"""
Train recognition model trên MỘT domain (hr | lr | sr_baseline | sr_improved)
với MỘT backbone cụ thể. Checkpoint tốt nhất chọn theo val set CÙNG domain.

Tối ưu GPU:
  - Mixed precision (AMP) khi chạy trên CUDA
  - DataLoader: pin_memory + persistent_workers
  - Tự động fallback sang CPU nếu CUDA hết bộ nhớ (OOM) giữa lúc train, tự
    thử lại GPU sau một số batch (xem utils/device_manager.py)

Chạy ví dụ:
    python train_recognition.py --config configs/config.yaml --domain hr --backbone mobilenet_v2
    python train_recognition.py --config configs/config.yaml --domain sr_improved --backbone resnet18

--init_ckpt (đã có từ trước): fine-tune/resume TRONG CÙNG 1 dataset (num_identities
GIỐNG NHAU giữa checkpoint và model hiện tại) — ví dụ domain sr_baseline fine-tune
từ chính checkpoint domain lr của CÙNG dataset đó (xem pipeline/run_multi_seed.sh).
Nạp state_dict TOÀN BỘ, strict=True — sẽ LỖI nếu num_identities khác nhau.

--init_ckpt_transfer ([MỚI]): TRANSFER LEARNING XUYÊN DATASET — ví dụ khởi tạo
từ checkpoint đã train trên EarVN1.0 (164 identity) để fine-tune trên AWE (100
identity). Vì đầu phân loại identity_head có kích thước PHỤ THUỘC num_identities
(khác nhau giữa 2 dataset), KHÔNG thể nạp strict=True toàn bộ như --init_ckpt.
Cờ này chỉ nạp các tensor CÙNG SHAPE (backbone features + embedding + gender_head
nếu num_genders giống nhau), tự động BỎ QUA và giữ nguyên khởi tạo ngẫu nhiên cho
identity_head (vì số lớp đầu ra khác nhau, không có cách nào "chuyển" trọng số
của 164 người sang đúng 100 người khác hoàn toàn) — xem hàm
load_transfer_checkpoint() bên dưới, có log rõ tensor nào được nạp/bỏ qua.

Chạy ví dụ fine-tune xuyên dataset (xem scripts/run_transfer_learning.sh):
    python train_recognition.py --config configs/config_awe_finetune.yaml \
        --domain sr_improved --backbone mobilenet_v2 \
        --init_ckpt_transfer runs/recognition_sr_improved_mobilenet_v2_seed42/best.pt \
        --seed 42 --run_suffix _finetuned_from_earvn1_seed42

--freeze_backbone ([MỚI]): dùng CÙNG --init_ckpt_transfer khi dataset đích quá ít dữ liệu
(ví dụ AWE, ~5.7 ảnh train/người) — đóng băng toàn bộ model.features (backbone đã học đặc
trưng từ dataset nguồn), chỉ train embedding + identity_head + gender_head. Giảm mạnh số
tham số phải học từ ít dữ liệu -> kỳ vọng giảm phương sai kết quả giữa các seed. Xem
scripts/run_transfer_learning_frozen.sh (chạy riêng, không đè kết quả transfer thường).
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.ear_dataset import EarDataset, build_label_map
from models.recognition_model import EarRecognitionNet, SUPPORTED_BACKBONES
from utils.device_manager import DeviceManager, move_optimizer_state
from utils.early_stopping import (EarlyStopping, save_state_dict as _save_state_dict,
                                   save_last_if_missing as _save_last_if_missing)
from utils.logger import setup_logger
from utils.metrics import compute_accuracy
from utils.seed import set_seed, seed_worker, seeded_generator


def load_transfer_checkpoint(model, ckpt_path, device, logger=None):
    """[MỚI] Nạp checkpoint TRANSFER LEARNING xuyên dataset — chỉ nạp các tensor
    CÙNG SHAPE với model hiện tại (backbone features, embedding, gender_head nếu
    num_genders khớp), BỎ QUA (giữ khởi tạo ngẫu nhiên/pretrained mặc định) các
    tensor lệch shape — thường chỉ là identity_head.weight/identity_head.bias vì
    num_identities khác nhau giữa 2 dataset. In rõ danh sách đã nạp/bỏ qua để
    kiểm chứng đúng những gì mong đợi (chỉ identity_head bị bỏ qua, không hơn)."""
    ckpt_state = torch.load(ckpt_path, map_location=device)
    model_state = model.state_dict()

    loaded_keys, skipped_keys = [], []
    for k, v in ckpt_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded_keys.append(k)
        else:
            skipped_keys.append(k)

    model.load_state_dict(model_state)

    msg = (f"[TRANSFER LEARNING] Khởi tạo từ checkpoint {ckpt_path}: "
           f"nạp {len(loaded_keys)}/{len(ckpt_state)} tensor khớp shape; "
           f"BỎ QUA {len(skipped_keys)} tensor lệch shape (giữ khởi tạo mới): {skipped_keys}")
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return loaded_keys, skipped_keys


def _forward_step(model, imgs, id_labels, gender_labels, device, cfg,
                   id_criterion, gender_criterion, optimizer, scaler, is_train):
    imgs = imgs.to(device, non_blocking=True)
    id_labels = id_labels.to(device, non_blocking=True)
    gender_labels = gender_labels.to(device, non_blocking=True)

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    # autocast chỉ thực sự bật khi ở CUDA — trên CPU sẽ là no-op (enabled=False)
    with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                         enabled=(device == "cuda")):
        id_logits, gender_logits, _ = model(imgs)
        loss_id = id_criterion(id_logits, id_labels)
        loss_gender = gender_criterion(gender_logits, gender_labels)
        loss = (cfg["recognition"]["identity_loss_weight"] * loss_id +
                cfg["recognition"]["gender_loss_weight"] * loss_gender)

    loss_val = loss.item()
    if is_train:
        if scaler is not None:
            # scaler.step() đã tự bỏ qua optimizer.step() khi grad không hữu
            # hạn — an toàn sẵn.
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # [SỬA — phát hiện qua review Q1, cùng lỗi đã sửa ở
            # train_sr_distill.py/train_sr.py] nhánh fallback CPU (scaler=None)
            # không có bảo vệ tự động — CrossEntropyLoss ổn định hơn nhiều so
            # với identity/saliency loss nên rủi ro thấp, nhưng chặn cho nhất
            # quán, chi phí gần như bằng 0.
            is_finite = loss_val == loss_val and loss_val not in (float("inf"), float("-inf"))
            if is_finite:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

    return loss_val, id_logits.detach(), gender_logits.detach(), id_labels, gender_labels


def run_epoch(model, loader, device_mgr, cfg, optimizer=None, scaler=None, logger=None,
              freeze_backbone_bn=False):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    # [MỚI — freeze_backbone] Nếu backbone bị đóng băng (requires_grad=False), vẫn phải
    # ép model.features.eval() ngay sau model.train() — nếu không, các lớp BatchNorm
    # trong backbone (dù không cập nhật affine weight/bias) vẫn tiếp tục cập nhật
    # running_mean/running_var theo minibatch nhỏ của dataset đích, làm trôi thống kê
    # BN học được từ dataset nguồn một cách âm thầm (không lỗi runtime, chỉ kém đi).
    if is_train and freeze_backbone_bn:
        model.features.eval()

    id_criterion = nn.CrossEntropyLoss()
    gender_criterion = nn.CrossEntropyLoss()

    total_loss, id_acc_sum, gender_acc_sum, n_batches, nan_batches = 0.0, 0.0, 0.0, 0, 0
    model_device = next(model.parameters()).device.type

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for imgs, id_labels, gender_labels in tqdm(loader, leave=False):
            device = device_mgr.current_device()

            if device != model_device:
                model.to(device)
                if optimizer is not None:
                    move_optimizer_state(optimizer, device)
                model_device = device
                if is_train and freeze_backbone_bn:
                    model.features.eval()

            try:
                loss, id_logits, gender_logits, id_l, gender_l = _forward_step(
                    model, imgs, id_labels, gender_labels, device, cfg,
                    id_criterion, gender_criterion, optimizer,
                    scaler if device == "cuda" else None, is_train)
            except RuntimeError as e:
                if device != "cuda" or "out of memory" not in str(e).lower():
                    raise  # lỗi khác, không phải OOM -> không nuốt lỗi, ném lại
                device_mgr.report_oom()
                device = "cpu"
                model.to(device)
                if optimizer is not None:
                    move_optimizer_state(optimizer, device)
                model_device = device
                if is_train and freeze_backbone_bn:
                    model.features.eval()
                # thử lại CHÍNH batch đó trên CPU, không bỏ qua dữ liệu
                loss, id_logits, gender_logits, id_l, gender_l = _forward_step(
                    model, imgs, id_labels, gender_labels, device, cfg,
                    id_criterion, gender_criterion, optimizer, None, is_train)

            if loss != loss or loss in (float("inf"), float("-inf")):
                nan_batches += 1
                continue

            total_loss += loss
            id_acc_sum += compute_accuracy(id_logits, id_l)
            gender_acc_sum += compute_accuracy(gender_logits, gender_l)
            n_batches += 1

    if nan_batches > 0 and logger:
        logger.info(f"  (lưu ý: {nan_batches} batch có loss NaN/Inf, đã bỏ qua khi tính trung bình)")

    if n_batches == 0:
        return float("nan"), 0.0, 0.0
    return (total_loss / n_batches, id_acc_sum / n_batches, gender_acc_sum / n_batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--domain", required=True,
                     help="hr | lr | sr_baseline | sr_improved | sr_ablation_<tên> (tùy ablation)")
    ap.add_argument("--backbone", required=True, choices=SUPPORTED_BACKBONES)
    ap.add_argument("--init_ckpt", default=None,
                     help="checkpoint khởi tạo để fine-tune/resume TRONG CÙNG dataset "
                          "(num_identities phải giống nhau — nạp strict=True). KHÔNG dùng "
                          "cờ này để chuyển dataset khác, dùng --init_ckpt_transfer.")
    ap.add_argument("--init_ckpt_transfer", default=None,
                     help="[MỚI] checkpoint khởi tạo để TRANSFER LEARNING XUYÊN DATASET "
                          "(num_identities có thể khác nhau, ví dụ EarVN1.0 (164) -> AWE (100)) "
                          "— chỉ nạp tensor cùng shape, tự bỏ qua/khởi tạo lại identity_head. "
                          "Xem load_transfer_checkpoint().")
    ap.add_argument("--seed", type=int, default=None,
                     help="ghi đè seed trong config — dùng để chạy multi-seed, đo độ ổn định")
    ap.add_argument("--run_suffix", default="",
                     help="hậu tố thêm vào tên thư mục runs/, tránh ghi đè khi chạy nhiều seed")
    ap.add_argument("--num_workers", type=int, default=4,
                     help="[MỚI — 2026-08-30, sự cố thật] số DataLoader worker — mặc định 4 "
                          "(giữ NGUYÊN hành vi cũ). Đặt 0 khi server dùng chung bị job khác chiếm "
                          "gần hết /dev/shm (num_workers=0 chạy DataLoader trong process chính, "
                          "không cần shared memory liên-process, xem chú thích tương tự trong "
                          "train_sr_distill.py::_make_hrlr_loaders).")
    ap.add_argument("--freeze_backbone", action="store_true",
                     help="[MỚI] Đóng băng TOÀN BỘ model.features (backbone trích đặc trưng) — "
                          "chỉ train embedding + identity_head + gender_head ('linear probe'). "
                          "Dùng cho fine-tune xuyên dataset khi dữ liệu đích rất ít (ví dụ AWE), "
                          "để giảm số tham số học từ ít dữ liệu -> giảm phương sai giữa seed. "
                          "Có ý nghĩa nhất khi dùng CÙNG --init_ckpt_transfer (backbone đã học "
                          "đặc trưng tốt từ dataset nguồn); dùng riêng --freeze_backbone không có "
                          "--init_ckpt_transfer sẽ đóng băng backbone ở trạng thái khởi tạo ngẫu "
                          "nhiên/ImageNet thô, không hợp lý cho identity ear recognition.")
    args = ap.parse_args()

    if args.init_ckpt and args.init_ckpt_transfer:
        raise ValueError("Chỉ dùng MỘT trong hai: --init_ckpt (cùng dataset) hoặc "
                          "--init_ckpt_transfer (xuyên dataset), không dùng cả hai cùng lúc.")

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["split"]["seed"] = args.seed

    set_seed(cfg["split"]["seed"])

    splits_root = cfg["paths"]["splits_root"]
    splits_json = f"{splits_root}/splits.json"
    label_map = build_label_map(splits_json)

    domain_root = f"{splits_root}/{args.domain}"
    image_size = cfg["image"]["hr_size"]

    train_set = EarDataset(domain_root, "train", splits_json, label_map, image_size, train=True)
    val_set = EarDataset(domain_root, "val", splits_json, label_map, image_size, train=False)

    # pin_memory + persistent_workers: tối ưu tốc độ nạp dữ liệu lên GPU
    # [SỬA — 2026-08-30] persistent_workers PHẢI theo (num_workers>0) — PyTorch
    # raise ValueError nếu persistent_workers=True mà num_workers=0 (trường hợp
    # dùng --num_workers 0 để né /dev/shm hết chỗ trên server dùng chung).
    nw = args.num_workers
    loader_kwargs = dict(num_workers=nw, pin_memory=torch.cuda.is_available(),
                          persistent_workers=(nw > 0))
    # [MỚI — phát hiện qua review Q1] worker_init_fn + generator cố định:
    # cudnn.deterministic (set_seed()) không đủ để tái lập tuyệt đối khi
    # num_workers>0 — thứ tự shuffle giờ tường minh qua generator seed riêng
    # thay vì dựa ngầm vào RNG toàn cục, và mỗi worker subprocess được seed
    # lại rõ ràng (xem utils/seed.py::seed_worker/seeded_generator).
    train_loader = DataLoader(train_set, batch_size=cfg["recognition"]["batch_size"],
                               shuffle=True, worker_init_fn=seed_worker if nw > 0 else None,
                               generator=seeded_generator(cfg["split"]["seed"]), **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg["recognition"]["batch_size"],
                             shuffle=False, worker_init_fn=seed_worker if nw > 0 else None, **loader_kwargs)

    run_name = f"recognition_{args.domain}_{args.backbone}{args.run_suffix}"
    run_dir = Path(cfg["paths"]["runs_root"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir, name="train")
    device_mgr = DeviceManager(logger=logger)
    logger.info(f"=== Bắt đầu train '{run_name}' ===")
    logger.info(f"Device ưu tiên: {device_mgr.preferred} | AMP: "
                f"{'bật' if device_mgr.preferred == 'cuda' else 'tắt (không có GPU)'}")
    logger.info(f"Domain: {args.domain} | Backbone: {args.backbone} | "
                f"Train: {len(train_set)} ảnh | Val: {len(val_set)} ảnh")

    # pretrained=True (ImageNet) CHỈ khi không khởi tạo từ checkpoint nào khác —
    # nếu dùng init_ckpt hoặc init_ckpt_transfer thì trọng số sẽ bị ghi đè ngay
    # sau đó, tải ImageNet weight lúc này chỉ tốn thời gian vô ích.
    use_pretrained_imagenet = (args.init_ckpt is None and args.init_ckpt_transfer is None)
    model = EarRecognitionNet(
        num_identities=cfg["num_identities"],
        num_genders=cfg["num_genders"],
        embedding_dim=cfg["recognition"]["embedding_dim"],
        backbone=args.backbone,
        pretrained=use_pretrained_imagenet,
    ).to(device_mgr.preferred)

    if args.init_ckpt:
        model.load_state_dict(torch.load(args.init_ckpt, map_location=device_mgr.preferred))
        logger.info(f"Khởi tạo từ checkpoint có sẵn để fine-tune (cùng dataset): {args.init_ckpt}")

    if args.init_ckpt_transfer:
        load_transfer_checkpoint(model, args.init_ckpt_transfer, device_mgr.preferred, logger=logger)

    if args.freeze_backbone:
        if not args.init_ckpt_transfer:
            logger.info(
                "[CẢNH BÁO] --freeze_backbone dùng mà KHÔNG có --init_ckpt_transfer — backbone "
                "bị đóng băng ở trạng thái khởi tạo ngẫu nhiên/ImageNet thô (chưa học đặc trưng "
                "ear-specific nào), nhiều khả năng học rất kém. Cờ này được thiết kế để dùng "
                "CÙNG --init_ckpt_transfer.")
        n_frozen = sum(p.numel() for p in model.features.parameters())
        n_total = sum(p.numel() for p in model.parameters())
        for p in model.features.parameters():
            p.requires_grad = False
        logger.info(
            f"[FREEZE BACKBONE] Đóng băng toàn bộ model.features: "
            f"{n_frozen:,}/{n_total:,} tham số ({100 * n_frozen / n_total:.1f}%) KHÔNG cập nhật. "
            f"Chỉ train: embedding + identity_head + gender_head "
            f"({n_total - n_frozen:,} tham số, {100 * (n_total - n_frozen) / n_total:.1f}%).")

    # filter(requires_grad) -> vô hại khi không freeze gì (trả về y hệt model.parameters()),
    # nhưng BẮT BUỘC khi có freeze_backbone: đưa param đã requires_grad=False vào Adam vẫn
    # chạy được (không lỗi) nhưng tốn bộ nhớ lưu trạng thái optimizer (m, v) vô ích cho những
    # tham số không bao giờ có gradient để cập nhật.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["recognition"]["lr"],
        weight_decay=cfg["recognition"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device_mgr.preferred == "cuda"))

    max_epochs = cfg["recognition"]["max_epochs"]
    patience = cfg["recognition"]["patience"]
    stopper = EarlyStopping(patience=patience, mode="max")

    for epoch in range(max_epochs):
        train_loss, train_id_acc, train_gender_acc = run_epoch(
            model, train_loader, device_mgr, cfg, optimizer=optimizer, scaler=scaler, logger=logger,
            freeze_backbone_bn=args.freeze_backbone)
        val_loss, val_id_acc, val_gender_acc = run_epoch(
            model, val_loader, device_mgr, cfg, optimizer=None, scaler=None, logger=logger,
            freeze_backbone_bn=args.freeze_backbone)

        oom_note = f" | OOM fallback: {device_mgr.total_oom_events} lần" \
            if device_mgr.total_oom_events > 0 else ""
        logger.info(
            f"[{run_name}] epoch {epoch + 1}/{max_epochs} "
            f"(early-stop counter: {stopper.counter}/{patience}){oom_note} | "
            f"train_id_acc={train_id_acc:.4f} train_gender_acc={train_gender_acc:.4f} | "
            f"VAL_ID_ACC={val_id_acc:.4f} VAL_GENDER_ACC={val_gender_acc:.4f} "
            f"val_loss={val_loss:.4f}"
        )

        is_best = stopper.step(val_id_acc)
        if is_best:
            # luôn lưu checkpoint ở CPU state_dict để load lại được bất kể device lúc train
            _save_state_dict(model, run_dir / "best.pt")
            logger.info(f"  -> checkpoint tốt nhất mới (val_id_acc={val_id_acc:.4f}), đã lưu.")

        if stopper.should_stop:
            logger.info(
                f"EARLY STOPPING tại epoch {epoch + 1}: val_id_acc không cải thiện "
                f"sau {patience} epoch liên tiếp. Best val_id_acc={stopper.best_str}"
            )
            break

    # [SỬA — bổ sung sau code review, vòng 4, điểm 3] xem giải thích tương tự
    # trong train_sr.py — đảm bảo LUÔN có best.pt kể cả khi val_id_acc là
    # NaN (ví dụ do numerically-unstable embedding) ở MỌI epoch.
    _save_last_if_missing(run_dir / "best.pt", model, logger, "best.pt")
    logger.info(f"=== Hoàn tất train '{run_name}'. Best val_id_acc={stopper.best_str}. "
                f"Tổng số lần fallback CPU do OOM: {device_mgr.total_oom_events}. "
                f"Checkpoint: {run_dir / 'best.pt'} ===")


if __name__ == "__main__":
    main()
