"""
Train mô hình SR nhẹ (ESPCN/FSRCNN) trên cặp HR-LR, dùng L1 pixel loss.
Đây là baseline SR "thuần pixel" — ở Giai đoạn 3, bạn sẽ thêm identity-aware loss
(ví dụ cosine loss giữa embedding của ảnh SR và embedding của ảnh HR từ recognition
model đã train ở cấu hình hr_hr) để biến nó thành SR "task-driven".

Chạy:
    python train_sr.py --config configs/config.yaml --sr_arch fsrcnn

Fine-tune span_official (SR baseline) từ checkpoint chính thức tác giả (dùng
lúc thiết lập ban đầu, xem pipeline/04_train_teacher_and_span_baseline.sh):
    python train_sr.py --config configs/config.yaml --sr_arch span_official \
        --pretrained_path checkpoints/span_pretrained_x4.pth

[MỚI] Fine-tune span_official/edsr TRANSFER LEARNING xuyên dataset (ví dụ
khởi tạo từ checkpoint span_baseline ĐÃ fine-tune trên EarVN1.0, fine-tune
tiếp trên AWE — để so sánh CÔNG BẰNG với span_tiny cũng được transfer learning,
xem scripts/run_transfer_learning.sh). BẮT BUỘC dùng --run_suffix trong
trường hợp này để KHÔNG ghi đè checkpoint "runs_<đích>/sr_span_official/best.pt"
gốc (checkpoint đó vẫn đang được dùng làm teacher_ckpt cho span_tiny, và làm
nguồn sinh ảnh domain sr_baseline hiện có — ghi đè sẽ làm hỏng khả năng tái
lập các kết quả trước đó):
    python train_sr.py --config configs/config_awe_finetune.yaml --sr_arch span_official \
        --pretrained_path runs/sr_span_official/best.pt \
        --run_suffix _finetuned_from_earvn1
    (--pretrained_path ở đây trỏ tới checkpoint ĐÃ train của dataset NGUỒN,
    không phải checkpoint pretrained gốc của tác giả — build_official_span()
    tự nhận diện đúng định dạng, xem models/span_official_wrapper.py)
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.hrlr_pair_dataset import HRLRPairDataset
from models.sr_models import build_sr_model
from utils.device_manager import DeviceManager, move_optimizer_state
from utils.early_stopping import (EarlyStopping, save_state_dict as _save_state_dict,
                                   save_last_if_missing as _save_last_if_missing)
from utils.logger import setup_logger
from utils.seed import set_seed, seed_worker, seeded_generator


def _forward_step(model, lr_img, hr_img, device, criterion, optimizer, scaler, is_train,
                   use_amp=True):
    lr_img = lr_img.to(device, non_blocking=True)
    hr_img = hr_img.to(device, non_blocking=True)

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    # [MỚI — sửa lỗi NaN loss phát hiện trên AWEx] span_official dùng phép scale nội
    # bộ img_range=255 ngay ở lớp đầu (models/span_official_wrapper.py) — kết hợp với
    # autocast FP16 có thể tràn số (FP16 tối đa ~65504) qua nhiều lớp conv, gây NaN
    # loss NGAY TỪ epoch 1 (đã tái lập 2 lần trên cùng dữ liệu AWEx, không phải may rủi
    # ngẫu nhiên theo seed). Các kiến trúc khác không dùng scale 255 nội bộ nên không
    # gặp vấn đề này -> chỉ tắt autocast cho riêng span_official (use_amp=False truyền
    # từ main()), giữ nguyên FP16 cho mọi kiến trúc khác (không đổi hành vi cũ của chúng).
    with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                         enabled=(device == "cuda" and use_amp)):
        sr_img = model(lr_img)
        loss = criterion(sr_img, hr_img)

    loss_val = loss.item()
    if is_train:
        if scaler is not None:
            # scaler.step() đã tự bỏ qua optimizer.step() khi phát hiện grad
            # không hữu hạn (hành vi chuẩn của GradScaler) — an toàn sẵn.
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # [SỬA — phát hiện qua review Q1, cùng lỗi đã sửa ở
            # train_sr_distill.py/train_sr_learned_prune.py] nhánh KHÔNG dùng
            # GradScaler (use_amp=False cho span_official, hoặc fallback CPU)
            # không có lớp bảo vệ tự động nào — trước đây backward()/step()
            # chạy vô điều kiện. span_official CHÍNH LÀ kiến trúc đã có lịch sử
            # NaN thật (xem comment phía trên) nên đáng chặn ở đây cho nhất
            # quán, dù rủi ro thấp hơn train_sr_distill.py (pixel loss L1 ổn
            # định hơn identity/saliency loss).
            is_finite = loss_val == loss_val and loss_val not in (float("inf"), float("-inf"))
            if is_finite:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

    return loss_val * lr_img.size(0), lr_img.size(0)


def run_epoch(model, loader, device_mgr, criterion, optimizer=None, scaler=None, logger=None,
              use_amp=True):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_n, nan_batches = 0.0, 0, 0
    model_device = next(model.parameters()).device.type

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for lr_img, hr_img in tqdm(loader, leave=False):
            device = device_mgr.current_device()

            if device != model_device:
                model.to(device)
                if optimizer is not None:
                    move_optimizer_state(optimizer, device)
                model_device = device

            try:
                loss_sum, n = _forward_step(
                    model, lr_img, hr_img, device, criterion, optimizer,
                    scaler if device == "cuda" else None, is_train, use_amp=use_amp)
            except RuntimeError as e:
                if device != "cuda" or "out of memory" not in str(e).lower():
                    raise
                device_mgr.report_oom()
                device = "cpu"
                model.to(device)
                if optimizer is not None:
                    move_optimizer_state(optimizer, device)
                model_device = device
                loss_sum, n = _forward_step(
                    model, lr_img, hr_img, device, criterion, optimizer, None, is_train,
                    use_amp=use_amp)

            # Bỏ qua batch có loss NaN/Inf khi tính trung bình hiển thị — trọng
            # số KHÔNG bị hỏng: nhánh có GradScaler tự bỏ qua optimizer.step()
            # khi phát hiện grad không hữu hạn; nhánh không có scaler (fallback
            # CPU, hoặc span_official tắt AMP) tự chặn bằng is_finite check
            # trong _forward_step() (xem sửa đổi NaN gradient, review Q1) —
            # cả 2 đường đều an toàn, chỉ khác cơ chế.
            if loss_sum != loss_sum or loss_sum in (float("inf"), float("-inf")):
                nan_batches += 1
                continue

            total_loss += loss_sum
            total_n += n

    if nan_batches > 0 and logger:
        logger.info(f"  (lưu ý: {nan_batches} batch có loss NaN/Inf, đã bỏ qua khi tính trung bình "
                     f"— trọng số không bị ảnh hưởng, xem chú thích trong run_epoch())")

    return total_loss / total_n if total_n > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sr_arch", default=None, help="ghi đè arch trong config nếu cần")
    ap.add_argument("--pretrained_path", default=None,
                     help="checkpoint pretrained chính thức HOẶC checkpoint đã train của dataset "
                          "khác (transfer learning) để fine-tune (dùng với --sr_arch span_official, "
                          "xem scripts/setup_span_official.sh và scripts/run_transfer_learning.sh)")
    ap.add_argument("--run_suffix", default="",
                     help="[MỚI] hậu tố thêm vào tên thư mục runs/sr_<arch>, tránh ghi đè checkpoint "
                          "gốc khi fine-tune/transfer learning (ví dụ _finetuned_from_earvn1). "
                          "QUAN TRỌNG: không truyền cờ này khi chạy pipeline chính bình thường "
                          "(pipeline/04_train_teacher_and_span_baseline.sh) — chỉ dùng khi cố ý "
                          "muốn lưu checkpoint riêng, không đè bản gốc.")
    ap.add_argument("--seed", type=int, default=None,
                     help="[MỚI — phát hiện qua review Q1] ghi đè seed trong config. Trước đây "
                          "train_sr.py KHÔNG có cờ này — toàn bộ SR (EDSR/span_official/Track A) "
                          "luôn train ở đúng 1 seed mặc định config.yaml (n=1), trong khi recognition "
                          "downstream lặp lại 5 seed — nghĩa là phương sai do CHÍNH seed train SR "
                          "(không chỉ downstream) chưa từng được đo. Cờ này thêm NĂNG LỰC train SR "
                          "nhiều seed (dùng cùng --run_suffix để tránh ghi đè checkpoint, ví dụ "
                          "--seed 123 --run_suffix _seed123) — CHƯA tự động chạy multi-seed SR trong "
                          "pipeline nào (cần bổ sung script điều phối + tốn thêm compute đáng kể, "
                          "là quyết định nghiên cứu/ngân sách GPU, không tự ý bật mặc định).")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["split"]["seed"] = args.seed

    set_seed(cfg["split"]["seed"])

    arch = args.sr_arch or cfg["sr"]["arch"]
    scale = cfg["image"]["scale"]

    splits_root = cfg["paths"]["splits_root"]
    train_set = HRLRPairDataset(f"{splits_root}/hr", f"{splits_root}/lr", "train")
    val_set = HRLRPairDataset(f"{splits_root}/hr", f"{splits_root}/lr", "val")

    loader_kwargs = dict(num_workers=4, pin_memory=torch.cuda.is_available(),
                          persistent_workers=True)
    # [MỚI — phát hiện qua review Q1] worker_init_fn + generator cố định, xem
    # giải thích đầy đủ trong train_recognition.py / utils/seed.py.
    train_loader = DataLoader(train_set, batch_size=cfg["sr"]["batch_size"],
                               shuffle=True, worker_init_fn=seed_worker,
                               generator=seeded_generator(cfg["split"]["seed"]), **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg["sr"]["batch_size"],
                             shuffle=False, **loader_kwargs)

    run_dir = Path(cfg["paths"]["runs_root"]) / f"sr_{arch}{args.run_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir, name="train")
    device_mgr = DeviceManager(logger=logger)
    logger.info(f"=== Bắt đầu train SR '{arch}' (scale={scale}) ===")
    logger.info(f"Device ưu tiên: {device_mgr.preferred}")
    logger.info(f"Train: {len(train_set)} cặp ảnh | Val: {len(val_set)} cặp ảnh")
    if args.pretrained_path:
        logger.info(f"Khởi tạo từ checkpoint có sẵn: {args.pretrained_path}")

    model = build_sr_model(arch, scale, pretrained_path=args.pretrained_path).to(device_mgr.preferred)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["sr"]["lr"])

    # [MỚI — sửa lỗi NaN loss trên AWEx] span_official dùng img_range=255 nội bộ,
    # dễ tràn số FP16 khi bật autocast -> tắt AMP riêng cho kiến trúc này (train FP32,
    # chậm hơn chút nhưng ổn định số học). Mọi kiến trúc khác giữ nguyên AMP như cũ.
    use_amp = (arch != "span_official") and (device_mgr.preferred == "cuda")
    if arch == "span_official":
        logger.info("[FP32] Tắt mixed-precision (AMP) cho span_official — kiến trúc này "
                     "dùng img_range=255 nội bộ, dễ tràn số FP16 (đã tái lập NaN loss "
                     "reproducible trên AWEx trước khi vá). Train chậm hơn nhưng ổn định.")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.L1Loss()

    max_epochs = cfg["sr"]["max_epochs"]
    patience = cfg["sr"]["patience"]
    stopper = EarlyStopping(patience=patience, mode="min")  # L1 loss: càng thấp càng tốt

    for epoch in range(max_epochs):
        train_loss = run_epoch(model, train_loader, device_mgr, criterion,
                                optimizer=optimizer, scaler=scaler, logger=logger, use_amp=use_amp)
        val_loss = run_epoch(model, val_loader, device_mgr, criterion,
                              optimizer=None, scaler=None, logger=logger, use_amp=use_amp)

        oom_note = f" | OOM fallback: {device_mgr.total_oom_events} lần" \
            if device_mgr.total_oom_events > 0 else ""
        logger.info(
            f"[{arch}] epoch {epoch + 1}/{max_epochs} "
            f"(early-stop counter: {stopper.counter}/{patience}){oom_note} | "
            f"train_L1={train_loss:.4f} VAL_L1={val_loss:.4f}"
        )

        is_best = stopper.step(val_loss)
        if is_best:
            _save_state_dict(model, run_dir / "best.pt")
            logger.info(f"  -> checkpoint tốt nhất mới (val_L1={val_loss:.4f}), đã lưu.")

        if stopper.should_stop:
            logger.info(
                f"EARLY STOPPING tại epoch {epoch + 1}: val_L1 không cải thiện "
                f"sau {patience} epoch liên tiếp. Best val_L1={stopper.best_str}"
            )
            break

    # [SỬA — bổ sung sau code review, vòng 4, điểm 3] TRƯỚC ĐÂY: nếu val_L1
    # là NaN/Inf ở MỌI epoch, is_best không bao giờ True -> best.pt KHÔNG BAO
    # GIỜ được tạo -> job train này tự nó không crash, nhưng bước SAU (data/
    # build_sr.py, eval_sr_quality.py cố load run_dir/best.pt) sẽ
    # FileNotFoundError, và log dòng cuối bên dưới vẫn nói "Checkpoint: ...best.pt"
    # dù file không tồn tại (gây hiểu nhầm). Đảm bảo LUÔN có 1 file tại đây
    # (xem save_last_if_missing() để biết vì sao checkpoint này — nếu phải
    # dùng tới — KHÔNG đáng tin).
    _save_last_if_missing(run_dir / "best.pt", model, logger, "best.pt")
    logger.info(f"=== Hoàn tất train SR '{arch}'. Best val_L1={stopper.best_str}. "
                f"Tổng số lần fallback CPU do OOM: {device_mgr.total_oom_events}. "
                f"Checkpoint: {run_dir / 'best.pt'} ===")


if __name__ == "__main__":
    main()
