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
from utils.early_stopping import EarlyStopping
from utils.logger import setup_logger
from utils.metrics import compute_accuracy
from utils.seed import set_seed


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

    if is_train:
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    return loss.item(), id_logits.detach(), gender_logits.detach(), id_labels, gender_labels


def run_epoch(model, loader, device_mgr, cfg, optimizer=None, scaler=None, logger=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

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
                     help="checkpoint khởi tạo để fine-tune thay vì train from-scratch")
    args = ap.parse_args()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["split"]["seed"])

    splits_root = cfg["paths"]["splits_root"]
    splits_json = f"{splits_root}/splits.json"
    label_map = build_label_map(splits_json)

    domain_root = f"{splits_root}/{args.domain}"
    image_size = cfg["image"]["hr_size"]

    train_set = EarDataset(domain_root, "train", splits_json, label_map, image_size, train=True)
    val_set = EarDataset(domain_root, "val", splits_json, label_map, image_size, train=False)

    # pin_memory + persistent_workers: tối ưu tốc độ nạp dữ liệu lên GPU
    loader_kwargs = dict(num_workers=4, pin_memory=torch.cuda.is_available(),
                          persistent_workers=True)
    train_loader = DataLoader(train_set, batch_size=cfg["recognition"]["batch_size"],
                               shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg["recognition"]["batch_size"],
                             shuffle=False, **loader_kwargs)

    run_name = f"recognition_{args.domain}_{args.backbone}"
    run_dir = Path(cfg["paths"]["runs_root"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir, name="train")
    device_mgr = DeviceManager(logger=logger)
    logger.info(f"=== Bắt đầu train '{run_name}' ===")
    logger.info(f"Device ưu tiên: {device_mgr.preferred} | AMP: "
                f"{'bật' if device_mgr.preferred == 'cuda' else 'tắt (không có GPU)'}")
    logger.info(f"Domain: {args.domain} | Backbone: {args.backbone} | "
                f"Train: {len(train_set)} ảnh | Val: {len(val_set)} ảnh")

    model = EarRecognitionNet(
        num_identities=cfg["num_identities"],
        num_genders=cfg["num_genders"],
        embedding_dim=cfg["recognition"]["embedding_dim"],
        backbone=args.backbone,
        pretrained=(args.init_ckpt is None),
    ).to(device_mgr.preferred)

    if args.init_ckpt:
        model.load_state_dict(torch.load(args.init_ckpt, map_location=device_mgr.preferred))
        logger.info(f"Khởi tạo từ checkpoint có sẵn để fine-tune: {args.init_ckpt}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["recognition"]["lr"],
        weight_decay=cfg["recognition"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device_mgr.preferred == "cuda"))

    max_epochs = cfg["recognition"]["max_epochs"]
    patience = cfg["recognition"]["patience"]
    stopper = EarlyStopping(patience=patience, mode="max")

    for epoch in range(max_epochs):
        train_loss, train_id_acc, train_gender_acc = run_epoch(
            model, train_loader, device_mgr, cfg, optimizer=optimizer, scaler=scaler, logger=logger)
        val_loss, val_id_acc, val_gender_acc = run_epoch(
            model, val_loader, device_mgr, cfg, optimizer=None, scaler=None, logger=logger)

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
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, run_dir / "best.pt")
            logger.info(f"  -> checkpoint tốt nhất mới (val_id_acc={val_id_acc:.4f}), đã lưu.")

        if stopper.should_stop:
            logger.info(
                f"EARLY STOPPING tại epoch {epoch + 1}: val_id_acc không cải thiện "
                f"sau {patience} epoch liên tiếp. Best val_id_acc={stopper.best:.4f}"
            )
            break

    logger.info(f"=== Hoàn tất train '{run_name}'. Best val_id_acc={stopper.best:.4f}. "
                f"Tổng số lần fallback CPU do OOM: {device_mgr.total_oom_events}. "
                f"Checkpoint: {run_dir / 'best.pt'} ===")


if __name__ == "__main__":
    main()
