"""
Train mô hình SR nhẹ (ESPCN/FSRCNN) trên cặp HR-LR, dùng L1 pixel loss.
Đây là baseline SR "thuần pixel" — ở Giai đoạn 3, bạn sẽ thêm identity-aware loss
(ví dụ cosine loss giữa embedding của ảnh SR và embedding của ảnh HR từ recognition
model đã train ở cấu hình hr_hr) để biến nó thành SR "task-driven".

Chạy:
    python train_sr.py --config configs/config.yaml --sr_arch fsrcnn
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
from utils.early_stopping import EarlyStopping
from utils.logger import setup_logger
from utils.seed import set_seed


def _forward_step(model, lr_img, hr_img, device, criterion, optimizer, scaler, is_train):
    lr_img = lr_img.to(device, non_blocking=True)
    hr_img = hr_img.to(device, non_blocking=True)

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                         enabled=(device == "cuda")):
        sr_img = model(lr_img)
        loss = criterion(sr_img, hr_img)

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

    return loss.item() * lr_img.size(0), lr_img.size(0)


def run_epoch(model, loader, device_mgr, criterion, optimizer=None, scaler=None, logger=None):
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
                    scaler if device == "cuda" else None, is_train)
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
                    model, lr_img, hr_img, device, criterion, optimizer, None, is_train)

            # Bỏ qua batch có loss NaN/Inf khi tính trung bình hiển thị — do
            # GradScaler đã tự bỏ qua bước cập nhật cho batch đó (không hỏng
            # trọng số), chỉ cần không để nó làm "bẩn" số liệu trung bình epoch.
            if loss_sum != loss_sum or loss_sum in (float("inf"), float("-inf")):
                nan_batches += 1
                continue

            total_loss += loss_sum
            total_n += n

    if nan_batches > 0 and logger:
        logger.info(f"  (lưu ý: {nan_batches} batch có loss NaN/Inf, đã bỏ qua khi tính trung bình "
                     f"— GradScaler tự bỏ qua cập nhật cho các batch này, không ảnh hưởng trọng số)")

    return total_loss / total_n if total_n > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sr_arch", default=None, help="ghi đè arch trong config nếu cần")
    ap.add_argument("--pretrained_path", default=None,
                     help="checkpoint pretrained chính thức để fine-tune (dùng với "
                          "--sr_arch span_official, xem scripts/setup_span_official.sh)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["split"]["seed"])

    arch = args.sr_arch or cfg["sr"]["arch"]
    scale = cfg["image"]["scale"]

    splits_root = cfg["paths"]["splits_root"]
    train_set = HRLRPairDataset(f"{splits_root}/hr", f"{splits_root}/lr", "train")
    val_set = HRLRPairDataset(f"{splits_root}/hr", f"{splits_root}/lr", "val")

    loader_kwargs = dict(num_workers=4, pin_memory=torch.cuda.is_available(),
                          persistent_workers=True)
    train_loader = DataLoader(train_set, batch_size=cfg["sr"]["batch_size"],
                               shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg["sr"]["batch_size"],
                             shuffle=False, **loader_kwargs)

    run_dir = Path(cfg["paths"]["runs_root"]) / f"sr_{arch}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir, name="train")
    device_mgr = DeviceManager(logger=logger)
    logger.info(f"=== Bắt đầu train SR '{arch}' (scale={scale}) ===")
    logger.info(f"Device ưu tiên: {device_mgr.preferred}")
    logger.info(f"Train: {len(train_set)} cặp ảnh | Val: {len(val_set)} cặp ảnh")

    model = build_sr_model(arch, scale, pretrained_path=args.pretrained_path).to(device_mgr.preferred)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["sr"]["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device_mgr.preferred == "cuda"))
    criterion = nn.L1Loss()

    max_epochs = cfg["sr"]["max_epochs"]
    patience = cfg["sr"]["patience"]
    stopper = EarlyStopping(patience=patience, mode="min")  # L1 loss: càng thấp càng tốt

    for epoch in range(max_epochs):
        train_loss = run_epoch(model, train_loader, device_mgr, criterion,
                                optimizer=optimizer, scaler=scaler, logger=logger)
        val_loss = run_epoch(model, val_loader, device_mgr, criterion,
                              optimizer=None, scaler=None, logger=logger)

        oom_note = f" | OOM fallback: {device_mgr.total_oom_events} lần" \
            if device_mgr.total_oom_events > 0 else ""
        logger.info(
            f"[{arch}] epoch {epoch + 1}/{max_epochs} "
            f"(early-stop counter: {stopper.counter}/{patience}){oom_note} | "
            f"train_L1={train_loss:.4f} VAL_L1={val_loss:.4f}"
        )

        is_best = stopper.step(val_loss)
        if is_best:
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, run_dir / "best.pt")
            logger.info(f"  -> checkpoint tốt nhất mới (val_L1={val_loss:.4f}), đã lưu.")

        if stopper.should_stop:
            logger.info(
                f"EARLY STOPPING tại epoch {epoch + 1}: val_L1 không cải thiện "
                f"sau {patience} epoch liên tiếp. Best val_L1={stopper.best:.4f}"
            )
            break

    logger.info(f"=== Hoàn tất train SR '{arch}'. Best val_L1={stopper.best:.4f}. "
                f"Tổng số lần fallback CPU do OOM: {device_mgr.total_oom_events}. "
                f"Checkpoint: {run_dir / 'best.pt'} ===")


if __name__ == "__main__":
    main()
