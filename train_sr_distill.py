"""
Giai đoạn 3 — cải tiến SPAN: train SPAN (student, nhẹ) với loss tổng hợp gồm
3 thành phần, dùng 2 "giám khảo" đã train sẵn và đóng băng:
  - Teacher SR nặng (ví dụ EDSR) — cung cấp distillation loss
  - Recognition model đã train trên domain HR — cung cấp identity-aware loss

QUAN TRỌNG: chỉ backprop qua Student SPAN. Teacher SR và recognition model
CHỈ dùng để forward (torch.no_grad), không cập nhật trọng số của chúng.

Kiến trúc/tốc độ của SPAN không đổi so với bản baseline — chỉ cách TRAIN thay
đổi. Lúc triển khai thực tế chỉ dùng riêng Student, không cần 2 giám khảo này
nữa, nên KHÔNG ảnh hưởng đến latency/params lúc inference.

Chạy:
    python train_sr_distill.py --config configs/config.yaml

Chạy với lambda tùy chỉnh (dùng cho ablation, xem pipeline/run_ablation.sh):
    python train_sr_distill.py --config configs/config.yaml \
        --lambda_pixel 1.0 --lambda_distill 0.5 --lambda_identity 0.0 \
        --run_suffix _ablation_pixel_distill

Chạy với kiến trúc student khác (dùng cho ablation kiến trúc, ví dụ
span_large — xem pipeline/run_span_large_ablation.sh):
    python train_sr_distill.py --config configs/config.yaml \
        --student_arch span_large

Chạy FINE-TUNE / TRANSFER LEARNING xuyên dataset (ví dụ: khởi tạo từ
checkpoint span_tiny đã train trên EarVN1.0, fine-tune tiếp trên AWE — xem
scripts/run_transfer_learning.sh):
    python train_sr_distill.py --config configs/config_awe_finetune.yaml \
        --init_ckpt runs/sr_improved_span_tiny/best.pt \
        --run_suffix _finetuned_from_earvn1
    (an toàn dùng strict=True vì model SR không có tầng phụ thuộc số lượng
    identity — kiến trúc giống hệt nhau giữa mọi dataset, không có rủi ro
    lệch shape như bên recognition.)
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.hrlr_pair_dataset import HRLRPairDataset
from models.sr_models import build_sr_model
from models.recognition_model import EarRecognitionNet
from utils.device_manager import DeviceManager, move_optimizer_state
from utils.early_stopping import EarlyStopping
from utils.logger import setup_logger
from utils.seed import set_seed


def compute_total_loss(student_out, hr_img, teacher_out, recognition_model, cfg):
    l1 = nn.L1Loss()

    loss_pixel = l1(student_out, hr_img)
    loss_distill = l1(student_out, teacher_out)

    # Tính identity loss (cosine_similarity qua model giám khảo) ở fp32 TƯỜNG
    # MINH, không để dưới autocast fp16 của khối bên ngoài — cosine_similarity
    # kém ổn định số học ở fp16 (dễ chia gần-0/gần-0 ra NaN), đặc biệt khi
    # chuỗi qua nhiều model liên tiếp. Đã quan sát thấy gây NaN gần như toàn
    # bộ batch khi để dưới autocast — ép fp32 riêng phần này để khắc phục.
    with torch.autocast(device_type=student_out.device.type, enabled=False):
        student_out_f32 = student_out.float()
        hr_img_f32 = hr_img.float()
        with torch.no_grad():
            hr_emb = recognition_model.embed(hr_img_f32)
        student_emb = recognition_model.embed(student_out_f32)
        loss_identity = 1 - F.cosine_similarity(student_emb, hr_emb, dim=1, eps=1e-4).mean()

    ci = cfg["sr_improve"]
    total = (ci["lambda_pixel"] * loss_pixel +
             ci["lambda_distill"] * loss_distill +
             ci["lambda_identity"] * loss_identity)

    return total, {
        "pixel": loss_pixel.item(),
        "distill": loss_distill.item(),
        "identity": loss_identity.item(),
    }


def _move_all_to(device, student, teacher, recognition_model, optimizer):
    student.to(device)
    teacher.to(device)
    recognition_model.to(device)
    move_optimizer_state(optimizer, device)


def _forward_step(student, teacher, recognition_model, lr_img, hr_img, device,
                   cfg, optimizer, scaler, is_train):
    lr_img = lr_img.to(device, non_blocking=True)
    hr_img = hr_img.to(device, non_blocking=True)

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    # LƯU Ý: KHÔNG dùng AMP/autocast ở đây (khác với train_sr.py/
    # train_recognition.py). SPAN chính thức nhân giá trị nội bộ lên tới
    # img_range=255 (xem span_arch.py: x = (x-mean)*img_range), biên độ hoạt
    # động lớn dễ tràn số dưới fp16 — càng dễ tràn hơn khi chuỗi qua 3 model
    # liên tiếp (student->teacher->recognition) trong cùng 1 lần forward.
    # Đã quan sát thực tế: bật AMP ở đây làm ~100% batch NaN kể cả lúc
    # validation (không qua backward/GradScaler) -> lỗi tràn số ở forward,
    # không phải lỗi gradient. fp32 chậm hơn nhưng ổn định tuyệt đối.
    student_out = student(lr_img)
    with torch.no_grad():
        teacher_out = teacher(lr_img)
    loss, parts = compute_total_loss(student_out, hr_img, teacher_out,
                                      recognition_model, cfg)

    if is_train:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

    return loss.item(), parts


def run_epoch(student, teacher, recognition_model, loader, device_mgr, cfg,
              optimizer=None, scaler=None, logger=None):
    is_train = optimizer is not None
    student.train() if is_train else student.eval()

    totals = {"pixel": 0.0, "distill": 0.0, "identity": 0.0, "total": 0.0}
    n, nan_batches = 0, 0
    model_device = next(student.parameters()).device.type

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for lr_img, hr_img in tqdm(loader, leave=False):
            device = device_mgr.current_device()

            if device != model_device:
                _move_all_to(device, student, teacher, recognition_model, optimizer)
                model_device = device

            try:
                loss_val, parts = _forward_step(
                    student, teacher, recognition_model, lr_img, hr_img, device,
                    cfg, optimizer, scaler if device == "cuda" else None, is_train)
            except RuntimeError as e:
                if device != "cuda" or "out of memory" not in str(e).lower():
                    raise
                device_mgr.report_oom()
                device = "cpu"
                _move_all_to(device, student, teacher, recognition_model, optimizer)
                model_device = device
                loss_val, parts = _forward_step(
                    student, teacher, recognition_model, lr_img, hr_img, device,
                    cfg, optimizer, None, is_train)

            if loss_val != loss_val or loss_val in (float("inf"), float("-inf")):
                nan_batches += 1
                continue

            for k, v in parts.items():
                totals[k] += v
            totals["total"] += loss_val
            n += 1

    if nan_batches > 0 and logger:
        logger.info(f"  (lưu ý: {nan_batches} batch có loss NaN/Inf, đã bỏ qua khi tính trung bình)")

    if n == 0:
        return {k: float("nan") for k in totals}
    return {k: v / n for k, v in totals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lambda_pixel", type=float, default=None,
                     help="ghi đè sr_improve.lambda_pixel trong config (dùng cho ablation)")
    ap.add_argument("--lambda_distill", type=float, default=None,
                     help="ghi đè sr_improve.lambda_distill trong config (dùng cho ablation)")
    ap.add_argument("--lambda_identity", type=float, default=None,
                     help="ghi đè sr_improve.lambda_identity trong config (dùng cho ablation)")
    ap.add_argument("--student_arch", default=None,
                     help="ghi đè sr_improve.student_arch trong config (dùng cho ablation kiến "
                          "trúc, ví dụ --student_arch span_large)")
    ap.add_argument("--init_ckpt", default=None,
                     help="[MỚI] checkpoint SPAN student có sẵn để khởi tạo trọng số trước khi "
                          "train — dùng cho FINE-TUNE/TRANSFER LEARNING xuyên dataset (ví dụ "
                          "khởi tạo từ span_tiny đã train trên EarVN1.0 rồi fine-tune trên AWE). "
                          "Nạp strict=True (an toàn: model SR không có tầng phụ thuộc số lượng "
                          "identity/dataset, kiến trúc giống hệt nhau mọi dataset).")
    ap.add_argument("--run_suffix", default="",
                     help="hậu tố thêm vào tên thư mục runs/, tránh ghi đè checkpoint "
                          "khi chạy nhiều cấu hình ablation/fine-tune khác nhau")
    ap.add_argument("--seed", type=int, default=None,
                     help="ghi đè seed trong config — dùng để chạy multi-seed, đo độ ổn định")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["split"]["seed"] = args.seed

    set_seed(cfg["split"]["seed"])
    ci = cfg["sr_improve"]

    if args.lambda_pixel is not None:
        ci["lambda_pixel"] = args.lambda_pixel
    if args.lambda_distill is not None:
        ci["lambda_distill"] = args.lambda_distill
    if args.lambda_identity is not None:
        ci["lambda_identity"] = args.lambda_identity
    if args.student_arch is not None:
        ci["student_arch"] = args.student_arch

    scale = cfg["image"]["scale"]
    splits_root = cfg["paths"]["splits_root"]

    train_set = HRLRPairDataset(f"{splits_root}/hr", f"{splits_root}/lr", "train")
    val_set = HRLRPairDataset(f"{splits_root}/hr", f"{splits_root}/lr", "val")
    loader_kwargs = dict(num_workers=4, pin_memory=torch.cuda.is_available(),
                          persistent_workers=True)
    train_loader = DataLoader(train_set, batch_size=ci["batch_size"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=ci["batch_size"], shuffle=False, **loader_kwargs)

    student_arch = ci.get("student_arch", cfg["sr"]["arch"])
    student_pretrained = ci.get("student_pretrained_path")

    run_dir = Path(cfg["paths"]["runs_root"]) / f"sr_improved_{student_arch}{args.run_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir, name="train")
    logger.info(f"Student architecture: {student_arch} | Lambda: pixel={ci['lambda_pixel']} "
                f"distill={ci['lambda_distill']} identity={ci['lambda_identity']}")
    device_mgr = DeviceManager(logger=logger)
    device = device_mgr.preferred
    logger.info(f"=== Bắt đầu cải tiến SPAN (distillation + identity-aware loss) ===")
    logger.info(f"Device ưu tiên: {device}")
    logger.info(f"Teacher: {ci['teacher_arch']} | Recognition giám khảo: {ci['frozen_recognition_ckpt']}")
    logger.info(f"Train: {len(train_set)} cặp ảnh | Val: {len(val_set)} cặp ảnh")

    # --- Student: kiến trúc NÉN (student_arch, ví dụ span_tiny) — model sẽ được deploy ---
    student = build_sr_model(student_arch, scale, pretrained_path=student_pretrained).to(device)

    # [MỚI] Khởi tạo student từ checkpoint có sẵn — dùng cho fine-tune/transfer learning.
    # An toàn strict=True vì SPAN/span_tiny/span_large không có tầng nào phụ thuộc
    # num_identities hay bất kỳ thông tin riêng của dataset — kiến trúc giống hệt
    # nhau dù train trên dataset nào, nên không có rủi ro lệch shape.
    if args.init_ckpt:
        student.load_state_dict(torch.load(args.init_ckpt, map_location=device))
        logger.info(f"[TRANSFER LEARNING] Khởi tạo student từ checkpoint có sẵn: {args.init_ckpt} "
                    f"(nạp toàn bộ trọng số, strict=True — an toàn vì không có tầng phụ thuộc dataset)")

    # --- Teacher: SPAN baseline (đã chứng minh chất lượng tốt), ĐÓNG BĂNG ---
    teacher = build_sr_model(ci["teacher_arch"], scale).to(device)
    teacher.load_state_dict(torch.load(ci["teacher_ckpt"], map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # --- Recognition model train trên domain HR, ĐÓNG BĂNG, dùng làm giám khảo ---
    recognition_model = EarRecognitionNet(
        num_identities=cfg["num_identities"],
        num_genders=cfg["num_genders"],
        embedding_dim=cfg["recognition"]["embedding_dim"],
        backbone="mobilenet_v2",   # backbone dùng làm giám khảo, có thể đổi tùy checkpoint đã có
        pretrained=False,
    ).to(device)
    recognition_model.load_state_dict(
        torch.load(ci["frozen_recognition_ckpt"], map_location=device))
    recognition_model.eval()
    for p in recognition_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(student.parameters(), lr=ci["lr"])
    # KHÔNG dùng GradScaler/AMP cho script này (xem ghi chú trong _forward_step
    # về lý do img_range=255 của SPAN dễ tràn số dưới fp16 khi chuỗi 3 model).

    max_epochs = ci["max_epochs"]
    patience = ci["patience"]
    stopper = EarlyStopping(patience=patience, mode="min")  # total loss: càng thấp càng tốt

    for epoch in range(max_epochs):
        train_stats = run_epoch(student, teacher, recognition_model, train_loader,
                                 device_mgr, cfg, optimizer=optimizer, scaler=None, logger=logger)
        val_stats = run_epoch(student, teacher, recognition_model, val_loader,
                               device_mgr, cfg, optimizer=None, scaler=None, logger=logger)

        oom_note = f" | OOM fallback: {device_mgr.total_oom_events} lần" \
            if device_mgr.total_oom_events > 0 else ""
        logger.info(
            f"[sr_improved] epoch {epoch + 1}/{max_epochs} "
            f"(early-stop counter: {stopper.counter}/{patience}){oom_note} | "
            f"train_total={train_stats['total']:.4f} pixel={train_stats['pixel']:.4f} "
            f"distill={train_stats['distill']:.4f} identity={train_stats['identity']:.4f} | "
            f"VAL_TOTAL={val_stats['total']:.4f}"
        )

        is_best = stopper.step(val_stats["total"])
        if is_best:
            torch.save({k: v.cpu() for k, v in student.state_dict().items()}, run_dir / "best.pt")
            logger.info(f"  -> checkpoint tốt nhất mới (val_total={val_stats['total']:.4f}), đã lưu.")

        if stopper.should_stop:
            logger.info(
                f"EARLY STOPPING tại epoch {epoch + 1}: val_total không cải thiện "
                f"sau {patience} epoch liên tiếp. Best val_total={stopper.best:.4f}"
            )
            break

    logger.info(f"=== Hoàn tất cải tiến SPAN. Best val_total={stopper.best:.4f}. "
                f"Tổng số lần fallback CPU do OOM: {device_mgr.total_oom_events}. "
                f"Checkpoint: {run_dir / 'best.pt'} ===")


if __name__ == "__main__":
    main()
