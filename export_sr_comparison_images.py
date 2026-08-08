"""
Xuất ảnh so sánh trực quan: LR (bicubic thường) | SR baseline | SR improved |
HR (ground truth) — trên CÙNG một tập ảnh mẫu, ghép thành 1 hàng ngang có
nhãn + PSNR, để đánh giá định tính bằng mắt bên cạnh số liệu PSNR/SSIM định
lượng đã có trong sr_quality.csv.

Chạy:
    python export_sr_comparison_images.py --config configs/config.yaml \
        --sr_baseline_ckpt runs/sr_span_official/best.pt --sr_baseline_arch span_official \
        --sr_improved_ckpt runs/sr_improved_span_tiny/best.pt --sr_improved_arch span_tiny \
        --n_samples 20 --out_dir results/sr_comparison_images
"""
import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from datasets.hrlr_pair_dataset import HRLRPairDataset
from models.sr_models import build_sr_model
from utils.metrics import compute_psnr, compute_ssim

to_pil = transforms.ToPILImage()


def tensor_to_pil(t):
    return to_pil(t.clamp(0, 1).cpu())


def add_label(img, text):
    """Thêm 1 dòng nhãn phía trên ảnh (nền trắng), trả về ảnh cao hơn 1 chút."""
    label_h = 18
    new_img = Image.new("RGB", (img.width, img.height + label_h), (255, 255, 255))
    new_img.paste(img, (0, label_h))
    draw = ImageDraw.Draw(new_img)
    draw.text((3, 2), text, fill=(0, 0, 0), font=ImageFont.load_default())
    return new_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sr_baseline_ckpt", required=True)
    ap.add_argument("--sr_baseline_arch", required=True)
    ap.add_argument("--sr_improved_ckpt", required=True)
    ap.add_argument("--sr_improved_arch", required=True)
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--display_scale", type=int, default=3,
                     help="phóng to ảnh lên bao nhiêu lần để dễ nhìn bằng mắt")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scale = cfg["image"]["scale"]
    hr_size = cfg["image"]["hr_size"]
    splits_root = cfg["paths"]["splits_root"]

    test_set = HRLRPairDataset(f"{splits_root}/hr", f"{splits_root}/lr", "test")
    n = min(args.n_samples, len(test_set))
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(test_set)), n)

    baseline = build_sr_model(args.sr_baseline_arch, scale)
    baseline.load_state_dict(torch.load(args.sr_baseline_ckpt, map_location=device))
    baseline.to(device).eval()

    improved = build_sr_model(args.sr_improved_arch, scale)
    improved.load_state_dict(torch.load(args.sr_improved_ckpt, map_location=device))
    improved.to(device).eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    display_size = hr_size * args.display_scale
    csv_rows = []

    with torch.no_grad():
        for i, idx in enumerate(indices):
            lr_img, hr_img = test_set[idx]
            lr_batch = lr_img.unsqueeze(0).to(device)
            hr_batch = hr_img.unsqueeze(0).to(device)

            sr_baseline_out = baseline(lr_batch)
            sr_improved_out = improved(lr_batch)

            psnr_baseline = compute_psnr(sr_baseline_out, hr_batch)
            ssim_baseline = compute_ssim(sr_baseline_out, hr_batch)
            psnr_improved = compute_psnr(sr_improved_out, hr_batch)
            ssim_improved = compute_ssim(sr_improved_out, hr_batch)

            # LR phóng to bằng bicubic THƯỜNG (không SR) — làm mốc "không xử lý gì"
            lr_upsampled = F.interpolate(lr_batch, size=hr_batch.shape[-2:],
                                          mode="bicubic", align_corners=False).clamp(0, 1)

            panels = [
                (lr_upsampled[0], "LR (bicubic, khong SR)"),
                (sr_baseline_out[0], f"SR baseline  PSNR={psnr_baseline:.1f}dB"),
                (sr_improved_out[0], f"SR improved  PSNR={psnr_improved:.1f}dB"),
                (hr_batch[0], "HR (ground truth)"),
            ]

            # NEAREST khi phóng to: không làm mượt thêm, giữ đúng pixel model tạo ra
            resized_labeled = [
                add_label(tensor_to_pil(t).resize((display_size, display_size), Image.NEAREST), label)
                for t, label in panels
            ]

            gap = 10
            total_w = sum(p.width for p in resized_labeled) + gap * (len(resized_labeled) - 1)
            total_h = resized_labeled[0].height
            combined = Image.new("RGB", (total_w, total_h), (255, 255, 255))
            x = 0
            for p in resized_labeled:
                combined.paste(p, (x, 0))
                x += p.width + gap

            combined.save(out_dir / f"sample_{i:02d}.png")

            csv_rows.append({
                "sample_idx": i, "dataset_idx": idx,
                "psnr_baseline_db": round(psnr_baseline, 3), "ssim_baseline": round(ssim_baseline, 4),
                "psnr_improved_db": round(psnr_improved, 3), "ssim_improved": round(ssim_improved, 4),
            })

    with open(out_dir / "comparison_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    avg_psnr_b = sum(r["psnr_baseline_db"] for r in csv_rows) / n
    avg_psnr_i = sum(r["psnr_improved_db"] for r in csv_rows) / n
    print(f"Đã xuất {n} ảnh so sánh vào {out_dir}/ (sample_00.png ... sample_{n-1:02d}.png)")
    print(f"PSNR trung bình trên {n} mẫu này — baseline: {avg_psnr_b:.2f}dB | improved: {avg_psnr_i:.2f}dB")
    print(f"Chi tiết từng ảnh: {out_dir}/comparison_metrics.csv")


if __name__ == "__main__":
    main()
