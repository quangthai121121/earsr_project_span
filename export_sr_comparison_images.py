"""
Xuất ảnh so sánh trực quan: LR (bicubic thường) | SR baseline | SR improved |
HR (ground truth) — trên CÙNG một tập ảnh mẫu, ghép thành 1 hàng ngang có
nhãn + PSNR, để đánh giá định tính bằng mắt bên cạnh số liệu PSNR/SSIM định
lượng đã có trong sr_quality.csv.

Chạy (bản gốc, lưu ảnh cho toàn bộ n_samples — dùng cho bộ 20 ảnh đã trích
dẫn trong bài, KHÔNG đổi hành vi này):
    python export_sr_comparison_images.py --config configs/config.yaml \
        --sr_baseline_ckpt runs/sr_span_official/best.pt --sr_baseline_arch span_official \
        --sr_improved_ckpt runs/sr_improved_span_tiny/best.pt --sr_improved_arch span_tiny \
        --n_samples 20 --out_dir results/sr_comparison_images

Chạy (bản tìm outlier — [MỚI, trả lời phản biện mục #6]: bộ 20 ảnh hiện tại
chỉ chọn tuần tự, không tìm ca thất bại, khoảng cách PSNR lớn nhất hiện có
chỉ +2.645dB. Tính PSNR/SSIM trên n_samples LỚN (vd 150-200) nhưng CHỈ lưu
ảnh PNG cho top-K mẫu có khoảng cách PSNR baseline-improved lớn nhất theo cả
2 chiều, để tránh ghi hàng trăm ảnh không cần thiết. CSV vẫn ghi đủ số liệu
cho toàn bộ n_samples):
    python export_sr_comparison_images.py --config configs/config.yaml \
        --sr_baseline_ckpt runs/sr_span_official/best.pt --sr_baseline_arch span_official \
        --sr_improved_ckpt runs/sr_improved_span_tiny/best.pt --sr_improved_arch span_tiny \
        --n_samples 150 --save_top_k_gap 10 --out_dir results/sr_comparison_search
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
    ap.add_argument("--save_top_k_gap", type=int, default=0,
                     help="Nếu >0: CSV vẫn tính đủ n_samples, nhưng CHỈ lưu ảnh PNG cho "
                          "top-K mẫu có |PSNR_baseline - PSNR_improved| lớn nhất theo mỗi "
                          "chiều (tìm ca improved thua/thắng rõ nhất). Nếu =0 (mặc định): "
                          "giữ nguyên hành vi gốc, lưu ảnh cho toàn bộ n_samples.")
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

    def save_composite(lr_batch, sr_baseline_out, sr_improved_out, hr_batch,
                        psnr_baseline, psnr_improved, save_path):
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

        combined.save(save_path)

    csv_rows = []

    if args.save_top_k_gap <= 0:
        # Hành vi gốc: lưu ảnh cho TOÀN BỘ n_samples (không đổi, dùng cho bộ
        # 20 ảnh đã trích dẫn trực tiếp trong bài — sample_00.png, sample_06.png, ...).
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

                save_composite(lr_batch, sr_baseline_out, sr_improved_out, hr_batch,
                               psnr_baseline, psnr_improved, out_dir / f"sample_{i:02d}.png")

                csv_rows.append({
                    "sample_idx": i, "dataset_idx": idx,
                    "psnr_baseline_db": round(psnr_baseline, 3), "ssim_baseline": round(ssim_baseline, 4),
                    "psnr_improved_db": round(psnr_improved, 3), "ssim_improved": round(ssim_improved, 4),
                    "gap_baseline_minus_improved_db": round(psnr_baseline - psnr_improved, 3),
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
        return

    # [MỚI] Chế độ tìm outlier: pass 1 — chỉ tính PSNR/SSIM cho TOÀN BỘ
    # n_samples (không lưu ảnh, rẻ vì chỉ inference). pass 2 — chỉ chạy lại
    # inference + lưu ảnh PNG cho top-K mẫu có |gap| lớn nhất mỗi chiều, để
    # tránh ghi hàng trăm ảnh không cần thiết lên đĩa.
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

            csv_rows.append({
                "sample_idx": i, "dataset_idx": idx,
                "psnr_baseline_db": round(psnr_baseline, 3), "ssim_baseline": round(ssim_baseline, 4),
                "psnr_improved_db": round(psnr_improved, 3), "ssim_improved": round(ssim_improved, 4),
                "gap_baseline_minus_improved_db": round(psnr_baseline - psnr_improved, 3),
            })

    with open(out_dir / "comparison_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    k = min(args.save_top_k_gap, n)
    # gap > 0: improved KÉM hơn baseline (ca thất bại của improved -- cái ta đang tìm)
    # gap < 0: improved TỐT hơn baseline (để đối chiếu, không phải "thất bại")
    worst_for_improved = sorted(csv_rows, key=lambda r: r["gap_baseline_minus_improved_db"], reverse=True)[:k]
    best_for_improved = sorted(csv_rows, key=lambda r: r["gap_baseline_minus_improved_db"])[:k]

    selected = {}
    for tag, rows in (("worst", worst_for_improved), ("best", best_for_improved)):
        for rank, r in enumerate(rows):
            selected[r["sample_idx"]] = (tag, rank, r)

    with torch.no_grad():
        for sample_idx, (tag, rank, row) in selected.items():
            idx = indices[sample_idx]
            lr_img, hr_img = test_set[idx]
            lr_batch = lr_img.unsqueeze(0).to(device)
            hr_batch = hr_img.unsqueeze(0).to(device)

            sr_baseline_out = baseline(lr_batch)
            sr_improved_out = improved(lr_batch)

            save_path = out_dir / f"{tag}{rank:02d}_sample{sample_idx:03d}_gap{row['gap_baseline_minus_improved_db']:+.2f}dB.png"
            save_composite(lr_batch, sr_baseline_out, sr_improved_out, hr_batch,
                           row["psnr_baseline_db"], row["psnr_improved_db"], save_path)

    avg_psnr_b = sum(r["psnr_baseline_db"] for r in csv_rows) / n
    avg_psnr_i = sum(r["psnr_improved_db"] for r in csv_rows) / n
    max_gap = max(r["gap_baseline_minus_improved_db"] for r in csv_rows)
    min_gap = min(r["gap_baseline_minus_improved_db"] for r in csv_rows)
    print(f"Đã tính PSNR/SSIM cho {n} mẫu (không lưu ảnh cho toàn bộ), lưu {len(selected)} ảnh "
          f"outlier (top-{k} mỗi chiều) vào {out_dir}/")
    print(f"PSNR trung bình trên {n} mẫu — baseline: {avg_psnr_b:.2f}dB | improved: {avg_psnr_i:.2f}dB")
    print(f"Khoảng cách PSNR (baseline - improved): lớn nhất (improved thua) = {max_gap:+.3f}dB, "
          f"nhỏ nhất (improved thắng) = {min_gap:+.3f}dB")
    print(f"Chi tiết từng mẫu ({n} dòng): {out_dir}/comparison_metrics.csv")
    print(f"Ảnh outlier: {out_dir}/worst00_... (improved kém nhất) ... {out_dir}/best00_... (improved tốt nhất)")


if __name__ == "__main__":
    main()
