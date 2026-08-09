"""
Đo chất lượng ảnh SR (PSNR, SSIM) + hiệu năng tính toán (params, FLOPs, latency)
cho MỘT model SR trên tập test — độc lập với accuracy nhận diện downstream.
Ghi (append) kết quả vào 1 file CSV dùng chung cho nhiều model.

Chạy ví dụ (gọi 3 lần cho 3 model, cùng ghi vào 1 file):
    python eval_sr_quality.py --config configs/config.yaml \
        --arch edsr --ckpt runs/sr_edsr/best.pt --label edsr_teacher \
        --out_csv results/sr_quality.csv

    python eval_sr_quality.py --config configs/config.yaml \
        --arch span_official --ckpt runs/sr_span_official/best.pt --label span_baseline \
        --out_csv results/sr_quality.csv

    python eval_sr_quality.py --config configs/config.yaml \
        --arch span_official --ckpt runs/sr_improved_span_official/best.pt --label span_improved \
        --out_csv results/sr_quality.csv
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.hrlr_pair_dataset import HRLRPairDataset
from models.sr_models import build_sr_model
from utils.metrics import (compute_psnr, compute_ssim, count_params,
                            count_params_deploy_mode, count_flops, measure_latency)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True,
                     help="tên hiển thị trong CSV, ví dụ: edsr_teacher, span_baseline, span_improved")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scale = cfg["image"]["scale"]
    hr_size = cfg["image"]["hr_size"]
    lr_size = hr_size // scale
    splits_root = cfg["paths"]["splits_root"]

    test_set = HRLRPairDataset(f"{splits_root}/hr", f"{splits_root}/lr", "test")
    loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4)

    model = build_sr_model(args.arch, scale)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for lr_img, hr_img in loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            sr_img = model(lr_img)
            for i in range(sr_img.size(0)):
                psnr_sum += compute_psnr(sr_img[i:i + 1], hr_img[i:i + 1])
                ssim_sum += compute_ssim(sr_img[i:i + 1], hr_img[i:i + 1])
                n += 1

    avg_psnr = psnr_sum / n
    avg_ssim = ssim_sum / n

    params_m = count_params(model)
    params_deploy_m = count_params_deploy_mode(model)
    try:
        flops_g = count_flops(model, (1, 3, lr_size, lr_size), device=device)
    except ImportError as e:
        print(f"Cảnh báo: {e}. Bỏ qua FLOPs (điền NA).")
        flops_g = None
    latency_ms = measure_latency(model, (1, 3, lr_size, lr_size), device)

    result = {
        "label": args.label,
        "arch": args.arch,
        "psnr_db": round(avg_psnr, 3),
        "ssim": round(avg_ssim, 4),
        "params_M": round(params_m, 3),
        "params_deploy_M": round(params_deploy_m, 4),
        "flops_G": round(flops_g, 4) if flops_g is not None else "NA",
        "latency_ms": round(latency_ms, 3),
        "n_test_images": n,
    }

    print(f"\n{'=' * 60}")
    print(f"CHẤT LƯỢNG SR | label={args.label} | arch={args.arch}")
    print(f"{'=' * 60}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"params_M = tổng CẢ 2 nhánh (train+eval) nếu model dùng reparameterization "
          f"(ví dụ span_official) — dùng để so sánh NỘI BỘ giữa các model trong project này.")
    print(f"params_deploy_M = CHỈ nhánh đã hợp nhất (deploy/inference thật) — dùng để so "
          f"sánh với số liệu TÁC GIẢ CÔNG BỐ trong bài báo gốc (thường báo cáo theo kiểu này).")
    print(f"{'=' * 60}\n")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(result)
    print(f"Đã ghi (append) vào {out_path}")


if __name__ == "__main__":
    main()
