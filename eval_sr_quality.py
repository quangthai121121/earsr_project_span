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
from utils.metrics import (compute_psnr, compute_ssim, compute_psnr_roi, compute_ssim_roi,
                            count_params, count_params_deploy_mode, count_flops, measure_latency,
                            load_lpips_model, compute_lpips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True,
                     help="tên hiển thị trong CSV, ví dụ: edsr_teacher, span_baseline, span_improved")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--skip_lpips", action="store_true",
                     help="Bỏ qua tính LPIPS (nếu chưa cài package `lpips` hoặc muốn chạy nhanh "
                          "để kiểm tra pipeline trước) — CSV sẽ ghi 'NA' cho cột lpips.")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scale = cfg["image"]["scale"]
    hr_size = cfg["image"]["hr_size"]
    lr_size = hr_size // scale
    splits_root = cfg["paths"]["splits_root"]

    # [SỬA — đợt 7] return_bbox=True: lấy thêm vùng ROI thật (không phải viền
    # đệm đen letterbox) từ splits.json, dùng để đo PSNR/SSIM ĐÚNG trên vùng
    # tai thật — xem utils/metrics.py::compute_psnr_roi/compute_ssim_roi và
    # datasets/hrlr_pair_dataset.py. Vẫn giữ song song số liệu full-canvas
    # (đặt tên *_full_canvas) để minh bạch/đối chiếu, không âm thầm xoá số cũ.
    splits_json = f"{splits_root}/splits.json"
    test_set = HRLRPairDataset(f"{splits_root}/hr", f"{splits_root}/lr", "test",
                                return_bbox=True, splits_json=splits_json)
    loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4)

    model = build_sr_model(args.arch, scale)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # [MỚI — bổ sung journal Q1] LPIPS: chỉ số cảm nhận (perceptual), bổ sung
    # cho PSNR/SSIM vốn tương quan yếu với chất lượng cảm nhận thật — xem
    # utils/metrics.py::load_lpips_model()/compute_lpips(). Tải model 1 LẦN
    # ở đây (trước vòng lặp), KHÔNG tải lại mỗi ảnh. Đo ROI-only (đồng bộ với
    # PSNR/SSIM đã sửa đợt 7 — không tính viền đệm đen letterbox).
    lpips_model = None
    if not args.skip_lpips:
        try:
            lpips_model = load_lpips_model(device=device, net="alex")
        except ImportError as e:
            print(f"Cảnh báo: {e}. Bỏ qua LPIPS (điền NA). Chạy lại với package `lpips` đã cài "
                  f"để có số liệu đầy đủ cho bài báo.")

    psnr_roi_sum, ssim_roi_sum, lpips_roi_sum = 0.0, 0.0, 0.0
    psnr_full_sum, ssim_full_sum = 0.0, 0.0
    n = 0
    with torch.no_grad():
        for lr_img, hr_img, bbox in loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            x0_b, y0_b, w_b, h_b = bbox
            sr_img = model(lr_img)
            for i in range(sr_img.size(0)):
                psnr_full_sum += compute_psnr(sr_img[i:i + 1], hr_img[i:i + 1])
                ssim_full_sum += compute_ssim(sr_img[i:i + 1], hr_img[i:i + 1])

                bbox_i = (int(x0_b[i]), int(y0_b[i]), int(w_b[i]), int(h_b[i]))
                psnr_roi_sum += compute_psnr_roi(sr_img[i:i + 1], hr_img[i:i + 1], bbox_i)
                ssim_roi_sum += compute_ssim_roi(sr_img[i:i + 1], hr_img[i:i + 1], bbox_i)
                if lpips_model is not None:
                    x0, y0, w, h = bbox_i
                    sr_roi = sr_img[i:i + 1, :, y0:y0 + h, x0:x0 + w]
                    hr_roi = hr_img[i:i + 1, :, y0:y0 + h, x0:x0 + w]
                    lpips_roi_sum += compute_lpips(lpips_model, sr_roi, hr_roi)
                n += 1

    avg_psnr = psnr_roi_sum / n
    avg_ssim = ssim_roi_sum / n
    avg_lpips = (lpips_roi_sum / n) if lpips_model is not None else None
    avg_psnr_full = psnr_full_sum / n
    avg_ssim_full = ssim_full_sum / n

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
        "psnr_db": round(avg_psnr, 3),                      # [SỬA] giờ là ROI-only (vùng tai thật)
        "ssim": round(avg_ssim, 4),                          # [SỬA] giờ là ROI-only
        "lpips": round(avg_lpips, 4) if avg_lpips is not None else "NA",  # [MỚI] càng THẤP càng tốt
        "psnr_db_full_canvas": round(avg_psnr_full, 3),      # [MỚI] số cũ (gồm viền đen) — đối chiếu
        "ssim_full_canvas": round(avg_ssim_full, 4),         # [MỚI] số cũ (gồm viền đen) — đối chiếu
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
    print(f"psnr_db/ssim/lpips = CHỈ tính trên vùng ROI thật (không phải viền đệm đen letterbox) "
          f"— [SỬA đợt 7] dùng số liệu NÀY cho bài báo, so sánh được với literature SR chuẩn.")
    print(f"lpips = [MỚI] chỉ số cảm nhận (perceptual) — CÀNG THẤP CÀNG TỐT (ngược hướng với "
          f"PSNR/SSIM). 'NA' nếu chưa cài package `lpips` hoặc chạy với --skip_lpips.")
    print(f"psnr_db_full_canvas/ssim_full_canvas = số liệu CŨ (gồm cả viền đen) — giữ lại để "
          f"đối chiếu minh bạch, KHÔNG dùng để báo cáo trong bài báo (bị thổi phồng bởi viền đen).")
    print(f"{'=' * 60}\n")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(result.keys())
    write_header = not out_path.exists()

    if not write_header:
        # [MỚI — sửa lỗi phát hiện qua code review] Đợt 7 đổi schema của
        # `result` (thêm psnr_db_full_canvas/ssim_full_canvas): 9 cột cũ ->
        # 11 cột mới. Nếu out_csv đã tồn tại từ lần chạy TRƯỚC đợt 7 (header
        # 9 cột cũ trên đĩa) và ta chỉ eval lại (đúng hướng dẫn README mục
        # "Điểm 2 chỉ đổi cách đo, không cần train lại") mà KHÔNG xoá file cũ
        # trước, `write_header=False` (vì file đã tồn tại) nhưng vẫn
        # writerow() với 11 giá trị theo thứ tự MỚI — dữ liệu sẽ LỆCH CỘT so
        # với header 9-cột cũ trên đĩa, không có exception nào báo (CSV vẫn
        # "hợp lệ" về mặt cú pháp, chỉ sai vị trí). Chặn lại đây, báo lỗi rõ
        # ràng thay vì âm thầm ghi sai — cùng tinh thần "thà crash to còn hơn
        # sai âm thầm" đã áp dụng ở các chỗ khác (vd kiểm tra mutual-exclusivity
        # --init_ckpt/--init_ckpt_transfer trong train_recognition.py).
        with open(out_path, "r", newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != fieldnames:
            raise RuntimeError(
                f"LỖI: {out_path} đã tồn tại với schema KHÁC (header hiện có trên đĩa: "
                f"{existing_header}), không khớp schema của script hiện tại ({fieldnames}). "
                f"Ghi thêm vào đây sẽ làm LỆCH CỘT toàn bộ dữ liệu (giá trị nằm sai vị trí "
                f"header, KHÔNG có lỗi runtime nào báo hiệu).\n"
                f"Cách xử lý: đổi tên/backup {out_path} (giữ lại nếu cần số liệu cũ để đối "
                f"chiếu) rồi chạy lại — script sẽ tự tạo file mới với header đúng."
            )

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(result)
    print(f"Đã ghi (append) vào {out_path}")


if __name__ == "__main__":
    main()
