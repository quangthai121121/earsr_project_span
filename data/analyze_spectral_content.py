"""
[MỚI — Trụ cột 6, reviewer round 2, Phần A] So sánh phổ công suất (power
spectral density, PSD) theo bán kính tần số giữa HR (ground truth), no-SR
(bicubic), span_tiny, span_baseline, span_large trên cùng tập ảnh test --
trả lời "việc nén sâu 6->3 khối THỰC SỰ làm mất thông tin ở dải tần số nào?"

Bổ trợ cho eval_frequency_ablation.py (Phần B, kiểm tra nhận dạng CẦN dải
tần số nào): script này đo phía SR (nén MẤT dải tần số nào), không đụng gì
tới model nhận dạng. Cả 2 phần dùng chung utils/frequency_filters.py.

KHÔNG cần train/sinh ảnh gì mới -- chỉ đọc lại các domain ảnh ĐÃ CÓ SẴN trên
đĩa (splits/hr, splits/lr, splits/sr_baseline, splits/sr_improved,
splits/sr_span_large), tính PSD trung bình-hoá theo bán kính cho từng ảnh
test rồi lấy trung bình toàn tập.

Chạy:
    python data/analyze_spectral_content.py --config configs/config.yaml \
        --out_csv results/frequency_ablation/spectral_content.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.frequency_filters import radial_power_spectrum  # noqa: E402

DOMAINS = {
    "hr": "HR (ground truth)",
    "lr": "no-SR (bicubic)",
    "sr_baseline": "span_baseline",
    "sr_improved": "span_tiny",
    "sr_span_large": "span_large",
}
N_BINS = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n_bins", type=int, default=N_BINS)
    ap.add_argument("--max_images", type=int, default=None,
                     help="[TÙY CHỌN] giới hạn số ảnh test dùng để tính PSD trung bình (mặc định: "
                          "dùng TOÀN BỘ tập test) -- chỉ để chạy nhanh khi kiểm tra pipeline.")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    splits_root = cfg["paths"]["splits_root"]
    image_size = cfg["image"]["hr_size"]
    splits_json = f"{splits_root}/splits.json"

    with open(splits_json, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_entries = splits["test"]
    if args.max_images is not None:
        test_entries = test_entries[:args.max_images]

    missing_domains = [d for d in DOMAINS if not (Path(splits_root) / d).is_dir()]
    if missing_domains:
        raise FileNotFoundError(
            f"Thiếu domain(s) {missing_domains} trong {splits_root} -- chạy pipeline chính "
            f"(build_lr/build_sr cho span_tiny/span_baseline/span_large) trước.")

    rows = []
    for domain, label in DOMAINS.items():
        domain_test_dir = Path(splits_root) / domain / "test"
        psd_sum = np.zeros(args.n_bins)
        n_used, n_missing = 0, 0
        for entry in test_entries:
            pid = entry["person_id"]
            filename = Path(entry["path"]).name
            img_path = domain_test_dir / pid / filename
            if not img_path.exists():
                n_missing += 1
                continue
            img = Image.open(img_path).convert("RGB")
            if img.size != (image_size, image_size):
                img = img.resize((image_size, image_size), Image.BICUBIC)
            psd_sum += radial_power_spectrum(img, n_bins=args.n_bins)
            n_used += 1

        if n_used == 0:
            print(f"CẢNH BÁO: domain '{domain}' không đọc được ảnh nào (0/{len(test_entries)}), bỏ qua.")
            continue
        if n_missing > 0:
            print(f"CẢNH BÁO: domain '{domain}' thiếu {n_missing}/{len(test_entries)} ảnh test.")

        psd_mean = psd_sum / n_used
        for bin_idx, val in enumerate(psd_mean):
            rows.append({
                "domain": domain, "label": label, "n_images": n_used,
                "freq_bin": bin_idx, "freq_bin_frac": round(bin_idx / args.n_bins, 4),
                "mean_normalized_power": round(float(val), 6),
            })
        print(f"{label:<20} ({domain}): {n_used} ảnh, PSD 5 bin thấp nhất = "
              f"{[round(v, 4) for v in psd_mean[:5]]}, 5 bin cao nhất = "
              f"{[round(v, 4) for v in psd_mean[-5:]]}")

    if not rows:
        print("Không có domain nào đọc được -- không ghi CSV.")
        return

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nĐã ghi {out_path}")

    # [MỚI] So sánh trực tiếp span_tiny vs span_baseline/span_large: ở dải tần số
    # nào 2 đường cong PSD lệch nhau NHIỀU nhất? Trả lời trực tiếp "nén mất
    # thông tin ở tần số nào" bằng 1 con số dễ đọc thay vì phải tự so 2 cột CSV.
    by_domain = {}
    for r in rows:
        by_domain.setdefault(r["domain"], [0.0] * args.n_bins)[r["freq_bin"]] = r["mean_normalized_power"]

    if "sr_improved" in by_domain:
        print("\n== So sánh span_tiny với các domain khác, theo dải tần số (thấp -> cao) ==")
        tiny = np.array(by_domain["sr_improved"])
        low_half = slice(0, args.n_bins // 2)
        high_half = slice(args.n_bins // 2, args.n_bins)
        for other in ["sr_baseline", "sr_span_large", "hr", "lr"]:
            if other not in by_domain:
                continue
            other_arr = np.array(by_domain[other])
            diff = np.abs(tiny - other_arr)
            print(f"  span_tiny vs {DOMAINS[other]:<20}: |diff| dải tần thấp (nửa dưới) = "
                  f"{diff[low_half].sum():.4f} | dải tần cao (nửa trên) = {diff[high_half].sum():.4f}")


if __name__ == "__main__":
    main()
