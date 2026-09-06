"""
[MỚI — Trụ cột 6, Phần B, sửa hiệu năng phát hiện qua tự review] Bộ lọc tần
số (utils/frequency_filters.py) không phụ thuộc backbone/seed nhận dạng nào
-- CÙNG 1 ảnh đã lọc được dùng lại cho cả 11 backbone x 5 seed. Bản đầu của
eval_frequency_ablation.py lọc TRỰC TIẾP mỗi lần được gọi (55 lần độc lập,
mỗi backbone/seed lọc lại TOÀN BỘ tập test từ đầu) -- lãng phí ~55x công tính
FFT giống hệt nhau (đo thực tế: ~17 phút lãng phí trên toàn bộ lần chạy).
Script này BUILD MỘT LẦN 16 domain ảnh đã lọc (khớp đúng quy ước "build 1
lần, dùng lại nhiều lần" của build_sr.py/build_lr.py trong project này), rồi
eval_frequency_ablation.py chỉ ĐỌC LẠI domain có sẵn -- không lọc lại.

Chỉ build cho split "test" (ablation này chỉ eval trên test set, không cần
train/val) và chỉ cho 1 domain nguồn (mặc định sr_improved = span_tiny,
domain dùng xuyên suốt Phần B).

Chạy:
    python data/build_frequency_filtered_domains.py --config configs/config.yaml
"""
import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.frequency_filters import apply_frequency_filter  # noqa: E402

# PHẢI khớp CHÍNH XÁC danh sách cutoff trong eval_frequency_ablation.py --
# đổi 1 trong 2 nơi mà quên đổi nơi kia sẽ khiến eval_frequency_ablation.py
# tìm domain không tồn tại (lỗi rõ ràng, không âm thầm sai) hoặc bỏ sót điều
# kiện lọc mới (không lỗi, nhưng thiếu số liệu).
LOWPASS_CUTOFFS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
HIGHPASS_CUTOFFS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]


def domain_dir_name(mode: str, cutoff_frac: float) -> str:
    """Tên thư mục domain cho 1 điều kiện lọc -- DÙNG CHUNG giữa script này
    và eval_frequency_ablation.py (import trực tiếp hàm này, không copy-paste
    lại công thức đặt tên ở 2 nơi)."""
    return f"freq_{mode}_{cutoff_frac:.2f}".replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--source_domain", default="sr_improved",
                     help="domain ảnh NGUỒN để lọc (mặc định sr_improved = span_tiny, "
                          "domain dùng xuyên suốt Phần B của Trụ cột 6).")
    ap.add_argument("--out_root", default=None,
                     help="mặc định: <splits_root>/freq_ablation_<source_domain>")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    splits_root = cfg["paths"]["splits_root"]
    splits_json = f"{splits_root}/splits.json"
    source_dir = Path(splits_root) / args.source_domain / "test"
    out_root = Path(args.out_root) if args.out_root else Path(splits_root) / f"freq_ablation_{args.source_domain}"

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy {source_dir} -- domain '{args.source_domain}' chưa được build.")

    with open(splits_json, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_entries = splits["test"]

    conditions = [("lowpass", c) for c in LOWPASS_CUTOFFS] + [("highpass", c) for c in HIGHPASS_CUTOFFS]
    print(f"Nguồn: {source_dir} ({len(test_entries)} ảnh test)")
    print(f"Đích:  {out_root}/<16 thư mục điều kiện lọc>/test/...")

    for mode, cutoff in conditions:
        dname = domain_dir_name(mode, cutoff)
        dest_dir = out_root / dname / "test"

        n_existing = sum(1 for e in test_entries
                         if (dest_dir / e["person_id"] / Path(e["path"]).name).exists())
        if n_existing == len(test_entries):
            print(f">>> Bỏ qua {dname} (đã có đủ {len(test_entries)}/{len(test_entries)} ảnh)")
            continue
        if 0 < n_existing < len(test_entries):
            print(f">>> {dname}: tiếp tục dở dang ({n_existing}/{len(test_entries)} ảnh đã có)")

        n_written = 0
        for entry in test_entries:
            pid = entry["person_id"]
            filename = Path(entry["path"]).name
            src_path = source_dir / pid / filename
            dst_path = dest_dir / pid / filename
            if dst_path.exists():
                continue
            if not src_path.exists():
                print(f"CẢNH BÁO: thiếu ảnh nguồn {src_path}, bỏ qua.")
                continue
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            # [SỬA — bug phát hiện qua review lại lần 2] TRƯỚC ĐÂY resize lên
            # image_size RỒI mới lọc -- với domain đã đúng kích thước sẵn
            # (sr_improved/sr_baseline/sr_span_large/hr, đều = hr_size) thì vô
            # hại, NHƯNG nếu dùng --source_domain lr (20x20, NHỎ hơn hr_size)
            # sẽ lọc tần số SAU KHI đã nội suy phóng to -- lẫn tần số GIẢ do
            # bicubic sinh ra, đúng lỗi mà bản trước (filter trước Resize
            # trong EarDataset, xem lịch sử eval_frequency_ablation.py) đã cố
            # tình tránh. Lọc ở ĐÚNG độ phân giải gốc trên đĩa; việc phóng lên
            # image_size cho vừa input model đã được EarDataset tự làm ở bước
            # eval (evaluate_one_condition() -> EarDataset(..., image_size)).
            img = Image.open(src_path).convert("RGB")
            filtered = apply_frequency_filter(img, mode, cutoff)
            filtered.save(dst_path)
            n_written += 1
        print(f">>> {dname}: đã ghi {n_written} ảnh mới")

    print("\nHOÀN TẤT.")


if __name__ == "__main__":
    main()
