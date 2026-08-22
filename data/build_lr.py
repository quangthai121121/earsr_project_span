"""
Bước 2: từ splits.json, tạo tập HR (letterbox — giữ tỷ lệ, không méo) và
tập LR (downsample bicubic từ HR đã letterbox).

Chạy:
    python data/build_lr.py --splits splits/splits.json --hr_size 128 --scale 4 --out_dir splits
"""
import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.letterbox import letterbox_resize  # noqa: E402


def process_split(entries, split_name, hr_size, scale, out_dir):
    lr_size = hr_size // scale
    hr_out = Path(out_dir) / "hr" / split_name
    lr_out = Path(out_dir) / "lr" / split_name
    hr_out.mkdir(parents=True, exist_ok=True)
    lr_out.mkdir(parents=True, exist_ok=True)

    for entry in tqdm(entries, desc=f"[{split_name}] tạo HR/LR"):
        src = Path(entry["path"])
        pid = entry["person_id"]
        (hr_out / pid).mkdir(exist_ok=True)
        (lr_out / pid).mkdir(exist_ok=True)

        with Image.open(src) as im:
            im = im.convert("RGB")
            hr_img = letterbox_resize(im, hr_size)          # giữ tỷ lệ, không méo
            lr_img = hr_img.resize((lr_size, lr_size), Image.BICUBIC)

        hr_img.save(hr_out / pid / src.name)
        lr_img.save(lr_out / pid / src.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--hr_size", type=int, default=128)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    import json
    with open(args.splits, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # [SỬA — bug phát hiện qua code review] trước đây chỉ mkdir(exist_ok=True),
    # không xoá thư mục đích cũ -> nếu splits.json thay đổi (ví dụ sau khi sửa
    # lỗi split-by-identity), ảnh của lần chạy TRƯỚC vẫn còn sót lại lẫn với
    # ảnh mới trên đĩa (số file thật > số entry trong splits.json). Không gây
    # data leakage khi train (EarDataset đọc danh sách ảnh từ splits.json, không
    # quét thư mục) nhưng phá hỏng mục đích kiểm tra vật lý bằng mắt và lãng phí
    # ổ đĩa. Xoá sạch trước khi ghi lại để splits/hr, splits/lr luôn khớp CHÍNH
    # XÁC với splits.json hiện tại.
    for sub in ("hr", "lr"):
        sub_dir = Path(args.out_dir) / sub
        if sub_dir.exists():
            print(f"Xoá thư mục cũ: {sub_dir}")
            shutil.rmtree(sub_dir)

    for split_name, entries in splits.items():
        process_split(entries, split_name, args.hr_size, args.scale, args.out_dir)

    print("Hoàn tất tạo tập HR (letterbox) và LR cho train/val/test.")
    print(f"LR có kích thước {args.hr_size // args.scale}x{args.hr_size // args.scale}, "
          f"HR có kích thước {args.hr_size}x{args.hr_size}.")


if __name__ == "__main__":
    main()
