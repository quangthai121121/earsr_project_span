"""
Đọc splits/threshold_report.txt (sinh bởi data/prepare_splits.py --auto) và
cập nhật configs/config.yaml (grouping.hr_source_min, grouping.too_small_max,
image.hr_size) để các bước sau dùng đúng ngưỡng vừa chọn tự động — tránh phải
sửa tay, tránh lệch giữa report và config.

Dùng thay thế theo dòng (regex) thay vì yaml.safe_dump toàn bộ, để GIỮ NGUYÊN
các comment giải thích đã có sẵn trong config.yaml.
"""
import argparse
import re


def parse_report(path: str) -> dict:
    values = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            values[key] = int(val)
    return values


def patch_value(text: str, key: str, new_value: int) -> str:
    """Thay số sau 'key:' bằng new_value, giữ nguyên comment/indent phía sau."""
    pattern = re.compile(rf"^(\s*{re.escape(key)}:\s*)\d+(.*)$", re.MULTILINE)
    new_text, n = pattern.subn(rf"\g<1>{new_value}\g<2>", text)
    if n == 0:
        raise ValueError(f"Không tìm thấy key '{key}:' trong config để cập nhật.")
    return new_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    report = parse_report(args.report)

    with open(args.config, "r", encoding="utf-8") as f:
        text = f.read()

    text = patch_value(text, "hr_source_min", report["hr_source_min"])
    text = patch_value(text, "too_small_max", report["too_small_max"])
    text = patch_value(text, "hr_size", report["hr_size"])

    with open(args.config, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Đã cập nhật {args.config}: "
          f"hr_source_min={report['hr_source_min']}, "
          f"too_small_max={report['too_small_max']}, "
          f"hr_size={report['hr_size']}")


if __name__ == "__main__":
    main()
