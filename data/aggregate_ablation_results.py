"""
Gộp kết quả 4 cấu hình ablation (pixel_only / pixel_distill / pixel_identity /
full) thành 1 bảng CSV, kèm giá trị lambda tương ứng — sẵn sàng dán vào bảng
ablation trong docs/03_span_improvement.md.

Chạy:
    python data/aggregate_ablation_results.py --results_dir results --out_csv results/ablation.csv
"""
import argparse
import csv
import json
from pathlib import Path

LAMBDA_MAP = {
    "pixel_only": (1.0, 0.0, 0.0),
    "pixel_distill": (1.0, 1.0, 0.0),
    "pixel_identity": (1.0, 0.0, 0.1),
    "full": (1.0, 1.0, 0.1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = []
    for f in sorted(Path(args.results_dir).glob("ablation_*.json")):
        name = f.stem.replace("ablation_", "")
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        lp, ld, li = LAMBDA_MAP.get(name, (None, None, None))
        rows.append({
            "config_name": name,
            "lambda_pixel": lp,
            "lambda_distill": ld,
            "lambda_identity": li,
            "identity_accuracy": data["identity_accuracy"],
            "gender_accuracy": data["gender_accuracy"],
            "backbone": data["backbone"],
        })

    if not rows:
        print("Không tìm thấy file ablation_*.json nào trong", args.results_dir)
        return

    fieldnames = ["config_name", "lambda_pixel", "lambda_distill", "lambda_identity",
                  "identity_accuracy", "gender_accuracy", "backbone"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Đã ghi {args.out_csv} — {len(rows)} dòng.")


if __name__ == "__main__":
    main()
