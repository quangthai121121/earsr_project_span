"""
Gộp kết quả 4 cấu hình ablation (pixel_only / pixel_distill / pixel_identity /
full) thành 1 bảng CSV, kèm giá trị lambda tương ứng — sẵn sàng dán vào bảng
ablation trong bài báo (xem RUNBOOK_EarVN1.0.md mục 10.1).

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
        # [SỬA — bug phát hiện qua code review] glob "ablation_*.json" cũng khớp
        # "ablation_kdv2_*.json" (sinh bởi pipeline/run_ablation_kd_v2.sh, recipe
        # HOÀN TOÀN KHÁC — 2x2 feat x multijudge, không phải pixel/distill/identity
        # của ablation này) — nếu chạy lại aggregator này trên cùng thư mục
        # results/ đã có cả 2 loại file, các dòng "kdv2_*" sẽ lọt vào bảng với
        # lambda=None (không có trong LAMBDA_MAP) — bỏ qua tường minh thay vì
        # âm thầm ghi hàng rác vào CSV.
        if name not in LAMBDA_MAP:
            continue
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        lp, ld, li = LAMBDA_MAP[name]
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

    # [MỚI — phát hiện qua code review] cảnh báo rõ nếu thiếu cấu hình so với
    # kỳ vọng (LAMBDA_MAP), tránh CSV trông "đầy đủ" trong khi thực ra có thí
    # nghiệm chưa chạy xong bị lặng lẽ bỏ qua.
    found_names = {r["config_name"] for r in rows}
    missing = sorted(set(LAMBDA_MAP) - found_names)
    if missing:
        print(f"!!! CẢNH BÁO: thiếu {len(missing)}/{len(LAMBDA_MAP)} cấu hình kỳ vọng: {missing}")

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
