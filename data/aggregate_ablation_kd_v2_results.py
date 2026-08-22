"""
[MỚI] Gộp kết quả ablation 2x2 cho feature-level KD (lambda_feat) x multi-judge
identity loss (identity_judges) — xem pipeline/run_ablation_kd_v2.sh và
train_sr_distill.py. File RIÊNG BIỆT với data/aggregate_ablation_results.py
(glob pattern khác: "ablation_kdv2_*.json" thay vì "ablation_*.json") để
KHÔNG đụng/ghi đè lên ablation 4-cấu-hình cũ (pixel_only/pixel_distill/
pixel_identity/full) đã có sẵn trong project.

Chạy:
    python data/aggregate_ablation_kd_v2_results.py --results_dir results \
        --out_csv results/ablation_kd_v2.csv
"""
import argparse
import csv
import json
from pathlib import Path

# Khớp đúng CONFIGS trong pipeline/run_ablation_kd_v2.sh — (lambda_pixel,
# lambda_distill, lambda_feat, lambda_identity).
LAMBDA_MAP = {
    "kdv2_baseline": (1.0, 1.0, 0.0, 0.0),
    "kdv2_feat": (1.0, 1.0, 0.5, 0.0),
    "kdv2_multijudge": (1.0, 1.0, 0.0, 0.1),
    "kdv2_full": (1.0, 1.0, 0.5, 0.1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = []
    for f in sorted(Path(args.results_dir).glob("ablation_kdv2_*.json")):
        name = f.stem.replace("ablation_", "")
        # [SỬA — nhất quán với aggregate_ablation_results.py] tên lạ (không khớp
        # LAMBDA_MAP) bị bỏ qua tường minh thay vì ghi hàng rác với lambda=None.
        if name not in LAMBDA_MAP:
            print(f"[bỏ qua] {f.name}: config_name '{name}' không có trong LAMBDA_MAP")
            continue
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        lp, ld, lf, li = LAMBDA_MAP[name]
        rows.append({
            "config_name": name,
            "lambda_pixel": lp,
            "lambda_distill": ld,
            "lambda_feat": lf,
            "lambda_identity": li,
            "identity_accuracy": data["identity_accuracy"],
            "gender_accuracy": data["gender_accuracy"],
            "backbone": data["backbone"],
        })

    if not rows:
        print("Không tìm thấy file ablation_kdv2_*.json nào trong", args.results_dir)
        return

    # [MỚI — phát hiện qua code review] cảnh báo rõ nếu thiếu cấu hình so với
    # kỳ vọng (LAMBDA_MAP), tránh REPORT.md/CSV trông "đầy đủ" trong khi thực
    # ra có thí nghiệm chưa chạy xong bị lặng lẽ bỏ qua.
    found_names = {r["config_name"] for r in rows}
    missing = sorted(set(LAMBDA_MAP) - found_names)
    if missing:
        print(f"!!! CẢNH BÁO: thiếu {len(missing)}/{len(LAMBDA_MAP)} cấu hình kỳ vọng: {missing}")

    fieldnames = ["config_name", "lambda_pixel", "lambda_distill", "lambda_feat",
                  "lambda_identity", "identity_accuracy", "gender_accuracy", "backbone"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Đã ghi {args.out_csv} — {len(rows)} dòng.")
    baseline = next((r for r in rows if r["config_name"] == "kdv2_baseline"), None)
    if baseline:
        print("\nSo với kdv2_baseline (recipe cũ, chỉ pixel+distill output-level):")
        for r in rows:
            if r["config_name"] == "kdv2_baseline":
                continue
            diff = r["identity_accuracy"] - baseline["identity_accuracy"]
            print(f"  {r['config_name']:20s} identity_accuracy={r['identity_accuracy']:.4f} "
                  f"(diff={diff:+.4f})")


if __name__ == "__main__":
    main()
