"""
Tổng hợp kết quả quét lambda_identity (nhiều mức x nhiều seed), tính trung
bình +- độ lệch chuẩn, và kiểm định ý nghĩa thống kê bằng PAIRED T-TEST
(so với mức lambda_identity=0, tức "chỉ distillation" — cấu hình đang thắng
trong ablation cũ) — thay vì chỉ so sánh thô "chênh lệch > độ lệch chuẩn"
như bước kiểm tra nhanh trước đó.

LƯU Ý QUAN TRỌNG: với chỉ 3 seed/cấu hình, p-value có độ tin cậy thấp (bậc tự
do = 2) — đây là bước SÀNG LỌC nhanh để chọn ứng viên, không phải kết quả cuối
dùng để viết vào bài báo. Sau khi chọn được lambda_identity tối ưu, PHẢI chạy
lại với ít nhất 5 seed cho riêng cấu hình đã chọn (Bước 2) để có số liệu đủ
mạnh cho journal.

Chạy:
    python data/aggregate_lambda_sweep.py --results_dir results/lambda_sweep \
        --out_csv results/lambda_sweep/lambda_sweep_summary.csv
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    # Đọc toàn bộ file acc_lid<X>_seed<Y>.json, gom theo giá trị lambda_identity
    by_lambda = defaultdict(dict)  # {lambda_id_str: {seed: accuracy}}
    for f in sorted(Path(args.results_dir).glob("acc_lid*_seed*.json")):
        # tên dạng: acc_lid0.1_seed42.json
        stem = f.stem.replace("acc_", "")
        lid_part, seed_part = stem.split("_seed")
        lambda_id = lid_part.replace("lid", "")
        seed = seed_part
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        by_lambda[lambda_id][seed] = data["identity_accuracy"]

    if not by_lambda:
        print(f"Không tìm thấy file acc_lid*_seed*.json nào trong {args.results_dir}")
        return

    # Sắp theo giá trị lambda tăng dần
    lambdas_sorted = sorted(by_lambda.keys(), key=lambda x: float(x))

    rows = []
    stats_by_lambda = {}
    for lam in lambdas_sorted:
        values = list(by_lambda[lam].values())
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        stats_by_lambda[lam] = values
        rows.append({
            "lambda_identity": lam, "n_seeds": len(values),
            "mean_identity_accuracy": round(mean, 4),
            "std_identity_accuracy": round(std, 4),
            "all_values": ";".join(f"{v:.4f}" for v in values),
        })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Đã ghi {out_path}\n")
    print("== Trung bình +- độ lệch chuẩn theo từng mức lambda_identity ==")
    for lam in lambdas_sorted:
        vals = stats_by_lambda[lam]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  lambda_identity={lam:<6} {mean:.4f} +- {std:.4f}  (n={len(vals)})")

    # Kiểm định paired t-test: so mỗi mức lambda với mức 0.0 (baseline: chỉ distillation)
    if "0.0" in stats_by_lambda:
        baseline_vals = stats_by_lambda["0.0"]
        print(f"\n== Paired t-test so với lambda_identity=0.0 (chỉ distillation) ==")
        print("LƯU Ý: n=3 seed cho p-value độ tin cậy THẤP — chỉ dùng để sàng lọc "
              "ứng viên, không phải kết luận cuối cùng.\n")
        for lam in lambdas_sorted:
            if lam == "0.0":
                continue
            vals = stats_by_lambda[lam]
            if len(vals) != len(baseline_vals):
                print(f"  lambda_identity={lam}: số seed không khớp với baseline, bỏ qua t-test")
                continue
            t_stat, p_value = stats.ttest_rel(vals, baseline_vals)
            mean_diff = statistics.mean(vals) - statistics.mean(baseline_vals)
            direction = "TỐT HƠN" if mean_diff > 0 else "TỆ HƠN"
            sig = "có xu hướng ý nghĩa (p<0.10)" if p_value < 0.10 else "chưa rõ ý nghĩa (p>=0.10)"
            print(f"  lambda_identity={lam:<6} chênh lệch={mean_diff:+.4f}  "
                  f"p-value={p_value:.4f}  {direction}, {sig}")

    best_lam = max(lambdas_sorted, key=lambda x: statistics.mean(stats_by_lambda[x]))
    print(f"\n>>> Mức lambda_identity có trung bình accuracy cao nhất: {best_lam} "
          f"(mean={statistics.mean(stats_by_lambda[best_lam]):.4f})")
    print(">>> Xem thêm kết quả t-test ở trên trước khi chốt — mức cao nhất chưa "
          "chắc đã KHÁC BIỆT CÓ Ý NGHĨA so với các mức lân cận.")


if __name__ == "__main__":
    main()
