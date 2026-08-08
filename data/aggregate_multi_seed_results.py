"""
Gộp kết quả nhiều seed cho từng domain, tính trung bình +- độ lệch chuẩn.
Tự động cảnh báo nếu chênh lệch giữa 2 domain nhỏ hơn độ lệch chuẩn quan sát
được (dấu hiệu chênh lệch có thể chỉ là nhiễu ngẫu nhiên, không đủ tin cậy
để kết luận domain này "tốt hơn" domain kia).

Chạy:
    python data/aggregate_multi_seed_results.py --results_dir results/multi_seed \
        --out_csv results/multi_seed/multi_seed_summary.csv
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    by_domain = defaultdict(list)
    for f in sorted(Path(args.results_dir).glob("*_seed*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        by_domain[data["test_domain"]].append(data["identity_accuracy"])

    if not by_domain:
        print(f"Không tìm thấy file *_seed*.json nào trong {args.results_dir}")
        return

    rows = []
    stats = {}
    for domain, values in by_domain.items():
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[domain] = (mean, std)
        rows.append({
            "domain": domain, "n_seeds": len(values),
            "mean_identity_accuracy": round(mean, 4),
            "std_identity_accuracy": round(std, 4),
            "min": round(min(values), 4), "max": round(max(values), 4),
            "all_values": ";".join(f"{v:.4f}" for v in values),
        })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Đã ghi {out_path}\n")
    print("== Kết quả trung bình +- độ lệch chuẩn theo domain ==")
    for domain, (mean, std) in stats.items():
        print(f"  {domain:<15} {mean:.4f} +- {std:.4f}  (n={len(by_domain[domain])} seed)")

    # So sánh từng cặp domain, cảnh báo nếu chênh lệch < độ lệch chuẩn gộp
    print("\n== Kiểm tra ý nghĩa chênh lệch giữa các domain ==")
    domains = list(stats.keys())
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            d1, d2 = domains[i], domains[j]
            mean1, std1 = stats[d1]
            mean2, std2 = stats[d2]
            diff = abs(mean1 - mean2)
            pooled_std = (std1 + std2) / 2
            if pooled_std > 0 and diff < pooled_std:
                print(f"  !!! {d1} vs {d2}: chênh lệch {diff:.4f} NHỎ HƠN độ lệch chuẩn "
                      f"trung bình ({pooled_std:.4f}) -> KHÔNG đủ tin cậy để kết luận "
                      f"domain nào tốt hơn, cần thêm seed.")
            else:
                better = d1 if mean1 > mean2 else d2
                print(f"  OK: {d1} vs {d2}: chênh lệch {diff:.4f} > độ lệch chuẩn "
                      f"({pooled_std:.4f}) -> '{better}' có vẻ tốt hơn thật, "
                      f"không chỉ do nhiễu ngẫu nhiên.")


if __name__ == "__main__":
    main()
