"""
Gộp kết quả nhiều seed cho từng (backbone, domain), tính trung bình +- độ
lệch chuẩn. Tự động cảnh báo nếu chênh lệch giữa 2 domain (CÙNG backbone)
nhỏ hơn độ lệch chuẩn quan sát được (dấu hiệu chênh lệch có thể chỉ là nhiễu
ngẫu nhiên, không đủ tin cậy để kết luận domain này "tốt hơn" domain kia).

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

    # key = (backbone, domain) -> list giá trị accuracy qua các seed
    by_key = defaultdict(list)
    for f in sorted(Path(args.results_dir).glob("*_seed*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        key = (data["backbone"], data["test_domain"])
        by_key[key].append(data["identity_accuracy"])

    if not by_key:
        print(f"Không tìm thấy file *_seed*.json nào trong {args.results_dir}")
        return

    rows = []
    stats = {}
    for (backbone, domain), values in by_key.items():
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[(backbone, domain)] = (mean, std)
        rows.append({
            "backbone": backbone, "domain": domain, "n_seeds": len(values),
            "mean_identity_accuracy": round(mean, 4),
            "std_identity_accuracy": round(std, 4),
            "min": round(min(values), 4), "max": round(max(values), 4),
            "all_values": ";".join(f"{v:.4f}" for v in values),
        })

    # sắp theo backbone rồi theo thứ tự domain quen thuộc
    domain_order = {"lr": 0, "sr_baseline": 1, "sr_improved": 2}
    rows.sort(key=lambda r: (r["backbone"], domain_order.get(r["domain"], 9)))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Đã ghi {out_path}\n")

    backbones = sorted({b for b, d in stats.keys()})
    for backbone in backbones:
        print(f"\n=== Backbone: {backbone} ===")
        print("-- Trung bình +- độ lệch chuẩn theo domain --")
        domains_this_backbone = [d for b, d in stats.keys() if b == backbone]
        domains_this_backbone.sort(key=lambda d: domain_order.get(d, 9))
        for domain in domains_this_backbone:
            mean, std = stats[(backbone, domain)]
            n = len(by_key[(backbone, domain)])
            print(f"  {domain:<15} {mean:.4f} +- {std:.4f}  (n={n} seed)")

        print("-- Kiểm tra ý nghĩa chênh lệch giữa các domain (cùng backbone) --")
        for i in range(len(domains_this_backbone)):
            for j in range(i + 1, len(domains_this_backbone)):
                d1, d2 = domains_this_backbone[i], domains_this_backbone[j]
                mean1, std1 = stats[(backbone, d1)]
                mean2, std2 = stats[(backbone, d2)]
                diff = abs(mean1 - mean2)
                pooled_std = (std1 + std2) / 2
                if pooled_std > 0 and diff < pooled_std:
                    print(f"  !!! {d1} vs {d2}: chênh lệch {diff:.4f} NHỎ HƠN độ lệch chuẩn "
                          f"trung bình ({pooled_std:.4f}) -> KHÔNG đủ tin cậy, cần thêm seed.")
                else:
                    better = d1 if mean1 > mean2 else d2
                    print(f"  OK: {d1} vs {d2}: chênh lệch {diff:.4f} > độ lệch chuẩn "
                          f"({pooled_std:.4f}) -> '{better}' tốt hơn thật, không chỉ do nhiễu.")


if __name__ == "__main__":
    main()
