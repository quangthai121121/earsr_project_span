"""
[MỚI] Tổng hợp multi-seed cho real_lr_holdout (xem pipeline/
run_real_lr_holdout_multiseed.sh) — trả lời câu hỏi: đảo ngược no-SR >
span_tiny > span_baseline quan sát được ở n=1 (Section res-main) có LẶP LẠI
qua nhiều seed hay chỉ là nhiễu của riêng 1 seed checkpoint?

Tái sử dụng ĐÚNG bộ hàm thống kê đã kiểm chứng trong
data/aggregate_multi_seed_results.py (paired t-test, Cohen's dz, Wilcoxon,
CI95%, MDES) — không viết lại logic thống kê ở 2 nơi khác nhau.

Chạy:
    python data/aggregate_real_lr_holdout_multiseed.py \
        --results_dir results/real_lr_holdout_multiseed \
        --out_csv results/real_lr_holdout_multiseed/real_lr_holdout_multiseed_summary.csv
"""
import argparse
import csv
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_multi_seed_results import cohens_d_paired, paired_ci95, wilcoxon_paired_p, mdes_paired  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    # condition -> {seed: identity_accuracy}
    by_condition = defaultdict(dict)
    n_files = 0
    for f in sorted(Path(args.results_dir).glob("real_lr_holdout_seed*.csv")):
        n_files += 1
        with open(f, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                seed = row["seed"]
                by_condition[row["condition"]][seed] = float(row["identity_accuracy"])

    if n_files == 0:
        print(f"Không tìm thấy file real_lr_holdout_seed*.csv nào trong {args.results_dir}")
        return

    conditions = ["no_sr", "sr_baseline", "sr_improved"]
    missing = [c for c in conditions if c not in by_condition]
    if missing:
        print(f"CẢNH BÁO: thiếu điều kiện {missing} trong dữ liệu đọc được.")

    # ================= Bảng 1: trung bình +- độ lệch chuẩn theo điều kiện =================
    rows = []
    for condition in conditions:
        if condition not in by_condition:
            continue
        vals = list(by_condition[condition].values())
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        rows.append({
            "condition": condition, "n_seeds": len(vals),
            "mean_identity_accuracy": round(mean, 4), "std_identity_accuracy": round(std, 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4),
            "all_values": ";".join(f"{v:.4f}" for v in vals),
            # [SỬA — bug phát hiện qua review] sorted() thường (string) xếp
            # "123" trước "42" (so sánh alphabet, không phải số) -> cột hiển
            # thị lệch thứ tự dù không ảnh hưởng thống kê (mean/std/paired
            # test không phụ thuộc thứ tự). Sort theo int cho đúng trực giác
            # người đọc CSV.
            "seeds": ";".join(sorted(by_condition[condition].keys(), key=int)),
        })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã ghi {out_path}\n")

    # ================= Bảng 2: kiểm định từng cặp điều kiện =================
    present_conditions = [c for c in conditions if c in by_condition]
    pairs = list(combinations(present_conditions, 2))
    n_comparisons = max(1, len(pairs))
    pairwise_rows = []
    for cond_a, cond_b in pairs:
        seeds_a, seeds_b = by_condition[cond_a], by_condition[cond_b]
        common_seeds = sorted(set(seeds_a.keys()) & set(seeds_b.keys()), key=int)
        if len(common_seeds) < 2:
            pairwise_rows.append({
                "condition_a": cond_a, "condition_b": cond_b, "n_common_seeds": len(common_seeds),
                "mean_diff_b_minus_a": "", "ci95_lower": "", "ci95_upper": "", "cohens_d": "",
                "p_raw": "", "p_bonferroni": "", "wilcoxon_p": "",
                "mdes_d_alpha10": "", "mdes_accuracy_alpha10": "",
                "n_comparisons": n_comparisons, "note": "< 2 seed chung -> không đủ để kiểm định",
            })
            continue
        vals_a = [seeds_a[s] for s in common_seeds]
        vals_b = [seeds_b[s] for s in common_seeds]
        mean_diff = statistics.mean(vals_b) - statistics.mean(vals_a)
        _, p_raw = stats.ttest_rel(vals_b, vals_a)
        # [SỬA — bug phát hiện qua test end-to-end] Nếu hiệu số bằng NHAU
        # TUYỆT ĐỐI qua mọi seed (std_diff=0 -- ví dụ checkpoint chưa train
        # thật, chỉ xảy ra trong test, gần như không thể với dữ liệu thật 173
        # ảnh qua nhiều seed độc lập), ttest_rel trả p_raw=NaN (0/0). Trước
        # đây `min(1.0, nan * n_comparisons)` TÌNH CỜ ra 1.0 nhờ cách
        # min() xử lý NaN trong Python (nan < 1.0 là False), KHÔNG PHẢI do
        # chủ động xử lý -- đổi thứ tự tham số min() sẽ cho NaN thay vì 1.0.
        # Chặn tường minh, cùng tinh thần phòng vệ số học đã áp dụng cho
        # cohens_d_paired() trong aggregate_multi_seed_results.py.
        p_bonferroni = "NA" if p_raw != p_raw else min(1.0, p_raw * n_comparisons)
        d = cohens_d_paired(vals_b, vals_a)
        ci_lo, ci_hi = paired_ci95(vals_b, vals_a)
        wilcoxon_p = wilcoxon_paired_p(vals_b, vals_a)
        mdes_d_a10, mdes_acc_a10 = mdes_paired(vals_b, vals_a, alpha=0.10)
        pairwise_rows.append({
            "condition_a": cond_a, "condition_b": cond_b, "n_common_seeds": len(common_seeds),
            "mean_diff_b_minus_a": round(mean_diff, 4),
            "ci95_lower": round(ci_lo, 4) if ci_lo is not None else "NA",
            "ci95_upper": round(ci_hi, 4) if ci_hi is not None else "NA",
            "cohens_d": round(d, 4) if d is not None else "NA (std hiệu số~0)",
            "p_raw": round(p_raw, 4), "p_bonferroni": p_bonferroni if p_bonferroni == "NA" else round(p_bonferroni, 4),
            "wilcoxon_p": round(wilcoxon_p, 4) if wilcoxon_p is not None else "NA",
            "mdes_d_alpha10": round(mdes_d_a10, 3) if mdes_d_a10 is not None else "NA",
            "mdes_accuracy_alpha10": round(mdes_acc_a10, 4) if mdes_acc_a10 is not None else "NA",
            "n_comparisons": n_comparisons,
            "note": "hiệu số bằng 0 tuyệt đối qua mọi seed -> không kiểm định được" if p_bonferroni == "NA"
                    else ("có ý nghĩa sau Bonferroni (p<0.10)" if p_bonferroni < 0.10
                          else "chưa đủ ý nghĩa sau Bonferroni (p>=0.10)"),
        })

    pairwise_path = out_path.with_name(out_path.stem + "_pairwise.csv")
    with open(pairwise_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairwise_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pairwise_rows)
    print(f"Đã ghi {pairwise_path}\n")

    print("=== Tóm tắt ===")
    for r in rows:
        print(f"  {r['condition']:<14} {r['mean_identity_accuracy']:.4f} +- "
              f"{r['std_identity_accuracy']:.4f}  (n={r['n_seeds']} seed)")
    for pr in pairwise_rows:
        if pr["mean_diff_b_minus_a"] == "":
            print(f"  {pr['condition_a']} vs {pr['condition_b']}: {pr['note']}")
            continue
        print(f"  {pr['condition_a']} vs {pr['condition_b']}: diff(b-a)={pr['mean_diff_b_minus_a']:+.4f}  "
              f"d={pr['cohens_d']}  p_raw={pr['p_raw']}  p_bonferroni={pr['p_bonferroni']}  "
              f"({pr['note']})")


if __name__ == "__main__":
    main()
