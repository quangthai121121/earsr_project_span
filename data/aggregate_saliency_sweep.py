"""
[MỚI] Tổng hợp kết quả quét lambda_saliency (nhiều mức x nhiều seed), tính
trung bình +- độ lệch chuẩn, và kiểm định ý nghĩa thống kê bằng PAIRED T-TEST
(so với mức lambda_saliency=0, tức "không dùng saliency-weighted loss") —
BẢN SAO CÓ SỬA TÊN BIẾN của data/aggregate_lambda_sweep.py (đã dùng cho
lambda_identity), xem file đó để biết chi tiết phương pháp thống kê.

LƯU Ý QUAN TRỌNG: với chỉ 3 seed/cấu hình, p-value có độ tin cậy thấp — đây
là bước SÀNG LỌC nhanh để chọn ứng viên, không phải kết quả cuối dùng để viết
vào bài báo. Sau khi chọn được lambda_saliency tối ưu, PHẢI chạy lại với ít
nhất 5 seed x cả 5 backbone (xem pipeline/run_multi_seed_kdv2.sh) trước khi
đưa vào bảng kết quả chính thức.

[MỚI — phát hiện qua review Q1] Thêm Wilcoxon signed-rank (robustness check
phi tham số, không giả định normality như t-test), khoảng tin cậy 95% cho
mean_diff (paired_ci95()), và MDES (mdes_paired() — định lượng "thiếu power
tới mức nào" bằng noncentral-t chính xác) — xem chi tiết trong
aggregate_lambda_sweep.py (cùng phương pháp, bản sao đổi tên biến).

Chạy:
    python data/aggregate_saliency_sweep.py --results_dir results/lambda_saliency_sweep \
        --out_csv results/lambda_saliency_sweep/saliency_sweep_summary.csv
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats
from scipy.optimize import brentq


def cohens_d_paired(values_a, values_b):
    """Cohen's d cho mẫu GHÉP CẶP (dz), xem cùng hàm trong
    aggregate_lambda_sweep.py để biết chi tiết diễn giải."""
    diffs = [a - b for a, b in zip(values_a, values_b)]
    if len(diffs) < 2:
        return None
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs)
    if std_diff < 1e-9:
        return None
    return mean_diff / std_diff


def paired_ci95(values_a, values_b):
    """[MỚI — phát hiện qua review Q1] Khoảng tin cậy 95% cho mean(a-b), xem
    cùng hàm trong aggregate_lambda_sweep.py để biết chi tiết diễn giải."""
    diffs = [a - b for a, b in zip(values_a, values_b)]
    n = len(diffs)
    if n < 2:
        return None, None
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs)
    se = std_diff / (n ** 0.5)
    if se < 1e-12:
        return mean_diff, mean_diff
    t_crit = stats.t.ppf(0.975, df=n - 1)
    margin = t_crit * se
    return mean_diff - margin, mean_diff + margin


def wilcoxon_paired_p(values_a, values_b):
    """[MỚI — phát hiện qua review Q1] Wilcoxon signed-rank test, xem cùng
    hàm trong aggregate_lambda_sweep.py để biết chi tiết diễn giải (robustness
    check phi tham số cho paired t-test, KHÔNG kỳ vọng đạt cùng ngưỡng ý
    nghĩa với n=3 seed)."""
    diffs = [a - b for a, b in zip(values_a, values_b)]
    if len(diffs) < 2 or all(abs(d) < 1e-12 for d in diffs):
        return None
    try:
        _, p = stats.wilcoxon(values_a, values_b)
        return p
    except ValueError:
        return None


def mdes_paired(values_a, values_b, alpha=0.05, power=0.80):
    """[MỚI — phát hiện qua review Q1] Minimum Detectable Effect Size (MDES) —
    xem giải thích đầy đủ trong data/aggregate_multi_seed_results.py cùng tên
    hàm (cùng phương pháp: noncentral-t chính xác, root-finding bằng brentq).
    Trả lời định lượng "n=3 seed + Bonferroni power thấp tới mức nào"."""
    diffs = [a - b for a, b in zip(values_a, values_b)]
    n = len(diffs)
    if n < 2:
        return None, None
    std_diff = statistics.stdev(diffs)
    if std_diff < 1e-9:
        return None, None

    df = n - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    def power_at_delta(delta):
        return (1 - stats.nct.cdf(t_crit, df, delta)) + stats.nct.cdf(-t_crit, df, delta)

    lo, hi = 1e-6, 3.0
    while power_at_delta(hi) < power:
        hi *= 1.5
        if hi > 30:
            return None, None
    delta = brentq(lambda d: power_at_delta(d) - power, lo, hi)
    mdes_d = delta / (n ** 0.5)
    return mdes_d, mdes_d * std_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    # Đọc toàn bộ file acc_lsal<X>_seed<Y>.json, gom theo giá trị lambda_saliency
    by_lambda = defaultdict(dict)  # {lambda_saliency_str: {seed: accuracy}}
    for f in sorted(Path(args.results_dir).glob("acc_lsal*_seed*.json")):
        # tên dạng: acc_lsal0.3_seed42.json
        stem = f.stem.replace("acc_", "")
        lsal_part, seed_part = stem.split("_seed")
        lambda_sal = lsal_part.replace("lsal", "")
        seed = seed_part
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        by_lambda[lambda_sal][seed] = data["identity_accuracy"]

    if not by_lambda:
        print(f"Không tìm thấy file acc_lsal*_seed*.json nào trong {args.results_dir}")
        return

    lambdas_sorted = sorted(by_lambda.keys(), key=lambda x: float(x))

    stats_by_lambda = {}
    for lam in lambdas_sorted:
        stats_by_lambda[lam] = list(by_lambda[lam].values())

    pvalues_by_lambda = {}
    cohens_d_by_lambda = {}
    ci95_by_lambda = {}
    wilcoxon_by_lambda = {}
    mdes_by_lambda = {}
    if "0.0" in by_lambda:
        baseline_by_seed = by_lambda["0.0"]
        candidate_lambdas = [lam for lam in lambdas_sorted if lam != "0.0"]
        n_comparisons = max(1, len(candidate_lambdas))
        for lam in candidate_lambdas:
            common_seeds = sorted(set(by_lambda[lam].keys()) & set(baseline_by_seed.keys()))
            if len(common_seeds) < len(by_lambda[lam]) or len(common_seeds) < len(baseline_by_seed):
                print(f"Cảnh báo: lambda_saliency={lam} và baseline (0.0) không cùng bộ seed "
                      f"đầy đủ — chỉ dùng {len(common_seeds)} seed chung để ghép cặp t-test/Cohen's d.")
            if len(common_seeds) < 2:
                continue
            vals = [by_lambda[lam][s] for s in common_seeds]
            baseline_vals_paired = [baseline_by_seed[s] for s in common_seeds]
            _, p_raw = stats.ttest_rel(vals, baseline_vals_paired)
            pvalues_by_lambda[lam] = (p_raw, min(1.0, p_raw * n_comparisons))
            cohens_d_by_lambda[lam] = cohens_d_paired(vals, baseline_vals_paired)
            ci95_by_lambda[lam] = paired_ci95(vals, baseline_vals_paired)
            wilcoxon_by_lambda[lam] = wilcoxon_paired_p(vals, baseline_vals_paired)
            # [SỬA — phát hiện qua review Q1] alpha=0.10 khớp đúng ngưỡng "có ý
            # nghĩa" thật dùng ở sig_raw/sig_corrected bên dưới (p<0.10).
            mdes_by_lambda[lam] = (
                mdes_paired(vals, baseline_vals_paired, alpha=0.10),
                mdes_paired(vals, baseline_vals_paired, alpha=0.10 / n_comparisons),
            )

    rows = []
    for lam in lambdas_sorted:
        values = stats_by_lambda[lam]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        p_raw, p_corrected = pvalues_by_lambda.get(lam, ("", ""))
        d = cohens_d_by_lambda.get(lam, "")
        ci_lo, ci_hi = ci95_by_lambda.get(lam, (None, None))
        wilcoxon_p = wilcoxon_by_lambda.get(lam)
        (mdes_d_a10, mdes_acc_a10), (mdes_d_bonf, mdes_acc_bonf) = mdes_by_lambda.get(
            lam, ((None, None), (None, None)))
        rows.append({
            "lambda_saliency": lam, "n_seeds": len(values),
            "mean_identity_accuracy": round(mean, 4),
            "std_identity_accuracy": round(std, 4),
            "p_value_vs_baseline_raw": round(p_raw, 4) if p_raw != "" else "",
            "p_value_vs_baseline_bonferroni": round(p_corrected, 4) if p_corrected != "" else "",
            "wilcoxon_p_vs_baseline": round(wilcoxon_p, 4) if wilcoxon_p is not None else "",
            "cohens_d_vs_baseline": round(d, 4) if isinstance(d, float) else "",
            "ci95_diff_lower": round(ci_lo, 4) if ci_lo is not None else "",
            "ci95_diff_upper": round(ci_hi, 4) if ci_hi is not None else "",
            "mdes_d_alpha10": round(mdes_d_a10, 3) if mdes_d_a10 is not None else "",
            "mdes_accuracy_alpha10": round(mdes_acc_a10, 4) if mdes_acc_a10 is not None else "",
            "mdes_d_bonferroni": round(mdes_d_bonf, 3) if mdes_d_bonf is not None else "",
            "mdes_accuracy_bonferroni": round(mdes_acc_bonf, 4) if mdes_acc_bonf is not None else "",
            "all_values": ";".join(f"{v:.4f}" for v in values),
        })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Đã ghi {out_path} (đã gồm p_value_vs_baseline_raw/_bonferroni + cohens_d_vs_baseline)\n")
    print("== Trung bình +- độ lệch chuẩn theo từng mức lambda_saliency ==")
    for lam in lambdas_sorted:
        vals = stats_by_lambda[lam]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  lambda_saliency={lam:<6} {mean:.4f} +- {std:.4f}  (n={len(vals)})")

    if "0.0" in stats_by_lambda:
        baseline_vals = stats_by_lambda["0.0"]
        candidate_lambdas = [lam for lam in lambdas_sorted if lam != "0.0"]
        n_comparisons = len(candidate_lambdas)
        print(f"\n== Paired t-test so với lambda_saliency=0.0 (tắt saliency-weighted loss) ==")
        print(f"LƯU Ý: n=3 seed cho p-value độ tin cậy THẤP — chỉ dùng để sàng lọc "
              f"ứng viên, không phải kết luận cuối cùng.")
        print(f"LƯU Ý THÊM: {n_comparisons} phép so sánh đồng thời với cùng baseline "
              f"-> đã hiệu chỉnh Bonferroni (p_corrected = min(1, p_raw x {n_comparisons})) "
              f"để không đánh giá quá lạc quan mức ý nghĩa thống kê.\n")
        for lam in candidate_lambdas:
            vals = stats_by_lambda[lam]
            if lam not in pvalues_by_lambda:
                print(f"  lambda_saliency={lam}: số seed không khớp với baseline, bỏ qua t-test")
                continue
            p_value, p_corrected = pvalues_by_lambda[lam]
            mean_diff = statistics.mean(vals) - statistics.mean(baseline_vals)
            direction = "TỐT HƠN" if mean_diff > 0 else "TỆ HƠN"
            sig_raw = "có xu hướng ý nghĩa (p_raw<0.10)" if p_value < 0.10 else "chưa rõ ý nghĩa (p_raw>=0.10)"
            sig_corrected = "CÒN ý nghĩa SAU hiệu chỉnh (p_corrected<0.10)" if p_corrected < 0.10 \
                else "KHÔNG còn ý nghĩa sau hiệu chỉnh (p_corrected>=0.10)"
            d = cohens_d_by_lambda.get(lam)
            d_str = f"d={d:+.3f}" if d is not None else "d=NA (std hiệu số=0 hoặc <2 cặp)"
            ci_lo, ci_hi = ci95_by_lambda.get(lam, (None, None))
            ci_str = f"CI95%=[{ci_lo:+.4f}, {ci_hi:+.4f}]" if ci_lo is not None else "CI95%=NA"
            wp = wilcoxon_by_lambda.get(lam)
            wp_str = f"wilcoxon_p={wp:.4f}" if wp is not None else "wilcoxon_p=NA"
            print(f"  lambda_saliency={lam:<6} chênh lệch={mean_diff:+.4f}  {ci_str}  "
                  f"p_raw={p_value:.4f} ({sig_raw})  "
                  f"p_corrected={p_corrected:.4f} ({direction}, {sig_corrected})  "
                  f"Cohen's {d_str}  {wp_str}")
            (mdes_d_a10, mdes_acc_a10), (mdes_d_bonf, mdes_acc_bonf) = mdes_by_lambda.get(
                lam, ((None, None), (None, None)))
            mdes_str_a10 = f"d={mdes_d_a10:.3f} (~{mdes_acc_a10:+.4f})" if mdes_d_a10 is not None else "NA"
            mdes_str_bonf = f"d={mdes_d_bonf:.3f} (~{mdes_acc_bonf:+.4f})" if mdes_d_bonf is not None else "NA"
            print(f"      MDES (power=80%): alpha=.10 -> {mdes_str_a10}  |  "
                  f"sau Bonferroni -> {mdes_str_bonf}")

    best_lam = max(lambdas_sorted, key=lambda x: statistics.mean(stats_by_lambda[x]))
    print(f"\n>>> Mức lambda_saliency có trung bình accuracy cao nhất: {best_lam} "
          f"(mean={statistics.mean(stats_by_lambda[best_lam]):.4f})")
    print(">>> Xem thêm kết quả t-test ở trên trước khi chốt — mức cao nhất chưa "
          "chắc đã KHÁC BIỆT CÓ Ý NGHĨA so với các mức lân cận.")


if __name__ == "__main__":
    main()
