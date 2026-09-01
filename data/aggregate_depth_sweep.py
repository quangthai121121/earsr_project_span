"""
[MỚI — trả lời phản biện] Tổng hợp đường cong accuracy-vs-depth (n_blocks =
1..6) từ 3 nguồn: 4 điểm mới (results/depth_sweep/*.json, do
pipeline/run_depth_sweep.sh sinh ra) + 2 điểm đã có sẵn KHÔNG cần train lại:
    - 3 khối = span_tiny (domain sr_improved, results/multi_seed/*.json)
    - 6 khối = span_large (domain sr_span_large, results/span_large_ablation/*.json)
      -- CÙNG class SPAN tự viết với span_tiny (xem ghi chú phương pháp
      Section~arch bài báo) -- so sánh depth SẠCH hơn span_baseline vì không
      lẫn khác biệt Conv3XC/reparameterization.

Với mỗi n_blocks, tính mean/std qua n=5 seed (mobilenet_v2), rồi so sánh MỖI
điểm nén (1-5 khối) với điểm 6 khối (uncompressed) bằng paired t-test/Cohen's
d/Wilcoxon, Bonferroni-corrected qua family so sánh -- trả lời trực tiếp "nén
tới đâu thì bắt đầu ảnh hưởng đáng kể tới downstream accuracy", đúng câu hỏi
tiêu đề/abstract đặt ra mà bản gốc (chỉ 1 điểm 3-vs-6 khối) không trả lời được.

Chạy:
    python data/aggregate_depth_sweep.py --config configs/config.yaml \
        --depth_sweep_dir results/depth_sweep --out_csv results/depth_sweep/depth_sweep_summary.csv
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_multi_seed_results import cohens_d_paired, paired_ci95, wilcoxon_paired_p, tost_paired  # noqa: E402

BACKBONE = "mobilenet_v2"
SEEDS = [42, 123, 2024, 44, 999]


def _read_seed_values(json_dir, domain, backbone, seeds):
    """Đọc identity_accuracy cho từng seed, trả về dict seed:int -> value.
    Bỏ qua seed thiếu file (in cảnh báo) thay vì crash -- cho phép chạy dở
    dang / kiểm tra sớm mà không phá toàn bộ script."""
    values = {}
    for seed in seeds:
        f = Path(json_dir) / f"{domain}_{backbone}_seed{seed}.json"
        if not f.exists():
            print(f"CẢNH BÁO: thiếu {f}")
            continue
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        values[seed] = data["identity_accuracy"]
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--depth_sweep_dir", default=None,
                     help="mặc định: <results_root>/depth_sweep")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--equivalence_margin", type=float, default=0.01,
                     help="[MỚI — trả lời phản biện] biên tương đương cho TOST (accuracy, mặc định "
                          "0.01 = 1 điểm %%) -- 'n.s.' trong bảng pairwise KHÔNG chứng minh được "
                          "'tương đương', chỉ có nghĩa 'chưa đủ bằng chứng khác biệt' -- xem "
                          "aggregate_multi_seed_results.py::tost_paired().")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    results_root = cfg["paths"]["results_root"]
    depth_dir = Path(args.depth_sweep_dir) if args.depth_sweep_dir else Path(results_root) / "depth_sweep"
    out_path = Path(args.out_csv) if args.out_csv else depth_dir / "depth_sweep_summary.csv"

    per_depth_values = {}  # n_blocks:int -> {seed:int -> accuracy}
    for depth in [1, 2, 4, 5]:
        per_depth_values[depth] = _read_seed_values(depth_dir, f"sr_depth{depth}", BACKBONE, SEEDS)
    per_depth_values[3] = _read_seed_values(Path(results_root) / "multi_seed", "sr_improved", BACKBONE, SEEDS)
    per_depth_values[6] = _read_seed_values(Path(results_root) / "span_large_ablation", "sr_span_large", BACKBONE, SEEDS)

    missing_depths = [d for d, v in per_depth_values.items() if len(v) < 2]
    if missing_depths:
        print(f"CẢNH BÁO: các n_blocks sau thiếu đủ dữ liệu (< 2 seed): {sorted(missing_depths)} "
              f"-- bảng/kiểm định cho các điểm đó sẽ không đầy đủ.")

    rows = []
    for depth in sorted(per_depth_values.keys()):
        by_seed = per_depth_values[depth]
        seeds_present = sorted(by_seed.keys())
        values = [by_seed[s] for s in seeds_present]
        n = len(values)
        mean = statistics.mean(values) if values else None
        std = statistics.stdev(values) if n > 1 else 0.0
        rows.append({
            "n_blocks": depth, "n_seeds": n, "seeds": ";".join(str(s) for s in seeds_present),
            "mean_identity_accuracy": round(mean, 4) if mean is not None else "NA",
            "std_identity_accuracy": round(std, 4),
            "all_values": ";".join(f"{v:.4f}" for v in values),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã ghi {out_path}")
    print("\n== Đường cong accuracy-vs-depth (mobilenet_v2, n=5 seed mỗi điểm) ==")
    for r in rows:
        print(f"  n_blocks={r['n_blocks']}: {r['mean_identity_accuracy']} +- {r['std_identity_accuracy']} (n={r['n_seeds']})")

    if 6 not in per_depth_values or len(per_depth_values[6]) < 2:
        print("\nKhông đủ dữ liệu ở n_blocks=6 để so sánh pairwise -- bỏ qua bước này.")
        return

    ref_by_seed = per_depth_values[6]
    compare_depths = [d for d in [1, 2, 3, 4, 5] if d in per_depth_values and len(per_depth_values[d]) >= 2]

    # [SỬA — phát hiện qua review lại logic, không phải qua chạy thử] Trước
    # đây n_comparisons (dùng làm hệ số Bonferroni) tính TRƯỚC vòng lặp, dựa
    # trên số điểm CÓ ĐỦ >=2 seed của riêng điểm đó -- nhưng 1 điểm có thể bị
    # "continue" bên trong vòng lặp nếu seed của nó KHÔNG TRÙNG với seed của
    # điểm tham chiếu (n_blocks=6), ví dụ dữ liệu thu thập dở dang trên
    # server dùng chung. Khi đó n_comparisons dùng để hiệu chỉnh các so sánh
    # CÒN LẠI sẽ KHÔNG khớp số dòng thật sự xuất hiện trong bảng kết quả --
    # về hướng quá thận trọng (không sai thành "có ý nghĩa" khi thực ra
    # không), nhưng vẫn là hệ số sai. Sửa bằng cách tính raw p-value trước
    # (pass 1), rồi mới áp Bonferroni với family size = số dòng THẬT SỰ có
    # trong bảng (pass 2).
    raw_results = []
    for depth in compare_depths:
        this_by_seed = per_depth_values[depth]
        common_seeds = sorted(set(this_by_seed.keys()) & set(ref_by_seed.keys()))
        if len(common_seeds) < 2:
            print(f"CẢNH BÁO: n_blocks={depth} vs 6 -- không đủ seed chung ({len(common_seeds)}), bỏ qua.")
            continue
        values_a = [this_by_seed[s] for s in common_seeds]  # n_blocks=depth (nén)
        values_b = [ref_by_seed[s] for s in common_seeds]   # n_blocks=6 (uncompressed, tham chiếu)
        d = cohens_d_paired(values_a, values_b)
        ci_lo, ci_hi = paired_ci95(values_a, values_b)
        w_p = wilcoxon_paired_p(values_a, values_b)
        t_stat, p_raw = stats.ttest_rel(values_a, values_b)
        mean_diff = statistics.mean([a - b for a, b in zip(values_a, values_b)])
        # [MỚI — trả lời phản biện] "n.s." (p_bonf>=0.05) KHÔNG chứng minh
        # được "tương đương" -- xem giải thích đầy đủ trong
        # aggregate_multi_seed_results.py::tost_paired().
        tost_result = tost_paired(values_a, values_b, margin=args.equivalence_margin)
        raw_results.append({
            "n_blocks": depth, "n_blocks_ref": 6, "n_common_seeds": len(common_seeds),
            "mean_diff": mean_diff, "cohens_d": d, "p_raw": p_raw, "wilcoxon_p": w_p,
            "ci95_lower": ci_lo, "ci95_upper": ci_hi, "tost_result": tost_result,
        })

    if not raw_results:
        print("\nKhông có so sánh pairwise nào đủ dữ liệu.")
        return

    n_comparisons = len(raw_results)  # family size THẬT, khớp đúng số dòng sẽ ghi ra
    pairwise_rows = []
    for r in raw_results:
        p_raw = r["p_raw"]
        p_bonf = "NA" if p_raw != p_raw else min(1.0, p_raw * n_comparisons)
        pairwise_rows.append({
            "n_blocks": r["n_blocks"], "n_blocks_ref": r["n_blocks_ref"],
            "n_common_seeds": r["n_common_seeds"],
            "mean_diff": round(r["mean_diff"], 4),
            "cohens_d": round(r["cohens_d"], 3) if r["cohens_d"] is not None else "NA",
            "p_raw": round(p_raw, 4) if p_raw == p_raw else "NA",
            "p_bonferroni": round(p_bonf, 4) if isinstance(p_bonf, float) else p_bonf,
            "wilcoxon_p": round(r["wilcoxon_p"], 4) if r["wilcoxon_p"] is not None else "NA",
            "ci95_lower": round(r["ci95_lower"], 4) if r["ci95_lower"] is not None else "NA",
            "ci95_upper": round(r["ci95_upper"], 4) if r["ci95_upper"] is not None else "NA",
            "tost_p": round(r["tost_result"]["p_tost"], 4) if r["tost_result"] is not None else "NA",
            "equivalent_within_margin": r["tost_result"]["equivalent"] if r["tost_result"] is not None else "NA",
        })

    pairwise_path = out_path.with_name(out_path.stem + "_pairwise.csv")
    with open(pairwise_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairwise_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pairwise_rows)
    print(f"Đã ghi {pairwise_path}")
    print(f"\n== So sánh mỗi điểm nén với 6 khối (uncompressed), Bonferroni qua family "
          f"{n_comparisons} so sánh ==")
    for r in pairwise_rows:
        # [SỬA — phát hiện qua review, không phải qua chạy thử] dòng in console
        # trước đây KHÔNG hiển thị kết quả TOST (chỉ có trong CSV) -- nếu chỉ
        # đọc console log (thói quen gửi kết quả trong suốt session này) sẽ bỏ
        # lỡ hoàn toàn phân biệt NHST/TOST, đúng lỗi phương pháp reviewer chỉ
        # ra. Nhúng thẳng vào dòng in, khớp cách aggregate_multi_seed_results.py
        # đã làm.
        # [SỬA — bug phát hiện qua chạy thử ngay sau khi viết] "equivalent" từ
        # tost_paired() là kết quả so sánh scipy (p_tost < alpha) -> kiểu
        # numpy.bool_, KHÔNG phải bool gốc Python. "numpy.bool_(True) is True"
        # cho ra False (so sánh định danh, không phải giá trị) -- dùng "is"
        # khiến MỌI dòng rơi vào nhánh "TOST: NA" dù CSV có giá trị thật (đã
        # tái lập: chạy thử ngay sau khi thêm dòng in này, toàn bộ 5 dòng đều
        # in "TOST: NA"). Dùng truthy-check thường (bool(...)), an toàn với
        # cả bool gốc lẫn numpy.bool_.
        eq = r["equivalent_within_margin"]
        tost_str = ("TOST: NA" if eq == "NA"
                    else f"tương đương ±{args.equivalence_margin} (TOST p={r['tost_p']})" if eq
                    else f"KHÔNG chứng minh được tương đương (TOST p={r['tost_p']})")
        print(f"  {r['n_blocks']} khối vs 6 khối: diff={r['mean_diff']}, d={r['cohens_d']}, "
              f"p_bonf={r['p_bonferroni']} | {tost_str}")


if __name__ == "__main__":
    main()
