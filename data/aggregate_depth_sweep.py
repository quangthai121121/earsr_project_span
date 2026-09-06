"""
[MỚI — trả lời phản biện] Tổng hợp đường cong accuracy-vs-depth (n_blocks =
1..6) từ 3 nguồn: 4 điểm mới (results/depth_sweep/*.json, do
pipeline/run_depth_sweep.sh + run_depth_sweep_new_backbones.sh sinh ra) + 2
điểm đã có sẵn KHÔNG cần train lại:
    - 3 khối = span_tiny (domain sr_improved, results/multi_seed/*.json)
    - 6 khối = span_large (domain sr_span_large, results/span_large_ablation/*.json)
      -- CÙNG class SPAN tự viết với span_tiny (xem ghi chú phương pháp
      Section~arch bài báo) -- so sánh depth SẠCH hơn span_baseline vì không
      lẫn khác biệt Conv3XC/reparameterization.

[SỬA — Trụ cột 2, reviewer round 2] Trước đây BACKBONE="mobilenet_v2" hard-code
-- reviewer chỉ ra depth sweep chỉ chạy trên 1 backbone đại diện, không đủ để
khái quát hoá. Giờ tự phát hiện backbone từ TRƯỜNG "backbone" trong mỗi JSON
(khớp đúng quy ước aggregate_multi_seed_results.py::main(), KHÔNG đoán qua tên
file) -- script này tự lên bảng cho MỌI backbone có đủ dữ liệu ở cả 6 điểm
n_blocks, không cần biết trước danh sách backbone nào đã chạy.

Với mỗi (backbone, n_blocks), tính mean/std qua n=5 seed, rồi so sánh MỖI điểm
nén (1-5 khối) với điểm 6 khối (uncompressed) bằng paired t-test/Cohen's d/
Wilcoxon, Bonferroni-corrected qua family so sánh CỦA RIÊNG backbone đó (khớp
đúng quy ước per-backbone Bonferroni gia đình đã dùng xuyên suốt project) --
trả lời trực tiếp "nén tới đâu thì bắt đầu ảnh hưởng đáng kể tới downstream
accuracy", đúng câu hỏi tiêu đề/abstract đặt ra mà bản gốc (chỉ 1 điểm 3-vs-6
khối, 1 backbone) không trả lời được.

Chạy:
    python data/aggregate_depth_sweep.py --config configs/config.yaml \
        --depth_sweep_dir results/depth_sweep --out_csv results/depth_sweep/depth_sweep_summary.csv
"""
import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_multi_seed_results import cohens_d_paired, paired_ci95, wilcoxon_paired_p, tost_paired  # noqa: E402

SEEDS = [42, 123, 2024, 44, 999]
SEED_RE = re.compile(r"_seed(\d+)\.json$")
DEPTH_RE = re.compile(r"^sr_depth(\d+)$")


def _scan_dir(json_dir, domain_predicate, depth_of_domain):
    """Quét MỌI file *_seed*.json trong json_dir, đọc "backbone" + xác định
    n_blocks từ nội dung JSON (KHÔNG đoán qua tên file) -- trả về
    dict (backbone, n_blocks) -> {seed:int -> accuracy}. domain_predicate
    lọc theo test_domain; depth_of_domain ánh xạ test_domain -> n_blocks.
    Bỏ qua file thiếu field/không khớp seed (in cảnh báo) thay vì crash."""
    import json as json_mod
    result = defaultdict(dict)
    if not Path(json_dir).is_dir():
        return result
    for f in sorted(Path(json_dir).glob("*_seed*.json")):
        m = SEED_RE.search(f.name)
        if not m:
            continue
        seed = int(m.group(1))
        with open(f, "r", encoding="utf-8") as fh:
            data = json_mod.load(fh)
        test_domain = data.get("test_domain")
        if test_domain is None or not domain_predicate(test_domain):
            continue
        backbone = data["backbone"]
        depth = depth_of_domain(test_domain)
        result[(backbone, depth)][seed] = data["identity_accuracy"]
    return result


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

    # (backbone, n_blocks) -> {seed:int -> accuracy}, gộp từ 3 nguồn.
    per_backbone_depth = defaultdict(dict)

    for key, by_seed in _scan_dir(
            depth_dir,
            domain_predicate=lambda d: DEPTH_RE.match(d) is not None,
            depth_of_domain=lambda d: int(DEPTH_RE.match(d).group(1))).items():
        per_backbone_depth[key].update(by_seed)

    for key, by_seed in _scan_dir(
            Path(results_root) / "multi_seed",
            domain_predicate=lambda d: d == "sr_improved",
            depth_of_domain=lambda d: 3).items():
        per_backbone_depth[key].update(by_seed)

    for key, by_seed in _scan_dir(
            Path(results_root) / "span_large_ablation",
            domain_predicate=lambda d: d == "sr_span_large",
            depth_of_domain=lambda d: 6).items():
        per_backbone_depth[key].update(by_seed)

    backbones = sorted({b for b, _ in per_backbone_depth.keys()})
    if not backbones:
        print("CẢNH BÁO: không tìm thấy dữ liệu depth-sweep nào.")
        return
    print(f"Backbone tìm thấy: {backbones}")

    all_summary_rows = []
    all_pairwise_rows = []

    for backbone in backbones:
        per_depth_values = {d: per_backbone_depth.get((backbone, d), {}) for d in range(1, 7)}
        missing_depths = [d for d, v in per_depth_values.items() if len(v) < 2]
        if missing_depths:
            print(f"CẢNH BÁO [{backbone}]: thiếu đủ dữ liệu (< 2 seed) ở n_blocks={sorted(missing_depths)} "
                  f"-- bỏ qua backbone này (cần đủ cả 6 điểm để dựng đường cong).")
            continue

        for depth in sorted(per_depth_values.keys()):
            by_seed = per_depth_values[depth]
            seeds_present = sorted(by_seed.keys())
            values = [by_seed[s] for s in seeds_present]
            n = len(values)
            mean = statistics.mean(values)
            std = statistics.stdev(values) if n > 1 else 0.0
            all_summary_rows.append({
                "backbone": backbone, "n_blocks": depth, "n_seeds": n,
                "seeds": ";".join(str(s) for s in seeds_present),
                "mean_identity_accuracy": round(mean, 4),
                "std_identity_accuracy": round(std, 4),
                "all_values": ";".join(f"{v:.4f}" for v in values),
            })

        ref_by_seed = per_depth_values[6]
        compare_depths = [d for d in [1, 2, 3, 4, 5] if len(per_depth_values[d]) >= 2]

        # [Giữ nguyên logic 2-pass đã sửa trước đây] tính raw p-value trước
        # (pass 1), rồi mới áp Bonferroni với family size = số dòng THẬT SỰ
        # có trong bảng của backbone NÀY (pass 2) -- không lẫn family size
        # giữa các backbone khác nhau.
        raw_results = []
        for depth in compare_depths:
            this_by_seed = per_depth_values[depth]
            common_seeds = sorted(set(this_by_seed.keys()) & set(ref_by_seed.keys()))
            if len(common_seeds) < 2:
                print(f"CẢNH BÁO [{backbone}]: n_blocks={depth} vs 6 -- không đủ seed chung "
                      f"({len(common_seeds)}), bỏ qua.")
                continue
            values_a = [this_by_seed[s] for s in common_seeds]
            values_b = [ref_by_seed[s] for s in common_seeds]
            d = cohens_d_paired(values_a, values_b)
            ci_lo, ci_hi = paired_ci95(values_a, values_b)
            w_p = wilcoxon_paired_p(values_a, values_b)
            t_stat, p_raw = stats.ttest_rel(values_a, values_b)
            mean_diff = statistics.mean([a - b for a, b in zip(values_a, values_b)])
            tost_result = tost_paired(values_a, values_b, margin=args.equivalence_margin)
            raw_results.append({
                "backbone": backbone, "n_blocks": depth, "n_blocks_ref": 6,
                "n_common_seeds": len(common_seeds),
                "mean_diff": mean_diff, "cohens_d": d, "p_raw": p_raw, "wilcoxon_p": w_p,
                "ci95_lower": ci_lo, "ci95_upper": ci_hi, "tost_result": tost_result,
            })

        if not raw_results:
            continue
        n_comparisons = len(raw_results)
        for r in raw_results:
            p_raw = r["p_raw"]
            p_bonf = "NA" if p_raw != p_raw else min(1.0, p_raw * n_comparisons)
            all_pairwise_rows.append({
                "backbone": r["backbone"], "n_blocks": r["n_blocks"], "n_blocks_ref": r["n_blocks_ref"],
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

    if not all_summary_rows:
        print("\nKhông có backbone nào đủ dữ liệu cả 6 điểm n_blocks -- không ghi CSV.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_summary_rows)
    print(f"Đã ghi {out_path}")
    for backbone in backbones:
        rows_this = [r for r in all_summary_rows if r["backbone"] == backbone]
        if not rows_this:
            continue
        print(f"\n== Đường cong accuracy-vs-depth ({backbone}, n=5 seed mỗi điểm) ==")
        for r in rows_this:
            print(f"  n_blocks={r['n_blocks']}: {r['mean_identity_accuracy']} +- {r['std_identity_accuracy']} (n={r['n_seeds']})")

    if not all_pairwise_rows:
        print("\nKhông có so sánh pairwise nào đủ dữ liệu.")
        return

    pairwise_path = out_path.with_name(out_path.stem + "_pairwise.csv")
    with open(pairwise_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_pairwise_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_pairwise_rows)
    print(f"\nĐã ghi {pairwise_path}")
    for backbone in backbones:
        rows_this = [r for r in all_pairwise_rows if r["backbone"] == backbone]
        if not rows_this:
            continue
        n_comparisons = len(rows_this)
        print(f"\n== [{backbone}] So sánh mỗi điểm nén với 6 khối (uncompressed), Bonferroni qua "
              f"family {n_comparisons} so sánh ==")
        for r in rows_this:
            eq = r["equivalent_within_margin"]
            tost_str = ("TOST: NA" if eq == "NA"
                        else f"tương đương ±{args.equivalence_margin} (TOST p={r['tost_p']})" if eq
                        else f"KHÔNG chứng minh được tương đương (TOST p={r['tost_p']})")
            print(f"  {r['n_blocks']} khối vs 6 khối: diff={r['mean_diff']}, d={r['cohens_d']}, "
                  f"p_bonf={r['p_bonferroni']} | {tost_str}")


if __name__ == "__main__":
    main()
