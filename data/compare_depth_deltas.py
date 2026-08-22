"""
[MỚI — Mục 5.7(i), ablation attention-parameterization]

Kiểm định trực tiếp giả thuyết cơ chế nêu ở Mục 6 của bản thảo: "SPAN chịu
được nén sâu tốt vì attention không có tham số học (chỉ mất computation,
không mất capacity)". Nếu đúng, một kiến trúc có attention THAM SỐ HỌC ĐƯỢC
(SAFMN/SMFANet) phải mất accuracy NHIỀU HƠN SPAN khi bị giảm CÙNG TỶ LỆ số
khối (ở đây: một nửa).

Đây KHÔNG phải so sánh span_tiny vs SAFMN/SMFANet ở cấu hình đầy đủ (đó là
Bảng 5/README mục 11, đã có). Đây là so sánh CHIỀU DỌC, NỘI BỘ từng kiến
trúc:

    Δ_arch  = acc(arch, full-depth)   - acc(arch, half-depth)     [cùng seed]
    Δ_SPAN  = acc(span_full)          - acc(span_tiny)            [cùng seed]

    [SỬA — lỗi confound phát hiện qua review] Δ_SPAN BẮT BUỘC dùng domain
    "sr_span_large" (span_large, 6 khối, CÙNG recipe distillation với
    span_tiny — xem RUN_ALL_span_large_ablation.sh), TUYỆT ĐỐI KHÔNG dùng
    "sr_baseline" (teacher SPAN train bằng pixel-loss THUẦN, KHÔNG cùng
    recipe). Dùng "sr_baseline" sẽ trộn lẫn 2 biến (độ sâu VÀ recipe train)
    vào cùng 1 delta, làm hỏng chính mục đích cô lập biến độ sâu của
    ablation này — xem docstring đầu RUN_ALL_span_large_ablation.sh.

rồi kiểm định (Δ_arch - Δ_SPAN) qua các seed bằng paired t-test 1 mẫu so
với 0 (H0: hai mức sụt bằng nhau) + Cohen's d ghép cặp — dùng ĐÚNG công
thức/ngưỡng std<1e-9 đã sửa trong data/aggregate_multi_seed_results.py (xem
docstring hàm cohens_d_paired ở đó để biết lý do dùng ngưỡng thay vì ==0).

diff_of_diffs > 0 và có ý nghĩa thống kê (sau Bonferroni theo 5 backbone)
=> arch mất accuracy NHIỀU HƠN SPAN khi giảm cùng tỷ lệ khối => ủng hộ giả
thuyết Mục 6. diff_of_diffs <= 0 hoặc không có ý nghĩa => KHÔNG ủng hộ (cần
báo cáo trung thực như mọi kết quả âm tính khác trong project này).

Chạy (ví dụ sau khi có đủ n=5 seed cho cả half-depth lẫn full-depth):
    python data/compare_depth_deltas.py \\
        --results_dir results/attention_param_ablation_safmn_half4/combined \\
        --arch_full_domain sr_improved_safmn --arch_half_domain sr_improved_safmn_half4 \\
        --span_full_domain sr_span_large --span_tiny_domain sr_improved \\
        --out_csv results/attention_param_ablation_safmn_half4/depth_delta_comparison_safmn.csv
"""
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats

SEED_RE = re.compile(r"_seed(\d+)\.json$")

# Ngưỡng ý nghĩa thống kê — GIỮ ĐÚNG quy ước đã định nghĩa trong bản thảo
# (Mục 3.5): p_Bonf < 0.05 = "significant", 0.05 <= p_Bonf < 0.10 = "trend"
# (không bao giờ gọi là significant), p_Bonf >= 0.10 = "n.s.". Không tự ý
# đổi ngưỡng ở riêng script này — nhất quán là điều reviewer sẽ kiểm tra.
ALPHA_SIG = 0.05
ALPHA_TREND = 0.10


def cohens_d_paired(diffs):
    """Cohen's d ghép cặp (dz) = trung bình(diff) / độ lệch chuẩn(diff).
    Cùng công thức + ngưỡng std<1e-9 (thay vì ==0) đã sửa trong
    data/aggregate_multi_seed_results.py — xem docstring ở đó."""
    if len(diffs) < 2:
        return None
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs)
    if std_diff < 1e-9:
        return None
    return mean_diff / std_diff


def verdict_label(p_bonf):
    if p_bonf is None:
        return "không đủ dữ liệu"
    if p_bonf < ALPHA_SIG:
        return "significant"
    if p_bonf < ALPHA_TREND:
        return "trend"
    return "n.s."


def load_accuracy_by_domain(results_dir):
    """key = (backbone, test_domain) -> {seed:int -> identity_accuracy:float}"""
    by_key = defaultdict(dict)
    n_skipped = 0
    for f in sorted(Path(results_dir).glob("*_seed*.json")):
        m = SEED_RE.search(f.name)
        if not m:
            n_skipped += 1
            continue
        seed = int(m.group(1))
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        key = (data["backbone"], data["test_domain"])
        by_key[key][seed] = data["identity_accuracy"]
    if n_skipped:
        print(f"CẢNH BÁO: bỏ qua {n_skipped} file không khớp pattern *_seed<N>.json")
    return by_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True,
                     help="thư mục chứa TẤT CẢ json cần thiết (arch full-depth, "
                          "arch half-depth, sr_span_large, sr_improved) — dùng thư mục "
                          "'combined' do RUN_ALL_attention_param_ablation*.sh tạo sẵn (BẮT BUỘC "
                          "đã copy thêm JSON từ results/span_large_ablation/ vào đó)")
    ap.add_argument("--arch_full_domain", required=True,
                     help="ví dụ sr_improved_safmn")
    ap.add_argument("--arch_half_domain", required=True,
                     help="ví dụ sr_improved_safmn_half4")
    ap.add_argument("--span_full_domain", default="sr_span_large",
                     help="[SỬA] domain SPAN full-depth CÙNG recipe distillation với span_tiny "
                          "(xem RUN_ALL_span_large_ablation.sh) — KHÔNG dùng 'sr_baseline' "
                          "(teacher pixel-loss thuần, sẽ trộn lẫn biến recipe vào biến độ sâu).")
    ap.add_argument("--span_tiny_domain", default="sr_improved")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    by_key = load_accuracy_by_domain(args.results_dir)
    backbones = sorted({b for (b, _) in by_key.keys()})
    if not backbones:
        raise SystemExit(f"Không tìm thấy JSON hợp lệ nào trong {args.results_dir}")

    rows = []
    for backbone in backbones:
        acc_arch_full = by_key.get((backbone, args.arch_full_domain), {})
        acc_arch_half = by_key.get((backbone, args.arch_half_domain), {})
        acc_span_base = by_key.get((backbone, args.span_full_domain), {})
        acc_span_tiny = by_key.get((backbone, args.span_tiny_domain), {})

        # Chỉ dùng seed có ĐỦ CẢ 4 domain (ghép cặp đúng nghĩa, tránh lệch seed
        # giữa arch và SPAN làm sai kiểm định — cùng nguyên tắc ghép qua khoá
        # seed tường minh đã áp dụng trong aggregate_multi_seed_results.py).
        common_seeds = sorted(set(acc_arch_full) & set(acc_arch_half)
                               & set(acc_span_base) & set(acc_span_tiny))

        delta_arch = [acc_arch_full[s] - acc_arch_half[s] for s in common_seeds]
        delta_span = [acc_span_base[s] - acc_span_tiny[s] for s in common_seeds]
        diff_of_diffs = [da - ds for da, ds in zip(delta_arch, delta_span)]

        n = len(common_seeds)
        if n >= 2:
            t_stat, p_raw = stats.ttest_1samp(diff_of_diffs, popmean=0.0)
            d = cohens_d_paired(diff_of_diffs)
        else:
            p_raw, d = None, None

        rows.append({
            "backbone": backbone,
            "n_seeds_matched": n,
            "seeds": ",".join(str(s) for s in common_seeds),
            "delta_arch_mean": round(statistics.mean(delta_arch), 4) if delta_arch else None,
            "delta_span_mean": round(statistics.mean(delta_span), 4) if delta_span else None,
            "diff_of_diffs_mean": round(statistics.mean(diff_of_diffs), 4) if diff_of_diffs else None,
            "cohens_d": round(d, 3) if d is not None else None,
            "p_raw": round(p_raw, 4) if p_raw is not None else None,
        })

    # Bonferroni theo số backbone thực sự có đủ dữ liệu (>=2 seed) — KHÔNG
    # đếm cả các backbone thiếu dữ liệu vào mẫu số hiệu chỉnh.
    n_testable = sum(1 for r in rows if r["p_raw"] is not None)
    for r in rows:
        if r["p_raw"] is None:
            r["p_bonferroni"] = None
        else:
            r["p_bonferroni"] = round(min(r["p_raw"] * max(n_testable, 1), 1.0), 4)
        r["verdict"] = verdict_label(r["p_bonferroni"])

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["backbone", "n_seeds_matched", "seeds", "delta_arch_mean",
                  "delta_span_mean", "diff_of_diffs_mean", "cohens_d", "p_raw",
                  "p_bonferroni", "verdict"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nĐã ghi {out_path}\n")
    print(f"{'backbone':<20}{'Δ_arch':>10}{'Δ_SPAN':>10}{'diff':>10}{'d':>8}{'p_bonf':>10}  verdict")
    for r in rows:
        print(f"{r['backbone']:<20}"
              f"{r['delta_arch_mean'] if r['delta_arch_mean'] is not None else float('nan'):>10.4f}"
              f"{r['delta_span_mean'] if r['delta_span_mean'] is not None else float('nan'):>10.4f}"
              f"{r['diff_of_diffs_mean'] if r['diff_of_diffs_mean'] is not None else float('nan'):>10.4f}"
              f"{r['cohens_d'] if r['cohens_d'] is not None else float('nan'):>8.2f}"
              f"{r['p_bonferroni'] if r['p_bonferroni'] is not None else float('nan'):>10.4f}"
              f"  {r['verdict']}")

    n_support = sum(1 for r in rows
                     if r["verdict"] == "significant" and (r["diff_of_diffs_mean"] or 0) > 0)
    print(f"\n=> {n_support}/{len(rows)} backbone ủng hộ có ý nghĩa thống kê giả thuyết Mục 6 "
          f"(diff_of_diffs > 0 nghĩa là {args.arch_full_domain.split('_')[-1]} mất accuracy "
          f"nhiều hơn SPAN khi giảm cùng tỷ lệ khối).")
    print("Diễn giải KHÔNG tự động: xem cả delta_arch_mean/delta_span_mean, không chỉ verdict — "
          "1 kiến trúc có thể 'thắng' hoặc 'thua' SPAN mà vẫn không đạt ý nghĩa thống kê do n nhỏ.")


if __name__ == "__main__":
    main()
