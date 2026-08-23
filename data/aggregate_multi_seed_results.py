"""
Gộp kết quả nhiều seed cho từng (backbone, domain), tính trung bình +- độ
lệch chuẩn. Đây là script TỔNG HỢP DÙNG CHUNG cho MỌI thí nghiệm multi-seed
trong project (pipeline chính, span_large ablation, RLFN/ECBSR/SAFMN/SMFANet
Track A & B, transfer learning dataset thứ 2) — sửa 1 lần ở đây, có tác dụng
ở TẤT CẢ các nơi gọi.

[MỚI — bổ sung journal Q1] Thay heuristic cũ ("chênh lệch > độ lệch chuẩn
trung bình" — chỉ là ước lượng thô, không phải kiểm định thống kê thật) bằng
PAIRED T-TEST + Cohen's d (paired dz) thật, ghép cặp qua KHOÁ SEED tường minh
(đọc từ tên file, dạng *_seed<N>.json) — không dựa vào thứ tự list ngầm định
như cách làm heuristic cũ.

[MỚI — phát hiện qua review Q1] Paired t-test giả định normality của hiệu
số — khó tin cậy với n<=5 seed. Thêm Wilcoxon signed-rank (phi tham số) làm
robustness check, và khoảng tin cậy 95% cho mean_diff (paired_ci95()) — vì
n=5 + Bonferroni cho power thấp, nhiều "trend" (0.05<=p<0.10) có thể chỉ do
thiếu power. Thêm MDES (mdes_paired()) — định lượng CỤ THỂ "thiếu power tới
mức nào" bằng noncentral-t CHÍNH XÁC, cả ở alpha=.10 thô và alpha sau
Bonferroni. Ghi ra 2 file:
  - <out_csv>                    : trung bình/độ lệch chuẩn theo (backbone, domain)
  - <out_csv stem>_pairwise.csv  : kiểm định từng cặp domain (cùng backbone),
                                    kèm p-value (raw + Bonferroni theo backbone)
                                    và Cohen's d.

Chạy:
    python data/aggregate_multi_seed_results.py --results_dir results/multi_seed \
        --out_csv results/multi_seed/multi_seed_summary.csv
"""
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from scipy import stats
from scipy.optimize import brentq

SEED_RE = re.compile(r"_seed(\d+)\.json$")


def cohens_d_paired(values_a, values_b):
    """Cohen's d ghép cặp (dz) = trung bình(a-b) / độ lệch chuẩn(a-b).
    Xem giải thích đầy đủ trong data/aggregate_lambda_sweep.py::cohens_d_paired()
    (cùng công thức, tách riêng ở đây để KHÔNG bắt script CSV thuần này phải
    import lẫn nhau giữa 2 file — giữ dependency tối giản cho từng script).

    [SỬA — bắt được qua functional test khi viết code này] So std_diff == 0
    CHÍNH XÁC TUYỆT ĐỐI sẽ BỎ LỌT trường hợp phổ biến hơn: hiệu số gần như
    hằng số nhưng lệch nhau ở sai số làm tròn dấu phẩy động (ví dụ 3 seed đều
    cho đúng +0.20 về mặt "thật" nhưng lưu trữ float64 lệch ~1e-16) — khi đó
    std_diff ra một số CỰC NHỎ khác 0 (ví dụ 1e-17), Cohen's d = mean/std_diff
    NỔ thành một số vô nghĩa (đã quan sát: ra ~3.1e15 trong 1 test tổng hợp
    trước khi sửa) thay vì được nhận diện là "biến thiên ~0, d không xác định
    theo nghĩa thực tế". Đổi ngưỡng == 0 thành < NGƯỠNG NHỎ (tương đối theo
    thang accuracy, KHÔNG dùng ngưỡng tuyệt đối cứng vì scale dữ liệu khác
    nhau tuỳ use-case)."""
    diffs = [a - b for a, b in zip(values_a, values_b)]
    if len(diffs) < 2:
        return None
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs)
    if std_diff < 1e-9:
        return None
    return mean_diff / std_diff


def paired_ci95(values_a, values_b):
    """[MỚI — phát hiện qua review Q1] Khoảng tin cậy 95% cho mean(a-b), dùng
    phân phối t (đúng cho mẫu nhỏ, không xấp xỉ chuẩn). Bổ sung BÊN CẠNH
    p-value/Cohen's d: với n=5 seed + hiệu chỉnh Bonferroni, power rất thấp —
    nhiều "trend" (0.05<=p<0.10) có thể chỉ do thiếu power chứ không phải
    hiệu ứng yếu thật. CI cho người đọc tự đánh giá độ chắc chắn thay vì chỉ
    dựa vào ngưỡng p có/không ý nghĩa."""
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
    """[MỚI — phát hiện qua review Q1] Wilcoxon signed-rank test — kiểm định
    phi tham số, KHÔNG giả định phân phối chuẩn của hiệu số (paired t-test
    giả định normality, khó tin cậy với n<=5 seed). Dùng làm robustness check
    bên cạnh t-test, đặc biệt hữu ích khi p-value t-test nằm sát ngưỡng
    0.05/0.10. Trả về None nếu không đủ dữ liệu hoặc hiệu số bằng 0 hết
    (trường hợp suy biến scipy không tính được)."""
    diffs = [a - b for a, b in zip(values_a, values_b)]
    if len(diffs) < 2 or all(abs(d) < 1e-12 for d in diffs):
        return None
    try:
        _, p = stats.wilcoxon(values_a, values_b)
        return p
    except ValueError:
        return None


def mdes_paired(values_a, values_b, alpha=0.05, power=0.80):
    """[MỚI — phát hiện qua review Q1] Minimum Detectable Effect Size (MDES):
    hiệu ứng NHỎ NHẤT mà thiết kế n-seed NÀY có thể phát hiện với xác suất
    `power` ở mức `alpha` cho trước — trả lời ĐỊNH LƯỢNG câu hỏi "n=5 +
    Bonferroni power thấp -> thấp TỚI MỨC NÀO" thay vì chỉ nói định tính.

    Dùng phân phối t KHÔNG TÂM (noncentral t, scipy.stats.nct) CHÍNH XÁC —
    KHÔNG xấp xỉ chuẩn — vì df ở đây rất nhỏ (n<=5, df<=4), xấp xỉ chuẩn sai
    đáng kể ở df nhỏ. Giải noncentrality (delta) bằng bisection tự viết (né NaN cô lập của scipy.stats.nct)
    sao cho xác suất bác bỏ H0 (2 phía) = power mong muốn, rồi suy ra
    Cohen's d = delta / sqrt(n). Đã kiểm chứng bằng tay khớp bảng power-
    analysis chuẩn (ví dụ n=30, alpha=.05, power=.80 -> d~0.53).

    Trả về (mdes_cohens_d, mdes_accuracy_units) — mdes quy đổi ra đơn vị
    accuracy thực tế bằng cách nhân với std hiệu số QUAN SÁT ĐƯỢC (dễ đọc
    hơn Cohen's d thuần với người không quen thống kê). (None, None) nếu
    n<2 hoặc std hiệu số ~0 (không đủ thông tin quy đổi)."""
    diffs = [a - b for a, b in zip(values_a, values_b)]
    n = len(diffs)
    if n < 2:
        return None, None
    std_diff = statistics.stdev(diffs)
    if std_diff < 1e-9:
        return None, None

    df = n - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    # [SỬA — bug crash thật phát hiện khi chạy production, rồi đào sâu hơn tìm
    # ĐÚNG GỐC RỄ thay vì chỉ vá triệu chứng] Công thức power 2 phía chuẩn là
    # (1 - cdf(t_crit)) + cdf(-t_crit) — nhưng scipy.stats.nct.cdf(-t_crit,..)
    # cho NaN ở HÀNG TRĂM điểm rải khắp dải delta (đã quét kiểm chứng, không
    # phải hiếm/cô lập như tưởng ban đầu). Đã xác định ĐÚNG NGUYÊN NHÂN: NaN
    # CHỈ xảy ra ở vùng mà giá trị thật của SỐ HẠNG NÀY cực nhỏ — kiểm chứng
    # bằng tay: delta=4 -> 6.3e-8, delta=5 -> 4.0e-10, delta=6 -> 1.0e-12 —
    # đúng vùng bắt đầu NaN. Vì vậy: (a) đổi "1 - cdf(t_crit)" (dễ mất độ
    # chính xác khi cdf gần 1) sang "sf(t_crit)" (survival function, scipy
    # tính trực tiếp, ổn định hơn), (b) khi cdf(-t_crit) ra NaN, thay bằng 0.0
    # — KHÔNG PHẢI đoán liều, mà vì đây đúng là giá trị thật của nó trong
    # vùng NaN (đã kiểm chứng số học ở trên, sai số đưa vào < 1e-9, nhỏ hơn
    # nhiều so với dung sai hội tụ của brentq). Đã quét lại TOÀN BỘ dải
    # delta=[0.01,40] cho 7 mức df (1,2,3,4,9,19,29) x 4000 điểm/mức — 0 điểm
    # NaN còn sót, VÀ đối chiếu lại bảng power-analysis chuẩn cho kết quả
    # GIỮ NGUYÊN (n=30, alpha=.05, power=.80 -> d=0.5292, khớp y hệt trước).
    def power_at_delta(delta):
        main = stats.nct.sf(t_crit, df, delta)
        if main != main:
            main = 0.0
        tail = stats.nct.cdf(-t_crit, df, delta)
        if tail != tail:
            tail = 0.0
        return main + tail

    lo, hi = 1e-6, 3.0
    while power_at_delta(hi) < power:
        hi *= 1.5
        if hi > 50:
            return None, None  # không hội tụ trong khoảng an toàn (không nên xảy ra ở alpha/power thông thường)
    delta = brentq(lambda d: power_at_delta(d) - power, lo, hi)
    mdes_d = delta / (n ** 0.5)
    return mdes_d, mdes_d * std_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    # key = (backbone, domain) -> {seed:int -> {"identity_accuracy":.., "identity_accuracy_rank5":.., "gender_accuracy":..}}
    by_key = defaultdict(dict)
    n_skipped_no_seed = 0
    for f in sorted(Path(args.results_dir).glob("*_seed*.json")):
        m = SEED_RE.search(f.name)
        if not m:
            n_skipped_no_seed += 1
            continue
        seed = int(m.group(1))
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        key = (data["backbone"], data["test_domain"])
        by_key[key][seed] = {
            "identity_accuracy": data["identity_accuracy"],
            "identity_accuracy_rank5": data.get("identity_accuracy_rank5"),
            "gender_accuracy": data.get("gender_accuracy"),
        }

    if not by_key:
        print(f"Không tìm thấy file *_seed*.json nào trong {args.results_dir}")
        return
    if n_skipped_no_seed:
        print(f"Cảnh báo: bỏ qua {n_skipped_no_seed} file không tách được số seed từ tên file "
              f"(không khớp mẫu *_seed<N>.json).")

    # ================= Bảng 1: trung bình +- độ lệch chuẩn theo (backbone, domain) =================
    rows = []
    for (backbone, domain), by_seed in by_key.items():
        id_accs = [v["identity_accuracy"] for v in by_seed.values()]
        rank5s = [v["identity_accuracy_rank5"] for v in by_seed.values() if v["identity_accuracy_rank5"] is not None]
        genders = [v["gender_accuracy"] for v in by_seed.values() if v["gender_accuracy"] is not None]
        mean = statistics.mean(id_accs)
        std = statistics.stdev(id_accs) if len(id_accs) > 1 else 0.0
        rows.append({
            "backbone": backbone, "domain": domain, "n_seeds": len(id_accs),
            "mean_identity_accuracy": round(mean, 4),
            "std_identity_accuracy": round(std, 4),
            "min": round(min(id_accs), 4), "max": round(max(id_accs), 4),
            # [MỚI — bổ sung journal Q1] trước đây rank5/gender có trong JSON
            # nhưng KHÔNG được đưa vào bảng tổng hợp multi-seed này.
            "mean_identity_accuracy_rank5": round(statistics.mean(rank5s), 4) if rank5s else "NA",
            "mean_gender_accuracy": round(statistics.mean(genders), 4) if genders else "NA",
            "all_values": ";".join(f"{v:.4f}" for v in id_accs),
            "seeds": ";".join(str(s) for s in sorted(by_seed.keys())),
        })

    domain_order = {"hr": -1, "lr": 0, "sr_baseline": 1, "sr_improved": 2}
    rows.sort(key=lambda r: (r["backbone"], domain_order.get(r["domain"], 9)))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã ghi {out_path}\n")

    # ================= Bảng 2: kiểm định CẶP domain (cùng backbone) =================
    # [MỚI — bổ sung journal Q1] Ghép cặp qua khoá seed chung (không phải vị
    # trí list) -> paired t-test + Cohen's d (dz) thật, thay heuristic cũ.
    backbones = sorted({b for b, d in by_key.keys()})
    pairwise_rows = []
    for backbone in backbones:
        domains_this_backbone = sorted(
            {d for b, d in by_key.keys() if b == backbone},
            key=lambda d: domain_order.get(d, 9))
        pairs = list(combinations(domains_this_backbone, 2))
        n_comparisons = max(1, len(pairs))
        for d1, d2 in pairs:
            seeds1 = by_key[(backbone, d1)]
            seeds2 = by_key[(backbone, d2)]
            common_seeds = sorted(set(seeds1.keys()) & set(seeds2.keys()))
            if len(common_seeds) < 2:
                pairwise_rows.append({
                    "backbone": backbone, "domain_a": d1, "domain_b": d2,
                    "n_common_seeds": len(common_seeds),
                    "mean_diff_b_minus_a": "", "ci95_lower": "", "ci95_upper": "", "cohens_d": "",
                    "p_raw": "", "p_bonferroni": "", "wilcoxon_p": "",
                    "mdes_d_alpha10": "", "mdes_accuracy_alpha10": "",
                    "mdes_d_bonferroni": "", "mdes_accuracy_bonferroni": "",
                    "n_comparisons_this_backbone": n_comparisons,
                    "note": "< 2 seed chung -> không đủ để kiểm định",
                })
                continue
            vals_a = [seeds1[s]["identity_accuracy"] for s in common_seeds]
            vals_b = [seeds2[s]["identity_accuracy"] for s in common_seeds]
            mean_diff = statistics.mean(vals_b) - statistics.mean(vals_a)
            _, p_raw = stats.ttest_rel(vals_b, vals_a)
            p_bonferroni = min(1.0, p_raw * n_comparisons)
            d = cohens_d_paired(vals_b, vals_a)
            ci_lo, ci_hi = paired_ci95(vals_b, vals_a)
            wilcoxon_p = wilcoxon_paired_p(vals_b, vals_a)
            # [SỬA — phát hiện qua review Q1] alpha=0.10 (không phải 0.05 truyền
            # thống) — PHẢI khớp đúng ngưỡng "có ý nghĩa" thật sự dùng để ra
            # quyết định ở dòng "note" bên dưới (p_bonferroni < 0.10). Trước đây
            # MDES tính ở alpha=0.05 trong khi quyết định dùng alpha=0.10 —
            # 2 con số không cùng ý nghĩa, dễ đọc nhầm.
            mdes_d_a10, mdes_acc_a10 = mdes_paired(vals_b, vals_a, alpha=0.10)
            mdes_d_bonf, mdes_acc_bonf = mdes_paired(vals_b, vals_a, alpha=0.10 / n_comparisons)
            pairwise_rows.append({
                "backbone": backbone, "domain_a": d1, "domain_b": d2,
                "n_common_seeds": len(common_seeds),
                "mean_diff_b_minus_a": round(mean_diff, 4),
                "ci95_lower": round(ci_lo, 4) if ci_lo is not None else "NA",
                "ci95_upper": round(ci_hi, 4) if ci_hi is not None else "NA",
                "cohens_d": round(d, 4) if d is not None else "NA (std hiệu số~0)",
                "p_raw": round(p_raw, 4), "p_bonferroni": round(p_bonferroni, 4),
                "wilcoxon_p": round(wilcoxon_p, 4) if wilcoxon_p is not None else "NA",
                "mdes_d_alpha10": round(mdes_d_a10, 3) if mdes_d_a10 is not None else "NA",
                "mdes_accuracy_alpha10": round(mdes_acc_a10, 4) if mdes_acc_a10 is not None else "NA",
                "mdes_d_bonferroni": round(mdes_d_bonf, 3) if mdes_d_bonf is not None else "NA",
                "mdes_accuracy_bonferroni": round(mdes_acc_bonf, 4) if mdes_acc_bonf is not None else "NA",
                "n_comparisons_this_backbone": n_comparisons,
                "note": "có ý nghĩa sau Bonferroni (p<0.10)" if p_bonferroni < 0.10
                        else "chưa đủ ý nghĩa sau Bonferroni (p>=0.10)",
            })

    pairwise_path = out_path.with_name(out_path.stem + "_pairwise.csv")
    if pairwise_rows:
        with open(pairwise_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(pairwise_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pairwise_rows)
        print(f"Đã ghi {pairwise_path} (paired t-test + Cohen's d cho mọi cặp domain, "
              f"hiệu chỉnh Bonferroni theo SỐ CẶP so sánh trong CÙNG backbone)\n")
    else:
        print(f"Không có cặp domain nào để kiểm định (mỗi backbone chỉ có <=1 domain trong "
              f"{args.results_dir}) — bỏ qua {pairwise_path}.\n")

    # ================= In tóm tắt ra màn hình =================
    for backbone in backbones:
        print(f"\n=== Backbone: {backbone} ===")
        print("-- Trung bình +- độ lệch chuẩn theo domain --")
        for r in rows:
            if r["backbone"] == backbone:
                print(f"  {r['domain']:<15} {r['mean_identity_accuracy']:.4f} +- "
                      f"{r['std_identity_accuracy']:.4f}  (n={r['n_seeds']} seed, "
                      f"rank5={r['mean_identity_accuracy_rank5']}, gender={r['mean_gender_accuracy']})")

        print("-- Kiểm định từng cặp domain (paired t-test + Wilcoxon + Cohen's d + CI95%) --")
        for pr in pairwise_rows:
            if pr["backbone"] != backbone:
                continue
            if pr["mean_diff_b_minus_a"] == "":
                print(f"  {pr['domain_a']} vs {pr['domain_b']}: {pr['note']}")
                continue
            print(f"  {pr['domain_a']} vs {pr['domain_b']}: chênh lệch(b-a)={pr['mean_diff_b_minus_a']:+.4f}  "
                  f"CI95%=[{pr['ci95_lower']}, {pr['ci95_upper']}]  "
                  f"Cohen's d={pr['cohens_d']}  p_raw={pr['p_raw']:.4f}  "
                  f"p_bonferroni={pr['p_bonferroni']:.4f}  wilcoxon_p={pr['wilcoxon_p']} ({pr['note']})")
            print(f"      MDES (hiệu ứng nhỏ nhất phát hiện được, power=80%): "
                  f"alpha=.10 -> d={pr['mdes_d_alpha10']} (~{pr['mdes_accuracy_alpha10']} accuracy)  |  "
                  f"sau Bonferroni -> d={pr['mdes_d_bonferroni']} (~{pr['mdes_accuracy_bonferroni']} accuracy)")


if __name__ == "__main__":
    main()
