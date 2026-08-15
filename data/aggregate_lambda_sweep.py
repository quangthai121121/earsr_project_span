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

[MỚI — bổ sung journal Q1] Ngoài p-value (raw + Bonferroni), giờ tính thêm
Cohen's d GHÉP CẶP (paired dz) cho mỗi mức lambda so với baseline (0.0) — xem
hàm cohens_d_paired(). Việc ghép cặp giờ dùng KHOÁ SEED tường minh (sửa lỗi
tiềm ẩn: bản trước ghép theo vị trí trong list, chỉ đúng nếu mọi mức lambda
dùng ĐÚNG CÙNG bộ seed).

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


def cohens_d_paired(values_a, values_b):
    """
    [MỚI — bổ sung journal Q1] Cohen's d cho mẫu GHÉP CẶP (dz — "paired" hay
    "repeated measures" Cohen's d): dz = trung bình(a-b) / độ lệch chuẩn(a-b).
    Bổ sung BÊN CẠNH p-value — p-value chỉ nói "có ý nghĩa thống kê hay
    không" (phụ thuộc cỡ mẫu), KHÔNG nói "chênh lệch lớn hay nhỏ về mặt thực
    tế". Nhiều venue Q1 khó hiện yêu cầu báo cáo cả 2 để tránh trường hợp
    p-value nhỏ nhưng effect size không đáng kể (hay ngược lại, effect size
    lớn nhưng p-value không nhỏ vì cỡ mẫu ít, như tình huống n=3 seed ở đây).

    Diễn giải kinh điển (Cohen 1988): |d|<0.2 rất nhỏ, ~0.2 nhỏ, ~0.5 vừa,
    ~0.8 lớn — NHƯNG đây là quy ước tổng quát, không thay thế cho việc đọc
    trực tiếp mean_diff theo đơn vị accuracy thực tế.

    values_a, values_b: 2 danh sách cùng độ dài, ĐÃ ghép cặp đúng theo seed
    (xem cách gọi bên dưới — dùng khoá seed tường minh, không dựa vào thứ tự
    list ngầm định).
    """
    diffs = [a - b for a, b in zip(values_a, values_b)]
    if len(diffs) < 2:
        return None  # cần >=2 cặp để tính độ lệch chuẩn của hiệu số
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs)
    # [SỬA — bắt được qua functional test] dùng ngưỡng nhỏ thay vì so sánh
    # == 0 tuyệt đối: hiệu số "gần như hằng số" vẫn có thể ra std_diff khác 0
    # cực nhỏ do sai số dấu phẩy động, làm d = mean/std_diff nổ thành số vô
    # nghĩa (đã quan sát ra ~3.1e15 trong 1 test tổng hợp) thay vì None.
    if std_diff < 1e-9:
        return None  # biến thiên ~0 -> Cohen's d không xác định theo nghĩa thực tế
    return mean_diff / std_diff


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

    stats_by_lambda = {}
    for lam in lambdas_sorted:
        stats_by_lambda[lam] = list(by_lambda[lam].values())

    # [MỚI — đợt 7] Tính p_raw/p_corrected TRƯỚC khi ghi CSV, để lưu luôn vào
    # file (bản trước chỉ in ra màn hình, không lưu — dễ mất nếu không chép
    # tay stdout). n_comparisons/Bonferroni: xem chú thích chi tiết bên dưới.
    #
    # [SỬA — bổ sung journal Q1] Bản trước ghép cặp (vals[i] với
    # baseline_vals[i]) dựa vào THỨ TỰ trong list — chỉ đúng NGẪU NHIÊN vì
    # by_lambda[lam] là dict {seed: accuracy} và list(...values()) giữ thứ
    # tự chèn, mà thứ tự chèn lại phụ thuộc thứ tự sort tên file (tình cờ
    # giống nhau giữa các mức lambda NẾU dùng đúng 1 bộ seed cố định cho mọi
    # mức — đúng với cách run_lambda_sweep.sh dùng, nhưng KHÔNG có gì đảm bảo
    # nếu sau này ai đó sweep thêm 1 mức lambda với bộ seed khác). Giờ ghép
    # cặp TƯỜNG MINH qua khoá seed chung giữa 2 mức (an toàn dù bộ seed lệch
    # nhau), đồng thời tính thêm Cohen's d (paired) bên cạnh p-value.
    pvalues_by_lambda = {}
    cohens_d_by_lambda = {}
    if "0.0" in by_lambda:
        baseline_by_seed = by_lambda["0.0"]
        candidate_lambdas = [lam for lam in lambdas_sorted if lam != "0.0"]
        n_comparisons = max(1, len(candidate_lambdas))
        for lam in candidate_lambdas:
            common_seeds = sorted(set(by_lambda[lam].keys()) & set(baseline_by_seed.keys()))
            if len(common_seeds) < len(by_lambda[lam]) or len(common_seeds) < len(baseline_by_seed):
                print(f"Cảnh báo: lambda_identity={lam} và baseline (0.0) không cùng bộ seed "
                      f"đầy đủ — chỉ dùng {len(common_seeds)} seed chung để ghép cặp t-test/Cohen's d.")
            if len(common_seeds) < 2:
                continue
            vals = [by_lambda[lam][s] for s in common_seeds]
            baseline_vals_paired = [baseline_by_seed[s] for s in common_seeds]
            _, p_raw = stats.ttest_rel(vals, baseline_vals_paired)
            pvalues_by_lambda[lam] = (p_raw, min(1.0, p_raw * n_comparisons))
            cohens_d_by_lambda[lam] = cohens_d_paired(vals, baseline_vals_paired)

    rows = []
    for lam in lambdas_sorted:
        values = stats_by_lambda[lam]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        p_raw, p_corrected = pvalues_by_lambda.get(lam, ("", ""))
        d = cohens_d_by_lambda.get(lam, "")
        rows.append({
            "lambda_identity": lam, "n_seeds": len(values),
            "mean_identity_accuracy": round(mean, 4),
            "std_identity_accuracy": round(std, 4),
            "p_value_vs_baseline_raw": round(p_raw, 4) if p_raw != "" else "",
            "p_value_vs_baseline_bonferroni": round(p_corrected, 4) if p_corrected != "" else "",
            "cohens_d_vs_baseline": round(d, 4) if isinstance(d, float) else "",
            "all_values": ";".join(f"{v:.4f}" for v in values),
        })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Đã ghi {out_path} (đã gồm p_value_vs_baseline_raw/_bonferroni + cohens_d_vs_baseline)\n")
    print("== Trung bình +- độ lệch chuẩn theo từng mức lambda_identity ==")
    for lam in lambdas_sorted:
        vals = stats_by_lambda[lam]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  lambda_identity={lam:<6} {mean:.4f} +- {std:.4f}  (n={len(vals)})")

    # Kiểm định paired t-test: so mỗi mức lambda với mức 0.0 (baseline: chỉ distillation)
    if "0.0" in stats_by_lambda:
        baseline_vals = stats_by_lambda["0.0"]
        # [SỬA — đợt 7, sửa lỗi phát hiện qua code review] Bản trước so sánh
        # N mức lambda cùng lúc với baseline (N phép kiểm định độc lập trên
        # CÙNG 1 tập dữ liệu) mà KHÔNG hiệu chỉnh so sánh bội (multiple
        # comparisons) — làm tăng tỷ lệ báo sai dương tính (false positive)
        # so với ngưỡng alpha danh nghĩa đã chọn. Thêm hiệu chỉnh Bonferroni
        # (p_corrected = min(1, p_raw * n_comparisons)) — giữ NGUYÊN p-value
        # thô để đối chiếu minh bạch, KHÔNG xoá số cũ, chỉ bổ sung cột đã
        # hiệu chỉnh và đổi câu kết luận "có ý nghĩa" dựa trên số ĐÃ hiệu
        # chỉnh. Bonferroni là lựa chọn BẢO THỦ nhất (kiểm soát chặt family-
        # wise error rate) — phù hợp bước sàng lọc để tránh chọn nhầm lambda
        # do may rủi thống kê với n=3 seed vốn đã ít.
        candidate_lambdas = [lam for lam in lambdas_sorted if lam != "0.0"]
        n_comparisons = len(candidate_lambdas)
        print(f"\n== Paired t-test so với lambda_identity=0.0 (chỉ distillation) ==")
        print(f"LƯU Ý: n=3 seed cho p-value độ tin cậy THẤP — chỉ dùng để sàng lọc "
              f"ứng viên, không phải kết luận cuối cùng.")
        print(f"LƯU Ý THÊM [MỚI]: {n_comparisons} phép so sánh đồng thời với cùng baseline "
              f"-> đã hiệu chỉnh Bonferroni (p_corrected = min(1, p_raw x {n_comparisons})) "
              f"để không đánh giá quá lạc quan mức ý nghĩa thống kê.\n")
        for lam in candidate_lambdas:
            vals = stats_by_lambda[lam]
            if lam not in pvalues_by_lambda:
                print(f"  lambda_identity={lam}: số seed không khớp với baseline, bỏ qua t-test")
                continue
            p_value, p_corrected = pvalues_by_lambda[lam]  # tái dùng, không tính lại lần 2
            mean_diff = statistics.mean(vals) - statistics.mean(baseline_vals)
            direction = "TỐT HƠN" if mean_diff > 0 else "TỆ HƠN"
            sig_raw = "có xu hướng ý nghĩa (p_raw<0.10)" if p_value < 0.10 else "chưa rõ ý nghĩa (p_raw>=0.10)"
            sig_corrected = "CÒN ý nghĩa SAU hiệu chỉnh (p_corrected<0.10)" if p_corrected < 0.10 \
                else "KHÔNG còn ý nghĩa sau hiệu chỉnh (p_corrected>=0.10)"
            d = cohens_d_by_lambda.get(lam)
            d_str = f"d={d:+.3f}" if d is not None else "d=NA (std hiệu số=0 hoặc <2 cặp)"
            print(f"  lambda_identity={lam:<6} chênh lệch={mean_diff:+.4f}  "
                  f"p_raw={p_value:.4f} ({sig_raw})  "
                  f"p_corrected={p_corrected:.4f} ({direction}, {sig_corrected})  "
                  f"Cohen's {d_str}")

    best_lam = max(lambdas_sorted, key=lambda x: statistics.mean(stats_by_lambda[x]))
    print(f"\n>>> Mức lambda_identity có trung bình accuracy cao nhất: {best_lam} "
          f"(mean={statistics.mean(stats_by_lambda[best_lam]):.4f})")
    print(">>> Xem thêm kết quả t-test ở trên trước khi chốt — mức cao nhất chưa "
          "chắc đã KHÁC BIỆT CÓ Ý NGHĨA so với các mức lân cận.")


if __name__ == "__main__":
    main()
