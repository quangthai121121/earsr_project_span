"""
[MỚI — phát hiện qua review Q1] Tổng hợp phương sai do CHÍNH SEED TRAIN SR
(không phải seed của recognition downstream) — trả lời trực tiếp mối lo ngại
reviewer: "toàn bộ kết luận thống kê chỉ điều kiện trên 1 checkpoint SR cụ
thể (seed=42, n=1); nếu SR 'may/xui' ở seed đó, hiệu ứng quan sát được có
thể phản ánh riêng lần chạy đó chứ không phải đặc tính kiến trúc."

Đọc các file *_srseed<N>.json do pipeline/run_sr_seed_variance.sh sinh ra
(mỗi file: 1 checkpoint SR train ở seed N riêng, fine-tune recognition với
SEED DOWNSTREAM CỐ ĐỊNH — cô lập đúng 1 biến là seed SR, không trộn với
phương sai downstream đã đo ở run_multi_seed.sh).

Nếu có --downstream_multiseed_csv (ví dụ results/multi_seed/multi_seed_summary.csv),
đọc thêm std_identity_accuracy của domain sr_improved (cùng backbone) — đó là
phương sai do SEED DOWNSTREAM (đã biết từ trước) — để so sánh cạnh nhau với
phương sai do SEED SR (mới đo được ở đây). Đây là bằng chứng ĐỊNH LƯỢNG trực
tiếp cho câu hỏi "phương sai do SR-seed lớn/nhỏ ra sao so với phương sai
downstream đã báo cáo".

Nếu có --sr_quality_csv (results/sr_seed_variance/sr_quality_srseed.csv, do
pipeline/run_sr_seed_variance.sh tự sinh qua eval_sr_quality.py), tính thêm
mean/std/CI95% của PSNR/SSIM theo seed-SR — bằng chứng TRỰC TIẾP NHẤT (chất
lượng ảnh SR tự nó biến động ra sao theo seed, TRƯỚC CẢ khi tính đến nhiễu từ
recognition downstream) — ghi ra file riêng <out_csv stem>_sr_quality.csv.

Chạy (ví dụ span_tiny -- --downstream_domain mặc định "sr_improved" nên có thể
bỏ qua; span_baseline/span_large BẮT BUỘC truyền đúng domain tương ứng, xem
--help, nếu không sẽ so sánh nhầm với phương sai downstream của span_tiny):
    python data/aggregate_sr_seed_variance.py --results_dir results/sr_seed_variance_span_tiny \
        --out_csv results/sr_seed_variance_span_tiny/sr_seed_variance_summary.csv \
        --downstream_multiseed_csv results/multi_seed/multi_seed_summary.csv \
        --sr_quality_csv results/sr_seed_variance_span_tiny/sr_quality_srseed.csv
"""
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats

SEED_RE = re.compile(r"_srseed(\d+)\.json$")
LABEL_SEED_RE = re.compile(r"_srseed(\d+)$")


def ci95_from_values(values):
    """Khoảng tin cậy 95% cho trung bình (phân phối t, đúng cho mẫu nhỏ) —
    cùng phương pháp với paired_ci95() ở các file aggregate_*.py khác, ở đây
    KHÔNG ghép cặp (không có baseline để trừ) mà tính trực tiếp trên chính
    dãy giá trị theo SR-seed."""
    n = len(values)
    if n < 2:
        return None, None
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    se = std / (n ** 0.5)
    if se < 1e-12:
        return mean, mean
    t_crit = stats.t.ppf(0.975, df=n - 1)
    margin = t_crit * se
    return mean - margin, mean + margin


def aggregate_sr_quality(csv_path):
    """Đọc sr_quality_srseed.csv (do eval_sr_quality.py append 1 dòng/seed),
    tách seed từ cột 'label' (dạng '<arch>_srseed<N>'), trả về list dict
    (1 dict/metric: psnr_db, ssim) với mean/std/CI95%/values — hoặc None nếu
    file không tồn tại/rỗng.

    [SỬA — phát hiện qua review] eval_sr_quality.py LUÔN append (không dedup) —
    nếu 1 bước bị lỗi giữa chừng rồi chạy lại (dễ xảy ra trên server dùng
    chung), dòng của seed đó bị nhân đôi trong CSV, và code CŨ ở đây cộng dồn
    TẤT CẢ dòng khớp pattern vào 1 list phẳng không phân biệt theo seed —
    silently đếm sai n (ví dụ n=4 dù chỉ có 3 seed thật) và lệch mean/std mà
    KHÔNG có cảnh báo nào. Giờ key theo SEED (dict), dòng SAU đè dòng TRƯỚC
    (khớp "chạy lại ghi đè kết quả cũ" là ý định hợp lý nhất khi rerun), và
    IN CẢNH BÁO rõ ràng nếu phát hiện trùng — để người dùng biết mà kiểm tra
    lại file thay vì tin nhầm số liệu.
    """
    if not csv_path or not Path(csv_path).exists():
        return None
    per_seed = {}  # seed:int -> {"psnr_db": float, "ssim": float}
    n_duplicate_rows = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = LABEL_SEED_RE.search(row.get("label", ""))
            if not m:
                continue
            seed = int(m.group(1))
            if seed in per_seed:
                n_duplicate_rows += 1
            entry = per_seed.setdefault(seed, {})
            for metric in ("psnr_db", "ssim"):
                try:
                    entry[metric] = float(row[metric])
                except (KeyError, ValueError):
                    pass

    if not per_seed:
        return None
    if n_duplicate_rows:
        print(f"CẢNH BÁO: {csv_path} có {n_duplicate_rows} dòng label trùng seed (có thể do "
              f"chạy lại 1 bước sau lỗi) — chỉ dùng dòng CUỐI CÙNG cho mỗi seed, dòng cũ bị bỏ. "
              f"Kiểm tra lại file gốc nếu nghi ngờ số liệu.")

    rows_by_metric = {"psnr_db": [], "ssim": []}
    for entry in per_seed.values():
        for metric in rows_by_metric:
            if metric in entry:
                rows_by_metric[metric].append(entry[metric])

    out = []
    for metric, values in rows_by_metric.items():
        if len(values) < 2:
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        ci_lo, ci_hi = ci95_from_values(values)
        out.append({
            "metric": metric, "n_sr_seeds": len(values),
            "mean": round(mean, 4), "std": round(std, 4),
            "ci95_lower": round(ci_lo, 4) if ci_lo is not None else "NA",
            "ci95_upper": round(ci_hi, 4) if ci_hi is not None else "NA",
            "min": round(min(values), 4), "max": round(max(values), 4),
            "values": ";".join(f"{v:.4f}" for v in values),
        })
    return out


def read_downstream_std(csv_path, backbone, domain="sr_improved"):
    if not csv_path or not Path(csv_path).exists():
        return None
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("backbone") == backbone and row.get("domain") == domain:
                try:
                    return float(row["std_identity_accuracy"])
                except (KeyError, ValueError):
                    return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--downstream_multiseed_csv", default=None,
                     help="[tuỳ chọn] results/multi_seed/multi_seed_summary.csv — để so sánh "
                          "std do SR-seed (mới đo) với std do seed downstream (đã biết)")
    ap.add_argument("--sr_quality_csv", default=None,
                     help="[tuỳ chọn] results/sr_seed_variance/sr_quality_srseed.csv — tính thêm "
                          "phương sai PSNR/SSIM của CHÍNH checkpoint SR theo seed (bằng chứng trực "
                          "tiếp nhất, trước cả khi tính đến nhiễu downstream)")
    ap.add_argument("--downstream_domain", default="sr_improved",
                     help="[MỚI — mở rộng đa kiến trúc] domain cần đọc std downstream-seed từ "
                          "--downstream_multiseed_csv để so sánh — PHẢI khớp đúng kiến trúc đang đo "
                          "SR-seed ở đây (span_tiny -> 'sr_improved' [mặc định, giữ hành vi cũ], "
                          "span_baseline -> 'sr_baseline', span_large -> 'sr_span_large'). Trước đây "
                          "hardcode 'sr_improved' -- đúng khi script chỉ dùng cho span_tiny, nhưng SAI "
                          "cho span_baseline/span_large nếu không đổi cờ này (sẽ so sánh nhầm với "
                          "phương sai downstream của span_tiny thay vì của chính kiến trúc đang đo).")
    args = ap.parse_args()

    by_backbone = defaultdict(dict)  # backbone -> {sr_seed:int -> data_dict}
    n_skipped = 0
    for f in sorted(Path(args.results_dir).glob("*_srseed*.json")):
        m = SEED_RE.search(f.name)
        if not m:
            n_skipped += 1
            continue
        sr_seed = int(m.group(1))
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        by_backbone[data["backbone"]][sr_seed] = data

    if not by_backbone:
        print(f"Không tìm thấy file *_srseed<N>.json nào trong {args.results_dir}")
        return
    if n_skipped:
        print(f"Cảnh báo: bỏ qua {n_skipped} file không khớp mẫu *_srseed<N>.json")

    rows = []
    for backbone, by_seed in by_backbone.items():
        sr_seeds = sorted(by_seed.keys())
        id_accs = [by_seed[s]["identity_accuracy"] for s in sr_seeds]
        n = len(id_accs)
        mean = statistics.mean(id_accs)
        std = statistics.stdev(id_accs) if n > 1 else 0.0
        ci_lo, ci_hi = ci95_from_values(id_accs)
        downstream_std = read_downstream_std(args.downstream_multiseed_csv, backbone,
                                              domain=args.downstream_domain)

        row = {
            "backbone": backbone,
            "n_sr_seeds": n,
            "sr_seeds": ";".join(str(s) for s in sr_seeds),
            "mean_identity_accuracy": round(mean, 4),
            "std_identity_accuracy_due_to_sr_seed": round(std, 4),
            "ci95_lower": round(ci_lo, 4) if ci_lo is not None else "NA",
            "ci95_upper": round(ci_hi, 4) if ci_hi is not None else "NA",
            "min": round(min(id_accs), 4), "max": round(max(id_accs), 4),
            "all_values": ";".join(f"{v:.4f}" for v in id_accs),
            "std_identity_accuracy_due_to_downstream_seed": round(downstream_std, 4)
                if downstream_std is not None else "NA (chưa cung cấp --downstream_multiseed_csv "
                                                    "hoặc chưa chạy pipeline/run_multi_seed.sh)",
        }
        if downstream_std is not None and downstream_std > 1e-9:
            row["ratio_sr_seed_std_over_downstream_seed_std"] = round(std / downstream_std, 3)
        else:
            row["ratio_sr_seed_std_over_downstream_seed_std"] = "NA"
        rows.append(row)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã ghi {out_path}\n")

    print("== Phương sai do SEED TRAIN SR (tách biệt khỏi phương sai downstream) ==")
    for r in rows:
        print(f"\n--- Backbone: {r['backbone']} (n={r['n_sr_seeds']} SR-seed: {r['sr_seeds']}) ---")
        print(f"  identity_accuracy: {r['mean_identity_accuracy']:.4f} "
              f"+- {r['std_identity_accuracy_due_to_sr_seed']:.4f}  "
              f"CI95%=[{r['ci95_lower']}, {r['ci95_upper']}]  "
              f"(min={r['min']}, max={r['max']})")
        if isinstance(r["std_identity_accuracy_due_to_downstream_seed"], float):
            print(f"  So sánh: std do SR-seed = {r['std_identity_accuracy_due_to_sr_seed']:.4f}  "
                  f"vs  std do downstream-seed (đã biết, run_multi_seed.sh) = "
                  f"{r['std_identity_accuracy_due_to_downstream_seed']:.4f}  "
                  f"(tỷ lệ = {r['ratio_sr_seed_std_over_downstream_seed_std']})")
            if r["ratio_sr_seed_std_over_downstream_seed_std"] not in ("NA",) and \
                    isinstance(r["ratio_sr_seed_std_over_downstream_seed_std"], (int, float)) and \
                    r["ratio_sr_seed_std_over_downstream_seed_std"] > 1.0:
                print("  !!! LƯU Ý: phương sai do SR-seed LỚN HƠN phương sai downstream đã báo cáo "
                      "— cần nêu rõ trong Limitations, kết luận multi-seed downstream KHÔNG đủ để "
                      "khẳng định kết quả ổn định với seed SR.")
        else:
            print(f"  (chưa có số liệu downstream-seed để so sánh — "
                  f"{r['std_identity_accuracy_due_to_downstream_seed']})")

    # [SỬA — trước đây hardcode "n=3 ... cần thêm seed" bất kể n thực tế, gây
    # hiểu lầm khi đã chạy pipeline/run_sr_seed_variance_extra_seeds.sh lên
    # n=5 (log vẫn nói "cần thêm" dù đã đủ) -- giờ đọc n nhỏ nhất thực tế
    # trong rows để quyết định thông điệp đúng.
    min_n = min(r["n_sr_seeds"] for r in rows)
    if min_n < 5:
        print(f"\nLƯU Ý: n={min_n} SR-seed là bước sàng lọc, không phải bằng chứng cuối cùng — "
              "nếu muốn kiểm định thống kê chính thức (t-test/CI) cho riêng trục SR-seed, cần "
              "thêm seed (khuyến nghị n=5, khớp quy ước n=5 seed downstream của project — xem "
              "pipeline/run_sr_seed_variance_extra_seeds.sh).")
    else:
        print(f"\nĐã đạt n={min_n} SR-seed, khớp quy ước n=5 seed downstream của project — đủ để "
              "dùng làm bằng chứng chính thức, không chỉ sàng lọc.")

    sr_quality_rows = aggregate_sr_quality(args.sr_quality_csv)
    if sr_quality_rows:
        sr_quality_path = out_path.with_name(out_path.stem + "_sr_quality.csv")
        with open(sr_quality_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sr_quality_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sr_quality_rows)
        print(f"\nĐã ghi {sr_quality_path} (phương sai PSNR/SSIM của CHÍNH checkpoint SR theo seed)")
        print("== Phương sai chất lượng SR (PSNR/SSIM) theo seed-SR — bằng chứng trực tiếp nhất ==")
        for r in sr_quality_rows:
            print(f"  {r['metric']:<10} {r['mean']:.4f} +- {r['std']:.4f}  "
                  f"CI95%=[{r['ci95_lower']}, {r['ci95_upper']}]  (min={r['min']}, max={r['max']}, "
                  f"n={r['n_sr_seeds']})")
    elif args.sr_quality_csv:
        print(f"\n(Không đọc được dữ liệu PSNR/SSIM hợp lệ từ {args.sr_quality_csv} — bỏ qua.)")


if __name__ == "__main__":
    main()
