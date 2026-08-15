"""
Gộp toàn bộ file JSON kết quả (từ eval_recognition.py) thành 1 bảng CSV duy nhất,
sắp theo backbone rồi domain — dễ copy thẳng vào bảng kết quả bài báo/luận án.

Chạy:
    python data/aggregate_results.py --results_dir results --out_csv results/summary.csv
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = []
    for f in sorted(Path(args.results_dir).glob("*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            rows.append(json.load(fh))

    if not rows:
        print("Không tìm thấy file JSON kết quả nào.")
        return

    rows.sort(key=lambda r: (r["backbone"], r["test_domain"]))

    # [SỬA — bổ sung journal Q1, lỗi phát hiện qua rà soát lại] Bản trước
    # THIẾU "identity_accuracy_rank5" dù eval_recognition.py đã tính và ghi
    # vào từng file JSON từ trước — bị ÂM THẦM RỚT khỏi summary.csv (không
    # lỗi runtime, chỉ đơn giản không có trong fieldnames nên không được
    # chọn ra). Giờ thêm lại, cộng thêm 3 cột MỚI (identity_auc, identity_eer,
    # n_genuine_pairs) từ đợt bổ sung verification-setting. Dùng r.get(k, "NA")
    # thay vì r[k] để KHÔNG lỗi nếu gộp lẫn JSON cũ (chạy trước đợt bổ sung
    # này, chưa có các trường mới) với JSON mới trong cùng results_dir — script
    # này GHI ĐÈ (mode "w") toàn bộ summary.csv mỗi lần chạy nên không có rủi
    # ro lệch cột kiểu append như đã sửa ở eval_sr_quality.py, chỉ cần đảm bảo
    # không KeyError khi thiếu trường.
    fieldnames = ["backbone", "train_domain", "test_domain",
                  "identity_accuracy", "identity_accuracy_rank5", "gender_accuracy",
                  "identity_auc", "identity_eer", "n_genuine_pairs",
                  "params_M", "latency_ms"]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "NA") for k in fieldnames})

    print(f"Đã ghi {args.out_csv} — {len(rows)} dòng kết quả.")

    # In nhanh bảng so sánh sr_baseline vs sr_improved theo từng backbone, để dễ soi ngay
    print("\n== So sánh nhanh accuracy theo backbone (identity_accuracy) ==")
    by_backbone = {}
    for r in rows:
        by_backbone.setdefault(r["backbone"], {})[r["test_domain"]] = r["identity_accuracy"]

    header = f"{'backbone':<20} {'hr':>8} {'lr':>8} {'sr_base':>10} {'sr_improv':>10}"
    print(header)
    for backbone, vals in by_backbone.items():
        print(f"{backbone:<20} "
              f"{vals.get('hr', float('nan')):>8} "
              f"{vals.get('lr', float('nan')):>8} "
              f"{vals.get('sr_baseline', float('nan')):>10} "
              f"{vals.get('sr_improved', float('nan')):>10}")


if __name__ == "__main__":
    main()
