"""
Gộp TOÀN BỘ kết quả rải rác trong results/ (summary.csv, sr_quality.csv,
real_lr_holdout.csv, training_summary.csv, sr_comparison_images/) vào 1 thư
mục duy nhất, kèm 1 file REPORT.md tổng hợp sẵn các bảng chính — đọc 1 file
là thấy hết, không cần mở nhiều CSV riêng lẻ.

Chạy (thường được gọi tự động từ pipeline/run_full_pipeline_and_report.sh,
không cần chạy tay):
    python data/generate_final_report.py --config configs/config.yaml \
        --results_dir results --out_dir results/final_report
"""
import argparse
import csv
import shutil
from pathlib import Path

import yaml


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def md_table(rows, columns=None):
    if not rows:
        return "*(không có dữ liệu)*\n"
    cols = columns or list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    student_arch = cfg["sr_improve"].get("student_arch", cfg["sr"]["arch"])
    lambda_pixel = cfg["sr_improve"]["lambda_pixel"]
    lambda_distill = cfg["sr_improve"]["lambda_distill"]
    lambda_identity = cfg["sr_improve"]["lambda_identity"]

    # --- Đọc toàn bộ CSV liên quan ---
    summary_rows = read_csv(results_dir / "summary.csv")
    sr_quality_rows = read_csv(results_dir / "sr_quality.csv")
    real_lr_rows = read_csv(results_dir / "real_lr_holdout.csv")
    training_summary_rows = read_csv(results_dir / "training_summary.csv")

    # --- Copy toàn bộ file gốc vào out_dir để tiện lưu trữ/gửi đi ---
    for fname in ["summary.csv", "sr_quality.csv", "real_lr_holdout.csv",
                  "training_summary.csv"]:
        src = results_dir / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)

    images_src = results_dir / "sr_comparison_images"
    if images_src.exists():
        images_dst = out_dir / "sr_comparison_images"
        if images_dst.exists():
            shutil.rmtree(images_dst)
        shutil.copytree(images_src, images_dst)

    # --- Bảng chính: accuracy theo backbone x domain (chỉ domain chính, bỏ dòng chẩn đoán hr_sr_*) ---
    main_rows = []
    for r in summary_rows:
        train_d = r.get("train_domain", "")
        test_d = r.get("test_domain", "")
        if train_d == test_d and train_d in ("hr", "lr", "sr_baseline", "sr_improved"):
            main_rows.append({
                "backbone": r.get("backbone", ""),
                "domain": train_d,
                "identity_accuracy": r.get("identity_accuracy", ""),
                # [MỚI — bổ sung journal Q1] rank-5, AUC/EER (verification-setting)
                # đã có sẵn trong summary.csv (xem data/aggregate_results.py) nhưng
                # trước đây KHÔNG được đưa vào REPORT.md — chỉ có identity_accuracy
                # top-1 + gender_accuracy, khá mỏng cho phần Results của bài báo.
                "identity_accuracy_rank5": r.get("identity_accuracy_rank5", ""),
                "identity_auc": r.get("identity_auc", ""),
                "identity_eer": r.get("identity_eer", ""),
                "gender_accuracy": r.get("gender_accuracy", ""),
            })
    # sắp theo backbone rồi theo thứ tự domain hr->lr->sr_baseline->sr_improved
    domain_order = {"hr": 0, "lr": 1, "sr_baseline": 2, "sr_improved": 3}
    main_rows.sort(key=lambda r: (r["backbone"], domain_order.get(r["domain"], 9)))

    # [MỚI — phát hiện qua code review] không có bước nào trước đây đối chiếu
    # số dòng thu được với số tổ hợp backbone x domain KỲ VỌNG theo config —
    # nếu 1 thí nghiệm chưa chạy xong, dòng đó lặng lẽ biến mất khỏi REPORT.md
    # mà không ai biết, dễ tưởng bảng đã đầy đủ.
    expected_backbones = cfg["recognition"]["backbones"]
    expected_combos = {(b, d) for b in expected_backbones for d in domain_order}
    found_combos = {(r["backbone"], r["domain"]) for r in main_rows}
    missing_combos = sorted(expected_combos - found_combos)
    if missing_combos:
        print(f"!!! CẢNH BÁO: thiếu {len(missing_combos)}/{len(expected_combos)} tổ hợp "
              f"backbone x domain trong summary.csv: {missing_combos}")

    # --- Bảng SR quality: chỉ giữ dòng mới nhất cho mỗi label (phòng khi file bị append nhiều lần) ---
    sr_quality_latest = {}
    for r in sr_quality_rows:
        sr_quality_latest[r.get("label", "")] = r  # dict giữ lại dòng cuối cùng cho mỗi label
    sr_quality_dedup = list(sr_quality_latest.values())

    report_lines = []
    report_lines.append("# BÁO CÁO TỔNG HỢP KẾT QUẢ — TỰ ĐỘNG SINH\n")
    report_lines.append(f"Kiến trúc student hiện tại: **{student_arch}** | "
                         f"Lambda: pixel={lambda_pixel}, distill={lambda_distill}, "
                         f"identity={lambda_identity}\n")

    # [SỬA — bug phát hiện qua review Q1] bản trước chỉ liệt kê "flops_G",
    # thiếu "macs_G" (cột mới sau khi sửa lỗi nhãn MACs/FLOPs trong
    # utils/metrics.py::count_flops()) — REPORT.md sẽ thiếu số MACs dù CSV
    # gốc đã có, trong khi nhiều bài NTIRE báo MACs (gọi nhầm là "FLOPs").
    report_lines.append("## 1. Chất lượng ảnh SR (PSNR/SSIM/LPIPS/Params/MACs/FLOPs/Latency)\n")
    report_lines.append(md_table(sr_quality_dedup,
                                  columns=["label", "arch", "psnr_db", "ssim", "lpips", "params_M",
                                           "params_deploy_M", "macs_G", "flops_G", "latency_ms"]))

    report_lines.append("\n## 2. Độ chính xác nhận diện — theo backbone x domain\n")
    report_lines.append(md_table(main_rows))

    if real_lr_rows:
        report_lines.append("\n## 3. Kiểm chứng trên ảnh LR thật (real_lr_holdout)\n")
        report_lines.append(md_table(real_lr_rows))

    if training_summary_rows:
        report_lines.append("\n## 4. Tóm tắt training (epoch dừng, OOM, thời gian)\n")
        report_lines.append(md_table(training_summary_rows,
                                      columns=["run_name", "epochs_run", "early_stopped",
                                               "oom_fallback_count", "duration_minutes"]))

    if images_src.exists():
        n_images = len(list((out_dir / "sr_comparison_images").glob("sample_*.png")))
        report_lines.append(f"\n## 5. Ảnh so sánh trực quan\n")
        report_lines.append(f"Xem thư mục `sr_comparison_images/` ({n_images} ảnh mẫu, "
                             f"mỗi ảnh gồm LR | SPAN baseline | {student_arch} | HR).\n")

    with open(out_dir / "REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Đã gộp toàn bộ kết quả vào: {out_dir}/")
    print(f"  - {out_dir}/REPORT.md  <- đọc file này trước tiên")
    for fname in ["summary.csv", "sr_quality.csv", "real_lr_holdout.csv", "training_summary.csv"]:
        if (out_dir / fname).exists():
            print(f"  - {out_dir}/{fname}")
    if images_src.exists():
        print(f"  - {out_dir}/sr_comparison_images/")


if __name__ == "__main__":
    main()
