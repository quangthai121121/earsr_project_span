#!/bin/bash
# CHẠY 1 LƯỢT: Bước 6 (nén SPAN) -> Bước 7 (recognition) -> Bước 8 (benchmark)
# -> xuất ảnh so sánh -> gộp TOÀN BỘ kết quả vào results/final_report/.
#
# Dùng khi đã đổi kiến trúc span_tiny hoặc trọng số lambda trong config.yaml,
# cần chạy lại toàn bộ và thu thập kết quả 1 lần, không phải chạy từng lệnh.
#
# YÊU CẦU: Bước 1-5 đã chạy xong trước đó (script này tự kiểm tra, báo lỗi rõ
# ràng nếu thiếu, không tự chạy lại Bước 1-5 vì tốn thời gian không cần thiết
# nếu dữ liệu/checkpoint gốc chưa đổi).

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results"
FINAL_DIR="${RESULTS_DIR}/final_report"  # [SỬA — bug lẫn dataset] trước đây hardcode "results/final_report",
# không ăn theo RESULTS_DIR -> khi genericize cho dataset thứ 2 (AWE...) sẽ ghi ĐÈ vào results/final_report
# của EarVN1.0 thay vì results_<dataset>/final_report. Giờ suy ra từ RESULTS_DIR, luôn đúng dataset.

echo "################################################################"
echo "# KIỂM TRA ĐIỀU KIỆN TIÊN QUYẾT (Bước 1-5 đã xong chưa)        #"
echo "################################################################"

check_exists() {
    if [ ! -e "$1" ]; then
        echo "!!! THIẾU: $1"
        echo "!!! Chạy trước: $2"
        exit 1
    fi
}

check_exists "splits/hr/train" "bash pipeline/01_survey_and_prepare_data.sh"
check_exists "checkpoints/span_pretrained_x4.pth" "bash pipeline/02_setup_span_official.sh (+ tải checkpoint bằng tay)"
check_exists "runs/recognition_hr_mobilenet_v2/best.pt" "bash pipeline/03_train_baseline_recognition.sh"
check_exists "runs/sr_span_official/best.pt" "bash pipeline/04_train_teacher_and_span_baseline.sh"
check_exists "runs/recognition_sr_baseline_mobilenet_v2/best.pt" "bash pipeline/05_train_recognition_sr_baseline.sh"
echo ">>> OK: Bước 1-5 đã sẵn sàng."
echo ""

STUDENT_ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")

echo "################################################################"
echo "# DỌN DỮ LIỆU CŨ CỦA span_tiny/sr_improved (tránh lẫn checkpoint cũ) #"
echo "################################################################"
rm -rf "runs/sr_improved_${STUDENT_ARCH}" "splits/sr_improved"
rm -rf runs/recognition_sr_improved_*
echo ">>> Đã dọn xong."
echo ""

echo "################################################################"
echo "# BƯỚC 6: Nén SPAN (${STUDENT_ARCH})                          #"
echo "################################################################"
bash pipeline/06_improve_span.sh

echo ""
echo "################################################################"
echo "# BƯỚC 7: Train recognition trên domain sr_improved            #"
echo "################################################################"
bash pipeline/07_train_recognition_sr_improved.sh

echo ""
echo "################################################################"
echo "# BƯỚC 8: Benchmark + tổng hợp bảng kết quả chính               #"
echo "################################################################"
bash pipeline/08_benchmark_and_aggregate.sh

echo ""
echo "################################################################"
echo "# BƯỚC 9: Xuất ảnh so sánh trực quan (HR | LR | baseline | improved) #"
echo "################################################################"
SR_ARCH=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['sr']['arch'])")
python export_sr_comparison_images.py --config "$CONFIG" \
    --sr_baseline_ckpt "runs/sr_${SR_ARCH}/best.pt" --sr_baseline_arch "$SR_ARCH" \
    --sr_improved_ckpt "runs/sr_improved_${STUDENT_ARCH}/best.pt" --sr_improved_arch "$STUDENT_ARCH" \
    --n_samples 20 --out_dir "$RESULTS_DIR/sr_comparison_images"

echo ""
echo "################################################################"
echo "# GỘP TOÀN BỘ KẾT QUẢ VÀO 1 NƠI: $FINAL_DIR                    #"
echo "################################################################"
python data/generate_final_report.py --config "$CONFIG" --results_dir "$RESULTS_DIR" \
    --out_dir "$FINAL_DIR"

echo ""
echo "################################################################"
echo "# HOÀN TẤT. Xem toàn bộ kết quả tại: $FINAL_DIR/               #"
echo "################################################################"
echo "  - $FINAL_DIR/REPORT.md          <- đọc file này trước tiên, tổng hợp mọi bảng số liệu"
echo "  - $FINAL_DIR/summary.csv"
echo "  - $FINAL_DIR/sr_quality.csv"
echo "  - $FINAL_DIR/sr_comparison_images/  <- ảnh HR/LR/baseline/improved"
