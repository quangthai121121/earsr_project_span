#!/bin/bash
# Chạy toàn bộ 8 bước liên tiếp. Xem RUNBOOK_EarVN1.0.md mục 1-9 để biết chi
# tiết từng bước, hoặc chạy tay từng file 01_*.sh ... 08_*.sh nếu muốn theo
# dõi kỹ hơn / dừng giữa chừng.
set -e

bash pipeline/01_survey_and_prepare_data.sh

bash pipeline/02_setup_span_official.sh
if [ ! -f "checkpoints/span_pretrained_x4.pth" ]; then
    echo ""
    echo "!!! Dừng tại đây — cần bạn xác nhận thủ công checkpoint/kiến trúc SPAN"
    echo "!!! (xem hướng dẫn ở trên). Chạy lại 'bash pipeline/run_all.sh' sau khi xong."
    exit 0
fi

bash pipeline/03_train_baseline_recognition.sh
bash pipeline/04_train_teacher_and_span_baseline.sh
bash pipeline/05_train_recognition_sr_baseline.sh
bash pipeline/06_improve_span.sh
bash pipeline/07_train_recognition_sr_improved.sh
bash pipeline/08_benchmark_and_aggregate.sh

echo ""
echo "=== TOÀN BỘ PIPELINE HOÀN TẤT ==="
