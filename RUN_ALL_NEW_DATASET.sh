#!/bin/bash
# ================================================================
# CHẠY TỪ ĐẦU ĐẾN CUỐI CHO 1 DATASET MỚI (AWE/AWEx/IIT Delhi/...)
# Gộp: setup config+pipeline -> 8 bước chính -> multi-seed cơ bản (lr/
# sr_baseline/sr_improved) -> span_large ablation (Tier 1) -> tổng hợp.
# CHẠY 1 LỆNH DUY NHẤT, không cần can thiệp giữa chừng (trừ khi có lỗi).
# ================================================================
#
# YÊU CẦU TRƯỚC KHI CHẠY (làm thủ công, KHÔNG tự động — có chủ đích, xem lý
# do bên dưới):
#   1. Đã tải dataset gốc (AWE/AWEx/IIT Delhi...) về máy.
#   2. Đã chạy convert_generic_dataset_to_project_format.py với --dry_run
#      TRƯỚC, đọc kỹ output (số identity/ảnh có khớp thông tin dataset công
#      bố không), rồi mới chạy thật (bỏ --dry_run) để tạo
#      raw_data/<ten_dataset>_raw/NNN/*.jpg
#      (KHÔNG gộp bước này vào script tự động — đây là bước kiểm tra bằng
#      mắt bắt buộc, sai ở đây sẽ lan sang toàn bộ kết quả sau, không nên
#      tự động hoá bỏ qua bước kiểm tra người dùng)
#   3. Đã copy 3 file bản đã patch vào thư mục gốc project (LÀM 1 LẦN, dùng
#      chung cho mọi dataset kể cả EarVN1.0):
#         train_sr_distill.py -> ./train_sr_distill.py
#         metrics.py           -> ./utils/metrics.py
#         eval_recognition.py  -> ./eval_recognition.py
#   4. Đã copy 3 file script vào đúng vị trí (LÀM 1 LẦN):
#         convert_generic_dataset_to_project_format.py -> ./data/
#         setup_second_dataset_pipeline.sh              -> ./scripts/
#         RUN_ALL_span_large_ablation.sh                -> thư mục gốc project
#   5. File này (RUN_ALL_NEW_DATASET.sh) đặt ở thư mục gốc project.
#
# DÙNG:
#   bash RUN_ALL_NEW_DATASET.sh <ten_dataset> <so_identity>
# Ví dụ:
#   bash RUN_ALL_NEW_DATASET.sh awe 100
#   bash RUN_ALL_NEW_DATASET.sh awex 346
#   bash RUN_ALL_NEW_DATASET.sh iitdelhi 121   # xác nhận lại số identity thật
#
# CẢNH BÁO THỜI GIAN: script này chạy GẦN NHƯ TOÀN BỘ khối lượng đã tốn cho
# EarVN1.0 (8 bước chính + multi-seed 5 backbone x 3 seed x 3 domain = 45 lần
# train recognition + span_large ablation thêm 15 lần) — có thể mất RẤT
# NHIỀU giờ tùy phần cứng. Cân nhắc chạy qua đêm / dùng `nohup ... &` hoặc
# `tmux`/`screen` để không bị ngắt giữa chừng khi mất kết nối.

set -e

if [ $# -ne 2 ]; then
    echo "Dùng: bash RUN_ALL_NEW_DATASET.sh <ten_dataset> <so_identity>"
    echo "Ví dụ: bash RUN_ALL_NEW_DATASET.sh awe 100"
    exit 1
fi

NAME="$1"
NUM_ID="$2"
CONFIG_NEW="configs/config_${NAME}.yaml"
PIPELINE_NEW="pipeline_${NAME}"
RAW_DIR_NEW="raw_data/${NAME}_raw"

echo "################################################################"
echo "# KIỂM TRA TIỀN ĐỀ                                              #"
echo "################################################################"

MISSING=0
if [ ! -d "$RAW_DIR_NEW" ] || [ -z "$(ls -A "$RAW_DIR_NEW" 2>/dev/null)" ]; then
    echo "LỖI: $RAW_DIR_NEW trống hoặc chưa tồn tại."
    echo "     -> chạy convert_generic_dataset_to_project_format.py (--dry_run TRƯỚC, rồi chạy thật) để tạo dữ liệu ở đây."
    MISSING=1
fi
if [ ! -f "train_sr_distill.py" ] || ! grep -q "student_arch" train_sr_distill.py 2>/dev/null; then
    echo "LỖI: ./train_sr_distill.py chưa phải bản đã patch — copy đè bản đã patch vào trước."
    MISSING=1
fi
if [ ! -f "utils/metrics.py" ] || ! grep -q "compute_topk_accuracy" utils/metrics.py 2>/dev/null; then
    echo "LỖI: ./utils/metrics.py chưa phải bản đã patch — copy đè bản đã patch vào trước."
    MISSING=1
fi
if [ ! -f "eval_recognition.py" ] || ! grep -q "all_id_rank5_hits" eval_recognition.py 2>/dev/null; then
    echo "LỖI: ./eval_recognition.py chưa phải bản đã patch — copy đè bản đã patch vào trước."
    MISSING=1
fi
if [ ! -f "data/convert_generic_dataset_to_project_format.py" ]; then
    echo "LỖI: chưa thấy data/convert_generic_dataset_to_project_format.py"; MISSING=1
fi
if [ ! -f "scripts/setup_second_dataset_pipeline.sh" ]; then
    echo "LỖI: chưa thấy scripts/setup_second_dataset_pipeline.sh"; MISSING=1
fi
if [ ! -f "RUN_ALL_span_large_ablation.sh" ]; then
    echo "LỖI: chưa thấy ./RUN_ALL_span_large_ablation.sh"; MISSING=1
fi
if [ ! -f "checkpoints/span_pretrained_x4.pth" ]; then
    echo "LỖI: chưa thấy checkpoints/span_pretrained_x4.pth (checkpoint SPAN pretrained gốc,"
    echo "     dùng chung cho mọi dataset — phải có sẵn từ lúc setup EarVN1.0)."
    MISSING=1
fi

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "DỪNG LẠI — thiếu tiền đề ở trên, xem chi tiết phía trên rồi chạy lại."
    exit 1
fi
echo "OK — mọi tiền đề đã sẵn sàng."

echo ""
echo "################################################################"
echo "# [A] Thiết lập config + pipeline riêng cho '$NAME'              #"
echo "################################################################"
if [ -f "$CONFIG_NEW" ] || [ -d "$PIPELINE_NEW" ]; then
    echo ">>> $CONFIG_NEW / $PIPELINE_NEW đã tồn tại — bỏ qua bước thiết lập, dùng lại cấu hình cũ."
    echo "    (nếu muốn thiết lập lại từ đầu, xóa 2 thứ trên trước khi chạy script này)"
else
    bash scripts/setup_second_dataset_pipeline.sh "$NAME" "$NUM_ID"
fi

echo ""
echo "################################################################"
echo "# [B] Chạy 8 bước chính (1 lần chạy cơ bản)                      #"
echo "################################################################"
bash "${PIPELINE_NEW}/01_survey_and_prepare_data.sh"
bash "${PIPELINE_NEW}/03_train_baseline_recognition.sh"
bash "${PIPELINE_NEW}/04_train_teacher_and_span_baseline.sh"
bash "${PIPELINE_NEW}/05_train_recognition_sr_baseline.sh"
bash "${PIPELINE_NEW}/06_improve_span.sh"
bash "${PIPELINE_NEW}/07_train_recognition_sr_improved.sh"
bash "${PIPELINE_NEW}/08_benchmark_and_aggregate.sh"

echo ""
echo "################################################################"
echo "# [C] Multi-seed cơ bản (lr / sr_baseline / sr_improved)         #"
echo "#     5 backbone x 3 seed x 3 domain = 45 lần train recognition  #"
echo "################################################################"
bash "${PIPELINE_NEW}/run_multi_seed.sh"

echo ""
echo "################################################################"
echo "# [D] Tier 1 architecture ablation (span_large)                  #"
echo "################################################################"
bash RUN_ALL_span_large_ablation.sh "$CONFIG_NEW"

RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_NEW'))['paths']['results_root'])")

echo ""
echo "################################################################"
echo "# HOÀN TẤT TOÀN BỘ cho dataset '$NAME'                          #"
echo "################################################################"
echo "File kết quả cần gửi lại để viết journal:"
echo "  - ${RESULTS_ROOT}/summary.csv"
echo "  - ${RESULTS_ROOT}/sr_quality.csv"
echo "  - ${RESULTS_ROOT}/multi_seed/multi_seed_summary.csv"
echo "  - ${RESULTS_ROOT}/span_large_ablation/multi_seed_summary_4domains.csv"
echo "  - splits_${NAME}/dataset_stats.csv"
