#!/bin/bash
# Tạo bản sao config + pipeline RIÊNG cho 1 dataset thứ 2 (hoặc thứ 3),
# KHÔNG đụng vào configs/config.yaml hay pipeline/ gốc (vẫn dùng cho EarVN1.0).
# Chạy 1 lần cho mỗi dataset mới.
#
# Dùng:  bash scripts/setup_second_dataset_pipeline.sh <ten_dataset> <so_identity>
# Ví dụ: bash scripts/setup_second_dataset_pipeline.sh awe 100
#        bash scripts/setup_second_dataset_pipeline.sh awex 346
#        bash scripts/setup_second_dataset_pipeline.sh iitdelhi 121   # XÁC NHẬN LẠI số identity thật khi tải dataset
#
# CHẠY TỪ THƯ MỤC GỐC CỦA PROJECT (earsr_project/), nơi có sẵn configs/config.yaml
# và pipeline/.

set -e

if [ $# -ne 2 ]; then
    echo "Dùng: bash scripts/setup_second_dataset_pipeline.sh <ten_dataset> <so_identity>"
    echo "Ví dụ: bash scripts/setup_second_dataset_pipeline.sh awe 100"
    exit 1
fi

NAME="$1"
NUM_ID="$2"
CONFIG_NEW="configs/config_${NAME}.yaml"
PIPELINE_NEW="pipeline_${NAME}"
RAW_DIR_NEW="raw_data/${NAME}_raw"

if [ ! -f "configs/config.yaml" ] || [ ! -d "pipeline" ]; then
    echo "LỖI: không thấy configs/config.yaml hoặc pipeline/ — chạy script này từ thư mục gốc project."
    exit 1
fi

if [ -f "$CONFIG_NEW" ] || [ -d "$PIPELINE_NEW" ]; then
    echo "LỖI: $CONFIG_NEW hoặc $PIPELINE_NEW đã tồn tại — xóa/đổi tên trước nếu muốn tạo lại từ đầu."
    exit 1
fi

echo ">>> [1/3] Tạo $CONFIG_NEW từ configs/config.yaml..."
cp configs/config.yaml "$CONFIG_NEW"

# Đường dẫn dữ liệu/kết quả RIÊNG cho dataset mới — không dùng chung với EarVN1.0
sed -i "s|raw_images: \".*\"|raw_images: \"${RAW_DIR_NEW}\"|" "$CONFIG_NEW"
sed -i "s|splits_root: \"splits\"|splits_root: \"splits_${NAME}\"|" "$CONFIG_NEW"
sed -i "s|runs_root: \"runs\"|runs_root: \"runs_${NAME}\"|" "$CONFIG_NEW"
sed -i "s|results_root: \"results\"|results_root: \"results_${NAME}\"|" "$CONFIG_NEW"
sed -i "s|num_identities: 164|num_identities: ${NUM_ID}|" "$CONFIG_NEW"

# Bài số 1 KHÔNG dùng gender cho dataset phụ — tắt hẳn gender loss để tránh
# nhiễu vào embedding chung. Lý do: prepare_splits.py::gender_from_person_id()
# giả định quy ước riêng của EarVN1.0 (001-098=nam, 099-164=nữ theo person_id
# đánh số lại) — KHÔNG đúng cho dataset khác, nhãn gender sinh ra ở đây chỉ là
# placeholder vô nghĩa, không nên dùng để huấn luyện hay báo cáo.
sed -i "s|gender_loss_weight: 0.5|gender_loss_weight: 0.0|" "$CONFIG_NEW"

# 2 đường dẫn checkpoint này viết cứng "runs/..." trong config gốc (không tự
# theo runs_root) — phải trỏ đúng sang runs_<dataset>/ để không lẫn với EarVN1.0.
sed -i "s|teacher_ckpt: \"runs/sr_span_official/best.pt\"|teacher_ckpt: \"runs_${NAME}/sr_span_official/best.pt\"|" "$CONFIG_NEW"
sed -i "s|frozen_recognition_ckpt: \"runs/recognition_hr_mobilenet_v2/best.pt\"|frozen_recognition_ckpt: \"runs_${NAME}/recognition_hr_mobilenet_v2/best.pt\"|" "$CONFIG_NEW"

echo ">>> [2/3] Tạo $PIPELINE_NEW/ (sao chép pipeline/ gốc, đổi CONFIG + RAW_DIR bên trong)..."
cp -r pipeline "$PIPELINE_NEW"

# BƯỚC 2 (setup_span_official) KHÔNG cần chạy lại — dùng chung checkpoint SPAN
# pretrained gốc (đã tải 1 lần cho EarVN1.0, không phụ thuộc dataset ear nào).
rm -f "$PIPELINE_NEW/02_setup_span_official.sh"

for f in "$PIPELINE_NEW"/*.sh; do
    sed -i "s|CONFIG=\"configs/config.yaml\"|CONFIG=\"${CONFIG_NEW}\"|" "$f"
    sed -i "s|RAW_DIR=\"raw_data/EarVN1.0_raw\"|RAW_DIR=\"${RAW_DIR_NEW}\"|" "$f"
done

mkdir -p "$RAW_DIR_NEW"

echo ">>> [3/3] Kiểm tra nhanh: đếm số dòng CONFIG đã đổi đúng trong từng script..."
grep -L "CONFIG=\"${CONFIG_NEW}\"" "$PIPELINE_NEW"/*.sh 2>/dev/null | grep -v run_all.sh \
    && echo "   (các file trên KHÔNG có dòng CONFIG= — bình thường nếu là script không cần CONFIG, kiểm tra lại nếu nghi ngờ)" \
    || echo "   OK: mọi script .sh đều đã trỏ đúng $CONFIG_NEW"

echo ""
echo "################################################################"
echo "HOÀN TẤT thiết lập cho dataset '$NAME'."
echo "################################################################"
echo "1. Chuyển đổi dữ liệu gốc sang đúng định dạng (xem"
echo "   convert_generic_dataset_to_project_format.py), copy/symlink kết quả vào:"
echo "       $RAW_DIR_NEW/"
echo "   (chạy --dry_run trước để kiểm tra kế hoạch chuyển đổi)"
echo ""
echo "2. Chạy tuần tự (BỎ QUA bước 02, đã xóa — dùng chung checkpoint SPAN gốc):"
echo "       bash ${PIPELINE_NEW}/01_survey_and_prepare_data.sh"
echo "       bash ${PIPELINE_NEW}/03_train_baseline_recognition.sh"
echo "       bash ${PIPELINE_NEW}/04_train_teacher_and_span_baseline.sh"
echo "       bash ${PIPELINE_NEW}/05_train_recognition_sr_baseline.sh"
echo "       bash ${PIPELINE_NEW}/06_improve_span.sh"
echo "       bash ${PIPELINE_NEW}/07_train_recognition_sr_improved.sh"
echo "       bash ${PIPELINE_NEW}/08_benchmark_and_aggregate.sh"
echo ""
echo "3. Đa seed (khuyến nghị bắt buộc trước khi dùng cho bài báo):"
echo "       bash ${PIPELINE_NEW}/run_multi_seed.sh"
echo ""
echo "Toàn bộ kết quả sẽ nằm trong results_${NAME}/ (KHÔNG lẫn với results/ của EarVN1.0)."
