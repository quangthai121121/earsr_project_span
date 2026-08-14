#!/bin/bash
# Tạo bản sao config + pipeline RIÊNG cho 1 dataset thứ 2 (hoặc thứ 3),
# KHÔNG đụng vào configs/config.yaml hay pipeline/ gốc (vẫn dùng cho EarVN1.0).
# Chạy 1 lần cho mỗi dataset mới.
#
# BẢN SỬA LỖI: bản trước chỉ đổi biến CONFIG=/RAW_DIR= nên bỏ sót nhiều
# đường dẫn "runs/", "splits/", "results/" bị viết cứng (hardcode) ngay
# trong thân các script pipeline/*.sh — gây lỗi FileNotFoundError VÀ rủi ro
# ghi đè dữ liệu splits/ của EarVN1.0 khi chạy dataset phụ. Bản này sed-thay
# TOÀN BỘ các đường dẫn đó, đã đối chiếu với nội dung THẬT của cả 8 script
# (01,03,04,05,06,07,08 + run_multi_seed.sh) trước khi viết.
#
# Dùng:  bash scripts/setup_second_dataset_pipeline.sh <ten_dataset> <so_identity>
# Ví dụ: bash scripts/setup_second_dataset_pipeline.sh awe 100
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
    echo "LỖI: $CONFIG_NEW hoặc $PIPELINE_NEW đã tồn tại."
    echo "     -> Nếu đang khắc phục lỗi từ lần chạy trước: xóa cả 2 rồi chạy lại script này:"
    echo "        rm -rf $CONFIG_NEW $PIPELINE_NEW"
    exit 1
fi

echo ">>> [1/3] Tạo $CONFIG_NEW từ configs/config.yaml..."
cp configs/config.yaml "$CONFIG_NEW"

sed -i "s|raw_images: \".*\"|raw_images: \"${RAW_DIR_NEW}\"|" "$CONFIG_NEW"
sed -i "s|splits_root: \"splits\"|splits_root: \"splits_${NAME}\"|" "$CONFIG_NEW"
sed -i "s|runs_root: \"runs\"|runs_root: \"runs_${NAME}\"|" "$CONFIG_NEW"
sed -i "s|results_root: \"results\"|results_root: \"results_${NAME}\"|" "$CONFIG_NEW"
sed -i "s|num_identities: 164|num_identities: ${NUM_ID}|" "$CONFIG_NEW"
sed -i "s|gender_loss_weight: 0.5|gender_loss_weight: 0.0|" "$CONFIG_NEW"
sed -i "s|teacher_ckpt: \"runs/sr_span_official/best.pt\"|teacher_ckpt: \"runs_${NAME}/sr_span_official/best.pt\"|" "$CONFIG_NEW"
sed -i "s|frozen_recognition_ckpt: \"runs/recognition_hr_mobilenet_v2/best.pt\"|frozen_recognition_ckpt: \"runs_${NAME}/recognition_hr_mobilenet_v2/best.pt\"|" "$CONFIG_NEW"

echo ">>> [2/3] Tạo $PIPELINE_NEW/ (sao chép pipeline/ gốc, sửa TOÀN BỘ đường dẫn bên trong)..."
cp -r pipeline "$PIPELINE_NEW"

# BƯỚC 2 (setup_span_official) KHÔNG cần chạy lại — dùng chung checkpoint SPAN
# pretrained gốc (đã tải 1 lần cho EarVN1.0, không phụ thuộc dataset ear nào).
rm -f "$PIPELINE_NEW/02_setup_span_official.sh"

for f in "$PIPELINE_NEW"/*.sh; do
    # 1) Biến CONFIG=/RAW_DIR= (như bản trước)
    sed -i "s|CONFIG=\"configs/config.yaml\"|CONFIG=\"${CONFIG_NEW}\"|" "$f"
    sed -i "s|RAW_DIR=\"raw_data/EarVN1.0_raw\"|RAW_DIR=\"${RAW_DIR_NEW}\"|" "$f"

    # 2) Biến RESULTS_DIR="results..." viết cứng trong 08 và run_multi_seed.sh/
    #    run_ablation.sh/run_lambda_sweep.sh (ví dụ RESULTS_DIR="results/multi_seed")
    sed -i "s|RESULTS_DIR=\"results|RESULTS_DIR=\"results_${NAME}|g" "$f"

    # 3) MỌI đường dẫn "runs/..." viết cứng rải rác trong thân script (checkpoint
    #    paths — đã đối chiếu: KHÔNG có trường hợp "runs/" là substring của từ
    #    khác trong các script này, an toàn để thay toàn bộ)
    sed -i "s|runs/|runs_${NAME}/|g" "$f"

    # 4) MỌI đường dẫn "splits/..." viết cứng (bao gồm cả các dòng echo hiển thị
    #    threshold_report.txt — sửa luôn cho khớp, tránh gây hiểu lầm)
    sed -i "s|splits/|splits_${NAME}/|g" "$f"

    # 5) Trường hợp riêng: "--out_dir splits" KHÔNG có dấu / ở cuối dòng
    #    (data/build_lr.py trong bước 01) — pattern ở (4) không bắt được vì
    #    thiếu dấu "/", cần xử lý riêng, neo cuối dòng cho chính xác.
    sed -i "s|--out_dir splits\$|--out_dir splits_${NAME}|" "$f"

    # 6) Tham chiếu "pipeline/..." (gọi tới pipeline/_update_config_from_report.py,
    #    và các dòng echo gợi ý bước tiếp theo) -> trỏ đúng sang bản sao
    #    pipeline_<dataset>/ (file _update_config_from_report.py đã được copy
    #    cùng ở bước cp -r phía trên, nên đường dẫn mới vẫn tồn tại và chạy đúng)
    sed -i "s|pipeline/|${PIPELINE_NEW}/|g" "$f"
done

mkdir -p "$RAW_DIR_NEW"

echo ">>> [3/3] Kiểm tra nhanh: tìm dấu vết đường dẫn CHƯA được sửa (nếu có)..."
LEFTOVER=$(grep -lE '(^|[^_a-zA-Z0-9])(runs|splits|results)/' "$PIPELINE_NEW"/*.sh 2>/dev/null | \
    xargs -I{} grep -nE '(^|[^_a-zA-Z0-9])(runs|splits|results)/' {} 2>/dev/null | \
    grep -v "runs_${NAME}/" | grep -v "splits_${NAME}/" | grep -v "results_${NAME}/" || true)
if [ -n "$LEFTOVER" ]; then
    echo "CẢNH BÁO: vẫn còn dấu vết đường dẫn chưa được sed thay — kiểm tra thủ công trước khi chạy:"
    echo "$LEFTOVER"
else
    echo "   OK: không còn đường dẫn runs/splits/results viết cứng chưa sửa."
fi

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
