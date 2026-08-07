#!/bin/bash
# BƯỚC 1/8: Khảo sát EarVN1.0 raw, TỰ ĐỘNG chọn ngưỡng lọc dựa trên số liệu
# thật, cập nhật configs/config.yaml theo ngưỡng đã chọn, rồi tạo HR/LR.
#
# Trước khi chạy: copy dữ liệu EarVN1.0 vào raw_data/EarVN1.0_raw/
# (xem raw_data/EarVN1.0_raw/README_ĐẶT_DATA_VÀO_ĐÂY.md)

set -e

RAW_DIR="raw_data/EarVN1.0_raw"
CONFIG="configs/config.yaml"

if [ ! -d "$RAW_DIR" ] || [ -z "$(ls -A "$RAW_DIR" 2>/dev/null | grep -v README)" ]; then
    echo "LỖI: $RAW_DIR trống. Copy dữ liệu EarVN1.0 vào đó trước, xem README trong thư mục này."
    exit 1
fi

echo "################################################################"
echo "# BƯỚC 1/8: Khảo sát + tự động chọn ngưỡng + tạo HR/LR         #"
echo "################################################################"

echo ">>> Khảo sát + tự động chọn ngưỡng + chia train/val/test..."
python data/prepare_splits.py --raw_dir "$RAW_DIR" --out_dir splits/ \
    --auto --report_out splits/threshold_report.txt

echo ""
echo ">>> Cập nhật configs/config.yaml theo ngưỡng vừa chọn..."
python pipeline/_update_config_from_report.py \
    --report splits/threshold_report.txt --config "$CONFIG"

echo ""
echo ">>> Đọc lại hr_size/scale từ config, tạo tập HR (letterbox) + LR (downsample)..."
HR_SIZE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['hr_size'])")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")
python data/build_lr.py --splits splits/splits.json --hr_size "$HR_SIZE" --scale "$SCALE" --out_dir splits

echo ""
echo "HOÀN TẤT BƯỚC 1/8. Xem splits/threshold_report.txt để biết ngưỡng đã chọn."
echo "Chạy tiếp: bash pipeline/02_setup_span_official.sh"
