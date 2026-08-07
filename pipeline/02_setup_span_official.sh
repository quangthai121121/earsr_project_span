#!/bin/bash
# BƯỚC 2/8: Tải code + checkpoint SPAN chính thức (chỉ cần chạy 1 lần).
set -e

echo "################################################################"
echo "# BƯỚC 2/8: Tải SPAN chính thức                                #"
echo "################################################################"

if [ -f "checkpoints/span_pretrained_x4.pth" ]; then
    echo "Đã có checkpoints/span_pretrained_x4.pth, bỏ qua."
else
    bash scripts/setup_span_official.sh
    echo ""
    echo "!!! DỪNG LẠI: làm 3 việc thủ công theo hướng dẫn ở trên (đặt tên file"
    echo "!!! checkpoint đúng, đối chiếu span_arch.py với wrapper) TRƯỚC KHI chạy tiếp."
    exit 0
fi

echo ""
echo "HOÀN TẤT BƯỚC 2/8. Chạy tiếp: bash pipeline/03_train_baseline_recognition.sh"
