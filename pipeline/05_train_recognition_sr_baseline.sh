#!/bin/bash
# BƯỚC 5/8: Train recognition trên domain sr_baseline (xác nhận giả thuyết:
# SPAN baseline có thua domain lr hay không), cho từng backbone.
set -e

CONFIG="configs/config.yaml"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")

echo "################################################################"
echo "# BƯỚC 5/8: Train recognition trên domain sr_baseline           #"
echo "################################################################"

for BACKBONE in "${BACKBONES[@]}"; do
    echo ">>> [domain=sr_baseline] backbone=$BACKBONE (fine-tune từ checkpoint lr)"
    python train_recognition.py --config "$CONFIG" --domain sr_baseline --backbone "$BACKBONE" \
        --init_ckpt "runs/recognition_lr_${BACKBONE}/best.pt"
done

echo ""
echo "HOÀN TẤT BƯỚC 5/8."
echo ">>> Xem nhanh kết quả baseline trước khi tốn thời gian cải tiến SPAN:"
echo ">>> so sánh runs/recognition_lr_*/train.log vs runs/recognition_sr_baseline_*/train.log"
echo "Chạy tiếp: bash pipeline/06_improve_span.sh"
