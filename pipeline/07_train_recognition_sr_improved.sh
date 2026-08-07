#!/bin/bash
# BƯỚC 7/8: Train recognition trên domain sr_improved cho từng backbone.
set -e

CONFIG="configs/config.yaml"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")

echo "################################################################"
echo "# BƯỚC 7/8: Train recognition trên domain sr_improved           #"
echo "################################################################"

for BACKBONE in "${BACKBONES[@]}"; do
    echo ">>> [domain=sr_improved] backbone=$BACKBONE (fine-tune từ checkpoint lr)"
    python train_recognition.py --config "$CONFIG" --domain sr_improved --backbone "$BACKBONE" \
        --init_ckpt "runs/recognition_lr_${BACKBONE}/best.pt"
done

echo ""
echo "HOÀN TẤT BƯỚC 7/8. Chạy tiếp: bash pipeline/08_benchmark_and_aggregate.sh"
