#!/bin/bash
# BƯỚC 3/8: Train recognition trên domain HR (trần lý thuyết + "giám khảo"
# cho identity loss sau này) và domain LR (baseline không SR), cho từng backbone.
set -e

CONFIG="configs/config.yaml"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")

echo "################################################################"
echo "# BƯỚC 3/8: Train recognition baseline (domain HR, LR)         #"
echo "################################################################"

for BACKBONE in "${BACKBONES[@]}"; do
    echo ">>> [domain=hr] backbone=$BACKBONE"
    python train_recognition.py --config "$CONFIG" --domain hr --backbone "$BACKBONE"

    echo ">>> [domain=lr] backbone=$BACKBONE (fine-tune từ checkpoint hr)"
    python train_recognition.py --config "$CONFIG" --domain lr --backbone "$BACKBONE" \
        --init_ckpt "runs/recognition_hr_${BACKBONE}/best.pt"
done

echo ""
echo "HOÀN TẤT BƯỚC 3/8. Chạy tiếp: bash pipeline/04_train_teacher_and_span_baseline.sh"
