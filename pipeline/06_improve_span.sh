#!/bin/bash
# BƯỚC 6/8: Cải tiến SPAN — distillation từ EDSR + identity-aware loss, sinh
# tập sr_improved.
set -e

CONFIG="configs/config.yaml"

echo "################################################################"
echo "# BƯỚC 6/8: Cải tiến SPAN (đóng góp chính)                     #"
echo "################################################################"

echo ">>> Train SPAN cải tiến..."
python train_sr_distill.py --config "$CONFIG"

echo ""
echo ">>> Sinh tập sr_improved..."
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")
python data/build_sr.py --lr_dir splits/lr --sr_ckpt runs/sr_improved_span_official/best.pt \
    --arch span_official --scale "$SCALE" --out_dir splits/sr_improved

echo ""
echo "HOÀN TẤT BƯỚC 6/8. Chạy tiếp: bash pipeline/07_train_recognition_sr_improved.sh"
