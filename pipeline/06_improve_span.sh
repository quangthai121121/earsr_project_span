#!/bin/bash
# BƯỚC 6/8: Cải tiến SPAN — NÉN kiến trúc (student nhẹ hơn) qua distillation từ
# SPAN baseline + identity-aware loss (trọng số thấp), sinh tập sr_improved.
set -e

CONFIG="configs/config.yaml"

echo "################################################################"
echo "# BƯỚC 6/8: Nén SPAN — nhẹ hơn/nhanh hơn baseline (đóng góp chính) #"
echo "################################################################"

echo ">>> Train SPAN đã nén (student_arch)..."
python train_sr_distill.py --config "$CONFIG"

echo ""
echo ">>> Sinh tập sr_improved..."
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")
STUDENT_ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")
python data/build_sr.py --lr_dir splits/lr --sr_ckpt "runs/sr_improved_${STUDENT_ARCH}/best.pt" \
    --arch "$STUDENT_ARCH" --scale "$SCALE" --out_dir splits/sr_improved

echo ""
echo "HOÀN TẤT BƯỚC 6/8. Chạy tiếp: bash pipeline/07_train_recognition_sr_improved.sh"
