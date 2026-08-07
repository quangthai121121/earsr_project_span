#!/bin/bash
# BƯỚC 4/8: Train Teacher EDSR (trần lý thuyết hướng SR) và fine-tune SPAN
# baseline từ checkpoint chính thức, sinh tập sr_baseline.
set -e

CONFIG="configs/config.yaml"

echo "################################################################"
echo "# BƯỚC 4/8: Train Teacher EDSR + SPAN baseline                 #"
echo "################################################################"

echo ">>> Train Teacher EDSR..."
python train_sr.py --config "$CONFIG" --sr_arch edsr

echo ""
echo ">>> Fine-tune SPAN baseline từ checkpoint chính thức (pixel loss thuần túy)..."
python train_sr.py --config "$CONFIG" --sr_arch span_official \
    --pretrained_path checkpoints/span_pretrained_x4.pth

echo ""
echo ">>> Sinh tập sr_baseline từ SPAN vừa fine-tune..."
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")
python data/build_sr.py --lr_dir splits/lr --sr_ckpt runs/sr_span_official/best.pt \
    --arch span_official --scale "$SCALE" --out_dir splits/sr_baseline

echo ""
echo "HOÀN TẤT BƯỚC 4/8. Chạy tiếp: bash pipeline/05_train_recognition_sr_baseline.sh"
