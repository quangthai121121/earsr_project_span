#!/bin/bash
# Chạy 4 cấu hình ablation (pixel_only / pixel_distill / pixel_identity / full)
# để cô lập tác dụng của từng thành phần loss (docs/03_span_improvement.md).
# Dùng 1 backbone duy nhất (mobilenet_v2) để tiết kiệm thời gian — đổi
# BACKBONE bên dưới nếu muốn chạy ablation trên backbone khác hoặc cả 5.
#
# YÊU CẦU: đã chạy xong pipeline 01-05 (có sẵn checkpoint recognition_lr_*,
# EDSR teacher, recognition_hr_mobilenet_v2 làm giám khảo).

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results"
BACKBONE="mobilenet_v2"
mkdir -p "$RESULTS_DIR"

ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

declare -A CONFIGS=(
    [pixel_only]="1.0 0.0 0.0"
    [pixel_distill]="1.0 0.5 0.0"
    [pixel_identity]="1.0 0.0 0.5"
    [full]="1.0 0.5 0.5"
)

for NAME in pixel_only pixel_distill pixel_identity full; do
    read -r LP LD LI <<< "${CONFIGS[$NAME]}"
    echo "################################################################"
    echo "# Ablation: $NAME (lambda_pixel=$LP lambda_distill=$LD lambda_identity=$LI)"
    echo "################################################################"

    python train_sr_distill.py --config "$CONFIG" \
        --lambda_pixel "$LP" --lambda_distill "$LD" --lambda_identity "$LI" \
        --run_suffix "_ablation_${NAME}"

    python data/build_sr.py --lr_dir splits/lr \
        --sr_ckpt "runs/sr_improved_${ARCH}_ablation_${NAME}/best.pt" \
        --arch "$ARCH" --scale "$SCALE" --out_dir "splits/sr_ablation_${NAME}"

    python train_recognition.py --config "$CONFIG" --domain "sr_ablation_${NAME}" \
        --backbone "$BACKBONE" --init_ckpt "runs/recognition_lr_${BACKBONE}/best.pt"

    python eval_recognition.py --config "$CONFIG" \
        --ckpt "runs/recognition_sr_ablation_${NAME}_${BACKBONE}/best.pt" \
        --backbone "$BACKBONE" --train_domain "sr_ablation_${NAME}" \
        --test_domain "sr_ablation_${NAME}" \
        --out_json "$RESULTS_DIR/ablation_${NAME}.json"
done

echo ""
echo ">>> Tổng hợp kết quả ablation..."
python data/aggregate_ablation_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/ablation.csv"

echo ""
echo "HOÀN TẤT ablation. Kết quả: $RESULTS_DIR/ablation.csv"
