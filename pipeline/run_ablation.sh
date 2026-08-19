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

# [SỬA — bổ sung sau code review, vòng 5, điểm 2] Xem giải thích đầy đủ trong
# pipeline/run_prune_sparsity_screen.sh — script này CŨNG dùng checkpoint
# KHÔNG suffix seed. Fail sớm thay vì mất hàng giờ (4 cấu hình) rồi mới lỗi.
LR_CKPT_NOSUFFIX="runs/recognition_lr_${BACKBONE}/best.pt"
if [ ! -f "$LR_CKPT_NOSUFFIX" ]; then
    echo "LỖI: thiếu $LR_CKPT_NOSUFFIX (checkpoint KHÔNG suffix seed)."
    echo "     -> chạy 'python train_recognition.py --config $CONFIG --domain lr"
    echo "        --backbone $BACKBONE' (xem pipeline/03_train_baseline_recognition.sh)"
    echo "        trước — KHÁC với checkpoint '_seed<N>' của run_multi_seed.sh."
    exit 1
fi

# pixel_identity / full cần judge HR — kiểm tra NGAY (identity_judges trong
# config.yaml là 3 backbone, không chỉ mobilenet_v2).
source "$(dirname "$0")/_check_hr_judges.sh"
check_hr_judges

declare -A CONFIGS=(
    [pixel_only]="1.0 0.0 0.0"
    [pixel_distill]="1.0 1.0 0.0"
    [pixel_identity]="1.0 0.0 0.1"
    [full]="1.0 1.0 0.1"
)

for NAME in pixel_only pixel_distill pixel_identity full; do
    read -r LP LD LI <<< "${CONFIGS[$NAME]}"
    echo "################################################################"
    echo "# Ablation: $NAME (lambda_pixel=$LP lambda_distill=$LD lambda_identity=$LI)"
    echo "################################################################"

    # [SỬA — confound phát hiện qua code review] PIN TƯỜNG MINH lambda_feat=0
    # và lambda_saliency=0 — ablation này CHỈ cô lập pixel/distill/identity,
    # không được để 2 cơ chế feature-KD/saliency (thêm sau) âm thầm bật lên
    # qua default config.yaml (dù default hiện đã là 0.0, pin rõ ở đây để
    # script này KHÔNG BAO GIỜ phụ thuộc vào default tương lai của config).
    python train_sr_distill.py --config "$CONFIG" \
        --lambda_pixel "$LP" --lambda_distill "$LD" --lambda_feat 0 --lambda_saliency 0 \
        --lambda_identity "$LI" \
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
