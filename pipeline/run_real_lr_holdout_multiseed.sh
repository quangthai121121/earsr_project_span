#!/bin/bash
# [MỚI — 2026-08-30] Mở rộng real_lr_holdout (Section res-main / res-awex-
# transfer trong bài báo) từ n=1 (chỉ seed mặc định) sang multi-seed thật —
# ĐÚNG lý do đã áp dụng cho block-position ablation trước đây: kết quả n=1
# KHÔNG đủ để phân biệt "hiệu ứng thật" (đảo ngược no-SR > span_tiny >
# span_baseline do resolution mismatch, như bài báo đang giải thích) với
# "nhiễu của riêng 1 seed checkpoint". Script này chạy lại đúng 3 điều kiện
# (no_sr, sr_baseline, sr_improved) trên real_lr_holdout.json, CHO TỪNG SEED
# recognition đã có sẵn từ pipeline/run_multi_seed.sh (KHÔNG train lại gì cả
# — chỉ eval lại bằng checkpoint recognition seed-khớp).
#
# YÊU CẦU: đã chạy xong pipeline/run_multi_seed.sh (checkpoint
# recognition_{lr,sr_baseline,sr_improved}_mobilenet_v2_seed{SEED}) cho các
# seed dưới đây, và có sẵn checkpoint SR cố định seed=42
# (runs/sr_${SR_ARCH}/best.pt, runs/sr_improved_${STUDENT_ARCH}/best.pt).

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results/real_lr_holdout_multiseed"
BACKBONE="mobilenet_v2"
SEEDS=(42 123 2024 44 999)
# [MỚI — 2026-08-30, sự cố thật] xem giải thích đầy đủ trong
# pipeline/run_block_position_ablation.sh — giữ NGUYÊN cùng cơ chế ở đây để
# nhất quán nếu server vẫn đang bị chia sẻ nặng lúc chạy.
NUM_WORKERS="${NUM_WORKERS:-4}"
mkdir -p "$RESULTS_DIR"

SR_ARCH=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['sr']['arch'])")
STUDENT_ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")
SR_BASELINE_CKPT="runs/sr_${SR_ARCH}/best.pt"
SR_IMPROVED_CKPT="runs/sr_improved_${STUDENT_ARCH}/best.pt"

echo "Kiểm tra tiền đề..."
MISSING=0
if [ ! -f "$SR_BASELINE_CKPT" ]; then
    echo "LỖI: thiếu $SR_BASELINE_CKPT" >&2; MISSING=1
fi
if [ ! -f "$SR_IMPROVED_CKPT" ]; then
    echo "LỖI: thiếu $SR_IMPROVED_CKPT" >&2; MISSING=1
fi
for SEED in "${SEEDS[@]}"; do
    for DOMAIN in lr sr_baseline sr_improved; do
        CKPT="runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: thiếu $CKPT -> chạy pipeline/run_multi_seed.sh (và _extra_seeds.sh cho seed 44/999) trước." >&2
            MISSING=1
        fi
    done
done
if [ "$MISSING" -eq 1 ]; then
    echo "DỪNG LẠI — thiếu tiền đề ở trên." >&2
    exit 1
fi
echo "OK — đủ tiền đề, bắt đầu chạy (chỉ eval, không train lại)."

for SEED in "${SEEDS[@]}"; do
    echo "----------------------------------------------------------------"
    echo "seed=$SEED"
    echo "----------------------------------------------------------------"
    python eval_real_lr_holdout.py --config "$CONFIG" --backbone "$BACKBONE" \
        --sr_baseline_ckpt "$SR_BASELINE_CKPT" --sr_baseline_arch "$SR_ARCH" \
        --sr_improved_ckpt "$SR_IMPROVED_CKPT" --sr_improved_arch "$STUDENT_ARCH" \
        --run_suffix "_seed${SEED}" --seed_label "$SEED" --num_workers "$NUM_WORKERS" \
        --out_csv "$RESULTS_DIR/real_lr_holdout_seed${SEED}.csv"
done

echo ""
echo ">>> Tổng hợp multi-seed (paired t-test, Cohen's d, Wilcoxon, CI95%, MDES)..."
python data/aggregate_real_lr_holdout_multiseed.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/real_lr_holdout_multiseed_summary.csv"

echo ""
echo "HOÀN TẤT. Xem $RESULTS_DIR/real_lr_holdout_multiseed_summary.csv và"
echo "$RESULTS_DIR/real_lr_holdout_multiseed_summary_pairwise.csv để biết đảo ngược"
echo "no-SR > span_tiny > span_baseline có LẶP LẠI qua nhiều seed hay chỉ là nhiễu n=1."
