#!/bin/bash
# Train lại ĐÚNG phần span_tiny transfer-learning fine-tune cho AWEx, sau khi
# đã fix teacher (span_official). KHÔNG đụng span_baseline_transfer (đang tốt).
#
# Nguyên nhân: configs/config_awex_finetune.yaml giữ nguyên teacher_ckpt trỏ
# vào runs_awex/sr_span_official/best.pt — lần fine-tune trước (12:39) chạy
# TRƯỚC khi span_official được train lại sạch (16:58), nên học theo teacher
# hỏng. Giờ teacher đã sạch, chỉ cần train lại đúng bước này.
#
# CHẠY TỪ THƯ MỤC GỐC PROJECT.

set -e

TARGET="awex"
SOURCE_LABEL="earvn1"
TGT_CONFIG="configs/config_${TARGET}.yaml"
TGT_FT_CONFIG="configs/config_${TARGET}_finetune.yaml"
RESULTS_DIR="results_${TARGET}"
RUNS_DIR="runs_${TARGET}"
SPLITS_DIR="splits_${TARGET}"
SUFFIX="_finetuned_from_${SOURCE_LABEL}"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(42 123 2024 44 999)

echo "################################################################"
echo "# BƯỚC 0 — Kiểm tra teacher đã sạch chưa"
echo "################################################################"
if ! python -c "
import yaml
cfg = yaml.safe_load(open('$TGT_FT_CONFIG'))
print(cfg['sr_improve']['teacher_ckpt'])
" ; then
    echo "LỖI: không đọc được teacher_ckpt từ $TGT_FT_CONFIG"; exit 1
fi
echo ">>> Xác nhận teacher_ckpt ở trên trỏ đúng checkpoint span_official ĐÃ SỬA (timestamp mới)."

echo ""
echo "################################################################"
echo "# BƯỚC 1 — Dọn artifact cũ (span_tiny fine-tune học từ teacher hỏng)"
echo "################################################################"
rm -rf "${RUNS_DIR}/sr_improved_span_tiny${SUFFIX}"
rm -rf "${SPLITS_DIR}/sr_improved_transfer"
rm -rf "${RUNS_DIR}"/recognition_sr_improved_transfer_*${SUFFIX}*

echo ""
echo "################################################################"
echo "# BƯỚC 2 — Fine-tune lại span_tiny (teacher giờ đã sạch)"
echo "################################################################"
python train_sr_distill.py --config "$TGT_FT_CONFIG" \
    --init_ckpt "runs/sr_improved_span_tiny/best.pt" \
    --run_suffix "$SUFFIX"

echo ""
echo "################################################################"
echo "# BƯỚC 3 — Eval chất lượng SR + sinh lại ảnh sr_improved_transfer"
echo "################################################################"
python eval_sr_quality.py --config "$TGT_CONFIG" --arch span_tiny \
    --ckpt "${RUNS_DIR}/sr_improved_span_tiny${SUFFIX}/best.pt" \
    --label "span_tiny_finetuned_from_${SOURCE_LABEL}_v2" \
    --out_csv "${RESULTS_DIR}/sr_quality_transfer.csv"

python data/build_sr.py --lr_dir "${SPLITS_DIR}/lr" \
    --sr_ckpt "${RUNS_DIR}/sr_improved_span_tiny${SUFFIX}/best.pt" \
    --arch span_tiny --scale 4 --out_dir "${SPLITS_DIR}/sr_improved_transfer"

echo ""
echo "################################################################"
echo "# BƯỚC 4 — Recognition multi-seed lại cho domain sr_improved_transfer (n=5)"
echo "################################################################"
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        SRC_CKPT="runs/recognition_sr_improved_${BACKBONE}_seed${SEED}/best.pt"
        python train_recognition.py --config "$TGT_FT_CONFIG" --domain "sr_improved_transfer" --backbone "$BACKBONE" \
            --init_ckpt_transfer "$SRC_CKPT" \
            --seed "$SEED" --run_suffix "${SUFFIX}_seed${SEED}"
        python eval_recognition.py --config "$TGT_CONFIG" \
            --ckpt "${RUNS_DIR}/recognition_sr_improved_transfer_${BACKBONE}${SUFFIX}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "sr_improved_transfer" --test_domain "sr_improved_transfer" \
            --out_json "${RESULTS_DIR}/multi_seed_transfer/sr_improved_transfer_${BACKBONE}_seed${SEED}.json"
    done
done

python data/aggregate_multi_seed_results.py --results_dir "${RESULTS_DIR}/multi_seed_transfer" \
    --out_csv "${RESULTS_DIR}/multi_seed_transfer/multi_seed_summary_transfer.csv"

echo ""
echo "HOÀN TẤT. Gửi lại:"
echo "  ${RESULTS_DIR}/sr_quality_transfer.csv"
echo "  ${RESULTS_DIR}/multi_seed_transfer/multi_seed_summary_transfer.csv + _pairwise.csv"
