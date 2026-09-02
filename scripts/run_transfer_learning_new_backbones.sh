#!/bin/bash
# [MỚI — mở rộng 7 backbone mới cho AWEx transfer-learning (full fine-tune),
# xem models/recognition_model.py] scripts/run_transfer_learning.sh (5
# backbone gốc) đã hoàn tất BƯỚC 1-5 (zero-shot SR, fine-tune SR tiny +
# baseline trên dataset đích, sinh lại ảnh sr_baseline_transfer/
# sr_improved_transfer) — các bước này KHÔNG phụ thuộc backbone recognition,
# nên KHÔNG lặp lại. Script này CHỈ lặp lại đúng BƯỚC 6/6 (fine-tune nhận
# diện) cho 7 backbone MỚI, tái sử dụng nguyên vẹn 2 checkpoint SR đã
# fine-tune + 2 domain ảnh đã sinh sẵn.
#
# Ghi kết quả vào CÙNG results_<đích>/multi_seed_transfer/ như script gốc --
# tổng hợp lại sẽ tự ra bảng đủ 12 backbone.
#
# DÙNG:
#   bash scripts/run_transfer_learning_new_backbones.sh <ten_dataset_dich> <ten_de_dat_cho_dataset_nguon>
# Ví dụ:
#   bash scripts/run_transfer_learning_new_backbones.sh awex earvn1
#
# TIỀN ĐỀ:
#   1. Đã chạy xong scripts/run_transfer_learning.sh cho CÙNG target/source
#      (có sẵn ${RUNS_DIR}/sr_improved_span_tiny${SUFFIX}/best.pt,
#      ${RUNS_DIR}/sr_span_official${SUFFIX}/best.pt,
#      ${SPLITS_DIR}/sr_improved_transfer/, ${SPLITS_DIR}/sr_baseline_transfer/,
#      và $TGT_FT_CONFIG).
#   2. Đã chạy xong pipeline/run_multi_seed_new_backbones.sh cho dataset
#      NGUỒN (EarVN1.0) — cần checkpoint
#      runs/recognition_{lr,sr_baseline,sr_improved}_<backbone mới>_seed<seed>/best.pt
#      làm nguồn cho --init_ckpt_transfer.
#
# CẢNH BÁO THỜI GIAN: 3 domain x 7 backbone x 5 seed = 105 lần train+eval
# recognition (không train lại SR).

set -e

TARGET="$1"
SOURCE_LABEL="$2"

if [ -z "$TARGET" ] || [ -z "$SOURCE_LABEL" ]; then
    echo "Dùng: bash scripts/run_transfer_learning_new_backbones.sh <ten_dataset_dich> <ten_de_dat_cho_dataset_nguon>"
    echo "Ví dụ: bash scripts/run_transfer_learning_new_backbones.sh awex earvn1"
    exit 1
fi

TGT_CONFIG="configs/config_${TARGET}.yaml"
TGT_FT_CONFIG="configs/config_${TARGET}_finetune.yaml"
RESULTS_DIR="results_${TARGET}"
RUNS_DIR="runs_${TARGET}"
SUFFIX="_finetuned_from_${SOURCE_LABEL}"
NEW_BACKBONES=("shufflenet_v2_x1_0" "squeezenet1_1" "mnasnet1_0" "mobilenet_v3_large" "regnet_y_400mf" "mobileone_s0" "lcnet_100")
SEEDS=(42 123 2024 44 999)
NUM_WORKERS="${NUM_WORKERS:-4}"

echo "################################################################"
echo "# KIỂM TRA TIỀN ĐỀ"
echo "################################################################"
MISSING=0
for f in "$TGT_CONFIG" "$TGT_FT_CONFIG"; do
    if [ ! -f "$f" ]; then
        echo "LỖI: thiếu $f -> chạy scripts/run_transfer_learning.sh trước." >&2
        MISSING=1
    fi
done
if [ ! -f "${RUNS_DIR}/sr_improved_span_tiny${SUFFIX}/best.pt" ]; then
    echo "LỖI: thiếu ${RUNS_DIR}/sr_improved_span_tiny${SUFFIX}/best.pt" >&2
    echo "     -> chạy scripts/run_transfer_learning.sh $TARGET $SOURCE_LABEL trước." >&2
    MISSING=1
fi
if [ ! -f "${RUNS_DIR}/sr_span_official${SUFFIX}/best.pt" ]; then
    echo "LỖI: thiếu ${RUNS_DIR}/sr_span_official${SUFFIX}/best.pt" >&2
    MISSING=1
fi
SPLITS_DIR="splits_${TARGET}"
if [ ! -d "${SPLITS_DIR}/sr_improved_transfer" ] || [ ! -d "${SPLITS_DIR}/sr_baseline_transfer" ]; then
    echo "LỖI: thiếu ${SPLITS_DIR}/sr_improved_transfer/ hoặc sr_baseline_transfer/" >&2
    echo "     -> chạy scripts/run_transfer_learning.sh $TARGET $SOURCE_LABEL trước." >&2
    MISSING=1
fi
for BACKBONE in "${NEW_BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for SRC_DOMAIN in lr sr_baseline sr_improved; do
            CKPT="runs/recognition_${SRC_DOMAIN}_${BACKBONE}_seed${SEED}/best.pt"
            if [ ! -f "$CKPT" ]; then
                echo "LỖI: thiếu checkpoint nguồn $CKPT" >&2
                echo "     -> chạy pipeline/run_multi_seed_new_backbones.sh cho dataset nguồn trước." >&2
                MISSING=1
            fi
        done
    done
done
if [ "$MISSING" -eq 1 ]; then
    echo "DỪNG LẠI — thiếu tiền đề ở trên." >&2
    exit 1
fi
echo "OK — đủ tiền đề, bắt đầu chạy (không train lại SR, không sinh lại ảnh)."

mkdir -p "${RESULTS_DIR}/multi_seed_transfer"

for BACKBONE in "${NEW_BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for PAIR in "lr:lr" "sr_baseline_transfer:sr_baseline" "sr_improved_transfer:sr_improved"; do
            TGT_DOMAIN="${PAIR%%:*}"
            SRC_DOMAIN="${PAIR##*:}"

            OUT_JSON="${RESULTS_DIR}/multi_seed_transfer/${TGT_DOMAIN}_${BACKBONE}_seed${SEED}.json"
            if [ -f "$OUT_JSON" ]; then
                echo ">>> Bỏ qua backbone=$BACKBONE seed=$SEED domain_đích=$TGT_DOMAIN (đã có JSON)"
                continue
            fi

            echo "---- backbone=$BACKBONE seed=$SEED domain_đích=$TGT_DOMAIN (nguồn: $SRC_DOMAIN) ----"
            SRC_CKPT="runs/recognition_${SRC_DOMAIN}_${BACKBONE}_seed${SEED}/best.pt"

            python train_recognition.py --config "$TGT_FT_CONFIG" --domain "$TGT_DOMAIN" --backbone "$BACKBONE" \
                --init_ckpt_transfer "$SRC_CKPT" \
                --seed "$SEED" --run_suffix "${SUFFIX}_seed${SEED}" \
                --num_workers "$NUM_WORKERS"

            python eval_recognition.py --config "$TGT_CONFIG" \
                --ckpt "${RUNS_DIR}/recognition_${TGT_DOMAIN}_${BACKBONE}${SUFFIX}_seed${SEED}/best.pt" \
                --backbone "$BACKBONE" --train_domain "$TGT_DOMAIN" --test_domain "$TGT_DOMAIN" \
                --out_json "$OUT_JSON" --num_workers "$NUM_WORKERS"
        done
    done
done

echo ""
echo ">>> Tổng hợp lại (giờ đủ 12 backbone: 5 gốc + 7 mới)..."
python data/aggregate_multi_seed_results.py --results_dir "${RESULTS_DIR}/multi_seed_transfer" \
    --out_csv "${RESULTS_DIR}/multi_seed_transfer/multi_seed_summary_transfer.csv"

echo ""
echo "HOÀN TẤT transfer learning cho 7 backbone mới ('$TARGET', nguồn: '$SOURCE_LABEL')."
echo "Kết quả: ${RESULTS_DIR}/multi_seed_transfer/multi_seed_summary_transfer.csv"
echo "và ${RESULTS_DIR}/multi_seed_transfer/multi_seed_summary_transfer_pairwise.csv"
