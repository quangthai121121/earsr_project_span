#!/bin/bash
# [MỚI — mở rộng 7 backbone mới cho AWEx transfer-learning (frozen backbone),
# xem models/recognition_model.py] scripts/run_transfer_learning_frozen.sh (5
# backbone gốc) đã hoàn tất — script này CHỈ lặp lại bước fine-tune nhận
# diện với --freeze_backbone cho 7 backbone MỚI, tái sử dụng nguyên vẹn 2
# checkpoint SR đã fine-tune + 2 domain ảnh đã sinh sẵn từ
# scripts/run_transfer_learning.sh (bước 1-5, không phụ thuộc backbone).
#
# Ghi kết quả vào CÙNG results_<đích>/multi_seed_transfer_frozen/ như script
# gốc -- tổng hợp lại sẽ tự ra bảng đủ 12 backbone.
#
# DÙNG:
#   bash scripts/run_transfer_learning_frozen_new_backbones.sh <ten_dataset_dich> <ten_de_dat_cho_dataset_nguon>
# Ví dụ:
#   bash scripts/run_transfer_learning_frozen_new_backbones.sh awex earvn1
#
# TIỀN ĐỀ: giống hệt scripts/run_transfer_learning_new_backbones.sh (checkpoint
# SR đã fine-tune + ảnh *_transfer đã sinh từ scripts/run_transfer_learning.sh,
# VÀ checkpoint recognition_{lr,sr_baseline,sr_improved}_<backbone mới>_seed<seed>
# của dataset NGUỒN từ pipeline/run_multi_seed_new_backbones.sh).
#
# CẢNH BÁO THỜI GIAN: 3 domain x 7 backbone x 5 seed = 105 lần train+eval
# recognition (không train lại SR).

set -e

TARGET="$1"
SOURCE_LABEL="$2"

if [ -z "$TARGET" ] || [ -z "$SOURCE_LABEL" ]; then
    echo "Dùng: bash scripts/run_transfer_learning_frozen_new_backbones.sh <ten_dataset_dich> <ten_de_dat_cho_dataset_nguon>"
    echo "Ví dụ: bash scripts/run_transfer_learning_frozen_new_backbones.sh awex earvn1"
    exit 1
fi

TGT_CONFIG="configs/config_${TARGET}.yaml"
TGT_FT_CONFIG="configs/config_${TARGET}_finetune.yaml"
RESULTS_DIR="results_${TARGET}"
RUNS_DIR="runs_${TARGET}"
SPLITS_DIR="splits_${TARGET}"
SUFFIX="_finetuned_from_${SOURCE_LABEL}"
OUT_SUBDIR="${RESULTS_DIR}/multi_seed_transfer_frozen"
NEW_BACKBONES=("shufflenet_v2_x1_0" "squeezenet1_1" "mnasnet1_0" "mobilenet_v3_large" "regnet_y_400mf" "mobileone_s0" "lcnet_100")
SEEDS=(42 123 2024 44 999)
NUM_WORKERS="${NUM_WORKERS:-4}"

echo "################################################################"
echo "# KIỂM TRA TIỀN ĐỀ"
echo "################################################################"
MISSING=0
for f in "$TGT_CONFIG" "$TGT_FT_CONFIG"; do
    if [ ! -f "$f" ]; then
        echo "LỖI: thiếu $f — chạy scripts/run_transfer_learning.sh trước." >&2
        MISSING=1
    fi
done
if [ ! -d "${SPLITS_DIR}/sr_baseline_transfer" ] || [ ! -d "${SPLITS_DIR}/sr_improved_transfer" ]; then
    echo "LỖI: thiếu ${SPLITS_DIR}/sr_baseline_transfer hoặc sr_improved_transfer —" >&2
    echo "     chạy xong scripts/run_transfer_learning.sh (bước 1-5) trước." >&2
    MISSING=1
fi
for BACKBONE in "${NEW_BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DOMAIN in lr sr_baseline sr_improved; do
            CKPT="runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt"
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
echo "OK: đủ tiền đề, bắt đầu chạy."

mkdir -p "$OUT_SUBDIR"

echo ""
echo "################################################################"
echo "# Fine-tune nhận diện (FREEZE BACKBONE): 3 domain x 7 backbone x 5 seed"
echo "################################################################"
for BACKBONE in "${NEW_BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for PAIR in "lr:lr" "sr_baseline_transfer:sr_baseline" "sr_improved_transfer:sr_improved"; do
            TGT_DOMAIN="${PAIR%%:*}"
            SRC_DOMAIN="${PAIR##*:}"

            OUT_JSON="${OUT_SUBDIR}/${TGT_DOMAIN}_${BACKBONE}_seed${SEED}.json"
            if [ -f "$OUT_JSON" ]; then
                echo ">>> Bỏ qua backbone=$BACKBONE seed=$SEED domain_đích=$TGT_DOMAIN (đã có JSON)"
                continue
            fi

            echo "---- backbone=$BACKBONE seed=$SEED domain_đích=$TGT_DOMAIN (nguồn: $SRC_DOMAIN) [FROZEN] ----"
            SRC_CKPT="runs/recognition_${SRC_DOMAIN}_${BACKBONE}_seed${SEED}/best.pt"

            python train_recognition.py --config "$TGT_FT_CONFIG" --domain "$TGT_DOMAIN" --backbone "$BACKBONE" \
                --init_ckpt_transfer "$SRC_CKPT" --freeze_backbone \
                --seed "$SEED" --run_suffix "${SUFFIX}_frozen_seed${SEED}" \
                --num_workers "$NUM_WORKERS"

            python eval_recognition.py --config "$TGT_CONFIG" \
                --ckpt "${RUNS_DIR}/recognition_${TGT_DOMAIN}_${BACKBONE}${SUFFIX}_frozen_seed${SEED}/best.pt" \
                --backbone "$BACKBONE" --train_domain "$TGT_DOMAIN" --test_domain "$TGT_DOMAIN" \
                --out_json "$OUT_JSON" --num_workers "$NUM_WORKERS"
        done
    done
done

echo ""
echo ">>> Tổng hợp lại (giờ đủ 12 backbone: 5 gốc + 7 mới)..."
python data/aggregate_multi_seed_results.py --results_dir "$OUT_SUBDIR" \
    --out_csv "${OUT_SUBDIR}/multi_seed_summary_transfer_frozen.csv"

echo ""
echo "HOÀN TẤT transfer learning FROZEN cho 7 backbone mới ('$TARGET', nguồn: '$SOURCE_LABEL')."
echo "Kết quả: ${OUT_SUBDIR}/multi_seed_summary_transfer_frozen.csv"
echo "và ${OUT_SUBDIR}/multi_seed_summary_transfer_frozen_pairwise.csv"
