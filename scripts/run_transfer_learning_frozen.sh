#!/bin/bash
# [MỚI] Biến thể freeze-backbone của scripts/run_transfer_learning.sh — CHỈ chạy lại
# bước 6 (fine-tune nhận diện) với --freeze_backbone, TÁI SỬ DỤNG checkpoint SR đã
# fine-tune và ảnh sr_baseline_transfer/sr_improved_transfer đã sinh từ
# run_transfer_learning.sh (bước 1-5) — KHÔNG train lại SR, KHÔNG sinh lại ảnh.
#
# Mục đích: dataset đích quá ít dữ liệu (ví dụ AWE, ~5.7 ảnh train/người) khiến
# fine-tune TOÀN BỘ recognition model có phương sai rất lớn giữa các seed (xem
# results_awe/multi_seed_transfer/*_pairwise.csv — mọi p_bonferroni đều >=0.13).
# Đóng băng backbone (đã học đặc trưng tốt từ dataset NGUỒN qua init_ckpt_transfer),
# chỉ train embedding+heads — giảm số tham số học từ ít dữ liệu, kỳ vọng giảm nhiễu.
#
# GIỮ ĐÚNG PROTOCOL: không đổi ngưỡng lọc dữ liệu, không đổi split, không đổi seed,
# không đổi domain nào so với run_transfer_learning.sh — CHỈ thêm 1 cờ --freeze_backbone
# vào đúng bước fine-tune nhận diện. Ghi kết quả vào thư mục RIÊNG
# (multi_seed_transfer_frozen/) — không đè kết quả transfer thường, để so sánh
# frozen vs không-frozen minh bạch.
#
# YÊU CẦU TRƯỚC KHI CHẠY: đã chạy XONG scripts/run_transfer_learning.sh cho cùng
# TARGET/SOURCE_LABEL (cần checkpoint SR đã fine-tune + ảnh *_transfer đã sinh).
#
# Dùng:
#   bash scripts/run_transfer_learning_frozen.sh <ten_dataset_dich> <ten_de_dat_cho_dataset_nguon>
# Ví dụ:
#   bash scripts/run_transfer_learning_frozen.sh awe earvn1

set -e

TARGET="$1"
SOURCE_LABEL="$2"

if [ -z "$TARGET" ] || [ -z "$SOURCE_LABEL" ]; then
    echo "Dùng: bash scripts/run_transfer_learning_frozen.sh <ten_dataset_dich> <ten_de_dat_cho_dataset_nguon>"
    echo "Ví dụ: bash scripts/run_transfer_learning_frozen.sh awe earvn1"
    exit 1
fi

TGT_CONFIG="configs/config_${TARGET}.yaml"
TGT_FT_CONFIG="configs/config_${TARGET}_finetune.yaml"
RESULTS_DIR="results_${TARGET}"
RUNS_DIR="runs_${TARGET}"
SPLITS_DIR="splits_${TARGET}"
SUFFIX="_finetuned_from_${SOURCE_LABEL}"
OUT_SUBDIR="${RESULTS_DIR}/multi_seed_transfer_frozen"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(42 123 2024 44 999)

echo "################################################################"
echo "# KIỂM TRA TIỀN ĐỀ (tái sử dụng từ run_transfer_learning.sh)"
echo "################################################################"
for f in "$TGT_CONFIG" "$TGT_FT_CONFIG"; do
    if [ ! -f "$f" ]; then
        echo "LỖI: thiếu $f — chạy scripts/run_transfer_learning.sh trước (script này chỉ chạy lại bước 6, không tạo config)."
        exit 1
    fi
done
if [ ! -d "${SPLITS_DIR}/sr_baseline_transfer" ] || [ ! -d "${SPLITS_DIR}/sr_improved_transfer" ]; then
    echo "LỖI: thiếu ${SPLITS_DIR}/sr_baseline_transfer hoặc sr_improved_transfer —"
    echo "     chạy xong scripts/run_transfer_learning.sh (bước 1-5) trước."
    exit 1
fi
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DOMAIN in lr sr_baseline sr_improved; do
            CKPT="runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt"
            if [ ! -f "$CKPT" ]; then
                echo "LỖI: thiếu checkpoint nguồn $CKPT (giống yêu cầu của run_transfer_learning.sh)."
                exit 1
            fi
        done
    done
done
echo "OK: đủ tiền đề, bắt đầu chạy."

mkdir -p "$OUT_SUBDIR"

echo ""
echo "################################################################"
echo "# Fine-tune nhận diện (FREEZE BACKBONE): 3 domain x 5 backbone x 5 seed"
echo "################################################################"
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        # Cặp giống hệt run_transfer_learning.sh bước 6:
        #   domain đích mới       <- domain nguồn để lấy checkpoint init
        #   lr                    <- lr
        #   sr_baseline_transfer  <- sr_baseline
        #   sr_improved_transfer  <- sr_improved
        for PAIR in "lr:lr" "sr_baseline_transfer:sr_baseline" "sr_improved_transfer:sr_improved"; do
            TGT_DOMAIN="${PAIR%%:*}"
            SRC_DOMAIN="${PAIR##*:}"
            echo "---- backbone=$BACKBONE seed=$SEED domain_đích=$TGT_DOMAIN (nguồn: $SRC_DOMAIN) [FROZEN] ----"
            SRC_CKPT="runs/recognition_${SRC_DOMAIN}_${BACKBONE}_seed${SEED}/best.pt"

            python train_recognition.py --config "$TGT_FT_CONFIG" --domain "$TGT_DOMAIN" --backbone "$BACKBONE" \
                --init_ckpt_transfer "$SRC_CKPT" --freeze_backbone \
                --seed "$SEED" --run_suffix "${SUFFIX}_frozen_seed${SEED}"

            python eval_recognition.py --config "$TGT_CONFIG" \
                --ckpt "${RUNS_DIR}/recognition_${TGT_DOMAIN}_${BACKBONE}${SUFFIX}_frozen_seed${SEED}/best.pt" \
                --backbone "$BACKBONE" --train_domain "$TGT_DOMAIN" --test_domain "$TGT_DOMAIN" \
                --out_json "${OUT_SUBDIR}/${TGT_DOMAIN}_${BACKBONE}_seed${SEED}.json"
        done
    done
done

python data/aggregate_multi_seed_results.py --results_dir "$OUT_SUBDIR" \
    --out_csv "${OUT_SUBDIR}/multi_seed_summary_transfer_frozen.csv"

echo ""
echo "################################################################"
echo "HOÀN TẤT transfer learning (FREEZE BACKBONE) cho '$TARGET' (nguồn: '$SOURCE_LABEL')."
echo "################################################################"
echo "File kết quả cần gửi lại để tổng hợp:"
echo "  ${OUT_SUBDIR}/multi_seed_summary_transfer_frozen.csv"
echo "  ${OUT_SUBDIR}/multi_seed_summary_transfer_frozen_pairwise.csv"
echo ""
echo "So sánh với kết quả KHÔNG freeze (đã có sẵn):"
echo "  ${RESULTS_DIR}/multi_seed_transfer/multi_seed_summary_transfer.csv"
echo "-> Xem std_identity_accuracy và p_bonferroni giữa 2 file để biết freeze-backbone"
echo "   có thực sự giảm phương sai / tăng khả năng đạt ý nghĩa thống kê hay không."
