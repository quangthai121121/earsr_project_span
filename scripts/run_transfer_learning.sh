#!/bin/bash
# [MỚI] Chạy TOÀN BỘ thí nghiệm TRANSFER LEARNING: khởi tạo từ checkpoint đã
# train trên dataset NGUỒN (mặc định EarVN1.0, configs/config.yaml) rồi
# fine-tune sang dataset ĐÍCH (ví dụ AWE) — bổ sung/so sánh với protocol
# "train from scratch" đã chạy trước đó cho dataset đích.
#
# 4 bước:
#   1. Zero-shot SR   : áp thẳng checkpoint span_tiny (nguồn) lên ảnh test đích,
#                       KHÔNG train lại — đo mức độ tổng quát hoá "trần".
#   2. Fine-tune SR   : train tiếp span_tiny (khởi tạo từ checkpoint nguồn) trên
#                       tập train ảnh đích (LR/epoch giảm so với from-scratch).
#   3. Fine-tune nhận diện: 3 domain (lr, sr_baseline, sr_improved) x 5 backbone
#                       x 3 seed, MỖI seed khởi tạo từ checkpoint CÙNG DOMAIN
#                       CÙNG SEED của dataset nguồn (so sánh công bằng, ghép cặp
#                       đúng seed với kết quả from-scratch multi_seed_summary.csv
#                       đã có) — dùng --init_ckpt_transfer (chỉ nạp tensor cùng
#                       shape, tự bỏ qua/khởi tạo lại identity_head vì số người
#                       khác nhau giữa 2 dataset).
#   4. Tổng hợp       : gộp kết quả đa seed như run_multi_seed.sh.
#
# QUY ƯỚC ĐẶT TÊN (để không lẫn với kết quả from-scratch đã có):
#   - checkpoint/run: hậu tố "_finetuned_from_<ten_dataset_nguon>"
#     ví dụ: runs_awe/sr_improved_span_tiny_finetuned_from_earvn1/best.pt
#            runs_awe/recognition_sr_improved_mobilenet_v2_finetuned_from_earvn1_seed42/best.pt
#   - kết quả SR      : results_<đích>/sr_quality_transfer.csv (RIÊNG, không đè
#                       results_<đích>/sr_quality.csv của from-scratch)
#   - kết quả nhận diện: results_<đích>/multi_seed_transfer/ (RIÊNG thư mục, không
#                       đè results_<đích>/multi_seed/ của from-scratch)
#
# LƯU Ý PHẠM VI: chỉ áp dụng fine-tune cho 3 domain lr/sr_baseline/sr_improved
# (không làm domain hr — hr chỉ là mốc tham chiếu trên, không phải trọng tâm
# nghiên cứu). Nếu cần domain hr, chạy thủ công theo mẫu bước 3 bên dưới.
#
# Dùng:
#   bash scripts/run_transfer_learning.sh <ten_dataset_dich> <ten_de_dat_cho_dataset_nguon>
# Ví dụ:
#   bash scripts/run_transfer_learning.sh awe earvn1
#
# YÊU CẦU TRƯỚC KHI CHẠY:
#   - Dataset nguồn (configs/config.yaml, LUÔN LUÔN là dataset nguồn — không đổi
#     được qua tham số) đã chạy xong pipeline chính + run_multi_seed.sh:
#       runs/sr_improved_span_tiny/best.pt
#       runs/recognition_<domain>_<backbone>_seed<seed>/best.pt
#       (domain = lr/sr_baseline/sr_improved, backbone = 5 cái, seed = 42/123/2024)
#   - Dataset đích đã chạy xong pipeline from-scratch (RUN_ALL_NEW_DATASET.sh)
#     trước đó, ít nhất tới bước có: runs_<đích>/sr_span_official/best.pt (teacher)

set -e

TARGET="$1"
SOURCE_LABEL="$2"

if [ -z "$TARGET" ] || [ -z "$SOURCE_LABEL" ]; then
    echo "Dùng: bash scripts/run_transfer_learning.sh <ten_dataset_dich> <ten_de_dat_cho_dataset_nguon>"
    echo "Ví dụ: bash scripts/run_transfer_learning.sh awe earvn1"
    exit 1
fi

SRC_CONFIG="configs/config.yaml"
TGT_CONFIG="configs/config_${TARGET}.yaml"
TGT_FT_CONFIG="configs/config_${TARGET}_finetune.yaml"
RESULTS_DIR="results_${TARGET}"
RUNS_DIR="runs_${TARGET}"
SUFFIX="_finetuned_from_${SOURCE_LABEL}"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(42 123 2024)
DOMAINS=("lr" "sr_baseline" "sr_improved")

echo "################################################################"
echo "# KIỂM TRA TIỀN ĐỀ"
echo "################################################################"
for f in "$SRC_CONFIG" "$TGT_CONFIG"; do
    if [ ! -f "$f" ]; then echo "LỖI: thiếu $f"; exit 1; fi
done
if [ ! -f "runs/sr_improved_span_tiny/best.pt" ]; then
    echo "LỖI: thiếu checkpoint nguồn runs/sr_improved_span_tiny/best.pt"
    echo "     -> chạy xong pipeline chính (Kịch bản A) cho dataset nguồn trước."
    exit 1
fi
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            CKPT="runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt"
            if [ ! -f "$CKPT" ]; then
                echo "LỖI: thiếu checkpoint nguồn $CKPT"
                echo "     -> chạy xong pipeline/run_multi_seed.sh cho dataset nguồn trước."
                exit 1
            fi
        done
    done
done
if [ ! -f "${RUNS_DIR}/sr_span_official/best.pt" ]; then
    echo "LỖI: thiếu ${RUNS_DIR}/sr_span_official/best.pt (teacher của dataset đích)"
    echo "     -> chạy xong bước 04 của pipeline_${TARGET} (RUN_ALL_NEW_DATASET.sh) trước."
    exit 1
fi
# [BỔ SUNG SAU KHI RÀ SOÁT LẠI] train_sr_distill.py LUÔN nạp frozen_recognition_ckpt
# để tính loss_identity mỗi batch, kể cả khi lambda_identity=0.0 (không ảnh hưởng
# gradient nhưng vẫn bắt buộc load+forward được checkpoint) — thiếu file này sẽ
# crash giữa chừng bước 2/4. Kiểm tra trước, không để crash giữa chừng.
if [ ! -f "${RUNS_DIR}/recognition_hr_mobilenet_v2/best.pt" ]; then
    echo "LỖI: thiếu ${RUNS_DIR}/recognition_hr_mobilenet_v2/best.pt"
    echo "     (frozen_recognition_ckpt trong config_${TARGET}.yaml — cần cho bước fine-tune SR,"
    echo "     dù lambda_identity=0.0 thì code vẫn bắt buộc load+forward qua checkpoint này)"
    echo "     -> chạy xong bước 03 của pipeline_${TARGET} (RUN_ALL_NEW_DATASET.sh) trước."
    exit 1
fi
if [ ! -f "splits_${TARGET}/splits.json" ]; then
    echo "LỖI: thiếu splits_${TARGET}/splits.json — dữ liệu dataset đích chưa được chuẩn bị."
    echo "     -> chạy xong bước 01 của pipeline_${TARGET} (RUN_ALL_NEW_DATASET.sh) trước."
    exit 1
fi
echo "OK: đủ tiền đề, bắt đầu chạy."

mkdir -p "$RESULTS_DIR" "${RESULTS_DIR}/multi_seed_transfer"

echo ""
echo "################################################################"
echo "# BƯỚC 0/4 — Tạo config fine-tune: $TGT_FT_CONFIG"
echo "################################################################"
python scripts/make_finetune_config.py --in_config "$TGT_CONFIG" --out_config "$TGT_FT_CONFIG"

echo ""
echo "################################################################"
echo "# BƯỚC 1/4 — Zero-shot SR: span_tiny (nguồn: $SOURCE_LABEL) áp thẳng lên $TARGET"
echo "################################################################"
python eval_sr_quality.py --config "$TGT_CONFIG" --arch span_tiny \
    --ckpt "runs/sr_improved_span_tiny/best.pt" \
    --label "span_tiny_zeroshot_from_${SOURCE_LABEL}" \
    --out_csv "${RESULTS_DIR}/sr_quality_transfer.csv"

echo ""
echo "################################################################"
echo "# BƯỚC 2/4 — Fine-tune SR (span_tiny) trên $TARGET, khởi tạo từ checkpoint nguồn"
echo "################################################################"
python train_sr_distill.py --config "$TGT_FT_CONFIG" \
    --init_ckpt "runs/sr_improved_span_tiny/best.pt" \
    --run_suffix "$SUFFIX"

python eval_sr_quality.py --config "$TGT_CONFIG" --arch span_tiny \
    --ckpt "${RUNS_DIR}/sr_improved_span_tiny${SUFFIX}/best.pt" \
    --label "span_tiny_finetuned_from_${SOURCE_LABEL}" \
    --out_csv "${RESULTS_DIR}/sr_quality_transfer.csv"

echo ""
echo "################################################################"
echo "# BƯỚC 3/4 — Fine-tune nhận diện: 3 domain x 5 backbone x 3 seed trên $TARGET"
echo "################################################################"
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            echo "---- backbone=$BACKBONE seed=$SEED domain=$DOMAIN ----"
            SRC_CKPT="runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt"

            python train_recognition.py --config "$TGT_FT_CONFIG" --domain "$DOMAIN" --backbone "$BACKBONE" \
                --init_ckpt_transfer "$SRC_CKPT" \
                --seed "$SEED" --run_suffix "${SUFFIX}_seed${SEED}"

            python eval_recognition.py --config "$TGT_CONFIG" \
                --ckpt "${RUNS_DIR}/recognition_${DOMAIN}_${BACKBONE}${SUFFIX}_seed${SEED}/best.pt" \
                --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
                --out_json "${RESULTS_DIR}/multi_seed_transfer/${DOMAIN}_${BACKBONE}_seed${SEED}.json"
        done
    done
done

echo ""
echo "################################################################"
echo "# BƯỚC 4/4 — Tổng hợp kết quả fine-tune đa seed"
echo "################################################################"
python data/aggregate_multi_seed_results.py --results_dir "${RESULTS_DIR}/multi_seed_transfer" \
    --out_csv "${RESULTS_DIR}/multi_seed_transfer/multi_seed_summary_transfer.csv"

echo ""
echo "################################################################"
echo "HOÀN TẤT transfer learning cho '$TARGET' (nguồn: '$SOURCE_LABEL')."
echo "################################################################"
echo "So sánh trực tiếp với kết quả from-scratch đã có trước đó:"
echo "  SR quality      : ${RESULTS_DIR}/sr_quality.csv (from-scratch)"
echo "                    vs ${RESULTS_DIR}/sr_quality_transfer.csv (zero-shot + fine-tuned)"
echo "  Recognition acc : ${RESULTS_DIR}/multi_seed/multi_seed_summary.csv (from-scratch)"
echo "                    vs ${RESULTS_DIR}/multi_seed_transfer/multi_seed_summary_transfer.csv (fine-tuned)"
