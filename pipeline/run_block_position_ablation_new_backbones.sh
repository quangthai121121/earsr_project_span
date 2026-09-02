#!/bin/bash
# [MỚI — mở rộng 7 backbone mới cho block-position ablation, xem
# models/recognition_model.py] pipeline/run_block_position_ablation.sh (n=3)
# + _extra_seeds.sh (n=5) đã kiểm chứng trên 5 backbone gốc: không có hiệu
# ứng đáng kể của vị trí khối giữ lại. Mở rộng thêm 7 backbone MỚI để biết
# kết luận "vị trí khối không ảnh hưởng" có giữ nguyên trên kiến trúc khác
# họ hay không.
#
# KHÔNG train lại SR — TÁI SỬ DỤNG 3 checkpoint
# runs/sr_improved_<student_arch>_position_<variant>/best.pt và ảnh đã sinh
# sẵn ở splits/sr_position_<variant>/ (SR_SEED=42 cố định, đã có từ
# pipeline/run_block_position_ablation.sh) — CHỈ train+eval recognition cho
# 7 backbone mới, thẳng n=5 seed (không qua bước n=3 trung gian vì đây là
# backbone hoàn toàn mới, không có lịch sử n=3 cần nối tiếp).
#
# Ghi kết quả vào CÙNG results/block_position_screen/ như 2 script gốc --
# tổng hợp lại sẽ tự ra bảng đủ 12 backbone.
#
# DÙNG:
#   bash pipeline/run_block_position_ablation_new_backbones.sh [đường_dẫn_config]
#
# TIỀN ĐỀ:
#   1. Đã chạy xong pipeline/run_block_position_ablation.sh (ảnh SR ở
#      splits/sr_position_{keepfirst,keeplast,interleaved}/, không phụ thuộc
#      backbone).
#   2. Đã chạy xong pipeline/run_multi_seed_new_backbones.sh (checkpoint
#      recognition_lr_<backbone mới>_seed<seed>/best.pt, n=5, cho cả 7
#      backbone mới).
#
# CẢNH BÁO THỜI GIAN: 3 domain x 7 backbone x 5 seed = 105 lần train
# recognition (nhẹ, không train lại SR).

set -e

CONFIG="${1:-configs/config.yaml}"
RESULTS_DIR="results/block_position_screen"
NEW_BACKBONES=("shufflenet_v2_x1_0" "squeezenet1_1" "mnasnet1_0" "mobilenet_v3_large" "regnet_y_400mf" "mobileone_s0" "lcnet_100")
SEEDS=(42 123 2024 44 999)
DOMAINS=("sr_position_keepfirst" "sr_position_keeplast" "sr_position_interleaved")
NUM_WORKERS="${NUM_WORKERS:-4}"
mkdir -p "$RESULTS_DIR"

echo "Kiểm tra tiền đề..."
MISSING=0
for DOMAIN in "${DOMAINS[@]}"; do
    if [ ! -d "splits/${DOMAIN}" ] || [ -z "$(find "splits/${DOMAIN}" -type f -print -quit 2>/dev/null)" ]; then
        echo "LỖI: splits/${DOMAIN} không tồn tại hoặc rỗng." >&2
        echo "     -> chạy pipeline/run_block_position_ablation.sh trước (Bước 1, sinh SR)." >&2
        MISSING=1
    fi
done
for BACKBONE in "${NEW_BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: thiếu $CKPT -> chạy pipeline/run_multi_seed_new_backbones.sh trước." >&2
            MISSING=1
        fi
    done
done
if [ "$MISSING" -eq 1 ]; then
    echo "DỪNG LẠI — thiếu tiền đề ở trên." >&2
    exit 1
fi
echo "OK — đủ tiền đề, bắt đầu chạy (không train lại SR)."

for DOMAIN in "${DOMAINS[@]}"; do
    for BACKBONE in "${NEW_BACKBONES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            OUT_JSON="$RESULTS_DIR/${DOMAIN}_${BACKBONE}_seed${SEED}.json"
            if [ -f "$OUT_JSON" ]; then
                echo ">>> Bỏ qua $DOMAIN backbone=$BACKBONE seed=$SEED (đã có JSON)"
                continue
            fi

            echo "----------------------------------------------------------------"
            echo "domain=$DOMAIN | backbone=$BACKBONE | seed=$SEED"
            echo "----------------------------------------------------------------"
            LR_CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
            python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" --backbone "$BACKBONE" \
                --init_ckpt "$LR_CKPT" --seed "$SEED" --run_suffix "_seed${SEED}" \
                --num_workers "$NUM_WORKERS"

            python eval_recognition.py --config "$CONFIG" \
                --ckpt "runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt" \
                --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
                --out_json "$OUT_JSON" --num_workers "$NUM_WORKERS"
        done
    done
done

echo ""
echo ">>> Tổng hợp lại (giờ đủ 12 backbone: 5 gốc + 7 mới)..."
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/block_position_screen_summary.csv"

echo ""
echo "HOÀN TẤT. Kết quả:"
echo "$RESULTS_DIR/block_position_screen_summary.csv và"
echo "$RESULTS_DIR/block_position_screen_summary_pairwise.csv."
