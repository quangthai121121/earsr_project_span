#!/bin/bash
# [MỚI — mở rộng 7 backbone mới cho KD v2 ablation (tab:kdv2-ablation), xem
# models/recognition_model.py] pipeline/run_multi_seed_kdv2.sh + _extra_seeds.sh
# đã kiểm chứng trên 5 backbone gốc (kết luận: không cải thiện đáng kể so với
# recipe chính). Đây là 1 trong 3 "negative-result ablation" của bài báo, đủ
# 5 backbone như bảng kết quả chính, nên mở rộng cùng lúc để nhất quán.
#
# KHÔNG train lại SR (checkpoint runs/sr_improved_<student_arch>_kdv2/best.pt
# và ảnh splits/sr_improved_kdv2/ không phụ thuộc backbone, đã có sẵn từ
# pipeline/run_multi_seed_kdv2.sh chạy 1 lần) — CHỈ thêm multi-seed
# recognition cho 7 backbone MỚI, thẳng n=5 seed.
#
# DÙNG:
#   bash pipeline/run_multi_seed_kdv2_new_backbones.sh [đường_dẫn_config]
#
# TIỀN ĐỀ:
#   - Đã chạy xong pipeline/run_multi_seed_kdv2.sh (ảnh splits/sr_improved_kdv2/
#     đã có, không phụ thuộc backbone).
#   - Đã chạy xong pipeline/run_multi_seed_new_backbones.sh (checkpoint
#     recognition_lr_<backbone mới>_seed<seed>/best.pt, n=5, cho cả 7
#     backbone mới).
#
# CẢNH BÁO THỜI GIAN: 7 backbone x 5 seed = 35 lần train+eval recognition
# (không train lại SR).

set -e

CONFIG="${1:-configs/config.yaml}"
RESULTS_DIR="results/multi_seed_kdv2"
NEW_BACKBONES=("shufflenet_v2_x1_0" "squeezenet1_1" "mnasnet1_0" "mobilenet_v3_large" "regnet_y_400mf" "mobileone_s0" "lcnet_100")
SEEDS=(42 123 2024 44 999)
DOMAIN="sr_improved_kdv2"
mkdir -p "$RESULTS_DIR"

echo "Kiểm tra tiền đề..."
MISSING=0
if [ ! -d "splits/${DOMAIN}" ] || [ -z "$(find "splits/${DOMAIN}" -type f -print -quit 2>/dev/null)" ]; then
    echo "LỖI: splits/${DOMAIN} không tồn tại hoặc rỗng." >&2
    echo "     -> chạy pipeline/run_multi_seed_kdv2.sh trước (sinh SR)." >&2
    MISSING=1
fi
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

for BACKBONE in "${NEW_BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUT_JSON="$RESULTS_DIR/${DOMAIN}_${BACKBONE}_seed${SEED}.json"
        if [ -f "$OUT_JSON" ]; then
            echo ">>> Bỏ qua backbone=$BACKBONE seed=$SEED (đã có JSON)"
            continue
        fi

        echo "----------------------------------------------------------------"
        echo "backbone=$BACKBONE | seed=$SEED | domain=$DOMAIN"
        echo "----------------------------------------------------------------"
        LR_CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" --backbone "$BACKBONE" \
            --init_ckpt "$LR_CKPT" --seed "$SEED" --run_suffix "_seed${SEED}"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
            --out_json "$OUT_JSON"
    done
done

echo ""
echo ">>> Tổng hợp lại (giờ đủ 12 backbone: 5 gốc + 7 mới)..."
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/multi_seed_kdv2_summary.csv"

echo ""
echo "HOÀN TẤT. Kết quả: $RESULTS_DIR/multi_seed_kdv2_summary.csv"
echo "và $RESULTS_DIR/multi_seed_kdv2_summary_pairwise.csv."
