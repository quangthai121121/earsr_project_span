#!/bin/bash
# [MỚI — 2026-08-29] Chạy THÊM 2 seed (44, 999) cho domain "sr_improved_saliency"
# (lambda_saliency=1.0), để cộng với 3 seed đã có (42, 123, 2024 — từ
# pipeline/run_multi_seed_saliency.sh) thành ĐỦ 5 SEED, giống hệt mẫu đã dùng
# cho pipeline/run_multi_seed_kdv2_extra_seeds.sh.
#
# KHÔNG train lại SR — TÁI SỬ DỤNG checkpoint
# runs/sr_improved_<student_arch>_saliency/best.pt và ảnh đã sinh sẵn ở
# splits/sr_improved_saliency/ (đúng seed cố định SR_SEED=42 như
# run_multi_seed_saliency.sh) — CHỈ multi-seed ở bước train_recognition.py.
#
# YÊU CẦU:
#   1. Đã chạy xong pipeline/run_multi_seed_saliency.sh (có sẵn ảnh SR ở
#      splits/sr_improved_saliency/ và 3 seed 42/123/2024 trong
#      results/multi_seed_saliency/).
#   2. Đã chạy xong pipeline/run_multi_seed_extra_seeds.sh (có sẵn checkpoint
#      recognition_lr_<backbone>_seed{44,999}/best.pt cho cả 5 backbone).
#
# CẢNH BÁO THỜI GIAN: 5 backbone x 2 seed = 10 lần train recognition (nhẹ,
# không train lại SR).

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results/multi_seed_saliency"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(44 999)
DOMAIN="sr_improved_saliency"
mkdir -p "$RESULTS_DIR"

echo "Kiểm tra tiền đề..."
MISSING=0
if [ ! -d "splits/${DOMAIN}" ] || [ -z "$(find "splits/${DOMAIN}" -type f -print -quit 2>/dev/null)" ]; then
    echo "LỖI: splits/${DOMAIN} không tồn tại hoặc rỗng (không có file ảnh nào)." >&2
    echo "     -> chạy pipeline/run_multi_seed_saliency.sh trước (Bước 1/2, sinh SR)." >&2
    MISSING=1
fi
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: thiếu $CKPT -> chạy pipeline/run_multi_seed_extra_seeds.sh trước." >&2
            MISSING=1
        fi
    done
done
if [ "$MISSING" -eq 1 ]; then
    echo "DỪNG LẠI — thiếu tiền đề ở trên." >&2
    exit 1
fi
echo "OK — đủ tiền đề, bắt đầu chạy (không train lại SR)."

for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "----------------------------------------------------------------"
        echo "backbone=$BACKBONE | seed=$SEED | domain=$DOMAIN"
        echo "----------------------------------------------------------------"
        LR_CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" --backbone "$BACKBONE" \
            --init_ckpt "$LR_CKPT" --seed "$SEED" --run_suffix "_seed${SEED}"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
            --out_json "$RESULTS_DIR/${DOMAIN}_${BACKBONE}_seed${SEED}.json"
    done
done

echo ""
echo ">>> Tổng hợp lại — giờ đọc ĐỦ 5 seed (42,123,2024,44,999) cho domain $DOMAIN..."
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/multi_seed_saliency_summary.csv"

echo ""
echo "HOÀN TẤT. Kết quả (giờ dựa trên n=5 seed): $RESULTS_DIR/multi_seed_saliency_summary.csv"
echo "và $RESULTS_DIR/multi_seed_saliency_summary_pairwise.csv."
