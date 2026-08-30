#!/bin/bash
# [MỚI — 2026-08-30] Chạy THÊM 2 seed (44, 999) cho cả 3 domain vị trí khối
# ("sr_position_keepfirst/keeplast/interleaved"), để cộng với 3 seed đã có
# (42, 123, 2024 — từ pipeline/run_block_position_ablation.sh) thành ĐỦ 5
# SEED, đúng mẫu pipeline/run_multi_seed_saliency_extra_seeds.sh.
#
# LÝ DO MỞ RỘNG (xem results/block_position_screen/*_pairwise.csv, n=3):
# 1/15 cặp có ý nghĩa sau Bonferroni (efficientnet_b0: keepfirst > interleaved,
# d=8.80) VÀ xu hướng nhất quán ở 4/5 backbone (keeplast xếp hạng cao nhất,
# keepfirst — thiết kế span_tiny hiện tại — không bao giờ xếp hạng cao nhất) —
# đủ tiêu chí "khác biệt đáng chú ý" đã đặt ra trước khi chạy để mở rộng lên
# n=5, thay vì dừng ở n=3 kết luận vội.
#
# KHÔNG train lại SR — TÁI SỬ DỤNG 3 checkpoint
# runs/sr_improved_<student_arch>_position_<variant>/best.pt và ảnh đã sinh
# sẵn ở splits/sr_position_<variant>/ (đúng seed cố định SR_SEED=42 như
# pipeline/run_block_position_ablation.sh) — CHỈ multi-seed thêm ở bước
# train_recognition.py, giống hệt nguyên tắc "SR train 1 lần, seed cố định;
# chỉ recognition lặp seed" đã áp dụng xuyên suốt project.
#
# YÊU CẦU:
#   1. Đã chạy xong pipeline/run_block_position_ablation.sh (có sẵn ảnh SR ở
#      splits/sr_position_{keepfirst,keeplast,interleaved}/ và 3 seed
#      42/123/2024 trong results/block_position_screen/).
#   2. Đã chạy xong pipeline/run_multi_seed_extra_seeds.sh (có sẵn checkpoint
#      recognition_lr_<backbone>_seed{44,999}/best.pt cho cả 5 backbone).
#
# CẢNH BÁO THỜI GIAN: 3 domain x 5 backbone x 2 seed = 30 lần train
# recognition (nhẹ, không train lại SR).

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results/block_position_screen"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(44 999)
DOMAINS=("sr_position_keepfirst" "sr_position_keeplast" "sr_position_interleaved")
# [MỚI — 2026-08-30, sự cố thật] xem giải thích đầy đủ trong
# pipeline/run_block_position_ablation.sh — giữ NGUYÊN cùng cơ chế ở đây để
# nhất quán nếu server vẫn đang bị chia sẻ nặng lúc chạy tiếp seed 44/999.
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

for DOMAIN in "${DOMAINS[@]}"; do
    for BACKBONE in "${BACKBONES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            # Bỏ qua tổ hợp ĐÃ CHẠY XONG THẬT SỰ (cho phép chạy lại an toàn
            # nếu bị ngắt giữa chừng, giống mẫu run_block_position_ablation.sh).
            if [ -f "$RESULTS_DIR/${DOMAIN}_${BACKBONE}_seed${SEED}.json" ]; then
                echo ">>> Bỏ qua $DOMAIN backbone=$BACKBONE seed=$SEED (đã có JSON từ lần chạy trước)"
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
                --out_json "$RESULTS_DIR/${DOMAIN}_${BACKBONE}_seed${SEED}.json" \
                --num_workers "$NUM_WORKERS"
        done
    done
done

echo ""
echo ">>> Tổng hợp lại — giờ đọc ĐỦ 5 seed (42,123,2024,44,999) cho cả 3 domain..."
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/block_position_screen_summary.csv"

echo ""
echo "HOÀN TẤT. Kết quả (giờ dựa trên n=5 seed):"
echo "$RESULTS_DIR/block_position_screen_summary.csv và"
echo "$RESULTS_DIR/block_position_screen_summary_pairwise.csv."
