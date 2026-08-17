#!/bin/bash
# [MỚI — bổ sung journal Q1, đợt 8] Chạy THÊM 2 seed (44, 999) cho 3 domain
# chính (lr, sr_baseline, sr_improved) x 5 backbone, để cộng với 3 seed đã có
# (42, 123, 2024 — từ pipeline/run_multi_seed.sh) thành ĐỦ 5 SEED/cấu hình.
#
# LÝ DO CẦN: n=3 seed là mức TỐI THIỂU để chạy paired t-test, nhưng lực kiểm
# định (statistical power) khá yếu — dễ ra "chưa đủ ý nghĩa sau Bonferroni"
# ngay cả khi hiệu ứng thật tồn tại (type II error). n=5 là mức thường được
# chấp nhận trong benchmark deep learning (ví dụ nhiều bài NAS/robustness
# report 5 seed), tăng lực kiểm định mà không tốn quá nhiều thời gian.
#
# QUAN TRỌNG — Y HỆT logic pipeline/run_multi_seed.sh (KHÔNG được lệch):
#   - sr_baseline/sr_improved PHẢI fine-tune từ checkpoint LR CỦA CHÍNH SEED
#     ĐÓ (không dùng chung checkpoint hr) — đúng chuỗi hr -> lr -> sr_*.
#   - Ghi vào CÙNG thư mục results/multi_seed/ (không tạo thư mục riêng) —
#     để bước tổng hợp cuối đọc được ĐỦ 5 seed cùng lúc, không phải gộp tay.
#
# CHẠY SAU KHI: đã chạy xong pipeline/run_multi_seed.sh (có sẵn checkpoint
# recognition_lr_<backbone>_seed{42,123,2024} làm nền so sánh "trước/sau").
#
# CẢNH BÁO THỜI GIAN: 5 backbone x 2 seed x 3 domain = 30 lần train recognition
# (thêm vào 45 lần đã chạy trước đó cho seed 42/123/2024).

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results/multi_seed"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(44 999)
mkdir -p "$RESULTS_DIR"

# Kiểm tra tiền đề: cần checkpoint recognition_hr_<backbone> (điểm khởi đầu
# chuỗi fine-tune hr -> lr) đã có từ pipeline/03_train_baseline_recognition.sh.
MISSING=0
for BACKBONE in "${BACKBONES[@]}"; do
    if [ ! -f "runs/recognition_hr_${BACKBONE}/best.pt" ]; then
        echo "LỖI: chưa thấy runs/recognition_hr_${BACKBONE}/best.pt"
        echo "     -> chạy 'bash pipeline/03_train_baseline_recognition.sh' trước."
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo "DỪNG LẠI — thiếu tiền đề ở trên."
    exit 1
fi

for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "################################################################"
        echo "# backbone=$BACKBONE | seed=$SEED | domain=lr"
        echo "################################################################"
        python train_recognition.py --config "$CONFIG" --domain lr --backbone "$BACKBONE" \
            --init_ckpt "runs/recognition_hr_${BACKBONE}/best.pt" \
            --seed "$SEED" --run_suffix "_seed${SEED}"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain lr --test_domain lr \
            --out_json "$RESULTS_DIR/lr_${BACKBONE}_seed${SEED}.json"

        for DOMAIN in sr_baseline sr_improved; do
            echo "################################################################"
            echo "# backbone=$BACKBONE | seed=$SEED | domain=$DOMAIN"
            echo "################################################################"

            python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" --backbone "$BACKBONE" \
                --init_ckpt "runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt" \
                --seed "$SEED" --run_suffix "_seed${SEED}"

            python eval_recognition.py --config "$CONFIG" \
                --ckpt "runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt" \
                --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
                --out_json "$RESULTS_DIR/${DOMAIN}_${BACKBONE}_seed${SEED}.json"
        done
    done
done

echo ""
echo ">>> Tổng hợp lại — giờ đọc ĐỦ 5 seed (42,123,2024,44,999) cho mỗi (backbone, domain)..."
echo ">>> (Yêu cầu: results/multi_seed/ đã có sẵn 45 file *_seed{42,123,2024}.json từ"
echo ">>> pipeline/run_multi_seed.sh — script này CHỈ thêm 30 file *_seed{44,999}.json,"
echo ">>> không xoá/ghi đè file cũ.)"
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/multi_seed_summary.csv"

echo ""
echo "HOÀN TẤT. Kết quả (giờ dựa trên n=5 seed): $RESULTS_DIR/multi_seed_summary.csv"
echo "và $RESULTS_DIR/multi_seed_summary_pairwise.csv (paired t-test + Cohen's d, n=5)."
