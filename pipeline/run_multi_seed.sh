#!/bin/bash
# Chạy lại 3 domain chính (lr, sr_baseline, sr_improved) với NHIỀU SEED khác
# nhau, cho CẢ 5 BACKBONE — ĐÚNG THEO CHUỖI fine-tune như pipeline chính
# (hr -> lr -> sr_baseline, hr -> lr -> sr_improved), chỉ khác là lặp lại với
# nhiều seed để đo độ ổn định. Quan trọng: sr_baseline/sr_improved phải
# fine-tune từ checkpoint LR CỦA CHÍNH SEED ĐÓ (không dùng chung 1 checkpoint
# hr cho cả 3 domain — làm vậy là THÍ NGHIỆM KHÁC, không so sánh công bằng
# được với kết quả chính).
#
# CẢNH BÁO THỜI GIAN: 5 backbone x 3 seed x 3 domain = 45 lần train recognition.
# Tốn thời gian đáng kể hơn nhiều so với bản chỉ 1 backbone — chạy khi có đủ
# thời gian, cân nhắc chạy qua đêm nếu máy không quá nhanh.

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results/multi_seed"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(42 123 2024)
mkdir -p "$RESULTS_DIR"

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

            # QUAN TRỌNG: fine-tune từ checkpoint LR CỦA CHÍNH SEED NÀY, đúng
            # chuỗi như pipeline chính — không dùng chung checkpoint hr.
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
echo ">>> Tổng hợp — tính trung bình +- độ lệch chuẩn qua các seed, theo từng backbone..."
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/multi_seed_summary.csv"

echo ""
echo "HOÀN TẤT. Kết quả: $RESULTS_DIR/multi_seed_summary.csv"
