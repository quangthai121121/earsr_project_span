#!/bin/bash
# Chạy lại 3 domain chính (lr, sr_baseline, sr_improved) với NHIỀU SEED khác
# nhau, cùng khởi tạo từ 1 checkpoint HR chung (không train lại HR, tiết kiệm
# thời gian) — để đo độ ổn định của kết luận "sr_improved > lr" và
# "sr_baseline > sr_improved", tránh kết luận vội từ 1 lần chạy có thể chỉ là
# nhiễu ngẫu nhiên (đã phát hiện: cùng 1 cấu hình, 2 lần chạy có thể lệch
# ~0.8 điểm % chỉ do random seed).
#
# Dùng 1 backbone duy nhất (mobilenet_v2) để tiết kiệm thời gian.

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results/multi_seed"
BACKBONE="mobilenet_v2"
SEEDS=(42 123 2024)
mkdir -p "$RESULTS_DIR"

for SEED in "${SEEDS[@]}"; do
    for DOMAIN in lr sr_baseline sr_improved; do
        echo "################################################################"
        echo "# seed=$SEED | domain=$DOMAIN | backbone=$BACKBONE"
        echo "################################################################"

        python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" --backbone "$BACKBONE" \
            --init_ckpt "runs/recognition_hr_${BACKBONE}/best.pt" \
            --seed "$SEED" --run_suffix "_seed${SEED}"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
            --out_json "$RESULTS_DIR/${DOMAIN}_seed${SEED}.json"
    done
done

echo ""
echo ">>> Tổng hợp — tính trung bình +- độ lệch chuẩn qua các seed..."
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/multi_seed_summary.csv"

echo ""
echo "HOÀN TẤT. Kết quả: $RESULTS_DIR/multi_seed_summary.csv"
