#!/bin/bash
# [MỚI — mở rộng 7 backbone mới, ưu tiên nhẹ, xem models/recognition_model.py]
# Bảng kết quả chính (lr, sr_baseline, sr_improved) cho 7 backbone MỚI:
# shufflenet_v2_x1_0, squeezenet1_1, mnasnet1_0, mobilenet_v3_large,
# regnet_y_400mf, mobileone_s0, lcnet_100 — KHÔNG chạy lại 5 backbone gốc.
#
# Khác với run_multi_seed.sh + run_multi_seed_extra_seeds.sh (lịch sử: chạy
# n=3 trước rồi mới mở rộng n=5 vì n=3 dữ liệu đã có sẵn từ trước khi n=5
# thành chuẩn), 7 backbone này CHƯA có dữ liệu nào — chạy thẳng n=5 seed
# trong 1 script duy nhất, không cần bước n=3 trung gian không cần thiết.
#
# Domain ảnh (splits/lr, splits/sr_baseline, splits/sr_improved) KHÔNG phụ
# thuộc backbone — không cần build lại, chỉ cần train+eval recognition.
#
# Ghi kết quả vào CÙNG results/multi_seed/ như script gốc --
# data/aggregate_multi_seed_results.py tự group theo backbone (đọc field
# "backbone" trong JSON, không phải tên file) nên khi tổng hợp lại sẽ tự
# động ra bảng đủ 12 backbone (5 gốc + 7 mới), không cần sửa script tổng hợp.
#
# TIỀN ĐỀ: checkpoint recognition_hr_<backbone> (KHÔNG suffix seed) cho mỗi
# backbone mới CHƯA tồn tại (khác 5 backbone gốc đã có từ Bước 3/8) — script
# này TỰ TRAIN bước đó trước (1 lần/backbone, seed mặc định trong config,
# giống hệt logic pipeline/03_train_baseline_recognition.sh) rồi mới vào
# vòng lặp multi-seed.
#
# DÙNG:
#   bash pipeline/run_multi_seed_new_backbones.sh [đường_dẫn_config]
#
# CẢNH BÁO THỜI GIAN: 7 backbone x (1 lần train hr) + 7 x 5 seed x 3 domain
# (lr, sr_baseline, sr_improved) = 7 + 105 lần train+eval recognition. Không
# train lại SR (span_tiny/span_baseline) — chỉ recognition.

set -e

CONFIG="${1:-configs/config.yaml}"
RESULTS_DIR="results/multi_seed"
NEW_BACKBONES=("shufflenet_v2_x1_0" "squeezenet1_1" "mnasnet1_0" "mobilenet_v3_large" "regnet_y_400mf" "mobileone_s0" "lcnet_100")
SEEDS=(42 123 2024 44 999)
NUM_WORKERS="${NUM_WORKERS:-4}"
mkdir -p "$RESULTS_DIR"

echo "================================================================"
echo "Config     = $CONFIG"
echo "Backbones MỚI = ${NEW_BACKBONES[*]}"
echo "Seeds      = ${SEEDS[*]} (n=5 ngay từ đầu, không qua bước n=3)"
echo "================================================================"

for BACKBONE in "${NEW_BACKBONES[@]}"; do
    HR_CKPT="runs/recognition_hr_${BACKBONE}/best.pt"
    if [ -f "$HR_CKPT" ]; then
        echo ">>> Bỏ qua train recognition_hr cho backbone=$BACKBONE (đã có checkpoint)"
    else
        echo "################################################################"
        echo "# [Tiền đề] backbone=$BACKBONE | domain=hr (không suffix seed)"
        echo "################################################################"
        python train_recognition.py --config "$CONFIG" --domain hr --backbone "$BACKBONE" \
            --num_workers "$NUM_WORKERS"
    fi
done

for BACKBONE in "${NEW_BACKBONES[@]}"; do
    HR_CKPT="runs/recognition_hr_${BACKBONE}/best.pt"
    for SEED in "${SEEDS[@]}"; do
        LR_JSON="$RESULTS_DIR/lr_${BACKBONE}_seed${SEED}.json"
        if [ -f "$LR_JSON" ]; then
            echo ">>> Bỏ qua backbone=$BACKBONE seed=$SEED domain=lr (đã có JSON)"
        else
            echo "################################################################"
            echo "# backbone=$BACKBONE | seed=$SEED | domain=lr"
            echo "################################################################"
            python train_recognition.py --config "$CONFIG" --domain lr --backbone "$BACKBONE" \
                --init_ckpt "$HR_CKPT" \
                --seed "$SEED" --run_suffix "_seed${SEED}" \
                --num_workers "$NUM_WORKERS"

            python eval_recognition.py --config "$CONFIG" \
                --ckpt "runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt" \
                --backbone "$BACKBONE" --train_domain lr --test_domain lr \
                --out_json "$LR_JSON" --num_workers "$NUM_WORKERS"
        fi

        for DOMAIN in sr_baseline sr_improved; do
            OUT_JSON="$RESULTS_DIR/${DOMAIN}_${BACKBONE}_seed${SEED}.json"
            if [ -f "$OUT_JSON" ]; then
                echo ">>> Bỏ qua backbone=$BACKBONE seed=$SEED domain=$DOMAIN (đã có JSON)"
                continue
            fi

            echo "################################################################"
            echo "# backbone=$BACKBONE | seed=$SEED | domain=$DOMAIN"
            echo "################################################################"

            # QUAN TRỌNG (xem run_multi_seed.sh gốc): fine-tune từ checkpoint LR
            # CỦA CHÍNH SEED NÀY, không dùng chung 1 checkpoint cho mọi seed.
            python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" --backbone "$BACKBONE" \
                --init_ckpt "runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt" \
                --seed "$SEED" --run_suffix "_seed${SEED}" \
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
    --out_csv "$RESULTS_DIR/multi_seed_summary.csv"

echo ""
echo "HOÀN TẤT. Kết quả: $RESULTS_DIR/multi_seed_summary.csv"
echo "và $RESULTS_DIR/multi_seed_summary_pairwise.csv"
