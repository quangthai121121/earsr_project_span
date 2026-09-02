#!/bin/bash
# [MỚI — mở rộng 7 backbone mới cho AWEx from-scratch track, xem
# models/recognition_model.py] Bản sao của pipeline/run_multi_seed_new_backbones.sh
# (EarVN1.0) áp dụng cho AWEx (từ đầu, không transfer), CHỈ khác đường dẫn
# runs_awex/ splits_awex/ results_awex/ thay vì runs/ splits/ results/ —
# xem pipeline_awex/ (sinh ra bởi scripts/setup_second_dataset_pipeline.sh
# trên server, không có trong repo local) để đối chiếu quy ước đặt tên gốc
# nếu cần, nhưng script này KHÔNG phụ thuộc file đó, tự đứng độc lập.
#
# Domain ảnh (splits_awex/lr, splits_awex/sr_baseline, splits_awex/sr_improved)
# KHÔNG phụ thuộc backbone — không cần build lại, chỉ cần train+eval
# recognition. Thẳng n=5 seed (không qua bước n=3 trung gian, giống lý do đã
# nêu ở bản EarVN1.0).
#
# Ghi kết quả vào CÙNG results_awex/multi_seed/ như pipeline AWEx gốc --
# data/aggregate_multi_seed_results.py tự group theo backbone, tổng hợp lại
# sẽ tự ra bảng đủ 12 backbone.
#
# DÙNG:
#   bash pipeline/run_multi_seed_new_backbones_awex.sh [đường_dẫn_config_awex]
#   mặc định: configs/config_awex.yaml
#
# TIỀN ĐỀ: pipeline AWEx gốc (RUN_ALL_NEW_DATASET.sh awex) đã chạy xong ít
# nhất tới bước có splits_awex/{lr,sr_baseline,sr_improved}/ (không phụ
# thuộc backbone, dùng chung cho mọi backbone).
#
# CẢNH BÁO THỜI GIAN: 7 backbone x (1 lần train hr) + 7 x 5 seed x 3 domain
# (lr, sr_baseline, sr_improved) = 7 + 105 lần train+eval recognition. Không
# train lại SR.

set -e

CONFIG="${1:-configs/config_awex.yaml}"
RUNS_DIR="runs_awex"
SPLITS_DIR="splits_awex"
RESULTS_DIR="results_awex/multi_seed"
NEW_BACKBONES=("shufflenet_v2_x1_0" "squeezenet1_1" "mnasnet1_0" "mobilenet_v3_large" "regnet_y_400mf" "mobileone_s0" "lcnet_100")
SEEDS=(42 123 2024 44 999)
NUM_WORKERS="${NUM_WORKERS:-4}"
mkdir -p "$RESULTS_DIR"

echo "================================================================"
echo "Config     = $CONFIG"
echo "Backbones MỚI = ${NEW_BACKBONES[*]}"
echo "Seeds      = ${SEEDS[*]} (n=5 ngay từ đầu, không qua bước n=3)"
echo "================================================================"

MISSING=0
for DOMAIN in lr sr_baseline sr_improved; do
    if [ ! -d "${SPLITS_DIR}/${DOMAIN}" ] || [ -z "$(find "${SPLITS_DIR}/${DOMAIN}" -type f -print -quit 2>/dev/null)" ]; then
        echo "LỖI: ${SPLITS_DIR}/${DOMAIN} không tồn tại hoặc rỗng." >&2
        echo "     -> chạy pipeline AWEx gốc (RUN_ALL_NEW_DATASET.sh awex) trước." >&2
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo "DỪNG LẠI — thiếu tiền đề ở trên." >&2
    exit 1
fi

for BACKBONE in "${NEW_BACKBONES[@]}"; do
    HR_CKPT="${RUNS_DIR}/recognition_hr_${BACKBONE}/best.pt"
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
    HR_CKPT="${RUNS_DIR}/recognition_hr_${BACKBONE}/best.pt"
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
                --ckpt "${RUNS_DIR}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt" \
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

            python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" --backbone "$BACKBONE" \
                --init_ckpt "${RUNS_DIR}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt" \
                --seed "$SEED" --run_suffix "_seed${SEED}" \
                --num_workers "$NUM_WORKERS"

            python eval_recognition.py --config "$CONFIG" \
                --ckpt "${RUNS_DIR}/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt" \
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
