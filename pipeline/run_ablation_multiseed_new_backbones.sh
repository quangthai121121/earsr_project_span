#!/bin/bash
# [MỚI — mở rộng 7 backbone mới cho loss-ablation, xem models/recognition_model.py]
# pipeline/run_ablation_multiseed.sh (mobilenet_v2) + run_ablation_multiseed_backbones.sh
# (resnet18, efficientnet_b0) đã kiểm chứng loss-ablation trên 3 backbone --
# phát hiện chính (identity-loss gây hại) chỉ RÕ trên mobilenet_v2, yếu/không
# có trên 2 backbone kia. Mở rộng thêm 7 backbone MỚI để biết phát hiện này
# có lặp lại ở kiến trúc khác họ (shufflenet/squeezenet/mnasnet/mobilenetv3-
# large/regnet/mobileone/lcnet) hay chỉ là đặc thù của vài backbone đã thử.
#
# KHÔNG train lại SR (4 checkpoint sr_ablation_<name> không phụ thuộc
# backbone, đã có sẵn từ pipeline/run_ablation.sh) -- CHỈ train+eval
# recognition cho 7 backbone mới trên 4 domain sr_ablation_<name> đã build
# sẵn ảnh.
#
# Ghi kết quả vào CÙNG results/ablation_multiseed/ như 2 script trước --
# data/aggregate_multi_seed_results.py tự group theo backbone (đọc field
# "backbone" trong JSON), tổng hợp lại sẽ tự ra bảng đủ 10 backbone.
#
# QUAN TRỌNG (xem pipeline/run_multi_seed_new_backbones.sh): fine-tune domain
# sr_ablation_* PHẢI dùng checkpoint LR của ĐÚNG seed downstream đang dùng
# (recognition_lr_<backbone>_seed<seed>/best.pt) -- không dùng chung 1
# checkpoint cho mọi seed.
#
# DÙNG:
#   bash pipeline/run_ablation_multiseed_new_backbones.sh [đường_dẫn_config]
#
# TIỀN ĐỀ: đã chạy xong pipeline/run_ablation.sh (4 checkpoint SR ablation +
# 4 thư mục ảnh, không phụ thuộc backbone) VÀ
# pipeline/run_multi_seed_new_backbones.sh (checkpoint recognition_lr_<backbone
# mới>_seed<seed>, n=5 seed, cho cả 7 backbone mới).
#
# CẢNH BÁO THỜI GIAN: 4 cấu hình x 7 backbone x 5 seed = 140 lần train+eval
# recognition (không train SR).

set -e

CONFIG="${1:-configs/config.yaml}"
RESULTS_DIR="results/ablation_multiseed"
NEW_BACKBONES=("shufflenet_v2_x1_0" "squeezenet1_1" "mnasnet1_0" "mobilenet_v3_large" "regnet_y_400mf" "mobileone_s0" "lcnet_100")
SEEDS=(42 123 2024 44 999)
CONFIGS_ORDER=(pixel_only pixel_distill pixel_identity full)
mkdir -p "$RESULTS_DIR"

echo "################################################################"
echo "# BƯỚC 0 — Kiểm tra tiền đề"
echo "################################################################"
MISSING=0
for NAME in "${CONFIGS_ORDER[@]}"; do
    SPLIT_DIR="splits/sr_ablation_${NAME}"
    if [ ! -d "$SPLIT_DIR" ]; then
        echo "LỖI: chưa thấy $SPLIT_DIR -> chạy 'bash pipeline/run_ablation.sh' trước."
        MISSING=1
    fi
done
for BACKBONE in "${NEW_BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        LR_CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$LR_CKPT" ]; then
            echo "LỖI: chưa thấy $LR_CKPT"
            echo "     -> chạy 'bash pipeline/run_multi_seed_new_backbones.sh' trước,"
            echo "        đảm bảo bao gồm backbone $BACKBONE."
            MISSING=1
        fi
    done
done
if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "DỪNG LẠI — thiếu tiền đề ở trên."
    exit 1
fi
echo "OK — mọi tiền đề đã sẵn sàng (dùng lại checkpoint SR ablation có sẵn, không train lại SR)."

for NAME in "${CONFIGS_ORDER[@]}"; do
    for BACKBONE in "${NEW_BACKBONES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            OUT_JSON="$RESULTS_DIR/ablation_${NAME}_${BACKBONE}_seed${SEED}.json"
            if [ -f "$OUT_JSON" ]; then
                echo ">>> Bỏ qua ablation=$NAME backbone=$BACKBONE seed=$SEED (đã có JSON)"
                continue
            fi

            echo "################################################################"
            echo "# ablation=$NAME | backbone=$BACKBONE | seed=$SEED"
            echo "################################################################"

            python train_recognition.py --config "$CONFIG" --domain "sr_ablation_${NAME}" \
                --backbone "$BACKBONE" \
                --init_ckpt "runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt" \
                --seed "$SEED" --run_suffix "_seed${SEED}"

            python eval_recognition.py --config "$CONFIG" \
                --ckpt "runs/recognition_sr_ablation_${NAME}_${BACKBONE}_seed${SEED}/best.pt" \
                --backbone "$BACKBONE" --train_domain "sr_ablation_${NAME}" \
                --test_domain "sr_ablation_${NAME}" \
                --out_json "$OUT_JSON"
        done
    done
done

echo ""
echo ">>> Tổng hợp lại (giờ đủ 10 backbone: mobilenet_v2 + resnet18 + efficientnet_b0 + 7 backbone mới)..."
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/ablation_multiseed_summary.csv"

echo ""
echo "HOÀN TẤT. Đọc ${RESULTS_DIR}/ablation_multiseed_summary_pairwise.csv --"
echo "giờ có 10 dòng backbone, cho phép kiểm tra tính nhất quán qua backbone rộng hơn nhiều."
