#!/bin/bash
# [MỚI — đóng nốt khoảng trống loss-ablation, xem models/recognition_model.py]
# tab:loss (Section~sec:res-loss) hiện chỉ có 9/11 backbone: 3 backbone gốc
# (mobilenet_v2, resnet18, efficientnet_b0, qua run_ablation_multiseed.sh +
# run_ablation_multiseed_backbones.sh) + 6 backbone mới (qua
# run_ablation_multiseed_new_backbones.sh, sau khi bỏ mnasnet1_0). HAI backbone
# CÒN LẠI trong nhóm 5 backbone gốc -- ghostnet_100 và mobilenet_v3_small --
# CHƯA TỪNG được đưa vào danh sách của run_ablation_multiseed_new_backbones.sh
# (danh sách đó chỉ liệt kê 7 backbone MỚI, không phải phần bù của 3 backbone
# gốc đã thử) nên chưa từng chạy ablation này ở bất kỳ giai đoạn nào. Script
# này CHỈ nhắm đúng 2 backbone còn thiếu đó, không đụng backbone nào khác.
#
# KHÔNG train lại SR (4 checkpoint sr_ablation_<name> không phụ thuộc
# backbone, đã có sẵn từ pipeline/run_ablation.sh) -- CHỈ train+eval
# recognition cho 2 backbone còn thiếu trên 4 domain sr_ablation_<name> đã
# build sẵn ảnh. Checkpoint recognition_lr_<backbone>_seed<seed> cho cả 2
# backbone này PHẢI đã có sẵn (ghostnet_100/mobilenet_v3_small nằm trong
# nhóm 5 backbone dùng từ đầu, qua pipeline/run_multi_seed.sh -- không phải
# nhóm 6 backbone thêm sau).
#
# Ghi kết quả vào CÙNG results/ablation_multiseed/ như 3 script trước --
# data/aggregate_multi_seed_results.py tự group theo backbone, tổng hợp lại
# sẽ tự ra bảng đủ 11 backbone (khớp đúng eleven main-comparison backbones).
#
# DÙNG:
#   bash pipeline/run_ablation_multiseed_final_backbones.sh [đường_dẫn_config]
#
# TIỀN ĐỀ: đã chạy xong pipeline/run_ablation.sh (4 checkpoint SR ablation +
# 4 thư mục ảnh, không phụ thuộc backbone) VÀ có sẵn checkpoint
# recognition_lr_ghostnet_100_seed<seed>/best.pt +
# recognition_lr_mobilenet_v3_small_seed<seed>/best.pt cho cả 5 seed (từ
# pipeline/run_multi_seed.sh, đã chạy từ lâu vì đây là 2 trong 5 backbone gốc).
#
# CẢNH BÁO THỜI GIAN: 4 cấu hình x 2 backbone x 5 seed = 40 lần train+eval
# recognition (không train SR) -- nhẹ hơn nhiều so với 140 lần của bước
# 7-backbone trước đó.

set -e

CONFIG="${1:-configs/config.yaml}"
RESULTS_DIR="results/ablation_multiseed"
FINAL_BACKBONES=("ghostnet_100" "mobilenet_v3_small")
SEEDS=(42 123 2024 44 999)
CONFIGS_ORDER=(pixel_only pixel_distill pixel_identity full)
NUM_WORKERS="${NUM_WORKERS:-4}"
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
for BACKBONE in "${FINAL_BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        LR_CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$LR_CKPT" ]; then
            echo "LỖI: chưa thấy $LR_CKPT"
            echo "     -> $BACKBONE thuộc nhóm 5 backbone gốc, kiểm tra lại"
            echo "        pipeline/run_multi_seed.sh đã chạy đủ n=5 seed cho backbone này chưa."
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
    for BACKBONE in "${FINAL_BACKBONES[@]}"; do
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
                --seed "$SEED" --run_suffix "_seed${SEED}" \
                --num_workers "$NUM_WORKERS"

            python eval_recognition.py --config "$CONFIG" \
                --ckpt "runs/recognition_sr_ablation_${NAME}_${BACKBONE}_seed${SEED}/best.pt" \
                --backbone "$BACKBONE" --train_domain "sr_ablation_${NAME}" \
                --test_domain "sr_ablation_${NAME}" \
                --out_json "$OUT_JSON" --num_workers "$NUM_WORKERS"
        done
    done
done

echo ""
echo ">>> Tổng hợp lại (giờ đủ 11 backbone: mobilenet_v2 + resnet18 + efficientnet_b0 +"
echo "    6 backbone thêm sau + ghostnet_100 + mobilenet_v3_small)..."
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/ablation_multiseed_summary.csv"

echo ""
echo "HOÀN TẤT. Đọc ${RESULTS_DIR}/ablation_multiseed_summary_pairwise.csv --"
echo "giờ có đủ 11 dòng backbone, khớp toàn bộ eleven main-comparison backbones."
