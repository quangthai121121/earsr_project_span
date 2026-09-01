#!/bin/bash
# [MỚI — trả lời phản biện] pipeline/run_ablation_multiseed.sh (loss-component
# ablation, Section~res-loss) chỉ chạy trên 1 backbone (mobilenet_v2), trong
# khi bảng kết quả chính dùng cả 5 backbone -- reviewer chỉ ra kết luận
# "identity loss luôn có hại" / "distillation không đóng góp có ý nghĩa" dựa
# trên 1 backbone duy nhất, chưa biết có nhất quán qua backbone khác không.
#
# Mở rộng thêm 2 backbone đại diện (ResNet-18, EfficientNet-B0) -- KHÔNG cần
# train lại SR (4 checkpoint sr_ablation_<name> không phụ thuộc backbone, đã
# có sẵn từ pipeline/run_ablation.sh), CHỈ thêm fine-tune+eval recognition
# cho 2 backbone mới trên 4 domain sr_ablation_<name> đã build sẵn ảnh.
#
# Ghi kết quả vào CÙNG results/ablation_multiseed/ như script gốc --
# data/aggregate_multi_seed_results.py KHÔNG cần sửa gì, đã tự group theo
# backbone (đọc field "backbone" trong JSON, không phải từ tên file).
#
# QUAN TRỌNG (xem pipeline/run_multi_seed.sh): fine-tune domain sr_ablation_*
# PHẢI dùng checkpoint LR của ĐÚNG seed downstream đang dùng
# (recognition_lr_<backbone>_seed<seed>/best.pt) -- không dùng chung 1
# checkpoint cho mọi seed.
#
# DÙNG:
#   bash pipeline/run_ablation_multiseed_backbones.sh [đường_dẫn_config]
#
# TIỀN ĐỀ: đã chạy xong pipeline/run_ablation.sh (4 checkpoint SR ablation +
# 4 thư mục ảnh) và pipeline/run_multi_seed.sh + _extra_seeds.sh cho 2
# backbone mới (recognition_lr_resnet18_seed<S>, recognition_lr_efficientnet_b0_seed<S>,
# n=5 seed).
#
# CẢNH BÁO THỜI GIAN: 4 cấu hình x 2 backbone x 5 seed = 40 lần train+eval
# recognition (không train SR).

set -e

CONFIG="${1:-configs/config.yaml}"
RESULTS_DIR="results/ablation_multiseed"
NEW_BACKBONES=("resnet18" "efficientnet_b0")
SEEDS=(42 123 2024 44 999)
CONFIGS_ORDER=(pixel_only pixel_distill pixel_identity full)
mkdir -p "$RESULTS_DIR"

ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")

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
            echo "     -> chạy 'bash pipeline/run_multi_seed.sh' (+ _extra_seeds.sh) trước,"
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
echo ">>> Tổng hợp lại (giờ đủ 3 backbone: mobilenet_v2 + resnet18 + efficientnet_b0)..."
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/ablation_multiseed_summary.csv"

echo ""
echo "HOÀN TẤT. Đọc ${RESULTS_DIR}/ablation_multiseed_summary_pairwise.csv --"
echo "giờ có 3 dòng backbone thay vì 1, cho phép kiểm tra tính nhất quán qua backbone."
