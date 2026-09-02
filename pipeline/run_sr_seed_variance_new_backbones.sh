#!/bin/bash
# [MỚI — mở rộng 7 backbone mới cho SR-seed-variance, xem
# models/recognition_model.py] pipeline/run_sr_seed_variance_backbones.sh đã
# mở rộng phép đo phương sai do SR-training-seed từ 1 backbone
# (mobilenet_v2) sang đủ 5 backbone gốc. Mở rộng thêm 7 backbone MỚI để biết
# tỉ lệ phương sai SR-seed/downstream-seed đã quan sát có giữ nguyên trên
# kiến trúc khác họ hay không.
#
# KHÔNG train lại SR (checkpoint SR không phụ thuộc backbone, đã có sẵn từ
# pipeline/run_sr_seed_variance.sh + _extra_seeds.sh) — CHỈ thêm bước
# fine-tune+eval recognition cho 7 backbone mới trên các domain SR-seed đã
# build sẵn ảnh.
#
# Ghi kết quả vào CÙNG results/sr_seed_variance_<arch>/ như script gốc —
# data/aggregate_sr_seed_variance.py tự group theo backbone, tổng hợp lại sẽ
# tự ra bảng đủ 12 backbone.
#
# QUAN TRỌNG: fine-tune domain SR PHẢI dùng checkpoint LR của ĐÚNG seed
# downstream đang dùng (luôn là 42, CỐ ĐỊNH, khớp thiết kế gốc).
#
# DÙNG:
#   bash pipeline/run_sr_seed_variance_new_backbones.sh [đường_dẫn_config] [kiến_trúc]
#   ví dụ: bash pipeline/run_sr_seed_variance_new_backbones.sh configs/config.yaml span_baseline
#
# TIỀN ĐỀ: đã chạy xong pipeline/run_sr_seed_variance.sh (+ _extra_seeds.sh)
# cho CÙNG config + CÙNG ARCH (ảnh SR ở mọi domain <base_domain>[_srseed<N>])
# VÀ pipeline/run_multi_seed_new_backbones.sh (checkpoint
# recognition_lr_<backbone mới>_seed42/best.pt cho cả 7 backbone mới).

set -e

CONFIG="${1:-configs/config.yaml}"
ARCH="${2:-span_tiny}"
RECOGNITION_SEED=42
NEW_BACKBONES=("shufflenet_v2_x1_0" "squeezenet1_1" "mnasnet1_0" "mobilenet_v3_large" "regnet_y_400mf" "mobileone_s0" "lcnet_100")
NUM_WORKERS="${NUM_WORKERS:-4}"

if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không thấy $CONFIG"
    exit 1
fi

case "$ARCH" in
    span_tiny)
        BASE_DOMAIN="sr_improved"
        SR_SEEDS=(42 123 2024 44 999)
        ;;
    span_baseline)
        BASE_DOMAIN="sr_baseline"
        SR_SEEDS=(42 123 2024 44 999)
        ;;
    span_large)
        BASE_DOMAIN="sr_span_large"
        SR_SEEDS=(42 123 2024 44 999)
        ;;
    *)
        echo "LỖI: kiến trúc '$ARCH' không hợp lệ — phải là span_tiny|span_baseline|span_large" >&2
        exit 1
        ;;
esac

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
SPLITS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['splits_root'])")
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")

RESULTS_DIR="${RESULTS_ROOT}/sr_seed_variance_${ARCH}"
if [ ! -d "$RESULTS_DIR" ]; then
    echo "LỖI: không thấy $RESULTS_DIR — chạy 'bash pipeline/run_sr_seed_variance.sh $CONFIG $ARCH' trước."
    exit 1
fi

echo "================================================================"
echo "Config          = $CONFIG"
echo "Kiến trúc (ARCH)= $ARCH"
echo "SR seeds        = ${SR_SEEDS[*]} (dùng lại checkpoint/ảnh có sẵn, KHÔNG train lại)"
echo "Backbone mới    = ${NEW_BACKBONES[*]}"
echo "Downstream seed = $RECOGNITION_SEED (CỐ ĐỊNH)"
echo "Results dir     = $RESULTS_DIR"
echo "================================================================"

MISSING=0
for SEED in "${SR_SEEDS[@]}"; do
    if [ "$SEED" -eq 42 ]; then
        DOMAIN_NAME="$BASE_DOMAIN"
    else
        DOMAIN_NAME="${BASE_DOMAIN}_srseed${SEED}"
    fi
    if [ ! -d "${SPLITS_ROOT}/${DOMAIN_NAME}" ]; then
        echo "LỖI: thiếu ${SPLITS_ROOT}/${DOMAIN_NAME} (ảnh SR seed=$SEED chưa build)."
        MISSING=1
    fi
done
for BACKBONE in "${NEW_BACKBONES[@]}"; do
    CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${RECOGNITION_SEED}/best.pt"
    if [ ! -f "$CKPT" ]; then
        echo "LỖI: thiếu $CKPT — chạy pipeline/run_multi_seed_new_backbones.sh trước."
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo "DỪNG LẠI — thiếu tiền đề ở trên."
    exit 1
fi
echo "OK — đủ tiền đề, bắt đầu chạy."

for SEED in "${SR_SEEDS[@]}"; do
    if [ "$SEED" -eq 42 ]; then
        DOMAIN_NAME="$BASE_DOMAIN"
    else
        DOMAIN_NAME="${BASE_DOMAIN}_srseed${SEED}"
    fi

    for BACKBONE in "${NEW_BACKBONES[@]}"; do
        if [ -f "${RESULTS_DIR}/${BACKBONE}_srseed${SEED}.json" ]; then
            echo ">>> Bỏ qua backbone=$BACKBONE seed=$SEED (đã có JSON từ lần chạy trước)"
            continue
        fi

        echo ""
        echo "################################################################"
        echo "# $ARCH | SR-seed=$SEED | backbone=$BACKBONE | domain=$DOMAIN_NAME #"
        echo "################################################################"
        INIT_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${RECOGNITION_SEED}/best.pt"

        RUN_SUFFIX_REC="_srseed${SEED}"
        python train_recognition.py --config "$CONFIG" --domain "$DOMAIN_NAME" --backbone "$BACKBONE" \
            --init_ckpt "$INIT_CKPT" \
            --seed "$RECOGNITION_SEED" --run_suffix "$RUN_SUFFIX_REC" \
            --num_workers "$NUM_WORKERS"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "${RUNS_ROOT}/recognition_${DOMAIN_NAME}_${BACKBONE}${RUN_SUFFIX_REC}/best.pt" \
            --backbone "$BACKBONE" --train_domain "$DOMAIN_NAME" --test_domain "$DOMAIN_NAME" \
            --out_json "${RESULTS_DIR}/${BACKBONE}_srseed${SEED}.json" --num_workers "$NUM_WORKERS"
    done
done

echo ""
echo ">>> Tổng hợp lại (giờ đủ 12 backbone) cho $ARCH..."
DOWNSTREAM_CSV="${RESULTS_ROOT}/multi_seed/multi_seed_summary.csv"
if [ "$ARCH" = "span_large" ] && [ -f "${RESULTS_ROOT}/span_large_ablation/multi_seed_summary_4domains.csv" ]; then
    DOWNSTREAM_CSV="${RESULTS_ROOT}/span_large_ablation/multi_seed_summary_4domains.csv"
fi
python data/aggregate_sr_seed_variance.py --results_dir "$RESULTS_DIR" \
    --out_csv "${RESULTS_DIR}/sr_seed_variance_summary.csv" \
    --downstream_multiseed_csv "$DOWNSTREAM_CSV" \
    --downstream_domain "$BASE_DOMAIN" \
    --sr_quality_csv "${RESULTS_DIR}/sr_quality_srseed.csv"

echo ""
echo "HOÀN TẤT ($ARCH, đủ 12 backbone). Kết quả: ${RESULTS_DIR}/sr_seed_variance_summary.csv"
echo "LƯU Ý: file summary CSV giờ có 12 dòng (1/backbone) thay vì 5 — xem cột 'backbone'."
