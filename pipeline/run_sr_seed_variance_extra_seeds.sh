#!/bin/bash
# [MỚI — mở rộng theo yêu cầu reviewer] Mở rộng pipeline/run_sr_seed_variance.sh
# từ n=3 SR-seed (42,123,2024) lên n=5 (thêm 44,999) — CÙNG khuôn mẫu
# "_extra_seeds.sh" đã dùng cho RUN_ALL_span_large_ablation_extra_seeds.sh /
# pipeline/run_multi_seed_extra_seeds.sh trong project.
#
# LÝ DO: chạy n=3 (screening tier) cho cả 3 kiến trúc trước, phát hiện
# span_baseline có std do SR-seed LỚN HƠN ~3.6x so với std downstream đã báo
# cáo (0.0305 vs 0.0084) — hiệu ứng bất ngờ, đủ lớn để cần bằng chứng vững
# hơn n=3 trước khi đưa vào bài báo (đúng quy ước n=3->n=5 áp dụng nhất quán
# cho các phần khác của bài). span_tiny (ratio 1.16x) và span_large không cần
# mở rộng ngay — chỉ chạy script này cho ARCH cần thêm bằng chứng.
#
# KHÔNG train lại 3 checkpoint SR-seed cũ (42/123/2024) — CHỈ train thêm 2
# checkpoint SR-seed mới (44, 999), dùng CHÍNH XÁC cùng dispatch train/lambda
# theo kiến trúc như pipeline/run_sr_seed_variance.sh (xem case "$ARCH" ở
# dưới — PHẢI sửa đồng bộ cả 2 file nếu sau này đổi recipe train, tương tự
# lưu ý đã ghi ở file gốc).
#
# DÙNG (Y HỆT cú pháp bản gốc, thêm đúng 2 seed):
#   bash pipeline/run_sr_seed_variance_extra_seeds.sh [đường_dẫn_config] [kiến_trúc]
#   ví dụ: bash pipeline/run_sr_seed_variance_extra_seeds.sh configs/config.yaml span_baseline
#
# TIỀN ĐỀ: đã chạy xong pipeline/run_sr_seed_variance.sh CÙNG config + CÙNG
# ARCH trước đó (cần sẵn kết quả n=3 trong results/sr_seed_variance_<arch>/).

set -e

CONFIG="${1:-configs/config.yaml}"
ARCH="${2:-span_tiny}"
BACKBONE="mobilenet_v2"
RECOGNITION_SEED=42
EXTRA_SEEDS=(44 999)
NUM_WORKERS="${NUM_WORKERS:-4}"

if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không thấy $CONFIG"
    exit 1
fi

# [Y HỆT case-statement của run_sr_seed_variance.sh — xem giải thích đầy đủ ở đó]
case "$ARCH" in
    span_tiny)
        STUDENT_ARCH="span_tiny"
        TRAIN_SCRIPT="distill"
        BASE_RUN_DIR="sr_improved_span_tiny"
        BASE_DOMAIN="sr_improved"
        ;;
    span_baseline)
        STUDENT_ARCH="span_official"
        TRAIN_SCRIPT="baseline"
        BASE_RUN_DIR="sr_span_official"
        BASE_DOMAIN="sr_baseline"
        if [ ! -f "checkpoints/span_pretrained_x4.pth" ]; then
            echo "LỖI: thiếu checkpoints/span_pretrained_x4.pth (checkpoint SPAN chính thức, "
            echo "     cần để fine-tune span_baseline, xem scripts/setup_span_official.sh)." >&2
            exit 1
        fi
        ;;
    span_large)
        STUDENT_ARCH="span_large"
        TRAIN_SCRIPT="distill"
        BASE_RUN_DIR="sr_improved_span_large"
        BASE_DOMAIN="sr_span_large"
        ;;
    *)
        echo "LỖI: kiến trúc '$ARCH' không hợp lệ — phải là span_tiny|span_baseline|span_large" >&2
        exit 1
        ;;
esac

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
SPLITS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['splits_root'])")
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

RESULTS_DIR="${RESULTS_ROOT}/sr_seed_variance_${ARCH}"
if [ ! -d "$RESULTS_DIR" ]; then
    echo "LỖI: không thấy $RESULTS_DIR — chạy 'bash pipeline/run_sr_seed_variance.sh $CONFIG $ARCH' "
    echo "     (bản n=3 gốc) trước khi chạy script mở rộng này."
    exit 1
fi

echo "================================================================"
echo "Config          = $CONFIG"
echo "Kiến trúc (ARCH)= $ARCH  (--arch kỹ thuật = $STUDENT_ARCH)"
echo "Backbone        = $BACKBONE (đại diện)"
echo "SR seeds THÊM   = ${EXTRA_SEEDS[*]}  (giữ nguyên 42;123;2024 đã có)"
echo "Downstream seed = $RECOGNITION_SEED (CỐ ĐỊNH)"
echo "Results dir     = $RESULTS_DIR"
echo "================================================================"

INIT_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${RECOGNITION_SEED}/best.pt"
if [ ! -f "$INIT_CKPT" ]; then
    INIT_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}/best.pt"
fi
if [ ! -f "$INIT_CKPT" ]; then
    echo "LỖI: không thấy checkpoint recognition domain 'lr' cho backbone $BACKBONE."
    exit 1
fi

for SEED in "${EXTRA_SEEDS[@]}"; do
    RUN_SUFFIX_SR="_srseed${SEED}"
    DOMAIN_NAME="${BASE_DOMAIN}_srseed${SEED}"
    SR_DOMAIN_DIR="${SPLITS_ROOT}/${DOMAIN_NAME}"
    SR_CKPT="${RUNS_ROOT}/${BASE_RUN_DIR}${RUN_SUFFIX_SR}/best.pt"

    echo ""
    echo "################################################################"
    echo "# [1/2] Train SR '$ARCH' — seed=$SEED (checkpoint MỚI, thêm cho n=5)  #"
    echo "################################################################"
    if [ "$TRAIN_SCRIPT" = "distill" ]; then
        if [ "$ARCH" = "span_tiny" ]; then
            python train_sr_distill.py --config "$CONFIG" --student_arch "$STUDENT_ARCH" \
                --lambda_pixel 1.0 --lambda_distill 1.0 --lambda_feat 0 --lambda_saliency 0 --lambda_identity 0 \
                --seed "$SEED" --run_suffix "$RUN_SUFFIX_SR" --num_workers "$NUM_WORKERS"
        else
            python train_sr_distill.py --config "$CONFIG" --student_arch "$STUDENT_ARCH" \
                --seed "$SEED" --run_suffix "$RUN_SUFFIX_SR" --num_workers "$NUM_WORKERS"
        fi
    else
        python train_sr.py --config "$CONFIG" --sr_arch "$STUDENT_ARCH" \
            --pretrained_path checkpoints/span_pretrained_x4.pth \
            --seed "$SEED" --run_suffix "$RUN_SUFFIX_SR" --num_workers "$NUM_WORKERS"
    fi

    echo ""
    echo "### [2/2] Sinh ảnh SR + đo chất lượng + fine-tune recognition — seed=$SEED ###"
    python data/build_sr.py --lr_dir "${SPLITS_ROOT}/lr" --sr_ckpt "$SR_CKPT" \
        --arch "$STUDENT_ARCH" --scale "$SCALE" --out_dir "$SR_DOMAIN_DIR"

    python eval_sr_quality.py --config "$CONFIG" --arch "$STUDENT_ARCH" --ckpt "$SR_CKPT" \
        --label "${ARCH}_srseed${SEED}" --out_csv "${RESULTS_DIR}/sr_quality_srseed.csv"

    RUN_SUFFIX_REC="_srseed${SEED}"
    python train_recognition.py --config "$CONFIG" --domain "$DOMAIN_NAME" --backbone "$BACKBONE" \
        --init_ckpt "$INIT_CKPT" \
        --seed "$RECOGNITION_SEED" --run_suffix "$RUN_SUFFIX_REC" --num_workers "$NUM_WORKERS"

    python eval_recognition.py --config "$CONFIG" \
        --ckpt "${RUNS_ROOT}/recognition_${DOMAIN_NAME}_${BACKBONE}${RUN_SUFFIX_REC}/best.pt" \
        --backbone "$BACKBONE" --train_domain "$DOMAIN_NAME" --test_domain "$DOMAIN_NAME" \
        --out_json "${RESULTS_DIR}/${BACKBONE}_srseed${SEED}.json" --num_workers "$NUM_WORKERS"
done

echo ""
echo ">>> Tổng hợp lại (giờ đủ n=5 SR-seed: 42,123,2024,44,999) cho $ARCH..."
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
echo "HOÀN TẤT ($ARCH, n=5). Kết quả: ${RESULTS_DIR}/sr_seed_variance_summary.csv"
