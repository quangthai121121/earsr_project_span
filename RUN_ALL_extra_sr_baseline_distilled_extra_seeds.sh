#!/bin/bash
# [MỚI — bổ sung journal Q1, đợt 9] Mở rộng Track B
# (RUN_ALL_extra_sr_baseline_distilled.sh) từ n=3 seed lên n=5 seed (thêm
# 44, 999) — Track B là bảng so sánh CHÍNH với span_tiny (cùng recipe
# distillation), nên đồng bộ seed ở đây quan trọng hơn Track A.
#
# KHÔNG train lại SR (chỉ 1 lần, seed=42 mặc định, đã có sẵn từ lần chạy
# RUN_ALL_extra_sr_baseline_distilled.sh gốc). CHỈ thêm 2 seed cho bước
# train_recognition.py trên domain sr_improved_<arch> đã có sẵn.
#
# DÙNG (Y HỆT cú pháp bản gốc):
#   bash RUN_ALL_extra_sr_baseline_distilled_extra_seeds.sh <rlfn|rlfn_adapted|ecbsr|safmn|smfanet> [đường_dẫn_config]
#
# TIỀN ĐỀ (kiểm tra ở BƯỚC 0):
#   - Đã chạy xong RUN_ALL_extra_sr_baseline_distilled.sh <arch> [config] (bản gốc)
#   - Đã chạy xong pipeline/run_multi_seed_extra_seeds.sh (đúng CONFIG này)

set -e

ARCH="$1"
CONFIG="${2:-configs/config.yaml}"

if [ -z "$ARCH" ] || [[ ! "$ARCH" =~ ^(rlfn|rlfn_adapted|ecbsr|safmn|smfanet)$ ]]; then
    echo "LỖI: thiếu hoặc sai tên kiến trúc."
    echo "Dùng: bash RUN_ALL_extra_sr_baseline_distilled_extra_seeds.sh <rlfn|rlfn_adapted|ecbsr|safmn|smfanet> [đường_dẫn_config]"
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không tìm thấy config '$CONFIG'."
    exit 1
fi

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")

RESULTS_DIR="${RESULTS_ROOT}/extra_sr_baseline_${ARCH}_distilled"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(44 999)

echo "################################################################"
echo "# BƯỚC 0 — Kiểm tra tiền đề (arch=$ARCH, config=$CONFIG)         #"
echo "################################################################"
MISSING=0
if [ ! -f "${RUNS_ROOT}/sr_improved_${ARCH}/best.pt" ]; then
    echo "LỖI: chưa thấy ${RUNS_ROOT}/sr_improved_${ARCH}/best.pt"
    echo "     -> chạy 'bash RUN_ALL_extra_sr_baseline_distilled.sh $ARCH $CONFIG' (bản gốc) trước."
    MISSING=1
fi
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: chưa thấy $CKPT -> chạy 'bash pipeline/run_multi_seed_extra_seeds.sh' trước."
            MISSING=1
        fi
    done
done
if [ "$MISSING" -eq 1 ]; then
    echo "DỪNG LẠI — thiếu tiền đề ở trên."
    exit 1
fi
echo "OK — bắt đầu chạy."
mkdir -p "$RESULTS_DIR"

echo ""
echo "################################################################"
echo "# BƯỚC 1 — Thêm seed 44,999 cho recognition trên domain sr_improved_${ARCH} #"
echo "################################################################"
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        LR_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"

        echo "### backbone=$BACKBONE | seed=$SEED | domain=sr_improved_${ARCH}"
        python train_recognition.py --config "$CONFIG" --domain "sr_improved_${ARCH}" \
            --backbone "$BACKBONE" --init_ckpt "$LR_CKPT" \
            --seed "$SEED" --run_suffix "_seed${SEED}"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "${RUNS_ROOT}/recognition_sr_improved_${ARCH}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "sr_improved_${ARCH}" --test_domain "sr_improved_${ARCH}" \
            --out_json "${RESULTS_DIR}/sr_improved_${ARCH}_${BACKBONE}_seed${SEED}.json"
    done
done

echo ""
echo "################################################################"
echo "# BƯỚC 2 — Tổng hợp lại (giờ đủ 5 seed: 42,123,2024,44,999)      #"
echo "################################################################"
rm -rf "${RESULTS_DIR}/combined"
mkdir -p "${RESULTS_DIR}/combined"
if ls "${RESULTS_ROOT}/multi_seed"/*_seed*.json >/dev/null 2>&1; then
    cp "${RESULTS_ROOT}/multi_seed"/*_seed*.json "${RESULTS_DIR}/combined/"
else
    echo "CẢNH BÁO: không thấy ${RESULTS_ROOT}/multi_seed/*_seed*.json — chỉ tổng hợp được domain sr_improved_${ARCH}."
fi
cp "${RESULTS_DIR}"/sr_improved_${ARCH}_*_seed*.json "${RESULTS_DIR}/combined/"

python data/aggregate_multi_seed_results.py --results_dir "${RESULTS_DIR}/combined" \
    --out_csv "${RESULTS_DIR}/multi_seed_summary_distilled_${ARCH}.csv"

echo ""
echo "HOÀN TẤT (arch: $ARCH distilled, giờ n=5 seed). Kết quả:"
echo "  - ${RESULTS_DIR}/multi_seed_summary_distilled_${ARCH}.csv"
echo "  - ${RESULTS_DIR}/multi_seed_summary_distilled_${ARCH}_pairwise.csv"
echo ""
echo "So sánh CÔNG BẰNG (cùng recipe, giờ cùng n=5 seed) với span_tiny: đọc domain"
echo "'sr_improved' trong ${RESULTS_ROOT}/multi_seed/multi_seed_summary.csv."
