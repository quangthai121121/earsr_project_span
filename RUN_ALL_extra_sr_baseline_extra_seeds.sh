#!/bin/bash
# [MỚI — bổ sung journal Q1, đợt 9] Mở rộng Track A (RUN_ALL_extra_sr_baseline.sh)
# từ n=3 seed lên n=5 seed (thêm 44, 999) — đồng bộ lực kiểm định thống kê
# với bảng so sánh chính (đã mở rộng qua pipeline/run_multi_seed_extra_seeds.sh).
#
# KHÔNG train lại SR (train_sr.py không có cờ --seed — SR của Track A LUÔN
# train ở seed=42 mặc định config.yaml, đây là quy ước có chủ đích, không
# phải thiếu sót — xem RUN_ALL_extra_sr_baseline.sh gốc). CHỈ thêm 2 seed
# cho bước train_recognition.py trên domain sr_<arch> đã có sẵn, rồi tổng
# hợp lại (giờ đọc đủ 5 seed).
#
# DÙNG (Y HỆT cú pháp bản gốc):
#   bash RUN_ALL_extra_sr_baseline_extra_seeds.sh <rlfn|rlfn_adapted|ecbsr|safmn|smfanet> [đường_dẫn_config]
#
# TIỀN ĐỀ (kiểm tra ở BƯỚC 0):
#   - Đã chạy xong RUN_ALL_extra_sr_baseline.sh <arch> [config] (bản gốc, seed 42/123/2024)
#   - Đã chạy xong pipeline/run_multi_seed_extra_seeds.sh (đúng CONFIG này) —
#     cần checkpoint recognition_lr_<backbone>_seed{44,999}

set -e

ARCH="$1"
CONFIG="${2:-configs/config.yaml}"

if [ -z "$ARCH" ] || [[ ! "$ARCH" =~ ^(rlfn|rlfn_adapted|ecbsr|safmn|smfanet)$ ]]; then
    echo "LỖI: thiếu hoặc sai tên kiến trúc."
    echo "Dùng: bash RUN_ALL_extra_sr_baseline_extra_seeds.sh <rlfn|rlfn_adapted|ecbsr|safmn|smfanet> [đường_dẫn_config]"
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không tìm thấy config '$CONFIG'."
    exit 1
fi

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")

RESULTS_DIR="${RESULTS_ROOT}/extra_sr_baseline_${ARCH}"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(44 999)

echo "################################################################"
echo "# BƯỚC 0 — Kiểm tra tiền đề (arch=$ARCH, config=$CONFIG)         #"
echo "################################################################"
MISSING=0
if [ ! -f "${RUNS_ROOT}/sr_${ARCH}/best.pt" ]; then
    echo "LỖI: chưa thấy ${RUNS_ROOT}/sr_${ARCH}/best.pt"
    echo "     -> chạy 'bash RUN_ALL_extra_sr_baseline.sh $ARCH $CONFIG' (bản gốc) trước."
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
echo "# BƯỚC 1 — Thêm seed 44,999 cho recognition trên domain sr_${ARCH} #"
echo "################################################################"
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        LR_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"

        echo "### backbone=$BACKBONE | seed=$SEED | domain=sr_${ARCH}"
        python train_recognition.py --config "$CONFIG" --domain "sr_${ARCH}" \
            --backbone "$BACKBONE" --init_ckpt "$LR_CKPT" \
            --seed "$SEED" --run_suffix "_seed${SEED}"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "${RUNS_ROOT}/recognition_sr_${ARCH}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "sr_${ARCH}" --test_domain "sr_${ARCH}" \
            --out_json "${RESULTS_DIR}/sr_${ARCH}_${BACKBONE}_seed${SEED}.json"
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
    echo "CẢNH BÁO: không thấy ${RESULTS_ROOT}/multi_seed/*_seed*.json — chỉ tổng hợp được domain sr_${ARCH}."
fi
cp "${RESULTS_DIR}"/sr_${ARCH}_*_seed*.json "${RESULTS_DIR}/combined/"

python data/aggregate_multi_seed_results.py --results_dir "${RESULTS_DIR}/combined" \
    --out_csv "${RESULTS_DIR}/multi_seed_summary_with_${ARCH}.csv"

echo ""
echo "HOÀN TẤT (arch: $ARCH, giờ n=5 seed). Kết quả:"
echo "  - ${RESULTS_DIR}/multi_seed_summary_with_${ARCH}.csv"
echo "  - ${RESULTS_DIR}/multi_seed_summary_with_${ARCH}_pairwise.csv"
