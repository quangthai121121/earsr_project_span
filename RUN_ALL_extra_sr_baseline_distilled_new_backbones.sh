#!/bin/bash
# [MỚI — mở rộng 7 backbone mới cho context-comparison (Table 10,
# span_tiny vs.\ 4 kiến trúc SR nhẹ khác), xem models/recognition_model.py]
# RUN_ALL_extra_sr_baseline_distilled.sh + _extra_seeds.sh đã kiểm chứng trên
# 5 backbone gốc, cho cả 4 kiến trúc (rlfn_adapted, ecbsr, safmn, smfanet).
# Đây LÀ bảng đủ 5 backbone (KHÔNG phải single-backbone screening như
# λ_identity sweep/depth-sweep — đã xác nhận lại qua caption Table 10: "all
# five recognition backbones"), nên PHẢI mở rộng cùng lúc với bảng chính,
# không phải giữ nguyên MobileNetV2-only như đã hiểu nhầm lúc lập kế hoạch
# ban đầu.
#
# PHIÊN BẢN THAM SỐ HÓA — mọi đường dẫn suy ra TỪ CHÍNH CONFIG được truyền vào.
#
# KHÔNG train lại SR (checkpoint sr_improved_<arch>/best.pt không phụ thuộc
# backbone, đã có sẵn từ RUN_ALL_extra_sr_baseline_distilled.sh chạy 1 lần) —
# CHỈ thêm multi-seed recognition cho 7 backbone MỚI, thẳng n=5 seed.
#
# DÙNG:
#   bash RUN_ALL_extra_sr_baseline_distilled_new_backbones.sh <rlfn_adapted|ecbsr|safmn|smfanet> [đường_dẫn_config]
#
# (Không bao gồm "rlfn" thường — kiến trúc đó KHÔNG nằm trong Table 10, hiện
# vẫn n=3 cho cả 5 backbone gốc, xem Limitations của bài báo.)
#
# TIỀN ĐỀ:
#   - Đã chạy xong RUN_ALL_extra_sr_baseline_distilled.sh <arch> [config] (bản gốc).
#   - Đã chạy xong pipeline/run_multi_seed_new_backbones.sh[_awex.sh] — cần
#     checkpoint recognition_lr_<backbone mới>_seed<seed> đủ n=5.
#
# CẢNH BÁO THỜI GIAN: 7 backbone x 5 seed = 35 lần train+eval recognition
# (không train lại SR).

set -e

ARCH="$1"
CONFIG="${2:-configs/config.yaml}"

if [ -z "$ARCH" ] || [[ ! "$ARCH" =~ ^(rlfn_adapted|ecbsr|safmn|smfanet)$ ]]; then
    echo "LỖI: thiếu hoặc sai tên kiến trúc."
    echo "Dùng: bash RUN_ALL_extra_sr_baseline_distilled_new_backbones.sh <rlfn_adapted|ecbsr|safmn|smfanet> [đường_dẫn_config]"
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không tìm thấy config '$CONFIG'."
    exit 1
fi

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")

RESULTS_DIR="${RESULTS_ROOT}/extra_sr_baseline_${ARCH}_distilled"
NEW_BACKBONES=("shufflenet_v2_x1_0" "squeezenet1_1" "mnasnet1_0" "mobilenet_v3_large" "regnet_y_400mf" "mobileone_s0" "lcnet_100")
SEEDS=(42 123 2024 44 999)

echo "################################################################"
echo "# BƯỚC 0 — Kiểm tra tiền đề (arch=$ARCH, config=$CONFIG)         #"
echo "################################################################"
MISSING=0
if [ ! -f "${RUNS_ROOT}/sr_improved_${ARCH}/best.pt" ]; then
    echo "LỖI: chưa thấy ${RUNS_ROOT}/sr_improved_${ARCH}/best.pt"
    echo "     -> chạy 'bash RUN_ALL_extra_sr_baseline_distilled.sh $ARCH $CONFIG' (bản gốc) trước."
    MISSING=1
fi
for BACKBONE in "${NEW_BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: chưa thấy $CKPT -> chạy pipeline/run_multi_seed_new_backbones.sh[_awex.sh] trước."
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
echo "# Multi-seed recognition (7 backbone MỚI x 5 seed) trên domain sr_improved_${ARCH} #"
echo "################################################################"
for BACKBONE in "${NEW_BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUT_JSON="${RESULTS_DIR}/sr_improved_${ARCH}_${BACKBONE}_seed${SEED}.json"
        if [ -f "$OUT_JSON" ]; then
            echo ">>> Bỏ qua backbone=$BACKBONE seed=$SEED (đã có JSON)"
            continue
        fi

        LR_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        echo "### backbone=$BACKBONE | seed=$SEED | domain=sr_improved_${ARCH}"
        python train_recognition.py --config "$CONFIG" --domain "sr_improved_${ARCH}" \
            --backbone "$BACKBONE" --init_ckpt "$LR_CKPT" \
            --seed "$SEED" --run_suffix "_seed${SEED}"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "${RUNS_ROOT}/recognition_sr_improved_${ARCH}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "sr_improved_${ARCH}" --test_domain "sr_improved_${ARCH}" \
            --out_json "$OUT_JSON"
    done
done

echo ""
echo "################################################################"
echo "# Tổng hợp lại (giờ đủ 12 backbone: 5 gốc + 7 mới)                #"
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
echo "HOÀN TẤT (arch: $ARCH distilled, 12 backbone). Kết quả:"
echo "  - ${RESULTS_DIR}/multi_seed_summary_distilled_${ARCH}.csv"
echo "  - ${RESULTS_DIR}/multi_seed_summary_distilled_${ARCH}_pairwise.csv"
