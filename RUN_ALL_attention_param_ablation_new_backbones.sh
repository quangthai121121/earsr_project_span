#!/bin/bash
# [MỚI — mở rộng 7 backbone mới cho attention-parameterization ablation
# (tab:attn-ablation), xem models/recognition_model.py]
# RUN_ALL_attention_param_ablation.sh + _extra_seeds.sh đã kiểm chứng trên 5
# backbone gốc (kết luận: SAFMN/SMFANet không mất nhiều hơn SPAN từ việc cắt
# nửa độ sâu — bác bỏ giả thuyết "attention không tham số học chịu nén sâu
# tốt hơn"). Đây là 1 trong các bảng kết quả đủ 5 backbone của bài, nên mở
# rộng cùng lúc với các bảng khác để nhất quán.
#
# PHIÊN BẢN THAM SỐ HÓA — mọi đường dẫn suy ra TỪ CHÍNH CONFIG được truyền
# vào, giống quy ước của bản gốc.
#
# KHÔNG train lại SR half-depth (checkpoint sr_improved_<arch>_half<N>/best.pt
# không phụ thuộc backbone, đã có sẵn từ RUN_ALL_attention_param_ablation.sh
# chạy 1 lần) — CHỈ thêm multi-seed recognition cho 7 backbone MỚI, thẳng
# n=5 seed.
#
# DÙNG:
#   bash RUN_ALL_attention_param_ablation_new_backbones.sh <safmn|smfanet> [n_blocks_half] [đường_dẫn_config]
#
# TIỀN ĐỀ:
#   - Đã chạy xong RUN_ALL_attention_param_ablation.sh <arch> [n_blocks_half] [config] (bản gốc).
#   - Đã chạy xong pipeline/run_multi_seed_new_backbones.sh[_awex.sh] — cần
#     checkpoint recognition_lr_<backbone mới>_seed<seed> đủ n=5.
#   - Đã chạy xong RUN_ALL_extra_sr_baseline_distilled_new_backbones.sh <arch> [config] —
#     cần JSON full-depth cho 7 backbone mới để Δ_<arch> đủ n=5 khớp phía half-depth.
#   - Đã chạy xong RUN_ALL_span_large_ablation_new_backbones.sh [config] — cần
#     JSON sr_span_large cho 7 backbone mới (Δ_SPAN đối chứng).
#
# CẢNH BÁO THỜI GIAN: 7 backbone x 5 seed = 35 lần train+eval recognition
# (không train lại SR).

set -e

ARCH="$1"
N_HALF="${2:-4}"
CONFIG="${3:-configs/config.yaml}"

if [ -z "$ARCH" ] || [[ ! "$ARCH" =~ ^(safmn|smfanet)$ ]]; then
    echo "LỖI: thiếu hoặc sai tên kiến trúc."
    echo "Dùng: bash RUN_ALL_attention_param_ablation_new_backbones.sh <safmn|smfanet> [n_blocks_half] [đường_dẫn_config]"
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không tìm thấy config '$CONFIG'."
    exit 1
fi

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")

TAG="${ARCH}_half${N_HALF}"
DOMAIN="sr_improved_${TAG}"
RESULTS_DIR="${RESULTS_ROOT}/attention_param_ablation_${TAG}"
NEW_BACKBONES=("shufflenet_v2_x1_0" "squeezenet1_1" "mnasnet1_0" "mobilenet_v3_large" "regnet_y_400mf" "mobileone_s0" "lcnet_100")
SEEDS=(42 123 2024 44 999)
NUM_WORKERS="${NUM_WORKERS:-4}"

echo "################################################################"
echo "# BƯỚC 0 — Kiểm tra tiền đề (arch=$ARCH, n_blocks_half=$N_HALF, config=$CONFIG) #"
echo "################################################################"
MISSING=0
if [ ! -f "${RUNS_ROOT}/${DOMAIN}/best.pt" ]; then
    echo "LỖI: chưa thấy ${RUNS_ROOT}/${DOMAIN}/best.pt"
    echo "     -> chạy 'bash RUN_ALL_attention_param_ablation.sh $ARCH $N_HALF $CONFIG' (bản gốc) trước."
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
    echo ""
    echo "DỪNG LẠI — thiếu tiền đề ở trên."
    exit 1
fi
echo "OK — mọi tiền đề đã sẵn sàng."
mkdir -p "$RESULTS_DIR"

echo ""
echo "################################################################"
echo "# Multi-seed recognition (7 backbone MỚI x 5 seed) trên domain ${DOMAIN} #"
echo "################################################################"
for BACKBONE in "${NEW_BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUT_JSON="${RESULTS_DIR}/${DOMAIN}_${BACKBONE}_seed${SEED}.json"
        if [ -f "$OUT_JSON" ]; then
            echo ">>> Bỏ qua backbone=$BACKBONE seed=$SEED (đã có JSON)"
            continue
        fi

        LR_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        echo "### backbone=$BACKBONE | seed=$SEED | domain=${DOMAIN}"
        python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" \
            --backbone "$BACKBONE" --init_ckpt "$LR_CKPT" \
            --seed "$SEED" --run_suffix "_seed${SEED}" \
            --num_workers "$NUM_WORKERS"
        python eval_recognition.py --config "$CONFIG" \
            --ckpt "${RUNS_ROOT}/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
            --out_json "$OUT_JSON" --num_workers "$NUM_WORKERS"
    done
done

echo ""
echo "################################################################"
echo "# Gộp lại 'combined' (giờ đủ 12 backbone: 5 gốc + 7 mới)         #"
echo "################################################################"
rm -rf "${RESULTS_DIR}/combined"
mkdir -p "${RESULTS_DIR}/combined"
cp "${RESULTS_DIR}"/${DOMAIN}_*_seed*.json "${RESULTS_DIR}/combined/"
FULL_RESULTS_DIR="${RESULTS_ROOT}/extra_sr_baseline_${ARCH}_distilled"
if ls "${FULL_RESULTS_DIR}"/sr_improved_${ARCH}_*_seed*.json >/dev/null 2>&1; then
    cp "${FULL_RESULTS_DIR}"/sr_improved_${ARCH}_*_seed*.json "${RESULTS_DIR}/combined/"
else
    echo "CẢNH BÁO: không thấy ${FULL_RESULTS_DIR}/sr_improved_${ARCH}_*_seed*.json —"
    echo "          chạy RUN_ALL_extra_sr_baseline_distilled_new_backbones.sh $ARCH trước để"
    echo "          có đủ Delta_${ARCH} cho 7 backbone mới."
fi
if ls "${RESULTS_ROOT}/multi_seed"/*_seed*.json >/dev/null 2>&1; then
    cp "${RESULTS_ROOT}/multi_seed"/*_seed*.json "${RESULTS_DIR}/combined/"
fi
if ls "${RESULTS_ROOT}/span_large_ablation"/sr_span_large_*_seed*.json >/dev/null 2>&1; then
    cp "${RESULTS_ROOT}/span_large_ablation"/sr_span_large_*_seed*.json "${RESULTS_DIR}/combined/"
else
    echo "CẢNH BÁO: không thấy ${RESULTS_ROOT}/span_large_ablation/sr_span_large_*_seed*.json —"
    echo "          chạy RUN_ALL_span_large_ablation_new_backbones.sh $CONFIG trước để có đủ"
    echo "          Delta_SPAN đối chứng cho 7 backbone mới."
fi

echo ""
echo "################################################################"
echo "# HOÀN TẤT (arch: $ARCH, n_blocks_half: $N_HALF, 12 backbone)     #"
echo "################################################################"
echo "Tính kiểm định Δ_${ARCH} vs Δ_SPAN bằng:"
echo "  python data/compare_depth_deltas.py \\"
echo "      --results_dir ${RESULTS_DIR}/combined \\"
echo "      --arch_full_domain sr_improved_${ARCH} --arch_half_domain ${DOMAIN} \\"
echo "      --span_full_domain sr_span_large --span_tiny_domain sr_improved \\"
echo "      --out_csv ${RESULTS_DIR}/depth_delta_comparison_${ARCH}.csv"
