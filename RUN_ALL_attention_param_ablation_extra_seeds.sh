#!/bin/bash
# [MỚI — Mục 5.7(i)] Mở rộng RUN_ALL_attention_param_ablation.sh từ n=3 lên
# n=5 seed (thêm 44, 999) — khớp quy ước n=5 dùng xuyên suốt project (xem
# RUN_ALL_extra_sr_baseline_distilled_extra_seeds.sh, mẫu gốc của script
# này). KHÔNG train lại SR half-depth (chỉ 1 lần, seed=42 mặc định, đã có
# sẵn từ RUN_ALL_attention_param_ablation.sh) — CHỈ thêm 2 seed cho bước
# train_recognition.py trên domain half-depth đã có sẵn.
#
# DÙNG (Y HỆT cú pháp bản gốc):
#   bash RUN_ALL_attention_param_ablation_extra_seeds.sh <safmn|smfanet> [n_blocks_half] [đường_dẫn_config]
#
# TIỀN ĐỀ:
#   - Đã chạy xong RUN_ALL_attention_param_ablation.sh <arch> [n_blocks_half] [config] (bản gốc, n=3)
#   - Đã chạy xong pipeline/run_multi_seed_extra_seeds.sh (đúng CONFIG này) —
#     cần checkpoint recognition_lr_<backbone>_seed{44,999}
#   - Đã chạy xong RUN_ALL_extra_sr_baseline_distilled_extra_seeds.sh <arch> [config] —
#     cần JSON full-depth ở seed 44/999 để Δ_<arch> đủ n=5 khớp phía half-depth

set -e

ARCH="$1"
N_HALF="${2:-4}"
CONFIG="${3:-configs/config.yaml}"

if [ -z "$ARCH" ] || [[ ! "$ARCH" =~ ^(safmn|smfanet)$ ]]; then
    echo "LỖI: thiếu hoặc sai tên kiến trúc."
    echo "Dùng: bash RUN_ALL_attention_param_ablation_extra_seeds.sh <safmn|smfanet> [n_blocks_half] [đường_dẫn_config]"
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
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(44 999)

echo "################################################################"
echo "# BƯỚC 0 — Kiểm tra tiền đề (arch=$ARCH, n_blocks_half=$N_HALF, config=$CONFIG) #"
echo "################################################################"
MISSING=0
if [ ! -f "${RUNS_ROOT}/${DOMAIN}/best.pt" ]; then
    echo "LỖI: chưa thấy ${RUNS_ROOT}/${DOMAIN}/best.pt"
    echo "     -> chạy 'bash RUN_ALL_attention_param_ablation.sh $ARCH $N_HALF $CONFIG' (bản gốc) trước."
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
# [SỬA — đồng bộ với RUN_ALL_attention_param_ablation.sh bản gốc] Δ_SPAN đối
# chứng PHẢI dùng domain "sr_span_large" (CÙNG recipe distillation), KHÔNG
# dùng "sr_baseline" — cần seed 44/999 của domain này (từ
# RUN_ALL_span_large_ablation_extra_seeds.sh) để khớp n=5 với phía half-depth.
if ! ls "${RESULTS_ROOT}/span_large_ablation"/sr_span_large_*_seed44.json >/dev/null 2>&1 \
   || ! ls "${RESULTS_ROOT}/span_large_ablation"/sr_span_large_*_seed999.json >/dev/null 2>&1; then
    echo "LỖI: chưa thấy đủ JSON seed 44/999 của domain 'sr_span_large' trong ${RESULTS_ROOT}/span_large_ablation/"
    echo "     -> chạy 'bash RUN_ALL_span_large_ablation_extra_seeds.sh $CONFIG' trước."
    MISSING=1
fi
if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "DỪNG LẠI — thiếu tiền đề ở trên."
    exit 1
fi
echo "OK — mọi tiền đề đã sẵn sàng."
mkdir -p "$RESULTS_DIR"

echo ""
echo "################################################################"
echo "# Multi-seed recognition (44, 999) trên domain ${DOMAIN}         #"
echo "################################################################"
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        LR_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        echo "### backbone=$BACKBONE | seed=$SEED | domain=${DOMAIN}"
        python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" \
            --backbone "$BACKBONE" --init_ckpt "$LR_CKPT" \
            --seed "$SEED" --run_suffix "_seed${SEED}"
        python eval_recognition.py --config "$CONFIG" \
            --ckpt "${RUNS_ROOT}/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
            --out_json "${RESULTS_DIR}/${DOMAIN}_${BACKBONE}_seed${SEED}.json"
    done
done

echo ""
echo "################################################################"
echo "# Gộp thêm JSON mới (44, 999) vào 'combined'                    #"
echo "################################################################"
mkdir -p "${RESULTS_DIR}/combined"
cp "${RESULTS_DIR}"/${DOMAIN}_*_seed*.json "${RESULTS_DIR}/combined/"
FULL_RESULTS_DIR="${RESULTS_ROOT}/extra_sr_baseline_${ARCH}_distilled"
if ls "${FULL_RESULTS_DIR}"/sr_improved_${ARCH}_*_seed*.json >/dev/null 2>&1; then
    cp "${FULL_RESULTS_DIR}"/sr_improved_${ARCH}_*_seed*.json "${RESULTS_DIR}/combined/"
fi
if ls "${RESULTS_ROOT}/multi_seed"/*_seed*.json >/dev/null 2>&1; then
    cp "${RESULTS_ROOT}/multi_seed"/*_seed*.json "${RESULTS_DIR}/combined/"
fi
# [SỬA — đồng bộ với bản gốc] domain "sr_span_large" (CÙNG recipe distillation)
# — BẮT BUỘC cho Δ_SPAN đối chứng công bằng, KHÔNG dùng "sr_baseline".
if ls "${RESULTS_ROOT}/span_large_ablation"/sr_span_large_*_seed*.json >/dev/null 2>&1; then
    cp "${RESULTS_ROOT}/span_large_ablation"/sr_span_large_*_seed*.json "${RESULTS_DIR}/combined/"
fi

echo ""
echo "################################################################"
echo "# HOÀN TẤT n=5 seed (arch: $ARCH, n_blocks_half: $N_HALF)         #"
echo "################################################################"
echo "Tính kiểm định Δ_${ARCH} vs Δ_SPAN bằng:"
echo "  python data/compare_depth_deltas.py \\"
echo "      --results_dir ${RESULTS_DIR}/combined \\"
echo "      --arch_full_domain sr_improved_${ARCH} --arch_half_domain ${DOMAIN} \\"
echo "      --span_full_domain sr_span_large --span_tiny_domain sr_improved \\"
echo "      --out_csv ${RESULTS_DIR}/depth_delta_comparison_${ARCH}.csv"
