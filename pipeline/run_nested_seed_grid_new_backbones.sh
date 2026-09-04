#!/bin/bash
# [MỚI — mở rộng 7 backbone mới cho lưới nested SR-seed x downstream-seed,
# xem models/recognition_model.py] pipeline/run_nested_seed_grid.sh đã train
# đủ phần LÕI của lưới 5x5 (SR-seed x downstream-seed) cho 5 backbone gốc,
# dùng bởi data/analyze_nested_seed_variance.py để fit crossed mixed-effects
# model tách phương sai SR-seed / downstream-seed cho Table 3 (tab:main).
#
# Khi chạy data/analyze_nested_seed_variance.py với dữ liệu hiện có (không
# train gì thêm), 7 backbone mới chỉ có 9/25 ô (5 ô từ results/multi_seed/ +
# 4 ô từ results/sr_seed_variance_<domain>/) vì phần LÕI 4x4 = 16 ô chưa
# được train riêng cho chúng -- script này lấp đúng 16 ô còn thiếu đó, để cả
# 12 backbone đều có lưới đầy đủ 25/25 ô, cho độ chính xác ước lượng ngang
# 5 backbone gốc.
#
# KHÔNG train lại SR (ảnh SR-seed đã build sẵn từ
# pipeline/run_sr_seed_variance_new_backbones.sh) -- CHỈ train+eval
# recognition cho 16 ô lõi x 7 backbone mới x 2 domain (span_tiny,
# span_baseline).
#
# QUAN TRỌNG (giống bản gốc): fine-tune domain SR ở downstream-seed D phải
# khởi tạo từ checkpoint recognition_lr_<backbone>_seed<D>/best.pt (checkpoint
# LR của ĐÚNG seed D đó) -- các checkpoint này phải đã tồn tại từ
# pipeline/run_multi_seed_new_backbones.sh (n=5 seed, cả 7 backbone mới).
#
# DÙNG:
#   bash pipeline/run_nested_seed_grid_new_backbones.sh [đường_dẫn_config] [span_tiny|span_baseline]
#
# TIỀN ĐỀ:
#   - Đã chạy xong pipeline/run_sr_seed_variance_new_backbones.sh configs/config.yaml span_tiny
#     và ... span_baseline (ảnh SR-seed 123,2024,44,999 đã build, không phụ thuộc backbone,
#     dùng chung với bản gốc).
#   - Đã chạy xong pipeline/run_multi_seed_new_backbones.sh (checkpoint
#     recognition_lr_<backbone mới>_seed<seed>/best.pt, n=5 seed, cho cả 7
#     backbone mới).
#
# CẢNH BÁO THỜI GIAN: 16 ô x 7 backbone = 112 lần train+eval recognition MỖI
# domain -- chạy cho cả span_tiny và span_baseline (2 lần gọi script này) =
# 224 lần tổng cộng. Không cần train lại SR nào.

set -e

CONFIG="${1:-configs/config.yaml}"
CONDITION="${2:-span_tiny}"   # span_tiny | span_baseline -- ĐÚNG 2 domain trung tâm của Table 3
BACKBONES=("shufflenet_v2_x1_0" "squeezenet1_1" "mnasnet1_0" "mobilenet_v3_large" "regnet_y_400mf" "mobileone_s0" "lcnet_100")
SR_SEEDS=(123 2024 44 999)
DOWNSTREAM_SEEDS=(123 2024 44 999)
NUM_WORKERS="${NUM_WORKERS:-4}"

if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không thấy $CONFIG"
    exit 1
fi

case "$CONDITION" in
    span_tiny)
        DOMAIN_BASE="sr_improved"
        ;;
    span_baseline)
        DOMAIN_BASE="sr_baseline"
        ;;
    *)
        echo "LỖI: '$CONDITION' không hợp lệ -- phải là span_tiny|span_baseline (đúng 2 domain của Table 3)." >&2
        exit 1
        ;;
esac

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
SPLITS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['splits_root'])")
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")

RESULTS_DIR="${RESULTS_ROOT}/nested_seed_grid"
mkdir -p "$RESULTS_DIR"

echo "================================================================"
echo "Config     = $CONFIG"
echo "Condition  = $CONDITION (domain gốc: $DOMAIN_BASE)"
echo "SR seeds   = ${SR_SEEDS[*]}"
echo "Downstream seeds = ${DOWNSTREAM_SEEDS[*]}"
echo "Backbones MỚI = ${BACKBONES[*]}"
echo "Tổng lượt  = $((${#SR_SEEDS[@]} * ${#DOWNSTREAM_SEEDS[@]} * ${#BACKBONES[@]}))"
echo "Results dir = $RESULTS_DIR"
echo "================================================================"

echo "Kiểm tra tiền đề..."
MISSING=0
for SR_SEED in "${SR_SEEDS[@]}"; do
    if [ ! -d "${SPLITS_ROOT}/${DOMAIN_BASE}_srseed${SR_SEED}" ]; then
        echo "LỖI: thiếu ${SPLITS_ROOT}/${DOMAIN_BASE}_srseed${SR_SEED} -- chạy pipeline/run_sr_seed_variance_new_backbones.sh trước." >&2
        MISSING=1
    fi
done
for BACKBONE in "${BACKBONES[@]}"; do
    for D_SEED in "${DOWNSTREAM_SEEDS[@]}"; do
        CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${D_SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: thiếu $CKPT -- chạy pipeline/run_multi_seed_new_backbones.sh trước." >&2
            MISSING=1
        fi
    done
done
if [ "$MISSING" -eq 1 ]; then
    echo "DỪNG LẠI -- thiếu tiền đề ở trên." >&2
    exit 1
fi
echo "OK -- đủ tiền đề, bắt đầu chạy (không train lại SR)."

for SR_SEED in "${SR_SEEDS[@]}"; do
    SR_DOMAIN="${DOMAIN_BASE}_srseed${SR_SEED}"
    for D_SEED in "${DOWNSTREAM_SEEDS[@]}"; do
        for BACKBONE in "${BACKBONES[@]}"; do
            OUT_JSON="${RESULTS_DIR}/${CONDITION}_${BACKBONE}_srseed${SR_SEED}_dseed${D_SEED}.json"
            if [ -f "$OUT_JSON" ]; then
                echo ">>> Bỏ qua $CONDITION | backbone=$BACKBONE | SR-seed=$SR_SEED | d-seed=$D_SEED (đã có)"
                continue
            fi

            echo ""
            echo "################################################################"
            echo "# $CONDITION | backbone=$BACKBONE | SR-seed=$SR_SEED | d-seed=$D_SEED #"
            echo "################################################################"
            INIT_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${D_SEED}/best.pt"
            RUN_SUFFIX="_srseed${SR_SEED}_dseed${D_SEED}"

            python train_recognition.py --config "$CONFIG" --domain "$SR_DOMAIN" --backbone "$BACKBONE" \
                --init_ckpt "$INIT_CKPT" \
                --seed "$D_SEED" --run_suffix "$RUN_SUFFIX" --num_workers "$NUM_WORKERS"

            python eval_recognition.py --config "$CONFIG" \
                --ckpt "${RUNS_ROOT}/recognition_${SR_DOMAIN}_${BACKBONE}${RUN_SUFFIX}/best.pt" \
                --backbone "$BACKBONE" --train_domain "$SR_DOMAIN" --test_domain "$SR_DOMAIN" \
                --out_json "$OUT_JSON" --num_workers "$NUM_WORKERS"
        done
    done
done

echo ""
echo "HOÀN TẤT ($CONDITION). $((${#SR_SEEDS[@]} * ${#DOWNSTREAM_SEEDS[@]} * ${#BACKBONES[@]})) file JSON trong $RESULTS_DIR"
echo "Chạy tiếp cho domain còn lại (span_tiny/span_baseline), rồi:"
echo "  python data/analyze_nested_seed_variance.py --config $CONFIG"
