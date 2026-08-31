#!/bin/bash
# [MỚI — mở rộng theo phản biện] Table 3 (tab:main) so sánh span_tiny vs.
# span_baseline (vs.\ lr) ở n=5 downstream-seed nhưng CHỈ 1 SR-seed (42).
# Section~sec:sr-seed-variance đo phương sai do SR-seed riêng (1 SR-seed cố
# định downstream=42, holding backbone=mobilenet_v2 rồi mở rộng 5 backbone
# qua run_sr_seed_variance_backbones.sh) nhưng KHÔNG cho phép tách 2 nguồn
# phương sai (SR-seed, downstream-seed) một cách đồng thời/tường minh —
# reviewer yêu cầu: thiết kế NESTED (SR-seed x downstream-seed) + mixed-
# effects model cho ĐÚNG 2 domain trung tâm của Table 3 (span_tiny,
# span_baseline), cả 5 backbone.
#
# [SỬA — mở rộng theo yêu cầu, sau khi phát hiện qua mô phỏng] Bản đầu chỉ
# dùng SR-seed in {123,2024} (2 mức) cho phần lõi lưới -- mô phỏng lặp lại 30
# lần (không phải chạy thật, chỉ kiểm chứng phương pháp) cho thấy với chỉ 2-3
# mức SR-seed/condition, mixed-effects model có 30/30 lần gặp cảnh báo
# boundary/convergence VÀ ~13% khả năng báo SAI HƯỚNG phương sai chỉ vì nhiễu
# mẫu nhỏ -- không phải lỗi code, mà là giới hạn cỡ mẫu của thiết kế. Để giảm
# rủi ro này VÀ đảm bảo công bằng giữa span_tiny/span_baseline/span_large
# (trước đây chỉ span_baseline có n=5 SR-seed), cả 3 kiến trúc giờ đều mở
# rộng lên n=5 SR-seed (42,123,2024,44,999) qua
# pipeline/run_sr_seed_variance_extra_seeds.sh -- lưới lõi ở đây giờ dùng cả
# 4 SR-seed non-42 thay vì 2, tăng số nhóm ngẫu nhiên cho SR-seed từ 2-3 lên
# 4-5/condition, giảm đáng kể độ nhiễu ước lượng.
#
# Ma trận đầy đủ cần: SR-seed in {42,123,2024,44,999} x downstream-seed in
# {42,123,2024,44,999} = 25 ô/(domain x backbone). ĐÃ CÓ SẴN (không train gì
# thêm):
#   - hàng SR-seed=42 (5 downstream-seed)  -> results/multi_seed/*.json
#     (từ pipeline/run_multi_seed.sh + _extra_seeds.sh, đã dùng cho Table 3)
#   - cột downstream-seed=42 (SR-seed 123,2024,44,999) -> results/sr_seed_variance_<domain>/*.json
#     (từ pipeline/run_sr_seed_variance.sh + _extra_seeds.sh + run_sr_seed_variance_backbones.sh)
# CÒN THIẾU đúng phần LÕI của lưới: SR-seed in {123,2024,44,999} x
# downstream-seed in {123,2024,44,999} = 16 ô/(domain x backbone). Script này
# CHỈ train 16 ô đó (KHÔNG train lại SR — dùng ảnh SR-seed đã build sẵn).
#
# QUAN TRỌNG (xem pipeline/run_multi_seed.sh): fine-tune domain SR ở
# downstream-seed D phải khởi tạo từ checkpoint recognition_lr_<backbone>_seed<D>/best.pt
# (checkpoint LR của ĐÚNG seed D đó) -- các checkpoint này PHẢI đã tồn tại từ
# pipeline/run_multi_seed.sh (+ _extra_seeds.sh cho seed 44,999).
#
# DÙNG:
#   bash pipeline/run_nested_seed_grid.sh [đường_dẫn_config] [span_tiny|span_baseline]
#
# CẢNH BÁO THỜI GIAN: 16 ô x 5 backbone = 80 lần train+eval recognition MỖI
# domain -- chạy cho cả span_tiny và span_baseline (2 lần gọi script này) =
# 160 lần tổng cộng. Không cần train lại SR nào.

set -e

CONFIG="${1:-configs/config.yaml}"
CONDITION="${2:-span_tiny}"   # span_tiny | span_baseline -- ĐÚNG 2 domain trung tâm của Table 3
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
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
echo "Backbones  = ${BACKBONES[*]}"
echo "Tổng lượt  = $((${#SR_SEEDS[@]} * ${#DOWNSTREAM_SEEDS[@]} * ${#BACKBONES[@]}))"
echo "Results dir = $RESULTS_DIR"
echo "================================================================"

echo "Kiểm tra tiền đề..."
MISSING=0
for SR_SEED in "${SR_SEEDS[@]}"; do
    if [ ! -d "${SPLITS_ROOT}/${DOMAIN_BASE}_srseed${SR_SEED}" ]; then
        echo "LỖI: thiếu ${SPLITS_ROOT}/${DOMAIN_BASE}_srseed${SR_SEED} -- chạy pipeline/run_sr_seed_variance.sh (+ _extra_seeds.sh) trước."
        MISSING=1
    fi
done
for BACKBONE in "${BACKBONES[@]}"; do
    for D_SEED in "${DOWNSTREAM_SEEDS[@]}"; do
        CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${D_SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: thiếu $CKPT -- chạy pipeline/run_multi_seed.sh (+ _extra_seeds.sh) trước."
            MISSING=1
        fi
    done
done
if [ "$MISSING" -eq 1 ]; then
    echo "DỪNG LẠI -- thiếu tiền đề ở trên."
    exit 1
fi
echo "OK -- đủ tiền đề, bắt đầu chạy."

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
