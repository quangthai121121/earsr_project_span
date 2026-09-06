#!/bin/bash
# [MỚI — Trụ cột 2, reviewer round 2] pipeline/run_depth_sweep.sh chỉ chạy
# depth sweep (1,2,4,5 khối) trên MỘT backbone đại diện (mobilenet_v2).
# Reviewer chỉ ra: không đủ để khái quát hoá "nén được tới đâu" cho MỌI
# backbone -- cần mở rộng sang vài backbone khác nhau về họ kiến trúc.
#
# Chọn 4 backbone bổ sung, cùng mobilenet_v2 hiện có tạo thành 5 backbone
# trải rộng nhiều họ kiến trúc: resnet18 (residual chuẩn), efficientnet_b0
# (compound scaling), ghostnet_100 (ghost convolution), shufflenet_v2_x1_0
# (channel shuffle). 3 trong 4 backbone này (resnet18, efficientnet_b0,
# ghostnet_100) thuộc nhóm 5 backbone dùng từ đầu dự án; shufflenet_v2_x1_0
# thuộc nhóm 6 backbone thêm sau -- checkpoint recognition_lr_<backbone>_
# seed<seed> PHẢI đã có sẵn cho cả 4, không phân biệt nhóm.
#
# KHÔNG train lại SR: 4 domain ảnh sr_depth1/2/4/5 đã build sẵn từ
# pipeline/run_depth_sweep.sh, KHÔNG phụ thuộc backbone nhận dạng nào dùng
# nó (ảnh SR chỉ phụ thuộc n_blocks). Depth 3 (span_tiny, domain sr_improved)
# và depth 6 (span_large, domain sr_span_large) cũng ĐÃ CÓ SẴN cho toàn bộ
# 11 backbone từ lâu (pipeline/run_multi_seed*.sh và RUN_ALL_span_large_
# ablation*.sh) -- KHÔNG cần train gì thêm cho 2 điểm này, chỉ cần
# data/aggregate_depth_sweep.py tự đọc lại. Script này CHỈ train+eval
# recognition cho 4 backbone MỚI trên 4 domain sr_depth{1,2,4,5} đã có sẵn.
# Không gọi lại eval_sr_quality (PSNR/SSIM không phụ thuộc backbone, đã có
# trong results/depth_sweep/sr_quality_depth_sweep.csv).
#
# DÙNG:
#   bash pipeline/run_depth_sweep_new_backbones.sh [đường_dẫn_config]
#
# TIỀN ĐỀ: đã chạy xong pipeline/run_depth_sweep.sh (4 domain ảnh sr_depth1/
# 2/4/5 đã build) VÀ có checkpoint recognition_lr_<backbone>_seed<seed> cho
# cả 4 backbone mới, n=5 seed (từ pipeline/run_multi_seed.sh hoặc
# pipeline/run_multi_seed_new_backbones.sh, tuỳ backbone).
#
# CẢNH BÁO THỜI GIAN: 4 backbone x 4 depth x 5 seed = 80 lần train+eval
# recognition (không train SR, không build lại ảnh).

set -e

CONFIG="${1:-configs/config.yaml}"
NEW_BACKBONES=("resnet18" "efficientnet_b0" "ghostnet_100" "shufflenet_v2_x1_0")
DEPTHS=(1 2 4 5)
SEEDS=(42 123 2024 44 999)
NUM_WORKERS="${NUM_WORKERS:-4}"

if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không thấy $CONFIG"
    exit 1
fi

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
SPLITS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['splits_root'])")
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")

RESULTS_DIR="${RESULTS_ROOT}/depth_sweep"
mkdir -p "$RESULTS_DIR"

echo "################################################################"
echo "# Kiểm tra tiền đề"
echo "################################################################"
MISSING=0
for DEPTH in "${DEPTHS[@]}"; do
    if [ ! -d "${SPLITS_ROOT}/sr_depth${DEPTH}" ]; then
        echo "LỖI: thiếu ${SPLITS_ROOT}/sr_depth${DEPTH} -> chạy pipeline/run_depth_sweep.sh trước."
        MISSING=1
    fi
done
for BACKBONE in "${NEW_BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: thiếu $CKPT -> chạy pipeline/run_multi_seed.sh (hoặc _new_backbones.sh) trước."
            MISSING=1
        fi
    done
done
if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "DỪNG LẠI — thiếu tiền đề ở trên."
    exit 1
fi
echo "OK — mọi tiền đề đã sẵn sàng (dùng lại 4 domain ảnh SR đã có, không train lại SR)."

for DEPTH in "${DEPTHS[@]}"; do
    DOMAIN_NAME="sr_depth${DEPTH}"
    for BACKBONE in "${NEW_BACKBONES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            OUT_JSON="${RESULTS_DIR}/${DOMAIN_NAME}_${BACKBONE}_seed${SEED}.json"
            if [ -f "$OUT_JSON" ]; then
                echo ">>> Bỏ qua depth=$DEPTH backbone=$BACKBONE seed=$SEED (đã có JSON)"
                continue
            fi

            echo "################################################################"
            echo "# depth=$DEPTH | backbone=$BACKBONE | seed=$SEED"
            echo "################################################################"

            INIT_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
            RUN_SUFFIX_REC="_seed${SEED}"
            python train_recognition.py --config "$CONFIG" --domain "$DOMAIN_NAME" --backbone "$BACKBONE" \
                --init_ckpt "$INIT_CKPT" \
                --seed "$SEED" --run_suffix "$RUN_SUFFIX_REC" --num_workers "$NUM_WORKERS"

            python eval_recognition.py --config "$CONFIG" \
                --ckpt "${RUNS_ROOT}/recognition_${DOMAIN_NAME}_${BACKBONE}${RUN_SUFFIX_REC}/best.pt" \
                --backbone "$BACKBONE" --train_domain "$DOMAIN_NAME" --test_domain "$DOMAIN_NAME" \
                --out_json "$OUT_JSON" --num_workers "$NUM_WORKERS"
        done
    done
done

echo ""
echo ">>> Tổng hợp lại (giờ đủ 5 backbone cho depth sweep: mobilenet_v2 (gốc) + resnet18 +"
echo "    efficientnet_b0 + ghostnet_100 + shufflenet_v2_x1_0)..."
python data/aggregate_depth_sweep.py --config "$CONFIG" \
    --depth_sweep_dir "$RESULTS_DIR" \
    --out_csv "${RESULTS_DIR}/depth_sweep_summary.csv"

echo ""
echo "HOÀN TẤT. Đọc ${RESULTS_DIR}/depth_sweep_summary.csv và "
echo "${RESULTS_DIR}/depth_sweep_summary_pairwise.csv -- giờ có 1 dòng/nhóm dòng cho MỖI"
echo "trong 5 backbone, không chỉ mobilenet_v2."
