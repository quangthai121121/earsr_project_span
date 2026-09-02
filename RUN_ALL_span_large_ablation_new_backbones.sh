#!/bin/bash
# [MỚI — mở rộng 7 backbone mới cho depth ablation (Table 4, span_tiny vs.\
# span_large), xem models/recognition_model.py] RUN_ALL_span_large_ablation.sh
# (+ _extra_seeds.sh) đã kiểm chứng trên 5 backbone gốc — đây là 1 trong 2
# bảng kết quả TRUNG TÂM của bài báo (cùng tier với bảng so sánh chính), nên
# PHẢI được mở rộng cùng lúc với bảng chính, không phải một mục phụ.
#
# PHIÊN BẢN THAM SỐ HÓA — dùng CHUNG cho EarVN1.0 VÀ AWEx (giống hệt cách
# RUN_ALL_span_large_ablation.sh gốc đã làm — mọi đường dẫn suy ra TỪ CHÍNH
# CONFIG được truyền vào, không hardcode "runs/"/"results/").
#
# KHÔNG train lại SR span_large (checkpoint không phụ thuộc backbone, đã có
# sẵn từ RUN_ALL_span_large_ablation.sh chạy 1 lần cho đúng CONFIG này) — CHỈ
# thêm multi-seed recognition trên domain sr_span_large cho 7 backbone MỚI,
# thẳng n=5 seed (không qua bước n=3 trung gian, giống lý do đã nêu ở các
# script new_backbones khác).
#
# Bước tổng hợp cuối cùng gộp CHUNG với dữ liệu multi-seed chính (results*/
# multi_seed/*_seed*.json — đã có cả 5 backbone gốc VÀ 7 backbone mới nếu đã
# chạy pipeline/run_multi_seed_new_backbones.sh[_awex.sh] trước) — không cần
# sửa gì thêm, bảng tổng hợp sẽ tự đủ 12 backbone.
#
# DÙNG:
#   bash RUN_ALL_span_large_ablation_new_backbones.sh [đường_dẫn_config]
#   Không truyền gì -> mặc định configs/config.yaml (EarVN1.0)
#   Cho AWEx: bash RUN_ALL_span_large_ablation_new_backbones.sh configs/config_awex.yaml
#
# TIỀN ĐỀ:
#   - Đã chạy xong RUN_ALL_span_large_ablation.sh [CONFIG] (checkpoint
#     sr_improved_span_large + ảnh splits*/sr_span_large đã có, không phụ
#     thuộc backbone).
#   - Đã chạy xong pipeline/run_multi_seed_new_backbones.sh (EarVN1.0) hoặc
#     pipeline/run_multi_seed_new_backbones_awex.sh (AWEx) — cần checkpoint
#     recognition_lr_<backbone mới>_seed<seed>/best.pt cho cả 7 backbone mới.
#
# CẢNH BÁO THỜI GIAN: 7 backbone x 5 seed = 35 lần train+eval recognition
# (không train lại SR).

set -e

CONFIG="${1:-configs/config.yaml}"

if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không tìm thấy config '$CONFIG'."
    echo "Dùng: bash RUN_ALL_span_large_ablation_new_backbones.sh [đường_dẫn_config]"
    exit 1
fi

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")
SPLITS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['splits_root'])")

RESULTS_DIR="${RESULTS_ROOT}/span_large_ablation"
NEW_BACKBONES=("shufflenet_v2_x1_0" "squeezenet1_1" "mnasnet1_0" "mobilenet_v3_large" "regnet_y_400mf" "mobileone_s0" "lcnet_100")
SEEDS=(42 123 2024 44 999)

echo "================================================================"
echo "CONFIG        = $CONFIG"
echo "runs_root     = $RUNS_ROOT"
echo "results_root  = $RESULTS_ROOT"
echo "splits_root   = $SPLITS_ROOT"
echo "Backbones MỚI = ${NEW_BACKBONES[*]}"
echo "================================================================"

echo ""
echo "################################################################"
echo "# BƯỚC 0 — Kiểm tra tiền đề                                     #"
echo "################################################################"
MISSING=0
if [ ! -f "${RUNS_ROOT}/sr_improved_span_large/best.pt" ]; then
    echo "LỖI: chưa thấy ${RUNS_ROOT}/sr_improved_span_large/best.pt"
    echo "     -> chạy 'bash RUN_ALL_span_large_ablation.sh $CONFIG' (bản gốc) trước."
    MISSING=1
fi
if [ ! -d "${SPLITS_ROOT}/sr_span_large" ] || [ -z "$(find "${SPLITS_ROOT}/sr_span_large" -type f -print -quit 2>/dev/null)" ]; then
    echo "LỖI: ${SPLITS_ROOT}/sr_span_large không tồn tại hoặc rỗng."
    echo "     -> chạy 'bash RUN_ALL_span_large_ablation.sh $CONFIG' (bản gốc) trước."
    MISSING=1
fi
for BACKBONE in "${NEW_BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: chưa thấy $CKPT"
            echo "     -> chạy pipeline/run_multi_seed_new_backbones.sh (hoặc _awex.sh cho AWEx)"
            echo "        với CONFIG=$CONFIG trước."
            MISSING=1
        fi
    done
done
if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "DỪNG LẠI — thiếu tiền đề ở trên."
    exit 1
fi
echo "OK — mọi tiền đề đã sẵn sàng, bắt đầu chạy (không train lại SR)."
mkdir -p "$RESULTS_DIR"

echo ""
echo "################################################################"
echo "# Multi-seed recognition trên domain sr_span_large (7 backbone MỚI x 5 seed) #"
echo "################################################################"
for BACKBONE in "${NEW_BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUT_JSON="${RESULTS_DIR}/sr_span_large_${BACKBONE}_seed${SEED}.json"
        if [ -f "$OUT_JSON" ]; then
            echo ">>> Bỏ qua backbone=$BACKBONE seed=$SEED (đã có JSON)"
            continue
        fi

        LR_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"

        echo "### backbone=$BACKBONE | seed=$SEED | domain=sr_span_large"
        python train_recognition.py --config "$CONFIG" --domain sr_span_large \
            --backbone "$BACKBONE" --init_ckpt "$LR_CKPT" \
            --seed "$SEED" --run_suffix "_seed${SEED}"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "${RUNS_ROOT}/recognition_sr_span_large_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain sr_span_large --test_domain sr_span_large \
            --out_json "$OUT_JSON"
    done
done

echo ""
echo "################################################################"
echo "# Tổng hợp lại (giờ đủ 12 backbone: 5 gốc + 7 mới)               #"
echo "################################################################"
rm -rf "${RESULTS_DIR}/combined"
mkdir -p "${RESULTS_DIR}/combined"
if ls "${RESULTS_ROOT}/multi_seed"/*_seed*.json >/dev/null 2>&1; then
    cp "${RESULTS_ROOT}/multi_seed"/*_seed*.json "${RESULTS_DIR}/combined/"
else
    echo "CẢNH BÁO: không thấy ${RESULTS_ROOT}/multi_seed/*_seed*.json — chỉ tổng hợp được domain sr_span_large."
fi
cp "${RESULTS_DIR}"/sr_span_large_*_seed*.json "${RESULTS_DIR}/combined/"

python data/aggregate_multi_seed_results.py --results_dir "${RESULTS_DIR}/combined" \
    --out_csv "${RESULTS_DIR}/multi_seed_summary_4domains.csv"

echo ""
echo "################################################################"
echo "# HOÀN TẤT (config: $CONFIG, đủ 12 backbone)                    #"
echo "################################################################"
echo "Kết quả:"
echo "  - ${RESULTS_DIR}/multi_seed_summary_4domains.csv"
echo "  - ${RESULTS_DIR}/multi_seed_summary_4domains_pairwise.csv (lr/sr_baseline/sr_improved/sr_span_large, 12 backbone x 5 seed)"
