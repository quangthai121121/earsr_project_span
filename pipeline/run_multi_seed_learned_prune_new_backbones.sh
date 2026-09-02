#!/bin/bash
# [MỚI — mở rộng 7 backbone mới cho learned/differentiable block pruning
# ablation (tab:pruning-ablation), xem models/recognition_model.py]
# pipeline/run_multi_seed_learned_prune.sh + _extra_seeds.sh đã kiểm chứng
# trên 5 backbone gốc (kết luận: pruning học được kém hơn ĐÁNG KỂ so với
# chọn khối cố định của span_tiny, trên cả 5/5 backbone). Mở rộng cùng lúc
# với các "negative-result ablation" khác để nhất quán.
#
# KHÔNG train lại SR (checkpoint runs/sr_learned_prune_final/best.pt và ảnh
# splits/sr_learned_prune/ không phụ thuộc backbone, đã có sẵn) — CHỈ thêm
# multi-seed recognition cho 7 backbone MỚI, thẳng n=5 seed.
#
# DÙNG:
#   bash pipeline/run_multi_seed_learned_prune_new_backbones.sh [đường_dẫn_config]
#
# TIỀN ĐỀ:
#   - Đã chạy xong (bản recipe sạch, đã chốt) pipeline/run_multi_seed_learned_prune.sh
#     (ảnh splits/sr_learned_prune/ đã có, không phụ thuộc backbone).
#   - Đã chạy xong pipeline/run_multi_seed_new_backbones.sh (checkpoint
#     recognition_lr_<backbone mới>_seed<seed>/best.pt, n=5, cho cả 7
#     backbone mới).
#
# CẢNH BÁO THỜI GIAN: 7 backbone x 5 seed = 35 lần train+eval recognition
# (không train lại SR).

set -e

CONFIG="${1:-configs/config.yaml}"
RESULTS_DIR="results/multi_seed_learned_prune"
NEW_BACKBONES=("shufflenet_v2_x1_0" "squeezenet1_1" "mnasnet1_0" "mobilenet_v3_large" "regnet_y_400mf" "mobileone_s0" "lcnet_100")
SEEDS=(42 123 2024 44 999)
DOMAIN="sr_learned_prune"
mkdir -p "$RESULTS_DIR"

echo "Kiểm tra tiền đề..."
MISSING=0
if [ ! -d "splits/${DOMAIN}" ] || [ -z "$(find "splits/${DOMAIN}" -type f -print -quit 2>/dev/null)" ]; then
    echo "LỖI: splits/${DOMAIN} không tồn tại hoặc rỗng." >&2
    echo "     -> chạy pipeline/run_multi_seed_learned_prune.sh trước (sinh SR)." >&2
    MISSING=1
fi
for BACKBONE in "${NEW_BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: thiếu $CKPT -> chạy pipeline/run_multi_seed_new_backbones.sh trước." >&2
            MISSING=1
        fi
    done
done
# [Giống pipeline/run_multi_seed_learned_prune_extra_seeds.sh] Xác nhận
# checkpoint trên đĩa đúng recipe sạch đã chốt (identity_aware=false,
# uses_feature_kd=false, n_blocks_kept=3) trước khi trộn vào cùng bảng tổng
# hợp với dữ liệu 5 backbone gốc — tránh trộn nhầm 2 lần chạy khác recipe.
if [ -f "runs/sr_learned_prune_final/prune_metadata.json" ]; then
    METADATA_CHECK=$(python -c "
import json
m = json.load(open('runs/sr_learned_prune_final/prune_metadata.json'))
identity_aware = bool(m.get('identity_aware'))
uses_feature_kd = bool(m.get('uses_feature_kd'))
n_blocks = m.get('n_blocks_kept')
ok = (not identity_aware) and (not uses_feature_kd) and (n_blocks == 3)
print(f'{1 if ok else 0}|{identity_aware}|{uses_feature_kd}|{n_blocks}')
")
    IFS='|' read -r METADATA_OK META_IA META_FKD META_NBLOCKS <<< "$METADATA_CHECK"
    if [ "$METADATA_OK" != "1" ]; then
        echo "LỖI: runs/sr_learned_prune_final/prune_metadata.json không khớp recipe SẠCH" >&2
        echo "     mong đợi (identity_aware=false, uses_feature_kd=false, n_blocks_kept=3)." >&2
        echo "     Hiện tại: identity_aware=$META_IA, uses_feature_kd=$META_FKD, n_blocks_kept=$META_NBLOCKS." >&2
        MISSING=1
    else
        echo "OK — prune_metadata.json khớp đúng recipe sạch."
    fi
else
    echo "LỖI: chưa thấy runs/sr_learned_prune_final/prune_metadata.json." >&2
    MISSING=1
fi
if [ "$MISSING" -eq 1 ]; then
    echo "DỪNG LẠI — thiếu tiền đề ở trên." >&2
    exit 1
fi
echo "OK — đủ tiền đề, bắt đầu chạy (không train lại SR)."

for BACKBONE in "${NEW_BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUT_JSON="$RESULTS_DIR/${DOMAIN}_${BACKBONE}_seed${SEED}.json"
        if [ -f "$OUT_JSON" ]; then
            echo ">>> Bỏ qua backbone=$BACKBONE seed=$SEED (đã có JSON)"
            continue
        fi

        echo "----------------------------------------------------------------"
        echo "backbone=$BACKBONE | seed=$SEED | domain=$DOMAIN"
        echo "----------------------------------------------------------------"
        LR_CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" --backbone "$BACKBONE" \
            --init_ckpt "$LR_CKPT" --seed "$SEED" --run_suffix "_seed${SEED}"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
            --out_json "$OUT_JSON"
    done
done

echo ""
echo ">>> Tổng hợp lại (giờ đủ 12 backbone: 5 gốc + 7 mới)..."
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/multi_seed_learned_prune_summary.csv"

echo ""
echo "HOÀN TẤT. Kết quả: $RESULTS_DIR/multi_seed_learned_prune_summary.csv"
echo "và $RESULTS_DIR/multi_seed_learned_prune_summary_pairwise.csv."
