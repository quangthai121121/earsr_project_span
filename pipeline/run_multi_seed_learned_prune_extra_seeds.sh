#!/bin/bash
# [MỚI — bổ sung sau đánh giá kết quả thật, đợt 9] Chạy THÊM 2 seed (44, 999)
# cho domain "sr_learned_prune" (learned/differentiable block pruning), để
# cộng với 3 seed đã có (42, 123, 2024 — từ
# pipeline/run_multi_seed_learned_prune.sh) thành ĐỦ 5 SEED, giống hệt lý do
# đã áp dụng cho 3 domain chính ở pipeline/run_multi_seed_extra_seeds.sh.
#
# CHỈ CHẠY SAU KHI đã chạy LẠI (bản sửa mới, pin lambda_feat=0.5/identity=0.1
# + lambda_sparsity cho ra số khối gần 3 nhất) pipeline/run_prune_sparsity_screen.sh
# và pipeline/run_multi_seed_learned_prune.sh — script này KHÔNG train lại SR,
# chỉ thêm seed ở bước recognition trên domain đã có sẵn từ lần chạy đó.
#
# KHÔNG train lại SR — TÁI SỬ DỤNG checkpoint runs/sr_learned_prune_final/best.pt
# và ảnh đã sinh sẵn ở splits/sr_learned_prune/ — CHỈ multi-seed ở bước
# train_recognition.py.
#
# YÊU CẦU:
#   1. Đã chạy xong (bản sửa mới) pipeline/run_multi_seed_learned_prune.sh
#      (có sẵn splits/sr_learned_prune/ và 3 seed 42/123/2024 trong
#      results/multi_seed_learned_prune/).
#   2. Đã chạy xong pipeline/run_multi_seed_extra_seeds.sh (có sẵn checkpoint
#      recognition_lr_<backbone>_seed{44,999}/best.pt cho cả 5 backbone).
#
# CẢNH BÁO THỜI GIAN: 5 backbone x 2 seed = 10 lần train recognition (nhẹ,
# không train lại SR).

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results/multi_seed_learned_prune"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(44 999)
DOMAIN="sr_learned_prune"
mkdir -p "$RESULTS_DIR"

echo "Kiểm tra tiền đề..."
MISSING=0
# [SỬA — bổ sung sau code review, đợt 10] "-d" chỉ kiểm tra thư mục TỒN TẠI,
# không kiểm tra CÓ ẢNH thật hay không — kiểm tra có ít nhất 1 file bên trong.
if [ ! -d "splits/${DOMAIN}" ] || [ -z "$(find "splits/${DOMAIN}" -type f -print -quit 2>/dev/null)" ]; then
    echo "LỖI: splits/${DOMAIN} không tồn tại hoặc rỗng (không có file ảnh nào)." >&2
    echo "     -> chạy pipeline/run_multi_seed_learned_prune.sh trước (Bước 1/2, sinh SR)." >&2
    MISSING=1
fi
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: thiếu $CKPT -> chạy pipeline/run_multi_seed_extra_seeds.sh trước." >&2
            MISSING=1
        fi
    done
done
# [SỬA — 2026-08-29] Chốt kiểm tra CŨ (đợt 10) yêu cầu identity_aware=true,
# dựa trên niềm tin "cấu hình identity-aware (feat=0.5/identity=0.1) là đúng"
# — niềm tin đó đã bị BÁC BỎ DỨT ĐIỂM bằng multi-seed thật (xem
# pipeline/run_prune_sparsity_screen.sh, cùng ngày sửa). Recipe ĐÚNG hiện tại
# (feat=0/identity=0/saliency=0, khớp span_tiny) tất yếu cho identity_aware=
# FALSE — đây là kỳ vọng ĐÚNG, không phải dấu hiệu model cũ/hỏng. Đảo lại
# chiều kiểm tra: xác nhận model hiện tại ĐÚNG recipe sạch (identity_aware=
# false, uses_feature_kd=false) VÀ đúng n_blocks_kept=3 (khớp
# LAMBDA_SPARSITY=0.095 đã chốt trong run_multi_seed_learned_prune.sh) —
# vẫn giữ tinh thần gốc: phát hiện sớm nếu model trên đĩa là 1 lần chạy
# KHÁC (recipe khác hoặc lambda_sparsity khác) trước khi trộn nhầm JSON.
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
        echo "     -> Chạy lại pipeline/run_multi_seed_learned_prune.sh (bản hiện tại, đã pin" >&2
        echo "        LAMBDA_SPARSITY=0.095/feat=0/identity=0) trước khi thêm seed 44/999," >&2
        echo "     để không trộn 2 model khác nhau vào cùng bảng tổng hợp." >&2
        MISSING=1
    else
        echo "OK — prune_metadata.json khớp đúng recipe sạch (n_blocks_kept=3, identity_aware=false, uses_feature_kd=false)."
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

for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "----------------------------------------------------------------"
        echo "backbone=$BACKBONE | seed=$SEED | domain=$DOMAIN"
        echo "----------------------------------------------------------------"
        LR_CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" --backbone "$BACKBONE" \
            --init_ckpt "$LR_CKPT" --seed "$SEED" --run_suffix "_seed${SEED}"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
            --out_json "$RESULTS_DIR/${DOMAIN}_${BACKBONE}_seed${SEED}.json"
    done
done

echo ""
echo ">>> Tổng hợp lại — giờ đọc ĐỦ 5 seed (42,123,2024,44,999) cho domain $DOMAIN..."
echo ">>> (Yêu cầu: $RESULTS_DIR/ đã có sẵn 15 file *_seed{42,123,2024}.json từ"
echo ">>> pipeline/run_multi_seed_learned_prune.sh — script này CHỈ thêm 10 file"
echo ">>> *_seed{44,999}.json, không xoá/ghi đè file cũ. Nếu muốn so lại với"
echo ">>> 'sr_improved' (đủ 5 seed), copy JSON tương ứng từ results/multi_seed/"
echo ">>> vào $RESULTS_DIR/ trước khi chạy aggregate.)"
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/multi_seed_learned_prune_summary.csv"

echo ""
echo "HOÀN TẤT. Kết quả (giờ dựa trên n=5 seed): $RESULTS_DIR/multi_seed_learned_prune_summary.csv"
echo "và $RESULTS_DIR/multi_seed_learned_prune_summary_pairwise.csv."
