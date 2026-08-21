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
# [MỚI — bổ sung sau code review, đợt 10, điểm 3] Script này KHÔNG tự biết
# domain "sr_learned_prune" hiện có trên đĩa là model đã được sửa (identity-
# aware, đúng recipe pin mới) hay vẫn là model CŨ (đợt 8, 6 khối, không
# identity-aware) — nếu chạy nhầm trên model cũ, 10 JSON seed 44/999 sinh ra
# sẽ ứng với model INVALID, rồi bị trộn với 15 JSON seed 42/123/2024 (có thể
# đã được train lại đúng, hoặc cũng vẫn cũ) mà không có cảnh báo gì. Đọc trực
# tiếp prune_metadata.json (nguồn sự thật duy nhất, ghi lại lúc train SR) để
# xác nhận đây LÀ model identity-aware trước khi cho phép chạy tiếp.
if [ -f "runs/sr_learned_prune_final/prune_metadata.json" ]; then
    IS_IDENTITY_AWARE=$(python -c "
import json
m = json.load(open('runs/sr_learned_prune_final/prune_metadata.json'))
print('1' if m.get('identity_aware') else '0')
")
    N_BLOCKS_CHECK=$(python -c "import json; print(json.load(open('runs/sr_learned_prune_final/prune_metadata.json'))['n_blocks_kept'])")
    if [ "$IS_IDENTITY_AWARE" != "1" ]; then
        echo "LỖI: runs/sr_learned_prune_final/prune_metadata.json ghi identity_aware=false" >&2
        echo "     (n_blocks_kept=$N_BLOCKS_CHECK) — RẤT CÓ THỂ đây là model CŨ (đợt 8," >&2
        echo "     reconstruction prune, không identity-aware) chứ KHÔNG PHẢI model đã" >&2
        echo "     sửa (pin lambda_feat=0.5/lambda_identity=0.1). Chạy lại (bản mới)" >&2
        echo "     pipeline/run_multi_seed_learned_prune.sh trước khi thêm seed 44/999," >&2
        echo "     để không trộn 2 model khác nhau vào cùng bảng tổng hợp." >&2
        MISSING=1
    else
        echo "OK — prune_metadata.json xác nhận identity_aware=true (n_blocks_kept=$N_BLOCKS_CHECK)."
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
