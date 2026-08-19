#!/bin/bash
# [MỚI] Validate ĐẦY ĐỦ multi-seed x 5-backbone cho học pruning độ sâu
# (learned/differentiable block pruning, xem train_sr_learned_prune.py,
# models/sr_models.py::SPANLearnedPrune). Đây là BƯỚC BẮT BUỘC sau khi
# pipeline/run_prune_sparsity_screen.sh (1 backbone, 1 seed/mức — chỉ tín
# hiệu sàng lọc) đã xác định được lambda_sparsity thắng, trước khi đưa số
# liệu vào bảng kết quả chính thức của bài báo.
#
# **SỬA THAM SỐ NÀY** trước khi chạy, khớp đúng mức lambda_sparsity đã THẮNG
# ở run_prune_sparsity_screen.sh:
LAMBDA_SPARSITY=0.05

# [SỬA — bổ sung sau code review, vòng 2, điểm 3; LÀM RÕ THÊM ở vòng 5]
# **SỬA 3 THAM SỐ NÀY** khớp ĐÚNG cấu hình (lambda_feat/saliency/identity) đã
# dùng khi chạy run_prune_sparsity_screen.sh để chọn LAMBDA_SPARSITY ở trên —
# PHẢI NHẤT QUÁN giữa 2 bước (screen -> validate), nếu không kết quả
# multi-seed sẽ không phản ánh đúng cấu hình đã sàng lọc. Mặc định để 0 (TẮT
# cả 3). LƯU Ý: LAMBDA_FEAT KHÔNG phải tín hiệu nhận dạng (chỉ so khớp
# feature SR teacher) — chỉ LAMBDA_SALIENCY/LAMBDA_IDENTITY mới quyết định
# cờ "identity-aware" (xem prune_metadata.json::identity_aware, tách riêng
# khỏi uses_feature_kd). Nếu chưa sửa LAMBDA_SALIENCY/LAMBDA_IDENTITY, đây
# CHỈ là "reconstruction-aware pruning" (có hoặc không kèm feature-KD tùy
# LAMBDA_FEAT), KHÔNG phải "identity-aware pruning" như tên gọi/docstring mô
# tả (train_sr_learned_prune.py tự cảnh báo runtime khi 2 giá trị
# saliency/identity =0, không phụ thuộc LAMBDA_FEAT).
LAMBDA_FEAT=0.0
LAMBDA_SALIENCY=0.0
LAMBDA_IDENTITY=0.0

# SR model (pruning) CHỈ train 1 LẦN ở seed cố định — ĐÚNG protocol thống kê
# đã dùng xuyên suốt project (train SR nhiều seed bị đánh giá không khả thi
# về chi phí, chỉ downstream recognition được lặp lại qua seed để đo phương sai).
SR_SEED=42

CONFIG="configs/config.yaml"
RESULTS_DIR="results/multi_seed_learned_prune"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(42 123 2024)   # đổi/thêm seed nếu muốn khớp n=5 như bản thảo bài báo (thêm 44, 999)
DOMAIN="sr_learned_prune"

# YÊU CẦU: đã chạy xong pipeline/run_multi_seed.sh trước đó — script này TÁI
# SỬ DỤNG checkpoint recognition_lr_<backbone>_seed<seed> đã train sẵn cho
# từng seed, ĐÚNG chuỗi fine-tune hr -> lr -> sr_learned_prune (không dùng
# chung 1 checkpoint hr cho mọi seed).

set -e

mkdir -p "$RESULTS_DIR"

# [SỬA — lỗi phát hiện qua code review] đọc scale từ config thay vì hardcode 4.
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

# [SỬA — bổ sung sau code review, vòng 6] Xem giải thích đầy đủ trong
# pipeline/run_multi_seed_kdv2.sh — kiểm tra ĐỦ CẢ 15 checkpoint
# recognition_lr_<backbone>_seed<seed> NGAY TỪ ĐẦU, TRƯỚC Bước 1/2 (train SR
# learned-pruning, tốn hàng giờ), thay vì để phát hiện thiếu ở Bước 2/2 sau
# khi đã lãng phí toàn bộ thời gian train SR.
echo "Kiểm tra tiền đề: ${#BACKBONES[@]} backbone x ${#SEEDS[@]} seed = "\
"$((${#BACKBONES[@]} * ${#SEEDS[@]})) checkpoint recognition_lr_*_seed*..."
MISSING=0
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: thiếu $CKPT" >&2
            MISSING=1
        fi
    done
done
if [ "$MISSING" -eq 1 ]; then
    echo "-> chạy pipeline/run_multi_seed.sh (đủ 5 backbone x 3 seed, domain lr) trước." >&2
    exit 1
fi
echo "OK — đủ tiền đề recognition_lr_*_seed*, bắt đầu chạy."

if python -c "import sys; sys.exit(0 if float('$LAMBDA_SALIENCY') > 0 or float('$LAMBDA_IDENTITY') > 0 else 1)"; then
    source "$(dirname "$0")/_check_hr_judges.sh"
    check_hr_judges
fi

echo "################################################################"
echo "# Bước 1/2: Train SR (learned pruning) — 1 lần, seed cố định=$SR_SEED"
echo "# lambda_sparsity=$LAMBDA_SPARSITY"
echo "################################################################"
python train_sr_learned_prune.py --config "$CONFIG" \
    --lambda_feat "$LAMBDA_FEAT" --lambda_saliency "$LAMBDA_SALIENCY" \
    --lambda_identity "$LAMBDA_IDENTITY" \
    --lambda_sparsity "$LAMBDA_SPARSITY" \
    --seed "$SR_SEED" --run_suffix "_final"

N_BLOCKS=$(python -c "import json; print(json.load(open('runs/sr_learned_prune_final/prune_metadata.json'))['n_blocks_kept'])")
echo "-> Model đã cứng hoá còn lại $N_BLOCKS khối (xem runs/sr_learned_prune_final/prune_metadata.json)"

# [SỬA — bổ sung sau code review, vòng 4, điểm 2] span_tiny (cắt tay) LUÔN có
# đúng 3/6 khối. So sánh downstream accuracy của learned-pruning với
# sr_improved (span_tiny) chỉ CÔNG BẰNG VỀ KÍCH THƯỚC MODEL khi N_BLOCKS==3
# (cùng params/FLOPs) — nếu khác 3, đây vẫn là so sánh HỢP LỆ nhưng là so
# sánh 2 ĐIỂM KHÁC NHAU trên đường cong đánh đổi kích thước/accuracy, KHÔNG
# PHẢI "cùng kích thước, learned pruning tốt hơn/kém hơn thủ công" — phải nêu
# rõ số khối thực tế khi báo cáo, không ngầm định bằng span_tiny.
if [ "$N_BLOCKS" -ne 3 ]; then
    echo ""
    echo "!!! LƯU Ý: N_BLOCKS=$N_BLOCKS KHÁC 3 (số khối cố định của span_tiny) !!!"
    echo "    So sánh downstream accuracy bên dưới với domain 'sr_improved' (span_tiny)"
    echo "    KHÔNG PHẢI so sánh cùng kích thước model (params/FLOPs sẽ khác đi cùng"
    echo "    N_BLOCKS) — nếu muốn kết luận 'learned pruning tốt hơn cắt tay ở CÙNG"
    echo "    ngân sách tham số', cần chỉnh lambda_sparsity (xem"
    echo "    pipeline/run_prune_sparsity_screen.sh) để N_BLOCKS hội tụ về đúng 3."
    echo "    Nếu giữ nguyên $N_BLOCKS khối, hãy báo cáo như 1 điểm KHÁC trên đường"
    echo "    cong size-accuracy, kèm rõ params/FLOPs thực tế (xem sr_quality_final.csv)"
    echo "    thay vì so sánh thẳng với span_tiny như cùng kích thước."
    echo ""
fi

echo ""
echo ">>> Sinh tập $DOMAIN..."
python data/build_sr.py --lr_dir splits/lr \
    --sr_ckpt "runs/sr_learned_prune_final/best.pt" \
    --arch span_pruned --n_blocks "$N_BLOCKS" --scale "$SCALE" --out_dir "splits/${DOMAIN}"

echo ""
echo ">>> Đo chất lượng ảnh + params/FLOPs/latency của model đã cứng hoá..."
python eval_sr_quality.py --config "$CONFIG" --arch span_pruned --n_blocks "$N_BLOCKS" \
    --ckpt "runs/sr_learned_prune_final/best.pt" \
    --label "span_learned_prune_${N_BLOCKS}blocks" \
    --out_csv "$RESULTS_DIR/sr_quality_final.csv"

echo ""
echo "################################################################"
echo "# Bước 2/2: Recognition multi-seed x 5-backbone trên domain $DOMAIN"
echo "# (${#BACKBONES[@]} backbone x ${#SEEDS[@]} seed = $((${#BACKBONES[@]} * ${#SEEDS[@]})) lần train)"
echo "################################################################"
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        LR_CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$LR_CKPT" ]; then
            echo "LỖI: không tìm thấy $LR_CKPT" >&2
            echo "  -> chạy pipeline/run_multi_seed.sh trước (ít nhất domain lr, backbone=$BACKBONE, seed=$SEED) rồi thử lại." >&2
            exit 1
        fi

        echo "----------------------------------------------------------------"
        echo "backbone=$BACKBONE | seed=$SEED | domain=$DOMAIN"
        echo "----------------------------------------------------------------"
        python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" --backbone "$BACKBONE" \
            --init_ckpt "$LR_CKPT" --seed "$SEED" --run_suffix "_seed${SEED}"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
            --out_json "$RESULTS_DIR/${DOMAIN}_${BACKBONE}_seed${SEED}.json"
    done
done

echo ""
echo ">>> Tổng hợp (tái sử dụng aggregator chung, xem data/aggregate_multi_seed_results.py)..."
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/multi_seed_learned_prune_summary.csv"

echo ""
echo "HOÀN TẤT. Kết quả learned pruning: $RESULTS_DIR/multi_seed_learned_prune_summary.csv"
echo ""
echo ">>> ĐỂ SO SÁNH TRỰC TIẾP với 'sr_improved' (span_tiny, cắt tay) đã có sẵn"
echo "    từ pipeline/run_multi_seed.sh — copy JSON của domain đó vào cùng thư"
echo "    mục rồi chạy lại đúng lệnh aggregate ở trên:"
echo "      cp results/multi_seed/sr_improved_*.json $RESULTS_DIR/"
echo "      python data/aggregate_multi_seed_results.py --results_dir $RESULTS_DIR \\"
echo "          --out_csv $RESULTS_DIR/multi_seed_learned_prune_summary.csv"
