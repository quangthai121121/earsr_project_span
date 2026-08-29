#!/bin/bash
# [MỚI — 2026-08-29] Validate ĐẦY ĐỦ multi-seed x 5-backbone cho saliency-
# weighted identity-critical loss — xem train_sr_distill.py::compute_multi_judge_saliency.
# Đây là BƯỚC BẮT BUỘC sau khi pipeline/run_lambda_saliency_sweep.sh (1 backbone,
# n=3 seed — chỉ tín hiệu sàng lọc) cho thấy lambda_saliency=1.0 có xu hướng
# DƯƠNG (0.4855 vs baseline 0.4754, độ lệch chuẩn thấp nhất trong các mức đã
# quét) — cần n=5 x 5 backbone để kết luận chắc chắn, cùng chuẩn nghiêm ngặt
# đã áp dụng cho KD v2 và learned pruning (2 cơ chế kia đều đã đóng âm tính,
# có ý nghĩa thống kê đầy đủ — saliency là câu hỏi mở DUY NHẤT còn lại).
#
# **SỬA THAM SỐ NÀY** nếu muốn validate mức lambda_saliency khác (mặc định để
# sẵn 1.0 — mức cao nhất đã quét, có mean cao nhất VÀ std thấp nhất trong
# results/lambda_saliency_sweep/saliency_sweep_summary.csv):
LAMBDA_SALIENCY=1.0

# Recipe SẠCH (khớp lambda_feat=0/identity=0 đã dùng cho toàn bộ sweep sau
# 2026-08-24 — xem pipeline/run_lambda_saliency_sweep.sh) — CHỈ cô lập biến
# lambda_saliency, không lẫn feat-KD/multi-judge identity (đã bị bác bỏ).
LAMBDA_FEAT=0.0
LAMBDA_IDENTITY=0.0

# SR model CHỈ train 1 LẦN ở seed cố định — ĐÚNG protocol thống kê đã dùng
# xuyên suốt project (train SR nhiều seed bị đánh giá không khả thi về chi
# phí, chỉ downstream recognition được lặp lại qua seed để đo phương sai).
SR_SEED=42

CONFIG="configs/config.yaml"
RESULTS_DIR="results/multi_seed_saliency"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(42 123 2024)   # thêm 44, 999 ở pipeline/run_multi_seed_saliency_extra_seeds.sh cho n=5
DOMAIN="sr_improved_saliency"

# YÊU CẦU: đã chạy xong pipeline/run_multi_seed.sh trước đó — script này TÁI
# SỬ DỤNG checkpoint recognition_lr_<backbone>_seed<seed> đã train sẵn cho
# từng seed, ĐÚNG chuỗi fine-tune hr -> lr -> sr_improved_saliency (không dùng
# chung 1 checkpoint hr cho mọi seed).

set -e

mkdir -p "$RESULTS_DIR"

ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

# Kiểm tra ĐỦ CẢ 15 checkpoint recognition_lr_<backbone>_seed<seed> NGAY TỪ
# ĐẦU, trước Bước 1/2 (train SR, tốn hàng giờ) — fail nhanh nếu thiếu.
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

# lambda_saliency>0 -> cần 3 judge HR trước khi train SR.
source "$(dirname "$0")/_check_hr_judges.sh"
check_hr_judges

echo "################################################################"
echo "# Bước 1/2: Train SR (saliency-weighted loss) — 1 lần, seed cố định=$SR_SEED"
echo "# lambda_saliency=$LAMBDA_SALIENCY (feat=$LAMBDA_FEAT identity=$LAMBDA_IDENTITY)"
echo "################################################################"
python train_sr_distill.py --config "$CONFIG" \
    --lambda_pixel 1.0 --lambda_distill 1.0 \
    --lambda_feat "$LAMBDA_FEAT" --lambda_identity "$LAMBDA_IDENTITY" \
    --lambda_saliency "$LAMBDA_SALIENCY" \
    --seed "$SR_SEED" --run_suffix "_saliency"

echo ""
echo ">>> Sinh tập $DOMAIN..."
python data/build_sr.py --lr_dir splits/lr \
    --sr_ckpt "runs/sr_improved_${ARCH}_saliency/best.pt" \
    --arch "$ARCH" --scale "$SCALE" --out_dir "splits/${DOMAIN}"

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
        # QUAN TRỌNG: fine-tune từ checkpoint LR CỦA CHÍNH SEED NÀY, đúng
        # chuỗi hr -> lr -> sr_improved_saliency, KHÔNG dùng chung checkpoint hr.
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
    --out_csv "$RESULTS_DIR/multi_seed_saliency_summary.csv"

echo ""
echo "HOÀN TẤT. Kết quả riêng recipe mới: $RESULTS_DIR/multi_seed_saliency_summary.csv"
echo ""
echo ">>> ĐỂ SO SÁNH TRỰC TIẾP với 'lr' (no-SR) và 'sr_improved' (span_tiny recipe"
echo "    gốc) đã có sẵn từ pipeline/run_multi_seed.sh — copy JSON của 2 domain đó"
echo "    vào cùng thư mục rồi chạy lại đúng lệnh aggregate ở trên:"
echo "      cp results/multi_seed/lr_*.json $RESULTS_DIR/"
echo "      cp results/multi_seed/sr_improved_*.json $RESULTS_DIR/"
echo "      python data/aggregate_multi_seed_results.py --results_dir $RESULTS_DIR \\"
echo "          --out_csv $RESULTS_DIR/multi_seed_saliency_summary.csv"
