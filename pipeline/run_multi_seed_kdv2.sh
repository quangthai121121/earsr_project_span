#!/bin/bash
# [MỚI] Validate ĐẦY ĐỦ multi-seed x 5-backbone cho recipe KD v2 (multi-judge
# identity loss + feature-level KD — xem train_sr_distill.py,
# docs/03_span_improvement.md mục "Multi-Judge Ensemble Identity-Aware
# Distillation + Feature-level KD"). Đây là BƯỚC BẮT BUỘC sau khi
# pipeline/run_ablation_kd_v2.sh (1 backbone, 1 seed — chỉ là tín hiệu sàng
# lọc nhanh) đã xác định được cấu hình thắng, trước khi đưa số liệu vào bảng
# kết quả chính thức — tránh lặp lại đúng giới hạn "ablation loss chỉ chạy 1
# backbone" đã bị nêu ra khi review Table 3 của bản thảo bài báo hiện tại.
#
# **SỬA 2 THAM SỐ NÀY** trước khi chạy, khớp đúng cấu hình đã THẮNG ở
# run_ablation_kd_v2.sh (mặc định để sẵn "kdv2_full" — cả 2 cơ chế cùng lúc):
LAMBDA_FEAT=0.5
LAMBDA_IDENTITY=0.1
# Nếu cấu hình thắng là "kdv2_feat" (chỉ feature-KD): LAMBDA_IDENTITY=0.0
# Nếu cấu hình thắng là "kdv2_multijudge" (chỉ multi-judge): LAMBDA_FEAT=0.0

# SR model CHỈ train 1 LẦN ở seed cố định — ĐÚNG protocol thống kê đã dùng
# xuyên suốt project (xem docs/03_span_improvement.md, configs/config.yaml
# split.seed): train SR nhiều seed bị đánh giá không khả thi về chi phí, chỉ
# downstream recognition được lặp lại qua seed để đo phương sai.
SR_SEED=42

CONFIG="configs/config.yaml"
RESULTS_DIR="results/multi_seed_kdv2"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(42 123 2024)   # đổi/thêm seed nếu muốn khớp n=5 như bản thảo bài báo (thêm 44, 999)
DOMAIN="sr_improved_kdv2"

# YÊU CẦU: đã chạy xong pipeline/run_multi_seed.sh trước đó — script này TÁI
# SỬ DỤNG checkpoint recognition_lr_<backbone>_seed<seed> đã train sẵn cho
# từng seed, ĐÚNG chuỗi fine-tune hr -> lr -> sr_improved_kdv2 (không dùng
# chung 1 checkpoint hr cho mọi seed — nguyên tắc đã áp dụng nhất quán cho
# MỌI domain khác trong project này, xem README mục "Danh sách lỗi đã phát
# hiện", lỗi #9).

set -e

mkdir -p "$RESULTS_DIR"

ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

# [SỬA — bổ sung sau code review, vòng 6] TRƯỚC ĐÂY: 15 checkpoint
# recognition_lr_<backbone>_seed<seed> chỉ được kiểm tra BÊN TRONG vòng lặp
# "Bước 2/2", tức là SAU KHI "Bước 1/2" (train SR, tốn hàng giờ) đã chạy xong
# — nếu thiếu dù chỉ 1 checkpoint, toàn bộ thời gian train SR đó bị lãng phí
# trước khi phát hiện lỗi. Kiểm tra ĐỦ CẢ 15 checkpoint NGAY TỪ ĐẦU, trước
# Bước 1/2, để fail nhanh.
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

# kdv2_full mặc định lambda_identity>0 → cần 3 judge HR trước khi train SR.
if python -c "import sys; sys.exit(0 if float('$LAMBDA_IDENTITY') > 0 else 1)"; then
    source "$(dirname "$0")/_check_hr_judges.sh"
    check_hr_judges
fi

echo "################################################################"
echo "# Bước 1/2: Train SR (recipe KD v2) — 1 lần, seed cố định=$SR_SEED"
echo "# lambda_feat=$LAMBDA_FEAT lambda_identity=$LAMBDA_IDENTITY (multi-judge)"
echo "################################################################"
# [SỬA — confound phát hiện qua code review] PIN TƯỜNG MINH lambda_saliency=0
# — script này validate ĐÚNG cấu hình đã thắng ở run_ablation_kd_v2.sh (chỉ
# feat x multijudge, KHÔNG có saliency), không được để saliency âm thầm bật
# qua default config.yaml.
python train_sr_distill.py --config "$CONFIG" \
    --lambda_pixel 1.0 --lambda_distill 1.0 --lambda_saliency 0 \
    --lambda_feat "$LAMBDA_FEAT" --lambda_identity "$LAMBDA_IDENTITY" \
    --seed "$SR_SEED" --run_suffix "_kdv2"

echo ""
echo ">>> Sinh tập $DOMAIN..."
python data/build_sr.py --lr_dir splits/lr \
    --sr_ckpt "runs/sr_improved_${ARCH}_kdv2/best.pt" \
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
        # QUAN TRỌNG: fine-tune từ checkpoint LR CỦA CHÍNH SEED NÀY (đã train
        # sẵn bởi run_multi_seed.sh) — đúng chuỗi hr -> lr -> sr_improved_kdv2,
        # KHÔNG dùng chung 1 checkpoint hr cho mọi seed.
        python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" --backbone "$BACKBONE" \
            --init_ckpt "$LR_CKPT" --seed "$SEED" --run_suffix "_seed${SEED}"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
            --out_json "$RESULTS_DIR/${DOMAIN}_${BACKBONE}_seed${SEED}.json"
    done
done

echo ""
echo ">>> Tổng hợp (tái sử dụng aggregator chung — hoạt động với BẤT KỲ tên"
echo "    domain nào, đọc (backbone, test_domain) từ NỘI DUNG JSON, không phụ"
echo "    thuộc tên file/thư mục — xem data/aggregate_multi_seed_results.py)..."
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/multi_seed_kdv2_summary.csv"

echo ""
echo "HOÀN TẤT. Kết quả riêng recipe mới: $RESULTS_DIR/multi_seed_kdv2_summary.csv"
echo ""
echo ">>> ĐỂ SO SÁNH TRỰC TIẾP với 'lr' (no-SR) và 'sr_improved' (span_tiny recipe"
echo "    CŨ) đã có sẵn từ pipeline/run_multi_seed.sh — copy JSON của 2 domain đó"
echo "    vào cùng thư mục rồi chạy lại đúng lệnh aggregate ở trên (script tự gộp"
echo "    theo domain đọc từ JSON, sinh cả bảng paired t-test + Cohen's d cho MỌI"
echo "    cặp domain, bao gồm 'sr_improved' (cũ) vs 'sr_improved_kdv2' (mới)):"
echo "      cp results/multi_seed/lr_*.json $RESULTS_DIR/"
echo "      cp results/multi_seed/sr_improved_*.json $RESULTS_DIR/"
echo "      python data/aggregate_multi_seed_results.py --results_dir $RESULTS_DIR \\"
echo "          --out_csv $RESULTS_DIR/multi_seed_kdv2_summary.csv"
