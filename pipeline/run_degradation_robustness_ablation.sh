#!/bin/bash
# [MỚI — Mục 5.9 bài báo, giải pháp cho "Boundary Condition" real_lr_holdout]
# Reviewer chỉ ra: bài báo mới dừng ở mức PHÁT HIỆN lỗi khi gặp ảnh thực
# <20px (Section res-main) mà chưa đưa ra giải pháp khắc phục. Script này
# train lại span_tiny/span_baseline với degradation augmentation THẬT
# (blur+noise+JPEG ngẫu nhiên mỗi sample mỗi epoch, xem
# utils/degradation.py::random_degrade()) thay vì chỉ bicubic sạch, rồi kiểm
# tra 2 điều:
#   (a) real_lr_holdout có cải thiện không (mục tiêu chính, biến điểm yếu
#       thành ablation thành công theo đúng đề xuất reviewer)
#   (b) compression-safety (claim CHÍNH của bài báo) có còn giữ nguyên dưới
#       recipe training MỚI không (sanity-check bắt buộc — không được để
#       thay đổi recipe làm hỏng claim chính đã kiểm chứng kỹ trước đó)
#
# [LỰA CHỌN THIẾT KẾ] Screening scale trước: mobilenet_v2 only (khớp đúng
# backbone đã dùng cho real_lr_holdout multi-seed check gốc), n=5 seed cho
# bước recognition (rẻ, không train lại SR) — chỉ 2 checkpoint SR mới
# (span_tiny_robust, span_baseline_robust) là chi phí GPU thật, mỗi cái train
# ĐÚNG 1 LẦN ở seed cố định, đúng nguyên tắc "SR train 1 lần, chỉ recognition
# lặp seed" đã dùng xuyên suốt project. Nếu kết quả khả quan, mở rộng sang
# 5 backbone theo đúng mẫu n=3->n=5 / 1-backbone->5-backbone đã áp dụng cho
# các ablation khác (xem RUNBOOK).
#
# KHÔNG đụng vào recipe/checkpoint GỐC (span_tiny, span_baseline không suffix)
# — đây là 1 BIẾN THỂ MỚI hoàn toàn tách biệt (hậu tố "_robust"), giữ nguyên
# mọi kết quả đã có trong bài báo.

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results/degradation_robustness"
LRHOLDOUT_RESULTS_DIR="results/degradation_robustness_lrholdout"
BACKBONE="mobilenet_v2"
SEEDS=(42 123 2024 44 999)
NUM_WORKERS="${NUM_WORKERS:-4}"
mkdir -p "$RESULTS_DIR" "$LRHOLDOUT_RESULTS_DIR"

SR_ARCH=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['sr']['arch'])")
STUDENT_ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

echo "Kiểm tra tiền đề..."
MISSING=0
for SEED in "${SEEDS[@]}"; do
    CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
    if [ ! -f "$CKPT" ]; then
        echo "LỖI: thiếu $CKPT -> chạy pipeline/run_multi_seed.sh (+ _extra_seeds.sh) trước." >&2
        MISSING=1
    fi
done
if [ ! -f "checkpoints/span_pretrained_x4.pth" ]; then
    echo "LỖI: thiếu checkpoints/span_pretrained_x4.pth (checkpoint SPAN chính thức, xem "
    echo "     scripts/setup_span_official.sh)." >&2
    MISSING=1
fi
if [ "$MISSING" -eq 1 ]; then
    echo "DỪNG LẠI — thiếu tiền đề ở trên." >&2
    exit 1
fi
echo "OK — đủ tiền đề, bắt đầu chạy."

echo "################################################################"
echo "# Bước 1/5: Train span_tiny_robust (degradation augmentation)  #"
echo "################################################################"
if [ -f "runs/sr_improved_${STUDENT_ARCH}_robust/best.pt" ]; then
    echo ">>> Bỏ qua (đã có checkpoint từ lần chạy trước)"
else
    python train_sr_distill.py --config "$CONFIG" \
        --lambda_pixel 1.0 --lambda_distill 1.0 --lambda_feat 0 --lambda_saliency 0 --lambda_identity 0 \
        --degradation_augment --run_suffix "_robust" --num_workers "$NUM_WORKERS"
fi

echo ""
echo "################################################################"
echo "# Bước 2/5: Train span_baseline_robust (degradation augmentation) #"
echo "################################################################"
if [ -f "runs/sr_${SR_ARCH}_robust/best.pt" ]; then
    echo ">>> Bỏ qua (đã có checkpoint từ lần chạy trước)"
else
    python train_sr.py --config "$CONFIG" --sr_arch "$SR_ARCH" \
        --pretrained_path checkpoints/span_pretrained_x4.pth \
        --degradation_augment --run_suffix "_robust"
fi

echo ""
echo "################################################################"
echo "# Bước 3/5: Sinh domain sr_baseline_robust / sr_improved_robust  #"
echo "################################################################"
python data/build_sr.py --lr_dir splits/lr --sr_ckpt "runs/sr_${SR_ARCH}_robust/best.pt" \
    --arch "$SR_ARCH" --scale "$SCALE" --out_dir splits/sr_baseline_robust
python data/build_sr.py --lr_dir splits/lr --sr_ckpt "runs/sr_improved_${STUDENT_ARCH}_robust/best.pt" \
    --arch "$STUDENT_ARCH" --scale "$SCALE" --out_dir splits/sr_improved_robust

echo ""
echo "################################################################"
echo "# Bước 4/5: Fine-tune recognition + sanity-check benchmark chính #"
echo "# (n=${#SEEDS[@]} seed, KHÔNG train lại SR)                                #"
echo "################################################################"
for DOMAIN in sr_baseline_robust sr_improved_robust; do
    for SEED in "${SEEDS[@]}"; do
        if [ -f "$RESULTS_DIR/${DOMAIN}_${BACKBONE}_seed${SEED}.json" ]; then
            echo ">>> Bỏ qua $DOMAIN seed=$SEED (đã có JSON từ lần chạy trước)"
            continue
        fi
        echo "----------------------------------------------------------------"
        echo "domain=$DOMAIN | backbone=$BACKBONE | seed=$SEED"
        echo "----------------------------------------------------------------"
        LR_CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" --backbone "$BACKBONE" \
            --init_ckpt "$LR_CKPT" --seed "$SEED" --run_suffix "_seed${SEED}" --num_workers "$NUM_WORKERS"
        python eval_recognition.py --config "$CONFIG" \
            --ckpt "runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
            --out_json "$RESULTS_DIR/${DOMAIN}_${BACKBONE}_seed${SEED}.json" --num_workers "$NUM_WORKERS"
    done
done

echo ""
echo ">>> Tổng hợp sanity-check benchmark chính (compression-safety dưới robust training)..."
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/degradation_robustness_mainbench_summary.csv"

echo ""
echo "################################################################"
echo "# Bước 5/5: Đánh giá real_lr_holdout với checkpoint robust (n=${#SEEDS[@]}) #"
echo "################################################################"
for SEED in "${SEEDS[@]}"; do
    if [ -f "$LRHOLDOUT_RESULTS_DIR/real_lr_holdout_seed${SEED}.csv" ]; then
        echo ">>> Bỏ qua seed=$SEED (đã có CSV từ lần chạy trước)"
        continue
    fi
    python eval_real_lr_holdout.py --config "$CONFIG" --backbone "$BACKBONE" \
        --sr_baseline_ckpt "runs/sr_${SR_ARCH}_robust/best.pt" --sr_baseline_arch "$SR_ARCH" \
        --sr_improved_ckpt "runs/sr_improved_${STUDENT_ARCH}_robust/best.pt" --sr_improved_arch "$STUDENT_ARCH" \
        --run_suffix "_seed${SEED}" --seed_label "$SEED" --domain_suffix "_robust" \
        --num_workers "$NUM_WORKERS" \
        --out_csv "$LRHOLDOUT_RESULTS_DIR/real_lr_holdout_seed${SEED}.csv"
done

echo ""
echo ">>> Tổng hợp real_lr_holdout với checkpoint robust..."
python data/aggregate_real_lr_holdout_multiseed.py --results_dir "$LRHOLDOUT_RESULTS_DIR" \
    --out_prefix "$LRHOLDOUT_RESULTS_DIR/real_lr_holdout_robust"

echo ""
echo "HOÀN TẤT. So sánh 2 bộ kết quả:"
echo "  - GỐC (bicubic sạch):    results/real_lr_holdout_multiseed/real_lr_holdout_multiseed_{identity,gender}*.csv"
echo "  - ROBUST (degradation):  $LRHOLDOUT_RESULTS_DIR/real_lr_holdout_robust_{identity,gender}*.csv"
echo "Sanity-check benchmark chính (compression-safety dưới robust training):"
echo "  $RESULTS_DIR/degradation_robustness_mainbench_summary.csv"
echo "  $RESULTS_DIR/degradation_robustness_mainbench_summary_pairwise.csv"
