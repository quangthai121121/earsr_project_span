#!/bin/bash
# BƯỚC 8/8: Test toàn bộ ma trận backbone x domain (đã train ở bước 3,5,7),
# thêm cấu hình chẩn đoán domain gap (train=hr, test=sr_*), đo chất lượng SR
# (PSNR/SSIM/FLOPs), eval trên real_lr_holdout, tổng hợp CSV kết quả cuối cùng.
set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
DOMAINS=("hr" "lr" "sr_baseline" "sr_improved")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")
SR_ARCH=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['sr']['arch'])")
STUDENT_ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")

echo "################################################################"
echo "# BƯỚC 8/8: Test + tổng hợp kết quả                            #"
echo "################################################################"

echo ">>> [1/4] Test ma trận chính (train_domain == test_domain)..."
for BACKBONE in "${BACKBONES[@]}"; do
    for DOMAIN in "${DOMAINS[@]}"; do
        echo "    backbone=$BACKBONE | domain=$DOMAIN"
        python eval_recognition.py --config "$CONFIG" \
            --ckpt "runs/recognition_${DOMAIN}_${BACKBONE}/best.pt" \
            --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
            --out_json "$RESULTS_DIR/${DOMAIN}_${DOMAIN}_${BACKBONE}.json"
    done
done

echo ""
echo ">>> [2/4] Chẩn đoán domain gap (train=hr, test=sr_baseline/sr_improved)..."
for BACKBONE in "${BACKBONES[@]}"; do
    for TEST_DOMAIN in sr_baseline sr_improved; do
        echo "    backbone=$BACKBONE | train=hr | test=$TEST_DOMAIN"
        python eval_recognition.py --config "$CONFIG" \
            --ckpt "runs/recognition_hr_${BACKBONE}/best.pt" \
            --backbone "$BACKBONE" --train_domain hr --test_domain "$TEST_DOMAIN" \
            --out_json "$RESULTS_DIR/hr_${TEST_DOMAIN}_${BACKBONE}.json"
    done
done

echo ""
echo ">>> [3/4] Đo chất lượng ảnh SR (PSNR/SSIM/FLOPs)..."
python eval_sr_quality.py --config "$CONFIG" --arch edsr \
    --ckpt runs/sr_edsr/best.pt --label edsr_teacher \
    --out_csv "$RESULTS_DIR/sr_quality.csv"
python eval_sr_quality.py --config "$CONFIG" --arch "$SR_ARCH" \
    --ckpt "runs/sr_${SR_ARCH}/best.pt" --label span_baseline \
    --out_csv "$RESULTS_DIR/sr_quality.csv"
python eval_sr_quality.py --config "$CONFIG" --arch "$STUDENT_ARCH" \
    --ckpt "runs/sr_improved_${STUDENT_ARCH}/best.pt" --label span_improved_tiny \
    --out_csv "$RESULTS_DIR/sr_quality.csv"

echo ""
echo ">>> [4/4] Eval trên real_lr_holdout (LR thật, dùng backbone mobilenet_v2)..."
python eval_real_lr_holdout.py --config "$CONFIG" --backbone mobilenet_v2 \
    --sr_baseline_ckpt "runs/sr_${SR_ARCH}/best.pt" --sr_baseline_arch "$SR_ARCH" \
    --sr_improved_ckpt "runs/sr_improved_${STUDENT_ARCH}/best.pt" --sr_improved_arch "$STUDENT_ARCH" \
    --out_csv "$RESULTS_DIR/real_lr_holdout.csv"

echo ""
echo ">>> Tổng hợp bảng kết quả chính + trích xuất log training..."
python data/aggregate_results.py --results_dir "$RESULTS_DIR" --out_csv "$RESULTS_DIR/summary.csv"
python data/export_training_log_summary.py --runs_root runs --out_csv "$RESULTS_DIR/training_summary.csv"

echo ""
echo "################################################################"
echo "# HOÀN TẤT TOÀN BỘ PIPELINE (8/8)                              #"
echo "################################################################"
echo "Toàn bộ file CSV kết quả trong $RESULTS_DIR/:"
echo "  - summary.csv            : accuracy chính theo backbone x domain"
echo "  - sr_quality.csv          : PSNR/SSIM/params/FLOPs/latency của SR models"
echo "  - real_lr_holdout.csv    : accuracy trên LR thật (kiểm chứng tổng quát hóa)"
echo "  - training_summary.csv   : epoch dừng, best val, OOM count, thời gian train"
echo "  (chạy thêm 'bash pipeline/run_ablation.sh' riêng để có ablation.csv)"
echo ""
echo "Điền số liệu vào bảng kết quả bài báo (xem RUNBOOK_EarVN1.0.md mục 8)."
