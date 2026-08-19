#!/bin/bash
# [MỚI] Quét λ_saliency (saliency-weighted identity-critical pixel loss, xem
# compute_multi_judge_saliency trong train_sr_distill.py) theo ĐÚNG mẫu
# run_lambda_sweep.sh (đã dùng cho lambda_identity) — nhiều mức giá trị x
# nhiều seed, có paired t-test + Cohen's d ngay khi tìm điểm tối ưu.
#
# CHẠY SAU KHI đã chốt được cấu hình thắng ở run_ablation_kd_v2.sh (feat +
# multi-judge) — SỬA 2 biến LAMBDA_FEAT_FIXED/LAMBDA_IDENTITY_FIXED bên dưới
# khớp đúng cấu hình đó trước khi chạy, vì saliency loss cộng THÊM vào cùng
# công thức tổng, cần biết rõ đang cộng vào nền nào.
#
# Backbone: mobilenet_v2 (đại diện, giữ tốc độ chạy hợp lý cho bước TÌM cấu
# hình — sau khi chốt lambda_saliency mới mở rộng sang multi-seed x 5 backbone
# đầy đủ, xem pipeline/run_multi_seed_kdv2.sh).

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results/lambda_saliency_sweep"
BACKBONE="mobilenet_v2"
SEEDS=(42 123 2024)
LAMBDA_SALIENCY_VALUES=(0.0 0.15 0.3 0.6 1.0)

# **SỬA 2 THAM SỐ NÀY** khớp đúng cấu hình đã THẮNG ở run_ablation_kd_v2.sh
LAMBDA_FEAT_FIXED=0.5
LAMBDA_IDENTITY_FIXED=0.1

mkdir -p "$RESULTS_DIR"

STUDENT_ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

# [SỬA — bổ sung sau code review, vòng 5, điểm 2] Xem giải thích đầy đủ trong
# pipeline/run_prune_sparsity_screen.sh — script này CŨNG dùng checkpoint
# KHÔNG suffix seed. Fail sớm thay vì mất hàng giờ (5 mức x 3 seed = 15 lần
# train SR) rồi mới lỗi ở lần train_recognition.py ĐẦU TIÊN.
LR_CKPT_NOSUFFIX="runs/recognition_lr_${BACKBONE}/best.pt"
if [ ! -f "$LR_CKPT_NOSUFFIX" ]; then
    echo "LỖI: thiếu $LR_CKPT_NOSUFFIX (checkpoint KHÔNG suffix seed)."
    echo "     -> chạy 'python train_recognition.py --config $CONFIG --domain lr"
    echo "        --backbone $BACKBONE' (xem pipeline/03_train_baseline_recognition.sh)"
    echo "        trước — KHÁC với checkpoint '_seed<N>' của run_multi_seed.sh."
    exit 1
fi

# Mọi lần chạy ở đây có lambda_identity>0 và/hoặc lambda_saliency>0 → cần 3 judge.
source "$(dirname "$0")/_check_hr_judges.sh"
check_hr_judges

for LAMBDA_SAL in "${LAMBDA_SALIENCY_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        TAG="lsal${LAMBDA_SAL}_seed${SEED}"
        echo "################################################################"
        echo "# lambda_saliency=$LAMBDA_SAL (feat=$LAMBDA_FEAT_FIXED identity=$LAMBDA_IDENTITY_FIXED) | seed=$SEED"
        echo "################################################################"

        # 1) Train student với đúng seed + lambda_saliency này, giữ nguyên
        # feat/identity ở cấu hình đã chốt từ run_ablation_kd_v2.sh.
        python train_sr_distill.py --config "$CONFIG" \
            --lambda_pixel 1.0 --lambda_distill 1.0 \
            --lambda_feat "$LAMBDA_FEAT_FIXED" --lambda_identity "$LAMBDA_IDENTITY_FIXED" \
            --lambda_saliency "$LAMBDA_SAL" \
            --seed "$SEED" --run_suffix "_${TAG}"

        # 2) Sinh ảnh SR từ student vừa train
        python data/build_sr.py --lr_dir splits/lr \
            --sr_ckpt "runs/sr_improved_${STUDENT_ARCH}_${TAG}/best.pt" \
            --arch "$STUDENT_ARCH" --scale "$SCALE" \
            --out_dir "splits/sr_sal_sweep_${TAG}"

        # 3) Train recognition trên ảnh vừa sinh (từ checkpoint LR gốc, seed khớp)
        python train_recognition.py --config "$CONFIG" --domain "sr_sal_sweep_${TAG}" \
            --backbone "$BACKBONE" --init_ckpt "runs/recognition_lr_${BACKBONE}/best.pt" \
            --seed "$SEED"

        # 4) Đo cả accuracy lẫn chất lượng ảnh (PSNR/SSIM ROI) cho đủ số liệu
        python eval_recognition.py --config "$CONFIG" \
            --ckpt "runs/recognition_sr_sal_sweep_${TAG}_${BACKBONE}/best.pt" \
            --backbone "$BACKBONE" --train_domain "sr_sal_sweep_${TAG}" --test_domain "sr_sal_sweep_${TAG}" \
            --out_json "$RESULTS_DIR/acc_${TAG}.json"

        python eval_sr_quality.py --config "$CONFIG" --arch "$STUDENT_ARCH" \
            --ckpt "runs/sr_improved_${STUDENT_ARCH}_${TAG}/best.pt" \
            --label "lsal${LAMBDA_SAL}_seed${SEED}" \
            --out_csv "$RESULTS_DIR/sr_quality_sweep.csv"
    done
done

echo ""
echo ">>> Tổng hợp — tính trung bình +- độ lệch chuẩn cho từng mức lambda_saliency..."
python data/aggregate_saliency_sweep.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/saliency_sweep_summary.csv"

echo ""
echo "HOÀN TẤT. Xem $RESULTS_DIR/saliency_sweep_summary.csv để chọn lambda_saliency tối ưu."
echo "SAU KHI CHỌN XONG: cập nhật lambda_saliency trong configs/config.yaml, rồi mới chạy"
echo "multi-seed x 5-backbone đầy đủ (sửa LAMBDA_FEAT/LAMBDA_IDENTITY trong"
echo "pipeline/run_multi_seed_kdv2.sh và thêm --lambda_saliency vào lệnh train_sr_distill.py ở đó)."
