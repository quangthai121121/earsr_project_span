#!/bin/bash
# QUÉT λ_identity đúng chuẩn cho journal Q1: nhiều mức giá trị (không chỉ 2 điểm
# rời rạc như ablation cũ) x nhiều seed (đo được độ tin cậy thống kê ngay khi
# tìm điểm tối ưu). Mục đích: giải quyết mâu thuẫn "full" thua "pixel_distill"
# trong ablation cũ trước khi chạy các thí nghiệm lớn hơn — tránh phải làm lại
# nếu cấu hình tối ưu thay đổi.
#
# Cố định λ_pixel=1.0, λ_distill=1.0 (đã xác nhận distillation luôn có lợi ở
# mọi mức trong ablation cũ) — chỉ quét λ_identity.
#
# Backbone: mobilenet_v2 (đại diện, giữ tốc độ chạy hợp lý cho bước TÌM cấu
# hình — Bước 2 sau đó mới mở rộng cấu hình đã chốt sang cả 5 backbone).

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results/lambda_sweep"
BACKBONE="mobilenet_v2"
SEEDS=(42 123 2024)
LAMBDA_IDENTITY_VALUES=(0.0 0.05 0.1 0.2 0.3 0.5)
mkdir -p "$RESULTS_DIR"

STUDENT_ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

# [SỬA — bổ sung sau code review, vòng 5, điểm 2] Xem giải thích đầy đủ trong
# pipeline/run_prune_sparsity_screen.sh — script này CŨNG dùng checkpoint
# KHÔNG suffix seed. Fail sớm thay vì mất hàng giờ (6 mức x 3 seed = 18 lần
# train SR) rồi mới lỗi ở lần train_recognition.py ĐẦU TIÊN.
LR_CKPT_NOSUFFIX="runs/recognition_lr_${BACKBONE}/best.pt"
if [ ! -f "$LR_CKPT_NOSUFFIX" ]; then
    echo "LỖI: thiếu $LR_CKPT_NOSUFFIX (checkpoint KHÔNG suffix seed)."
    echo "     -> chạy 'python train_recognition.py --config $CONFIG --domain lr"
    echo "        --backbone $BACKBONE' (xem pipeline/03_train_baseline_recognition.sh)"
    echo "        trước — KHÁC với checkpoint '_seed<N>' của run_multi_seed.sh."
    exit 1
fi

# Các mức lambda_identity>0 cần 3 judge HR — kiểm tra NGAY, không để 3 seed
# ở mức 0.0 chạy xong rồi mới chết ở 0.05.
source "$(dirname "$0")/_check_hr_judges.sh"
check_hr_judges

for LAMBDA_ID in "${LAMBDA_IDENTITY_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        TAG="lid${LAMBDA_ID}_seed${SEED}"
        echo "################################################################"
        echo "# lambda_identity=$LAMBDA_ID | seed=$SEED"
        echo "################################################################"

        # 1) Train student với đúng seed + lambda này
        # [SỬA — confound phát hiện qua code review] PIN TƯỜNG MINH
        # lambda_feat=0/lambda_saliency=0 — script này chỉ quét lambda_identity,
        # KHÔNG được để 2 cơ chế feature-KD/saliency (thêm sau) âm thầm bật lên
        # qua default config.yaml (dù default hiện đã là 0.0, pin rõ ở đây để
        # script này KHÔNG BAO GIỜ phụ thuộc vào default tương lai của config).
        python train_sr_distill.py --config "$CONFIG" \
            --lambda_pixel 1.0 --lambda_distill 1.0 --lambda_feat 0 --lambda_saliency 0 \
            --lambda_identity "$LAMBDA_ID" \
            --seed "$SEED" --run_suffix "_${TAG}"

        # 2) Sinh ảnh SR từ student vừa train
        python data/build_sr.py --lr_dir splits/lr \
            --sr_ckpt "runs/sr_improved_${STUDENT_ARCH}_${TAG}/best.pt" \
            --arch "$STUDENT_ARCH" --scale "$SCALE" \
            --out_dir "splits/sr_sweep_${TAG}"

        # 3) Train recognition trên ảnh vừa sinh (từ checkpoint LR gốc, seed khớp)
        # Không cần --run_suffix riêng vì domain "sr_sweep_${TAG}" đã là duy nhất
        # (đã gồm cả lambda lẫn seed trong tên).
        python train_recognition.py --config "$CONFIG" --domain "sr_sweep_${TAG}" \
            --backbone "$BACKBONE" --init_ckpt "runs/recognition_lr_${BACKBONE}/best.pt" \
            --seed "$SEED"

        # 4) Đo cả accuracy lẫn chất lượng ảnh (PSNR/SSIM) cho đủ số liệu
        python eval_recognition.py --config "$CONFIG" \
            --ckpt "runs/recognition_sr_sweep_${TAG}_${BACKBONE}/best.pt" \
            --backbone "$BACKBONE" --train_domain "sr_sweep_${TAG}" --test_domain "sr_sweep_${TAG}" \
            --out_json "$RESULTS_DIR/acc_${TAG}.json"

        python eval_sr_quality.py --config "$CONFIG" --arch "$STUDENT_ARCH" \
            --ckpt "runs/sr_improved_${STUDENT_ARCH}_${TAG}/best.pt" \
            --label "lid${LAMBDA_ID}_seed${SEED}" \
            --out_csv "$RESULTS_DIR/sr_quality_sweep.csv"
    done
done

echo ""
echo ">>> Tổng hợp — tính trung bình +- độ lệch chuẩn cho từng mức lambda_identity..."
python data/aggregate_lambda_sweep.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/lambda_sweep_summary.csv"

echo ""
echo "HOÀN TẤT. Xem $RESULTS_DIR/lambda_sweep_summary.csv để chọn lambda_identity tối ưu."
echo ""
echo "!!! [SỬA — hướng dẫn cũ SAI, phát hiện qua review Q1] KHÔNG chỉ đổi"
echo "    lambda_identity trong configs/config.yaml rồi chạy thẳng multi-seed —"
echo "    pipeline/run_multi_seed.sh KHÔNG train lại SR, chỉ tái sử dụng ảnh"
echo "    splits/sr_improved đã sinh sẵn (từ pipeline/06_improve_span.sh, vốn"
echo "    PIN CỨNG lambda_identity=0 qua CLI, không đọc default config.yaml)."
echo "    Đổi config.yaml một mình sẽ KHÔNG có tác dụng gì — bảng kết quả chính"
echo "    vẫn phản ánh recipe pixel+distill cũ (lambda_identity=0), không phải"
echo "    lambda tối ưu vừa chọn, mà KHÔNG có lỗi/cảnh báo nào báo hiệu."
echo "    QUY TRÌNH ĐÚNG: (1) sửa lambda_identity trong configs/config.yaml,"
echo "    (2) sửa CÙNG giá trị đó vào dòng pin CLI trong"
echo "    pipeline/06_improve_span.sh, (3) chạy lại 06_improve_span.sh (train"
echo "    lại SR + sinh lại splits/sr_improved), (4) rồi MỚI chạy Bước 2"
echo "    (pipeline/run_multi_seed.sh, multi-seed x 5 backbone) trên ảnh SR MỚI."
