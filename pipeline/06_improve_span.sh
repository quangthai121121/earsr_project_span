#!/bin/bash
# BƯỚC 6/8: Cải tiến SPAN — NÉN kiến trúc (student nhẹ hơn) qua distillation từ
# SPAN baseline + identity-aware loss (trọng số thấp), sinh tập sr_improved.
#
# [SỬA — confound phát hiện qua code review] PIN TƯỜNG MINH toàn bộ lambda ở
# đây thay vì để train_sr_distill.py tự đọc default trong config.yaml — bước
# này SINH RA checkpoint dùng CHO TOÀN BỘ các bước sau + bảng kết quả chính
# của bài báo (main1.pdf), nên phải rõ ràng 100% recipe nào đang chạy, không
# phụ thuộc "default hiện tại của config.yaml là gì" (từng bị đổi ngầm 1 lần
# khi thêm lambda_feat/lambda_saliency, phá vỡ khả năng tái lập kết quả cũ —
# xem lịch sử sửa lỗi). Recipe dưới đây = ĐÚNG recipe đã dùng cho main1.pdf
# (pixel + output-distill, KHÔNG feature-KD/saliency/identity) — các cơ chế
# MỚI (feat/saliency/multi-judge identity) là ĐỀ XUẤT CẦN KIỂM ĐỊNH RIÊNG qua
# ablation (xem run_ablation_kd_v2.sh, run_lambda_saliency_sweep.sh,
# run_lambda_sweep.sh) trước khi được đưa vào làm recipe CHÍNH THỨC ở đây.
set -e

CONFIG="configs/config.yaml"

echo "################################################################"
echo "# BƯỚC 6/8: Nén SPAN — nhẹ hơn/nhanh hơn baseline (đóng góp chính) #"
echo "################################################################"

echo ">>> Train SPAN đã nén (student_arch)..."
python train_sr_distill.py --config "$CONFIG" \
    --lambda_pixel 1.0 --lambda_distill 1.0 --lambda_feat 0 --lambda_saliency 0 --lambda_identity 0

echo ""
echo ">>> Sinh tập sr_improved..."
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")
STUDENT_ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")
python data/build_sr.py --lr_dir splits/lr --sr_ckpt "runs/sr_improved_${STUDENT_ARCH}/best.pt" \
    --arch "$STUDENT_ARCH" --scale "$SCALE" --out_dir splits/sr_improved

echo ""
echo "HOÀN TẤT BƯỚC 6/8. Chạy tiếp: bash pipeline/07_train_recognition_sr_improved.sh"
