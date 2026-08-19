#!/bin/bash
# [MỚI] Sàng lọc nhanh hệ số lambda_sparsity cho học pruning độ sâu
# (train_sr_learned_prune.py) — 1 backbone, 1 seed/mức, chỉ để xem:
#   (a) mức sparsity nào cho ra SỐ KHỐI CÒN LẠI hợp lý (so sánh với span_tiny,
#       3/6 khối) sau khi harden_and_export(),
#   (b) accuracy downstream sơ bộ ở mỗi mức.
# ĐÂY LÀ TÍN HIỆU SÀNG LỌC NHANH, không phải bằng chứng cuối — sau khi chọn
# được mức lambda_sparsity, PHẢI chạy pipeline/run_multi_seed_learned_prune.sh
# (multi-seed x 5-backbone) trước khi đưa vào bảng kết quả chính thức.
#
# YÊU CẦU: đã chạy xong pipeline 01-05 + đã có checkpoint recognition domain
# "hr" cho các judge trong configs/config.yaml::sr_improve.identity_judges.

# [SỬA — bổ sung sau code review, vòng 2, điểm 3; LÀM RÕ THÊM ở vòng 5]
# **SỬA 3 THAM SỐ NÀY** trước khi chạy, khớp ĐÚNG cấu hình đã THẮNG ở
# pipeline/run_ablation_kd_v2.sh + run_multi_seed_kdv2.sh (mặc định để 0 —
# TẮT cả 3). LƯU Ý: LAMBDA_FEAT và (LAMBDA_SALIENCY/LAMBDA_IDENTITY) KHÔNG
# cùng vai trò — LAMBDA_FEAT (feature-KD so khớp SR teacher) KHÔNG dùng tín
# hiệu nhận dạng nào, chỉ LAMBDA_SALIENCY/LAMBDA_IDENTITY mới thật sự
# "identity-aware" (lấy tín hiệu từ recognition judge). Nếu CHỈ bật
# LAMBDA_FEAT (2 cái còn lại vẫn =0), script vẫn CHỈ là "reconstruction-aware
# pruning" (có thêm feature-KD), KHÔNG PHẢI "identity-aware pruning" như tên
# gọi/docstring train_sr_learned_prune.py mô tả — xem cảnh báo runtime trong
# train_sr_learned_prune.py::main() (chỉ tắt khi LAMBDA_SALIENCY=LAMBDA_IDENTITY=0,
# không phụ thuộc LAMBDA_FEAT) và cờ "identity_aware"/"uses_feature_kd" riêng
# biệt trong prune_metadata.json.
LAMBDA_FEAT=0.0
LAMBDA_SALIENCY=0.0
LAMBDA_IDENTITY=0.0

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results/prune_sparsity_screen"
BACKBONE="mobilenet_v2"
LAMBDA_SPARSITY_VALUES=(0.0 0.01 0.05 0.1 0.2)
mkdir -p "$RESULTS_DIR"

# [SỬA — lỗi phát hiện qua code review] đọc scale từ config thay vì hardcode 4
# — nhất quán với mọi script khác trong pipeline (nếu config đổi scale, script
# này không bị lệch âm thầm).
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

# [SỬA — bổ sung sau code review, vòng 5, điểm 2] Script này fine-tune
# recognition từ checkpoint KHÔNG suffix "runs/recognition_lr_${BACKBONE}/best.pt"
# (sinh ra bởi pipeline/03_train_baseline_recognition.sh) — KHÁC với checkpoint
# multi-seed "_seed<seed>" mà pipeline/check_prerequisites.sh và
# pipeline/run_multi_seed.sh kiểm tra/tạo ra. Nếu máy CHỈ chạy qua đường
# multi-seed (chỉ có *_seed42/_seed123/..., KHÔNG có bản không suffix), script
# sẽ train xong toàn bộ SR (hàng giờ, tốn GPU) rồi MỚI FileNotFoundError ở
# bước train_recognition.py đầu tiên — kiểm tra NGAY TỪ ĐẦU để fail nhanh.
LR_CKPT_NOSUFFIX="runs/recognition_lr_${BACKBONE}/best.pt"
if [ ! -f "$LR_CKPT_NOSUFFIX" ]; then
    echo "LỖI: thiếu $LR_CKPT_NOSUFFIX (checkpoint KHÔNG suffix seed, dùng để"
    echo "     init_ckpt cho recognition fine-tune ở script này)."
    echo "     -> chạy 'python train_recognition.py --config $CONFIG --domain lr"
    echo "        --backbone $BACKBONE' (không --seed/--run_suffix, xem"
    echo "        pipeline/03_train_baseline_recognition.sh) trước, KHÁC với"
    echo "        checkpoint '_seed<N>' mà pipeline/run_multi_seed.sh tạo ra."
    exit 1
fi

# Nếu user đã sửa LAMBDA_SALIENCY/IDENTITY > 0 thì cần judge TRƯỚC khi train SR.
if python -c "import sys; sys.exit(0 if float('$LAMBDA_SALIENCY') > 0 or float('$LAMBDA_IDENTITY') > 0 else 1)"; then
    source "$(dirname "$0")/_check_hr_judges.sh"
    check_hr_judges
fi

for LS in "${LAMBDA_SPARSITY_VALUES[@]}"; do
    TAG="lsp${LS}"
    echo "################################################################"
    echo "# lambda_sparsity=$LS"
    echo "################################################################"

    python train_sr_learned_prune.py --config "$CONFIG" \
        --lambda_feat "$LAMBDA_FEAT" --lambda_saliency "$LAMBDA_SALIENCY" \
        --lambda_identity "$LAMBDA_IDENTITY" \
        --lambda_sparsity "$LS" --run_suffix "_${TAG}"

    N_BLOCKS=$(python -c "import json; print(json.load(open('runs/sr_learned_prune_${TAG}/prune_metadata.json'))['n_blocks_kept'])")
    echo "-> lambda_sparsity=$LS giữ lại $N_BLOCKS khối (xem runs/sr_learned_prune_${TAG}/prune_metadata.json)"
    # [SỬA — bổ sung sau code review, vòng 4, điểm 2] span_tiny cố định 3/6 khối —
    # chỉ mức lambda_sparsity nào cho N_BLOCKS==3 mới "cùng kích thước" với span_tiny.
    if [ "$N_BLOCKS" -eq 3 ]; then
        echo "   (N_BLOCKS=3 -> CÙNG kích thước với span_tiny, ứng viên tốt để chốt lambda_sparsity)"
    fi

    python data/build_sr.py --lr_dir splits/lr \
        --sr_ckpt "runs/sr_learned_prune_${TAG}/best.pt" \
        --arch span_pruned --n_blocks "$N_BLOCKS" --scale "$SCALE" \
        --out_dir "splits/sr_prune_screen_${TAG}"

    python train_recognition.py --config "$CONFIG" --domain "sr_prune_screen_${TAG}" \
        --backbone "$BACKBONE" --init_ckpt "runs/recognition_lr_${BACKBONE}/best.pt"

    python eval_recognition.py --config "$CONFIG" \
        --ckpt "runs/recognition_sr_prune_screen_${TAG}_${BACKBONE}/best.pt" \
        --backbone "$BACKBONE" --train_domain "sr_prune_screen_${TAG}" \
        --test_domain "sr_prune_screen_${TAG}" \
        --out_json "$RESULTS_DIR/acc_${TAG}_nblocks${N_BLOCKS}.json"

    python eval_sr_quality.py --config "$CONFIG" --arch span_pruned --n_blocks "$N_BLOCKS" \
        --ckpt "runs/sr_learned_prune_${TAG}/best.pt" \
        --label "prune_${TAG}_nblocks${N_BLOCKS}" \
        --out_csv "$RESULTS_DIR/sr_quality_screen.csv"
done

echo ""
echo "HOÀN TẤT sàng lọc lambda_sparsity. Xem:"
echo "  - $RESULTS_DIR/acc_*.json          (accuracy downstream mỗi mức, tên file đã gồm n_blocks_kept)"
echo "  - $RESULTS_DIR/sr_quality_screen.csv (PSNR/SSIM/params/FLOPs/latency mỗi mức)"
echo "Chọn mức lambda_sparsity cho số khối + accuracy hợp lý nhất, rồi SỬA"
echo "lambda_sparsity mặc định trong configs/config.yaml, sau đó chạy"
echo "pipeline/run_multi_seed_learned_prune.sh để validate đầy đủ trước khi"
echo "đưa vào bảng kết quả chính thức."
