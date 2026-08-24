#!/bin/bash
# [MỚI] Quét λ_saliency (saliency-weighted identity-critical pixel loss, xem
# compute_multi_judge_saliency trong train_sr_distill.py) theo ĐÚNG mẫu
# run_lambda_sweep.sh (đã dùng cho lambda_identity) — nhiều mức giá trị x
# nhiều seed, có paired t-test + Cohen's d ngay khi tìm điểm tối ưu.
#
# [SỬA — confound phát hiện 2026-08-24] TRƯỚC ĐÂY pin cứng
# LAMBDA_FEAT_FIXED=0.5/LAMBDA_IDENTITY_FIXED=0.1 ("kdv2_full" — cấu hình
# TƯỞNG thắng ở run_ablation_kd_v2.sh khi đó). Sau đó multi-seed n=5 x 5
# backbone (results/multi_seed_kdv2/multi_seed_kdv2_summary_pairwise.csv) xác
# nhận DỨT ĐIỂM: cả feat-KD lẫn identity=0.1 đều KHÔNG giúp ích (chênh lệch
# -0.012..+0.0027, không backbone nào đạt p_raw<0.10). Tiếp tục quét
# lambda_saliency TRÊN NỀN 1 cấu hình đã biết là kém sẽ làm nhiễu (confound)
# kết quả — không đo được saliency loss có tác dụng RIÊNG hay không. Đổi về
# nền sạch (chỉ pixel+distill, đúng recipe span_tiny mặc định) để cô lập đúng
# biến lambda_saliency, cùng triết lý với confound-check đã làm cho learned
# pruning (Mục 13.2b trong pipeline/run_everything_from_scratch.sh).
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

# Nền SẠCH (đúng recipe span_tiny mặc định) — xem giải thích ở trên.
LAMBDA_FEAT_FIXED=0.0
LAMBDA_IDENTITY_FIXED=0.0

mkdir -p "$RESULTS_DIR"

# [MỚI — 2026-08-24, sự cố thật phát hiện khi chạy] eval_sr_quality.py ghi
# APPEND (không overwrite) vào $RESULTS_DIR/sr_quality_sweep.csv, và TAG
# ("lsal<λ>_seed<seed>") KHÔNG hề chứa thông tin lambda_feat/lambda_identity.
# Nếu $RESULTS_DIR còn dữ liệu từ 1 lần chạy TRƯỚC (recipe feat/identity
# KHÁC — ví dụ đã xảy ra thật: 1 lần chạy dở dang dùng feat=0.5/identity=0.1
# bị dừng giữa chừng, sau đó chạy lại với feat=0/identity=0), CSV sẽ có 2
# DÒNG TRÙNG NHÃN với giá trị khác nhau mà KHÔNG có cột nào phân biệt — làm
# sai lệch tổng hợp thống kê mà không có cảnh báo nào. Cùng loại kiểm tra đã
# có ở pipeline/run_prune_sparsity_screen.sh — chặn NGAY từ đầu, bắt buộc dọn
# sạch thư mục cũ trước khi chạy lại.
if [ -d "$RESULTS_DIR" ] && [ -n "$(ls -A "$RESULTS_DIR" 2>/dev/null)" ]; then
    echo "LỖI: $RESULTS_DIR đã tồn tại và không rỗng." >&2
    echo "     eval_sr_quality.py ghi APPEND vào sr_quality_sweep.csv — nếu thư mục" >&2
    echo "     này còn dữ liệu từ lần chạy TRƯỚC (có thể dùng recipe feat/identity" >&2
    echo "     KHÁC lần này), số liệu cũ/mới sẽ LẪN VÀO NHAU (trùng nhãn, khác giá trị)." >&2
    echo "     -> di chuyển thư mục cũ đi trước, ví dụ:" >&2
    echo "        mv $RESULTS_DIR ${RESULTS_DIR}_$(date +%Y%m%d_%H%M%S)" >&2
    echo "     rồi chạy lại script này." >&2
    exit 1
fi

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
