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

# [SỬA — 2026-08-25] TRƯỚC ĐÂY (2026-08-24) chặn cứng nếu $RESULTS_DIR không
# rỗng, để tránh 2 recipe khác nhau lẫn nhãn trong sr_quality_sweep.csv (xem
# sự cố đã xảy ra thật, đã dọn dữ liệu bị lẫn). Chốt đó giờ MÂU THUẪN trực
# tiếp với logic "bỏ qua tổ hợp đã xong" thêm bên dưới (cần $RESULTS_DIR có
# sẵn dữ liệu từ lần chạy TRƯỚC ĐÓ, đúng recipe, để biết tổ hợp nào khỏi train
# lại) — giữ cả 2 sẽ khiến script LUÔN thoát lỗi ngay từ đầu, không bao giờ
# resume được. Bỏ chặn cứng ở đây: rủi ro nhãn trùng lẫn recipe giờ đã được
# data/check_duplicate_labels.py đảm nhiệm — cổng chặn CHUNG, bắt buộc, chạy
# ngay trước khi tổng hợp báo cáo cuối (xem pipeline/run_everything_from_scratch.sh),
# bắt được ở BẤT KỲ CSV/script nào, không chỉ riêng script này.
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

        # [MỚI — 2026-08-25, sự cố thật] Bỏ qua tổ hợp ĐÃ CHẠY XONG THẬT SỰ —
        # script này chạy rất lâu (early-stopping thiếu min_delta khiến 1 tổ
        # hợp có thể chạy gần hết max_epochs, xem utils/early_stopping.py),
        # nếu phải dừng giữa chừng thì KHÔNG được train lại từ đầu các tổ hợp
        # đã xong — số liệu của chúng vẫn ĐÚNG, train lại là lãng phí GPU.
        #
        # [SỬA — 2026-08-25, review lại] Bước 3 (eval_recognition.py, ghi
        # acc_<TAG>.json) và Bước 4 (eval_sr_quality.py, ghi APPEND vào
        # sr_quality_sweep.csv) là 2 LỆNH PYTHON RIÊNG — nếu 1 lần dừng trước
        # đó rơi đúng giữa 2 lệnh này, acc_<TAG>.json tồn tại nhưng THIẾU dòng
        # CSV tương ứng, chỉ kiểm tra 1 file sẽ bỏ qua NHẦM tổ hợp còn thiếu
        # nửa dữ liệu mà không cảnh báo. Kiểm tra ĐỦ CẢ 2: acc JSON tồn tại VÀ
        # có dòng nhãn khớp trong sr_quality_sweep.csv (cột đầu tiên, đúng
        # label="lsal<λ>_seed<seed>" mà eval_sr_quality.py ghi ở Bước 4).
        if [ -f "$RESULTS_DIR/acc_${TAG}.json" ] \
           && [ -f "$RESULTS_DIR/sr_quality_sweep.csv" ] \
           && grep -q "^lsal${LAMBDA_SAL}_seed${SEED}," "$RESULTS_DIR/sr_quality_sweep.csv"; then
            echo ">>> Bỏ qua $TAG (đã có đủ acc_${TAG}.json + dòng CSV từ lần chạy trước)"
            continue
        fi

        echo "################################################################"
        echo "# lambda_saliency=$LAMBDA_SAL (feat=$LAMBDA_FEAT_FIXED identity=$LAMBDA_IDENTITY_FIXED) | seed=$SEED"
        echo "################################################################"

        # 1) Train student với đúng seed + lambda_saliency này, giữ nguyên
        # feat/identity ở cấu hình đã chốt từ run_ablation_kd_v2.sh.
        # --min_delta 0.001: xem utils/early_stopping.py — với lambda_saliency>0,
        # val_total có thể giảm đơn điệu cực nhỏ dần (diminishing returns) khiến
        # early-stop KHÔNG BAO GIỜ kích hoạt thật (counter luôn reset về 0 dù cải
        # thiện không đáng kể), chạy tới tận max_epochs=500 thay vì dừng sớm thật
        # như đã thấy với lambda_saliency=0.0 (~25-40 epoch). Xác nhận bằng mô
        # phỏng lại đúng pattern quan sát được (val_total giảm 0.1261->0.1259 ở
        # epoch 398->400): với min_delta=0.001, dừng ở epoch 99 thay vì chạy hết
        # 500; với chuỗi hội tụ thật (không phải diminishing-returns giả), KHÔNG
        # cắt ngang quá trình hội tụ (dừng đúng patience epoch sau khi thật sự
        # không còn cải thiện đáng kể). min_delta mặc định 0.0 ở MỌI script khác
        # trong project — thay đổi này CHỈ áp dụng ở đây.
        python train_sr_distill.py --config "$CONFIG" \
            --lambda_pixel 1.0 --lambda_distill 1.0 \
            --lambda_feat "$LAMBDA_FEAT_FIXED" --lambda_identity "$LAMBDA_IDENTITY_FIXED" \
            --lambda_saliency "$LAMBDA_SAL" --min_delta 0.001 \
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
