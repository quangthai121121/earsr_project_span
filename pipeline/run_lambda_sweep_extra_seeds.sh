#!/bin/bash
# [MỚI — sửa lỗi số liệu thật + trả lời phản biện] pipeline/run_lambda_sweep.sh
# chạy n=3 seed (42,123,2024) cho 6 mức lambda_identity -- kết quả THẬT
# (results/lambda_sweep/lambda_sweep_summary.csv) cho thấy KHÔNG mức nào đạt ý
# nghĩa Bonferroni ở n=3, dù hướng cực kỳ nhất quán (100% lambda>0 đều thấp
# hơn baseline). Bài báo hiện đang trích dẫn "d=-10.09, p=0.0163" cho
# lambda_identity=0.3 -- một con số ĐÃ XÁC NHẬN LÀ SAI: đối chiếu git log,
# claim này được commit vào paper/main.tex lúc 22/8 15:35, TRƯỚC KHI chính
# thí nghiệm lambda=0.3 chạy xong (22/8 22:05) -- không thể là số liệu thật
# của lần chạy này, chưa từng được verify.
#
# Theo đúng quy ước n=3->n=5 đã dùng nhất quán cho mọi ablation khác trong
# project (xem RUNBOOK), mở rộng thêm 2 seed (44, 999) cho CẢ 6 mức lambda
# trước khi viết lại claim này trong bài -- không thay thế số liệu sai bằng
# số liệu n=3 khi n=5 là chuẩn đã dùng cho phần còn lại của bài.
#
# KHÔNG train lại 3 seed cũ (42,123,2024) -- CHỈ thêm 2 seed mới, dùng ĐÚNG
# quy ước init checkpoint của script gốc: checkpoint recognition_lr KHÔNG
# hậu tố seed (dùng chung cho mọi seed/lambda trong sweep này -- khác với quy
# ước "mỗi seed dùng checkpoint lr của chính seed đó" ở run_multi_seed.sh,
# đây là lựa chọn thiết kế CÓ CHỦ ĐÍCH của bản gốc cho bước sàng lọc, giữ
# nguyên để không đổi ý nghĩa so sánh giữa 3 seed cũ và 2 seed mới).
#
# DÙNG:
#   bash pipeline/run_lambda_sweep_extra_seeds.sh [đường_dẫn_config]
#
# TIỀN ĐỀ: đã chạy xong pipeline/run_lambda_sweep.sh (n=3, có sẵn
# results/lambda_sweep/*.json) cho CÙNG config.

set -e

CONFIG="${1:-configs/config.yaml}"
RESULTS_DIR="results/lambda_sweep"
BACKBONE="mobilenet_v2"
SEEDS=(44 999)
LAMBDA_IDENTITY_VALUES=(0.0 0.05 0.1 0.2 0.3 0.5)
mkdir -p "$RESULTS_DIR"

STUDENT_ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

LR_CKPT_NOSUFFIX="runs/recognition_lr_${BACKBONE}/best.pt"
if [ ! -f "$LR_CKPT_NOSUFFIX" ]; then
    echo "LỖI: thiếu $LR_CKPT_NOSUFFIX (checkpoint KHÔNG suffix seed)."
    echo "     -> chạy 'python train_recognition.py --config $CONFIG --domain lr"
    echo "        --backbone $BACKBONE' (xem pipeline/03_train_baseline_recognition.sh) trước."
    exit 1
fi
if [ ! -d "results/lambda_sweep" ] || [ ! -f "results/lambda_sweep/lambda_sweep_summary.csv" ]; then
    echo "LỖI: chưa thấy results/lambda_sweep/lambda_sweep_summary.csv -> chạy"
    echo "     'bash pipeline/run_lambda_sweep.sh $CONFIG' (bản n=3 gốc) trước."
    exit 1
fi

source "$(dirname "$0")/_check_hr_judges.sh"
check_hr_judges

echo "================================================================"
echo "Config     = $CONFIG"
echo "Lambda values = ${LAMBDA_IDENTITY_VALUES[*]}"
echo "Seeds THÊM = ${SEEDS[*]} (giữ nguyên 42,123,2024 đã có)"
echo "================================================================"

for LAMBDA_ID in "${LAMBDA_IDENTITY_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        TAG="lid${LAMBDA_ID}_seed${SEED}"
        OUT_JSON="$RESULTS_DIR/acc_${TAG}.json"
        SR_CKPT="runs/sr_improved_${STUDENT_ARCH}_${TAG}/best.pt"

        # [SỬA — phát hiện qua review, bản vá lần trước CHƯA ĐỦ] Trước đây
        # dùng "continue" ngay khi OUT_JSON đã tồn tại -- điều này nhảy qua
        # LUÔN CẢ bước kiểm tra eval_sr_quality bên dưới (dù đã tách logic
        # riêng), vì "continue" thoát khỏi TOÀN BỘ vòng lặp con, không chỉ
        # phần train/eval_recognition. Nếu script bị ngắt giữa
        # eval_recognition (đã ghi OUT_JSON) và eval_sr_quality, chạy lại sẽ
        # continue ngay từ đầu, khiến dòng PSNR/SSIM của đúng seed đó VĨNH
        # VIỄN bị thiếu mà không có cảnh báo -- đã tái lập bằng test trước
        # khi sửa. Đổi "continue" thành if/else để KHÔNG thoát sớm, luôn đi
        # tiếp xuống bước kiểm tra eval_sr_quality độc lập bên dưới.
        if [ -f "$OUT_JSON" ]; then
            echo ">>> Bỏ qua train+eval_recognition cho lambda_identity=$LAMBDA_ID seed=$SEED (đã có JSON)"
        else
            echo ""
            echo "################################################################"
            echo "# lambda_identity=$LAMBDA_ID | seed=$SEED (THÊM cho n=5)"
            echo "################################################################"

            python train_sr_distill.py --config "$CONFIG" \
                --lambda_pixel 1.0 --lambda_distill 1.0 --lambda_feat 0 --lambda_saliency 0 \
                --lambda_identity "$LAMBDA_ID" \
                --seed "$SEED" --run_suffix "_${TAG}"

            python data/build_sr.py --lr_dir splits/lr \
                --sr_ckpt "$SR_CKPT" \
                --arch "$STUDENT_ARCH" --scale "$SCALE" \
                --out_dir "splits/sr_sweep_${TAG}"

            python train_recognition.py --config "$CONFIG" --domain "sr_sweep_${TAG}" \
                --backbone "$BACKBONE" --init_ckpt "$LR_CKPT_NOSUFFIX" \
                --seed "$SEED"

            python eval_recognition.py --config "$CONFIG" \
                --ckpt "runs/recognition_sr_sweep_${TAG}_${BACKBONE}/best.pt" \
                --backbone "$BACKBONE" --train_domain "sr_sweep_${TAG}" --test_domain "sr_sweep_${TAG}" \
                --out_json "$OUT_JSON"
        fi

        # Kiểm tra eval_sr_quality ĐỘC LẬP với nhánh trên -- luôn chạy tới
        # đây dù train+eval_recognition ở trên bị bỏ qua hay không, để không
        # bao giờ âm thầm bỏ sót 1 dòng PSNR/SSIM (xem giải thích ở trên).
        QUALITY_CSV="$RESULTS_DIR/sr_quality_sweep.csv"
        QUALITY_LABEL="lid${LAMBDA_ID}_seed${SEED}"
        if [ -f "$QUALITY_CSV" ] && grep -q "^${QUALITY_LABEL}," "$QUALITY_CSV"; then
            echo ">>> Bỏ qua eval_sr_quality (đã có dòng ${QUALITY_LABEL} trong $QUALITY_CSV)"
        else
            python eval_sr_quality.py --config "$CONFIG" --arch "$STUDENT_ARCH" \
                --ckpt "$SR_CKPT" \
                --label "$QUALITY_LABEL" \
                --out_csv "$QUALITY_CSV"
        fi
    done
done

echo ""
echo ">>> Tổng hợp lại (giờ đủ n=5 seed: 42,123,2024,44,999) cho cả 6 mức lambda_identity..."
python data/aggregate_lambda_sweep.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/lambda_sweep_summary.csv"

echo ""
echo "HOÀN TẤT (n=5). Kết quả: $RESULTS_DIR/lambda_sweep_summary.csv"
