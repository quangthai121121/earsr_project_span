#!/bin/bash
# [MỚI — phát hiện qua review Q1, MỞ RỘNG theo yêu cầu reviewer vòng sau]
# Đo phương sai do CHÍNH SEED TRAIN SR, tách biệt khỏi phương sai downstream
# (recognition seed) đã đo ở pipeline/run_multi_seed.sh.
#
# BỐI CẢNH: toàn bộ pipeline chính chỉ train SR (span_tiny/span_official/
# span_large) ĐÚNG 1 LẦN (seed=42 mặc định config.yaml) rồi tái sử dụng ảnh SR
# sinh ra cho MỌI seed downstream (recognition). Nghĩa là mọi kết luận thống
# kê hiện có chỉ điều kiện trên 1 checkpoint SR cụ thể — nếu checkpoint đó
# "may/xui" ở seed=42, hiệu ứng quan sát được có thể phản ánh riêng lần chạy
# đó chứ không phải đặc tính kiến trúc.
#
# [MỞ RỘNG] Reviewer yêu cầu tường minh: "thực hiện ít nhất 3 seed cho các mô
# hình SR CHÍNH (span_tiny, span_baseline, span_large)" — KHÔNG chỉ 1 kiến
# trúc đại diện như bản trước. Script này giờ THAM SỐ HOÁ theo kiến trúc
# (đối số thứ 2, xem DÙNG bên dưới) — GỌI LẠI 3 LẦN, mỗi lần 1 kiến trúc, để
# phủ đủ ma trận reviewer yêu cầu. Không gộp cả 3 vào 1 lần chạy vì mỗi kiến
# trúc dùng script train khác nhau (train_sr_distill.py cho span_tiny/
# span_large, train_sr.py cho span_baseline) và cần results_dir riêng (tránh
# đụng tên file JSON — cùng backbone "mobilenet_v2" cho cả 3 kiến trúc).
#
# THIẾT KẾ: cô lập ĐÚNG 1 biến — chỉ đổi seed lúc TRAIN SR, GIỮ NGUYÊN seed
# downstream (recognition) cố định ở RECOGNITION_SEED cho cả 3 lần — nếu để
# seed downstream cũng đổi theo, không tách được phương sai do SR-seed ra
# khỏi phương sai downstream đã biết.
#
# QUAN TRỌNG — recipe PIN CỨNG cho span_tiny (khớp CHÍNH XÁC
# pipeline/06_improve_span.sh): lambda_pixel=1.0, lambda_distill=1.0,
# lambda_feat=0, lambda_saliency=0, lambda_identity=0. span_baseline
# (train_sr.py) và span_large KHÔNG pin lambda — khớp đúng cách 2 kiến trúc
# đó được train ở pipeline chính (04_train_teacher_and_span_baseline.sh
# không nhận lambda nào; RUN_ALL_span_large_ablation.sh CỐ Ý đọc lambda trực
# tiếp từ config, không pin, xem giải thích trong chính file đó). Nếu SAU NÀY
# bảng kết quả chính đổi recipe (span_tiny đổi lambda, hoặc span_large/
# span_baseline đổi cách gọi), PHẢI SỬA LẠI khớp trước khi chạy — nếu không,
# script này đo phương sai của 2 RECIPE KHÁC NHAU thay vì phương sai của CÙNG
# 1 recipe qua các seed, làm sai lệch hoàn toàn kết luận.
#
# TIỀN ĐỀ: đã chạy xong pipeline chính (Bước 1-7, xem RUNBOOK_EarVN1.0.md)
# cho $CONFIG — cần sẵn có (tuỳ ARCH được chọn):
#   span_tiny:     <runs_root>/sr_improved_span_tiny/best.pt (seed=42, TÁI SỬ DỤNG)
#                  <splits_root>/sr_improved/                 (ảnh SR seed=42, TÁI SỬ DỤNG)
#   span_baseline: <runs_root>/sr_span_official/best.pt       (seed=42, TÁI SỬ DỤNG)
#                  <splits_root>/sr_baseline/                 (ảnh SR seed=42, TÁI SỬ DỤNG)
#   span_large:    <runs_root>/sr_improved_span_large/best.pt (seed=42, TÁI SỬ DỤNG)
#                  <splits_root>/sr_span_large/                (ảnh SR seed=42, TÁI SỬ DỤNG)
#   + <runs_root>/recognition_lr_<backbone>[_seed42]/best.pt   (init cho fine-tune, mọi ARCH)
#   + (chỉ span_baseline) checkpoints/span_pretrained_x4.pth
#
# DÙNG:
#   bash pipeline/run_sr_seed_variance.sh [đường_dẫn_config] [kiến_trúc]
#   (config: không truyền -> mặc định configs/config.yaml, khớp EarVN1.0;
#    truyền configs/config_awex.yaml để chạy cho AWEx — script tham số hoá
#    qua config, dùng CHUNG cho cả 2 dataset như các RUN_ALL_*.sh khác)
#   (kiến trúc: không truyền -> mặc định span_tiny, GIỮ NGUYÊN hành vi bản
#    trước để không phá lệnh gọi cũ. Reviewer yêu cầu đủ 3 kiến trúc -> chạy
#    LẦN LƯỢT 3 lệnh:
#       bash pipeline/run_sr_seed_variance.sh configs/config.yaml span_tiny
#       bash pipeline/run_sr_seed_variance.sh configs/config.yaml span_baseline
#       bash pipeline/run_sr_seed_variance.sh configs/config.yaml span_large
#
# CẢNH BÁO THỜI GIAN: mỗi lần gọi = 2 lần train SR mới (seed 123, 2024; seed
# 42 tái sử dụng) + 3 lần fine-tune recognition (1 backbone đại diện) — chạy
# đủ cả 3 kiến trúc tốn gấp ~3 lần 1 lần chạy trước đây.

set -e

CONFIG="${1:-configs/config.yaml}"
ARCH="${2:-span_tiny}"          # span_tiny | span_baseline | span_large
BACKBONE="mobilenet_v2"         # 1 backbone đại diện, giống quy ước screening
                                 # khác trong project (run_ablation.sh,
                                 # run_prune_sparsity_screen.sh) — mở rộng
                                 # sang cả 5 backbone (xem BACKBONES trong
                                 # pipeline/run_multi_seed.sh) nếu muốn bằng
                                 # chứng mạnh hơn cho bài báo, tốn thêm ~5x.
RECOGNITION_SEED=42             # CỐ ĐỊNH — cô lập đúng biến SR-seed
SR_SEEDS=(42 123 2024)
NUM_WORKERS="${NUM_WORKERS:-4}"

if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không thấy $CONFIG"
    exit 1
fi

# [MỚI — mở rộng đa kiến trúc] STUDENT_ARCH = tên arch KỸ THUẬT truyền cho
# --student_arch/--sr_arch/--arch của các script train/eval (khác với ARCH ở
# trên, vốn là tên bài báo dùng để đặt tên results_dir/label cho dễ đọc).
case "$ARCH" in
    span_tiny)
        STUDENT_ARCH="span_tiny"
        TRAIN_SCRIPT="distill"
        BASE_RUN_DIR="sr_improved_span_tiny"
        BASE_DOMAIN="sr_improved"
        ;;
    span_baseline)
        STUDENT_ARCH="span_official"
        TRAIN_SCRIPT="baseline"
        BASE_RUN_DIR="sr_span_official"
        BASE_DOMAIN="sr_baseline"
        if [ ! -f "checkpoints/span_pretrained_x4.pth" ]; then
            echo "LỖI: thiếu checkpoints/span_pretrained_x4.pth (checkpoint SPAN chính thức, "
            echo "     cần để fine-tune span_baseline, xem scripts/setup_span_official.sh)." >&2
            exit 1
        fi
        ;;
    span_large)
        STUDENT_ARCH="span_large"
        TRAIN_SCRIPT="distill"
        BASE_RUN_DIR="sr_improved_span_large"
        BASE_DOMAIN="sr_span_large"
        ;;
    *)
        echo "LỖI: kiến trúc '$ARCH' không hợp lệ — phải là span_tiny|span_baseline|span_large" >&2
        exit 1
        ;;
esac

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
SPLITS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['splits_root'])")
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

# [MỚI — mở rộng đa kiến trúc] results_dir RIÊNG cho mỗi ARCH — bắt buộc,
# tránh đụng tên file "<backbone>_srseed<seed>.json" (backbone giống nhau =
# mobilenet_v2 cho cả 3 kiến trúc, nếu dùng chung 1 thư mục sẽ ghi đè lẫn
# nhau giữa các lần gọi cho 3 kiến trúc khác nhau).
RESULTS_DIR="${RESULTS_ROOT}/sr_seed_variance_${ARCH}"
mkdir -p "$RESULTS_DIR"

BASE_SR_CKPT="${RUNS_ROOT}/${BASE_RUN_DIR}/best.pt"
if [ ! -f "$BASE_SR_CKPT" ]; then
    echo "LỖI: không thấy $BASE_SR_CKPT — chạy xong pipeline chính cho $ARCH "
    echo "     (config $CONFIG) trước khi chạy script này."
    exit 1
fi

echo "================================================================"
echo "Config          = $CONFIG"
echo "Kiến trúc (ARCH)= $ARCH  (--arch kỹ thuật = $STUDENT_ARCH)"
echo "Backbone        = $BACKBONE (đại diện)"
echo "SR seeds        = ${SR_SEEDS[*]}"
echo "Downstream seed = $RECOGNITION_SEED (CỐ ĐỊNH cho cả 3 lần)"
echo "Results dir     = $RESULTS_DIR"
echo "================================================================"

for SEED in "${SR_SEEDS[@]}"; do
    RUN_SUFFIX_SR="_srseed${SEED}"
    DOMAIN_NAME="${BASE_DOMAIN}_srseed${SEED}"
    SR_DOMAIN_DIR="${SPLITS_ROOT}/${DOMAIN_NAME}"

    if [ "$SEED" -eq 42 ]; then
        SR_CKPT="$BASE_SR_CKPT"
        DOMAIN_NAME="$BASE_DOMAIN"
        SR_DOMAIN_DIR="${SPLITS_ROOT}/${BASE_DOMAIN}"
        echo ""
        echo "################################################################"
        echo "# SR seed=42 ($ARCH): TÁI SỬ DỤNG checkpoint/ảnh sẵn có (không train lại) #"
        echo "################################################################"
    else
        SR_CKPT="${RUNS_ROOT}/${BASE_RUN_DIR}${RUN_SUFFIX_SR}/best.pt"

        echo ""
        echo "################################################################"
        echo "# [1/2] Train SR '$ARCH' — seed=$SEED (checkpoint MỚI)           #"
        echo "################################################################"
        if [ "$TRAIN_SCRIPT" = "distill" ]; then
            if [ "$ARCH" = "span_tiny" ]; then
                # [SỬA — bug phát hiện qua review Q1] PIN TƯỜNG MINH khớp CHÍNH
                # XÁC recipe của 06_improve_span.sh (pixel + output-distill
                # thuần, KHÔNG feature-KD/saliency/identity) — nếu không pin,
                # sau này đổi config.yaml sẽ âm thầm train SR khác recipe với
                # bảng chính, khiến so sánh phương sai SR-seed không còn ý nghĩa.
                python train_sr_distill.py --config "$CONFIG" --student_arch "$STUDENT_ARCH" \
                    --lambda_pixel 1.0 --lambda_distill 1.0 --lambda_feat 0 --lambda_saliency 0 --lambda_identity 0 \
                    --seed "$SEED" --run_suffix "$RUN_SUFFIX_SR" --num_workers "$NUM_WORKERS"
            else
                # span_large: KHÔNG pin lambda — khớp đúng RUN_ALL_span_large_ablation.sh,
                # vốn CỐ Ý đọc lambda trực tiếp từ config (xem giải thích ở đó).
                python train_sr_distill.py --config "$CONFIG" --student_arch "$STUDENT_ARCH" \
                    --seed "$SEED" --run_suffix "$RUN_SUFFIX_SR" --num_workers "$NUM_WORKERS"
            fi
        else
            # span_baseline: train_sr.py, khớp đúng 04_train_teacher_and_span_baseline.sh
            python train_sr.py --config "$CONFIG" --sr_arch "$STUDENT_ARCH" \
                --pretrained_path checkpoints/span_pretrained_x4.pth \
                --seed "$SEED" --run_suffix "$RUN_SUFFIX_SR" --num_workers "$NUM_WORKERS"
        fi

        echo ""
        echo "################################################################"
        echo "# [2/2] Sinh ảnh SR từ checkpoint seed=$SEED                     #"
        echo "################################################################"
        python data/build_sr.py --lr_dir "${SPLITS_ROOT}/lr" --sr_ckpt "$SR_CKPT" \
            --arch "$STUDENT_ARCH" --scale "$SCALE" --out_dir "$SR_DOMAIN_DIR"
    fi

    echo ""
    echo "### Đo chất lượng SR (PSNR/SSIM/LPIPS) — checkpoint seed=$SEED ###"
    python eval_sr_quality.py --config "$CONFIG" --arch "$STUDENT_ARCH" --ckpt "$SR_CKPT" \
        --label "${ARCH}_srseed${SEED}" --out_csv "${RESULTS_DIR}/sr_quality_srseed.csv"

    echo ""
    echo "### Fine-tune recognition (seed downstream CỐ ĐỊNH=$RECOGNITION_SEED) trên checkpoint SR seed=$SEED ###"
    INIT_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${RECOGNITION_SEED}/best.pt"
    if [ ! -f "$INIT_CKPT" ]; then
        INIT_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}/best.pt"
    fi
    if [ ! -f "$INIT_CKPT" ]; then
        echo "LỖI: không thấy checkpoint recognition domain 'lr' cho backbone $BACKBONE "
        echo "     (${RUNS_ROOT}/recognition_lr_${BACKBONE}[_seed${RECOGNITION_SEED}]/best.pt)."
        echo "     Chạy xong Bước 3 (03_train_baseline_recognition.sh) trước."
        exit 1
    fi

    RUN_SUFFIX_REC="_srseed${SEED}"
    python train_recognition.py --config "$CONFIG" --domain "$DOMAIN_NAME" --backbone "$BACKBONE" \
        --init_ckpt "$INIT_CKPT" \
        --seed "$RECOGNITION_SEED" --run_suffix "$RUN_SUFFIX_REC" --num_workers "$NUM_WORKERS"

    python eval_recognition.py --config "$CONFIG" \
        --ckpt "${RUNS_ROOT}/recognition_${DOMAIN_NAME}_${BACKBONE}${RUN_SUFFIX_REC}/best.pt" \
        --backbone "$BACKBONE" --train_domain "$DOMAIN_NAME" --test_domain "$DOMAIN_NAME" \
        --out_json "${RESULTS_DIR}/${BACKBONE}_srseed${SEED}.json" --num_workers "$NUM_WORKERS"
done

echo ""
echo ">>> Tổng hợp phương sai do SR-seed cho $ARCH (tách biệt khỏi phương sai downstream)..."
python data/aggregate_sr_seed_variance.py --results_dir "$RESULTS_DIR" \
    --out_csv "${RESULTS_DIR}/sr_seed_variance_summary.csv" \
    --downstream_multiseed_csv "${RESULTS_ROOT}/multi_seed/multi_seed_summary.csv" \
    --downstream_domain "$BASE_DOMAIN" \
    --sr_quality_csv "${RESULTS_DIR}/sr_quality_srseed.csv"

echo ""
echo "HOÀN TẤT ($ARCH). Kết quả: ${RESULTS_DIR}/sr_seed_variance_summary.csv"
echo "Nếu reviewer yêu cầu đủ 3 kiến trúc, chạy thêm 2 lệnh còn lại (xem DÙNG ở đầu file)."
echo "Dùng số liệu này để viết rõ hơn phần Limitations — so sánh trực tiếp"
echo "phương sai do SR-seed (mới đo) với phương sai downstream đã báo cáo."
