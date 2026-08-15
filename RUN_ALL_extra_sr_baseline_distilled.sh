#!/bin/bash
# ================================================================
# [MỚI — đợt 7, Track B: so sánh CÙNG RECIPE với span_tiny/span_large]
#
# LÝ DO CÓ SCRIPT NÀY (sửa lỗi thiết kế thí nghiệm phát hiện qua review):
# RUN_ALL_extra_sr_baseline.sh (Track A) train rlfn/ecbsr/safmn bằng
# train_sr.py (pixel loss L1 thuần) — ĐÂY LÀ recipe của chính bài báo gốc
# 3 kiến trúc đó, nhưng KHÁC recipe của span_tiny/span_large (cả 2 đều train
# qua train_sr_distill.py: distillation từ teacher span_official + identity
# loss). So sánh thẳng span_tiny (Track A/B trộn) với rlfn/ecbsr/safmn (chỉ
# Track A) là so sánh LẪN LỘN 2 biến số (kiến trúc + recipe), không cô lập
# được riêng "kiến trúc nào tốt hơn". Script này train rlfn_adapted/ecbsr/
# safmn qua ĐÚNG CÙNG recipe train_sr_distill.py (cùng teacher, cùng identity
# loss, cùng lambda trong config) — cô lập biến kiến trúc, so sánh công bằng
# trực tiếp với span_tiny/span_large.
#
# DÙNG:
#   bash RUN_ALL_extra_sr_baseline_distilled.sh <rlfn|rlfn_adapted|ecbsr|safmn|smfanet> [đường_dẫn_config]
#
# TIỀN ĐỀ (giống hệt tiền đề của span_tiny — Bước 6 pipeline chính):
#   - teacher_ckpt (mặc định runs/sr_span_official/best.pt) đã train xong
#     (pipeline/04_train_teacher_and_span_baseline.sh)
#   - frozen_recognition_ckpt (mặc định runs/recognition_hr_mobilenet_v2/best.pt)
#     đã train xong (pipeline/03_train_baseline_recognition.sh)
#   - Đã chạy xong pipeline/run_multi_seed.sh (cần checkpoint lr_<backbone>_seed<seed>)
#
# SCRIPT NÀY LÀM GÌ:
#   1. Train <arch> qua train_sr_distill.py --student_arch <arch> (CÙNG
#      teacher/identity-loss/lambda trong config với span_tiny)
#   2. Sinh tập ảnh sr_improved_<arch>
#   3. Đo PSNR/SSIM/params/FLOPs/latency -> append vào sr_quality.csv,
#      nhãn "<arch>_distilled" (phân biệt với nhãn "<arch>" của Track A)
#   4. Multi-seed recognition (5 backbone x 3 seed = 15 lần) trên domain
#      sr_improved_<arch>
#   5. Tổng hợp CSV so sánh cùng span_tiny/span_large (cùng recipe)

set -e

ARCH="$1"
CONFIG="${2:-configs/config.yaml}"

if [ -z "$ARCH" ] || [[ ! "$ARCH" =~ ^(rlfn|rlfn_adapted|ecbsr|safmn|smfanet)$ ]]; then
    echo "LỖI: thiếu hoặc sai tên kiến trúc."
    echo "Dùng: bash RUN_ALL_extra_sr_baseline_distilled.sh <rlfn|rlfn_adapted|ecbsr|safmn|smfanet> [đường_dẫn_config]"
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không tìm thấy config '$CONFIG'."
    exit 1
fi

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")
SPLITS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['splits_root'])")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")
TEACHER_CKPT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['sr_improve']['teacher_ckpt'])")
FROZEN_REC_CKPT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['sr_improve']['frozen_recognition_ckpt'])")

RESULTS_DIR="${RESULTS_ROOT}/extra_sr_baseline_${ARCH}_distilled"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(42 123 2024)

echo "================================================================"
echo "ARCH (distilled) = $ARCH"
echo "CONFIG            = $CONFIG"
echo "teacher_ckpt      = $TEACHER_CKPT"
echo "frozen_rec_ckpt    = $FROZEN_REC_CKPT"
echo "================================================================"

echo ""
echo "################################################################"
echo "# BƯỚC 0/5 — Kiểm tra tiền đề (giống tiền đề của span_tiny)      #"
echo "################################################################"

MISSING=0
if [ ! -f "$TEACHER_CKPT" ]; then
    echo "LỖI: chưa thấy teacher_ckpt ($TEACHER_CKPT)"
    echo "     -> chạy 'bash pipeline/04_train_teacher_and_span_baseline.sh' trước."
    MISSING=1
fi
if [ ! -f "$FROZEN_REC_CKPT" ]; then
    echo "LỖI: chưa thấy frozen_recognition_ckpt ($FROZEN_REC_CKPT)"
    echo "     -> chạy 'bash pipeline/03_train_baseline_recognition.sh' trước."
    MISSING=1
fi
if [ ! -d "${SPLITS_ROOT}/lr" ] || [ ! -d "${SPLITS_ROOT}/hr" ]; then
    echo "LỖI: chưa thấy ${SPLITS_ROOT}/lr hoặc ${SPLITS_ROOT}/hr"
    MISSING=1
fi
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: chưa thấy $CKPT -> chạy 'bash pipeline/run_multi_seed.sh' trước."
            MISSING=1
        fi
    done
done

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "DỪNG LẠI — thiếu tiền đề ở trên, xem chi tiết phía trên rồi chạy lại."
    exit 1
fi
echo "OK — mọi tiền đề đã sẵn sàng, bắt đầu chạy."
mkdir -p "$RESULTS_DIR"

echo ""
echo "################################################################"
echo "# BƯỚC 1/5 — Train $ARCH qua train_sr_distill.py (CÙNG recipe span_tiny) #"
echo "################################################################"
python train_sr_distill.py --config "$CONFIG" --student_arch "$ARCH"

echo ""
echo "################################################################"
echo "# BƯỚC 2/5 — Sinh tập ảnh sr_improved_${ARCH}                    #"
echo "################################################################"
python data/build_sr.py --lr_dir "${SPLITS_ROOT}/lr" \
    --sr_ckpt "${RUNS_ROOT}/sr_improved_${ARCH}/best.pt" \
    --arch "$ARCH" --scale "$SCALE" --out_dir "${SPLITS_ROOT}/sr_improved_${ARCH}"

echo ""
echo "################################################################"
echo "# BƯỚC 3/5 — Đo PSNR/SSIM/params/FLOPs/latency cho $ARCH (distilled) #"
echo "################################################################"
python eval_sr_quality.py --config "$CONFIG" --arch "$ARCH" \
    --ckpt "${RUNS_ROOT}/sr_improved_${ARCH}/best.pt" --label "${ARCH}_distilled" \
    --out_csv "${RESULTS_ROOT}/sr_quality.csv"
echo ">>> Đã APPEND dòng '${ARCH}_distilled' vào ${RESULTS_ROOT}/sr_quality.csv."

echo ""
echo "################################################################"
echo "# BƯỚC 4/5 — Multi-seed recognition trên domain sr_improved_${ARCH} #"
echo "################################################################"
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        LR_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"

        echo "### backbone=$BACKBONE | seed=$SEED | domain=sr_improved_${ARCH}"
        python train_recognition.py --config "$CONFIG" --domain "sr_improved_${ARCH}" \
            --backbone "$BACKBONE" --init_ckpt "$LR_CKPT" \
            --seed "$SEED" --run_suffix "_seed${SEED}"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "${RUNS_ROOT}/recognition_sr_improved_${ARCH}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "sr_improved_${ARCH}" --test_domain "sr_improved_${ARCH}" \
            --out_json "${RESULTS_DIR}/sr_improved_${ARCH}_${BACKBONE}_seed${SEED}.json"
    done
done

echo ""
echo "################################################################"
echo "# BƯỚC 5/5 — Tổng hợp bảng so sánh CÙNG RECIPE (span_tiny/span_large) #"
echo "################################################################"
mkdir -p "${RESULTS_DIR}/combined"
if ls "${RESULTS_ROOT}/multi_seed"/*_seed*.json >/dev/null 2>&1; then
    cp "${RESULTS_ROOT}/multi_seed"/*_seed*.json "${RESULTS_DIR}/combined/"
fi
cp "${RESULTS_DIR}"/sr_improved_${ARCH}_*_seed*.json "${RESULTS_DIR}/combined/"

python data/aggregate_multi_seed_results.py --results_dir "${RESULTS_DIR}/combined" \
    --out_csv "${RESULTS_DIR}/multi_seed_summary_distilled_${ARCH}.csv"

echo ""
echo "################################################################"
echo "# HOÀN TẤT (arch: $ARCH, recipe: distillation | config: $CONFIG) #"
echo "################################################################"
echo "Kết quả:"
echo "  - ${RESULTS_ROOT}/sr_quality.csv                                       (đã thêm dòng ${ARCH}_distilled)"
echo "  - ${RESULTS_DIR}/multi_seed_summary_distilled_${ARCH}.csv   (so sánh CÙNG recipe với lr/sr_improved [span_tiny])"
echo ""
echo "So sánh CÔNG BẰNG (cùng recipe) với span_tiny: đọc domain 'sr_improved'"
echo "trong ${RESULTS_ROOT}/multi_seed/multi_seed_summary.csv cùng với domain"
echo "'sr_improved_${ARCH}' vừa sinh ra ở trên."
