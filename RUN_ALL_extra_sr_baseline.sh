#!/bin/bash
# ================================================================
# [MỚI — đợt 7, Track A: recipe pixel-loss chuẩn của CHÍNH bài báo gốc]
# CHẠY TỪ ĐẦU ĐẾN CUỐI — Baseline SR nhẹ BÊN NGOÀI họ SPAN
# (rlfn / rlfn_adapted / ecbsr / safmn).
# PHIÊN BẢN THAM SỐ HÓA theo cả TÊN KIẾN TRÚC lẫn CONFIG — dùng CHUNG cho
# EarVN1.0 gốc VÀ bất kỳ dataset thứ 2/3 nào.
#
# [SỬA — lỗi tài liệu đã phát hiện qua review] Bản trước ghi sai: "GIỐNG HỆT
# cách EDSR/span_large đang được đối xử". THỰC TẾ: chỉ EDSR train pixel-loss
# thuần qua train_sr.py như script này — span_large train qua
# train_sr_distill.py (--student_arch span_large), CÙNG recipe distillation
# với span_tiny (xem RUN_ALL_span_large_ablation.sh dòng gọi
# train_sr_distill.py). Nghĩa là: kết quả script NÀY (Track A) KHÔNG so sánh
# công bằng trực tiếp với span_tiny/span_large (khác recipe huấn luyện, chỉ
# khác kiến trúc thôi thì không đủ, còn lẫn biến "có/không distillation").
# Muốn so sánh công bằng CÙNG recipe với span_tiny/span_large, dùng
# RUN_ALL_extra_sr_baseline_distilled.sh (Track B, script riêng) thay vì
# script này. Track A (script này) vẫn hữu ích để đối chiếu số liệu với
# chính bài báo gốc RLFN/ECBSR/SAFMN (cả 3 bài đều tự train pixel-loss
# thuần, không dùng distillation) — chỉ đừng dùng Track A để kết luận
# "kiến trúc nào tốt hơn span_tiny", vì đó là so sánh lẫn 2 biến số.
# ================================================================
#
# DÙNG:
#   bash RUN_ALL_extra_sr_baseline.sh <rlfn|rlfn_adapted|ecbsr|safmn|smfanet> [đường_dẫn_config]
#
#   Không truyền config -> mặc định configs/config.yaml (EarVN1.0):
#       bash RUN_ALL_extra_sr_baseline.sh rlfn
#
#   Chạy cho dataset phụ (ví dụ AWE):
#       bash RUN_ALL_extra_sr_baseline.sh rlfn configs/config_awe.yaml
#
# TIỀN ĐỀ TRƯỚC KHI CHẠY (script tự kiểm tra ở BƯỚC 0, giống
# RUN_ALL_span_large_ablation.sh):
#   - Đã chạy xong pipeline 01-05 CHÍNH của đúng dataset đó (cần
#     splits/lr, splits/hr, checkpoint recognition_lr_<backbone>_seed<seed>)
#   - Đã chạy xong pipeline/run_multi_seed.sh của đúng dataset đó
#
# SCRIPT NÀY LÀM GÌ (mọi đường dẫn suy ra TỪ CONFIG, không hardcode
# "runs/"/"results/" — chạy đúng cho mọi dataset, giống hệt quy ước của
# RUN_ALL_span_large_ablation.sh):
#   0. Kiểm tra tiền đề
#   1. Train <arch> from-scratch (pixel loss L1 thuần túy, qua train_sr.py —
#      ĐÚNG cách EDSR đang được đối xử — xem cảnh báo Track A/B ở đầu file)
#   2. Sinh tập ảnh sr_<arch>
#   3. Đo PSNR/SSIM/params/FLOPs/latency cho <arch> -> append vào
#      <results_root>/sr_quality.csv
#   4. Multi-seed (3 seed x 5 backbone = 15 lần train recognition) trên
#      domain sr_<arch>, tái dùng checkpoint lr_<backbone>_seed<seed> đã có
#   5. Tổng hợp CSV so sánh domain sr_<arch> cùng lr/sr_baseline/sr_improved

set -e

ARCH="$1"
CONFIG="${2:-configs/config.yaml}"

if [ -z "$ARCH" ] || [[ ! "$ARCH" =~ ^(rlfn|rlfn_adapted|ecbsr|safmn|smfanet)$ ]]; then
    echo "LỖI: thiếu hoặc sai tên kiến trúc."
    echo "Dùng: bash RUN_ALL_extra_sr_baseline.sh <rlfn|rlfn_adapted|ecbsr|safmn|smfanet> [đường_dẫn_config]"
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

RESULTS_DIR="${RESULTS_ROOT}/extra_sr_baseline_${ARCH}"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(42 123 2024)

echo "================================================================"
echo "ARCH          = $ARCH"
echo "CONFIG        = $CONFIG"
echo "runs_root     = $RUNS_ROOT"
echo "results_root  = $RESULTS_ROOT"
echo "splits_root   = $SPLITS_ROOT"
echo "================================================================"

echo ""
echo "################################################################"
echo "# BƯỚC 0/5 — Kiểm tra tiền đề trước khi chạy                    #"
echo "################################################################"

MISSING=0
if [ ! -d "${SPLITS_ROOT}/lr" ] || [ ! -d "${SPLITS_ROOT}/hr" ]; then
    echo "LỖI: chưa thấy ${SPLITS_ROOT}/lr hoặc ${SPLITS_ROOT}/hr"
    echo "     -> chạy xong pipeline 01 (đúng CONFIG=$CONFIG) trước."
    MISSING=1
fi
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: chưa thấy $CKPT"
            echo "     -> chạy 'bash pipeline/run_multi_seed.sh' (đúng CONFIG=$CONFIG) trước."
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
echo "# BƯỚC 1/5 — Train $ARCH from-scratch (pixel loss L1, 1 lần)    #"
echo "################################################################"
python train_sr.py --config "$CONFIG" --sr_arch "$ARCH"

echo ""
echo "################################################################"
echo "# BƯỚC 2/5 — Sinh tập ảnh sr_${ARCH}                             #"
echo "################################################################"
python data/build_sr.py --lr_dir "${SPLITS_ROOT}/lr" \
    --sr_ckpt "${RUNS_ROOT}/sr_${ARCH}/best.pt" \
    --arch "$ARCH" --scale "$SCALE" --out_dir "${SPLITS_ROOT}/sr_${ARCH}"

echo ""
echo "################################################################"
echo "# BƯỚC 3/5 — Đo PSNR/SSIM/params/FLOPs/latency cho $ARCH         #"
echo "################################################################"
python eval_sr_quality.py --config "$CONFIG" --arch "$ARCH" \
    --ckpt "${RUNS_ROOT}/sr_${ARCH}/best.pt" --label "$ARCH" \
    --out_csv "${RESULTS_ROOT}/sr_quality.csv"
echo ">>> Đã APPEND dòng '$ARCH' vào ${RESULTS_ROOT}/sr_quality.csv (không ghi đè dòng cũ)."

echo ""
echo "################################################################"
echo "# BƯỚC 4/5 — Multi-seed recognition trên domain sr_${ARCH}       #"
echo "#             (5 backbone x 3 seed = 15 lần, tái dùng ckpt lr)   #"
echo "################################################################"
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        LR_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"

        echo "### backbone=$BACKBONE | seed=$SEED | domain=sr_${ARCH}"
        python train_recognition.py --config "$CONFIG" --domain "sr_${ARCH}" \
            --backbone "$BACKBONE" --init_ckpt "$LR_CKPT" \
            --seed "$SEED" --run_suffix "_seed${SEED}"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "${RUNS_ROOT}/recognition_sr_${ARCH}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "sr_${ARCH}" --test_domain "sr_${ARCH}" \
            --out_json "${RESULTS_DIR}/sr_${ARCH}_${BACKBONE}_seed${SEED}.json"
    done
done

echo ""
echo "################################################################"
echo "# BƯỚC 5/5 — Tổng hợp bảng so sánh cùng lr/sr_baseline/sr_improved #"
echo "################################################################"
mkdir -p "${RESULTS_DIR}/combined"
if ls "${RESULTS_ROOT}/multi_seed"/*_seed*.json >/dev/null 2>&1; then
    cp "${RESULTS_ROOT}/multi_seed"/*_seed*.json "${RESULTS_DIR}/combined/"
else
    echo "CẢNH BÁO: không thấy ${RESULTS_ROOT}/multi_seed/*_seed*.json — chỉ tổng hợp được domain sr_${ARCH}."
fi
cp "${RESULTS_DIR}"/sr_${ARCH}_*_seed*.json "${RESULTS_DIR}/combined/"

python data/aggregate_multi_seed_results.py --results_dir "${RESULTS_DIR}/combined" \
    --out_csv "${RESULTS_DIR}/multi_seed_summary_with_${ARCH}.csv"

echo ""
echo "################################################################"
echo "# HOÀN TẤT (arch: $ARCH | config: $CONFIG)                       #"
echo "################################################################"
echo "Kết quả:"
echo "  - ${RESULTS_ROOT}/sr_quality.csv                                (đã thêm dòng $ARCH)"
echo "  - ${RESULTS_DIR}/multi_seed_summary_with_${ARCH}.csv  (lr/sr_baseline/sr_improved/sr_${ARCH})"
