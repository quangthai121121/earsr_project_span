#!/bin/bash
# ================================================================
# CHẠY TỪ ĐẦU ĐẾN CUỐI — Tier 1 Architecture Ablation (span_large)
# PHIÊN BẢN THAM SỐ HÓA — dùng CHUNG cho EarVN1.0 gốc VÀ bất kỳ dataset
# thứ 2/3 nào đã thiết lập qua scripts/setup_second_dataset_pipeline.sh.
# Không cần sửa file này khi đổi dataset — chỉ đổi tham số dòng lệnh.
# ================================================================
#
# DÙNG:
#   bash RUN_ALL_span_large_ablation.sh [đường_dẫn_config]
#
#   Không truyền gì -> mặc định configs/config.yaml (EarVN1.0):
#       bash RUN_ALL_span_large_ablation.sh
#
#   Chạy cho dataset phụ (ví dụ AWE, sau khi đã chạy
#   scripts/setup_second_dataset_pipeline.sh awe 100):
#       bash RUN_ALL_span_large_ablation.sh configs/config_awe.yaml
#
# CÁCH DÙNG BAN ĐẦU (làm 1 lần, dùng chung cho mọi lần chạy sau):
#   1. Copy 3 file bản đã patch vào ĐÚNG vị trí trong thư mục gốc project:
#         train_sr_distill.py   -> đè lên ./train_sr_distill.py
#         metrics.py            -> đè lên ./utils/metrics.py
#         eval_recognition.py   -> đè lên ./eval_recognition.py
#   2. Đặt file này vào thư mục gốc project.
#
# TIỀN ĐỀ TRƯỚC KHI CHẠY VỚI 1 CONFIG BẤT KỲ (script tự kiểm tra ở BƯỚC 0):
#   - Đã chạy xong pipeline 01-08 CHÍNH của đúng dataset đó
#     (runs_root/sr_improved_span_tiny/best.pt đã tồn tại)
#   - Đã chạy xong run_multi_seed.sh của đúng dataset đó (checkpoint
#     runs_root/recognition_lr_<backbone>_seed<seed>/best.pt đủ 5 backbone x 3 seed)
#
# SCRIPT NÀY LÀM GÌ (tự động, mọi đường dẫn suy ra TỪ CHÍNH CONFIG được truyền
# vào — không hardcode "runs/"/"results/" nên chạy đúng cho mọi dataset):
#   0. Kiểm tra file đã patch đúng chỗ + checkpoint tiền đề đã có đủ
#   1. Train span_large (1 lần, dùng đúng lambda cuối cùng đã chốt trong config)
#   2. Sinh tập ảnh sr_span_large
#   3. Đo PSNR/SSIM/params/FLOPs/latency cho span_large -> append vào <results_root>/sr_quality.csv
#   4. Multi-seed (3 seed x 5 backbone = 15 lần train recognition) trên domain
#      sr_span_large, tái dùng checkpoint lr_<backbone>_seed<seed> đã có sẵn
#   5. Tổng hợp CSV so sánh đủ 4 domain: lr / sr_baseline / sr_improved / sr_span_large

set -e

CONFIG="${1:-configs/config.yaml}"

if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không tìm thấy config '$CONFIG'."
    echo "Dùng: bash RUN_ALL_span_large_ablation.sh [đường_dẫn_config]"
    exit 1
fi

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")
SPLITS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['splits_root'])")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

RESULTS_DIR="${RESULTS_ROOT}/span_large_ablation"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(42 123 2024)

echo "================================================================"
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

if [ ! -f "train_sr_distill.py" ]; then
    echo "LỖI: không thấy ./train_sr_distill.py"; MISSING=1
fi
if ! grep -q "student_arch" train_sr_distill.py 2>/dev/null; then
    echo "LỖI: ./train_sr_distill.py chưa phải bản đã patch (không thấy '--student_arch')."
    echo "     -> copy đè file train_sr_distill.py (bản đã patch) vào đây trước."
    MISSING=1
fi
if [ ! -f "utils/metrics.py" ] || ! grep -q "compute_topk_accuracy" utils/metrics.py 2>/dev/null; then
    echo "LỖI: ./utils/metrics.py chưa phải bản đã patch (không thấy 'compute_topk_accuracy')."
    echo "     -> copy đè file metrics.py (bản đã patch) vào utils/metrics.py trước."
    MISSING=1
fi
if [ ! -f "eval_recognition.py" ] || ! grep -q "all_id_rank5_hits" eval_recognition.py 2>/dev/null; then
    echo "LỖI: ./eval_recognition.py chưa phải bản đã patch (không thấy 'all_id_rank5_hits')."
    echo "     -> copy đè file eval_recognition.py (bản đã patch) vào đây trước."
    MISSING=1
fi
if [ ! -f "${RUNS_ROOT}/sr_improved_span_tiny/best.pt" ]; then
    echo "LỖI: chưa thấy ${RUNS_ROOT}/sr_improved_span_tiny/best.pt"
    echo "     -> chạy xong pipeline 01-08 chính (đúng CONFIG=$CONFIG) trước."
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
echo "# BƯỚC 1/5 — Train span_large (1 lần)                           #"
echo "################################################################"
# [SỬA — bổ sung sau code review, vòng 4, điểm 1] Script này CỐ Ý KHÔNG pin
# cứng lambda_feat/saliency/identity — đúng docstring đầu file: mục đích là
# train span_large bằng "đúng lambda CUỐI CÙNG đã chốt trong config" (tức
# CÙNG recipe với span_tiny ở thời điểm chạy, để cô lập biến kiến trúc, không
# phải cô lập biến recipe). Pin cứng về đây sẽ làm HỎNG chính mục đích so
# sánh công bằng này nếu sau này bạn chốt lambda_feat/saliency>0 cho span_tiny.
# RỦI RO đã ghi nhận: nếu default trong config.yaml đổi SAU KHI script này đã
# chạy 1 lần với giá trị cũ, 2 lần chạy sẽ dùng 2 recipe khác nhau mà không
# có cảnh báo nào — để giảm rủi ro "âm thầm đổi recipe", IN RÕ giá trị lambda
# THẬT SỰ được dùng ngay dưới đây (để bạn ghi lại theo từng lần chạy, phát
# hiện ngay nếu vô tình khác input trước đó).
python -c "
import yaml
ci = yaml.safe_load(open('$CONFIG'))['sr_improve']
print('>>> Lambda THẬT SỰ dùng cho span_large (đọc từ $CONFIG, sr_improve.*):')
for k in ['lambda_pixel', 'lambda_distill', 'lambda_feat', 'lambda_saliency', 'lambda_identity']:
    print(f'    {k} = {ci.get(k, 0.0)}')
"
python train_sr_distill.py --config "$CONFIG" --student_arch span_large

echo ""
echo "################################################################"
echo "# BƯỚC 2/5 — Sinh tập ảnh sr_span_large                          #"
echo "################################################################"
python data/build_sr.py --lr_dir "${SPLITS_ROOT}/lr" \
    --sr_ckpt "${RUNS_ROOT}/sr_improved_span_large/best.pt" \
    --arch span_large --scale "$SCALE" --out_dir "${SPLITS_ROOT}/sr_span_large"

echo ""
echo "################################################################"
echo "# BƯỚC 3/5 — Đo PSNR/SSIM/params/FLOPs/latency cho span_large    #"
echo "################################################################"
python eval_sr_quality.py --config "$CONFIG" --arch span_large \
    --ckpt "${RUNS_ROOT}/sr_improved_span_large/best.pt" --label span_large \
    --out_csv "${RESULTS_ROOT}/sr_quality.csv"
echo ">>> Đã APPEND dòng 'span_large' vào ${RESULTS_ROOT}/sr_quality.csv (không ghi đè dòng cũ)."

echo ""
echo "################################################################"
echo "# BƯỚC 4/5 — Multi-seed recognition trên domain sr_span_large    #"
echo "#             (5 backbone x 3 seed = 15 lần, tái dùng ckpt lr)   #"
echo "################################################################"
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        LR_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"

        echo "### backbone=$BACKBONE | seed=$SEED | domain=sr_span_large"
        python train_recognition.py --config "$CONFIG" --domain sr_span_large \
            --backbone "$BACKBONE" --init_ckpt "$LR_CKPT" \
            --seed "$SEED" --run_suffix "_seed${SEED}"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "${RUNS_ROOT}/recognition_sr_span_large_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain sr_span_large --test_domain sr_span_large \
            --out_json "${RESULTS_DIR}/sr_span_large_${BACKBONE}_seed${SEED}.json"
    done
done

echo ""
echo "################################################################"
echo "# BƯỚC 5/5 — Tổng hợp bảng so sánh đủ 4 domain                   #"
echo "################################################################"
mkdir -p "${RESULTS_DIR}/combined"
if ls "${RESULTS_ROOT}/multi_seed"/*_seed*.json >/dev/null 2>&1; then
    cp "${RESULTS_ROOT}/multi_seed"/*_seed*.json "${RESULTS_DIR}/combined/"
else
    echo "CẢNH BÁO: không thấy ${RESULTS_ROOT}/multi_seed/*_seed*.json — chỉ tổng hợp được domain sr_span_large."
fi
cp "${RESULTS_DIR}"/sr_span_large_*_seed*.json "${RESULTS_DIR}/combined/"

python data/aggregate_multi_seed_results.py --results_dir "${RESULTS_DIR}/combined" \
    --out_csv "${RESULTS_DIR}/multi_seed_summary_4domains.csv"

echo ""
echo "################################################################"
echo "# HOÀN TẤT (config: $CONFIG)                                    #"
echo "################################################################"
echo "Kết quả:"
echo "  - ${RESULTS_ROOT}/sr_quality.csv                              (đã thêm dòng span_large)"
echo "  - ${RESULTS_DIR}/multi_seed_summary_4domains.csv    (lr/sr_baseline/sr_improved/sr_span_large, 5 backbone x 3 seed)"
echo ""
echo "Gửi lại 2 file trên (+ ${RESULTS_ROOT}/multi_seed/multi_seed_summary.csv nếu"
echo "chưa gửi) để viết phần Ablation kiến trúc (mục V.B trong khung bài báo)."
