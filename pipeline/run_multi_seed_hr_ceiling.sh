#!/bin/bash
# [MỚI — lấp lỗ hổng "trần HR" reviewer chỉ ra] Domain "hr" (train+eval trực
# tiếp trên ảnh HR gốc, không qua LR/SR) đã tồn tại đầy đủ trong
# train_recognition.py/eval_recognition.py từ đầu dự án, nhưng CHỈ từng chạy
# n=1 seed qua pipeline/03_train_baseline_recognition.sh +
# pipeline/08_benchmark_and_aggregate.sh (bộ pipeline gốc, đánh số 01-08) --
# checkpoint đó (runs/recognition_hr_<backbone>/best.pt, KHÔNG suffix seed)
# chỉ được dùng làm điểm khởi tạo (warm-start) cho domain "lr" trong mọi
# script multi-seed sau này (run_multi_seed.sh, run_multi_seed_new_backbones.sh,
# ...), KHÔNG hề được eval lại ở quy mô multi-seed. Không có file kết quả
# domain=hr nào tồn tại trong results/ hiện tại, và bài báo chưa báo cáo HR ở
# bất kỳ bảng nào -- kể cả cho 5 backbone gốc.
#
# Script này train+eval domain "hr" ở ĐÚNG chuẩn multi-seed (n=5) đã dùng cho
# lr/sr_baseline/sr_improved, cho ĐỦ 12 backbone (5 gốc + 7 mới), để có "trần
# lý thuyết" đúng nghĩa thống kê, không phải chỉ 1 lần chạy.
#
# QUAN TRỌNG (khác biệt với lr/sr_baseline/sr_improved): domain "hr" là domain
# ĐẦU CHUỖI, KHÔNG cần --init_ckpt (xem pipeline/03_train_baseline_recognition.sh
# -- gọi train_recognition.py --domain hr --backbone X mà KHÔNG có --init_ckpt
# nào cả, tự khởi tạo từ backbone pretrained ImageNet nội bộ). Vì vậy script
# này ĐƠN GIẢN hơn các script multi-seed khác, không có bước kiểm tra checkpoint
# tiền đề nào ngoài thư mục splits/hr/ (không phụ thuộc backbone).
#
# QUAN TRỌNG (tránh lỗi thống kê): KHÔNG ghi kết quả hr vào CHUNG
# results/multi_seed/ -- data/aggregate_multi_seed_results.py tính hệ số
# Bonferroni theo SỐ DOMAIN có mặt trong thư mục (family = C(n_domain, 2)).
# Table 3 hiện tại dùng family-of-3 (lr, sr_baseline, sr_improved); nếu trộn
# thêm hr vào CHUNG thư mục đó, mọi p-value ĐÃ BÁO CÁO trong Table 3 sẽ bị
# tính lại thành family-of-6 một cách ÂM THẦM, không ai để ý. Script này ghi
# hr vào thư mục RIÊNG (results/hr_ceiling/), rồi tự gộp bản sao (không phải
# di chuyển) của lr/sr_baseline/sr_improved đã có sẵn + hr mới vào
# results/hr_ceiling/combined/ để phân tích 4-domain RIÊNG BIỆT, không đụng
# vào results/multi_seed/multi_seed_summary.csv gốc.
#
# PHIÊN BẢN THAM SỐ HÓA -- dùng CHUNG cho EarVN1.0 VÀ AWEx (giống quy ước của
# RUN_ALL_span_large_ablation_new_backbones.sh): mọi đường dẫn (runs_root,
# splits_root, results_root) suy ra TỪ CHÍNH CONFIG được truyền vào, không
# hard-code "runs/"/"results/"/"splits/". KHÔNG cần viết script riêng cho
# AWEx -- chỉ cần đổi tham số config.
#
# DÙNG:
#   bash pipeline/run_multi_seed_hr_ceiling.sh [đường_dẫn_config]
#   Không truyền gì -> mặc định configs/config.yaml (EarVN1.0)
#   Cho AWEx: bash pipeline/run_multi_seed_hr_ceiling.sh configs/config_awex.yaml
#
# TIỀN ĐỀ:
#   - <splits_root>/hr/ đã tồn tại (ảnh HR gốc, không phụ thuộc backbone --
#     có sẵn từ đầu dự án cho cả 2 bộ dữ liệu, dùng chung cho
#     pipeline/03_train_baseline_recognition.sh / bước tương đương của AWEx).
#   - Đã chạy xong pipeline/run_multi_seed.sh + run_multi_seed_new_backbones.sh
#     (EarVN1.0) hoặc run_multi_seed_new_backbones_awex.sh (AWEx) -- cần
#     <results_root>/multi_seed/{lr,sr_baseline,sr_improved}_<backbone>_seed<seed>.json
#     cho cả 12 backbone để bước gộp cuối có đủ dữ liệu so sánh.
#
# CẢNH BÁO THỜI GIAN: 12 backbone x 5 seed = 60 lần train+eval recognition.
# Domain hr không phụ thuộc SR nào, không cần train lại SR.

set -e

CONFIG="${1:-configs/config.yaml}"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100" \
           "shufflenet_v2_x1_0" "squeezenet1_1" "mnasnet1_0" "mobilenet_v3_large" \
           "regnet_y_400mf" "mobileone_s0" "lcnet_100")
SEEDS=(42 123 2024 44 999)
NUM_WORKERS="${NUM_WORKERS:-4}"

if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không thấy $CONFIG" >&2
    exit 1
fi

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
SPLITS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['splits_root'])")
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")

RESULTS_DIR="${RESULTS_ROOT}/hr_ceiling"
MULTISEED_DIR="${RESULTS_ROOT}/multi_seed"
mkdir -p "$RESULTS_DIR"

echo "================================================================"
echo "Config     = $CONFIG"
echo "Backbones  = ${BACKBONES[*]} (${#BACKBONES[@]} backbone)"
echo "Seeds      = ${SEEDS[*]}"
echo "Results dir = $RESULTS_DIR"
echo "================================================================"

echo "Kiểm tra tiền đề..."
MISSING=0
if [ ! -d "${SPLITS_ROOT}/hr" ] || [ -z "$(find "${SPLITS_ROOT}/hr" -type f -print -quit 2>/dev/null)" ]; then
    echo "LỖI: ${SPLITS_ROOT}/hr không tồn tại hoặc rỗng -- domain HR chưa được build ảnh." >&2
    MISSING=1
fi
if [ "$MISSING" -eq 1 ]; then
    echo "DỪNG LẠI -- thiếu tiền đề ở trên." >&2
    exit 1
fi
echo "OK -- đủ tiền đề, bắt đầu chạy (domain hr không cần init_ckpt)."

for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUT_JSON="$RESULTS_DIR/hr_${BACKBONE}_seed${SEED}.json"
        if [ -f "$OUT_JSON" ]; then
            echo ">>> Bỏ qua backbone=$BACKBONE seed=$SEED (đã có $OUT_JSON)"
            continue
        fi

        echo ""
        echo "################################################################"
        echo "# backbone=$BACKBONE | seed=$SEED | domain=hr"
        echo "################################################################"
        python train_recognition.py --config "$CONFIG" --domain hr --backbone "$BACKBONE" \
            --seed "$SEED" --run_suffix "_seed${SEED}" --num_workers "$NUM_WORKERS"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "${RUNS_ROOT}/recognition_hr_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain hr --test_domain hr \
            --out_json "$OUT_JSON" --num_workers "$NUM_WORKERS"
    done
done

echo ""
echo "################################################################"
echo "# Gộp lại 'combined' (hr mới + lr/sr_baseline/sr_improved đã có) #"
echo "################################################################"
rm -rf "${RESULTS_DIR}/combined"
mkdir -p "${RESULTS_DIR}/combined"
cp "${RESULTS_DIR}"/hr_*_seed*.json "${RESULTS_DIR}/combined/"
if ls "${MULTISEED_DIR}"/*_seed*.json >/dev/null 2>&1; then
    cp "${MULTISEED_DIR}"/*_seed*.json "${RESULTS_DIR}/combined/"
else
    echo "CẢNH BÁO: không thấy ${MULTISEED_DIR}/*_seed*.json -- chỉ tổng hợp được domain hr,"
    echo "          chạy pipeline/run_multi_seed.sh + run_multi_seed_new_backbones.sh trước"
    echo "          để có đủ lr/sr_baseline/sr_improved cho phép so sánh 4-domain."
fi

python data/aggregate_multi_seed_results.py --results_dir "${RESULTS_DIR}/combined" \
    --out_csv "${RESULTS_DIR}/multi_seed_summary_with_hr.csv"

echo ""
echo "HOÀN TẤT (EarVN1.0, đủ 12 backbone, domain hr thêm vào n=5 seed). Kết quả:"
echo "  - ${RESULTS_DIR}/multi_seed_summary_with_hr.csv (4 domain: hr, lr, sr_baseline, sr_improved)"
echo "  - ${RESULTS_DIR}/multi_seed_summary_with_hr_pairwise.csv (family-of-6, RIÊNG BIỆT với Table 3 gốc)"
echo "LƯU Ý: results/multi_seed/multi_seed_summary.csv (Table 3 gốc, family-of-3) KHÔNG bị thay đổi."
