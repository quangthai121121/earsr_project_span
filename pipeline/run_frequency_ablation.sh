#!/bin/bash
# [MỚI — Trụ cột 6, reviewer round 2] Bài báo lập luận (Section~sec:positioning)
# rằng nén sâu an toàn vì tín hiệu nhận dạng tập trung ở tần số THẤP. Thí
# nghiệm duy nhất đã chạy (attention-parameterization ablation) chỉ bác bỏ
# MỘT giả thuyết hẹp hơn (attention cố định), KHÔNG trực tiếp kiểm chứng giả
# thuyết tần số thấp. Script này chạy 2 phần bổ trợ nhau để kiểm chứng trực
# tiếp, KHÔNG train lại gì cả (chỉ đọc lại ảnh/checkpoint đã có sẵn):
#
#   Phần A (data/analyze_spectral_content.py): so sánh phổ công suất giữa
#     HR/no-SR/span_tiny/span_baseline/span_large trên tập ảnh test đã có
#     sẵn -- nén MẤT thông tin ở dải tần số nào? Chạy 1 lần (không phụ thuộc
#     seed/backbone -- chỉ phân tích ảnh, không dùng model nhận dạng).
#
#   Phần B (eval_frequency_ablation.py): lọc bỏ 1 dải tần số khỏi ảnh
#     sr_improved (span_tiny) rồi eval lại checkpoint recognition ĐÃ TRAIN
#     SẴN (recognition_sr_improved_<backbone>_seed<seed>) -- nhận dạng CẦN
#     thông tin ở dải tần số nào?
#
# [SỬA — phát hiện qua tự phản biện trước khi giao, KHÔNG phải sau khi bị
# reviewer chỉ ra] Bản đầu chỉ chạy Phần B trên 1 backbone (MobileNetV2,
# "screening trước, mở rộng sau" -- đúng mẫu depth sweep/degradation-
# robustness). NHƯNG mẫu đó chỉ hợp lý khi mở rộng backbone tốn CHI PHÍ TRAIN
# THẬT (như 2 ablation kia) -- Phần B ở đây KHÔNG train gì cả, chỉ eval
# checkpoint recognition_sr_improved_<backbone>_seed<seed> ĐÃ CÓ SẴN cho
# TOÀN BỘ 11 backbone từ lâu (dùng chung cho Table~tab:main). Hơn nữa, chính
# ablation kiểm chứng cơ chế TRƯỚC ĐÓ trong bài báo này (attention-
# parameterization, Table~tab:attn-ablation) đã chạy ở quy mô ĐỦ 11 backbone,
# không phải 1 backbone đại diện -- giữ Phần B ở 1 backbone sẽ KHÔNG nhất
# quán với chính tiền lệ của bài báo, và lặp lại đúng kiểu thiếu sót reviewer
# đã chỉ ra ở Trụ cột 2/5. Vì chi phí mở rộng gần như 0 (không train lại),
# không có lý do chính đáng để dừng ở 1 backbone -- sửa ngay từ đầu.
#
# DÙNG:
#   bash pipeline/run_frequency_ablation.sh [đường_dẫn_config]
#
# TIỀN ĐỀ: đã chạy xong pipeline chính (domain hr/lr/sr_baseline/sr_improved/
# sr_span_large đã build) VÀ có checkpoint recognition_sr_improved_
# <backbone>_seed<seed>/best.pt cho cả 11 backbone x 5 seed (đã có sẵn từ
# pipeline/run_multi_seed.sh + run_multi_seed_new_backbones.sh -- CHÍNH các
# checkpoint dùng cho Table~tab:main, không cần train gì thêm).
#
# [SỬA — hiệu năng, phát hiện qua tự review] Bộ lọc tần số KHÔNG phụ thuộc
# backbone/seed -- lọc lại trong mỗi lần eval (55 lần) sẽ tính FFT giống hệt
# nhau 55 lần (đo thực tế lãng phí ~17 phút). Giờ tách riêng 1 bước BUILD 16
# domain ảnh đã lọc DUY NHẤT 1 LẦN (data/build_frequency_filtered_domains.py)
# trước vòng lặp backbone/seed -- Phần B chỉ ĐỌC LẠI, không lọc lại.
#
# CẢNH BÁO THỜI GIAN: KHÔNG train gì cả -- 1 lần phân tích ảnh (Phần A) + 1
# lần build 16 domain đã lọc + 11 backbone x 5 seed = 55 lần eval thuần tuý
# (không có backward/optimizer, không lọc lại) -- rẻ hơn NHIỀU so với mọi
# ablation cần train khác trong project này.

set -e

CONFIG="${1:-configs/config.yaml}"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100" \
           "shufflenet_v2_x1_0" "squeezenet1_1" "mobilenet_v3_large" "regnet_y_400mf" \
           "mobileone_s0" "lcnet_100")
SEEDS=(42 123 2024 44 999)
DOMAIN="sr_improved"
RESULTS_DIR="results/frequency_ablation"
NUM_WORKERS="${NUM_WORKERS:-4}"
mkdir -p "$RESULTS_DIR"

echo "################################################################"
echo "# Kiểm tra tiền đề"
echo "################################################################"
MISSING=0
RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
SPLITS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['splits_root'])")
for D in hr lr sr_baseline sr_improved sr_span_large; do
    if [ ! -d "${SPLITS_ROOT}/${D}" ]; then
        echo "LỖI: thiếu ${SPLITS_ROOT}/${D} -> chạy pipeline chính trước."
        MISSING=1
    fi
done
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="${RUNS_ROOT}/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: thiếu $CKPT -> chạy pipeline/run_multi_seed.sh (hoặc _new_backbones.sh) trước."
            MISSING=1
        fi
    done
done
if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "DỪNG LẠI — thiếu tiền đề ở trên."
    exit 1
fi
echo "OK — mọi tiền đề đã sẵn sàng (không train lại gì, chỉ đọc ảnh/checkpoint có sẵn)."

echo ""
echo "################################################################"
echo "# Phần A — Phân tích phổ tần số (1 lần, không phụ thuộc seed/backbone)"
echo "################################################################"
SPECTRAL_CSV="${RESULTS_DIR}/spectral_content.csv"
if [ -f "$SPECTRAL_CSV" ]; then
    echo ">>> Bỏ qua (đã có $SPECTRAL_CSV từ lần chạy trước)"
else
    python data/analyze_spectral_content.py --config "$CONFIG" --out_csv "$SPECTRAL_CSV"
fi

echo ""
echo "################################################################"
echo "# Phần B, bước 0 — Build 16 domain ảnh đã lọc (1 lần, dùng lại cho cả"
echo "# 11 backbone x 5 seed bên dưới -- KHÔNG lọc lại 55 lần, xem chú thích"
echo "# đầu data/build_frequency_filtered_domains.py)"
echo "################################################################"
python data/build_frequency_filtered_domains.py --config "$CONFIG" --source_domain "$DOMAIN"

echo ""
echo "################################################################"
echo "# Phần B — eval recognition đã train sẵn trên domain đã lọc (11 backbone x n=5 seed)"
echo "################################################################"
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUT_CSV="${RESULTS_DIR}/${BACKBONE}_seed${SEED}.csv"
        if [ -f "$OUT_CSV" ]; then
            echo ">>> Bỏ qua backbone=$BACKBONE seed=$SEED (đã có $OUT_CSV)"
            continue
        fi
        echo "----------------------------------------------------------------"
        echo "backbone=$BACKBONE | seed=$SEED"
        echo "----------------------------------------------------------------"
        CKPT="${RUNS_ROOT}/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt"
        python eval_frequency_ablation.py --config "$CONFIG" \
            --ckpt "$CKPT" --backbone "$BACKBONE" --domain "$DOMAIN" \
            --seed_label "$SEED" --num_workers "$NUM_WORKERS" \
            --out_csv "$OUT_CSV"
    done

    echo ">>> Tổng hợp backbone=$BACKBONE (gộp 5 seed thành mean/std mỗi điều kiện lọc)..."
    python data/aggregate_frequency_ablation.py --results_dir "$RESULTS_DIR" \
        --backbone "$BACKBONE" \
        --out_csv "${RESULTS_DIR}/${BACKBONE}_summary.csv"
done

echo ""
echo "HOÀN TẤT. Đọc ${RESULTS_DIR}/spectral_content.csv (Phần A) và"
echo "${RESULTS_DIR}/<backbone>_summary.csv cho từng backbone trong 11 backbone (Phần B,"
echo "mean+-std qua 5 seed mỗi backbone)."
