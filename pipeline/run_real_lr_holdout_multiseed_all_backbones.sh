#!/bin/bash
# [MỚI — mở rộng real_lr_holdout sang cả 12 backbone, theo yêu cầu người
# dùng] pipeline/run_real_lr_holdout_multiseed.sh chỉ chạy đúng 1 backbone
# (mobilenet_v2, hard-code) — đây là 1 trong 3 mục "MobileNetV2-only
# screening" đã xác nhận trước đây (cùng nhóm với λ_identity sweep,
# depth-sweep). Script này CHỦ ĐỘNG mở rộng ra cả 12 backbone, vì đây là bước
# CHỈ EVAL (không train gì cả) -- mọi checkpoint recognition_{lr,sr_baseline,
# sr_improved}_<backbone>_seed<seed> đã có sẵn từ pipeline/run_multi_seed.sh
# (5 backbone gốc) + pipeline/run_multi_seed_new_backbones.sh (7 backbone
# mới), nên chi phí thêm gần như chỉ là thời gian eval, không phải GPU-hour
# train.
#
# QUAN TRỌNG (phát hiện khi đọc lại data/aggregate_real_lr_holdout_multiseed.py
# trước khi viết script này): script tổng hợp đó KHÔNG có khái niệm backbone
# -- nó gộp MỌI file "real_lr_holdout_seed*.csv" tìm thấy trong results_dir
# thành 1 kết quả DUY NHẤT, bất kể backbone nào tạo ra file đó. Nếu chạy
# nhiều backbone vào CHUNG 1 results_dir như file gốc dùng
# (results/real_lr_holdout_multiseed/), số liệu của 12 backbone sẽ bị TRỘN
# LẪN vào 1 con số trung bình vô nghĩa. Để tránh sửa script tổng hợp đã được
# kiểm chứng (rủi ro thấp hơn nhiều so với sửa logic thống kê), mỗi backbone
# ở đây được ghi vào 1 THƯ MỤC RIÊNG
# (results/real_lr_holdout_multiseed_<backbone>/), rồi gọi lại đúng
# data/aggregate_real_lr_holdout_multiseed.py KHÔNG SỬA GÌ cho từng thư mục
# đó -- mỗi backbone tự có 1 bộ kết quả tổng hợp độc lập, không thể lẫn nhau.
#
# DÙNG:
#   bash pipeline/run_real_lr_holdout_multiseed_all_backbones.sh [đường_dẫn_config]
#
# TIỀN ĐỀ:
#   - Đã chạy xong pipeline/run_multi_seed.sh (+ _extra_seeds.sh) cho 5
#     backbone gốc, VÀ pipeline/run_multi_seed_new_backbones.sh cho 7
#     backbone mới -- cần checkpoint recognition_{lr,sr_baseline,sr_improved}
#     _<backbone>_seed<seed>/best.pt, n=5 seed, cho cả 12 backbone.
#   - Checkpoint SR cố định (span_official + span_improved student) đã có sẵn
#     từ pipeline chính, không phụ thuộc backbone.
#
# CẢNH BÁO THỜI GIAN: 12 backbone x 5 seed x 3 điều kiện = 180 lần EVAL (KHÔNG
# train gì cả -- chỉ forward pass qua splits/real_lr_holdout.json, rất nhẹ so
# với các script train+eval khác trong dự án).

set -e

CONFIG="${1:-configs/config.yaml}"
BACKBONES=("efficientnet_b0" "ghostnet_100" "mobilenet_v2" "mobilenet_v3_small" "resnet18" \
           "lcnet_100" "mnasnet1_0" "mobilenet_v3_large" "mobileone_s0" "regnet_y_400mf" \
           "shufflenet_v2_x1_0" "squeezenet1_1")
SEEDS=(42 123 2024 44 999)
NUM_WORKERS="${NUM_WORKERS:-4}"

if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không thấy $CONFIG" >&2
    exit 1
fi

SR_ARCH=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['sr']['arch'])")
STUDENT_ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")
SR_BASELINE_CKPT="runs/sr_${SR_ARCH}/best.pt"
SR_IMPROVED_CKPT="runs/sr_improved_${STUDENT_ARCH}/best.pt"

echo "================================================================"
echo "Config       = $CONFIG"
echo "Backbones    = ${BACKBONES[*]} (${#BACKBONES[@]} backbone)"
echo "Seeds        = ${SEEDS[*]}"
echo "================================================================"

echo "Kiểm tra tiền đề..."
MISSING=0
if [ ! -f "$SR_BASELINE_CKPT" ]; then
    echo "LỖI: thiếu $SR_BASELINE_CKPT" >&2; MISSING=1
fi
if [ ! -f "$SR_IMPROVED_CKPT" ]; then
    echo "LỖI: thiếu $SR_IMPROVED_CKPT" >&2; MISSING=1
fi
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DOMAIN in lr sr_baseline sr_improved; do
            CKPT="runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt"
            if [ ! -f "$CKPT" ]; then
                echo "LỖI: thiếu $CKPT -> chạy pipeline/run_multi_seed.sh (5 backbone gốc) hoặc" >&2
                echo "     pipeline/run_multi_seed_new_backbones.sh (7 backbone mới) trước." >&2
                MISSING=1
            fi
        done
    done
done
if [ "$MISSING" -eq 1 ]; then
    echo "DỪNG LẠI — thiếu tiền đề ở trên." >&2
    exit 1
fi
echo "OK — đủ tiền đề, bắt đầu chạy (chỉ eval, không train lại gì cả)."

for BACKBONE in "${BACKBONES[@]}"; do
    RESULTS_DIR="results/real_lr_holdout_multiseed_${BACKBONE}"
    mkdir -p "$RESULTS_DIR"

    echo ""
    echo "################################################################"
    echo "# backbone=$BACKBONE"
    echo "################################################################"

    for SEED in "${SEEDS[@]}"; do
        OUT_CSV="$RESULTS_DIR/real_lr_holdout_seed${SEED}.csv"
        if [ -f "$OUT_CSV" ]; then
            echo ">>> Bỏ qua backbone=$BACKBONE seed=$SEED (đã có $OUT_CSV)"
            continue
        fi

        echo "----------------------------------------------------------------"
        echo "backbone=$BACKBONE | seed=$SEED"
        echo "----------------------------------------------------------------"
        python eval_real_lr_holdout.py --config "$CONFIG" --backbone "$BACKBONE" \
            --sr_baseline_ckpt "$SR_BASELINE_CKPT" --sr_baseline_arch "$SR_ARCH" \
            --sr_improved_ckpt "$SR_IMPROVED_CKPT" --sr_improved_arch "$STUDENT_ARCH" \
            --run_suffix "_seed${SEED}" --seed_label "$SEED" --num_workers "$NUM_WORKERS" \
            --out_csv "$OUT_CSV"
    done

    echo ">>> Tổng hợp multi-seed cho backbone=$BACKBONE..."
    python data/aggregate_real_lr_holdout_multiseed.py --results_dir "$RESULTS_DIR" \
        --out_prefix "$RESULTS_DIR/real_lr_holdout_multiseed"
done

echo ""
echo "HOÀN TẤT. Kết quả từng backbone nằm riêng trong:"
echo "  results/real_lr_holdout_multiseed_<backbone>/real_lr_holdout_multiseed_{identity,gender}.csv"
echo "  results/real_lr_holdout_multiseed_<backbone>/real_lr_holdout_multiseed_{identity,gender}_pairwise.csv"
echo "Xem cột 'condition' (no_sr/sr_baseline/sr_improved) ở từng backbone để biết đảo ngược"
echo "thứ tự (no-SR > span_tiny > span_baseline) có lặp lại ở backbone nào ngoài MobileNetV2."
