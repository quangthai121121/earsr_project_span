#!/bin/bash
# [MỚI — Trụ cột 5, reviewer round 2] pipeline/run_degradation_robustness_
# ablation.sh chỉ kiểm tra giải pháp degradation-augmented training (Mục 5.9)
# trên MỘT backbone (mobilenet_v2). Reviewer chỉ ra: phát hiện chính (lỗi khi
# gặp ảnh thực <20px) vững, nhưng biện pháp KHẮC PHỤC chỉ có 1 backbone,
# không đạt ý nghĩa thống kê -- không đủ để khái quát hoá là giải pháp chung.
# Đúng comment sẵn có trong pipeline/run_degradation_robustness_ablation.sh
# ("Nếu kết quả khả quan, mở rộng sang 5 backbone") -- script này thực hiện
# đúng phần mở rộng đó.
#
# Dùng LẠI 4 backbone giống hệt pipeline/run_depth_sweep_new_backbones.sh
# (Trụ cột 2) để 2 phần mở rộng của round-2 review dùng chung một bộ "5
# backbone đại diện, trải rộng nhiều họ kiến trúc" -- resnet18, efficientnet_
# b0, ghostnet_100, shufflenet_v2_x1_0 -- cùng mobilenet_v2 đã có.
#
# KHÔNG train lại SR: 2 checkpoint SR robust (span_tiny_robust, span_
# baseline_robust) và 2 domain ảnh (sr_baseline_robust, sr_improved_robust)
# đã build sẵn từ pipeline/run_degradation_robustness_ablation.sh, KHÔNG phụ
# thuộc backbone nhận dạng nào dùng chúng. Script này CHỈ train+eval
# recognition cho 4 backbone MỚI, theo ĐÚNG mẫu ghi kết quả từng backbone vào
# thư mục RIÊNG (results/degradation_robustness_<backbone>/, results/
# degradation_robustness_lrholdout_<backbone>/) đã dùng trong pipeline/
# run_real_lr_holdout_multiseed_all_backbones.sh -- KHÔNG sửa data/aggregate_
# real_lr_holdout_multiseed.py (script đó không có khái niệm backbone trong
# logic gộp theo condition, và ĐÃ được dùng nguyên trạng theo đúng mẫu này
# cho real_lr_holdout gốc; sửa nó có nguy cơ ảnh hưởng ngược lại số liệu ĐÃ
# công bố của real_lr_holdout gốc).
#
# DÙNG:
#   bash pipeline/run_degradation_robustness_ablation_new_backbones.sh [đường_dẫn_config]
#
# TIỀN ĐỀ: đã chạy xong pipeline/run_degradation_robustness_ablation.sh (2
# checkpoint SR robust + 2 domain ảnh robust đã build) VÀ có checkpoint
# recognition_lr_<backbone>_seed<seed> cho cả 4 backbone mới, n=5 seed.
#
# CẢNH BÁO THỜI GIAN: 4 backbone x 2 domain x 5 seed = 40 lần train+eval
# recognition (sanity-check benchmark chính) + 4 backbone x 5 seed = 20 lần
# eval real_lr_holdout (chỉ eval, không train) -- không train lại SR nào.

set -e

CONFIG="${1:-configs/config.yaml}"
NEW_BACKBONES=("resnet18" "efficientnet_b0" "ghostnet_100" "shufflenet_v2_x1_0")
SEEDS=(42 123 2024 44 999)
NUM_WORKERS="${NUM_WORKERS:-4}"

if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không thấy $CONFIG"
    exit 1
fi

SR_ARCH=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['sr']['arch'])")
STUDENT_ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")

SR_BASELINE_ROBUST_CKPT="runs/sr_${SR_ARCH}_robust/best.pt"
SR_IMPROVED_ROBUST_CKPT="runs/sr_improved_${STUDENT_ARCH}_robust/best.pt"

echo "################################################################"
echo "# Kiểm tra tiền đề"
echo "################################################################"
MISSING=0
if [ ! -f "$SR_BASELINE_ROBUST_CKPT" ]; then
    echo "LỖI: thiếu $SR_BASELINE_ROBUST_CKPT -> chạy pipeline/run_degradation_robustness_ablation.sh trước."
    MISSING=1
fi
if [ ! -f "$SR_IMPROVED_ROBUST_CKPT" ]; then
    echo "LỖI: thiếu $SR_IMPROVED_ROBUST_CKPT -> chạy pipeline/run_degradation_robustness_ablation.sh trước."
    MISSING=1
fi
if [ ! -d "splits/sr_baseline_robust" ] || [ ! -d "splits/sr_improved_robust" ]; then
    echo "LỖI: thiếu splits/sr_baseline_robust hoặc splits/sr_improved_robust -> chạy"
    echo "     pipeline/run_degradation_robustness_ablation.sh trước."
    MISSING=1
fi
for BACKBONE in "${NEW_BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
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
echo "OK — mọi tiền đề đã sẵn sàng (dùng lại 2 checkpoint SR robust + 2 domain ảnh đã có)."

for BACKBONE in "${NEW_BACKBONES[@]}"; do
    RESULTS_DIR="results/degradation_robustness_${BACKBONE}"
    mkdir -p "$RESULTS_DIR"

    echo ""
    echo "################################################################"
    echo "# [1/2] Sanity-check benchmark chính (robust training) — backbone=$BACKBONE"
    echo "################################################################"
    for DOMAIN in sr_baseline_robust sr_improved_robust; do
        for SEED in "${SEEDS[@]}"; do
            OUT_JSON="$RESULTS_DIR/${DOMAIN}_${BACKBONE}_seed${SEED}.json"
            if [ -f "$OUT_JSON" ]; then
                echo ">>> Bỏ qua $DOMAIN seed=$SEED (đã có JSON)"
                continue
            fi
            echo "----------------------------------------------------------------"
            echo "domain=$DOMAIN | backbone=$BACKBONE | seed=$SEED"
            echo "----------------------------------------------------------------"
            LR_CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
            python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" --backbone "$BACKBONE" \
                --init_ckpt "$LR_CKPT" --seed "$SEED" --run_suffix "_seed${SEED}" --num_workers "$NUM_WORKERS"
            python eval_recognition.py --config "$CONFIG" \
                --ckpt "runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt" \
                --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
                --out_json "$OUT_JSON" --num_workers "$NUM_WORKERS"
        done
    done

    echo ">>> Tổng hợp sanity-check benchmark chính, backbone=$BACKBONE..."
    python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
        --out_csv "$RESULTS_DIR/degradation_robustness_mainbench_summary.csv"

    LRHOLDOUT_RESULTS_DIR="results/degradation_robustness_lrholdout_${BACKBONE}"
    mkdir -p "$LRHOLDOUT_RESULTS_DIR"

    echo ""
    echo "################################################################"
    echo "# [2/2] real_lr_holdout với checkpoint robust — backbone=$BACKBONE"
    echo "################################################################"
    for SEED in "${SEEDS[@]}"; do
        OUT_CSV="$LRHOLDOUT_RESULTS_DIR/real_lr_holdout_seed${SEED}.csv"
        if [ -f "$OUT_CSV" ]; then
            echo ">>> Bỏ qua seed=$SEED (đã có $OUT_CSV)"
            continue
        fi
        python eval_real_lr_holdout.py --config "$CONFIG" --backbone "$BACKBONE" \
            --sr_baseline_ckpt "$SR_BASELINE_ROBUST_CKPT" --sr_baseline_arch "$SR_ARCH" \
            --sr_improved_ckpt "$SR_IMPROVED_ROBUST_CKPT" --sr_improved_arch "$STUDENT_ARCH" \
            --run_suffix "_seed${SEED}" --seed_label "$SEED" --domain_suffix "_robust" \
            --num_workers "$NUM_WORKERS" \
            --out_csv "$OUT_CSV"
    done

    echo ">>> Tổng hợp real_lr_holdout robust, backbone=$BACKBONE..."
    python data/aggregate_real_lr_holdout_multiseed.py --results_dir "$LRHOLDOUT_RESULTS_DIR" \
        --out_prefix "$LRHOLDOUT_RESULTS_DIR/real_lr_holdout_robust"
done

echo ""
echo "HOÀN TẤT. Kết quả từng backbone nằm riêng trong:"
echo "  results/degradation_robustness_<backbone>/degradation_robustness_mainbench_summary*.csv"
echo "  results/degradation_robustness_lrholdout_<backbone>/real_lr_holdout_robust_{identity,gender}*.csv"
echo "So sánh với backbone gốc (mobilenet_v2):"
echo "  results/degradation_robustness/degradation_robustness_mainbench_summary*.csv"
echo "  results/degradation_robustness_lrholdout/real_lr_holdout_robust_{identity,gender}*.csv"
