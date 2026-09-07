#!/bin/bash
# [MỚI — thử nghiệm frequency-weighted loss, hướng novelty mới đề xuất sau
# nhận xét của thầy hướng dẫn: "đóng góp kỹ thuật hiện giờ chỉ là giảm
# n_blocks 6->3, cần thuyết phục hơn"]
#
# Ý TƯỞNG: thí nghiệm frequency-ablation đã có trong bài
# (Section~sec:freq-ablation, eval_frequency_ablation.py) chứng minh BẰNG
# THỰC NGHIỆM rằng tín hiệu nhận dạng trong ảnh tai độ phân giải thấp tập
# trung ở dải tần số THẤP (giữ 10% tần số thấp nhất vẫn giữ được 22-34%
# accuracy; bỏ 10% đó thì accuracy sập gần về ngẫu nhiên). Thí nghiệm đó chỉ
# DÙNG phát hiện này để GIẢI THÍCH tại sao nén sâu an toàn (eval trên
# checkpoint đã train sẵn, không train lại). Script này lần đầu tiên dùng
# phát hiện đó để THIẾT KẾ một loss training mới (lambda_freq trong
# train_sr_distill.py::compute_total_loss, xem
# utils/frequency_filters_torch.py) -- nếu có tín hiệu, đây là novelty kỹ
# thuật thật sự (không chỉ là ablation xác nhận), biến "chẩn đoán" thành
# "phương pháp đề xuất".
#
# ĐÂY LÀ TÍN HIỆU SÀNG LỌC NHANH (1 backbone, n=3 seed) — KHÔNG phải bằng
# chứng cuối cùng, giống đúng nguyên tắc "sàng lọc trước, mở rộng sau" đã
# dùng cho depth-sweep/KD v2/mọi ablation khác trong project (xem
# pipeline/run_ablation_kd_v2.sh). Nếu có cấu hình lambda_freq thắng rõ ràng
# ở đây, PHẢI validate lại bằng multi-seed (n=5) x cả 11 backbone trước khi
# đưa vào bài báo.
#
# YÊU CẦU:
#   - Đã chạy pipeline/00-02 (checkpoint EDSR teacher, SPAN baseline/span_official).
#   - Đã có checkpoint recognition_lr_mobilenet_v2_seed{42,123,2024}/best.pt
#     (từ pipeline/run_multi_seed.sh) — KHÔNG cần seed 44/999 (n=3 là đủ cho
#     bước sàng lọc này, giống thiết kế run_ablation_multiseed.sh ban đầu).
#   - KHÔNG cần checkpoint HR judge (lambda_identity=0 xuyên suốt sweep này —
#     cô lập đúng 1 biến, không trộn với novelty-ablation khác).

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results/freq_weighted_loss_screening"
BACKBONE="mobilenet_v2"
SEEDS=(42 123 2024)
FREQ_CUTOFF=0.1
mkdir -p "$RESULTS_DIR"

ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

echo "################################################################"
echo "# BƯỚC 0 — Kiểm tra tiền đề"
echo "################################################################"
MISSING=0
for SEED in "${SEEDS[@]}"; do
    LR_CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
    if [ ! -f "$LR_CKPT" ]; then
        echo "LỖI: chưa thấy $LR_CKPT -> chạy 'bash pipeline/run_multi_seed.sh' trước."
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "DỪNG LẠI — thiếu tiền đề ở trên, xem chi tiết rồi chạy lại."
    exit 1
fi
echo "OK — mọi tiền đề đã sẵn sàng."

# format mỗi dòng: lambda_freq (freq_cutoff cố định = $FREQ_CUTOFF cho cả sweep,
# khớp đúng cutoff dùng trong thí nghiệm frequency-ablation đã có trong bài)
declare -A CONFIGS=(
    [freqloss_baseline]="0.0"   # recipe cũ (đối chứng) — giống pipeline/06_improve_span.sh mặc định
    [freqloss_low]="0.25"
    [freqloss_mid]="0.5"
    [freqloss_high]="1.0"
)
CONFIGS_ORDER=(freqloss_baseline freqloss_low freqloss_mid freqloss_high)

echo ""
echo "################################################################"
echo "# BƯỚC 1 — Train SR (1 lần/cấu hình, seed mặc định config.yaml,"
echo "# giống đúng quy ước 'SR train 1 lần, chỉ multi-seed bước recognition'"
echo "# đã dùng xuyên suốt project — xem pipeline/run_ablation_multiseed.sh)"
echo "################################################################"
for NAME in "${CONFIGS_ORDER[@]}"; do
    LF="${CONFIGS[$NAME]}"
    echo "---- $NAME (lambda_freq=$LF, freq_cutoff=$FREQ_CUTOFF) ----"

    # [PIN TƯỜNG MINH] lambda_feat/lambda_saliency/lambda_identity/lambda_position=0
    # — sweep này CHỈ cô lập lambda_freq, không để cơ chế MỚI khác âm thầm bật
    # lên qua default config.yaml (giống đúng cách run_ablation_kd_v2.sh làm).
    python train_sr_distill.py --config "$CONFIG" \
        --lambda_pixel 1.0 --lambda_distill 1.0 \
        --lambda_feat 0 --lambda_saliency 0 --lambda_identity 0 --lambda_position 0 \
        --lambda_freq "$LF" --freq_cutoff "$FREQ_CUTOFF" \
        --run_suffix "_ablation_${NAME}"

    python data/build_sr.py --lr_dir splits/lr \
        --sr_ckpt "runs/sr_improved_${ARCH}_ablation_${NAME}/best.pt" \
        --arch "$ARCH" --scale "$SCALE" --out_dir "splits/sr_ablation_${NAME}"
done

echo ""
echo "################################################################"
echo "# BƯỚC 2 — Train recognition multi-seed (n=3), backbone=$BACKBONE"
echo "################################################################"
for NAME in "${CONFIGS_ORDER[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "---- freq_config=$NAME | backbone=$BACKBONE | seed=$SEED ----"

        python train_recognition.py --config "$CONFIG" --domain "sr_ablation_${NAME}" \
            --backbone "$BACKBONE" \
            --init_ckpt "runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt" \
            --seed "$SEED" --run_suffix "_seed${SEED}"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "runs/recognition_sr_ablation_${NAME}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "sr_ablation_${NAME}" \
            --test_domain "sr_ablation_${NAME}" \
            --out_json "$RESULTS_DIR/ablation_${NAME}_${BACKBONE}_seed${SEED}.json"
    done
done

echo ""
echo ">>> Tổng hợp — paired t-test + Cohen's d cho mọi cặp cấu hình (đặc biệt"
echo ">>> chú ý freqloss_baseline vs freqloss_{low,mid,high} = câu hỏi chính)..."
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/freq_weighted_loss_screening_summary.csv"

echo ""
echo "HOÀN TẤT. Đọc $RESULTS_DIR/freq_weighted_loss_screening_summary_pairwise.csv,"
echo "tìm các dòng domain_a=sr_ablation_freqloss_baseline so với 3 cấu hình còn lại."
echo "LƯU Ý: đây CHỈ là sàng lọc (1 backbone, n=3 seed) — nếu có cấu hình thắng rõ,"
echo "PHẢI validate lại bằng n=5 x 11 backbone trước khi đưa vào bài báo."
