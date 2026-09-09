#!/bin/bash
# [MỚI — thử nghiệm kiến trúc mới, hướng novelty đề xuất sau khi 4 thí nghiệm
# loss-engineering (KD v2, saliency, learned pruning, frequency-weighted loss)
# đều cho kết quả null/không cải thiện]
#
# Ý TƯỞNG: khác 4 lần thử trước (đều can thiệp vào LOSS), lần này can thiệp
# vào KIẾN TRÚC -- dựa trực tiếp trên phát hiện frequency-ablation đã có
# trong bài (tín hiệu nhận dạng gần như không phụ thuộc chi tiết tần số
# cao). models/sr_models.py::SPANDetailBottleneck thêm 1 bottleneck 1x1 hẹp
# ngay TRƯỚC lớp upsample cuối cùng (nơi tổng hợp chi tiết tần số cao), giữ
# NGUYÊN VẸN phần thân mạng (head + 3 khối SPAB + body_tail, residual dài
# không đổi) -- nếu giả thuyết đúng, thắt cổ chai đó không hại downstream
# accuracy trong khi cắt thêm tham số so với span_tiny.
#
# LƯU Ý QUAN TRỌNG (đã kiểm chứng bằng số trước khi chạy, xem
# models/test_span_detail_bottleneck.py): mức giảm tham số ròng khá khiêm
# tốn (~4-7% tuỳ bottleneck_channels, vì phần lớn tham số của span_tiny nằm
# ở 3 khối SPAB chứ không phải lớp chiếu cuối) -- mục tiêu chính của thí
# nghiệm này là KIỂM ĐỊNH GIẢ THUYẾT KIẾN TRÚC (bottleneck ở đúng chỗ tổng
# hợp tần số cao không hại accuracy), không phải để đạt mức nén ấn tượng.
#
# ĐÂY LÀ TÍN HIỆU SÀNG LỌC NHANH (1 backbone, n=3 seed) — giống đúng nguyên
# tắc đã dùng cho mọi ablation khác trong project (xem
# pipeline/run_ablation_kd_v2.sh, pipeline/run_freq_weighted_loss_screening.sh).
# Nếu có tín hiệu (bottleneck KHÔNG làm giảm accuracy có ý nghĩa so với
# span_tiny), PHẢI validate lại bằng multi-seed (n=5) x cả 11 backbone trước
# khi đưa vào bài báo.
#
# YÊU CẦU: giống hệt pipeline/run_freq_weighted_loss_screening.sh — đã chạy
# pipeline/00-02 (checkpoint EDSR teacher, SPAN baseline/span_official), và
# có checkpoint recognition_lr_mobilenet_v2_seed{42,123,2024}/best.pt.

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results/detail_bottleneck_screening"
BACKBONE="mobilenet_v2"
SEEDS=(42 123 2024)
mkdir -p "$RESULTS_DIR"

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

# format mỗi dòng: student_arch bottleneck_channels ("-" nghĩa là không truyền
# --student_bottleneck_channels, dùng cho baseline span_tiny không có bottleneck)
# [SỬA — tương thích bash 3.2 (mặc định macOS, dùng để review/test cục bộ) —
# associative array (declare -A) CHỈ hỗ trợ từ bash 4.0, không dùng được ở
# đây dù server chạy training thường có bash 4+. Dùng 3 mảng thường song
# song theo INDEX (tương thích bash 3.2 trở lên) thay vì associative array,
# để logic loop này review/test được trên MỌI bash, không riêng gì server.]
CONFIGS_ORDER=(bottleneck_baseline bottleneck_8 bottleneck_16 bottleneck_24)
ARCH_LIST=(span_tiny span_bottleneck span_bottleneck span_bottleneck)
BC_LIST=("-" 8 16 24)

echo ""
echo "################################################################"
echo "# BƯỚC 1 — Train SR (1 lần/cấu hình, seed mặc định config.yaml, giống"
echo "# đúng quy ước 'SR train 1 lần, chỉ multi-seed bước recognition')"
echo "################################################################"
for i in "${!CONFIGS_ORDER[@]}"; do
    NAME="${CONFIGS_ORDER[$i]}"
    ARCH="${ARCH_LIST[$i]}"
    BC="${BC_LIST[$i]}"
    echo "---- $NAME (student_arch=$ARCH, bottleneck_channels=$BC) ----"

    if [ "$BC" == "-" ]; then
        BC_FLAG=()
    else
        BC_FLAG=(--student_bottleneck_channels "$BC")
    fi

    # [PIN TƯỜNG MINH] mọi lambda mới (feat/saliency/identity/position/freq)=0
    # — chỉ cô lập đúng 1 biến (kiến trúc student), không trộn với các thí
    # nghiệm loss khác đã kết luận null.
    python train_sr_distill.py --config "$CONFIG" \
        --student_arch "$ARCH" "${BC_FLAG[@]}" \
        --lambda_pixel 1.0 --lambda_distill 1.0 \
        --lambda_feat 0 --lambda_saliency 0 --lambda_identity 0 --lambda_position 0 --lambda_freq 0 \
        --run_suffix "_ablation_${NAME}"

    BUILD_BC_FLAG=()
    if [ "$BC" != "-" ]; then
        BUILD_BC_FLAG=(--bottleneck_channels "$BC")
    fi
    python data/build_sr.py --lr_dir splits/lr \
        --sr_ckpt "runs/sr_improved_${ARCH}_ablation_${NAME}/best.pt" \
        --arch "$ARCH" --scale "$SCALE" "${BUILD_BC_FLAG[@]}" \
        --out_dir "splits/sr_ablation_${NAME}"
done

echo ""
echo "################################################################"
echo "# BƯỚC 2 — Train recognition multi-seed (n=3), backbone=$BACKBONE"
echo "################################################################"
for NAME in "${CONFIGS_ORDER[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "---- config=$NAME | backbone=$BACKBONE | seed=$SEED ----"

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
echo ">>> chú ý bottleneck_baseline vs bottleneck_{8,16,24} = câu hỏi chính)..."
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/detail_bottleneck_screening_summary.csv"

echo ""
echo "HOÀN TẤT. Đọc $RESULTS_DIR/detail_bottleneck_screening_summary_pairwise.csv,"
echo "tìm các dòng domain_a=sr_ablation_bottleneck_baseline so với 3 cấu hình còn lại."
echo "LƯU Ý: đây CHỈ là sàng lọc (1 backbone, n=3 seed) — nếu có cấu hình KHÔNG"
echo "thua kém baseline một cách rõ ràng, PHẢI validate lại bằng n=5 x 11 backbone"
echo "trước khi đưa vào bài báo."
