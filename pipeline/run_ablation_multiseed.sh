#!/bin/bash
# [MỚI — bổ sung journal Q1, đợt 8] Phiên bản MULTI-SEED của
# pipeline/run_ablation.sh — mục đích DUY NHẤT: trả lời bằng số liệu thật
# (paired t-test + Cohen's d), thay vì cảm tính, câu hỏi "distillation (KD)
# có thật sự giúp ích không?" (so sánh pixel_only vs pixel_distill), vốn
# trước đó chỉ có n=1 (results/ablation.csv gốc), KHÔNG đủ để kết luận theo
# hướng nào — xu hướng n=1 quan sát được (pixel_only > pixel_distill) thậm
# chí ngược với kỳ vọng, nhưng n=1 không có ý nghĩa thống kê nên KHÔNG được
# dùng để kết luận theo hướng nào cả.
#
# THIẾT KẾ (đúng quy ước "SR train 1 lần, chỉ multi-seed bước recognition"
# đã dùng xuyên suốt project — xem pipeline/run_multi_seed.sh,
# RUN_ALL_span_large_ablation.sh, RUN_ALL_extra_sr_baseline*.sh):
#   - KHÔNG train lại SR (train_sr_distill.py) — DÙNG LẠI checkpoint SR đã có
#     từ pipeline/run_ablation.sh (4 checkpoint runs/sr_improved_<arch>_ablation_<name>/best.pt,
#     train ở seed=42 mặc định của config.yaml) và ảnh đã sinh sẵn ở
#     splits/sr_ablation_<name>/. Nghĩa là: biến "chất lượng ảnh SR" CỐ ĐỊNH
#     giữa các seed — CHỈ biến thiên qua khởi tạo/thứ tự batch của bước
#     TRAIN RECOGNITION. Đây là giới hạn cần nêu rõ trong phần Limitations
#     của bài báo (không phải multi-seed "từ đầu đến cuối" bao gồm cả SR).
#   - Multi-seed (5 seed: 42,123,2024,44,999) CHỈ áp dụng cho train_recognition.py,
#     fine-tune từ checkpoint LR CỦA CHÍNH SEED ĐÓ (runs/recognition_lr_<backbone>_seed<seed>/best.pt)
#     — giống hệt nguyên tắc chuỗi hr->lr->sr_* dùng trong run_multi_seed.sh,
#     áp dụng cho cả 4 domain sr_ablation_<name> song song.
#   - 1 backbone duy nhất (mobilenet_v2), giống quy ước của run_ablation.sh gốc
#     (ablation dùng 1 backbone đại diện để tiết kiệm thời gian).
#
# TIỀN ĐỀ (BẮT BUỘC, kiểm tra ở BƯỚC 0):
#   1. Đã chạy xong pipeline/run_ablation.sh (có 4 checkpoint SR ablation +
#      4 thư mục ảnh splits/sr_ablation_<name>/).
#   2. Đã chạy xong pipeline/run_multi_seed.sh (checkpoint recognition_lr_mobilenet_v2_seed{42,123,2024})
#      VÀ pipeline_run_multi_seed_extra_seeds.sh (thêm seed 44,999) — tức
#      ĐỦ 5 seed của checkpoint recognition_lr_mobilenet_v2_seed<seed>.
#
# KẾT QUẢ: results/ablation_multiseed/ablation_multiseed_summary.csv (trung
# bình +- độ lệch chuẩn theo (backbone, domain)) và
# results/ablation_multiseed/ablation_multiseed_summary_pairwise.csv — chứa
# ĐỦ CẢ 6 CẶP so sánh trong C(4,2) (gồm đúng cặp pixel_only vs pixel_distill
# ta cần) với paired t-test (raw + Bonferroni theo 6 cặp) + Cohen's d thật.
#
# CẢNH BÁO THỜI GIAN: 4 cấu hình x 5 seed = 20 lần train recognition (nhẹ hơn
# nhiều so với multi-seed 5-backbone vì chỉ dùng 1 backbone).

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results/ablation_multiseed"
BACKBONE="mobilenet_v2"
SEEDS=(42 123 2024 44 999)
CONFIGS_ORDER=(pixel_only pixel_distill pixel_identity full)
mkdir -p "$RESULTS_DIR"

ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")

echo "################################################################"
echo "# BƯỚC 0 — Kiểm tra tiền đề"
echo "################################################################"
MISSING=0
for NAME in "${CONFIGS_ORDER[@]}"; do
    SR_CKPT="runs/sr_improved_${ARCH}_ablation_${NAME}/best.pt"
    SPLIT_DIR="splits/sr_ablation_${NAME}"
    if [ ! -f "$SR_CKPT" ]; then
        echo "LỖI: chưa thấy $SR_CKPT -> chạy 'bash pipeline/run_ablation.sh' trước."
        MISSING=1
    fi
    if [ ! -d "$SPLIT_DIR" ]; then
        echo "LỖI: chưa thấy $SPLIT_DIR -> chạy 'bash pipeline/run_ablation.sh' trước."
        MISSING=1
    fi
done
for SEED in "${SEEDS[@]}"; do
    LR_CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
    if [ ! -f "$LR_CKPT" ]; then
        echo "LỖI: chưa thấy $LR_CKPT"
        echo "     -> chạy 'bash pipeline/run_multi_seed.sh' (seed 42,123,2024) và/hoặc"
        echo "        'bash pipeline_run_multi_seed_extra_seeds.sh' (seed 44,999) trước."
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "DỪNG LẠI — thiếu tiền đề ở trên, xem chi tiết rồi chạy lại."
    exit 1
fi
echo "OK — mọi tiền đề đã sẵn sàng (dùng lại checkpoint SR ablation có sẵn, không train lại SR)."

for NAME in "${CONFIGS_ORDER[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "################################################################"
        echo "# ablation=$NAME | backbone=$BACKBONE | seed=$SEED"
        echo "################################################################"

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
echo ">>> Tổng hợp — paired t-test + Cohen's d cho cả 6 cặp cấu hình (đặc biệt"
echo ">>> chú ý cặp pixel_only vs pixel_distill = câu hỏi 'KD có tác dụng không')..."
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/ablation_multiseed_summary.csv"

echo ""
echo "HOÀN TẤT. Đọc $RESULTS_DIR/ablation_multiseed_summary_pairwise.csv, tìm dòng"
echo "domain_a=pixel_distill (hoặc pixel_only) domain_b=pixel_only (hoặc pixel_distill)"
echo "để có câu trả lời THẬT (không phải cảm tính) cho câu hỏi KD có tác dụng hay không."
echo "LƯU Ý khi viết bài: đây là multi-seed CHỈ ở bước recognition (SR cố định seed=42)"
echo "— nêu rõ giới hạn này trong phần Limitations."
