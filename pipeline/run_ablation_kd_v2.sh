#!/bin/bash
# [MỚI] Ablation 2x2 cho 2 cơ chế mới trong train_sr_distill.py:
#   - feature-level KD (lambda_feat, SRFeatureHook — models/sr_models.py)
#   - multi-judge ensemble identity loss (identity_judges — configs/config.yaml)
# Mục tiêu: cô lập tác dụng của TỪNG cơ chế + tổ hợp cả hai, trả lời trực
# tiếp câu hỏi "feature-KD/multi-judge có thực sự cải thiện downstream
# accuracy so với recipe cũ (chỉ pixel+distill output-level) không?"
#
# Dùng 1 backbone duy nhất (mobilenet_v2) để tiết kiệm thời gian — ĐÂY LÀ TÍN
# HIỆU SÀNG LỌC NHANH, không phải bằng chứng cuối cùng. Sau khi xác định được
# cấu hình thắng ở đây, PHẢI validate lại bằng multi-seed x 5-backbone (theo
# đúng mẫu pipeline/run_multi_seed.sh) trước khi đưa vào bảng kết quả chính
# thức của bài báo — 1 bài học đã rút ra từ chính project này (Table 3 trong
# bản thảo hiện tại cũng chỉ mới chạy 1 backbone, đã bị nêu là giới hạn).
#
# YÊU CẦU:
#   - Đã chạy xong pipeline 01-05 (checkpoint recognition_lr_mobilenet_v2,
#     EDSR teacher, SPAN baseline/span_official).
#   - Đã có ĐỦ 3 checkpoint recognition domain "hr" dùng làm multi-judge
#     (mobilenet_v2, resnet18, ghostnet_100) — xem
#     configs/config.yaml::sr_improve.identity_judges. Nếu thiếu, chạy lại
#     pipeline/03_train_baseline_recognition.sh cho backbone còn thiếu trước.

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results"
BACKBONE="mobilenet_v2"
mkdir -p "$RESULTS_DIR"

ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

# [SỬA — bổ sung sau code review, vòng 5, điểm 2] Xem giải thích đầy đủ trong
# pipeline/run_prune_sparsity_screen.sh — script này CŨNG dùng checkpoint
# KHÔNG suffix seed, KHÁC với checkpoint multi-seed mà check_prerequisites.sh
# kiểm tra. Fail sớm thay vì để mất hàng giờ train SR (4 cấu hình) rồi mới lỗi.
LR_CKPT_NOSUFFIX="runs/recognition_lr_${BACKBONE}/best.pt"
if [ ! -f "$LR_CKPT_NOSUFFIX" ]; then
    echo "LỖI: thiếu $LR_CKPT_NOSUFFIX (checkpoint KHÔNG suffix seed)."
    echo "     -> chạy 'python train_recognition.py --config $CONFIG --domain lr"
    echo "        --backbone $BACKBONE' (xem pipeline/03_train_baseline_recognition.sh)"
    echo "        trước — KHÁC với checkpoint '_seed<N>' của run_multi_seed.sh."
    exit 1
fi

# kdv2_multijudge / kdv2_full cần 3 judge HR — kiểm tra NGAY, không để
# kdv2_baseline+kdv2_feat chạy xong (hàng giờ) rồi mới chết.
source "$(dirname "$0")/_check_hr_judges.sh"
check_hr_judges

# format mỗi dòng: lambda_pixel lambda_distill lambda_feat lambda_identity
declare -A CONFIGS=(
    [kdv2_baseline]="1.0 1.0 0.0 0.0"        # recipe cũ (đối chứng) — giống pipeline/06_improve_span.sh mặc định
    [kdv2_feat]="1.0 1.0 0.5 0.0"            # CHỈ bật feature-level KD, cô lập tác dụng riêng
    [kdv2_multijudge]="1.0 1.0 0.0 0.1"      # CHỈ bật multi-judge identity loss, cô lập tác dụng riêng
    [kdv2_full]="1.0 1.0 0.5 0.1"            # cả 2 cơ chế cùng lúc — cấu hình đề xuất mới
)

for NAME in kdv2_baseline kdv2_feat kdv2_multijudge kdv2_full; do
    read -r LP LD LF LI <<< "${CONFIGS[$NAME]}"
    echo "################################################################"
    echo "# Ablation KD v2: $NAME (pixel=$LP distill=$LD feat=$LF identity=$LI)"
    echo "################################################################"

    # [SỬA — confound phát hiện qua code review] PIN TƯỜNG MINH lambda_saliency=0
    # — ablation 2x2 này CHỈ cô lập feat x multijudge, KHÔNG được để saliency
    # (thêm sau, có sweep riêng ở run_lambda_saliency_sweep.sh) âm thầm bật lên
    # qua default config.yaml.
    python train_sr_distill.py --config "$CONFIG" \
        --lambda_pixel "$LP" --lambda_distill "$LD" \
        --lambda_feat "$LF" --lambda_saliency 0 --lambda_identity "$LI" \
        --run_suffix "_ablation_${NAME}"

    python data/build_sr.py --lr_dir splits/lr \
        --sr_ckpt "runs/sr_improved_${ARCH}_ablation_${NAME}/best.pt" \
        --arch "$ARCH" --scale "$SCALE" --out_dir "splits/sr_ablation_${NAME}"

    python train_recognition.py --config "$CONFIG" --domain "sr_ablation_${NAME}" \
        --backbone "$BACKBONE" --init_ckpt "runs/recognition_lr_${BACKBONE}/best.pt"

    # LƯU Ý [phát hiện qua code review]: file "ablation_${NAME}.json" ở đây
    # (NAME đã có tiền tố "kdv2_", ví dụ ablation_kdv2_baseline.json) VẪN khớp
    # glob "ablation_*.json" của aggregator CŨ (data/aggregate_ablation_results.py)
    # — đã sửa TRỰC TIẾP glob của aggregator cũ để loại trừ "kdv2" thay vì đổi
    # tên ở đây (đổi tên sẽ phá tính tương thích với
    # data/aggregate_ablation_kd_v2_results.py đang dùng đúng tên này).
    python eval_recognition.py --config "$CONFIG" \
        --ckpt "runs/recognition_sr_ablation_${NAME}_${BACKBONE}/best.pt" \
        --backbone "$BACKBONE" --train_domain "sr_ablation_${NAME}" \
        --test_domain "sr_ablation_${NAME}" \
        --out_json "$RESULTS_DIR/ablation_${NAME}.json"
done

echo ""
echo ">>> Tổng hợp kết quả ablation KD v2..."
python data/aggregate_ablation_kd_v2_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/ablation_kd_v2.csv"

echo ""
echo "HOÀN TẤT ablation KD v2. Kết quả: $RESULTS_DIR/ablation_kd_v2.csv"
echo "Bước tiếp theo (BẮT BUỘC trước khi đưa vào bài báo): xác định cấu hình"
echo "thắng ở trên, rồi validate lại bằng multi-seed (n>=5) x cả 5 backbone,"
echo "theo đúng mẫu pipeline/run_multi_seed.sh — 1 backbone/1 seed ở bước này"
echo "chỉ là tín hiệu sàng lọc nhanh, KHÔNG đủ để kết luận thống kê."
