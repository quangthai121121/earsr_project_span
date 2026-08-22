#!/bin/bash
# [MỚI — phát hiện qua review Q1] Đo phương sai do CHÍNH SEED TRAIN SR
# (span_tiny), tách biệt khỏi phương sai downstream (recognition seed) đã đo
# ở pipeline/run_multi_seed.sh.
#
# BỐI CẢNH: toàn bộ pipeline chính chỉ train SR (span_tiny/span_official/mọi
# baseline) ĐÚNG 1 LẦN (seed=42 mặc định config.yaml) rồi tái sử dụng ảnh SR
# sinh ra cho MỌI seed downstream (recognition). Nghĩa là mọi kết luận thống
# kê hiện có chỉ điều kiện trên 1 checkpoint SR cụ thể — nếu checkpoint đó
# "may/xui" ở seed=42, hiệu ứng quan sát được có thể phản ánh riêng lần chạy
# đó chứ không phải đặc tính kiến trúc. Đây là quy ước có chủ đích (train SR
# nhiều seed tốn thêm 3-5x compute cho MỌI cấu hình SR trong bài báo — xem
# train_sr.py/train_sr_distill.py::--seed), không phải bug — script này bổ
# sung NĂNG LỰC ĐO trực tiếp phương sai đó cho 1 kiến trúc đại diện
# (span_tiny — đóng góp chính của luận án), không train lại toàn bộ ma trận
# kiến trúc x seed (quá tốn compute, không cần thiết để trả lời câu hỏi
# reviewer).
#
# THIẾT KẾ: cô lập ĐÚNG 1 biến — chỉ đổi seed lúc TRAIN SR, GIỮ NGUYÊN seed
# downstream (recognition) cố định ở RECOGNITION_SEED cho cả 3 lần — nếu để
# seed downstream cũng đổi theo, không tách được phương sai do SR-seed ra
# khỏi phương sai downstream đã biết.
#
# QUAN TRỌNG — recipe PIN CỨNG (khớp CHÍNH XÁC pipeline/06_improve_span.sh):
# lambda_pixel=1.0, lambda_distill=1.0, lambda_feat=0, lambda_saliency=0,
# lambda_identity=0. Nếu SAU NÀY bảng kết quả chính đổi sang recipe khác (ví
# dụ chọn lambda_identity>0 từ run_lambda_sweep.sh), PHẢI SỬA LẠI giá trị pin
# ở đây khớp với 06_improve_span.sh trước khi chạy — nếu không, script này đo
# phương sai của 2 RECIPE KHÁC NHAU thay vì phương sai của CÙNG 1 recipe qua
# các seed, làm sai lệch hoàn toàn kết luận.
#
# TIỀN ĐỀ: đã chạy xong pipeline chính (Bước 1-7, xem RUNBOOK_EarVN1.0.md)
# cho $CONFIG — cần sẵn có:
#   - <runs_root>/sr_improved_span_tiny/best.pt   (SR seed=42, TÁI SỬ DỤNG)
#   - <splits_root>/sr_improved/                   (ảnh SR seed=42, TÁI SỬ DỤNG)
#   - <runs_root>/recognition_lr_<backbone>[_seed42]/best.pt (init cho fine-tune)
#
# DÙNG:
#   bash pipeline/run_sr_seed_variance.sh [đường_dẫn_config]
#   (không truyền -> mặc định configs/config.yaml, khớp EarVN1.0; truyền
#    configs/config_awex.yaml để chạy cho AWEx — script tham số hoá qua
#    config, dùng CHUNG cho cả 2 dataset như các RUN_ALL_*.sh khác)
#
# CẢNH BÁO THỜI GIAN: 2 lần train SR mới (seed 123, 2024; seed 42 tái sử
# dụng) + 3 lần fine-tune recognition (1 backbone đại diện) — tốn thêm đáng
# kể so với multi-seed downstream thuần (vốn không train lại SR).

set -e

CONFIG="${1:-configs/config.yaml}"
STUDENT_ARCH="span_tiny"
BACKBONE="mobilenet_v2"        # 1 backbone đại diện, giống quy ước screening
                                 # khác trong project (run_ablation.sh,
                                 # run_prune_sparsity_screen.sh) — mở rộng
                                 # sang cả 5 backbone (xem BACKBONES trong
                                 # pipeline/run_multi_seed.sh) nếu muốn bằng
                                 # chứng mạnh hơn cho bài báo, tốn thêm ~5x.
RECOGNITION_SEED=42             # CỐ ĐỊNH — cô lập đúng biến SR-seed
SR_SEEDS=(42 123 2024)

if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không thấy $CONFIG"
    exit 1
fi

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
SPLITS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['splits_root'])")
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

RESULTS_DIR="${RESULTS_ROOT}/sr_seed_variance"
mkdir -p "$RESULTS_DIR"

BASE_SR_CKPT="${RUNS_ROOT}/sr_improved_${STUDENT_ARCH}/best.pt"
if [ ! -f "$BASE_SR_CKPT" ]; then
    echo "LỖI: không thấy $BASE_SR_CKPT — chạy xong pipeline chính (Bước 6, "
    echo "     06_improve_span.sh) cho $CONFIG trước khi chạy script này."
    exit 1
fi

echo "================================================================"
echo "Config          = $CONFIG"
echo "Student arch    = $STUDENT_ARCH"
echo "Backbone        = $BACKBONE (đại diện)"
echo "SR seeds        = ${SR_SEEDS[*]}"
echo "Downstream seed = $RECOGNITION_SEED (CỐ ĐỊNH cho cả 3 lần)"
echo "================================================================"

for SEED in "${SR_SEEDS[@]}"; do
    RUN_SUFFIX_SR="_srseed${SEED}"
    DOMAIN_NAME="sr_improved_srseed${SEED}"
    SR_DOMAIN_DIR="${SPLITS_ROOT}/${DOMAIN_NAME}"

    if [ "$SEED" -eq 42 ]; then
        SR_CKPT="$BASE_SR_CKPT"
        DOMAIN_NAME="sr_improved"
        SR_DOMAIN_DIR="${SPLITS_ROOT}/sr_improved"
        echo ""
        echo "################################################################"
        echo "# SR seed=42: TÁI SỬ DỤNG checkpoint/ảnh sẵn có (không train lại) #"
        echo "################################################################"
    else
        SR_CKPT="${RUNS_ROOT}/sr_improved_${STUDENT_ARCH}${RUN_SUFFIX_SR}/best.pt"

        echo ""
        echo "################################################################"
        echo "# [1/2] Train SR '$STUDENT_ARCH' — seed=$SEED (checkpoint MỚI)   #"
        echo "################################################################"
        # [SỬA — bug phát hiện qua review Q1] TRƯỚC ĐÂY không pin lambda, chỉ
        # dựa vào default hiện tại của config.yaml — giống ĐÚNG anti-pattern
        # mà pipeline/06_improve_span.sh đã tự cảnh báo ("default config từng
        # bị đổi ngầm 1 lần, phá vỡ khả năng tái lập"). PIN TƯỜNG MINH khớp
        # CHÍNH XÁC recipe của 06_improve_span.sh (pixel + output-distill
        # thuần, KHÔNG feature-KD/saliency/identity) — nếu không pin, sau này
        # đổi config.yaml (ví dụ sau khi chọn lambda từ lambda sweep) thì
        # script này sẽ âm thầm train SR khác recipe với bảng chính, khiến so
        # sánh phương sai SR-seed không còn ý nghĩa (đo phương sai của 2 recipe
        # khác nhau, không phải phương sai CÙNG 1 recipe qua các seed).
        python train_sr_distill.py --config "$CONFIG" --student_arch "$STUDENT_ARCH" \
            --lambda_pixel 1.0 --lambda_distill 1.0 --lambda_feat 0 --lambda_saliency 0 --lambda_identity 0 \
            --seed "$SEED" --run_suffix "$RUN_SUFFIX_SR"

        echo ""
        echo "################################################################"
        echo "# [2/2] Sinh ảnh SR từ checkpoint seed=$SEED                     #"
        echo "################################################################"
        python data/build_sr.py --lr_dir "${SPLITS_ROOT}/lr" --sr_ckpt "$SR_CKPT" \
            --arch "$STUDENT_ARCH" --scale "$SCALE" --out_dir "$SR_DOMAIN_DIR"
    fi

    echo ""
    echo "### Đo chất lượng SR (PSNR/SSIM/LPIPS) — checkpoint seed=$SEED ###"
    python eval_sr_quality.py --config "$CONFIG" --arch "$STUDENT_ARCH" --ckpt "$SR_CKPT" \
        --label "${STUDENT_ARCH}_srseed${SEED}" --out_csv "${RESULTS_DIR}/sr_quality_srseed.csv"

    echo ""
    echo "### Fine-tune recognition (seed downstream CỐ ĐỊNH=$RECOGNITION_SEED) trên checkpoint SR seed=$SEED ###"
    INIT_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${RECOGNITION_SEED}/best.pt"
    if [ ! -f "$INIT_CKPT" ]; then
        INIT_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}/best.pt"
    fi
    if [ ! -f "$INIT_CKPT" ]; then
        echo "LỖI: không thấy checkpoint recognition domain 'lr' cho backbone $BACKBONE "
        echo "     (${RUNS_ROOT}/recognition_lr_${BACKBONE}[_seed${RECOGNITION_SEED}]/best.pt)."
        echo "     Chạy xong Bước 3 (03_train_baseline_recognition.sh) trước."
        exit 1
    fi

    RUN_SUFFIX_REC="_srseed${SEED}"
    python train_recognition.py --config "$CONFIG" --domain "$DOMAIN_NAME" --backbone "$BACKBONE" \
        --init_ckpt "$INIT_CKPT" \
        --seed "$RECOGNITION_SEED" --run_suffix "$RUN_SUFFIX_REC"

    python eval_recognition.py --config "$CONFIG" \
        --ckpt "${RUNS_ROOT}/recognition_${DOMAIN_NAME}_${BACKBONE}${RUN_SUFFIX_REC}/best.pt" \
        --backbone "$BACKBONE" --train_domain "$DOMAIN_NAME" --test_domain "$DOMAIN_NAME" \
        --out_json "${RESULTS_DIR}/${BACKBONE}_srseed${SEED}.json"
done

echo ""
echo ">>> Tổng hợp phương sai do SR-seed (tách biệt khỏi phương sai downstream)..."
python data/aggregate_sr_seed_variance.py --results_dir "$RESULTS_DIR" \
    --out_csv "${RESULTS_DIR}/sr_seed_variance_summary.csv" \
    --downstream_multiseed_csv "${RESULTS_ROOT}/multi_seed/multi_seed_summary.csv" \
    --sr_quality_csv "${RESULTS_DIR}/sr_quality_srseed.csv"

echo ""
echo "HOÀN TẤT. Kết quả: ${RESULTS_DIR}/sr_seed_variance_summary.csv"
echo "Dùng số liệu này để viết rõ hơn phần Limitations — so sánh trực tiếp"
echo "phương sai do SR-seed (mới đo) với phương sai downstream đã báo cáo."
