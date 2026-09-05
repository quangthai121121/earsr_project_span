#!/bin/bash
# [MỚI — mở rộng PSNR/SSIM/LPIPS từ n=1 lên n=5 cho EDSR + 4 kiến trúc
# contextual dùng recipe Track B (rlfn_adapted/ecbsr/safmn/smfanet), theo
# yêu cầu review (bất đối xứng n=1-vs-n=5 giữa Table 10 và Table 11).
#
# KHÁC VỚI pipeline/run_sr_seed_variance.sh: script đó ĐO CẢ phương sai
# downstream-recognition do SR-seed gây ra (cần build_sr.py sinh domain ảnh
# + fine-tune recognition nhiều backbone), vì mục tiêu của nó là tách 2
# nguồn phương sai (Section~sec:sr-seed-variance). Ở ĐÂY mục tiêu hẹp hơn
# nhiều: CHỈ cần n=5 điểm PSNR/SSIM/LPIPS để báo cáo mean±std trong Table 10
# — KHÔNG cần build domain ảnh, KHÔNG cần fine-tune recognition gì cả.
# eval_sr_quality.py tự đánh giá TRỰC TIẾP từ checkpoint lên tập test
# (splits_root/hr, splits_root/lr), đã xác nhận qua đọc trực tiếp source
# (eval_sr_quality.py không đọc bất kỳ thư mục sr_<arch> nào) — nên
# data/build_sr.py không phải tiền đề của script này.
#
# 5 kiến trúc hỗ trợ:
#   - edsr: train_sr.py --sr_arch edsr (pixel-loss thuần, KHÔNG
#     --pretrained_path — khớp CHÍNH XÁC pipeline/04_train_teacher_and_span_baseline.sh)
#   - rlfn_adapted|ecbsr|safmn|smfanet: train_sr_distill.py --student_arch
#     <arch>, CÙNG recipe distillation với span_tiny (teacher/lambda đọc từ
#     config, KHÔNG pin cứng) — khớp CHÍNH XÁC
#     RUN_ALL_extra_sr_baseline_distilled.sh (script đã tạo ra checkpoint
#     seed=42/gốc hiện có trong Table 10). KHÔNG pin lambda ở đây (khác
#     run_sr_seed_variance.sh's case span_tiny) vì chính script gốc tạo ra
#     checkpoint seed=42 của các kiến trúc này CŨNG không pin — pin ở đây sẽ
#     có nguy cơ làm seed=42 (recipe cũ, có thể đã trôi so với config hiện
#     tại) và seed mới (recipe pin cứng) LỆCH NHAU, phá vỡ giả định "chỉ đổi
#     seed, giữ nguyên mọi thứ khác" của toàn bộ phân tích seed-variance.
#
# CHỦ ĐỘNG KHÔNG làm (theo quyết định phạm vi sau khi cân nhắc chi phí/lợi
# ích với người dùng): 5 dòng Track A (rlfn/rlfn_adapted/ecbsr/safmn/smfanet
# qua train_sr.py) và 2 dòng half-depth (SAFMN/SMFANet half-depth) — không
# phục vụ bất kỳ so sánh thống kê nào của bài báo (Track A chỉ đối chiếu với
# chính bài báo gốc; half-depth chỉ mô tả, ablation liên quan đã có kiểm
# định thống kê riêng ở MỨC ACCURACY, không phải PSNR).
#
# DÙNG:
#   bash pipeline/run_context_quality_seed_variance.sh [đường_dẫn_config] <edsr|rlfn_adapted|ecbsr|safmn|smfanet>
#   (không truyền config -> mặc định configs/config.yaml, khớp EarVN1.0;
#    contextual comparison hiện chỉ tồn tại trên EarVN1.0, xem Section~sec:limitations,
#    nên KHÔNG có ví dụ AWEx ở đây)
#
# TIỀN ĐỀ: checkpoint seed gốc (=42, xem giải thích trên) đã tồn tại:
#   edsr:          <runs_root>/sr_edsr/best.pt
#   rlfn_adapted:  <runs_root>/sr_improved_rlfn_adapted/best.pt
#   ecbsr:         <runs_root>/sr_improved_ecbsr/best.pt
#   safmn:         <runs_root>/sr_improved_safmn/best.pt
#   smfanet:       <runs_root>/sr_improved_smfanet/best.pt
# (edsr từ pipeline/04_train_teacher_and_span_baseline.sh; 4 kiến trúc còn
#  lại từ RUN_ALL_extra_sr_baseline_distilled.sh <arch> — đã tồn tại vì
#  Table 10 hiện tại đã báo cáo đúng các giá trị n=1 lấy từ các checkpoint đó)
#
# CẢNH BÁO THỜI GIAN: 4 lần train SR mới mỗi kiến trúc (seed 123/2024/44/999
# — seed gốc/42 tái sử dụng, không train lại) + 5 lần eval_sr_quality.py
# (nhẹ, không train). KHÔNG có bước fine-tune recognition nào trong script
# này (khác biệt cố ý so với run_sr_seed_variance.sh, xem giải thích trên).

set -e

CONFIG="${1:-configs/config.yaml}"
ARCH="$2"
SEEDS=(123 2024 44 999)
NUM_WORKERS="${NUM_WORKERS:-4}"

if [ -z "$ARCH" ] || [[ ! "$ARCH" =~ ^(edsr|rlfn_adapted|ecbsr|safmn|smfanet)$ ]]; then
    echo "LỖI: thiếu hoặc sai tên kiến trúc." >&2
    echo "Dùng: bash pipeline/run_context_quality_seed_variance.sh [đường_dẫn_config] <edsr|rlfn_adapted|ecbsr|safmn|smfanet>" >&2
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không thấy $CONFIG" >&2
    exit 1
fi

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")

case "$ARCH" in
    edsr)
        TRAIN_SCRIPT="plain"
        BASE_RUN_DIR="sr_edsr"
        LABEL_BASE="edsr_teacher"
        ;;
    rlfn_adapted|ecbsr|safmn|smfanet)
        TRAIN_SCRIPT="distill"
        BASE_RUN_DIR="sr_improved_${ARCH}"
        LABEL_BASE="${ARCH}_distilled"
        ;;
    *)
        echo "LỖI: kiến trúc '$ARCH' không hợp lệ." >&2
        exit 1
        ;;
esac

BASE_CKPT="${RUNS_ROOT}/${BASE_RUN_DIR}/best.pt"
if [ ! -f "$BASE_CKPT" ]; then
    echo "LỖI: không thấy $BASE_CKPT — chạy xong checkpoint seed gốc cho '$ARCH' trước:" >&2
    if [ "$ARCH" = "edsr" ]; then
        echo "      bash pipeline/04_train_teacher_and_span_baseline.sh" >&2
    else
        echo "      bash RUN_ALL_extra_sr_baseline_distilled.sh $ARCH $CONFIG" >&2
    fi
    exit 1
fi

RESULTS_DIR="${RESULTS_ROOT}/context_quality_seed_variance_${ARCH}"
mkdir -p "$RESULTS_DIR"
OUT_CSV="${RESULTS_DIR}/sr_quality_srseed.csv"

echo "================================================================"
echo "Config     = $CONFIG"
echo "Arch       = $ARCH"
echo "Seeds mới  = ${SEEDS[*]} (seed gốc=42, tái sử dụng checkpoint sẵn có: $BASE_CKPT)"
echo "Out CSV    = $OUT_CSV"
echo "================================================================"

BASE_LABEL="${LABEL_BASE}_srseed42"
if [ -f "$OUT_CSV" ] && grep -q "^${BASE_LABEL}," "$OUT_CSV"; then
    echo ">>> Bỏ qua seed=42 (đã có dòng '$BASE_LABEL' trong $OUT_CSV)"
else
    echo ""
    echo "### seed=42 (gốc): TÁI SỬ DỤNG checkpoint sẵn có (không train lại) ###"
    python eval_sr_quality.py --config "$CONFIG" --arch "$ARCH" --ckpt "$BASE_CKPT" \
        --label "$BASE_LABEL" --out_csv "$OUT_CSV"
fi

for SEED in "${SEEDS[@]}"; do
    LABEL="${LABEL_BASE}_srseed${SEED}"
    RUN_SUFFIX="_srseed${SEED}"
    NEW_CKPT="${RUNS_ROOT}/${BASE_RUN_DIR}${RUN_SUFFIX}/best.pt"

    if [ -f "$OUT_CSV" ] && grep -q "^${LABEL}," "$OUT_CSV"; then
        echo ""
        echo ">>> Bỏ qua $ARCH | seed=$SEED (đã có dòng '$LABEL' trong $OUT_CSV)"
        continue
    fi

    echo ""
    echo "################################################################"
    echo "# $ARCH | seed=$SEED (checkpoint MỚI)                            #"
    echo "################################################################"
    if [ "$TRAIN_SCRIPT" = "plain" ]; then
        python train_sr.py --config "$CONFIG" --sr_arch "$ARCH" \
            --seed "$SEED" --run_suffix "$RUN_SUFFIX" --num_workers "$NUM_WORKERS"
    else
        # [Y HỆT cảnh báo của RUN_ALL_extra_sr_baseline_distilled.sh — xem
        # giải thích đầy đủ ở đó] ECBSR + feature-KD (lambda_feat>0) khiến
        # SRFeatureHook fallback âm thầm. Với config.yaml mặc định
        # (lambda_feat=0.0) cảnh báo này không kích hoạt, giữ lại để an toàn
        # nếu config sau này đổi.
        if [ "$ARCH" = "ecbsr" ]; then
            LAMBDA_FEAT_CHECK=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['sr_improve'].get('lambda_feat', 0.0))")
            if python -c "exit(0 if float('$LAMBDA_FEAT_CHECK') > 0 else 1)"; then
                echo "!!! CẢNH BÁO: ARCH=ecbsr VÀ lambda_feat=$LAMBDA_FEAT_CHECK > 0 — xem RUN_ALL_extra_sr_baseline_distilled.sh để biết chi tiết fallback hook." >&2
            fi
        fi
        python train_sr_distill.py --config "$CONFIG" --student_arch "$ARCH" \
            --seed "$SEED" --run_suffix "$RUN_SUFFIX" --num_workers "$NUM_WORKERS"
    fi

    echo "### Đo PSNR/SSIM/LPIPS — $ARCH seed=$SEED ###"
    python eval_sr_quality.py --config "$CONFIG" --arch "$ARCH" --ckpt "$NEW_CKPT" \
        --label "$LABEL" --out_csv "$OUT_CSV"
done

echo ""
echo "HOÀN TẤT ($ARCH). Kết quả: $OUT_CSV"
echo "Kỳ vọng 5 dòng: ${LABEL_BASE}_srseed{42,123,2024,44,999}."
echo "Dùng cột psnr_db/ssim/lpips để tính mean±std, cập nhật dòng '$ARCH' trong Table 10."
