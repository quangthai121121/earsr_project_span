#!/bin/bash
# [MỚI — kiểm tra nhân quả cho Nghi ngờ B (shared-teacher confound), phát
# sinh từ Section~sec:freq-crossarch/Limitations] Nghi ngờ B: span_tiny,
# SAFMN, SMFANet (bản Track B, dùng cho Section~sec:freq-crossarch) đều
# được distill từ CÙNG 1 teacher -- nên phổ công suất (PSD) của 3 SR output
# này giống nhau bất thường (đã đo trong data/analyze_spectral_content.py),
# và pattern "nhận dạng phụ thuộc tần số thấp" lặp lại trên cả 3 kiến trúc
# CÓ THỂ chỉ là do 3 mô hình bị "đồng nhất hoá" một phần bởi teacher chung,
# không phải một tính chất kiến trúc-độc-lập của bài toán tái tạo tai ở
# 20x20px. Kiểm tra bằng HR control (domain "hr", eval_frequency_ablation.py)
# đã bác bỏ được Nghi ngờ A (backbone tự thiên vị tần số thấp) nhưng KHÔNG
# đụng tới Nghi ngờ B (HR nằm hoàn toàn ngoài họ distillation).
#
# Script này áp ĐÚNG bài test cũ (eval_frequency_ablation.py, KHÔNG train
# lại gì) lên domain sr_safmn / sr_smfanet -- SR output Track A, tức train
# BẰNG pixel-loss L1 THUẦN qua train_sr.py (RUN_ALL_extra_sr_baseline.sh),
# KHÔNG hề đi qua distillation, KHÔNG chạm teacher chung. Nếu pattern bất
# đối xứng (lowpass giữ acc tốt hơn highpass) vẫn lặp lại ở đây, đó là bằng
# chứng nhân quả trực tiếp bác bỏ Nghi ngờ B (pattern không cần teacher
# chung mới xuất hiện). Nếu pattern KHÔNG lặp lại / yếu hẳn đi, đó cũng là
# kết quả khoa học có giá trị -- cho thấy Nghi ngờ B có cơ sở thật.
#
# GIỚI HẠN QUAN TRỌNG (đã biết trước, không phải phát hiện mới của script
# này): RUN_ALL_extra_sr_baseline.sh (script train Track A) chỉ train
# recognition trên 5 backbone x 3 seed = 15 checkpoint/kiến trúc (bộ backbone
# CŨ, trước đợt mở rộng lên 11 backbone), KHÔNG PHẢI 11 backbone x 5 seed như
# Section~sec:freq-crossarch (Track B) -- kết quả ở đây có độ phủ backbone/
# seed THẤP HƠN, không so sánh ngang hàng trực tiếp về số lượng bằng chứng.
# Riêng SMFANet Track A đã biết hội tụ PSNR kém (20.906dB so với 26.330dB ở
# Track B, xem Table~tab:quality) -- nếu pattern khác đi ở SMFANet Track A,
# CẦN cân nhắc đó là do thiếu teacher hay do chất lượng SR tự thân kém, hai
# biến số bị trộn lẫn trong đúng 1 kiến trúc này. SAFMN Track A hội tụ tốt
# (27.355dB, ngang Track B của chính nó) nên là phép thử SẠCH hơn hẳn.
#
# TIỀN ĐỀ (kiểm tra tự động ở BƯỚC 0, CẢNH BÁO rồi BỎ QUA đúng kiến trúc
# thiếu, không dừng cả script nếu chỉ 1 trong 2 thiếu -- xem cùng lý do sửa
# lỗi ở pipeline/run_frequency_ablation_cross_arch.sh):
#   - splits/sr_safmn/, splits/sr_smfanet/ (SR output Track A trên test set)
#   - runs/recognition_sr_<arch>_<backbone>_seed<seed>/best.pt cho 5 backbone
#     x 3 seed x 2 kiến trúc (30 checkpoint) -- SẢN PHẨM CÓ SẴN của
#     RUN_ALL_extra_sr_baseline.sh <safmn|smfanet>, KHÔNG train mới gì nếu
#     script đó đã chạy xong trước đây (dùng để tính PSNR Track A trong
#     Table~tab:quality). Nếu CHƯA chạy xong (thiếu checkpoint), phải chạy
#     RUN_ALL_extra_sr_baseline.sh <safmn|smfanet> trước -- ĐÂY LÀ CHI PHÍ
#     THẬT (train recognition mới), không rẻ như các kiểm tra tần số khác
#     đã làm trong Section~sec:freq-crossarch (vốn chỉ tái sử dụng
#     checkpoint có sẵn).
#
# DÙNG:
#   bash pipeline/run_frequency_ablation_track_a.sh [abridged|full_sweep] [đường_dẫn_config]
#   (abridged mặc định: 2 điều kiện matched-severity cutoff=0.1, giống quy
#   ước ở run_frequency_ablation_cross_arch.sh)

set -e

MODE="${1:-abridged}"
CONFIG="${2:-configs/config.yaml}"

if [[ "$MODE" != "abridged" && "$MODE" != "full_sweep" ]]; then
    echo "LỖI: mode phải là 'abridged' hoặc 'full_sweep', nhận '$MODE'"
    echo "Dùng: bash pipeline/run_frequency_ablation_track_a.sh [abridged|full_sweep] [đường_dẫn_config]"
    exit 1
fi

# [Bộ backbone/seed ĐÚNG BẰNG bộ mà RUN_ALL_extra_sr_baseline.sh dùng để
# train Track A -- KHÔNG PHẢI bộ 11 backbone/5 seed của Track B. Cố tình
# nhỏ hơn, không mở rộng thêm ở đây (mở rộng = train recognition mới, chi
# phí thật, cần quyết định riêng chứ không lặng lẽ tự làm).]
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(42 123 2024)
NUM_WORKERS="${NUM_WORKERS:-4}"

DOMAINS=(sr_safmn sr_smfanet)
LABELS=(safmn_tracka smfanet_tracka)

if [ "$MODE" == "abridged" ]; then
    ONLY_CUTOFF_FLAG=(--only_cutoff 0.1)
else
    ONLY_CUTOFF_FLAG=()
fi

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
SPLITS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['splits_root'])")

echo "################################################################"
echo "# BƯỚC 0 — Kiểm tra tiền đề (mode=$MODE, ${#DOMAINS[@]} kiến trúc Track A)"
echo "################################################################"
READY=()
for i in "${!DOMAINS[@]}"; do
    DOMAIN="${DOMAINS[$i]}"
    LABEL="${LABELS[$i]}"
    DOMAIN_MISSING=0
    if [ ! -d "${SPLITS_ROOT}/${DOMAIN}" ]; then
        echo "CẢNH BÁO [$LABEL]: thiếu ${SPLITS_ROOT}/${DOMAIN} -> chạy "
        echo "  RUN_ALL_extra_sr_baseline.sh ${DOMAIN#sr_} trước. Kiến trúc này sẽ bị BỎ QUA ở lần chạy này."
        DOMAIN_MISSING=1
    fi
    for BACKBONE in "${BACKBONES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            CKPT="${RUNS_ROOT}/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt"
            if [ ! -f "$CKPT" ]; then
                echo "CẢNH BÁO [$LABEL]: thiếu $CKPT"
                DOMAIN_MISSING=1
            fi
        done
    done
    if [ "$DOMAIN_MISSING" -eq 1 ]; then
        READY[$i]=0
    else
        READY[$i]=1
        echo "OK [$LABEL]: đủ ${#BACKBONES[@]} backbone x ${#SEEDS[@]} seed checkpoint Track A."
    fi
done

ANY_READY=0
for r in "${READY[@]}"; do
    if [ "$r" -eq 1 ]; then ANY_READY=1; fi
done
if [ "$ANY_READY" -eq 0 ]; then
    echo ""
    echo "DỪNG LẠI — KHÔNG kiến trúc nào đủ tiền đề (xem cảnh báo ở trên)."
    echo "Cần chạy RUN_ALL_extra_sr_baseline.sh <safmn|smfanet> trước (train recognition mới, chi phí thật)."
    exit 1
fi
echo ""
echo "Sẽ chạy các kiến trúc đã sẵn sàng, bỏ qua kiến trúc còn thiếu (xem cảnh báo ở trên nếu có)."

for i in "${!DOMAINS[@]}"; do
    DOMAIN="${DOMAINS[$i]}"
    LABEL="${LABELS[$i]}"
    if [ "${READY[$i]}" -eq 0 ]; then
        echo ""
        echo ">>> Bỏ qua [$LABEL] (thiếu tiền đề, xem cảnh báo ở BƯỚC 0 phía trên)."
        continue
    fi
    RESULTS_DIR="results/frequency_ablation_${LABEL}_${MODE}"
    mkdir -p "$RESULTS_DIR"

    echo ""
    echo "################################################################"
    echo "# [$LABEL/$MODE] BƯỚC 1 — Build domain ảnh đã lọc"
    echo "################################################################"
    python data/build_frequency_filtered_domains.py --config "$CONFIG" --source_domain "$DOMAIN"

    echo ""
    echo "################################################################"
    echo "# [$LABEL/$MODE] BƯỚC 2 — Eval recognition đã train sẵn (Track A)"
    echo "################################################################"
    for BACKBONE in "${BACKBONES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            OUT_CSV="${RESULTS_DIR}/${BACKBONE}_seed${SEED}.csv"
            if [ -f "$OUT_CSV" ]; then
                echo ">>> Bỏ qua backbone=$BACKBONE seed=$SEED (đã có $OUT_CSV)"
                continue
            fi
            echo "---- [$LABEL/$MODE] backbone=$BACKBONE | seed=$SEED ----"
            CKPT="${RUNS_ROOT}/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt"
            python eval_frequency_ablation.py --config "$CONFIG" \
                --ckpt "$CKPT" --backbone "$BACKBONE" --domain "$DOMAIN" \
                --seed_label "$SEED" --num_workers "$NUM_WORKERS" \
                "${ONLY_CUTOFF_FLAG[@]}" \
                --out_csv "$OUT_CSV"
        done

        echo ">>> Tổng hợp [$LABEL/$MODE] backbone=$BACKBONE (gộp ${#SEEDS[@]} seed)..."
        python data/aggregate_frequency_ablation.py --results_dir "$RESULTS_DIR" \
            --backbone "$BACKBONE" \
            --out_csv "${RESULTS_DIR}/${BACKBONE}_summary.csv"
    done
done

echo ""
echo "HOÀN TẤT (mode=$MODE)."
PROCESSED=()
SKIPPED=()
for i in "${!DOMAINS[@]}"; do
    if [ "${READY[$i]}" -eq 1 ]; then
        PROCESSED+=("${LABELS[$i]}")
    else
        SKIPPED+=("${LABELS[$i]}")
    fi
done
echo "Đã chạy: ${PROCESSED[*]:-（không có）}"
if [ "${#SKIPPED[@]}" -gt 0 ]; then
    echo "ĐÃ BỎ QUA (thiếu tiền đề): ${SKIPPED[*]} -- cần chạy RUN_ALL_extra_sr_baseline.sh trước."
fi
echo "Đọc results/frequency_ablation_<label>_${MODE}/<backbone>_summary.csv."
echo "So sánh matched-severity (cutoff=0.1, lowpass vs highpass) với:"
echo "  - span_tiny (results/frequency_ablation/<backbone>_summary.csv)"
echo "  - SAFMN/SMFANet Track B (results/frequency_ablation_<safmn|smfanet>_fulldepth_${MODE}/<backbone>_summary.csv)"
echo "để xem pattern có LẶP LẠI khi KHÔNG có teacher chung (Track A) hay không."
echo "LƯU Ý: chỉ ${#BACKBONES[@]} backbone x ${#SEEDS[@]} seed (bộ cũ), không phải 11x5 -- đọc kết quả với độ tin cậy tương ứng."
