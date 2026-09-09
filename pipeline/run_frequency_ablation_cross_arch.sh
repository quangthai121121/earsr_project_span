#!/bin/bash
# [MỚI — kiểm chứng tổng quát hoá của phát hiện tần số thấp (Section~
# sec:freq-ablation) ra ngoài SPAN/span_tiny] Câu hỏi: tín hiệu nhận dạng
# tập trung ở tần số thấp là tính chất của BÀI TOÁN nhận dạng tai ở độ phân
# giải cực thấp (20x20px) nói chung, hay chỉ là đặc thù riêng của cách SPAN
# tái tạo ảnh? Script này áp lại ĐÚNG bài test đã dùng cho span_tiny
# (eval_frequency_ablation.py, KHÔNG train lại gì) lên SR output của SAFMN
# và SMFANet -- 2 kiến trúc SR dùng cơ chế khác hẳn SPAN (đã có trong bài,
# Section~sec:res-context và Section~sec:disc-depth), ở CẢ full-depth (n_blocks
# mặc định 8, dùng cho Table~tab:quality/tab:context) VÀ half-depth (n_blocks=4,
# dùng cho attention-parameterization ablation, Table~tab:attn-ablation) --
# 4 tổ hợp kiến trúc x độ sâu, tách bạch được "do kiến trúc" hay "do độ sâu".
#
# KHÔNG TRAIN LẠI GÌ CẢ -- SR output và checkpoint recognition cho cả 4 tổ
# hợp đã có sẵn từ 2 thí nghiệm khác của bài (RUN_ALL_extra_sr_baseline_
# distilled*.sh cho full-depth, RUN_ALL_attention_param_ablation*.sh cho
# half-depth) -- chỉ đọc lại, lọc FFT, rồi eval. Chi phí = số lượt inference
# thuần tuý, không có backward/optimizer.
#
# DÙNG:
#   bash pipeline/run_frequency_ablation_cross_arch.sh [abridged|full_sweep] [đường_dẫn_config]
#
#   abridged (mặc định): CHỈ 2 điều kiện matched-severity (lowpass 0.1,
#     highpass 0.1) mỗi tổ hợp -- đúng 2 điểm đã dùng để tóm tắt phát hiện
#     chính cho span_tiny (Table~tab:freq-ablation). Rẻ, chạy trước để có
#     tín hiệu nhanh.
#   full_sweep: đủ 16 điều kiện mỗi tổ hợp, đủ để vẽ đường cong kiểu Figure~
#     fig:freq-ablation cho cả SAFMN/SMFANet -- thuyết phục hơn nhưng tốn
#     gấp 8 lần số lượt inference. Chạy SAU khi "abridged" đã cho tín hiệu
#     đáng chạy tiếp.
#
# [SỬA — bug phát hiện qua tự review, TRƯỚC KHI giao] Bản đầu dùng CHUNG 1
# RESULTS_DIR cho cả 2 mode (chỉ đặt tên theo LABEL kiến trúc, không theo
# mode) -- nếu chạy "abridged" trước (ghi ra file CSV chỉ 2 dòng) rồi sau đó
# chạy "full_sweep" trên CÙNG thư mục, bước kiểm tra resumability
# (`if [ -f "$OUT_CSV" ]`) sẽ thấy file đã tồn tại và BỎ QUA -- lặp lại
# ĐÚNG file CSV cũ chỉ có 2/16 điều kiện, không hề chạy full_sweep thật, mà
# KHÔNG báo lỗi gì (âm thầm sai) -- đúng kịch bản "chạy abridged trước, full
# sau nếu có thời gian" mà thí nghiệm này được thiết kế để hỗ trợ. Sửa bằng
# cách đưa MODE vào tên RESULTS_DIR -- 2 mode ghi ra 2 thư mục tách biệt
# hoàn toàn, không thể lẫn lộn hay đọc nhầm file cũ.
# Đổi tên mode "full" -> "full_sweep" VÀ label kiến trúc "*_full" ->
# "*_fulldepth" (cùng lý do: "full" bị dùng cho 2 nghĩa khác nhau -- độ sâu
# kiến trúc và độ rộng sweep cutoff -- gây nhầm lẫn tên thư mục nếu ghép lại
# (ví dụ "frequency_ablation_safmn_full_full").
#
# TIỀN ĐỀ (kiểm tra tự động ở BƯỚC 0, dừng ngay nếu thiếu):
#   - splits/sr_improved_safmn/, splits/sr_improved_safmn_half4/,
#     splits/sr_improved_smfanet/, splits/sr_improved_smfanet_half4/ (SR
#     output trên test set, đã build bởi 2 script RUN_ALL_* nêu trên).
#   - runs/recognition_sr_improved_<arch>[_half4]_<backbone>_seed<seed>/best.pt
#     cho cả 11 backbone x 5 seed x 4 tổ hợp (220 checkpoint).

set -e

MODE="${1:-abridged}"
CONFIG="${2:-configs/config.yaml}"

if [[ "$MODE" != "abridged" && "$MODE" != "full_sweep" ]]; then
    echo "LỖI: mode phải là 'abridged' hoặc 'full_sweep', nhận '$MODE'"
    echo "Dùng: bash pipeline/run_frequency_ablation_cross_arch.sh [abridged|full_sweep] [đường_dẫn_config]"
    exit 1
fi

BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100" \
           "shufflenet_v2_x1_0" "squeezenet1_1" "mobilenet_v3_large" "regnet_y_400mf" \
           "mobileone_s0" "lcnet_100")
SEEDS=(42 123 2024 44 999)
NUM_WORKERS="${NUM_WORKERS:-4}"

# [Tương thích bash 3.2 -- xem giải thích trong pipeline/run_detail_bottleneck_screening.sh]
# Mảng thường song song theo index thay vì associative array.
DOMAINS=(sr_improved_safmn sr_improved_safmn_half4 sr_improved_smfanet sr_improved_smfanet_half4)
LABELS=(safmn_fulldepth safmn_halfdepth smfanet_fulldepth smfanet_halfdepth)

if [ "$MODE" == "abridged" ]; then
    ONLY_CUTOFF_FLAG=(--only_cutoff 0.1)
else
    ONLY_CUTOFF_FLAG=()
fi

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
SPLITS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['splits_root'])")

echo "################################################################"
echo "# BƯỚC 0 — Kiểm tra tiền đề (mode=$MODE, ${#DOMAINS[@]} tổ hợp kiến trúc x độ sâu)"
echo "################################################################"
# [SỬA — bug thiết kế phát hiện qua tự review lần 2] Bản trước dùng 1 cờ
# MISSING DUY NHẤT cho cả 4 tổ hợp -- nếu CHỈ 1 trong 4 tổ hợp thiếu tiền đề
# (rất có thể xảy ra thật: full-depth và half-depth đến từ 2 chuỗi script
# RUN_ALL_* KHÁC NHAU, chạy riêng theo từng kiến trúc -- hoàn toàn có thể
# safmn đã chạy xong cả 2 độ sâu nhưng smfanet thì chưa), script sẽ THOÁT
# HẲN, bỏ lỡ luôn 2-3 tổ hợp ĐÃ SẴN SÀNG mà không có lý do chính đáng. Giờ
# kiểm tra ĐỘC LẬP từng tổ hợp (mảng READY song song với DOMAINS/LABELS,
# tương thích bash 3.2 -- không dùng associative array), CẢNH BÁO rồi BỎ QUA
# đúng tổ hợp thiếu, vẫn chạy tiếp mọi tổ hợp đã sẵn sàng. Chỉ thoát hẳn nếu
# KHÔNG tổ hợp nào sẵn sàng (không có gì để làm).
READY=()
for i in "${!DOMAINS[@]}"; do
    DOMAIN="${DOMAINS[$i]}"
    LABEL="${LABELS[$i]}"
    DOMAIN_MISSING=0
    if [ ! -d "${SPLITS_ROOT}/${DOMAIN}" ]; then
        echo "CẢNH BÁO [$LABEL]: thiếu ${SPLITS_ROOT}/${DOMAIN} -> chạy "
        echo "  RUN_ALL_extra_sr_baseline_distilled*.sh hoặc RUN_ALL_attention_param_ablation*.sh "
        echo "  tương ứng trước. Tổ hợp này sẽ bị BỎ QUA ở lần chạy này."
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
        echo "OK [$LABEL]: đủ 11 backbone x 5 seed checkpoint."
    fi
done

ANY_READY=0
for r in "${READY[@]}"; do
    if [ "$r" -eq 1 ]; then ANY_READY=1; fi
done
if [ "$ANY_READY" -eq 0 ]; then
    echo ""
    echo "DỪNG LẠI — KHÔNG tổ hợp nào đủ tiền đề (xem cảnh báo ở trên), không có gì để chạy."
    exit 1
fi
echo ""
echo "Sẽ chạy các tổ hợp đã sẵn sàng, bỏ qua tổ hợp còn thiếu (xem cảnh báo ở trên nếu có)."

for i in "${!DOMAINS[@]}"; do
    DOMAIN="${DOMAINS[$i]}"
    LABEL="${LABELS[$i]}"
    if [ "${READY[$i]}" -eq 0 ]; then
        echo ""
        echo ">>> Bỏ qua tổ hợp [$LABEL] (thiếu tiền đề, xem cảnh báo ở BƯỚC 0 phía trên)."
        continue
    fi
    # [SỬA — xem chú thích đầu file] MODE nằm TRONG tên thư mục -- abridged
    # và full_sweep ghi ra 2 nơi tách biệt, không đè/đọc nhầm CSV của nhau.
    RESULTS_DIR="results/frequency_ablation_${LABEL}_${MODE}"
    mkdir -p "$RESULTS_DIR"

    echo ""
    echo "################################################################"
    echo "# [$LABEL/$MODE] BƯỚC 1 — Build domain ảnh đã lọc (1 lần, dùng chung cho"
    echo "# cả 2 mode -- build luôn đủ 16 điều kiện bất kể mode, vì bước build"
    echo "# này chỉ chạy 1 lần/domain trong khi bước eval chạy 55 lần/domain,"
    echo "# nên chi phí build không phải điểm nghẽn cần tối ưu theo mode)"
    echo "################################################################"
    python data/build_frequency_filtered_domains.py --config "$CONFIG" --source_domain "$DOMAIN"

    echo ""
    echo "################################################################"
    echo "# [$LABEL/$MODE] BƯỚC 2 — Eval recognition đã train sẵn (11 backbone x 5 seed)"
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

        echo ">>> Tổng hợp [$LABEL/$MODE] backbone=$BACKBONE (gộp 5 seed)..."
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
    echo "ĐÃ BỎ QUA (thiếu tiền đề): ${SKIPPED[*]} -- chạy lại script này sau khi bổ sung để có nốt."
fi
echo "Đọc results/frequency_ablation_<label>_${MODE}/<backbone>_summary.csv cho từng tổ hợp đã chạy."
echo "So sánh matched-severity (cutoff=0.1, lowpass vs highpass) với số liệu span_tiny đã có"
echo "trong bài (results/frequency_ablation/<backbone>_summary.csv, Table~tab:freq-ablation)"
echo "để xem pattern bất đối xứng có lặp lại trên cả 3 kiến trúc độc lập không."
