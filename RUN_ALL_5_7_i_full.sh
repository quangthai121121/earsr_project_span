#!/bin/bash
# ================================================================
# [MỚI — Mục 5.7(i)] Script tổng — chạy TOÀN BỘ ablation
# attention-parameterization (SAFMN + SMFANet, n=5 seed mỗi kiến
# trúc, rồi kiểm định thống kê) chỉ bằng MỘT lệnh.
#
# Đặt file này ở ĐÚNG thư mục gốc repo (cùng cấp với
# RUN_ALL_attention_param_ablation.sh, RUN_ALL_attention_param_ablation_extra_seeds.sh,
# data/compare_depth_deltas.py) trước khi chạy.
#
# DÙNG:
#   bash RUN_ALL_5_7_i_full.sh [đường_dẫn_config] [n_blocks_half]
#
# Mặc định: config=configs/config.yaml, n_blocks_half=4.
#
# TIỀN ĐỀ (script con sẽ tự kiểm tra và DỪNG NGAY nếu thiếu — xem
# BƯỚC 0 trong từng RUN_ALL_attention_param_ablation*.sh):
#   - checkpoint full-depth SAFMN + SMFANet (RUN_ALL_extra_sr_baseline_distilled.sh
#     <arch> [config], cả bản gốc lẫn _extra_seeds)
#   - recognition_lr_<backbone>_seed{42,123,2024,44,999} (pipeline/run_multi_seed.sh
#     + pipeline/run_multi_seed_extra_seeds.sh)
#   - JSON sr_span_large_*_seed{42,123,2024,44,999}.json
#     (RUN_ALL_span_large_ablation.sh + RUN_ALL_span_large_ablation_extra_seeds.sh)
#
# Nếu thiếu tiền đề nào, script DỪNG NGAY ở BƯỚC 0 của kiến trúc đó
# (không tốn giờ train oan) — chạy xong phần tiền đề còn thiếu rồi
# gọi lại script này, nó sẽ tự bỏ qua các bước đã xong (checkpoint đã
# tồn tại sẽ không train lại vì --run_suffix cố định theo n_blocks_half).
#
# set -e: bất kỳ bước nào lỗi (kể cả kiến trúc thứ nhất) sẽ dừng toàn
# bộ script ngay, KHÔNG âm thầm chạy tiếp sang kiến trúc thứ hai với
# trạng thái nghi ngờ — khớp quy ước fail-fast dùng xuyên suốt project.
set -e

CONFIG="${1:-configs/config.yaml}"
N_HALF="${2:-4}"
ARCHS=("safmn" "smfanet")

if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không tìm thấy config '$CONFIG'. Chạy script này từ thư mục gốc repo."
    exit 1
fi

for ARCH in "${ARCHS[@]}"; do
    echo ""
    echo "################################################################"
    echo "########   KIẾN TRÚC: $ARCH   (n_blocks_half=$N_HALF)   ################"
    echo "################################################################"

    echo ""
    echo ">>> [1/3] n=3 seed đầu (train half-depth + eval) — $ARCH"
    bash RUN_ALL_attention_param_ablation.sh "$ARCH" "$N_HALF" "$CONFIG"

    echo ""
    echo ">>> [2/3] +2 seed (44, 999) — $ARCH"
    bash RUN_ALL_attention_param_ablation_extra_seeds.sh "$ARCH" "$N_HALF" "$CONFIG"

    echo ""
    echo ">>> [3/3] Kiểm định Δ_${ARCH} vs Δ_SPAN — $ARCH"
    python data/compare_depth_deltas.py \
        --results_dir "results/attention_param_ablation_${ARCH}_half${N_HALF}/combined" \
        --arch_full_domain "sr_improved_${ARCH}" \
        --arch_half_domain "sr_improved_${ARCH}_half${N_HALF}" \
        --span_full_domain "sr_span_large" \
        --span_tiny_domain "sr_improved" \
        --out_csv "results/attention_param_ablation_${ARCH}_half${N_HALF}/depth_delta_comparison_${ARCH}.csv"

    echo ""
    echo ">>> XONG kiến trúc $ARCH."
done

echo ""
echo "################################################################"
echo "# HOÀN TẤT TOÀN BỘ Mục 5.7(i)                                   #"
echo "################################################################"
echo "Gửi lại 2 file này để hoàn thành phân tích + viết Section 5.7(i):"
for ARCH in "${ARCHS[@]}"; do
    echo "  - results/attention_param_ablation_${ARCH}_half${N_HALF}/depth_delta_comparison_${ARCH}.csv"
done
echo ""
echo "(Tuỳ chọn, nếu muốn có thêm bảng params/FLOPs/PSNR/latency half-depth vs"
echo "full-depth: gửi kèm các dòng mới trong results/sr_quality.csv có label"
echo "'safmn_half${N_HALF}_distilled' / 'smfanet_half${N_HALF}_distilled', do BƯỚC 3 của"
echo "RUN_ALL_attention_param_ablation.sh tự động append vào đó.)"