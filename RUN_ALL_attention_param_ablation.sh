#!/bin/bash
# ================================================================
# [MỚI — Mục 5.7(i), ablation attention-parameterization]
#
# MỤC ĐÍCH: kiểm định trực tiếp giả thuyết cơ chế nêu ở Mục 6 của bản thảo
# ("SPAN chịu nén sâu tốt vì attention không có tham số học, chỉ mất
# computation chứ không mất capacity"). Đây KHÔNG phải bảng so sánh ngữ cảnh
# (Bảng 5/README mục 11 — span_tiny vs SAFMN/SMFANet Ở CẤU HÌNH ĐẦY ĐỦ mặc
# định) — đó là câu hỏi khác. Câu hỏi ở ĐÂY là so sánh THEO CHIỀU DỌC, trong
# NỘI BỘ từng kiến trúc: nếu giảm ĐÚNG MỘT NỬA số khối của SAFMN/SMFANet
# (giữ mọi thứ khác cố định — cùng recipe distillation, cùng teacher, cùng
# lambda — giống hệt cách span_baseline (6 khối) -> span_tiny (3 khối)),
# thì mức SỤT accuracy downstream của SAFMN/SMFANet có LỚN HƠN mức sụt của
# SPAN không? Nếu có (và có ý nghĩa thống kê), đó là bằng chứng trực tiếp
# ủng hộ giả thuyết "attention không tham số chịu nén tốt hơn attention có
# tham số học" — một nguyên lý thiết kế tổng quát, không chỉ một quan sát
# cục bộ trên riêng SPAN.
#
# TIỀN ĐỀ:
#   - Đã chạy xong RUN_ALL_extra_sr_baseline_distilled.sh <arch> [config]
#     (bản FULL-DEPTH, n_blocks=8 mặc định) — script này TÁI SỬ DỤNG số liệu
#     đó làm nửa "trước" của phép so sánh, KHÔNG train lại.
#   - Đã có sẵn checkpoint span_tiny/span_baseline + multi-seed recognition
#     (pipeline chính, Bước 1-8) — dùng để tính Δ_SPAN đối chứng.
#   - Cần patch models/sr_models.py::build_sr_model() + train_sr_distill.py
#     (--student_n_blocks) đã áp dụng — xem CHANGELOG/diff kèm theo.
#
# DÙNG:
#   bash RUN_ALL_attention_param_ablation.sh <safmn|smfanet> [n_blocks_half] [đường_dẫn_config]
#   ví dụ:
#   bash RUN_ALL_attention_param_ablation.sh safmn 4
#   bash RUN_ALL_attention_param_ablation.sh smfanet 4
#
# n_blocks_half mặc định = 4 (một nửa của 8, giá trị mặc định gốc của cả
# SAFMN và SMFANet trong build_sr_model()) — chỉ cần đổi nếu bạn có lý do cụ
# thể để dùng tỷ lệ khác.
#
# LƯU Ý QUAN TRỌNG VỀ ECBSR: KHÔNG dùng script này cho ecbsr. Sau khi
# reparameterize, ECBSR gần như suy biến thành một conv đơn (xem báo cáo
# ablation trước) — không còn "attention có tham số học" thật để so sánh,
# nên không phải phép thử công bằng cho giả thuyết Mục 6.

set -e

ARCH="$1"
N_HALF="${2:-4}"
CONFIG="${3:-configs/config.yaml}"

if [ -z "$ARCH" ] || [[ ! "$ARCH" =~ ^(safmn|smfanet)$ ]]; then
    echo "LỖI: thiếu hoặc sai tên kiến trúc — script này CHỈ hỗ trợ safmn/smfanet"
    echo "     (attention có tham số học thật, khác ECBSR đã suy biến sau reparameterize)."
    echo "Dùng: bash RUN_ALL_attention_param_ablation.sh <safmn|smfanet> [n_blocks_half] [đường_dẫn_config]"
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không tìm thấy config '$CONFIG'."
    exit 1
fi

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")
SPLITS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['splits_root'])")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")
TEACHER_CKPT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['sr_improve']['teacher_ckpt'])")

TAG="${ARCH}_half${N_HALF}"
DOMAIN="sr_improved_${TAG}"
RESULTS_DIR="${RESULTS_ROOT}/attention_param_ablation_${TAG}"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(42 123 2024)   # 3 seed đầu — chạy RUN_ALL_attention_param_ablation_extra_seeds.sh
                       # để thêm 44/999, khớp quy ước n=5 dùng xuyên suốt project

echo "================================================================"
echo "ARCH        = $ARCH (half-depth, n_blocks=$N_HALF, mặc định gốc=8)"
echo "DOMAIN      = $DOMAIN"
echo "CONFIG      = $CONFIG"
echo "================================================================"

echo ""
echo "################################################################"
echo "# BƯỚC 0/5 — Kiểm tra tiền đề                                    #"
echo "################################################################"
MISSING=0
if [ ! -f "$TEACHER_CKPT" ]; then
    echo "LỖI: chưa thấy teacher_ckpt ($TEACHER_CKPT)"
    MISSING=1
fi
if [ ! -f "${RUNS_ROOT}/sr_improved_${ARCH}/best.pt" ]; then
    echo "LỖI: chưa thấy checkpoint full-depth ${RUNS_ROOT}/sr_improved_${ARCH}/best.pt"
    echo "     -> chạy 'bash RUN_ALL_extra_sr_baseline_distilled.sh $ARCH $CONFIG' trước"
    echo "        (script này TÁI SỬ DỤNG kết quả full-depth làm nửa 'trước' so sánh)."
    MISSING=1
fi
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: chưa thấy $CKPT -> chạy 'bash pipeline/run_multi_seed.sh' trước."
            MISSING=1
        fi
    done
done
# [SỬA — lỗi confound phát hiện qua review] Δ_SPAN đối chứng PHẢI dùng domain
# "sr_span_large" (CÙNG recipe distillation với span_tiny, xem
# RUN_ALL_span_large_ablation.sh) — KHÔNG dùng "sr_baseline" (teacher pixel-
# loss thuần, sẽ trộn lẫn biến recipe vào biến độ sâu). Kiểm tra tiền đề này
# NGAY TỪ ĐẦU (fail-fast) vì thiếu nó chỉ lộ ra ở BƯỚC cuối, sau hàng giờ train.
if ! ls "${RESULTS_ROOT}/span_large_ablation"/sr_span_large_*_seed*.json >/dev/null 2>&1; then
    echo "LỖI: chưa thấy JSON domain 'sr_span_large' trong ${RESULTS_ROOT}/span_large_ablation/"
    echo "     -> cần cho Δ_SPAN đối chứng (CÙNG recipe distillation, KHÔNG dùng sr_baseline)."
    echo "     -> chạy 'bash RUN_ALL_span_large_ablation.sh $CONFIG' trước."
    MISSING=1
fi
if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "DỪNG LẠI — thiếu tiền đề ở trên."
    exit 1
fi
echo "OK — mọi tiền đề đã sẵn sàng."
mkdir -p "$RESULTS_DIR"

echo ""
echo "################################################################"
echo "# BƯỚC 1/5 — Train $ARCH half-depth (n_blocks=$N_HALF) qua train_sr_distill.py #"
echo "################################################################"
# CÙNG recipe/teacher/lambda với bản full-depth đã train — CHỈ khác n_blocks
# của student. --run_suffix đảm bảo KHÔNG ghi đè checkpoint full-depht.
python train_sr_distill.py --config "$CONFIG" --student_arch "$ARCH" \
    --student_n_blocks "$N_HALF" --run_suffix "_half${N_HALF}"

echo ""
echo "################################################################"
echo "# BƯỚC 2/5 — Sinh tập ảnh ${DOMAIN}                              #"
echo "################################################################"
python data/build_sr.py --lr_dir "${SPLITS_ROOT}/lr" \
    --sr_ckpt "${RUNS_ROOT}/${DOMAIN}/best.pt" \
    --arch "$ARCH" --n_blocks "$N_HALF" --scale "$SCALE" \
    --out_dir "${SPLITS_ROOT}/${DOMAIN}"

echo ""
echo "################################################################"
echo "# BƯỚC 3/5 — Đo PSNR/SSIM/params/FLOPs/latency cho $ARCH half-depth #"
echo "################################################################"
python eval_sr_quality.py --config "$CONFIG" --arch "$ARCH" --n_blocks "$N_HALF" \
    --ckpt "${RUNS_ROOT}/${DOMAIN}/best.pt" --label "${TAG}_distilled" \
    --out_csv "${RESULTS_ROOT}/sr_quality.csv"
echo ">>> Đã APPEND dòng '${TAG}_distilled' vào ${RESULTS_ROOT}/sr_quality.csv."

echo ""
echo "################################################################"
echo "# BƯỚC 4/5 — Multi-seed recognition trên domain ${DOMAIN}        #"
echo "################################################################"
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        LR_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        echo "### backbone=$BACKBONE | seed=$SEED | domain=${DOMAIN}"
        python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" \
            --backbone "$BACKBONE" --init_ckpt "$LR_CKPT" \
            --seed "$SEED" --run_suffix "_seed${SEED}"
        python eval_recognition.py --config "$CONFIG" \
            --ckpt "${RUNS_ROOT}/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
            --out_json "${RESULTS_DIR}/${DOMAIN}_${BACKBONE}_seed${SEED}.json"
    done
done

echo ""
echo "################################################################"
echo "# BƯỚC 5/5 — Gộp JSON vào 1 thư mục 'combined' để so sánh Δ sau  #"
echo "################################################################"
mkdir -p "${RESULTS_DIR}/combined"
# Half-depth (arch, vừa chạy)
cp "${RESULTS_DIR}"/${DOMAIN}_*_seed*.json "${RESULTS_DIR}/combined/"
# Full-depth (arch, đã có sẵn từ RUN_ALL_extra_sr_baseline_distilled.sh)
FULL_RESULTS_DIR="${RESULTS_ROOT}/extra_sr_baseline_${ARCH}_distilled"
if ls "${FULL_RESULTS_DIR}"/sr_improved_${ARCH}_*_seed*.json >/dev/null 2>&1; then
    cp "${FULL_RESULTS_DIR}"/sr_improved_${ARCH}_*_seed*.json "${RESULTS_DIR}/combined/"
else
    echo "CẢNH BÁO: không thấy JSON full-depth trong ${FULL_RESULTS_DIR} — Δ_${ARCH} sẽ thiếu dữ liệu."
fi
# span_tiny (domain "sr_improved", pipeline chính) — cần cho Δ_SPAN đối chứng
if ls "${RESULTS_ROOT}/multi_seed"/*_seed*.json >/dev/null 2>&1; then
    cp "${RESULTS_ROOT}/multi_seed"/*_seed*.json "${RESULTS_DIR}/combined/"
else
    echo "CẢNH BÁO: không thấy JSON pipeline chính trong ${RESULTS_ROOT}/multi_seed — Δ_SPAN sẽ thiếu dữ liệu."
fi
# [SỬA — lỗi confound phát hiện qua review] span_large (domain "sr_span_large",
# CÙNG recipe distillation với span_tiny) — BẮT BUỘC cho Δ_SPAN đối chứng công
# bằng. TUYỆT ĐỐI KHÔNG dùng domain "sr_baseline" (teacher pixel-loss thuần,
# xem cảnh báo trong data/compare_depth_deltas.py) dù nó cũng có sẵn trong
# ${RESULTS_ROOT}/multi_seed đã copy ở trên.
if ls "${RESULTS_ROOT}/span_large_ablation"/sr_span_large_*_seed*.json >/dev/null 2>&1; then
    cp "${RESULTS_ROOT}/span_large_ablation"/sr_span_large_*_seed*.json "${RESULTS_DIR}/combined/"
else
    echo "CẢNH BÁO: không thấy JSON domain sr_span_large trong ${RESULTS_ROOT}/span_large_ablation — Δ_SPAN sẽ thiếu dữ liệu (BƯỚC 0 lẽ ra đã chặn trường hợp này)."
fi

echo ""
echo "################################################################"
echo "# HOÀN TẤT n=3 seed (arch: $ARCH, n_blocks_half: $N_HALF)         #"
echo "################################################################"
echo "Chạy tiếp để lên n=5 (khớp quy ước project):"
echo "  bash RUN_ALL_attention_param_ablation_extra_seeds.sh $ARCH $N_HALF $CONFIG"
echo ""
echo "Sau khi đủ n=5, tính kiểm định Δ_${ARCH} vs Δ_SPAN bằng:"
echo "  python data/compare_depth_deltas.py \\"
echo "      --results_dir ${RESULTS_DIR}/combined \\"
echo "      --arch_full_domain sr_improved_${ARCH} --arch_half_domain ${DOMAIN} \\"
echo "      --span_full_domain sr_span_large --span_tiny_domain sr_improved \\"
echo "      --out_csv ${RESULTS_DIR}/depth_delta_comparison_${ARCH}.csv"
