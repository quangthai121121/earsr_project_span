#!/bin/bash
# ================================================================
# CHẠY TOÀN BỘ PIPELINE EarVN1.0 TỪ ĐẦU (sau khi đã vá các bug nghiêm trọng:
# split, NaN gradient, RandomHorizontalFlip, MACs/FLOPs, micro-average
# accuracy — xem RUNBOOK_EarVN1.0.md mục "Danh sách lỗi đã phát hiện và vá").
#
# Điều phối lại các script ĐÃ CÓ SẴN (không viết lại logic của chúng), theo
# ĐÚNG thứ tự phụ thuộc trong RUNBOOK_EarVN1.0.md — kèm 1 LOG LỊCH SỬ
# (results/run_log/PIPELINE_RUN_LOG.md) ghi lại: bước nào chạy khi nào, lệnh
# chính xác gì, mất bao lâu, thành công hay thất bại — để sau này truy vết
# lại chính xác cách bảng kết quả cuối cùng được tạo ra.
#
# DÙNG:
#   bash pipeline/run_everything_from_scratch.sh                  # KHÔNG xoá dữ liệu cũ
#   bash pipeline/run_everything_from_scratch.sh --wipe-existing   # XOÁ runs/ splits/ results/ cũ trước khi chạy
#
# LƯU Ý AN TOÀN: --wipe-existing là hành động KHÔNG THỂ HOÀN TÁC (rm -rf).
# Cờ này phải được truyền tường minh lúc gọi lệnh (không hỏi lại giữa chừng)
# — vì script này thường chạy qua đêm/nhiều ngày dưới nohup/tmux, một dấu
# nhắc xác nhận giữa chừng sẽ làm treo job vô thời hạn. KHÔNG đụng tới
# raw_data/, external/, checkpoints/span_pretrained_x4.pth dù có --wipe-existing.
#
# BẬT/TẮT từng giai đoạn tốn compute lớn (Track A/B, KD v2, saliency, learned
# pruning) bằng cách sửa các biến RUN_* ngay dưới đây — mặc định TẤT CẢ bật
# ("chạy toàn bộ"), comment ước tính thời gian ghi ngay cạnh mỗi biến.

set -e

RUN_TRACK_AB=true          # Bước 10.6: RLFN/ECBSR/SAFMN/SMFANet Track A+B — RẤT TỐN (9 kiến trúc x train+multiseed)
RUN_KD_V2=true             # Mục 12: multi-judge + feature-KD — nhẹ (CHỈ sàng lọc, DỪNG lại chờ bạn chọn lambda trước khi multi-seed)
RUN_SALIENCY=true          # Mục 13.1: saliency-weighted loss sweep — vừa (6 mức x 3 seed)
RUN_LEARNED_PRUNE_SCREEN=true  # Mục 13.2: sàng lọc lambda_sparsity (screen + confound-check) — nhẹ, KHÔNG tự chạy multi-seed đầy đủ (cần quyết định người dùng sau khi xem kết quả sàng lọc)

CONFIG="configs/config.yaml"
RESULTS_DIR="results"
LOG_DIR="${RESULTS_DIR}/run_log"
MAIN_LOG="${LOG_DIR}/PIPELINE_RUN_LOG.md"

# ---------------------------------------------------------------
# [0] Xoá dữ liệu cũ (chỉ khi truyền --wipe-existing)
# ---------------------------------------------------------------
if [ "$1" == "--wipe-existing" ]; then
    echo "!!! --wipe-existing: xoá runs/ splits/ results/ (raw_data/, external/, checkpoints/ GIỮ NGUYÊN)"
    rm -rf runs splits results
    echo ">>> Đã xoá xong."
elif [ -n "$1" ]; then
    echo "LỖI: tham số không hợp lệ '$1'. Dùng --wipe-existing hoặc để trống."
    exit 1
fi

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------
# Hàm log: ghi lại lệnh + thời gian + kết quả vào MAIN_LOG (Markdown), toàn
# bộ stdout/stderr của bước đó lưu riêng ra 1 file trong LOG_DIR để không
# làm loãng log chính. Dừng NGAY nếu bước thất bại (giữ tinh thần set -e,
# nhưng ghi log RÕ RÀNG trước khi dừng thay vì chỉ có traceback bash).
run_step() {
    local step_name="$1"
    shift
    local safe_name
    safe_name=$(echo "$step_name" | tr -c '[:alnum:]_-' '_')
    local step_log="${LOG_DIR}/${safe_name}.log"
    local start_ts start_epoch end_ts dur rc

    start_ts=$(date '+%Y-%m-%d %H:%M:%S')
    start_epoch=$(date +%s)
    {
        echo "## ${step_name}"
        echo "- Bắt đầu: ${start_ts}"
        echo "- Lệnh: \`$*\`"
        echo "- Log chi tiết: \`${step_log}\`"
    } >> "$MAIN_LOG"

    # [MỚI — phát hiện qua chạy thật] TRƯỚC ĐÂY không in gì ra terminal cho tới
    # khi bước xong — với các bước tốn hàng giờ (train 10 model...), terminal
    # đứng yên trông như bị treo dù thực ra vẫn chạy (output nằm hết trong
    # step_log). In ngay dòng "ĐANG CHẠY" kèm đường dẫn log cụ thể để biết theo
    # dõi bằng `tail -f` ở đâu, không phải đoán/tính tay tên file đã bị tr biến đổi.
    echo ">>> [ĐANG CHẠY] ${step_name} — theo dõi: tail -f ${step_log}"

    if "$@" > "$step_log" 2>&1; then
        end_ts=$(date '+%Y-%m-%d %H:%M:%S')
        dur=$(( $(date +%s) - start_epoch ))
        {
            echo "- Kết thúc: ${end_ts} (${dur}s) — **THÀNH CÔNG**"
            echo ""
        } >> "$MAIN_LOG"
        echo ">>> [OK] ${step_name} (${dur}s)"
    else
        rc=$?
        end_ts=$(date '+%Y-%m-%d %H:%M:%S')
        {
            echo "- Kết thúc: ${end_ts} — **THẤT BẠI** (exit code ${rc})"
            echo ""
        } >> "$MAIN_LOG"
        echo "!!! [FAIL] ${step_name} — exit code ${rc}. Chi tiết: ${step_log}" >&2
        exit "$rc"
    fi
}

# ---------------------------------------------------------------
# Ghi header lịch sử: thời điểm chạy, commit git hiện tại (nếu có), trạng
# thái working tree — để biết CHÍNH XÁC bản code nào tạo ra kết quả này.
# ---------------------------------------------------------------
{
    echo "# LỊCH SỬ CHẠY PIPELINE — EarVN1.0"
    echo ""
    echo "- Bắt đầu toàn bộ: $(date '+%Y-%m-%d %H:%M:%S')"
    if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        echo "- Git commit: \`$(git rev-parse HEAD)\`"
        if [ -n "$(git status --short 2>/dev/null)" ]; then
            echo "- **CẢNH BÁO**: working tree có thay đổi CHƯA COMMIT lúc chạy — xem \`git status\`/\`git diff\` để biết chính xác code nào đã dùng, đừng chỉ tin vào commit hash ở trên."
        fi
    else
        echo "- (không phải git repo, không xác định được commit)"
    fi
    echo "- Cờ bật/tắt: RUN_TRACK_AB=${RUN_TRACK_AB}, RUN_KD_V2=${RUN_KD_V2}, RUN_SALIENCY=${RUN_SALIENCY}, RUN_LEARNED_PRUNE_SCREEN=${RUN_LEARNED_PRUNE_SCREEN}"
    echo ""
} >> "$MAIN_LOG"

# ---------------------------------------------------------------
# [1] Pipeline chính — Bước 1-9 (RUNBOOK_EarVN1.0.md mục 1-9)
# ---------------------------------------------------------------
# run_step "Bước 1 - Chuẩn bị dữ liệu"        bash pipeline/01_survey_and_prepare_data.sh
# run_step "Bước 2 - nêính thức"   bash pipeline/02_setup_span_official.sh
# run_step "Bước 3 - Recognition baseline"    bash pipeline/03_train_baseline_recognition.sh
# run_step "Bước 4 - EDSR + SPAN baseline"    bash pipeline/04_train_teacher_and_span_baseline.sh
# run_step "Bước 5 - Recognition sr_baseline" bash pipeline/05_train_recognition_sr_baseline.sh
# run_step "Bước 6 - Nen SPAN (span_tiny)"    bash pipeline/06_improve_span.sh
# run_step "Bước 7 - Recognition sr_improved" bash pipeline/07_train_recognition_sr_improved.sh
# run_step "Bước 8 - Benchmark + aggregate"   bash pipeline/08_benchmark_and_aggregate.sh

STUDENT_ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")
SR_ARCH=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['sr']['arch'])")
# run_step "Bước 9 - Xuat anh so sanh" \
#     python export_sr_comparison_images.py --config "$CONFIG" \
#         --sr_baseline_ckpt "runs/sr_${SR_ARCH}/best.pt" --sr_baseline_arch "$SR_ARCH" \
#         --sr_improved_ckpt "runs/sr_improved_${STUDENT_ARCH}/best.pt" --sr_improved_arch "$STUDENT_ARCH" \
#         --n_samples 20 --out_dir "${RESULTS_DIR}/sr_comparison_images"

# ---------------------------------------------------------------
# [2] Kiểm chứng độ tin cậy — Bước 10.1-10.5 (bắt buộc cho journal Q1)
# ---------------------------------------------------------------
# run_step "Buoc 10.1 - Ablation loss"                bash pipeline/run_ablation.sh
# run_step "Buoc 10.2 - Lambda identity sweep"        bash pipeline/run_lambda_sweep.sh
run_step "Buoc 10.3a - Multi-seed n=3"              bash pipeline/run_multi_seed.sh
run_step "Buoc 10.3b - Multi-seed extra seeds n=5"  bash pipeline/run_multi_seed_extra_seeds.sh
# [MỚI — phát hiện qua review Q1] Câu hỏi "distillation (KD) có thật sự giúp
# span_tiny không?" chỉ có bằng chứng n=1 (results/ablation.csv gốc) trước
# đây — script này trả lời bằng n=5 seed THẬT (paired t-test + Cohen's d cho
# cặp pixel_only vs pixel_distill), tái sử dụng checkpoint SR đã có từ Bước
# 10.1 (không train lại SR). Cần Bước 10.1 + 10.3b (checkpoint recognition_lr
# seed 44/999) đã xong trước đó — đúng thứ tự đã đặt ở trên.
run_step "Buoc 10.1b - Ablation loss multi-seed n=5 (KD co giup khong)" \
    bash pipeline/run_ablation_multiseed.sh
run_step "Buoc 10.4a - Span_large ablation n=3"     bash RUN_ALL_span_large_ablation.sh
run_step "Buoc 10.4b - Span_large ablation n=5"     bash RUN_ALL_span_large_ablation_extra_seeds.sh
run_step "Buoc 10.5 - SR-seed variance"             bash pipeline/run_sr_seed_variance.sh

# ---------------------------------------------------------------
# [3] Bước 10.6 — Baseline SR ngoài họ SPAN, Track A + Track B (TỐN COMPUTE)
# ---------------------------------------------------------------
if [ "$RUN_TRACK_AB" = true ]; then
    TRACK_A_ARCHS=(rlfn rlfn_adapted ecbsr safmn smfanet)
    TRACK_B_ARCHS=(rlfn_adapted ecbsr safmn smfanet)

    for ARCH in "${TRACK_A_ARCHS[@]}"; do
        run_step "Buoc 10.6 Track A - ${ARCH} (n=3)" bash RUN_ALL_extra_sr_baseline.sh "$ARCH"
        run_step "Buoc 10.6 Track A - ${ARCH} (n=5)" bash RUN_ALL_extra_sr_baseline_extra_seeds.sh "$ARCH"
    done
    for ARCH in "${TRACK_B_ARCHS[@]}"; do
        run_step "Buoc 10.6 Track B - ${ARCH} (n=3)" bash RUN_ALL_extra_sr_baseline_distilled.sh "$ARCH"
        run_step "Buoc 10.6 Track B - ${ARCH} (n=5)" bash RUN_ALL_extra_sr_baseline_distilled_extra_seeds.sh "$ARCH"
    done
else
    echo "(RUN_TRACK_AB=false — bỏ qua Bước 10.6, xem RUNBOOK_EarVN1.0.md mục 10.6 để chạy tay sau)" | tee -a "$MAIN_LOG"
fi

# ---------------------------------------------------------------
# [4] Mục 12 — KD v2 (multi-judge + feature-KD)
#
# [SỬA — bug phát hiện qua review Q1] TRƯỚC ĐÂY chạy sàng lọc rồi TỰ ĐỘNG
# chạy thẳng multi-seed validate với LAMBDA_FEAT=0.5/LAMBDA_IDENTITY=0.1 PIN
# CỨNG ngay trong pipeline/run_multi_seed_kdv2.sh — đây là cấu hình "thắng"
# từ lần chạy TRƯỚC (trên split/code CŨ, trước các bản vá NaN/split/flip),
# KHÔNG có gì đảm bảo vẫn là cấu hình thắng trên code/dữ liệu MỚI. Tự động
# chạy tiếp với giá trị pin cũ có thể validate NHẦM recipe — cùng loại lỗi
# đã tránh đúng cho learned pruning (mục 13.2) nhưng bị bỏ sót ở đây. Sửa:
# CHỈ chạy sàng lọc, DỪNG lại để người dùng tự xem + sửa 2 tham số trong
# pipeline/run_multi_seed_kdv2.sh trước khi chạy tay bước multi-seed.
# ---------------------------------------------------------------
if [ "$RUN_KD_V2" = true ]; then
    run_step "Muc 12 - KD v2 ablation (sang loc)" bash pipeline/run_ablation_kd_v2.sh

    echo "" | tee -a "$MAIN_LOG"
    echo "!!! DỪNG TỰ ĐỘNG ở mục 12 — xem results/ablation_kd_v2.csv (hoặc" | tee -a "$MAIN_LOG"
    echo "    ablation_kd_v2_summary.csv nếu đã tổng hợp), xác nhận LAMBDA_FEAT/" | tee -a "$MAIN_LOG"
    echo "    LAMBDA_IDENTITY nào thắng trên dữ liệu MỚI (có thể KHÁC 0.5/0.1 cũ" | tee -a "$MAIN_LOG"
    echo "    vì code/split đã thay đổi), sửa 2 dòng đầu pipeline/run_multi_seed_kdv2.sh" | tee -a "$MAIN_LOG"
    echo "    cho khớp, rồi mới chạy tay:" | tee -a "$MAIN_LOG"
    echo "        bash pipeline/run_multi_seed_kdv2.sh" | tee -a "$MAIN_LOG"
    echo "        bash pipeline/run_multi_seed_kdv2_extra_seeds.sh" | tee -a "$MAIN_LOG"
else
    echo "(RUN_KD_V2=false — bỏ qua mục 12)" | tee -a "$MAIN_LOG"
fi

# ---------------------------------------------------------------
# [5] Mục 13.1 — Saliency-weighted loss sweep
# ---------------------------------------------------------------
if [ "$RUN_SALIENCY" = true ]; then
    run_step "Muc 13.1 - Saliency lambda sweep" bash pipeline/run_lambda_saliency_sweep.sh
else
    echo "(RUN_SALIENCY=false — bỏ qua mục 13.1)" | tee -a "$MAIN_LOG"
fi

# ---------------------------------------------------------------
# [6] Mục 13.2 — Learned block pruning: sàng lọc (KHÔNG tự chạy multi-seed
# đầy đủ — cần bạn tự xem kết quả sàng lọc rồi CHỌN lambda_sparsity/có hay
# không feat-KD+identity trước khi chạy pipeline/run_multi_seed_learned_prune.sh,
# đây là quyết định con người, không tự động hoá được).
# ---------------------------------------------------------------
if [ "$RUN_LEARNED_PRUNE_SCREEN" = true ]; then
    run_step "Muc 13.2a - Sang loc lambda_sparsity (pin feat/identity kdv2_full)" \
        bash pipeline/run_prune_sparsity_screen.sh

    # [MỚI — thí nghiệm tách nhiễu bàn ở lượt trước] cùng lambda_sparsity=0.2
    # nhưng TẮT feat-KD/saliency/identity — cô lập đúng biến "cách chọn khối"
    # so với span_tiny (không lẫn hiệu ứng KD v2 đang trend yếu).
    PRUNE_CONFOUND_DIR="${RESULTS_DIR}/prune_confound_check"
    PRUNE_CONFOUND_TAG="lsp0.2_purepixel"
    mkdir -p "$PRUNE_CONFOUND_DIR"
    run_step "Muc 13.2b - Confound-check (pure pixel+distill, lambda_sparsity=0.2)" \
        python train_sr_learned_prune.py --config "$CONFIG" \
            --lambda_feat 0 --lambda_saliency 0 --lambda_identity 0 \
            --lambda_sparsity 0.2 --run_suffix "_${PRUNE_CONFOUND_TAG}"

    PRUNE_CONFOUND_NBLOCKS=$(python -c "import json; print(json.load(open('runs/sr_learned_prune_${PRUNE_CONFOUND_TAG}/prune_metadata.json'))['n_blocks_kept'])")
    echo "Confound-check: lambda_sparsity=0.2 (pure pixel+distill) giữ lại ${PRUNE_CONFOUND_NBLOCKS} khối (kỳ vọng 3, khớp span_tiny)" | tee -a "$MAIN_LOG"

    run_step "Muc 13.2b - Sinh anh SR (confound-check)" \
        python data/build_sr.py --lr_dir splits/lr \
            --sr_ckpt "runs/sr_learned_prune_${PRUNE_CONFOUND_TAG}/best.pt" \
            --arch span_pruned --n_blocks "$PRUNE_CONFOUND_NBLOCKS" \
            --scale "$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")" \
            --out_dir "splits/sr_prune_${PRUNE_CONFOUND_TAG}"

    run_step "Muc 13.2b - Recognition (confound-check)" \
        python train_recognition.py --config "$CONFIG" --domain "sr_prune_${PRUNE_CONFOUND_TAG}" \
            --backbone mobilenet_v2 --init_ckpt "runs/recognition_lr_mobilenet_v2/best.pt"

    run_step "Muc 13.2b - Eval accuracy (confound-check)" \
        python eval_recognition.py --config "$CONFIG" \
            --ckpt "runs/recognition_sr_prune_${PRUNE_CONFOUND_TAG}_mobilenet_v2/best.pt" \
            --backbone mobilenet_v2 --train_domain "sr_prune_${PRUNE_CONFOUND_TAG}" \
            --test_domain "sr_prune_${PRUNE_CONFOUND_TAG}" \
            --out_json "${PRUNE_CONFOUND_DIR}/acc_${PRUNE_CONFOUND_TAG}_nblocks${PRUNE_CONFOUND_NBLOCKS}.json"

    run_step "Muc 13.2b - Eval SR quality (confound-check)" \
        python eval_sr_quality.py --config "$CONFIG" --arch span_pruned --n_blocks "$PRUNE_CONFOUND_NBLOCKS" \
            --ckpt "runs/sr_learned_prune_${PRUNE_CONFOUND_TAG}/best.pt" \
            --label "prune_${PRUNE_CONFOUND_TAG}_nblocks${PRUNE_CONFOUND_NBLOCKS}" \
            --out_csv "${PRUNE_CONFOUND_DIR}/sr_quality_confound_check.csv"

    echo "" | tee -a "$MAIN_LOG"
    echo "!!! DỪNG TỰ ĐỘNG ở mục 13.2 — xem ${RESULTS_DIR}/prune_sparsity_screen/ VÀ ${PRUNE_CONFOUND_DIR}/," | tee -a "$MAIN_LOG"
    echo "    tự chọn lambda_sparsity (+ có dùng feat/identity hay không) rồi mới chạy tay:" | tee -a "$MAIN_LOG"
    echo "    bash pipeline/run_multi_seed_learned_prune.sh" | tee -a "$MAIN_LOG"
else
    echo "(RUN_LEARNED_PRUNE_SCREEN=false — bỏ qua mục 13.2)" | tee -a "$MAIN_LOG"
fi

# ---------------------------------------------------------------
# [7] Tổng hợp báo cáo cuối cùng
# ---------------------------------------------------------------
run_step "Tong hop REPORT.md cuoi cung" \
    python data/generate_final_report.py --config "$CONFIG" --results_dir "$RESULTS_DIR" \
        --out_dir "${RESULTS_DIR}/final_report"

{
    echo "---"
    echo "- Kết thúc toàn bộ: $(date '+%Y-%m-%d %H:%M:%S')"
} >> "$MAIN_LOG"

echo ""
echo "################################################################"
echo "# HOÀN TẤT. Đọc theo thứ tự:                                    #"
echo "################################################################"
echo "  1. ${RESULTS_DIR}/final_report/REPORT.md   <- bảng kết quả tổng hợp"
echo "  2. ${MAIN_LOG}   <- lịch sử đầy đủ: bước nào chạy khi nào, lệnh gì, bao lâu, có lỗi không"
echo "  3. ${LOG_DIR}/*.log   <- log chi tiết (stdout/stderr) của từng bước, khi cần truy vết sâu hơn"
