#!/bin/bash
# ================================================================
# CHẠY TOÀN BỘ PIPELINE AWEx (dataset thứ 2) TỪ ĐẦU — bản đối xứng với
# pipeline/run_everything_from_scratch.sh (EarVN1.0), theo đúng thứ tự
# Bước 0-5 trong RUNBOOK_AWEx.md.
#
# Điều phối lại các script ĐÃ CÓ SẴN (không viết lại logic của chúng) — kèm
# 1 LOG LỊCH SỬ (results_awex/run_log/PIPELINE_RUN_LOG_AWEX.md) giống hệt
# bản EarVN1.0, để truy vết sau này.
#
# DÙNG:
#   bash pipeline/run_everything_from_scratch_awex.sh                  # KHÔNG xoá dữ liệu cũ
#   bash pipeline/run_everything_from_scratch_awex.sh --wipe-existing   # XOÁ runs_awex/ splits_awex/
#                                                                        # results_awex/ pipeline_awex/
#                                                                        # configs/config_awex.yaml cũ
#
# LƯU Ý AN TOÀN: --wipe-existing CHỈ xoá dữ liệu mang hậu tố "_awex" (và
# pipeline_awex/, configs/config_awex.yaml) — TUYỆT ĐỐI KHÔNG đụng tới
# runs/, splits/, results/ (dữ liệu EarVN1.0, dataset NGUỒN mà Bước 4 transfer
# learning cần đọc từ đó). Cờ này phải được truyền tường minh (không hỏi lại
# giữa chừng) vì script chạy qua đêm dưới nohup/tmux.
#
# TIỀN ĐỀ BẮT BUỘC trước khi chạy script này:
#   1. raw_data/awex_raw/ đã tồn tại (336 thư mục 001-336) — xem RUNBOOK_AWEx.md
#      Bước 0 (tải + flatten + convert, CÓ BƯỚC --dry_run thủ công, không tự
#      động hoá được vì cần con người xác nhận số liệu thống kê hợp lý).
#   2. EarVN1.0 đã chạy xong ĐẦY ĐỦ n=5 seed cho 5 backbone (runs/recognition_
#      <lr|sr_baseline|sr_improved>_<backbone>_seed<42|123|2024|44|999>/best.pt)
#      — Bước 4 (transfer learning) đọc trực tiếp các checkpoint này làm
#      nguồn fine-tune, KHÔNG train lại từ đầu.
#
# NHẮC LẠI BÀI HỌC ĐÃ TRẢ GIÁ (2026-08 EarVN1.0): khi cần dừng tiến trình
# đang chạy, dùng `pkill -f run_everything_from_scratch_awex.sh` (và
# `pkill -f` tên script con đang chạy dở) — KHÔNG dùng `kill <PID>` một mình,
# vì không diệt được tiến trình con Python, để lại tiến trình mồ côi tranh
# chấp GPU qua nhiều lần restart.

set -e

RUN_SR_SEED_VARIANCE=true       # Bước 3: đo phương sai do SR-seed — nhẹ, nên bật
RUN_FREEZE_BACKBONE_SCREEN=false # Bước 5: CHỈ có ý nghĩa SAU KHI xem kết quả Bước 4
                                  # (RUNBOOK: "chỉ chạy sau khi đã xem kết quả Bước 4
                                  # và thấy phương sai giữa seed vẫn lớn") — quyết định
                                  # con người, KHÔNG bật mặc định, script tự dừng và in
                                  # hướng dẫn ở đúng chỗ thay vì đoán thay bạn.

CONFIG_AWEX="configs/config_awex.yaml"
RESULTS_DIR="results_awex"
LOG_DIR="${RESULTS_DIR}/run_log"
MAIN_LOG="${LOG_DIR}/PIPELINE_RUN_LOG_AWEX.md"

# ---------------------------------------------------------------
# [0] Xoá dữ liệu cũ (chỉ khi truyền --wipe-existing) — CHỈ đụng phần _awex.
# ---------------------------------------------------------------
if [ "$1" == "--wipe-existing" ]; then
    echo "!!! --wipe-existing: xoá runs_awex/ splits_awex/ results_awex/ pipeline_awex/"
    echo "    configs/config_awex.yaml (runs/ splits/ results/ của EarVN1.0 GIỮ NGUYÊN)"
    rm -rf runs_awex splits_awex results_awex pipeline_awex configs/config_awex.yaml
    echo ">>> Đã xoá xong."
elif [ -n "$1" ]; then
    echo "LỖI: tham số không hợp lệ '$1'. Dùng --wipe-existing hoặc để trống."
    exit 1
fi

mkdir -p "$LOG_DIR"

# [MỚI — 2026-08-30, phòng ngừa sự cố thật đã xảy ra với EarVN1.0] `kill <PID>`
# lên tiến trình bash CHA không tự động diệt tiến trình con Python đang chạy
# bên trong `run_step()` — đã gây tích luỹ nhiều `train_sr_distill.py` mồ côi
# tranh chấp cùng 1 GPU qua nhiều lần restart (xem RUNBOOK_EarVN1.0.md mục
# #25). Thay vì chỉ dựa vào việc LUÔN NHỚ dùng `pkill -f` đúng cách, thêm
# `trap` ở đây: khi script này nhận SIGTERM/SIGINT (dù bị dừng bằng cách nào,
# kể cả lỡ tay `kill <PID>` đúng lên PID này) — tự động diệt HẾT tiến trình
# con trực tiếp (`pkill -P $$`, dựa theo PID cha, không phụ thuộc tên lệnh cụ
# thể) trước khi thoát. Không thay thế được `pkill -f` khi CẦN diệt tiến
# trình cháu (con của con, ví dụ DataLoader worker) — vẫn nên dùng `pkill -f`
# khi có thể — nhưng đây là lưới an toàn cuối cùng nếu ai đó (kể cả tương lai)
# lỡ quên.
cleanup_children() {
    echo "" >&2
    echo "!!! Nhận tín hiệu dừng — đang diệt tiến trình con (tránh mồ côi tranh chấp GPU)..." >&2
    pkill -P $$ 2>/dev/null
    sleep 2
    pkill -9 -P $$ 2>/dev/null
    echo "!!! Đã dọn xong tiến trình con trực tiếp. Nếu vẫn còn tiến trình Python treo," >&2
    echo "    dùng: pkill -f train_sr_distill.py; pkill -f train_recognition.py; ..." >&2
    exit 1
}
trap cleanup_children SIGTERM SIGINT

# Hàm log — giống hệt bản EarVN1.0 (xem pipeline/run_everything_from_scratch.sh
# để biết lý do từng chi tiết: in "ĐANG CHẠY" ngay từ đầu để không tưởng bị
# treo, log riêng mỗi bước, dừng ngay khi có bước fail).
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
# Ghi header lịch sử — giống hệt bản EarVN1.0.
# ---------------------------------------------------------------
{
    echo "# LỊCH SỬ CHẠY PIPELINE — AWEx"
    echo ""
    echo "- Bắt đầu toàn bộ: $(date '+%Y-%m-%d %H:%M:%S')"
    if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        echo "- Git commit: \`$(git rev-parse HEAD)\`"
        if [ -n "$(git status --short 2>/dev/null)" ]; then
            echo "- **CẢNH BÁO**: working tree có thay đổi CHƯA COMMIT lúc chạy."
        fi
    fi
    echo "- Cờ bật/tắt: RUN_SR_SEED_VARIANCE=${RUN_SR_SEED_VARIANCE}, RUN_FREEZE_BACKBONE_SCREEN=${RUN_FREEZE_BACKBONE_SCREEN}"
    echo ""
} >> "$MAIN_LOG"

# ---------------------------------------------------------------
# [Kiểm tra tiền đề] — fail nhanh, TRƯỚC khi tốn bất kỳ giờ GPU nào.
# ---------------------------------------------------------------
echo ">>> Kiểm tra tiền đề..."
MISSING=0
if [ ! -d "raw_data/awex_raw" ] || [ -z "$(find raw_data/awex_raw -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null)" ]; then
    echo "LỖI: thiếu raw_data/awex_raw/ (hoặc rỗng) — xem RUNBOOK_AWEx.md Bước 0" >&2
    echo "     (tải + flatten_awex.sh + convert_generic_dataset_to_project_format.py" >&2
    echo "     --dry_run trước, xác nhận thống kê hợp lý, rồi mới chạy thật)." >&2
    MISSING=1
fi
EARVN_MISSING=0
for DOMAIN in lr sr_baseline sr_improved; do
    for BACKBONE in mobilenet_v2 mobilenet_v3_small resnet18 efficientnet_b0 ghostnet_100; do
        for SEED in 42 123 2024 44 999; do
            CKPT="runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt"
            if [ ! -f "$CKPT" ]; then
                EARVN_MISSING=$((EARVN_MISSING + 1))
            fi
        done
    done
done
if [ "$EARVN_MISSING" -gt 0 ]; then
    echo "LỖI: thiếu ${EARVN_MISSING}/75 checkpoint recognition n=5 seed của EarVN1.0" >&2
    echo "     (domain lr/sr_baseline/sr_improved x 5 backbone x 5 seed)." >&2
    echo "     -> Bước 4 (transfer learning) đọc trực tiếp các checkpoint này làm" >&2
    echo "        nguồn fine-tune — chạy xong pipeline/run_everything_from_scratch.sh" >&2
    echo "        (EarVN1.0) trước, đủ n=5, rồi mới chạy script này." >&2
    MISSING=1
fi
if [ "$MISSING" -eq 1 ]; then
    echo "DỪNG LẠI — thiếu tiền đề ở trên." >&2
    exit 1
fi
echo ">>> OK — đủ tiền đề (raw_data/awex_raw/ + checkpoint EarVN1.0 n=5)."

# ---------------------------------------------------------------
# [Bước 1] Pipeline chính AWEx from-scratch, n=3 seed (RUNBOOK Bước 1).
# [SỬA — kiểm chứng lại, không đoán] scripts/setup_second_dataset_pipeline.sh
# (được RUN_ALL_NEW_DATASET.sh gọi để tạo configs/config_awex.yaml +
# pipeline_awex/) KHÔNG tự bỏ qua nếu đã tồn tại — nó CHẶN CỨNG (exit 1,
# yêu cầu tự `rm -rf` trước). Nghĩa là: nếu Bước 1 fail GIỮA CHỪNG (1 trong
# 8 bước chính con của nó lỗi), script này (do `set -e`) sẽ dừng ngay, và
# CHẠY LẠI TRỰC TIẾP sẽ tự thất bại ở đúng chốt chặn đó (configs/config_awex.yaml
# + pipeline_awex/ đã tồn tại từ lần trước) — KHÔNG có resume tự động ở mức
# Bước 1. Muốn chạy lại Bước 1 từ đầu: dùng --wipe-existing (xoá sạch, an
# toàn, không đụng EarVN1.0) rồi chạy lại toàn bộ script này.
# ---------------------------------------------------------------
run_step "Buoc 1 - Pipeline chinh AWEx (from-scratch, n=3)" \
    bash RUN_ALL_NEW_DATASET.sh awex 336

# ---------------------------------------------------------------
# [Bước 2] Mở rộng n=3 -> n=5 (RUNBOOK Bước 2).
# ---------------------------------------------------------------
run_step "Buoc 2 - Mo rong n=3 sang n=5" \
    bash pipeline_awex/run_multi_seed_extra_seeds.sh

# ---------------------------------------------------------------
# [Bước 3] Đo phương sai do SR-seed (RUNBOOK Bước 3) — toggle-able.
# ---------------------------------------------------------------
if [ "$RUN_SR_SEED_VARIANCE" = true ]; then
    run_step "Buoc 3 - SR-seed variance (AWEx)" \
        bash pipeline/run_sr_seed_variance.sh "$CONFIG_AWEX"
else
    echo "(RUN_SR_SEED_VARIANCE=false — bỏ qua Bước 3)" | tee -a "$MAIN_LOG"
fi

# ---------------------------------------------------------------
# [Bước 4] Transfer learning từ EarVN1.0 — KẾT QUẢ CHÍNH của AWEx
# (RUNBOOK Bước 4, xem giải thích đầy đủ trong RUNBOOK_AWEx.md vì sao đây là
# số liệu chính, không phải from-scratch ở Bước 1).
# ---------------------------------------------------------------
run_step "Buoc 4 - Transfer learning tu EarVN1.0 (KET QUA CHINH)" \
    bash scripts/run_transfer_learning.sh awex earvn1

echo "" | tee -a "$MAIN_LOG"
echo "!!! DỪNG TỰ ĐỘNG trước Bước 5 — xem ${RESULTS_DIR}/multi_seed_transfer/" | tee -a "$MAIN_LOG"
echo "    multi_seed_summary_transfer_pairwise.csv, kiểm tra std_identity_accuracy" | tee -a "$MAIN_LOG"
echo "    (độ lệch chuẩn giữa seed) so với chênh lệch trung bình — CHỈ nếu phương" | tee -a "$MAIN_LOG"
echo "    sai vẫn LỚN (p_bonferroni gần như luôn >=0.10 do std quá cao) mới cần" | tee -a "$MAIN_LOG"
echo "    chạy tiếp Bước 5 (freeze-backbone, quyết định con người, xem RUNBOOK_AWEx.md" | tee -a "$MAIN_LOG"
echo "    mục Bước 5):" | tee -a "$MAIN_LOG"
echo "        bash scripts/run_transfer_learning_frozen.sh awex earvn1" | tee -a "$MAIN_LOG"

if [ "$RUN_FREEZE_BACKBONE_SCREEN" = true ]; then
    run_step "Buoc 5 - Freeze-backbone (giam phuong sai)" \
        bash scripts/run_transfer_learning_frozen.sh awex earvn1
else
    echo "(RUN_FREEZE_BACKBONE_SCREEN=false — bỏ qua Bước 5, xem hướng dẫn ở trên)" | tee -a "$MAIN_LOG"
fi

# ---------------------------------------------------------------
# [Kiểm tra sạch dữ liệu + tổng hợp báo cáo cuối cùng] — tái sử dụng
# NGUYÊN VẸN 3 công cụ đã dùng cho EarVN1.0 (đều đã tham số hoá --root/
# --results_dir, không có logic riêng cho dataset nào).
# ---------------------------------------------------------------
run_step "Kiem tra nhan trung lap truoc khi tong hop (AWEx)" \
    python data/check_duplicate_labels.py --root "$RESULTS_DIR"
run_step "Kiem tra NaN/Inf that truoc khi tong hop (AWEx)" \
    python data/check_nan_in_results.py --root "$RESULTS_DIR" runs_awex

run_step "Tong hop REPORT.md cuoi cung (AWEx)" \
    python data/generate_final_report.py --config "$CONFIG_AWEX" --results_dir "$RESULTS_DIR" \
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
echo "  2. ${MAIN_LOG}   <- lịch sử đầy đủ"
echo "  3. ${LOG_DIR}/*.log   <- log chi tiết từng bước"
