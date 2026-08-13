#!/bin/bash
# CHẠY LẠI TOÀN BỘ SAU KHI SỬA LỖI CÔNG THỨC SPAB (thứ tự residual/attention).
# Script này chạy TẤT CẢ thí nghiệm liên quan đến span_tiny, không chừa lại
# thí nghiệm nào — vì cả 4 nhóm dưới đây đều dùng chung models/sr_models.py
# nên đều bị ảnh hưởng bởi lỗi công thức đã phát hiện:
#   1. Pipeline chính (Bước 6-9): train span_tiny, benchmark, xuất ảnh
#   2. Ablation study (4 cấu hình loss)
#   3. Lambda sweep (6 mức x 3 seed)
#   4. Multi-seed (5 backbone x 3 seed x 3 domain)
#
# CẢNH BÁO THỜI GIAN: đây là toàn bộ khối lượng công việc lớn nhất trong dự
# án — tổng cộng có thể lên tới hàng chục giờ tùy tốc độ máy (đặc biệt bước
# 4 - multi-seed - là bước nặng nhất, 45 lần train recognition). Nên chạy
# qua đêm hoặc trên máy có thể chạy liên tục nhiều giờ không bị ngắt.

set -e

echo "################################################################"
echo "# BƯỚC 1/6: BACKUP TOÀN BỘ KẾT QUẢ CŨ (CODE LỖI)               #"
echo "################################################################"
BK="backup_code_loi_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BK"

cp -r runs/sr_improved_span_tiny "$BK/" 2>/dev/null || echo "  (chưa có sr_improved_span_tiny, bỏ qua)"
cp -r runs/recognition_sr_improved_* "$BK/" 2>/dev/null || echo "  (chưa có recognition_sr_improved_*, bỏ qua)"
cp -r runs/recognition_lr_*_seed* "$BK/" 2>/dev/null || echo "  (chưa có multi-seed lr, bỏ qua)"
cp -r runs/recognition_sr_baseline_*_seed* "$BK/" 2>/dev/null || echo "  (chưa có multi-seed sr_baseline, bỏ qua)"
cp -r runs/recognition_sr_improved_*_seed* "$BK/" 2>/dev/null || echo "  (chưa có multi-seed sr_improved, bỏ qua)"
cp -r runs/sr_improved_span_tiny_ablation_* "$BK/" 2>/dev/null || echo "  (chưa có ablation runs, bỏ qua)"
cp -r runs/recognition_sr_ablation_* "$BK/" 2>/dev/null || echo "  (chưa có ablation recognition, bỏ qua)"
cp -r runs/sr_improved_span_tiny_lid* "$BK/" 2>/dev/null || echo "  (chưa có lambda_sweep runs, bỏ qua)"
cp -r runs/recognition_sr_sweep_* "$BK/" 2>/dev/null || echo "  (chưa có lambda_sweep recognition, bỏ qua)"

cp -r results/multi_seed "$BK/" 2>/dev/null || echo "  (chưa có results/multi_seed, bỏ qua)"
cp -r results/lambda_sweep "$BK/" 2>/dev/null || echo "  (chưa có results/lambda_sweep, bỏ qua)"
cp results/ablation.csv "$BK/ablation_CU.csv" 2>/dev/null || echo "  (chưa có ablation.csv, bỏ qua)"
cp results/summary.csv "$BK/summary_CU.csv" 2>/dev/null || echo "  (chưa có summary.csv, bỏ qua)"
cp results/sr_quality.csv "$BK/sr_quality_CU.csv" 2>/dev/null || echo "  (chưa có sr_quality.csv, bỏ qua)"
cp results/real_lr_holdout.csv "$BK/real_lr_holdout_CU.csv" 2>/dev/null || echo "  (chưa có real_lr_holdout.csv, bỏ qua)"
cp -r results/final_report "$BK/" 2>/dev/null || echo "  (chưa có results/final_report, bỏ qua)"
cp models/sr_models.py "$BK/sr_models_LOI.py"

echo ">>> Đã backup vào: $BK/"
echo ""

echo "################################################################"
echo "# BƯỚC 2/6: XÁC MINH FILE sr_models.py ĐÃ ĐƯỢC SỬA ĐÚNG        #"
echo "################################################################"
if ! grep -q "u = x + h" models/sr_models.py; then
    echo "!!! LỖI: models/sr_models.py CHƯA được thay bằng bản đã sửa."
    echo "!!! Copy file sr_models.py đã tải về vào models/sr_models.py trước, rồi chạy lại script này."
    exit 1
fi
if grep -q "identity + out" models/sr_models.py; then
    echo "!!! LỖI: models/sr_models.py vẫn còn dấu vết công thức CŨ (sai)."
    echo "!!! Copy đè file sr_models.py đã sửa vào models/sr_models.py trước, rồi chạy lại script này."
    exit 1
fi
echo ">>> OK: file đã đúng công thức mới."
echo ""

echo "################################################################"
echo "# BƯỚC 3/6: XÓA TOÀN BỘ KẾT QUẢ CŨ LIÊN QUAN ĐẾN span_tiny     #"
echo "################################################################"
rm -rf runs/sr_improved_span_tiny
rm -rf runs/recognition_sr_improved_*
rm -rf runs/recognition_lr_*_seed* runs/recognition_sr_baseline_*_seed* runs/recognition_sr_improved_*_seed*
rm -rf runs/sr_improved_span_tiny_ablation_* runs/recognition_sr_ablation_*
rm -rf runs/sr_improved_span_tiny_lid* runs/recognition_sr_sweep_*
rm -rf splits/sr_improved
rm -rf results/multi_seed results/lambda_sweep
rm -f results/ablation.csv results/summary.csv results/sr_quality.csv results/real_lr_holdout.csv
rm -rf results/final_report
echo ">>> Đã xóa xong, sẵn sàng chạy lại từ đầu."
echo ""

echo "################################################################"
echo "# BƯỚC 4/6: PIPELINE CHÍNH (Bước 6-9: train + benchmark + ảnh) #"
echo "################################################################"
bash pipeline/run_full_pipeline_and_report.sh

echo ""
echo "################################################################"
echo "# BƯỚC 5/6: ABLATION STUDY (4 cấu hình loss)                   #"
echo "################################################################"
bash pipeline/run_ablation.sh

echo ""
echo "################################################################"
echo "# BƯỚC 6a/6: LAMBDA SWEEP (6 mức x 3 seed)                     #"
echo "################################################################"
bash pipeline/run_lambda_sweep.sh

echo ""
echo "################################################################"
echo "# BƯỚC 6b/6: MULTI-SEED ĐẦY ĐỦ (5 backbone x 3 seed x 3 domain)#"
echo "################################################################"
bash pipeline/run_multi_seed.sh

echo ""
echo "################################################################"
echo "# HOÀN TẤT TOÀN BỘ. Tổng hợp lại báo cáo cuối cùng...          #"
echo "################################################################"
python data/generate_final_report.py --config configs/config.yaml \
    --results_dir results --out_dir results/final_report

echo ""
echo "=== XONG TOÀN BỘ. Các file cần gửi để đối chiếu: ==="
echo "  results/final_report/REPORT.md"
echo "  results/ablation.csv"
echo "  results/lambda_sweep/lambda_sweep_summary.csv"
echo "  results/multi_seed/multi_seed_summary.csv"
echo "  (Bản cũ đã backup an toàn tại: $BK/)"
