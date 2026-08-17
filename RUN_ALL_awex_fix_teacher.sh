#!/bin/bash
# ================================================================
# KHÔI PHỤC AWEx sau lỗi teacher (span_official) bị NaN loss.
# ================================================================
# Bối cảnh: runs_awex/sr_span_official/best.pt (checkpoint span_official
# from-scratch trên AWEx) bị NaN loss trong training (đã xác nhận qua log
# thật, tái lập 2 lần). Checkpoint này là TEACHER cho: span_tiny, span_large,
# VÀ toàn bộ Track B distilled (rlfn/rlfn_adapted/ecbsr/safmn/smfanet bản
# "sr_improved_<arch>") — teacher_ckpt trỏ tới nó, lambda_distill=1.0 trong
# config_awex.yaml. Mọi model học từ teacher này đều cần train lại.
#
# ĐÃ VÁ: train_sr.py (tắt AMP riêng cho span_official — nguyên nhân NaN,
# xem CHANGELOG). File train_sr.py trong project PHẢI đã là bản vá này
# (kiểm tra: grep -c "use_amp" train_sr.py phải > 0) TRƯỚC khi chạy script này.
#
# KHÔNG cần đụng: domain "lr", Track A (sr_<arch> không "improved", pixel-loss
# thuần không qua teacher), toàn bộ track transfer learning (*_transfer,
# *_transfer_frozen — khởi tạo từ checkpoint SẠCH của EarVN1.0, không liên
# quan checkpoint hỏng của AWEx).
#
# CHẠY TỪ THƯ MỤC GỐC PROJECT. Tốn NHIỀU giờ — nên chạy qua đêm / tmux / nohup.

set -e

CONFIG="configs/config_awex.yaml"

echo "################################################################"
echo "# BƯỚC 0 — Kiểm tra tiền đề"
echo "################################################################"
if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không thấy $CONFIG"; exit 1
fi
if ! grep -q "use_amp" train_sr.py 2>/dev/null; then
    echo "LỖI: train_sr.py CHƯA phải bản đã vá (thiếu 'use_amp')."
    echo "     -> copy đè file train_sr.py đã vá vào đây trước khi chạy script này."
    exit 1
fi
echo "OK."

echo ""
echo "################################################################"
echo "# BƯỚC 1 — Dọn sạch mọi artifact phụ thuộc teacher hỏng"
echo "################################################################"
rm -rf runs_awex/sr_span_official
rm -rf runs_awex/sr_improved_span_tiny
rm -rf runs_awex/sr_span_large
for ARCH in rlfn rlfn_adapted ecbsr safmn smfanet; do
    rm -rf "runs_awex/sr_improved_${ARCH}"
done
rm -rf splits_awex/sr_baseline
rm -rf splits_awex/sr_improved
rm -rf splits_awex/sr_span_large
for ARCH in rlfn rlfn_adapted ecbsr safmn smfanet; do
    rm -rf "splits_awex/sr_improved_${ARCH}"
done
rm -rf runs_awex/recognition_sr_baseline_*
rm -rf runs_awex/recognition_sr_improved_*
rm -rf runs_awex/recognition_sr_span_large_*
rm -rf results_awex/multi_seed
rm -rf results_awex/span_large_ablation
# sr_quality.csv có dòng cũ của các arch trên (bị hỏng) — xoá để lần chạy
# lại ghi sạch, không lẫn số liệu cũ/mới cùng 1 nhãn.
mv -f results_awex/sr_quality.csv results_awex/sr_quality.csv.bak_broken 2>/dev/null || true
echo "Đã dọn xong. Backup sr_quality.csv cũ (nếu có) -> sr_quality.csv.bak_broken"

echo ""
echo "################################################################"
echo "# BƯỚC 2 — Train lại span_official (teacher, bản đã vá NaN)"
echo "################################################################"
python train_sr.py --config "$CONFIG" --sr_arch span_official \
    --pretrained_path checkpoints/span_pretrained_x4.pth
echo ">>> Kiểm tra log — KHÔNG được còn 'train_L1=nan':"
tail -10 runs_awex/sr_span_official/train.log
if grep -q "train_L1=nan" runs_awex/sr_span_official/train.log; then
    echo "LỖI: vẫn còn NaN sau khi vá — DỪNG LẠI, báo lại log này trước khi tiếp tục."
    exit 1
fi

echo ""
echo "################################################################"
echo "# BƯỚC 3 — Sinh lại ảnh sr_baseline từ teacher sạch"
echo "################################################################"
python data/build_sr.py --lr_dir splits_awex/lr --sr_ckpt runs_awex/sr_span_official/best.pt \
    --arch span_official --scale 4 --out_dir splits_awex/sr_baseline
python eval_sr_quality.py --config "$CONFIG" --arch span_official \
    --ckpt runs_awex/sr_span_official/best.pt --label span_baseline \
    --out_csv results_awex/sr_quality.csv

echo ""
echo "################################################################"
echo "# BƯỚC 4 — Train lại span_tiny (KD từ teacher sạch) + sinh ảnh sr_improved"
echo "################################################################"
bash pipeline_awex/06_improve_span.sh

echo ""
echo "################################################################"
echo "# BƯỚC 5 — Recognition multi-seed cho lr/sr_baseline/sr_improved (n=3 -> n=5)"
echo "################################################################"
bash pipeline_awex/run_multi_seed.sh
bash pipeline_awex/run_multi_seed_extra_seeds.sh

echo ""
echo "################################################################"
echo "# BƯỚC 6 — Train lại span_large (KD từ teacher sạch), n=3 -> n=5"
echo "################################################################"
bash RUN_ALL_span_large_ablation.sh "$CONFIG"
bash RUN_ALL_span_large_ablation_extra_seeds.sh "$CONFIG"

echo ""
echo "################################################################"
echo "# BƯỚC 7 — Train lại Track B distilled (rlfn/rlfn_adapted/ecbsr/safmn/smfanet), n=3 -> n=5"
echo "################################################################"
for ARCH in rlfn rlfn_adapted ecbsr safmn smfanet; do
    echo ">>> Track B: $ARCH"
    bash RUN_ALL_extra_sr_baseline_distilled.sh "$ARCH" "$CONFIG"
    bash RUN_ALL_extra_sr_baseline_distilled_extra_seeds.sh "$ARCH" "$CONFIG"
done

echo ""
echo "################################################################"
echo "HOÀN TẤT khôi phục AWEx sau lỗi teacher."
echo "################################################################"
echo "File cần gửi lại để đánh giá:"
echo "  results_awex/sr_quality.csv"
echo "  results_awex/multi_seed/multi_seed_summary.csv + _pairwise.csv"
echo "  results_awex/span_large_ablation/multi_seed_summary_4domains.csv + _pairwise.csv"
echo "  (các file *_distilled*/*extra_sr_baseline* liên quan Track B nếu có, theo tên script sinh ra)"
