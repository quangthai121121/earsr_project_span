#!/bin/bash
# [MỚI — bổ sung sau đánh giá kết quả thật, đợt 9] Chạy THÊM 2 seed (44, 999)
# cho domain "sr_improved_kdv2" (recipe multi-judge + feature-KD), để cộng
# với 3 seed đã có (42, 123, 2024 — từ pipeline/run_multi_seed_kdv2.sh) thành
# ĐỦ 5 SEED, giống hệt lý do đã áp dụng cho 3 domain chính ở
# pipeline/run_multi_seed_extra_seeds.sh.
#
# LÝ DO CẦN: kết quả multi-seed_kdv2 (n=3) hiện tại cho hiệu ứng NHỎ và
# KHÔNG NHẤT QUÁN giữa 5 backbone (chỉ 1/5 backbone — ghostnet_100 — đạt ý
# nghĩa thống kê so với sr_improved cũ, ở ngưỡng p_bonferroni<0.10). Với n=3,
# lực kiểm định (statistical power) yếu — không phân biệt được "hiệu ứng thật
# nhỏ" với "chưa đủ mẫu để phát hiện hiệu ứng". Tăng lên n=5 giúp làm rõ hơn.
#
# KHÔNG train lại SR — TÁI SỬ DỤNG checkpoint
# runs/sr_improved_<student_arch>_kdv2/best.pt (tên thư mục thật tuỳ theo
# configs/config.yaml::sr_improve.student_arch, xem cách build ARCH trong
# run_multi_seed_kdv2.sh) và ảnh đã sinh sẵn ở splits/sr_improved_kdv2/ (đúng
# seed cố định SR_SEED=42 như run_multi_seed_kdv2.sh) — CHỈ multi-seed ở bước
# train_recognition.py. Script này chỉ cần kiểm tra splits/sr_improved_kdv2/
# tồn tại (ảnh đã sinh), không cần biết tên checkpoint SR cụ thể.
#
# YÊU CẦU:
#   1. Đã chạy xong pipeline/run_multi_seed_kdv2.sh (có sẵn ảnh SR ở
#      splits/sr_improved_kdv2/ và 3 seed 42/123/2024 trong
#      results/multi_seed_kdv2/).
#   2. Đã chạy xong pipeline/run_multi_seed_extra_seeds.sh (có sẵn checkpoint
#      recognition_lr_<backbone>_seed{44,999}/best.pt cho cả 5 backbone —
#      ĐÚNG chuỗi hr -> lr, TÁI SỬ DỤNG như mọi domain khác, không train lại).
#
# CẢNH BÁO THỜI GIAN: 5 backbone x 2 seed = 10 lần train recognition (nhẹ,
# không train lại SR).

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results/multi_seed_kdv2"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(44 999)
DOMAIN="sr_improved_kdv2"
mkdir -p "$RESULTS_DIR"

echo "Kiểm tra tiền đề..."
MISSING=0
# [SỬA — bổ sung sau code review, đợt 10] "-d" chỉ kiểm tra thư mục TỒN TẠI,
# không kiểm tra CÓ ẢNH thật hay không (thư mục rỗng do build_sr.py lỗi giữa
# đường vẫn qua được check này) — kiểm tra có ít nhất 1 file bên trong (đệ quy
# qua train/val/test).
if [ ! -d "splits/${DOMAIN}" ] || [ -z "$(find "splits/${DOMAIN}" -type f -print -quit 2>/dev/null)" ]; then
    echo "LỖI: splits/${DOMAIN} không tồn tại hoặc rỗng (không có file ảnh nào)." >&2
    echo "     -> chạy pipeline/run_multi_seed_kdv2.sh trước (Bước 1/2, sinh SR)." >&2
    MISSING=1
fi
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: thiếu $CKPT -> chạy pipeline/run_multi_seed_extra_seeds.sh trước." >&2
            MISSING=1
        fi
    done
done
if [ "$MISSING" -eq 1 ]; then
    echo "DỪNG LẠI — thiếu tiền đề ở trên." >&2
    exit 1
fi
echo "OK — đủ tiền đề, bắt đầu chạy (không train lại SR)."

for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "----------------------------------------------------------------"
        echo "backbone=$BACKBONE | seed=$SEED | domain=$DOMAIN"
        echo "----------------------------------------------------------------"
        LR_CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" --backbone "$BACKBONE" \
            --init_ckpt "$LR_CKPT" --seed "$SEED" --run_suffix "_seed${SEED}"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
            --out_json "$RESULTS_DIR/${DOMAIN}_${BACKBONE}_seed${SEED}.json"
    done
done

echo ""
echo ">>> Tổng hợp lại — giờ đọc ĐỦ 5 seed (42,123,2024,44,999) cho domain $DOMAIN..."
echo ">>> (Yêu cầu: $RESULTS_DIR/ đã có sẵn 15 file *_seed{42,123,2024}.json từ"
echo ">>> pipeline/run_multi_seed_kdv2.sh — script này CHỈ thêm 10 file"
echo ">>> *_seed{44,999}.json, không xoá/ghi đè file cũ. Nếu muốn so lại với 'lr'"
echo ">>> và 'sr_improved' (đủ 5 seed), copy JSON tương ứng từ results/multi_seed/"
echo ">>> vào $RESULTS_DIR/ trước khi chạy aggregate.)"
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/multi_seed_kdv2_summary.csv"

echo ""
echo "HOÀN TẤT. Kết quả (giờ dựa trên n=5 seed): $RESULTS_DIR/multi_seed_kdv2_summary.csv"
echo "và $RESULTS_DIR/multi_seed_kdv2_summary_pairwise.csv."
