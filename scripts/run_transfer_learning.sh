#!/bin/bash
# [BẢN 2 — SỬA ĐỂ CÔNG BẰNG GIỮA span_tiny VÀ span_baseline]
# Chạy TOÀN BỘ thí nghiệm TRANSFER LEARNING: khởi tạo từ checkpoint đã train
# trên dataset NGUỒN (mặc định EarVN1.0) rồi fine-tune sang dataset ĐÍCH
# (ví dụ AWE) — bổ sung/so sánh với protocol "train from scratch" đã chạy
# trước đó.
#
# THAY ĐỔI SO VỚI BẢN 1 (lý do: bản 1 chỉ fine-tune span_tiny, còn span_baseline
# vẫn train from-scratch trên dataset đích -> so sánh KHÔNG công bằng, vì
# span_tiny được lợi từ dữ liệu nguồn còn span_baseline thì không):
#   - Fine-tune CẢ span_baseline (train_sr.py, không phải train_sr_distill.py
#     — 2 script train khác nhau, đã đọc lại source thật trước khi vá) từ
#     checkpoint span_baseline của dataset NGUỒN, y hệt cách làm cho span_tiny.
#   - Sau khi có CẢ 2 checkpoint SR đã fine-tune, SINH LẠI ảnh SR cho cả 2
#     domain (sr_baseline_transfer, sr_improved_transfer) từ CHÍNH 2 checkpoint
#     đó — KHÔNG tái dùng ảnh sr_baseline/sr_improved cũ (vốn sinh ra từ model
#     train from-scratch, sẽ làm bước nhận diện sau đó vẫn không công bằng dù
#     bản thân recognition model đã được transfer).
#   - Recognition fine-tune giờ train trên ẢNH MỚI này (domain
#     sr_baseline_transfer / sr_improved_transfer), không phải ảnh cũ.
#
# 6 bước:
#   1. Zero-shot SR (cả tiny lẫn baseline) — áp thẳng checkpoint nguồn, không train.
#   2. Fine-tune SR tiny (train_sr_distill.py --init_ckpt).
#   3. Fine-tune SR baseline (train_sr.py --pretrained_path + --run_suffix MỚI
#      thêm — BẮT BUỘC có run_suffix để không đè checkpoint sr_baseline gốc
#      đang dùng làm teacher cho span_tiny).
#   4. Eval chất lượng SR cho cả 2 (zero-shot + fine-tuned).
#   5. Sinh lại ảnh sr_baseline_transfer / sr_improved_transfer từ 2 checkpoint
#      VỪA fine-tune (data/build_sr.py).
#   6. Fine-tune nhận diện trên ảnh MỚI này (3 domain: lr, sr_baseline_transfer,
#      sr_improved_transfer x 5 backbone x 3 seed), rồi tổng hợp.
#
# QUY ƯỚC ĐẶT TÊN:
#   - checkpoint SR   : runs_<đích>/sr_improved_span_tiny_finetuned_from_<nguồn>/best.pt
#                       runs_<đích>/sr_span_official_finetuned_from_<nguồn>/best.pt
#   - ảnh SR mới      : splits_<đích>/sr_improved_transfer/, splits_<đích>/sr_baseline_transfer/
#                       (RIÊNG, không đè splits_<đích>/sr_improved|sr_baseline/ cũ)
#   - checkpoint nhận diện: runs_<đích>/recognition_<domain_moi>_<backbone>_finetuned_from_<nguồn>_seed<seed>/best.pt
#   - kết quả SR      : results_<đích>/sr_quality_transfer.csv (4 dòng: tiny x{zeroshot,finetuned}, baseline x{zeroshot,finetuned})
#   - kết quả nhận diện: results_<đích>/multi_seed_transfer/multi_seed_summary_transfer.csv
#     (domain trong file này sẽ là lr / sr_baseline_transfer / sr_improved_transfer)
#
# LƯU Ý PHẠM VI: không làm domain hr (mốc tham chiếu, không phải trọng tâm).
#
# Dùng:
#   bash scripts/run_transfer_learning.sh <ten_dataset_dich> <ten_de_dat_cho_dataset_nguon>
# Ví dụ:
#   bash scripts/run_transfer_learning.sh awe earvn1
#
# YÊU CẦU TRƯỚC KHI CHẠY (dataset NGUỒN = configs/config.yaml, cố định):
#   runs/sr_improved_span_tiny/best.pt
#   runs/sr_span_official/best.pt
#   runs/recognition_<domain>_<backbone>_seed<seed>/best.pt
#     (domain = lr/sr_baseline/sr_improved, backbone = 5 cái, seed = 42/123/2024/44/999 — [SỬA] cần ĐỦ 5 seed ở dataset
#     nguồn nữa, tức là pipeline/run_multi_seed.sh + pipeline/run_multi_seed_extra_seeds.sh
#     của EarVN1.0 đã chạy xong cả 2)
#   Dataset ĐÍCH đã chạy xong pipeline from-scratch (RUN_ALL_NEW_DATASET.sh)
#   ít nhất tới bước có runs_<đích>/sr_span_official/best.pt và
#   runs_<đích>/recognition_hr_mobilenet_v2/best.pt

set -e

TARGET="$1"
SOURCE_LABEL="$2"

if [ -z "$TARGET" ] || [ -z "$SOURCE_LABEL" ]; then
    echo "Dùng: bash scripts/run_transfer_learning.sh <ten_dataset_dich> <ten_de_dat_cho_dataset_nguon>"
    echo "Ví dụ: bash scripts/run_transfer_learning.sh awe earvn1"
    exit 1
fi

SRC_CONFIG="configs/config.yaml"
TGT_CONFIG="configs/config_${TARGET}.yaml"
TGT_FT_CONFIG="configs/config_${TARGET}_finetune.yaml"
RESULTS_DIR="results_${TARGET}"
RUNS_DIR="runs_${TARGET}"
SPLITS_DIR="splits_${TARGET}"
SUFFIX="_finetuned_from_${SOURCE_LABEL}"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(42 123 2024 44 999)

echo "################################################################"
echo "# KIỂM TRA TIỀN ĐỀ"
echo "################################################################"
for f in "$SRC_CONFIG" "$TGT_CONFIG"; do
    if [ ! -f "$f" ]; then echo "LỖI: thiếu $f"; exit 1; fi
done
if [ ! -f "runs/sr_improved_span_tiny/best.pt" ]; then
    echo "LỖI: thiếu checkpoint nguồn runs/sr_improved_span_tiny/best.pt"
    exit 1
fi
if [ ! -f "runs/sr_span_official/best.pt" ]; then
    echo "LỖI: thiếu checkpoint nguồn runs/sr_span_official/best.pt (span_baseline của nguồn)"
    exit 1
fi
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DOMAIN in lr sr_baseline sr_improved; do
            CKPT="runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt"
            if [ ! -f "$CKPT" ]; then
                echo "LỖI: thiếu checkpoint nguồn $CKPT"
                echo "     -> chạy xong pipeline/run_multi_seed.sh cho dataset nguồn trước."
                exit 1
            fi
        done
    done
done
if [ ! -f "${RUNS_DIR}/sr_span_official/best.pt" ]; then
    echo "LỖI: thiếu ${RUNS_DIR}/sr_span_official/best.pt (span_baseline from-scratch của đích)"
    exit 1
fi
# [SỬA — bổ sung sau code review, vòng 8, điểm 3] TRƯỚC ĐÂY luôn bắt buộc
# ${RUNS_DIR}/recognition_hr_mobilenet_v2/best.pt với comment "dù
# lambda_identity=0.0 vẫn bắt buộc load được" — comment này SAI/CŨ so với code
# hiện tại: train_sr_distill.py::build_judges() (đọc "sr_improve.identity_judges",
# fallback "frozen_recognition_ckpt" nếu rỗng) CHỈ được gọi khi
# `needs_judges = lambda_identity>0 or lambda_saliency>0` — nếu $TGT_CONFIG có
# cả 2 giá trị này =0 (mặc định kế thừa từ config.yaml gốc) thì KHÔNG cần
# checkpoint judge nào cả. Ngược lại, nếu $TGT_CONFIG bật identity trên dataset
# đích, Python đòi ĐỦ danh sách "identity_judges" (thường 3 backbone), trong
# khi check cũ chỉ xác nhận ĐÚNG 1 file (mobilenet_v2) — sẽ lọt qua đây rồi mới
# chết ngay lúc BƯỚC 2/6 bắt đầu (train_sr_distill.py::build_judges(), TRƯỚC
# vòng lặp epoch — không phải giữa epoch, nhưng vẫn phí thời gian tới đây).
# Đọc ĐÚNG "identity_judges"/"frozen_recognition_ckpt" từ CHÍNH $TGT_CONFIG
# (không hardcode tên backbone) để check khớp với những gì Python thực sự cần.
NEEDS_JUDGES=$(python -c "
import yaml
ci = yaml.safe_load(open('$TGT_CONFIG'))['sr_improve']
li = ci.get('lambda_identity', 0.0)
ls = ci.get('lambda_saliency', 0.0)
print('1' if (float(li) > 0 or float(ls) > 0) else '0')
")
if [ "$NEEDS_JUDGES" = "1" ]; then
    JUDGE_CKPTS=$(python -c "
import yaml
ci = yaml.safe_load(open('$TGT_CONFIG'))['sr_improve']
judges = ci.get('identity_judges') or [{'backbone': 'mobilenet_v2', 'ckpt': ci['frozen_recognition_ckpt']}]
for j in judges:
    print(j['ckpt'])
")
    MISSING_JUDGE=0
    while IFS= read -r CK; do
        if [ -n "$CK" ] && [ ! -f "$CK" ]; then
            echo "LỖI: thiếu checkpoint judge $CK (config đích $TGT_CONFIG có" >&2
            echo "     lambda_identity>0 hoặc lambda_saliency>0 nên train_sr_distill.py cần" >&2
            echo "     checkpoint này — xem sr_improve.identity_judges trong $TGT_CONFIG)." >&2
            MISSING_JUDGE=1
        fi
    done <<< "$JUDGE_CKPTS"
    if [ "$MISSING_JUDGE" -eq 1 ]; then
        echo "-> chạy pipeline/03_train_baseline_recognition.sh (domain hr) cho dataset đích" >&2
        echo "   để tạo đủ checkpoint judge trước khi chạy BƯỚC 2/6 (fine-tune SR tiny)." >&2
        exit 1
    fi
    echo "OK: đủ checkpoint judge cho identity/saliency loss (config đích có bật)."
else
    echo "OK: config đích lambda_identity=lambda_saliency=0 — không cần checkpoint judge nào."
fi
if [ ! -f "${SPLITS_DIR}/splits.json" ]; then
    echo "LỖI: thiếu ${SPLITS_DIR}/splits.json — dữ liệu dataset đích chưa được chuẩn bị."
    exit 1
fi
if [ ! -d "${SPLITS_DIR}/lr" ]; then
    echo "LỖI: thiếu thư mục ${SPLITS_DIR}/lr — cần cho bước sinh lại ảnh SR (data/build_sr.py)."
    exit 1
fi
echo "OK: đủ tiền đề, bắt đầu chạy."

mkdir -p "$RESULTS_DIR" "${RESULTS_DIR}/multi_seed_transfer"

echo ""
echo "################################################################"
echo "# BƯỚC 0/6 — Tạo config fine-tune: $TGT_FT_CONFIG"
echo "################################################################"
python scripts/make_finetune_config.py --in_config "$TGT_CONFIG" --out_config "$TGT_FT_CONFIG"

echo ""
echo "################################################################"
echo "# BƯỚC 1/6 — Zero-shot SR (tiny + baseline), nguồn: $SOURCE_LABEL -> $TARGET"
echo "################################################################"
python eval_sr_quality.py --config "$TGT_CONFIG" --arch span_tiny \
    --ckpt "runs/sr_improved_span_tiny/best.pt" \
    --label "span_tiny_zeroshot_from_${SOURCE_LABEL}" \
    --out_csv "${RESULTS_DIR}/sr_quality_transfer.csv"

python eval_sr_quality.py --config "$TGT_CONFIG" --arch span_official \
    --ckpt "runs/sr_span_official/best.pt" \
    --label "span_baseline_zeroshot_from_${SOURCE_LABEL}" \
    --out_csv "${RESULTS_DIR}/sr_quality_transfer.csv"

echo ""
echo "################################################################"
echo "# BƯỚC 2/6 — Fine-tune SR tiny trên $TARGET"
echo "################################################################"
# [SỬA — bổ sung sau code review, vòng 4, điểm 1] CỐ Ý KHÔNG pin cứng lambda
# feat/saliency/identity — script này fine-tune tiếp từ checkpoint span_tiny
# nguồn, nên PHẢI dùng CÙNG recipe (đọc từ $TGT_FT_CONFIG, vốn được sinh ra
# từ $TGT_CONFIG qua make_finetune_config.py, xem BƯỚC 0) như lúc train
# span_tiny ban đầu — chỉ IN RÕ giá trị lambda thật sự dùng để tránh đổi
# recipe âm thầm giữa các lần chạy nếu default config.yaml đổi sau này.
python -c "
import yaml
ci = yaml.safe_load(open('$TGT_FT_CONFIG'))['sr_improve']
print('>>> Lambda THẬT SỰ dùng để fine-tune span_tiny (đọc từ $TGT_FT_CONFIG, sr_improve.*):')
for k in ['lambda_pixel', 'lambda_distill', 'lambda_feat', 'lambda_saliency', 'lambda_identity']:
    print(f'    {k} = {ci.get(k, 0.0)}')
"
python train_sr_distill.py --config "$TGT_FT_CONFIG" \
    --init_ckpt "runs/sr_improved_span_tiny/best.pt" \
    --run_suffix "$SUFFIX"

echo ""
echo "################################################################"
echo "# BƯỚC 3/6 — Fine-tune SR baseline trên $TARGET (CÔNG BẰNG với tiny)"
echo "################################################################"
python train_sr.py --config "$TGT_FT_CONFIG" --sr_arch span_official \
    --pretrained_path "runs/sr_span_official/best.pt" \
    --run_suffix "$SUFFIX"

echo ""
echo "################################################################"
echo "# BƯỚC 4/6 — Eval chất lượng SR đã fine-tune (cả 2)"
echo "################################################################"
python eval_sr_quality.py --config "$TGT_CONFIG" --arch span_tiny \
    --ckpt "${RUNS_DIR}/sr_improved_span_tiny${SUFFIX}/best.pt" \
    --label "span_tiny_finetuned_from_${SOURCE_LABEL}" \
    --out_csv "${RESULTS_DIR}/sr_quality_transfer.csv"

python eval_sr_quality.py --config "$TGT_CONFIG" --arch span_official \
    --ckpt "${RUNS_DIR}/sr_span_official${SUFFIX}/best.pt" \
    --label "span_baseline_finetuned_from_${SOURCE_LABEL}" \
    --out_csv "${RESULTS_DIR}/sr_quality_transfer.csv"

echo ""
echo "################################################################"
echo "# BƯỚC 5/6 — Sinh lại ảnh SR (train/val/test) từ 2 checkpoint VỪA fine-tune"
echo "################################################################"
python data/build_sr.py --lr_dir "${SPLITS_DIR}/lr" \
    --sr_ckpt "${RUNS_DIR}/sr_improved_span_tiny${SUFFIX}/best.pt" \
    --arch span_tiny --scale 4 --out_dir "${SPLITS_DIR}/sr_improved_transfer"

python data/build_sr.py --lr_dir "${SPLITS_DIR}/lr" \
    --sr_ckpt "${RUNS_DIR}/sr_span_official${SUFFIX}/best.pt" \
    --arch span_official --scale 4 --out_dir "${SPLITS_DIR}/sr_baseline_transfer"

echo ""
echo "################################################################"
echo "# BƯỚC 6/6 — Fine-tune nhận diện trên ảnH MỚI: 3 domain x 5 backbone x 5 seed"
echo "################################################################"
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        # Cặp (domain đích mới, domain nguồn tương ứng để lấy checkpoint init):
        #   lr                  <- lr                  (ảnh LR không phụ thuộc model SR nào, giữ nguyên)
        #   sr_baseline_transfer <- sr_baseline          (ảnh mới sinh từ SR baseline đã fine-tune)
        #   sr_improved_transfer <- sr_improved          (ảnh mới sinh từ SR tiny đã fine-tune)
        for PAIR in "lr:lr" "sr_baseline_transfer:sr_baseline" "sr_improved_transfer:sr_improved"; do
            TGT_DOMAIN="${PAIR%%:*}"
            SRC_DOMAIN="${PAIR##*:}"
            echo "---- backbone=$BACKBONE seed=$SEED domain_đích=$TGT_DOMAIN (nguồn: $SRC_DOMAIN) ----"
            SRC_CKPT="runs/recognition_${SRC_DOMAIN}_${BACKBONE}_seed${SEED}/best.pt"

            python train_recognition.py --config "$TGT_FT_CONFIG" --domain "$TGT_DOMAIN" --backbone "$BACKBONE" \
                --init_ckpt_transfer "$SRC_CKPT" \
                --seed "$SEED" --run_suffix "${SUFFIX}_seed${SEED}"

            python eval_recognition.py --config "$TGT_CONFIG" \
                --ckpt "${RUNS_DIR}/recognition_${TGT_DOMAIN}_${BACKBONE}${SUFFIX}_seed${SEED}/best.pt" \
                --backbone "$BACKBONE" --train_domain "$TGT_DOMAIN" --test_domain "$TGT_DOMAIN" \
                --out_json "${RESULTS_DIR}/multi_seed_transfer/${TGT_DOMAIN}_${BACKBONE}_seed${SEED}.json"
        done
    done
done

python data/aggregate_multi_seed_results.py --results_dir "${RESULTS_DIR}/multi_seed_transfer" \
    --out_csv "${RESULTS_DIR}/multi_seed_transfer/multi_seed_summary_transfer.csv"

echo ""
echo "################################################################"
echo "HOÀN TẤT transfer learning CÔNG BẰNG cho '$TARGET' (nguồn: '$SOURCE_LABEL')."
echo "################################################################"
echo "File kết quả cần gửi lại để tổng hợp:"
echo "  ${RESULTS_DIR}/sr_quality_transfer.csv (giờ có 4 dòng: tiny/baseline x zero-shot/fine-tuned)"
echo "  ${RESULTS_DIR}/multi_seed_transfer/multi_seed_summary_transfer.csv (domain: lr, sr_baseline_transfer, sr_improved_transfer)"
echo ""
echo "So sánh công bằng: span_tiny (domain sr_improved_transfer) vs span_baseline"
echo "(domain sr_baseline_transfer) — CẢ HAI giờ đều được fine-tune từ dataset nguồn"
echo "như nhau, khác biệt còn lại chỉ đến từ kiến trúc, không còn lệch do 1 bên có"
echo "transfer learning còn bên kia thì không."
