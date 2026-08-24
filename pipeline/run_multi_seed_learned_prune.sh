#!/bin/bash
# [MỚI] Validate ĐẦY ĐỦ multi-seed x 5-backbone cho học pruning độ sâu
# (learned/differentiable block pruning, xem train_sr_learned_prune.py,
# models/sr_models.py::SPANLearnedPrune). Đây là BƯỚC BẮT BUỘC sau khi
# pipeline/run_prune_sparsity_screen.sh (1 backbone, 1 seed/mức — chỉ tín
# hiệu sàng lọc) đã xác định được lambda_sparsity thắng, trước khi đưa số
# liệu vào bảng kết quả chính thức của bài báo.
#
# **SỬA THAM SỐ NÀY** trước khi chạy, khớp đúng mức lambda_sparsity đã THẮNG
# ở run_prune_sparsity_screen.sh:
#
# [SỬA — bổ sung sau đánh giá kết quả thật, đợt 9] LẦN CHẠY TRƯỚC dùng
# LAMBDA_SPARSITY=0.05 — nhưng đó là giá trị lấy từ screen CŨ (lambda_feat=
# lambda_identity=0), và ở CHÍNH mức 0.05 đó, screen cũ đã cho n_blocks_kept=6
# (KHÔNG prune gì cả, xem results/prune_sparsity_screen/sr_quality_screen.csv
# dòng "prune_lsp0.05_nblocks6"). Nghĩa là lần multi-seed trước thực chất so
# sánh 1 model SPAN ĐẦY ĐỦ 6 khối (không nén) với span_tiny 3 khối — không có
# ý nghĩa gì cho claim "learned pruning". PHẢI chạy lại
# pipeline/run_prune_sparsity_screen.sh (đã pin lambda_feat=0.5/identity=0.1
# ở bản sửa mới) TRƯỚC, đọc kết quả n_blocks_kept ở CÁC mức lambda_sparsity
# mới, rồi điền lại giá trị cho ra khối GẦN 3 NHẤT vào đây.
#
# [SỬA — bổ sung sau code review, đợt 10, điểm 1] TRƯỚC ĐÂY để sẵn "0.05" làm
# giá trị mặc định — CHÍNH giá trị này đã gây ra lỗi đợt 8 (chạy nhầm với
# λ_feat/identity=0 VÀ 0.05 chưa qua screen mới nên vẫn là số "biết trước là
# sai", 6 khối, không nén). Nếu người dùng quên sửa và bấm chạy thẳng, script
# sẽ LẶP LẠI ĐÚNG lỗi cũ, tốn lại hàng giờ GPU. Đổi thành rỗng + fail-fast:
# BẮT BUỘC phải tự điền giá trị (đọc từ Bước 1 —
# pipeline/run_prune_sparsity_screen.sh bản mới) trước khi script chạy được.
# [SỬA — 2026-08-24] Đợt 9 pin lambda_feat=0.5/lambda_identity=0.1 vì tin đó
# là cấu hình THẮNG ở ablation KDv2 — niềm tin đó đã bị BÁC BỎ DỨT ĐIỂM bằng
# multi-seed n=5 x 5 backbone thật (xem giải thích đầy đủ trong
# pipeline/run_prune_sparsity_screen.sh, cùng ngày sửa). Đã đổi
# run_prune_sparsity_screen.sh về recipe SẠCH (feat=0/identity=0) — nghĩa là
# QUAN HỆ lambda_sparsity -> n_blocks_kept của LẦN SÀNG LỌC CŨ (0.2 -> 3 khối)
# KHÔNG còn đảm bảo đúng nữa (recipe khác thì gate học khác, có thể lệch mốc
# giống hệt lần "reconstruction thuần" trước đó: 0.1->5 khối, 0.2->2 khối,
# KHÔNG mốc nào ra đúng 3). BẮT BUỘC chạy lại pipeline/run_prune_sparsity_screen.sh
# (bản đã sửa) TRƯỚC, đọc n_blocks_kept mới ở từng mức, rồi mới điền lại giá
# trị cho ra khối GẦN 3 NHẤT vào đây — để trống + fail-fast, không đoán lại.
LAMBDA_SPARSITY=

# [SỬA — 2026-08-24] Đồng bộ với run_prune_sparsity_screen.sh: bỏ hẳn nhánh
# "identity-aware" (lambda_feat=0.5/identity=0.1, dựa trên cấu hình KDv2 đã bị
# bác bỏ) — dùng recipe SẠCH (đúng span_tiny: pixel+distill thuần), PHẢI khớp
# CHÍNH XÁC với cấu hình đã dùng ở run_prune_sparsity_screen.sh để chọn
# LAMBDA_SPARSITY ở trên (2 bước screen -> validate bắt buộc nhất quán).
LAMBDA_FEAT=0.0
LAMBDA_SALIENCY=0.0
LAMBDA_IDENTITY=0.0

# SR model (pruning) CHỈ train 1 LẦN ở seed cố định — ĐÚNG protocol thống kê
# đã dùng xuyên suốt project (train SR nhiều seed bị đánh giá không khả thi
# về chi phí, chỉ downstream recognition được lặp lại qua seed để đo phương sai).
SR_SEED=42

CONFIG="configs/config.yaml"
RESULTS_DIR="results/multi_seed_learned_prune"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(42 123 2024)   # đổi/thêm seed nếu muốn khớp n=5 như bản thảo bài báo (thêm 44, 999)
DOMAIN="sr_learned_prune"

# YÊU CẦU: đã chạy xong pipeline/run_multi_seed.sh trước đó — script này TÁI
# SỬ DỤNG checkpoint recognition_lr_<backbone>_seed<seed> đã train sẵn cho
# từng seed, ĐÚNG chuỗi fine-tune hr -> lr -> sr_learned_prune (không dùng
# chung 1 checkpoint hr cho mọi seed).

set -e

# [MỚI — bổ sung sau code review, đợt 10, điểm 1] Fail-fast nếu quên điền
# LAMBDA_SPARSITY (xem giải thích ở khai báo biến phía trên) — dừng NGAY từ
# đầu, trước khi tốn bất kỳ giờ GPU nào để train SR.
if [ -z "$LAMBDA_SPARSITY" ]; then
    echo "LỖI: LAMBDA_SPARSITY chưa được điền (đang rỗng)." >&2
    echo "     -> chạy 'bash pipeline/run_prune_sparsity_screen.sh' (bản mới, recipe" >&2
    echo "        SẠCH feat=0/identity=0) TRƯỚC, đọc n_blocks_kept ở" >&2
    echo "        mỗi mức lambda_sparsity trong results/prune_sparsity_screen/" >&2
    echo "        rồi sửa dòng 'LAMBDA_SPARSITY=' đầu file này thành giá trị cho ra" >&2
    echo "        số khối GẦN 3 NHẤT, sau đó chạy lại." >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# [MỚI — bổ sung sau code review, đợt 10, điểm 3] Nếu đã có JSON seed 44/999
# từ 1 lần chạy pipeline/run_multi_seed_learned_prune_extra_seeds.sh TRƯỚC
# (rất có thể trên domain "sr_learned_prune" CŨ, ứng với model 6-khối không
# identity-aware, đã bị coi là invalid) — CẢNH BÁO rõ vì Bước 2/2 dưới đây
# CHỈ ghi lại seed 42/123/2024, không tự xoá/ghi đè 2 file seed 44/999 cũ đó.
# Nếu không dọn, bước tổng hợp cuối sẽ TRỘN 3 seed mới (model đã sửa) với 2
# seed cũ (model 6-khối invalid) mà không có dấu hiệu cảnh báo nào khi đọc
# lại CSV.
STALE_EXTRA_SEED=0
for SEED in 44 999; do
    f="${RESULTS_DIR}/${DOMAIN}_mobilenet_v2_seed${SEED}.json"
    if [ -f "$f" ]; then
        STALE_EXTRA_SEED=1
    fi
done
if [ "$STALE_EXTRA_SEED" -eq 1 ]; then
    echo "CẢNH BÁO: đã thấy JSON seed 44/999 trong $RESULTS_DIR (khả năng từ 1 lần" >&2
    echo "  chạy run_multi_seed_learned_prune_extra_seeds.sh TRƯỚC — RẤT CÓ THỂ ứng" >&2
    echo "  với model learned-pruning CŨ, không phải model sắp train lại ở đây)." >&2
    echo "  Xoá các file '${DOMAIN}_*_seed44.json' / '${DOMAIN}_*_seed999.json' trong" >&2
    echo "  $RESULTS_DIR rồi chạy lại pipeline/run_multi_seed_learned_prune_extra_seeds.sh" >&2
    echo "  SAU KHI script này (Bước 2/2, seed 42/123/2024) chạy xong với model mới," >&2
    echo "  để tránh trộn 2 model khác nhau vào cùng bảng tổng hợp." >&2
    echo "  (Không tự xoá thay bạn — kiểm tra thủ công để chắc chắn trước khi mất dữ liệu.)" >&2
fi

# [SỬA — lỗi phát hiện qua code review] đọc scale từ config thay vì hardcode 4.
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

# [SỬA — bổ sung sau code review, vòng 6] Xem giải thích đầy đủ trong
# pipeline/run_multi_seed_kdv2.sh — kiểm tra ĐỦ CẢ 15 checkpoint
# recognition_lr_<backbone>_seed<seed> NGAY TỪ ĐẦU, TRƯỚC Bước 1/2 (train SR
# learned-pruning, tốn hàng giờ), thay vì để phát hiện thiếu ở Bước 2/2 sau
# khi đã lãng phí toàn bộ thời gian train SR.
echo "Kiểm tra tiền đề: ${#BACKBONES[@]} backbone x ${#SEEDS[@]} seed = "\
"$((${#BACKBONES[@]} * ${#SEEDS[@]})) checkpoint recognition_lr_*_seed*..."
MISSING=0
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: thiếu $CKPT" >&2
            MISSING=1
        fi
    done
done
if [ "$MISSING" -eq 1 ]; then
    echo "-> chạy pipeline/run_multi_seed.sh (đủ 5 backbone x 3 seed, domain lr) trước." >&2
    exit 1
fi
echo "OK — đủ tiền đề recognition_lr_*_seed*, bắt đầu chạy."

if python -c "import sys; sys.exit(0 if float('$LAMBDA_SALIENCY') > 0 or float('$LAMBDA_IDENTITY') > 0 else 1)"; then
    source "$(dirname "$0")/_check_hr_judges.sh"
    check_hr_judges
fi

echo "################################################################"
echo "# Bước 1/2: Train SR (learned pruning) — 1 lần, seed cố định=$SR_SEED"
echo "# lambda_sparsity=$LAMBDA_SPARSITY"
echo "################################################################"
python train_sr_learned_prune.py --config "$CONFIG" \
    --lambda_feat "$LAMBDA_FEAT" --lambda_saliency "$LAMBDA_SALIENCY" \
    --lambda_identity "$LAMBDA_IDENTITY" \
    --lambda_sparsity "$LAMBDA_SPARSITY" \
    --seed "$SR_SEED" --run_suffix "_final"

N_BLOCKS=$(python -c "import json; print(json.load(open('runs/sr_learned_prune_final/prune_metadata.json'))['n_blocks_kept'])")
echo "-> Model đã cứng hoá còn lại $N_BLOCKS khối (xem runs/sr_learned_prune_final/prune_metadata.json)"

# [SỬA — bổ sung sau code review, vòng 4, điểm 2] span_tiny (cắt tay) LUÔN có
# đúng 3/6 khối. So sánh downstream accuracy của learned-pruning với
# sr_improved (span_tiny) chỉ CÔNG BẰNG VỀ KÍCH THƯỚC MODEL khi N_BLOCKS==3
# (cùng params/FLOPs) — nếu khác 3, đây vẫn là so sánh HỢP LỆ nhưng là so
# sánh 2 ĐIỂM KHÁC NHAU trên đường cong đánh đổi kích thước/accuracy, KHÔNG
# PHẢI "cùng kích thước, learned pruning tốt hơn/kém hơn thủ công" — phải nêu
# rõ số khối thực tế khi báo cáo, không ngầm định bằng span_tiny.
if [ "$N_BLOCKS" -ne 3 ]; then
    echo ""
    echo "!!! LƯU Ý: N_BLOCKS=$N_BLOCKS KHÁC 3 (số khối cố định của span_tiny) !!!"
    echo "    So sánh downstream accuracy bên dưới với domain 'sr_improved' (span_tiny)"
    echo "    KHÔNG PHẢI so sánh cùng kích thước model (params/FLOPs sẽ khác đi cùng"
    echo "    N_BLOCKS) — nếu muốn kết luận 'learned pruning tốt hơn cắt tay ở CÙNG"
    echo "    ngân sách tham số', cần chỉnh lambda_sparsity (xem"
    echo "    pipeline/run_prune_sparsity_screen.sh) để N_BLOCKS hội tụ về đúng 3."
    echo "    Nếu giữ nguyên $N_BLOCKS khối, hãy báo cáo như 1 điểm KHÁC trên đường"
    echo "    cong size-accuracy, kèm rõ params/FLOPs thực tế (xem sr_quality_final.csv)"
    echo "    thay vì so sánh thẳng với span_tiny như cùng kích thước."
    echo ""
fi

echo ""
echo ">>> Sinh tập $DOMAIN..."
python data/build_sr.py --lr_dir splits/lr \
    --sr_ckpt "runs/sr_learned_prune_final/best.pt" \
    --arch span_pruned --n_blocks "$N_BLOCKS" --scale "$SCALE" --out_dir "splits/${DOMAIN}"

echo ""
echo ">>> Đo chất lượng ảnh + params/FLOPs/latency của model đã cứng hoá..."
python eval_sr_quality.py --config "$CONFIG" --arch span_pruned --n_blocks "$N_BLOCKS" \
    --ckpt "runs/sr_learned_prune_final/best.pt" \
    --label "span_learned_prune_${N_BLOCKS}blocks" \
    --out_csv "$RESULTS_DIR/sr_quality_final.csv"

echo ""
echo "################################################################"
echo "# Bước 2/2: Recognition multi-seed x 5-backbone trên domain $DOMAIN"
echo "# (${#BACKBONES[@]} backbone x ${#SEEDS[@]} seed = $((${#BACKBONES[@]} * ${#SEEDS[@]})) lần train)"
echo "################################################################"
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        LR_CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$LR_CKPT" ]; then
            echo "LỖI: không tìm thấy $LR_CKPT" >&2
            echo "  -> chạy pipeline/run_multi_seed.sh trước (ít nhất domain lr, backbone=$BACKBONE, seed=$SEED) rồi thử lại." >&2
            exit 1
        fi

        echo "----------------------------------------------------------------"
        echo "backbone=$BACKBONE | seed=$SEED | domain=$DOMAIN"
        echo "----------------------------------------------------------------"
        python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" --backbone "$BACKBONE" \
            --init_ckpt "$LR_CKPT" --seed "$SEED" --run_suffix "_seed${SEED}"

        python eval_recognition.py --config "$CONFIG" \
            --ckpt "runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt" \
            --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
            --out_json "$RESULTS_DIR/${DOMAIN}_${BACKBONE}_seed${SEED}.json"
    done
done

echo ""
echo ">>> Tổng hợp (tái sử dụng aggregator chung, xem data/aggregate_multi_seed_results.py)..."
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/multi_seed_learned_prune_summary.csv"

echo ""
echo "HOÀN TẤT. Kết quả learned pruning: $RESULTS_DIR/multi_seed_learned_prune_summary.csv"
echo ""
echo ">>> ĐỂ SO SÁNH TRỰC TIẾP với 'sr_improved' (span_tiny, cắt tay) đã có sẵn"
echo "    từ pipeline/run_multi_seed.sh — copy JSON của domain đó vào cùng thư"
echo "    mục rồi chạy lại đúng lệnh aggregate ở trên:"
echo "      cp results/multi_seed/sr_improved_*.json $RESULTS_DIR/"
echo "      python data/aggregate_multi_seed_results.py --results_dir $RESULTS_DIR \\"
echo "          --out_csv $RESULTS_DIR/multi_seed_learned_prune_summary.csv"
