#!/bin/bash
# [MỚI] Sàng lọc nhanh hệ số lambda_sparsity cho học pruning độ sâu
# (train_sr_learned_prune.py) — 1 backbone, 1 seed/mức, chỉ để xem:
#   (a) mức sparsity nào cho ra SỐ KHỐI CÒN LẠI hợp lý (so sánh với span_tiny,
#       3/6 khối) sau khi harden_and_export(),
#   (b) accuracy downstream sơ bộ ở mỗi mức.
# ĐÂY LÀ TÍN HIỆU SÀNG LỌC NHANH, không phải bằng chứng cuối — sau khi chọn
# được mức lambda_sparsity, PHẢI chạy pipeline/run_multi_seed_learned_prune.sh
# (multi-seed x 5-backbone) trước khi đưa vào bảng kết quả chính thức.
#
# YÊU CẦU: đã chạy xong pipeline 01-05 + đã có checkpoint recognition domain
# "hr" cho các judge trong configs/config.yaml::sr_improve.identity_judges.

# [SỬA — bổ sung sau code review, vòng 2, điểm 3; LÀM RÕ THÊM ở vòng 5]
# **SỬA 3 THAM SỐ NÀY** trước khi chạy, khớp ĐÚNG cấu hình đã THẮNG ở
# pipeline/run_ablation_kd_v2.sh + run_multi_seed_kdv2.sh. LƯU Ý: LAMBDA_FEAT
# và (LAMBDA_SALIENCY/LAMBDA_IDENTITY) KHÔNG cùng vai trò — LAMBDA_FEAT
# (feature-KD so khớp SR teacher) KHÔNG dùng tín hiệu nhận dạng nào, chỉ
# LAMBDA_SALIENCY/LAMBDA_IDENTITY mới thật sự "identity-aware" (lấy tín hiệu
# từ recognition judge). Nếu CHỈ bật LAMBDA_FEAT (2 cái còn lại vẫn =0), script
# vẫn CHỈ là "reconstruction-aware pruning" (có thêm feature-KD), KHÔNG PHẢI
# "identity-aware pruning" như tên gọi/docstring train_sr_learned_prune.py mô
# tả — xem cảnh báo runtime trong train_sr_learned_prune.py::main() (chỉ tắt
# khi LAMBDA_SALIENCY=LAMBDA_IDENTITY=0, không phụ thuộc LAMBDA_FEAT) và cờ
# "identity_aware"/"uses_feature_kd" riêng biệt trong prune_metadata.json.
#
# [SỬA — bổ sung sau code review, đợt 10] Dòng comment BẢN CŨ ghi "mặc định để
# 0 — TẮT cả 3" — ĐÃ LỖI THỜI, vì 3 giá trị dưới đây ĐÃ được pin sẵn =
# 0.5/0.0/0.1 (không còn là 0 nữa) sau khi phát hiện lần chạy đợt 8 chỉ ra
# "reconstruction prune" (xem đoạn giải thích ngay dưới). Không tắt cả 3 mặc
# định — PHẢI tự sửa lại nếu muốn tắt.
#
# [SỬA — bổ sung sau đánh giá kết quả thật, đợt 9] LẦN CHẠY TRƯỚC (để 3 giá
# trị này = 0) cho ra prune_metadata.json với identity_aware=false,
# uses_feature_kd=false — nghĩa là ĐÃ chạy "reconstruction prune" thuần, KHÔNG
# phải "identity-aware learned pruning" như bài báo claim. PIN CỨNG 2 giá trị
# đã thắng ở ablation KDv2 (kdv2_full: lambda_feat=0.5, lambda_identity=0.1)
# để lần chạy lại này thực sự test đúng cơ chế "identity-aware pruning".
LAMBDA_FEAT=0.5
LAMBDA_SALIENCY=0.0
LAMBDA_IDENTITY=0.1

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results/prune_sparsity_screen"
BACKBONE="mobilenet_v2"
# [SỬA — bổ sung sau đánh giá kết quả thật, đợt 9] LẦN CHẠY TRƯỚC (lambda_*
# identity đều =0) cho kết quả: 0.0/0.01/0.05 -> 6 khối (KHÔNG prune gì cả),
# 0.1 -> 5 khối, 0.2 -> 2 khối — KHÔNG có mốc nào ra đúng 3 khối (bằng
# span_tiny) để so sánh công bằng về kích thước. Set lambda_feat/identity vừa
# pin ở trên là 1 dạng regularization THÊM (ép model giữ khối nào giúp
# identity), nên hành vi giữ/bỏ khối CÓ THỂ đổi khác so với lần trước — quét
# lại TOÀN BỘ dải, tập trung dày hơn ở khoảng (0.1, 0.2) nơi ranh giới
# 5->3->2 khối nhiều khả năng nằm trong đó.
#
# [SỬA — bổ sung sau code review, đợt 10, điểm "dải sparsity bỏ 0.0"] Thêm lại
# mốc 0.0 làm ĐỐI CHỨNG "identity-aware nhưng KHÔNG phạt sparsity" (λ_feat=0.5,
# λ_identity=0.1, λ_sparsity=0) — cần mốc này để tách được 2 hiệu ứng riêng:
# (a) bật identity/feature loss có tự làm model muốn giữ ÍT/NHIỀU khối hơn
# ngay cả khi không bị phạt sparsity không, hay (b) hoàn toàn do λ_sparsity
# ép giảm khối. Nếu bỏ mốc 0.0, không có cách phân biệt 2 nguyên nhân này.
LAMBDA_SPARSITY_VALUES=(0.0 0.05 0.08 0.1 0.13 0.16 0.2)
mkdir -p "$RESULTS_DIR"

# [MỚI — bổ sung sau code review, đợt 10, điểm 2] eval_sr_quality.py GHI
# APPEND (không overwrite) vào $RESULTS_DIR/sr_quality_screen.csv, và các file
# acc_<tag>_nblocks<N>.json cũng KHÔNG bị xoá giữa các lần chạy. Nếu
# $RESULTS_DIR đã có dữ liệu từ lần chạy TRƯỚC (rất có thể là lần đợt 8, dùng
# λ_feat=λ_identity=0 — "reconstruction prune", KHÔNG identity-aware), lần
# chạy MỚI này (đã pin λ_feat/identity) sẽ ghi THÊM vào, để 2 loại số liệu
# (reconstruction-prune cũ và identity-aware mới) LẪN VÀO CÙNG 1 CSV mà không
# có cột nào phân biệt rõ — dễ đọc nhầm khi tổng hợp bài báo. Chặn NGAY từ
# đầu, yêu cầu di chuyển thư mục cũ đi trước khi chạy tiếp.
if [ -d "$RESULTS_DIR" ] && [ -n "$(ls -A "$RESULTS_DIR" 2>/dev/null)" ]; then
    echo "LỖI: $RESULTS_DIR đã tồn tại và không rỗng." >&2
    echo "     eval_sr_quality.py ghi APPEND vào sr_quality_screen.csv — nếu thư mục" >&2
    echo "     này còn dữ liệu từ lần sàng lọc TRƯỚC (có thể là bản reconstruction-" >&2
    echo "     prune cũ, không identity-aware), số liệu cũ/mới sẽ LẪN VÀO NHAU." >&2
    echo "     -> di chuyển thư mục cũ đi trước, ví dụ:" >&2
    echo "        mv $RESULTS_DIR ${RESULTS_DIR}_$(date +%Y%m%d_%H%M%S)" >&2
    echo "     rồi chạy lại script này." >&2
    exit 1
fi

# [SỬA — lỗi phát hiện qua code review] đọc scale từ config thay vì hardcode 4
# — nhất quán với mọi script khác trong pipeline (nếu config đổi scale, script
# này không bị lệch âm thầm).
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

# [SỬA — bổ sung sau code review, vòng 5, điểm 2] Script này fine-tune
# recognition từ checkpoint KHÔNG suffix "runs/recognition_lr_${BACKBONE}/best.pt"
# (sinh ra bởi pipeline/03_train_baseline_recognition.sh) — KHÁC với checkpoint
# multi-seed "_seed<seed>" mà pipeline/check_prerequisites.sh và
# pipeline/run_multi_seed.sh kiểm tra/tạo ra. Nếu máy CHỈ chạy qua đường
# multi-seed (chỉ có *_seed42/_seed123/..., KHÔNG có bản không suffix), script
# sẽ train xong toàn bộ SR (hàng giờ, tốn GPU) rồi MỚI FileNotFoundError ở
# bước train_recognition.py đầu tiên — kiểm tra NGAY TỪ ĐẦU để fail nhanh.
LR_CKPT_NOSUFFIX="runs/recognition_lr_${BACKBONE}/best.pt"
if [ ! -f "$LR_CKPT_NOSUFFIX" ]; then
    echo "LỖI: thiếu $LR_CKPT_NOSUFFIX (checkpoint KHÔNG suffix seed, dùng để"
    echo "     init_ckpt cho recognition fine-tune ở script này)."
    echo "     -> chạy 'python train_recognition.py --config $CONFIG --domain lr"
    echo "        --backbone $BACKBONE' (không --seed/--run_suffix, xem"
    echo "        pipeline/03_train_baseline_recognition.sh) trước, KHÁC với"
    echo "        checkpoint '_seed<N>' mà pipeline/run_multi_seed.sh tạo ra."
    exit 1
fi

# Nếu user đã sửa LAMBDA_SALIENCY/IDENTITY > 0 thì cần judge TRƯỚC khi train SR.
if python -c "import sys; sys.exit(0 if float('$LAMBDA_SALIENCY') > 0 or float('$LAMBDA_IDENTITY') > 0 else 1)"; then
    source "$(dirname "$0")/_check_hr_judges.sh"
    check_hr_judges
fi

for LS in "${LAMBDA_SPARSITY_VALUES[@]}"; do
    TAG="lsp${LS}"
    echo "################################################################"
    echo "# lambda_sparsity=$LS"
    echo "################################################################"

    python train_sr_learned_prune.py --config "$CONFIG" \
        --lambda_feat "$LAMBDA_FEAT" --lambda_saliency "$LAMBDA_SALIENCY" \
        --lambda_identity "$LAMBDA_IDENTITY" \
        --lambda_sparsity "$LS" --run_suffix "_${TAG}"

    N_BLOCKS=$(python -c "import json; print(json.load(open('runs/sr_learned_prune_${TAG}/prune_metadata.json'))['n_blocks_kept'])")
    echo "-> lambda_sparsity=$LS giữ lại $N_BLOCKS khối (xem runs/sr_learned_prune_${TAG}/prune_metadata.json)"
    # [SỬA — bổ sung sau code review, vòng 4, điểm 2] span_tiny cố định 3/6 khối —
    # chỉ mức lambda_sparsity nào cho N_BLOCKS==3 mới "cùng kích thước" với span_tiny.
    if [ "$N_BLOCKS" -eq 3 ]; then
        echo "   (N_BLOCKS=3 -> CÙNG kích thước với span_tiny, ứng viên tốt để chốt lambda_sparsity)"
    fi

    python data/build_sr.py --lr_dir splits/lr \
        --sr_ckpt "runs/sr_learned_prune_${TAG}/best.pt" \
        --arch span_pruned --n_blocks "$N_BLOCKS" --scale "$SCALE" \
        --out_dir "splits/sr_prune_screen_${TAG}"

    python train_recognition.py --config "$CONFIG" --domain "sr_prune_screen_${TAG}" \
        --backbone "$BACKBONE" --init_ckpt "runs/recognition_lr_${BACKBONE}/best.pt"

    python eval_recognition.py --config "$CONFIG" \
        --ckpt "runs/recognition_sr_prune_screen_${TAG}_${BACKBONE}/best.pt" \
        --backbone "$BACKBONE" --train_domain "sr_prune_screen_${TAG}" \
        --test_domain "sr_prune_screen_${TAG}" \
        --out_json "$RESULTS_DIR/acc_${TAG}_nblocks${N_BLOCKS}.json"

    python eval_sr_quality.py --config "$CONFIG" --arch span_pruned --n_blocks "$N_BLOCKS" \
        --ckpt "runs/sr_learned_prune_${TAG}/best.pt" \
        --label "prune_${TAG}_nblocks${N_BLOCKS}" \
        --out_csv "$RESULTS_DIR/sr_quality_screen.csv"
done

echo ""
echo "HOÀN TẤT sàng lọc lambda_sparsity. Xem:"
echo "  - $RESULTS_DIR/acc_*.json          (accuracy downstream mỗi mức, tên file đã gồm n_blocks_kept)"
echo "  - $RESULTS_DIR/sr_quality_screen.csv (PSNR/SSIM/params/FLOPs/latency mỗi mức)"
echo "Chọn mức lambda_sparsity cho số khối + accuracy hợp lý nhất, rồi SỬA"
echo "lambda_sparsity mặc định trong configs/config.yaml, sau đó chạy"
echo "pipeline/run_multi_seed_learned_prune.sh để validate đầy đủ trước khi"
echo "đưa vào bảng kết quả chính thức."
