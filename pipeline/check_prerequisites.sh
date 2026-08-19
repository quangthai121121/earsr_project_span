#!/bin/bash
# [MỚI] Kiểm tra nhanh xem đã có đủ checkpoint cần thiết để BỎ QUA Giai đoạn
# A/B (train từ đầu) và nhảy thẳng vào Giai đoạn C (2 cơ chế mới tăng
# novelty: KD v2, saliency-weighted loss, learned pruning) hay chưa.
# Chạy: bash pipeline/check_prerequisites.sh

BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
JUDGES=("mobilenet_v2" "resnet18" "ghostnet_100")
SEEDS=(42 123 2024)
missing=0

check() {
    if [ -f "$1" ]; then
        echo "  [OK]    $1"
    else
        echo "  [THIẾU] $1"
        missing=$((missing + 1))
    fi
}

# Mục 4 (multi_seed summary) chỉ cần khi SO SÁNH với bảng cũ — không chặn
# chạy ablation KD v2 / saliency / prune. Không tính vào `missing`.
check_optional() {
    if [ -f "$1" ]; then
        echo "  [OK]    $1"
    else
        echo "  [TUỲ CHỌN — thiếu] $1"
        echo "           (không chặn Giai đoạn C; chỉ cần khi so sánh với results/multi_seed cũ)"
    fi
}

# [SỬA — bổ sung sau code review, vòng 6] Trước đây script này KHÔNG kiểm tra
# splits/splits.json + splits/hr|lr/{train,val} — tiền đề CƠ BẢN NHẤT, cần cho
# MỌI script train (train_sr_distill.py, train_sr_learned_prune.py) chứ không
# riêng Giai đoạn C. Thiếu mục này, người chạy dễ tưởng "check_prerequisites.sh
# OK" là an toàn để bấm chạy trong khi chưa chuẩn bị dữ liệu.
check_dir_nonempty() {
    if [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; then
        echo "  [OK]    $1 ($(find "$1" -type f | wc -l | tr -d ' ') file)"
    else
        echo "  [THIẾU] $1 (không tồn tại hoặc rỗng)"
        missing=$((missing + 1))
    fi
}

echo "=== 0) Dữ liệu splits (bắt buộc cho MỌI script train, không riêng Giai đoạn C) ==="
check "splits/splits.json"
check_dir_nonempty "splits/hr/train"
check_dir_nonempty "splits/hr/val"
check_dir_nonempty "splits/lr/train"
check_dir_nonempty "splits/lr/val"

echo ""
echo "=== 1) SPAN pretrained + teacher ==="
check "checkpoints/span_pretrained_x4.pth"
check "runs/sr_span_official/best.pt"

echo ""
echo "=== 2) Recognition domain HR (dùng làm judge cho identity loss) ==="
for b in "${JUDGES[@]}"; do
    check "runs/recognition_hr_${b}/best.pt"
done

# [SỬA — bổ sung sau code review, vòng 8, điểm 2] TRƯỚC ĐÂY dùng check()
# (cộng vào `missing`, chặn "DUYỆT") cho ĐỦ 15 checkpoint (5 backbone x 3
# seed) — nhưng 15 file này CHỈ cần cho run_multi_seed_kdv2.sh /
# run_multi_seed_learned_prune.sh (validate multi-seed x multi-backbone đầy
# đủ, dùng để VIẾT BÀI BÁO). Các script sàng lọc nhanh 1-backbone
# (run_ablation.sh, run_ablation_kd_v2.sh, run_lambda_sweep.sh,
# run_lambda_saliency_sweep.sh, run_prune_sparsity_screen.sh) chỉ cần ĐÚNG 1
# checkpoint KHÔNG suffix (mục 3b bên dưới) — tự chúng đã check riêng lúc
# chạy. Coi cả 15 file này là "bắt buộc" khiến script này báo "THIẾU" ngay cả
# khi người dùng CHỈ muốn chạy ablation KD v2 (không phải điều kiện của
# train_sr_distill.py) — cùng lỗi logic đã sửa ở mục 4 (multi_seed_summary.csv).
# Đổi sang check_optional(), liệt kê rõ backbone/seed còn thiếu để tiện theo dõi.
echo ""
echo "=== 3) Recognition domain LR, per-seed (TUỲ CHỌN — chỉ cần cho"
echo "     run_multi_seed_kdv2.sh / run_multi_seed_learned_prune.sh; KHÔNG chặn"
echo "     ablation/sweep 1-backbone, xem mục 3b) ==="
n_seed_missing=0
for b in "${BACKBONES[@]}"; do
    for s in "${SEEDS[@]}"; do
        f="runs/recognition_lr_${b}_seed${s}/best.pt"
        if [ -f "$f" ]; then
            echo "  [OK]    $f"
        else
            echo "  [TUỲ CHỌN — thiếu] $f"
            n_seed_missing=$((n_seed_missing + 1))
        fi
    done
done
if [ "$n_seed_missing" -gt 0 ]; then
    echo "  -> thiếu $n_seed_missing/$(( ${#BACKBONES[@]} * ${#SEEDS[@]} )) checkpoint (không chặn"
    echo "     Giai đoạn C nếu chỉ chạy ablation/sweep 1-backbone; cần ĐỦ nếu muốn chạy"
    echo "     run_multi_seed_kdv2.sh / run_multi_seed_learned_prune.sh)."
fi

# [SỬA — bổ sung sau code review, vòng 5, điểm 2] Mục 3 ở trên CHỈ kiểm tra
# checkpoint có suffix "_seed<N>" (dùng cho run_multi_seed*.sh) — NHƯNG các
# script sàng lọc nhanh (run_ablation.sh, run_lambda_sweep.sh,
# run_ablation_kd_v2.sh, run_lambda_saliency_sweep.sh, run_prune_sparsity_screen.sh)
# lại cần checkpoint KHÔNG suffix "runs/recognition_lr_mobilenet_v2/best.pt"
# (sinh ra bởi pipeline/03_train_baseline_recognition.sh, domain "lr", KHÔNG
# --seed/--run_suffix). Thiếu mục kiểm tra riêng này từng khiến các script
# trên train SR xong (hàng giờ) rồi mới FileNotFoundError ở bước
# train_recognition.py đầu tiên — bổ sung kiểm tra ở đây để phát hiện SỚM.
echo ""
echo "=== 3b) Recognition domain LR, KHÔNG suffix (dùng cho các script sàng lọc nhanh:"
echo "     run_ablation.sh / run_lambda_sweep.sh / run_ablation_kd_v2.sh /"
echo "     run_lambda_saliency_sweep.sh / run_prune_sparsity_screen.sh — chỉ cần"
echo "     backbone mobilenet_v2, các script này không dùng cả 5 backbone) ==="
check "runs/recognition_lr_mobilenet_v2/best.pt"

echo ""
echo "=== 4) Kết quả multi_seed cũ (TUỲ CHỌN — chỉ để so sánh sr_improved vs recipe mới) ==="
check_optional "results/multi_seed/multi_seed_summary.csv"
n_json=$(ls results/multi_seed/sr_improved_*_seed*.json 2>/dev/null | wc -l | tr -d ' ')
echo "  -> tìm thấy $n_json file results/multi_seed/sr_improved_*_seed*.json (kỳ vọng: $(( ${#BACKBONES[@]} * ${#SEEDS[@]} )))"

echo ""
if [ "$missing" -eq 0 ]; then
    echo ">>> DUYỆT: đủ điều kiện chạy ablation/sweep 1-backbone (run_ablation*.sh,"
    echo "    run_lambda_sweep.sh, run_lambda_saliency_sweep.sh, run_prune_sparsity_screen.sh)."
    if [ "$n_seed_missing" -gt 0 ]; then
        echo "    LƯU Ý: còn thiếu $n_seed_missing checkpoint mục 3 (per-seed) — CẦN chạy"
        echo "    pipeline/run_multi_seed.sh trước nếu muốn chạy run_multi_seed_kdv2.sh /"
        echo "    run_multi_seed_learned_prune.sh (validate multi-seed x multi-backbone)."
    fi
else
    echo ">>> THIẾU $missing file bắt buộc — cần chạy phần Giai đoạn A/B tương ứng còn thiếu trước."
fi
