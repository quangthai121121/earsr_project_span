#!/bin/bash
# [MỚI — 2026-08-29] Quét MỊN lambda_sparsity trong khoảng [0.08, 0.1] — sàng lọc
# thô ban đầu (pipeline/run_prune_sparsity_screen.sh) cho thấy số khối NHẢY THẲNG
# tu 6 (lambda=0.08) xuong 4 (lambda=0.1), BO QUA hoan toan moc 3 khoi (kich thuoc
# span_tiny, 0.230M) — khong co diem nao de so sanh CUNG kich thuoc. Script nay
# CHỦ Ý ghi APPEND vao dung $RESULTS_DIR cua sang loc goc (khong phai loi giong
# sự cố saliency sweep — o day la bo sung CO CHU DICH, khong phai chay lai nham
# do bi dung giua chung, xem cotation trong run_prune_sparsity_screen.sh).
#
# Dung CHINH XAC cung recipe sach (feat=0/identity=0) da xac nhan cho
# run_prune_sparsity_screen.sh — PHAI khop de so sanh cong bang giua cac diem.
#
# YÊU CẦU: đã chạy pipeline/run_prune_sparsity_screen.sh (co san checkpoint
# recognition_lr_<backbone> khong suffix seed).

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results/prune_sparsity_screen"
BACKBONE="mobilenet_v2"
LAMBDA_FEAT=0.0
LAMBDA_SALIENCY=0.0
LAMBDA_IDENTITY=0.0
LAMBDA_SPARSITY_VALUES=(0.085 0.09 0.095)
mkdir -p "$RESULTS_DIR"

SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

LR_CKPT_NOSUFFIX="runs/recognition_lr_${BACKBONE}/best.pt"
if [ ! -f "$LR_CKPT_NOSUFFIX" ]; then
    echo "LỖI: thiếu $LR_CKPT_NOSUFFIX (checkpoint KHÔNG suffix seed)."
    exit 1
fi

# Nhat quan voi run_prune_sparsity_screen.sh: kiem tra judge NEU sau nay co ai
# doi LAMBDA_SALIENCY/IDENTITY o tren khac 0 (hien tai luon la 0.0 nen dieu
# kien duoi day khong kich hoat, nhung giu lai de khong am tham bo sot neu gia
# tri thay doi sau nay).
if python -c "import sys; sys.exit(0 if float('$LAMBDA_SALIENCY') > 0 or float('$LAMBDA_IDENTITY') > 0 else 1)"; then
    source "$(dirname "$0")/_check_hr_judges.sh"
    check_hr_judges
fi

for LS in "${LAMBDA_SPARSITY_VALUES[@]}"; do
    TAG="lsp${LS}"

    # Bo qua neu da chay xong (cung tinh than logic resume da them cho saliency sweep)
    EXISTING=$(ls "$RESULTS_DIR"/acc_${TAG}_nblocks*.json 2>/dev/null | head -1)
    if [ -n "$EXISTING" ]; then
        echo ">>> Bỏ qua $TAG (đã có $EXISTING)"
        continue
    fi

    echo "################################################################"
    echo "# lambda_sparsity=$LS (quet min, feat=0/identity=0/saliency=0)"
    echo "################################################################"

    python train_sr_learned_prune.py --config "$CONFIG" \
        --lambda_feat "$LAMBDA_FEAT" --lambda_saliency "$LAMBDA_SALIENCY" \
        --lambda_identity "$LAMBDA_IDENTITY" \
        --lambda_sparsity "$LS" --run_suffix "_${TAG}"

    N_BLOCKS=$(python -c "import json; print(json.load(open('runs/sr_learned_prune_${TAG}/prune_metadata.json'))['n_blocks_kept'])")
    echo "-> lambda_sparsity=$LS giữ lại $N_BLOCKS khối"
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
echo "HOÀN TẤT quét mịn. Xem lại toàn bộ (thô + mịn):"
echo "  cat $RESULTS_DIR/acc_*.json"
echo "  cat $RESULTS_DIR/sr_quality_screen.csv"
