# Hướng dẫn chạy lại toàn bộ thí nghiệm cho EarVN1.0

Áp dụng cho bản `earsr_project_span_v7g.zip` (bản mới nhất, đã gồm: 6 điểm sửa
lỗi đợt 7, RLFN/RLFN-adapted/ECBSR/SAFMN/SMFANet, Track A/B, LPIPS, verification
AUC/EER + confusion matrix, Cohen's d). Chạy đúng thứ tự dưới đây, không đảo
bước — nhiều bước sau phụ thuộc checkpoint của bước trước.

---

## Bước 0 — Chuẩn bị môi trường

```bash
pip install -r requirements.txt --break-system-packages   # gồm cả lpips mới thêm
python test_new_sr_archs.py                               # xác nhận nhanh mọi kiến trúc SR (gồm smfanet) chạy được, vài giây
```

Nếu Bước 0 lỗi (thiếu package, sai shape) — dừng lại xử lý trước khi chạy tiếp,
đừng để lỗi lan sang các bước tốn thời gian phía sau.

---

## Bước 1 — Pipeline chính (8 bước, tuần tự)

```bash
bash pipeline/01_survey_and_prepare_data.sh
bash pipeline/02_setup_span_official.sh     # + tải checkpoint pretrained thủ công theo README mục 2
bash pipeline/03_train_baseline_recognition.sh
bash pipeline/04_train_teacher_and_span_baseline.sh
bash pipeline/05_train_recognition_sr_baseline.sh
bash pipeline/06_improve_span.sh
bash pipeline/07_train_recognition_sr_improved.sh
bash pipeline/08_benchmark_and_aggregate.sh
```

**Kết quả**: `results/summary.csv`, `results/sr_quality.csv` (đã có cột `lpips`),
`results/real_lr_holdout.csv`, `results/training_summary.csv` — bảng kết quả
chính (span_tiny vs SPAN baseline vs EDSR, trên các domain hr/lr/sr_baseline/sr_improved).

---

## Bước 2 — Thí nghiệm bổ sung bắt buộc cho journal Q1

```bash
bash pipeline/run_ablation.sh          # ablation loss (pixel/distill/identity) -> results/ablation.csv
bash pipeline/run_lambda_sweep.sh      # quét lambda_identity x 3 seed + Cohen's d -> results/lambda_sweep/lambda_sweep_summary.csv
bash pipeline/run_multi_seed.sh        # multi-seed 5 backbone x 3 seed x 3 domain -> results/multi_seed/multi_seed_summary.csv + multi_seed_summary_pairwise.csv
bash RUN_ALL_span_large_ablation.sh    # ablation kiến trúc Tier-1 (span_tiny vs span_large — cô lập biến độ sâu)
```

`multi_seed_summary_pairwise.csv` là file **quan trọng nhất** để viết phần Results
có ý nghĩa thống kê — chứa paired t-test (raw + Bonferroni) + Cohen's d cho mọi
cặp domain, cùng backbone.

---

## Bước 3 — So sánh với SR nhẹ ngoài họ SPAN

### Track A — pixel-loss thuần (đối chiếu literature, KHÔNG so sánh công bằng trực tiếp với span_tiny)

```bash
bash RUN_ALL_extra_sr_baseline.sh rlfn
bash RUN_ALL_extra_sr_baseline.sh rlfn_adapted
bash RUN_ALL_extra_sr_baseline.sh ecbsr
bash RUN_ALL_extra_sr_baseline.sh safmn
bash RUN_ALL_extra_sr_baseline.sh smfanet
```

### Track B — cùng recipe distillation với span_tiny (dùng cho bảng kết quả CHÍNH)

```bash
bash RUN_ALL_extra_sr_baseline_distilled.sh rlfn_adapted
bash RUN_ALL_extra_sr_baseline_distilled.sh ecbsr
bash RUN_ALL_extra_sr_baseline_distilled.sh safmn
bash RUN_ALL_extra_sr_baseline_distilled.sh smfanet
```

Nếu thời gian hạn chế: **ưu tiên chạy Track B trước** — đây là bộ so sánh cô lập
đúng 1 biến (kiến trúc), dùng được cho phát biểu nhân quả trong bài báo. Track A
có thể lược bớt hoặc chạy sau, chỉ mang tính tham khảo/đối chiếu với số liệu tác
giả gốc công bố.

Cả 2 track đều tự động ghi LPIPS (`results/sr_quality.csv`) và AUC/EER/confusion
matrix (`results/*_confusion.csv`) — không cần lệnh riêng.

---

## Bước 4 — Gộp báo cáo cuối cùng

```bash
python data/generate_final_report.py --config configs/config.yaml \
    --results_dir results --out_dir results/final_report
```

**Đọc `results/final_report/REPORT.md` trước tiên** — đã gộp sẵn toàn bộ bảng:
PSNR/SSIM/LPIPS, accuracy (top-1/rank-5/AUC/EER/gender) theo backbone x domain,
real-LR holdout, training summary — kèm copy toàn bộ CSV gốc trong cùng thư mục,
không cần mở nhiều file rời rạc.

---

## Tổng kết quy mô

| Giai đoạn | Số lần train recognition |
|---|---|
| Pipeline chính (Bước 1) | tuần tự, không tính theo "lần" |
| Ablation (10.1) | 4 cấu hình x 1 backbone |
| Lambda sweep (10.2) | 6 mức x 3 seed = 18 |
| Multi-seed (10.3) | 5 backbone x 3 seed x 3 domain = 45 |
| span_large ablation | 5 backbone x 3 seed = 15 |
| Track A (5 kiến trúc) | 5 x 15 = 75 |
| Track B (4 kiến trúc) | 4 x 15 = 60 |

Tổng riêng phần multi-seed + SR mở rộng: ~213 lần train recognition. Có thể mất
nhiều giờ đến vài ngày tùy phần cứng — chạy qua đêm hoặc dùng `tmux`/`screen`/`nohup`.

## Sau khi chạy xong

Gửi lại các file sau để đánh giá có đủ mạnh cho journal Q1 chưa:
- `results/final_report/REPORT.md` (hoặc riêng từng CSV: `summary.csv`,
  `sr_quality.csv`, `multi_seed/multi_seed_summary.csv` +
  `multi_seed_summary_pairwise.csv`, `lambda_sweep/lambda_sweep_summary.csv`)
- Vài file `results/*_confusion.csv` đại diện (không cần gửi hết, chỉ cần backbone
  chính dùng cho bài báo)
