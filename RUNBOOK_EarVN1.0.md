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

**[Dễ bỏ sót] Ảnh so sánh trực quan LR | SPAN baseline | span_tiny | HR** — script này
đã có sẵn trong project nhưng KHÔNG nằm trong 8 bước pipeline chính (phải gọi
riêng), chạy được ngay sau khi Bước 1 xong (đã có đủ 2 checkpoint cần):

```bash
python export_sr_comparison_images.py --config configs/config.yaml \
    --sr_baseline_ckpt runs/sr_span_official/best.pt --sr_baseline_arch span_official \
    --sr_improved_ckpt runs/sr_improved_span_tiny/best.pt --sr_improved_arch span_tiny \
    --n_samples 20 --out_dir results/sr_comparison_images
```

Xuất `results/sr_comparison_images/sample_00.png ... sample_19.png` (mỗi ảnh 1 hàng:
LR bicubic thường | SR baseline kèm PSNR | SR improved (span_tiny) kèm PSNR | HR gốc)
+ `comparison_metrics.csv`. Dùng trực tiếp làm hình minh hoạ định tính trong bài báo.

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

## Bước 2.5 — [MỚI, đợt 8] Tăng lực kiểm định + trả lời câu hỏi KD bằng số liệu thật

Chỉ cần nếu muốn: (a) n=5 seed thay vì n=3 cho bảng kết quả chính (lực kiểm định
mạnh hơn), và (b) câu trả lời THẬT (không phải cảm tính từ n=1) cho câu hỏi
"distillation có giúp span_tiny không". Chạy đúng thứ tự:

```bash
bash pipeline/run_multi_seed_extra_seeds.sh   # thêm seed 44,999 -> n=5 cho lr/sr_baseline/sr_improved
bash pipeline/run_ablation_multiseed.sh        # multi-seed (5 seed) riêng bước recognition cho 4 cấu hình ablation
python debug_smfanet.py --config configs/config.yaml \
    --ckpt runs/sr_smfanet/best.pt --arch smfanet \
    --compare_ckpt runs/sr_rlfn_adapted/best.pt --compare_arch rlfn_adapted \
    --out_dir results/debug_smfanet   # chẩn đoán PSNR thấp bất thường của SMFANet, KHÔNG tự sửa
```

`results/ablation_multiseed/ablation_multiseed_summary_pairwise.csv` — đọc dòng
`pixel_distill` vs `pixel_only` để có câu trả lời KD có tác dụng hay không kèm
p-value + Cohen's d thật (thay vì chỉ dựa n=1). `results/debug_smfanet/report.txt`
— đọc trước khi quyết định giữ/loại SMFANet khỏi bảng so sánh chính.

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

## Bước 3.5 — [MỚI, đợt 9] Đồng bộ n=5 seed cho Track A/B + span_large-ablation

Chỉ cần nếu muốn lực kiểm định thống kê ĐỀU giữa mọi phần của bài báo (Bước 2.5
đã đưa bảng kết quả chính + ablation KD lên n=5, nhưng Track A/B và
span_large-ablation ở Bước 2 và Bước 3 vẫn n=3). Chạy SAU khi đã có:
`pipeline/run_multi_seed_extra_seeds.sh` (Bước 2.5), `RUN_ALL_span_large_ablation.sh`
(Bước 2, bản gốc), và Track A/B ở Bước 3 ngay trên đây (bản gốc):

```bash
bash RUN_ALL_span_large_ablation_extra_seeds.sh
for ARCH in rlfn rlfn_adapted ecbsr safmn smfanet; do
    bash RUN_ALL_extra_sr_baseline_extra_seeds.sh "$ARCH"
done
for ARCH in rlfn_adapted ecbsr safmn smfanet; do
    bash RUN_ALL_extra_sr_baseline_distilled_extra_seeds.sh "$ARCH"
done
```

Không train lại SR — chỉ thêm seed 44,999 cho bước recognition, ghi đè các CSV
tổng hợp bằng bản n=5 đầy đủ hơn (tên file giữ nguyên như bản n=3 gốc).

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
| [Bước 2.5] +2 seed cho bảng chính | 5 backbone x 2 seed x 3 domain = 30 |
| [Bước 2.5] ablation KD multi-seed | 4 cấu hình x 5 seed = 20 |
| [Bước 3.5] +2 seed span_large-ablation | 5 backbone x 2 seed = 10 |
| [Bước 3.5] +2 seed Track A (5 kiến trúc) | 5 x 10 = 50 |
| [Bước 3.5] +2 seed Track B (4 kiến trúc) | 4 x 10 = 40 |

Tổng riêng phần multi-seed + SR mở rộng (Bước 2+3): ~213 lần train recognition.
Nếu chạy thêm cả Bước 2.5 + 3.5 (đồng bộ n=5): +150 lần nữa (~363 tổng). Có thể
mất nhiều giờ đến vài ngày tùy phần cứng — chạy qua đêm hoặc dùng `tmux`/`screen`/`nohup`.
Bước 2.5/3.5 KHÔNG bắt buộc — n=3 đã đủ hợp lệ để chạy paired t-test, chỉ là lực
kiểm định yếu hơn n=5; có thể bỏ qua nếu thời gian hạn chế, nêu rõ n=3 trong
phần Limitations của bài báo.

## Sau khi chạy xong

Gửi lại các file sau để đánh giá có đủ mạnh cho journal Q1 chưa:
- `results/final_report/REPORT.md` (hoặc riêng từng CSV: `summary.csv`,
  `sr_quality.csv`, `multi_seed/multi_seed_summary.csv` +
  `multi_seed_summary_pairwise.csv`, `lambda_sweep/lambda_sweep_summary.csv`)
- Vài file `results/*_confusion.csv` đại diện (không cần gửi hết, chỉ cần backbone
  chính dùng cho bài báo)
