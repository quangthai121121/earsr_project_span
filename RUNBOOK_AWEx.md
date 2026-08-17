# Hướng dẫn chạy dataset thứ 2 = AWEx (đã bỏ AWE)

Áp dụng cho `earsr_project_span_v7m.zip` (bản mới nhất — có `--freeze_backbone`,
n=5 seed mặc định, đã sửa lỗi lẫn đường dẫn giữa các dataset). Quyết định:
**bỏ hẳn AWE, dataset thứ 2 chính thức là AWEx (336 người: 100 AWE + 16 CVLE +
220 New images, gộp theo README của AWEx).**

---

## Bước 0 — Chuyển sang code mới, giữ lại dữ liệu EarVN1.0 đã có

Code cũ bạn đang chạy thiếu nhiều fix quan trọng (bao gồm 1 lỗi có thể XOÁ
NHẦM `results/` của EarVN1.0 — xem CHANGELOG_v7.md Phần I). Giải nén
`earsr_project_span_v7m.zip` làm bản làm việc mới, rồi mang dữ liệu EarVN1.0
đã có sẵn qua (KHÔNG mang AWE qua nữa):

```bash
cd /đường/dẫn/tới/bản_v7m_mới_giải_nén
cp -r /đường/dẫn/tới/project_cũ/raw_data/EarVN1.0_raw raw_data/
cp -r /đường/dẫn/tới/project_cũ/splits ./
cp -r /đường/dẫn/tới/project_cũ/runs ./       # nếu có (checkpoint EarVN1.0 đã train)
cp -r /đường/dẫn/tới/project_cũ/results ./    # nếu có
```

Kiểm tra nhanh sau khi copy — 3 file đã patch phải đúng bản mới:

```bash
grep -q "student_arch" train_sr_distill.py && echo OK1
grep -q "compute_topk_accuracy" utils/metrics.py && echo OK2
grep -q "compute_topk_accuracy" eval_recognition.py && echo OK3
```

---

## Bước 1 — Chuẩn bị dữ liệu AWEx thô

Nếu chưa có `raw_data/awex_raw/` (336 thư mục `001`-`336`):

1. Tải AWEx (AWE + CVLE + New images), giải nén.
2. Gộp 3 thư mục con thành 1 pool bằng `flatten_awex.sh` (đã gửi trước đó):
   ```bash
   cd /đường/dẫn/tới/AWE-ex
   bash /đường/dẫn/tới/flatten_awex.sh
   ```
3. Convert sang định dạng project — **luôn `--dry_run` trước**:
   ```bash
   cd /đường/dẫn/tới/bản_v7m_mới_giải_nén
   python data/convert_generic_dataset_to_project_format.py \
       --in_dir "/đường/dẫn/tới/AWE-ex/awex_flat" --out_dir raw_data/awex_raw --dry_run
   ```
   Đọc kỹ dòng "Số ảnh/người: min/max/trung bình" — khớp kỳ vọng (~336 người,
   ~11-12 ảnh/người trung bình) mới bỏ `--dry_run` chạy thật.

Nếu đã có `raw_data/awex_raw/` rồi (như hiện tại) — bỏ qua bước này.

---

## Bước 2 — Pipeline chính cho AWEx (from-scratch, n=3 seed, điểm so sánh bắt buộc)

```bash
bash RUN_ALL_NEW_DATASET.sh awex 336
```

Tự động: tạo `configs/config_awex.yaml` + `pipeline_awex/`, chạy 8 bước
chính, multi-seed cơ bản (n=3: seed 42/123/2024), span_large ablation.
**Tốn nhiều giờ** — nên chạy qua đêm / dùng `tmux`/`nohup`.

---

## Bước 3 — Mở rộng n=3 → n=5 (đồng bộ lực kiểm định với EarVN1.0)

```bash
bash pipeline_awex/run_multi_seed_extra_seeds.sh
```

Chỉ thêm seed 44/999 cho recognition (không train lại SR).

---

## Bước 4 — Transfer learning từ EarVN1.0 (kết quả CHÍNH, khắc phục vấn đề đã gặp ở AWE)

**Yêu cầu tiền đề**: EarVN1.0 đã có đủ checkpoint recognition n=5 seed
(`runs/recognition_<domain>_<backbone>_seed<seed>/best.pt` với domain =
lr/sr_baseline/sr_improved, 5 backbone, seed = 42/123/2024/44/999) và
`runs_awex/sr_span_official/best.pt` + `runs_awex/recognition_hr_mobilenet_v2/best.pt`
từ Bước 2.

```bash
bash scripts/run_transfer_learning.sh awex earvn1
```

Mặc định đã n=5 seed. Kết quả: `results_awex/sr_quality_transfer.csv`,
`results_awex/multi_seed_transfer/multi_seed_summary_transfer.csv` +
`_pairwise.csv`.

---

## Bước 5 — (Tuỳ chọn, sau khi có kết quả Bước 4) Freeze-backbone — giảm phương sai

Chỉ chạy sau khi đã xem kết quả Bước 4 và thấy phương sai giữa seed vẫn lớn
(như đã xảy ra với AWE):

```bash
bash scripts/run_transfer_learning_frozen.sh awex earvn1
```

Kết quả: `results_awex/multi_seed_transfer_frozen/multi_seed_summary_transfer_frozen.csv`
+ `_pairwise.csv` — so với file Bước 4 để xem freeze có giúp giảm phương sai/
đạt ý nghĩa thống kê hơn không.

---

## Sau mỗi bước — gửi lại file gì

| Sau bước | Gửi lại |
|---|---|
| 1 (dry_run) | output dry_run trên chat (không cần file) |
| 2+3 | `splits_awex/dataset_stats.csv`, `results_awex/sr_quality.csv`, `results_awex/multi_seed/multi_seed_summary_pairwise.csv` |
| 4 | `results_awex/sr_quality_transfer.csv`, `results_awex/multi_seed_transfer/multi_seed_summary_transfer_pairwise.csv` |
| 5 | `results_awex/multi_seed_transfer_frozen/multi_seed_summary_transfer_frozen_pairwise.csv` |

Không cần zip toàn bộ `results_awex/` mỗi lần — các file CSV trên là đủ để
đánh giá, nhẹ hơn nhiều so với gửi cả checkpoint.
