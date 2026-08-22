# RUNBOOK — AWEx (dataset thứ 2)

Hướng dẫn chạy đầy đủ cho dataset thứ 2 = **AWEx** (336 người: 100 AWE + 16
CVLE + 220 New images, gộp theo README của AWEx) — dùng để kiểm chứng
`span_tiny` tổng quát hóa sang dataset khác ngoài EarVN1.0. Đã **bỏ hẳn AWE**
(dataset gốc, chỉ 100 người) vì quá nhỏ để có kết quả đáng tin cậy (xem mục
"Vì sao bỏ AWE" bên dưới). File này gộp toàn bộ nội dung liên quan AWEx
trước đây rải rác trong `CHANGELOG_v7.md` (Phần H/I/J) — xem
`RUNBOOK_EarVN1.0.md` cho dataset chính.

---

## Bước 0 — Chuẩn bị dữ liệu AWEx thô

Nếu chưa có `raw_data/awex_raw/` (336 thư mục `001`-`336`):

1. Tải AWEx (AWE + CVLE + New images), giải nén.
2. Gộp 3 thư mục con thành 1 pool bằng `flatten_awex.sh`:
   ```bash
   cd /đường/dẫn/tới/AWE-ex
   bash /đường/dẫn/tới/flatten_awex.sh
   ```
3. Convert sang định dạng project — **luôn `--dry_run` trước**:
   ```bash
   cd /đường/dẫn/tới/earsr_project_span
   python data/convert_generic_dataset_to_project_format.py \
       --in_dir "/đường/dẫn/tới/AWE-ex/awex_flat" --out_dir raw_data/awex_raw --dry_run
   ```
   Đọc kỹ dòng "Số ảnh/người: min/max/trung bình" — khớp kỳ vọng (~336
   người, ~11-12 ảnh/người trung bình) mới bỏ `--dry_run` chạy thật.

Nếu đã có `raw_data/awex_raw/` rồi — bỏ qua bước này.

---

## Bước 1 — Pipeline chính cho AWEx (from-scratch, n=3 seed, điểm so sánh bắt buộc)

```bash
bash RUN_ALL_NEW_DATASET.sh awex 336
```

Tự động: tạo `configs/config_awex.yaml` + `pipeline_awex/` (sao chép nguyên
`pipeline/` gốc rồi sed-thay toàn bộ đường dẫn `runs/`→`runs_awex/`,
`splits/`→`splits_awex/`, `results/`→`results_awex/` — xem "Cách ly dữ liệu
2 dataset" bên dưới), chạy 8 bước chính, multi-seed cơ bản (n=3: seed
42/123/2024), span_large ablation. **Tốn nhiều giờ** — nên chạy qua đêm.

---

## Bước 2 — Mở rộng n=3 → n=5 (đồng bộ lực kiểm định với EarVN1.0)

```bash
bash pipeline_awex/run_multi_seed_extra_seeds.sh
```

Chỉ thêm seed 44/999 cho recognition (không train lại SR).

---

## Bước 3 — Đo phương sai do CHÍNH seed train SR

Bước 1 chỉ train SR (`span_tiny`) đúng 1 lần (seed=42) rồi tái sử dụng cho
mọi seed downstream (xem mục 10.5 `RUNBOOK_EarVN1.0.md` để biết lý do). Script
dùng CHUNG cho cả 2 dataset (tham số hoá qua `--config`), chỉ cần trỏ đúng
config AWEx:

```bash
bash pipeline/run_sr_seed_variance.sh configs/config_awex.yaml
```

Kết quả: `results_awex/sr_seed_variance/sr_seed_variance_summary.csv` +
`_sr_quality.csv` — so sánh phương sai do SR-seed với phương sai downstream
đã có ở Bước 2. Với AWEx (ít dữ liệu/người hơn EarVN1.0), phương sai do
SR-seed có khả năng LỚN hơn tương đối — đáng chạy bước này trước khi chốt
số liệu chính thức cho dataset thứ 2.

---

## Bước 4 — Transfer learning từ EarVN1.0 (kết quả CHÍNH)

**Vì sao transfer learning là số liệu chính, không phải from-scratch (Bước
1)**: AWEx vẫn là dataset nhỏ (~11-12 ảnh/người) — SR/recognition
train-from-scratch có phương sai lớn giữa seed. Fine-tune từ checkpoint đã
học đặc trưng tai trên EarVN1.0 (164 người, nhiều ảnh/người hơn) ổn định
hơn hẳn — đã xác nhận qua dữ liệu AWE cũ: SR from-scratch trên AWE chỉ
23.976dB, trong khi transfer learning cho 27.045dB (zero-shot) → 27.218dB
(fine-tuned).

**Yêu cầu tiền đề**: EarVN1.0 đã có đủ checkpoint recognition n=5 seed
(`runs/recognition_<domain>_<backbone>_seed<seed>/best.pt` với domain =
lr/sr_baseline/sr_improved, 5 backbone, seed = 42/123/2024/44/999) và
`runs_awex/sr_span_official/best.pt` + `runs_awex/recognition_hr_mobilenet_v2/best.pt`
từ Bước 1.

```bash
bash scripts/run_transfer_learning.sh awex earvn1
```

Điều phối 6 bước: (1) zero-shot SR cho cả `span_tiny` VÀ `span_baseline`
(áp checkpoint nguồn thẳng lên ảnh đích, không train lại), (2) fine-tune SR
tiny, (3) fine-tune SR baseline (**quan trọng — so sánh công bằng**: cả 2
model đều được fine-tune, không chỉ `span_tiny` — nếu chỉ fine-tune
`span_tiny` mà để `span_baseline` train-from-scratch thuần, so sánh sẽ
thiên vị `span_tiny` một cách giả tạo), (4) eval chất lượng SR fine-tuned,
(5) sinh lại ảnh SR từ 2 checkpoint fine-tune vào domain riêng
(`sr_baseline_transfer`/`sr_improved_transfer`, KHÔNG đè domain from-scratch
cũ), (6) fine-tune nhận diện trên ảnh MỚI, mặc định n=5 seed.

Kết quả: `results_awex/sr_quality_transfer.csv` (4 dòng: zero-shot/fine-tuned
x tiny/baseline), `results_awex/multi_seed_transfer/multi_seed_summary_transfer.csv`
+ `_pairwise.csv`.

**Không có "zero-shot" cho recognition** (chỉ SR mới có): 164 người EarVN1.0
và ~336 người AWEx là 2 tập người hoàn toàn khác nhau — không có ý nghĩa
khi áp thẳng bộ phân loại của tập người này lên tập người khác.
`identity_head` bắt buộc học lại (fine-tune).

---

## Bước 5 — (Tuỳ chọn, sau khi có kết quả Bước 4) Freeze-backbone — giảm phương sai

**Bối cảnh**: kết quả AWE cũ (n=5, transfer learning) cho thấy toàn bộ 15
phép so sánh lr/sr_baseline_transfer/sr_improved_transfer (5 backbone) đều
KHÔNG có ý nghĩa thống kê (p_bonferroni ≥ 0.13, hầu hết ~1.0) — độ lệch
chuẩn giữa seed gần bằng chênh lệch trung bình, do dữ liệu quá ít. Chỉ chạy
bước này sau khi đã xem kết quả Bước 4 và thấy phương sai giữa seed vẫn lớn:

```bash
bash scripts/run_transfer_learning_frozen.sh awex earvn1
```

**Cơ chế** (`--freeze_backbone` trong `train_recognition.py`, chỉ có tác
dụng khi dùng CÙNG `--init_ckpt_transfer`): đóng băng toàn bộ
`model.features` (backbone trích đặc trưng, đã học từ EarVN1.0 qua
checkpoint nguồn), chỉ train `embedding` + `identity_head` + `gender_head`
— giảm mạnh số tham số phải học từ dữ liệu ít, kỳ vọng giảm phương sai giữa
seed. `model.features.eval()` được ép lại đúng sau mỗi epoch train (kể cả
khi model đổi device do fallback OOM) — nếu không, `running_mean`/
`running_var` của BatchNorm vẫn tiếp tục trôi theo minibatch nhỏ dù affine
weight/bias không cập nhật (lỗi âm thầm, không exception nào báo hiệu). Tái
sử dụng checkpoint SR đã fine-tune và ảnh đã sinh sẵn ở Bước 4 (không train
lại SR, không sinh lại ảnh — đảm bảo so sánh frozen vs không-frozen công
bằng, cùng 1 bộ ảnh SR).

**Không phá protocol EarVN1.0**: không đụng `hr_source_min`, không đụng
splits, không đổi seed set/domain — freeze-backbone chỉ là tinh chỉnh TRONG
nhánh transfer-learning vốn đã riêng cho dataset đích (EarVN1.0 là nguồn,
không cần/không thể áp ngược cờ này).

Kết quả ghi vào thư mục RIÊNG:
`results_awex/multi_seed_transfer_frozen/multi_seed_summary_transfer_frozen.csv`
+ `_pairwise.csv` — so trực tiếp `std_identity_accuracy`/`p_bonferroni` với
file Bước 4 (không-freeze) để xem freeze có thực sự giảm phương sai/tăng ý
nghĩa thống kê hay không — không giả định trước, phải chạy thật để biết.

---

## Cách ly dữ liệu giữa 2 dataset (đã kiểm chứng, không còn rủi ro)

`scripts/setup_second_dataset_pipeline.sh` sed-thay TOÀN BỘ đường dẫn
`runs/`, `splits/`, `results/` (cả dạng có `/` theo sau lẫn dạng gán biến
trần `="results"`) trong mọi script copy sang `pipeline_awex/` — đã audit
thực nghiệm (chạy thật, quét độc lập không dùng lại logic kiểm tra nội bộ
của chính script) xác nhận SẠCH, 0 dấu vết đường dẫn chưa gắn hậu tố
`_awex` còn sót. `scripts/run_transfer_learning.sh` (rủi ro cao nhất vì chủ
đích đọc từ CẢ 2 dataset) — mọi điểm GHI (`--out_json`/`--out_csv`/
`--out_dir`) đều dùng biến đã gắn hậu tố `_awex`; điểm ĐỌC không hậu tố
(`runs/sr_improved_span_tiny/best.pt`, `runs/recognition_..._seed<seed>/best.pt`)
là ĐÚNG THIẾT KẾ (đọc checkpoint NGUỒN từ EarVN1.0 để fine-tune sang AWEx).

**Việc quan trọng đã xoá hẳn**: `pipeline/run_all_after_fix.sh` (script
tiện ích cũ, chứa nhiều lệnh `cp`/`rm -rf` thao tác trực tiếp lên
`results/multi_seed`, `results/lambda_sweep`... KHÔNG ăn theo dataset nào —
nếu bị chạy nhầm dưới `pipeline_awex/` sẽ XOÁ THẬT dữ liệu `results/` của
EarVN1.0). Không còn tồn tại trong `pipeline/`, không bị copy sang
`pipeline_<dataset>/` nào trong tương lai.

---

## Vì sao bỏ AWE (dataset gốc), chuyển hẳn sang AWEx

AWE gốc chỉ 100 identity x ~10 ảnh/người → chia 70/15/15 chỉ còn ~5.7
ảnh train/người cho bước recognition — quá ít, multi-seed (n=3) cho std
0.15-0.21 trên mean chỉ 0.05-0.4 (đổi seed, kết quả có thể nhảy từ 7% lên
39%). AWEx (336 người, gộp AWE+CVLE+New images) có nhiều dữ liệu hơn đáng
kể, giảm bớt vấn đề này — dù vẫn cần Bước 3 (đo phương sai SR-seed) và Bước
5 (freeze-backbone) để xử lý phần còn lại của vấn đề phương sai cao.

---

## Sau mỗi bước — gửi lại file gì

| Sau bước | Gửi lại |
|---|---|
| 0 (dry_run) | output dry_run trên chat (không cần file) |
| 1+2 | `splits_awex/dataset_stats.csv`, `results_awex/sr_quality.csv`, `results_awex/multi_seed/multi_seed_summary_pairwise.csv` |
| 3 | `results_awex/sr_seed_variance/sr_seed_variance_summary.csv` |
| 4 | `results_awex/sr_quality_transfer.csv`, `results_awex/multi_seed_transfer/multi_seed_summary_transfer_pairwise.csv` |
| 5 | `results_awex/multi_seed_transfer_frozen/multi_seed_summary_transfer_frozen_pairwise.csv` |

Không cần zip toàn bộ `results_awex/` mỗi lần — các file CSV trên là đủ để
đánh giá, nhẹ hơn nhiều so với gửi cả checkpoint.

---

## Lưu ý — mọi bản vá chung áp dụng cho cả AWEx

Toàn bộ 20 lỗi đã phát hiện/vá liệt kê trong `RUNBOOK_EarVN1.0.md` (mục
"Danh sách lỗi đã phát hiện và vá") áp dụng CHUNG cho cả AWEx, vì
`pipeline_awex/` chỉ là bản sao sed-path của `pipeline/` gốc — không có
logic riêng. Đặc biệt lưu ý mục #14/16/17 (split, NaN gradient,
RandomHorizontalFlip) đều BẮT BUỘC train lại từ đầu nếu `pipeline_awex/`
đã được tạo TRƯỚC khi các bản vá đó tồn tại — xoá `configs/config_awex.yaml`
+ `pipeline_awex/` rồi chạy lại `RUN_ALL_NEW_DATASET.sh awex 336` để tạo bản
mới, tự động kế thừa toàn bộ bản vá từ `pipeline/` nguồn.
