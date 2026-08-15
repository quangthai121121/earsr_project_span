# CHANGELOG v6 — Sửa để so sánh CÔNG BẰNG span_tiny vs span_baseline

Nối tiếp `CHANGELOG_v5.md` (giữ nguyên toàn bộ nội dung đợt 1 + đợt 2 ở đó).
File này chỉ ghi thêm **đợt 3**.

## Đợt 3 — Fine-tune cả span_baseline (không chỉ span_tiny), sinh lại ảnh SR

**Lỗi ở bản v5**: script `run_transfer_learning.sh` chỉ fine-tune `span_tiny`
từ checkpoint EarVN1.0; `span_baseline` vẫn dùng bản train-from-scratch trên
AWE. Kết quả: so sánh `span_tiny` (được lợi transfer learning) với
`span_baseline` (không được lợi gì) là **không công bằng** — lý giải trực
tiếp hiện tượng `span_tiny` "vượt" `span_baseline` ở AWE nhưng KHÔNG xảy ra
ở EarVN1.0 (nơi cả 2 đều được train đầy đủ trên cùng lượng dữ liệu).

**Đã sửa, theo đúng yêu cầu: EarVN1.0 làm sao thì làm lại y hệt cho dataset
mới** — cả `span_tiny` VÀ `span_baseline` đều được fine-tune từ checkpoint
tương ứng của dataset nguồn.

### File thay đổi

- **`train_sr.py`** (script train `span_baseline`/`edsr` — KHÁC với
  `train_sr_distill.py` dùng cho `span_tiny`, đã đọc lại source thật để xác
  nhận trước khi vá, không đoán): thêm `--run_suffix` (chưa từng có ở bản
  gốc). **Bắt buộc phải có** — nếu không, fine-tune `span_baseline` sẽ ghi
  đè thẳng lên `runs_<đích>/sr_span_official/best.pt`, phá hỏng checkpoint
  đang dùng làm `teacher_ckpt` cho việc train `span_tiny` và làm nguồn sinh
  ảnh domain `sr_baseline` gốc. Cờ `--pretrained_path` (đã có sẵn từ trước)
  tái sử dụng đúng cho việc nạp checkpoint dataset nguồn — đã đọc
  `models/span_official_wrapper.py` để xác nhận cơ chế nạp tự nhận diện
  đúng định dạng checkpoint (không có/có tiền tố `params_ema`), hoạt động
  chính xác với checkpoint tự train của project (không chỉ checkpoint gốc
  tác giả).

- **`scripts/run_transfer_learning.sh`** (viết lại đáng kể, 4 bước → 6
  bước):
  1. Zero-shot SR cho **cả 2** model (tiny + baseline).
  2. Fine-tune SR tiny (giữ nguyên như v5).
  3. **[MỚI]** Fine-tune SR baseline (`train_sr.py --pretrained_path ...
     --run_suffix ...`).
  4. Eval chất lượng SR fine-tuned cho cả 2 → `sr_quality_transfer.csv` giờ
     có **4 dòng** thay vì 2.
  5. **[MỚI]** Sinh lại ảnh SR (`data/build_sr.py`) cho cả train/val/test từ
     **chính 2 checkpoint vừa fine-tune** — lưu vào domain mới
     `sr_improved_transfer` và `sr_baseline_transfer` (KHÔNG ghi đè
     `sr_improved`/`sr_baseline` cũ, vốn sinh từ model from-scratch).
  6. Fine-tune nhận diện — **giờ train trên ảnh MỚI** (`sr_baseline_transfer`,
     `sr_improved_transfer`) thay vì ảnh cũ, cộng domain `lr` (không đổi, vì
     ảnh LR không phụ thuộc model SR nào nên không cần sinh lại).

### Kiểm chứng đã làm trước khi giao (không đoán)

- Đọc lại thật `train_sr.py`, `models/span_official_wrapper.py`,
  `data/build_sr.py` từ GitHub trước khi viết patch — xác nhận đúng CLI,
  đúng cơ chế nạp checkpoint, đúng cấu trúc thư mục ảnh output.
- Diff trực tiếp bản gốc (fetch trong chính phiên làm việc này, trước khi
  vá) với bản đã vá cho cả `train_sr.py` và `train_recognition.py` — xác
  nhận MỌI thay đổi chỉ là dòng thêm mới (cờ mặc định an toàn), không dòng
  thực thi cũ nào bị đổi hành vi khi không truyền cờ mới. Kết luận: chạy lại
  `pipeline/01`→`08` + `run_multi_seed.sh` gốc cho EarVN1.0 (dataset nguồn)
  với bộ file đã vá cho kết quả **giống hệt** bản chưa vá.
- Kiểm tra cú pháp Python (`py_compile`) và Bash (`bash -n`) cho toàn bộ
  file trước khi đóng gói.

### Quy ước đặt tên (đợt 3, thay thế phần tương ứng ở v5)

| Thành phần | Đường dẫn |
|---|---|
| Checkpoint SR baseline fine-tuned | `runs_<đích>/sr_span_official_finetuned_from_<nguồn>/best.pt` |
| Ảnh SR baseline fine-tuned | `splits_<đích>/sr_baseline_transfer/{train,val,test}/<person_id>/*.jpg` |
| Ảnh SR tiny fine-tuned | `splits_<đích>/sr_improved_transfer/{train,val,test}/<person_id>/*.jpg` |
| Checkpoint nhận diện (domain mới) | `runs_<đích>/recognition_sr_baseline_transfer_<backbone>_finetuned_from_<nguồn>_seed<seed>/best.pt` (tương tự cho `sr_improved_transfer`, `lr`) |
| Kết quả SR | `results_<đích>/sr_quality_transfer.csv` — 4 dòng: `span_tiny_zeroshot_from_<nguồn>`, `span_tiny_finetuned_from_<nguồn>`, `span_baseline_zeroshot_from_<nguồn>`, `span_baseline_finetuned_from_<nguồn>` |
| Kết quả nhận diện | `results_<đích>/multi_seed_transfer/multi_seed_summary_transfer.csv` — domain: `lr`, `sr_baseline_transfer`, `sr_improved_transfer` |

## File cần đặt lại (đè lên bản v5 đã đặt trước đó)

| File | Đặt vào |
|---|---|
| `train_sr.py` | `./train_sr.py` (file NÀY CHƯA từng gửi ở v5 — file mới cần thêm, không phải đè) |
| `scripts/run_transfer_learning.sh` | `./scripts/run_transfer_learning.sh` (đè bản v5) |

Các file khác của v5 (`train_sr_distill.py`, `train_recognition.py`,
`eval_recognition.py`, `utils/metrics.py`, `data/convert_generic_...py`,
`scripts/setup_second_dataset_pipeline.sh`, `scripts/make_finetune_config.py`)
**không đổi** so với v5 — không cần đặt lại nếu đã đặt từ trước.

## Chạy lại

Vì bước 5/6 sinh ảnh MỚI (không đè ảnh cũ) và bước 6/6 train recognition
trên domain MỚI (không đè checkpoint recognition cũ của bản v5), có thể chạy
thẳng lại toàn bộ script mà không cần dọn dẹp gì trước — kết quả v5 cũ
(domain `sr_baseline`/`sr_improved` cũ) vẫn còn nguyên, kết quả v6 mới nằm ở
domain `_transfer` riêng biệt.

```bash
grep -q "run_suffix" train_sr.py && echo "OK: train_sr.py đã patch"
grep -q "BƯỚC 6/6" scripts/run_transfer_learning.sh && echo "OK: run_transfer_learning.sh là bản v6"

bash scripts/run_transfer_learning.sh awe earvn1
```

## File kết quả cần gửi lại sau khi chạy (thay thế phần tương ứng ở v5)

- `results_awe/sr_quality_transfer.csv` (4 dòng)
- `results_awe/multi_seed_transfer/multi_seed_summary_transfer.csv` (domain: `lr`, `sr_baseline_transfer`, `sr_improved_transfer`)
