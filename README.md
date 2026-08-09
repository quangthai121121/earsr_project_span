# EarSR-Recognition: SPAN nén nhẹ cho nhận diện danh tính/giới tính qua ảnh tai

Khung thí nghiệm đầy đủ cho luận án: **nén** kiến trúc SPAN (Swift
Parameter-free Attention Network, Wan et al. 2023, arXiv:2311.12770 — đoạt
giải Nhất NTIRE 2024 Efficient SR Challenge) để giảm kích thước/tăng tốc độ
xử lý, áp dụng lên ảnh tai độ phân giải thấp (EarVN1.0), đo tác động lên
accuracy nhận diện danh tính (164 người) và giới tính.

**Tài liệu này mô tả ĐÚNG trạng thái code hiện tại** (đã áp dụng toàn bộ bản
vá lỗi phát hiện được trong quá trình phát triển) — làm theo đúng thứ tự dưới
đây, một dự án clone mới sẽ tái lập được kết quả tương tự.

---

## 0. Cài đặt môi trường

```bash
git clone <repo_url> earsr_project
cd earsr_project
pip install -r requirements.txt --break-system-packages
```

**Lưu ý quan trọng — xung đột tên package**: nếu môi trường đã cài
`datasets` (thư viện HuggingFace, rất phổ biến), Python có thể nhầm giữa
package đó và thư mục `datasets/` cục bộ của project này. Project đã có sẵn
file `__init__.py` rỗng trong `datasets/`, `models/`, `utils/`, `data/`,
`pipeline/` để tránh xung đột — không cần làm gì thêm, nhưng nếu vẫn gặp lỗi
`ModuleNotFoundError`, chạy với `PYTHONPATH=.` đứng trước lệnh Python:

```bash
PYTHONPATH=. python <script>.py ...
```

---

## 1. Chuẩn bị dữ liệu EarVN1.0

```bash
# Copy dữ liệu gốc EarVN1.0 (thư mục con là NNN.Ten_nguoi) vào:
#   raw_data/EarVN1.0_raw/

bash pipeline/01_survey_and_prepare_data.sh
```

**Việc script này làm** (`data/prepare_splits.py --auto` + `data/build_lr.py`):

1. Khảo sát phân phối kích thước trên toàn bộ ảnh gốc (28.412 ảnh, 164 người).
2. Tự động chọn ngưỡng lọc `hr_source_min` sao cho giữ được nhiều dữ liệu nhất
   (từ 80% trở lên) mà không mất identity nào — với EarVN1.0 gốc, kết quả là
   **41px**.
3. Lọc + resize ảnh đạt ngưỡng về **80x80** bằng kỹ thuật **Letterbox** (giữ
   tỷ lệ khung hình gốc, đệm viền đen) → tập **HR**.
4. Downsample bicubic HR xuống **20x20** (scale=4) → tập **LR**.
5. **Chia train/val/test theo ẢNH trong từng người** (KHÔNG chia theo người) —
   bắt buộc vì bài toán là closed-set classification: nếu chia theo người, một
   số người sẽ hoàn toàn vắng mặt ở train, khiến model không thể học nhận ra
   họ (đã từng gặp lỗi này — accuracy nhận diện mãi mãi ~0%, xem
   `data/prepare_splits.py::split_by_identity()`).
6. Xuất `splits/dataset_stats.csv` (thống kê đầy đủ cho báo cáo) và
   `splits/real_lr_holdout.json` (~173 ảnh vốn dĩ đã nhỏ, dùng kiểm chứng bổ
   sung ở Bước 8, không dùng để train).

**Kết quả mong đợi** (khớp với `configs/config.yaml` mặc định):
`hr_source_min=41`, `hr_size=80`, Train=16.077 / Val=3.446 / Test=3.449 ảnh,
đủ 164/164 người ở cả 3 tập.

---

## 2. Tải kiến trúc + checkpoint SPAN chính thức

```bash
bash pipeline/02_setup_span_official.sh
```

Script này `git clone` repo `github.com/hongyuanyu/SPAN` vào `external/SPAN/`.
Sau đó **làm tay**:

1. Tải checkpoint pretrained (scale x4, ví dụ `spanx4_ch48.pth`) từ trang
   GitHub của tác giả, đặt đúng vào `checkpoints/span_pretrained_x4.pth`
   (chú ý đường dẫn — nhiều người hay copy nhầm vào thư mục con).
2. Không cần sửa file `external/SPAN/basicsr/archs/span_arch.py` — project
   đọc trực tiếp qua `models/span_official_wrapper.py` mà **không** import
   toàn bộ package `basicsr` (vốn có lỗi tương thích với torchvision mới do
   `torchvision.transforms.functional_tensor` đã bị xóa) — chỉ load đúng
   class `SPAN` cần dùng, kèm 1 lớp "giả lập" tối thiểu cho `ARCH_REGISTRY`.
3. Wrapper `SPANWithRescale` tự động vá lỗi chuẩn hóa: `forward()` gốc biến
   đổi ảnh theo `(x-mean)*255` nhưng thiếu bước biến đổi ngược ở output —
   wrapper tự thêm bước này để output luôn nằm trong `[0,1]`.

---

## 3. Train recognition baseline (domain `hr`, `lr`) — 5 backbone

```bash
bash pipeline/03_train_baseline_recognition.sh
```

Train **5 backbone x 2 domain = 10 model**: `mobilenet_v2`,
`mobilenet_v3_small`, `resnet18`, `efficientnet_b0`, `ghostnet_100`, mỗi cái
trên domain `hr` (train from-scratch) rồi domain `lr` (fine-tune từ
checkpoint `hr` cùng backbone).

**Lưu ý**: `ghostnet_100` cần tải pretrained weights từ HuggingFace Hub lần
đầu — có thể chậm nếu mạng yếu (không cần `HF_TOKEN`, chỉ giới hạn tốc độ).

---

## 4. Train EDSR teacher + SPAN baseline

```bash
bash pipeline/04_train_teacher_and_span_baseline.sh
```

1. Train **EDSR** (teacher nặng, dùng để đối chiếu chất lượng, KHÔNG dùng làm
   teacher cho student cuối — xem Bước 6) bằng pixel loss (L1) thuần túy.
2. Fine-tune **SPAN chính thức** (checkpoint pretrained từ Bước 2) trên
   EarVN1.0, cũng bằng pixel loss thuần túy → đây là **SPAN baseline**, mốc so
   sánh chính cho toàn bộ luận án.
3. Sinh tập ảnh `splits/sr_baseline/` (chạy LR qua SPAN baseline vừa train).

---

## 5. Train recognition trên domain `sr_baseline`

```bash
bash pipeline/05_train_recognition_sr_baseline.sh
```

Fine-tune tiếp từ checkpoint domain `lr` (đúng chuỗi `hr -> lr -> sr_baseline`)
cho cả 5 backbone.

---

## 6. Nén SPAN (`span_tiny`) — đóng góp chính của luận án

```bash
bash pipeline/06_improve_span.sh
```

**Kiến trúc `span_tiny`** (tự thiết kế, trong `models/sr_models.py`): giữ ý
tưởng "parameter-free attention" của SPAN gốc, nhưng bỏ kỹ thuật
reparameterization (Conv3XC — lưu song song 2 nhánh trọng số train/eval) và
hệ số khuếch đại kênh nội bộ (`gain=2`) → **0.875M tham số (giảm 60.9% so với
SPAN baseline 2.237M)**.

**Cách train**: distillation từ **SPAN baseline** (không phải EDSR) làm
teacher, kết hợp identity-aware loss từ model recognition đã train trên
domain `hr` làm "giám khảo":

```
L_total = lambda_pixel . L1(SR, HR) + lambda_distill . L1(SR, teacher) + lambda_identity . (1 - cos(f(SR), f(HR)))
```

Trọng số hiện tại trong `configs/config.yaml`: `lambda_pixel=1.0,
lambda_distill=1.0, lambda_identity=0.1` — **đang trong quá trình kiểm định
lại kỹ hơn** bằng `pipeline/run_lambda_sweep.sh` (xem mục "Thí nghiệm bổ
sung" bên dưới), vì ablation ban đầu cho thấy cấu hình này chưa hẳn tối ưu.

**Quan trọng — KHÔNG dùng AMP (mixed precision) cho bước train này**: giá trị
nội bộ của SPAN (do phép `(x-mean)*255`) rất dễ tràn số dưới fp16, đặc biệt
khi chuỗi qua 3 model (student->teacher->recognition) trong 1 lần forward —
đã quan sát thấy gây NaN hàng loạt khi bật AMP. `train_sr_distill.py` cố định
chạy fp32 kèm gradient clipping (`max_norm=1.0`).

Script cũng tự sinh tập ảnh `splits/sr_improved/` từ `span_tiny` vừa train.

---

## 7. Train recognition trên domain `sr_improved`

```bash
bash pipeline/07_train_recognition_sr_improved.sh
```

Tương tự Bước 5, fine-tune từ checkpoint domain `lr`.

---

## 8. Benchmark tổng hợp — bảng kết quả chính

```bash
bash pipeline/08_benchmark_and_aggregate.sh
```

Chạy toàn bộ ma trận test (backbone x domain), đo PSNR/SSIM/params/FLOPs/
latency cho 3 model SR (EDSR, SPAN baseline, span_tiny — **đã vá lỗi đo
latency**, xem ghi chú dưới), eval trên `real_lr_holdout`, và tổng hợp toàn
bộ log training. Kết quả nằm trong `results/`:

| File | Nội dung |
|---|---|
| `results/summary.csv` | Bảng chính: accuracy theo backbone x domain |
| `results/sr_quality.csv` | PSNR/SSIM/params/FLOPs/latency của 3 model SR |
| `results/real_lr_holdout.csv` | Accuracy trên ảnh LR thật (không phải downsample nhân tạo) |
| `results/training_summary.csv` | Epoch dừng, early-stop, OOM, thời gian train mỗi lần chạy |

**Ghi chú quan trọng về latency**: `utils/metrics.py::freeze_reparam_modules()`
tự động "đóng băng" cơ chế reparameterization của SPAN chính thức (module
`Conv3XC` vốn tính lại phép hợp nhất trọng số MỖI LẦN `forward()`, kể cả eval
mode — rất lãng phí khi đo lặp lại 100 lần). Nếu không có bản vá này, SPAN
baseline sẽ bị đo latency chậm giả tạo gấp ~4 lần so với thực tế.

---

## 9. Xuất ảnh so sánh trực quan (tuỳ chọn, hỗ trợ báo cáo)

```bash
python export_sr_comparison_images.py --config configs/config.yaml \
    --sr_baseline_ckpt runs/sr_span_official/best.pt --sr_baseline_arch span_official \
    --sr_improved_ckpt runs/sr_improved_span_tiny/best.pt --sr_improved_arch span_tiny \
    --n_samples 20 --out_dir results/sr_comparison_images
```

Xuất ảnh ghép 4 phần (LR bicubic | SPAN baseline | span_tiny | HR gốc) kèm
PSNR — dùng minh họa định tính bên cạnh số liệu định lượng.

---

## 10. Thí nghiệm bổ sung — kiểm chứng độ tin cậy (cho journal Q1)

Ba bước sau **không bắt buộc** để có bảng kết quả chính (Bước 8 đã đủ), nhưng
**bắt buộc** nếu mục tiêu là công bố journal Q1 — vì 1 lần chạy không đủ bằng
chứng thống kê để khẳng định kết luận.

### 10.1 Ablation — cô lập tác dụng từng thành phần loss

```bash
bash pipeline/run_ablation.sh
```

Chạy 4 cấu hình (`pixel_only`, `pixel_distill`, `pixel_identity`, `full`) trên
1 backbone đại diện (`mobilenet_v2`). Kết quả: `results/ablation.csv`.

**Phát hiện quan trọng cần biết trước khi dùng số liệu này cho bài báo**:
ablation ban đầu (1 lần chạy/cấu hình) cho thấy `pixel_distill` (chỉ
distillation, không identity) nhỉnh hơn `full` (cấu hình đang dùng) — đây là
lý do cần Bước 10.2 dưới đây trước khi chốt kết luận.

### 10.2 Quét lambda_identity + multi-seed — tìm cấu hình tối ưu có ý nghĩa thống kê

```bash
bash pipeline/run_lambda_sweep.sh
```

Quét 6 mức `lambda_identity` (0, 0.05, 0.1, 0.2, 0.3, 0.5) x 3 seed (42, 123,
2024) = 18 lần train student + recognition. Tự động tính `paired t-test` so
với mức `lambda_identity=0` (chỉ distillation). Kết quả:
`results/lambda_sweep/lambda_sweep_summary.csv`.

**Sau khi có kết quả**: cập nhật `lambda_identity` trong
`configs/config.yaml` (mục `sr_improve`) theo giá trị tối ưu tìm được, rồi
chạy lại Bước 6-8 với cấu hình đã chốt trước khi công bố số liệu chính thức.

### 10.3 Kiểm chứng độ ổn định — multi-seed cho bảng kết quả chính

```bash
bash pipeline/run_multi_seed.sh
```

Chạy lại 3 domain chính (`lr`, `sr_baseline`, `sr_improved`) với 3 seed độc
lập, **đúng theo chuỗi fine-tune** như pipeline chính (`hr -> lr ->
sr_baseline`/`sr_improved` — quan trọng: KHÔNG dùng chung 1 checkpoint `hr`
cho cả 3 domain, phải đi qua đúng domain `lr` trước, nếu không kết quả sẽ
không so sánh công bằng được với bảng chính). Kết quả:
`results/multi_seed/multi_seed_summary.csv`, kèm kiểm tra ý nghĩa thống kê
tự động (chênh lệch so với độ lệch chuẩn).

**Khuyến nghị cho journal**: hiện script chỉ chạy 1 backbone đại diện
(`mobilenet_v2`) và 3 seed. Để có độ tin cậy cao hơn, nên mở rộng
`BACKBONES` trong `pipeline/run_multi_seed.sh` thành vòng lặp cả 5 backbone,
và tăng số seed lên 5.

---

## Cấu trúc thư mục

```
earsr_project/
  raw_data/EarVN1.0_raw/          # dữ liệu gốc (tự copy vào)
  external/SPAN/                   # repo SPAN chính thức (tự clone qua Bước 2)
  checkpoints/span_pretrained_x4.pth  # checkpoint pretrained chính thức (tự tải qua Bước 2)
  configs/config.yaml               # toàn bộ hyperparameter, đường dẫn, ngưỡng lọc

  pipeline/
    01...08_*.sh                    # 8 bước chính, chạy tuần tự
    run_all.sh                      # chạy gộp cả 8 bước
    run_ablation.sh                 # Bước 10.1
    run_lambda_sweep.sh             # Bước 10.2
    run_multi_seed.sh               # Bước 10.3
    _update_config_from_report.py

  docs/                             # tài liệu phương pháp (điền số liệu sau khi chạy)
    01_data_preparation.md
    02_baseline_experiments.md
    03_span_improvement.md
    04_results_template.md

  data/
    prepare_splits.py               # khảo sát + chọn ngưỡng + letterbox + chia split (theo ẢNH)
    build_lr.py                     # downsample HR -> LR
    build_sr.py                     # chạy model SR: LR -> SR
    aggregate_results.py            # gộp JSON -> summary.csv (Bước 8)
    aggregate_ablation_results.py   # gộp JSON -> ablation.csv (Bước 10.1)
    aggregate_lambda_sweep.py       # gộp JSON -> lambda_sweep_summary.csv + t-test (Bước 10.2)
    aggregate_multi_seed_results.py # gộp JSON -> multi_seed_summary.csv (Bước 10.3)
    export_training_log_summary.py  # trích epoch/OOM/thời gian từ train.log -> CSV

  datasets/
    ear_dataset.py                  # Dataset đọc theo domain (hr/lr/sr_baseline/sr_improved)
    hrlr_pair_dataset.py            # Dataset đọc cặp (HR, LR), dùng chung cho train_sr*/eval_sr_quality

  models/
    sr_models.py                    # SPAN tự viết lại (span, span_tiny, span_large) + EDSR
    span_official_wrapper.py        # import SPAN CHÍNH THỨC, vá lỗi chuẩn hóa (SPANWithRescale)
    recognition_model.py            # factory 5 backbone, đo feat_dim động (tránh lệch metadata)

  scripts/setup_span_official.sh    # clone repo SPAN + hướng dẫn tải checkpoint

  utils/
    letterbox.py                    # resize giữ tỷ lệ, không méo hình tai
    metrics.py                      # accuracy, PSNR, SSIM, params, FLOPs, latency (đã vá đo công bằng)
    logger.py                       # log ra file + màn hình
    early_stopping.py
    device_manager.py                # tự fallback CPU khi GPU OOM
    seed.py

  train_sr.py                       # train SR thường (pixel loss) — SPAN baseline, EDSR
  train_sr_distill.py               # train span_tiny: distillation + identity loss (fp32, hỗ trợ --seed)
  train_recognition.py              # train recognition 1 domain + 1 backbone (hỗ trợ --seed, --init_ckpt)
  eval_recognition.py               # test checkpoint, hỗ trợ cross-domain
  eval_sr_quality.py                # đo PSNR/SSIM/FLOPs/latency cho 1 model SR
  eval_real_lr_holdout.py           # eval trên real_lr_holdout.json (LR thật)
  export_sr_comparison_images.py    # xuất ảnh so sánh trực quan (Bước 9)
```

---

## Danh sách lỗi đã phát hiện và vá trong quá trình phát triển

Ghi lại để người sau hiểu vì sao code có dạng như hiện tại — không phải thiết
kế thừa thãi, mà là bài học thực tế:

| # | Lỗi | Hậu quả nếu không vá | File vá |
|---|---|---|---|
| 1 | Chia split theo NGƯỜI thay vì theo ẢNH | `VAL_ID_ACC` mãi mãi ~0%, model không thể học | `data/prepare_splits.py` |
| 2 | `span_arch.py` gốc thiếu biến đổi ngược output | Output ngoài `[0,1]`, loss/PSNR sai hoàn toàn | `models/span_official_wrapper.py` |
| 3 | Import `basicsr` đầy đủ gây lỗi torchvision mới | `ModuleNotFoundError: functional_tensor` | `models/span_official_wrapper.py` |
| 4 | `count_params`/`feat_dim` GhostNet lệch metadata | `RuntimeError` shape mismatch khi train | `models/recognition_model.py` |
| 5 | AMP + identity loss (cosine similarity) dưới fp16 | NaN hàng loạt, PSNR giảm sâu | `train_sr_distill.py` (tắt AMP) |
| 6 | Learning rate quá cao sau khi tắt AMP | Loss tăng vọt rồi "chết" gradient qua `clamp` | `configs/config.yaml` (`lr: 1e-4`) |
| 7 | `Conv3XC` tính lại reparameterization mỗi `forward()` | Latency SPAN baseline bị đo chậm giả tạo ~4x | `utils/metrics.py` |
| 8 | Xung đột package `datasets/` cục bộ với HuggingFace | `ModuleNotFoundError` dù file đã đúng | `__init__.py` ở các thư mục con |
| 9 | Multi-seed dùng chung checkpoint `hr` thay vì đi qua `lr` | So sánh không công bằng, kết luận đảo ngược | `pipeline/run_multi_seed.sh` |

---

## Yêu cầu phần cứng

- GPU khuyến nghị (CPU vẫn chạy được nhờ `DeviceManager` tự fallback, nhưng
  chậm hơn nhiều).
- Dung lượng: `splits/hr` + `splits/lr` ~180MB (ảnh 80x80 + 20x20 số lượng
  lớn), checkpoint mỗi model recognition/SR vài MB đến vài chục MB.
- Thời gian: Bước 3 (10 model) và Bước 10.3 (multi-seed x 5 backbone nếu mở
  rộng) là các bước tốn thời gian nhất trong toàn bộ pipeline.
