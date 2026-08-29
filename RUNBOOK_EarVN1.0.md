# RUNBOOK — EarVN1.0

Hướng dẫn chạy đầy đủ, đúng thứ tự, cho toàn bộ pipeline nghiên cứu nén kiến
trúc SPAN (Swift Parameter-free Attention Network, Wan et al. 2023,
arXiv:2311.12770 — giải Nhất NTIRE 2024 Efficient SR Challenge) áp dụng lên
ảnh tai độ phân giải thấp EarVN1.0 (28.412 ảnh, 164 người), đo tác động lên
accuracy nhận diện danh tính/giới tính. File này gộp toàn bộ nội dung trước
đây rải rác ở `README.md`, `CHANGELOG_v7.md`, `pipeline/README.md`,
`docs/01-04*.md` — chỉ còn **file DUY NHẤT** cần đọc để chạy lại toàn bộ thí
nghiệm cho EarVN1.0 (xem `RUNBOOK_AWEx.md` cho dataset thứ 2).

**[MỚI] Chạy gộp toàn bộ Bước 1-13 bên dưới bằng 1 lệnh** (kèm log lịch sử
đầy đủ, xem chi tiết cơ chế trong chính file script):
```bash
bash pipeline/run_everything_from_scratch.sh --wipe-existing
```
Đây là script **điều phối lại** các bước 1-13 dưới đây (không thay thế nội
dung mô tả bên dưới) — vẫn nên đọc từng mục để hiểu RÕ từng bước làm gì
trước khi chạy gộp. **[LƯU Ý] File này chưa `git commit`** — chạy
`git add pipeline/run_everything_from_scratch.sh && git commit` nếu muốn
lưu lại cùng lịch sử code, tránh mất khi dọn dẹp working tree.

---

## 0. Cài đặt môi trường

```bash
git clone <repo_url> earsr_project
cd earsr_project
pip install -r requirements.txt --break-system-packages
```

**Xung đột tên package**: nếu môi trường đã cài `datasets` (thư viện
HuggingFace, rất phổ biến), Python có thể nhầm giữa package đó và thư mục
`datasets/` cục bộ của project này. Project đã có sẵn file `__init__.py`
rỗng trong `datasets/`, `models/`, `utils/`, `data/`, `pipeline/` để tránh
xung đột — nếu vẫn gặp `ModuleNotFoundError`, chạy với `PYTHONPATH=.` đứng
trước lệnh Python:
```bash
PYTHONPATH=. python <script>.py ...
```

---

## 1. Chuẩn bị dữ liệu EarVN1.0

```bash
# Copy dữ liệu gốc EarVN1.0 (thư mục con dạng NNN.Ten_nguoi) vào:
#   raw_data/EarVN1.0_raw/
bash pipeline/01_survey_and_prepare_data.sh
```

**Đặc điểm dữ liệu ảnh hưởng thiết kế**: độ phân giải không đồng nhất (một số
ảnh dưới 25×25px), vành tai không luôn nằm giữa khung (crop tự động), không
có cặp HR-LR thật đi kèm.

**Phương pháp luận** (`data/prepare_splits.py`):
1. Coi ảnh gốc là "HR tương đối" (không phải HR tuyệt đối) — quy ước phổ biến
   khi thiếu ground truth thật: chọn tập con đủ lớn làm "HR", tự tạo LR bằng
   downsample bicubic. Cần nêu rõ giả định này trong Limitations.
2. **Phân nhóm theo ngưỡng kích thước**: khảo sát phân phối kích thước toàn
   dataset (`--survey_only`, in percentile p10/p25/p50/p75/p90 cạnh ngắn), tự
   động chọn `hr_source_min` (ảnh ≥ ngưỡng → dùng làm HR-source) và
   `too_small_max` (ảnh < ngưỡng → gác riêng, `real_lr_holdout.json` — LR
   *thật* ngoài đời, dùng kiểm chứng bổ sung Bước 8, **không** dùng để train).
   Ảnh quá nhỏ không dùng làm HR vì phóng to sẽ tạo chi tiết giả (nội suy bịa
   ra), khiến model học sai lệch.
3. **Letterbox thay vì resize thẳng** (`utils/letterbox.py`): giữ nguyên tỷ
   lệ khung hình, resize cạnh dài nhất = kích thước mục tiêu, pad nền cho đủ
   hình vuông — resize thẳng sẽ kéo dãn/méo hình dạng vành tai (đặc trưng
   quan trọng cho nhận diện). Áp dụng nhất quán cho mọi ảnh trước khi tạo LR.
4. **Chia train/val/test theo ẢNH trong từng người** (KHÔNG chia theo người)
   — bài toán là closed-set classification (softmax cố định N=164 lớp); nếu
   chia theo người, một số người vắng mặt hoàn toàn ở train khiến model
   không thể học nhận ra họ (từng gặp lỗi này — accuracy nhận diện mãi mãi
   ~0%, xem `data/prepare_splits.py::split_by_identity()`). Tỷ lệ 70/15/15
   theo SỐ ẢNH (không phải số người); identity <3 ảnh sau lọc ngưỡng là ngoại
   lệ không tránh được (toàn bộ rơi vào train), được liệt kê rõ trong log.
5. **Nhãn giới tính**: suy tự động từ số thứ tự thư mục (001-098=nam,
   099-164=nữ) — **đã xác minh khớp đúng** mô tả chính thức bài báo gốc công
   bố dataset (Hoang, V.N., *"EarVN1.0: A new large-scale ear images dataset
   in the wild"*, Data in Brief, 2019): "98 males and 66 females... first 98
   folders belong to male class". Không phải suy đoán. Chỉ ảnh hưởng
   `gender_accuracy` (độc lập hoàn toàn với `identity_accuracy`).
6. Xuất `splits/dataset_stats.csv` (thống kê đầy đủ cho Bảng 1 — Dataset của
   bài báo) và `splits/real_lr_holdout.json`.

**Kết quả mong đợi** (khớp `configs/config.yaml` mặc định, đã xác nhận bằng
cách chạy lại thật `prepare_splits.py --auto` trên dữ liệu gốc):
`hr_source_min=41`, `hr_size=80`, Train=16.077 / Val=3.415 / Test=3.480 ảnh,
đủ 164/164 người ở cả 3 tập, percentile cạnh ngắn: min=13, p5=27, p10=32,
p20=41, p25=46, p50=77, p75=122, p90=161, max=472.

**Giả thuyết baseline cần kiểm chứng ở Bước 4** (động lực khoa học cho Bước
6): SPAN gốc (train bằng pixel loss L1 thuần, tối ưu PSNR) có thể cho
`identity_accuracy` **thấp hơn** baseline `lr_lr` — do pixel loss có xu
hướng làm mượt (over-smooth) mất chi tiết tần số cao (texture da, nếp gấp
sụn tai) quan trọng cho nhận diện, dù ảnh SR "đẹp hơn" bằng mắt/PSNR cao
hơn. Nếu đúng → động lực khoa học rõ ràng cho Bước 6. Nếu sai (SPAN gốc đã
thắng `lr_lr`) → vẫn tiếp tục Bước 6, nhưng đổi khung diễn giải sang "cải
thiện thêm biên độ" thay vì "sửa từ thua thành thắng" — vẫn là đóng góp hợp
lệ.

---

## 2. Tải kiến trúc + checkpoint SPAN chính thức

```bash
bash pipeline/02_setup_span_official.sh
```

`git clone` repo `github.com/hongyuanyu/SPAN` vào `external/SPAN/`. Sau đó
**làm tay**: tải checkpoint pretrained (scale x4, `spanx4_ch48.pth`) từ
GitHub tác giả, đặt vào `checkpoints/span_pretrained_x4.pth`.

Không cần sửa `external/SPAN/basicsr/archs/span_arch.py` — project đọc trực
tiếp qua `models/span_official_wrapper.py` mà **không** import toàn bộ
`basicsr` (có lỗi tương thích torchvision mới do
`torchvision.transforms.functional_tensor` đã bị xóa) — chỉ load đúng class
`SPAN`, kèm 1 lớp giả lập tối thiểu cho `ARCH_REGISTRY`. Wrapper
`SPANWithRescale` tự vá lỗi chuẩn hóa: `forward()` gốc biến đổi ảnh theo
`(x-mean)*255` nhưng thiếu bước biến đổi ngược ở output — wrapper tự thêm
bước này để output luôn trong `[0,1]`.

---

## 3. Train recognition baseline (domain `hr`, `lr`) — 5 backbone

```bash
bash pipeline/03_train_baseline_recognition.sh
```

Train **5 backbone x 2 domain = 10 model**: `mobilenet_v2`,
`mobilenet_v3_small`, `resnet18`, `efficientnet_b0`, `ghostnet_100` (dòng
kiến trúc chuyên cho lightweight face/biometric recognition, nền tảng của
GhostFaceNet) — mỗi cái trên domain `hr` (from-scratch) rồi `lr` (fine-tune
từ checkpoint `hr` cùng backbone).

**Quyết định fine-tune, không train from-scratch mọi domain**: EarVN1.0 chỉ
164 identity — không đủ lớn để train ổn định from-scratch nhiều backbone
trong thời gian hạn chế. Checkpoint tốt nhất luôn chọn theo **val set cùng
domain** đang train (không dùng val domain khác — tránh mismatch tiêu chí
chọn model với domain triển khai thực tế).

**Lưu ý**: `ghostnet_100` cần tải pretrained weights từ HuggingFace Hub lần
đầu — có thể chậm nếu mạng yếu (không cần `HF_TOKEN`).

---

## 4. Train EDSR teacher + SPAN baseline

```bash
bash pipeline/04_train_teacher_and_span_baseline.sh
```

1. Train **EDSR** (teacher nặng, đối chiếu chất lượng, KHÔNG dùng làm teacher
   cho student cuối — xem Bước 6) bằng pixel loss (L1) thuần.
2. Fine-tune **SPAN chính thức** (checkpoint pretrained Bước 2) trên
   EarVN1.0, cũng pixel loss thuần → **SPAN baseline**, mốc so sánh chính.
3. Sinh `splits/sr_baseline/` (LR qua SPAN baseline vừa train).

Dùng ma trận này để kiểm chứng giả thuyết ở Bước 1 — cột Δ (sr_baseline −
lr_lr) âm ở càng nhiều backbone → giả thuyết càng được củng cố, bằng chứng
định lượng cho Introduction/Motivation.

---

## 5. Train recognition trên domain `sr_baseline`

```bash
bash pipeline/05_train_recognition_sr_baseline.sh
```

Fine-tune tiếp từ checkpoint domain `lr` (đúng chuỗi `hr -> lr ->
sr_baseline`) cho cả 5 backbone.

---

## 6. Nén SPAN (`span_tiny`) — đóng góp chính của luận án

```bash
bash pipeline/06_improve_span.sh
```

**Kiến trúc `span_tiny`** (`models/sr_models.py`): giữ ý tưởng
"parameter-free attention" của SPAN gốc, bỏ reparameterization (Conv3XC —
lưu song song 2 nhánh trọng số train/eval), dùng `feat=48, n_blocks=3` (so
với SPAN baseline `feat=48, n_blocks=6` + Conv3XC) → **0.230M tham số, giảm
46.0% so với SPAN baseline đo ở chế độ DEPLOY (0.4263M — khớp 99.93% số liệu
SPAN-S tác giả gốc, Table 1 arXiv:2311.12770)** — so sánh CÔNG BẰNG theo chế
độ deploy, không phải tổng tham số bao gồm nhánh train dư thừa (2.237M).

**Cách train**: distillation từ **SPAN baseline** (không phải EDSR) làm
teacher, kết hợp identity-aware loss từ model recognition đã train trên
domain `hr`:

```
L_total = lambda_pixel . L1(SR, HR) + lambda_distill . L1(SR, teacher) + lambda_identity . (1 - cos(f(SR), f(HR)))
```

Trọng số hiện tại (`configs/config.yaml`): `lambda_pixel=1.0,
lambda_distill=1.0, lambda_identity=0.0`. Sau khi chạy Bước 10.2 (lambda
sweep) + kiểm định thống kê: **có bằng chứng identity loss > 0 gây hại chất
lượng SR** — đã hạ `lambda_identity` về 0.0.

**KHÔNG dùng AMP cho bước train này**: giá trị nội bộ SPAN (`(x-mean)*255`)
dễ tràn số dưới fp16, đặc biệt khi chuỗi qua 3 model (student→teacher→
recognition) trong 1 lần forward — đã quan sát NaN hàng loạt khi bật AMP.
`train_sr_distill.py` cố định fp32 kèm gradient clipping (`max_norm=1.0`),
**và [SỬA — bug nghiêm trọng phát hiện qua review Q1] giờ CHẶN
`backward()`/`optimizer.step()` khi loss NaN/Inf** — trước đây optimizer vẫn
cập nhật trọng số bằng gradient NaN trước khi bị lọc khỏi log hiển thị (với
Adam, 1 lần cập nhật NaN làm nhiễm vĩnh viễn buffer m/v của tham số đó).

Script tự sinh `splits/sr_improved/` từ `span_tiny` vừa train.

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

Chạy toàn bộ ma trận test (backbone x domain), đo PSNR/SSIM/MACs/FLOPs/
latency cho 3 model SR (EDSR, SPAN baseline, span_tiny), eval trên
`real_lr_holdout`, tổng hợp toàn bộ log training.

| File | Nội dung |
|---|---|
| `results/summary.csv` | Bảng chính: accuracy (identity top-1/rank-5, AUC/EER verification, gender) theo backbone x domain |
| `results/sr_quality.csv` | PSNR/SSIM/LPIPS/params/**macs_G/flops_G**/latency của các model SR |
| `results/real_lr_holdout.csv` | Accuracy trên ảnh LR thật (không phải downsample nhân tạo) — chỉ kiểm chứng phụ, KHÔNG dùng làm bảng accuracy chính (identity ở đây không tách khỏi identity trong train — cùng người, chỉ khác ảnh) |
| `results/training_summary.csv` | Epoch dừng, early-stop, OOM, thời gian train mỗi lần chạy |
| `results/<backbone>_<domain>_confusion.csv` | Ma trận nhầm lẫn identity (164x164), 1 file/lần chạy `eval_recognition.py` |

**Latency**: `utils/metrics.py::freeze_reparam_modules()` tự "đóng băng" cơ
chế reparameterization của SPAN chính thức (`Conv3XC` vốn tính lại phép hợp
nhất trọng số MỖI LẦN `forward()`, kể cả eval mode) — không có bản vá này,
SPAN baseline bị đo latency chậm giả tạo ~4x.

**MACs/FLOPs [SỬA — lỗi nhãn phát hiện qua review Q1]**: `count_flops()`
(`utils/metrics.py`) trước đây trả về 1 số bị gọi NHẦM là "FLOPs" nhưng thực
ra là MACs (`thop.profile()` trả MACs, không phải FLOPs — đã kiểm chứng bằng
tay: Conv2d 1x1 trên input 1x1x2x2 có MACs=4 đúng hand-calc, còn FLOPs=2xMACs
phải là 8, và `thop` trả đúng 4.0). Với bài báo "efficient SR" trích dẫn
NTIRE (RLFN/ECBSR/SAFMN/SMFANet công bố số liệu riêng), lệch 2x giữa
MACs/FLOPs là sai số nghiêm trọng khi đối chiếu literature — giờ CSV có cả 2
cột `macs_G` và `flops_G` (=2×macs_G), tự đối chiếu đúng quy ước từng
baseline được trích dẫn thay vì đoán.

**Accuracy micro-average [SỬA — bug phát hiện qua review Q1]**:
`eval_recognition.py`/`eval_real_lr_holdout.py` trước đây tính accuracy bằng
macro-average theo BATCH (cộng accuracy từng batch rồi chia số batch) — sai
với true top-1 accuracy khi test set không chia hết cho batch_size (batch
cuối nhỏ hơn vẫn được tính trọng số ngang bằng batch khác). Đã sửa sang
micro-average đúng nghĩa (tổng dự đoán đúng / tổng số ảnh). Lệch nhỏ, cùng
hướng ở mọi domain nên so sánh TƯƠNG ĐỐI vẫn hợp lệ trước đây, nhưng số
tuyệt đối giờ mới đúng chuẩn.

---

## 9. Xuất ảnh so sánh trực quan (tuỳ chọn, hỗ trợ báo cáo)

```bash
python export_sr_comparison_images.py --config configs/config.yaml \
    --sr_baseline_ckpt runs/sr_span_official/best.pt --sr_baseline_arch span_official \
    --sr_improved_ckpt runs/sr_improved_span_tiny/best.pt --sr_improved_arch span_tiny \
    --n_samples 20 --out_dir results/sr_comparison_images
```

Xuất ảnh ghép 4 phần (LR bicubic | SPAN baseline | span_tiny | HR gốc) kèm
PSNR — minh họa định tính bên cạnh số liệu định lượng.

---

## 10. Thí nghiệm bổ sung — kiểm chứng độ tin cậy (bắt buộc cho journal Q1)

Không bắt buộc để có bảng kết quả chính (Bước 8 đã đủ), nhưng bắt buộc nếu
mục tiêu là journal Q1 — 1 lần chạy không đủ bằng chứng thống kê.

### 10.1 Ablation — cô lập tác dụng từng thành phần loss

```bash
bash pipeline/run_ablation.sh
```

Chạy 4 cấu hình (`pixel_only`, `pixel_distill`, `pixel_identity`, `full`)
trên 1 backbone đại diện (`mobilenet_v2`). Kết quả: `results/ablation.csv`.

Ablation ban đầu (1 lần chạy/cấu hình) cho thấy `pixel_distill` nhỉnh hơn
`full` — lý do cần Bước 10.2. **[CẢNH BÁO — phát hiện qua review Q1]**
`pixel_identity`/`full` dùng identity loss qua `build_judges()`, đọc
`configs/config.yaml::sr_improve.identity_judges` — hiện có 3 judge
(mobilenet_v2/resnet18/ghostnet_100), KHÔNG PHẢI single-judge mobilenet_v2
như ablation ĐẦU TIÊN. Không có cờ CLI ghi đè `identity_judges` — chạy lại
script này BÂY GIỜ cho số liệu KHÔNG so sánh trực tiếp được với ablation cũ
(cơ chế identity loss khác nhau). Đừng trộn 2 bộ số liệu vào cùng 1 bảng mà
không chú thích rõ.

**[MỚI — phát hiện qua review Q1] Trả lời câu hỏi "KD có thật sự giúp
`span_tiny` không?" bằng n=5 seed thật** (trước đây chỉ n=1, xu hướng quan
sát được `pixel_only > pixel_distill` KHÔNG có ý nghĩa thống kê ở n=1):

```bash
bash pipeline/run_ablation_multiseed.sh   # cần Bước 10.1 + 10.3 (extra seeds) đã xong
```

Tái sử dụng checkpoint SR đã có từ `run_ablation.sh` (KHÔNG train lại SR —
đúng quy ước "SR train 1 lần" xuyên suốt project), chỉ multi-seed bước
recognition — **[SỬA — số liệu sai phát hiện qua review Q1] 1 backbone đại
diện (`mobilenet_v2`) x 5 seed x 4 cấu hình**, KHÔNG PHẢI 5 backbone (giống
quy ước 1-backbone của `run_ablation.sh` gốc, không phải quy ước 5-backbone
của `run_multi_seed.sh`). Kết quả:
`results/ablation_multiseed/ablation_multiseed_summary_pairwise.csv` — đọc
cặp `pixel_distill` vs `pixel_only` để có câu trả lời KD có tác dụng hay
không kèm p-value + Cohen's d thật.

### 10.2 Quét lambda_identity + multi-seed

```bash
bash pipeline/run_lambda_sweep.sh
```

Quét 6 mức `lambda_identity` (0, 0.05, 0.1, 0.2, 0.3, 0.5) x 3 seed (42, 123,
2024) = 18 lần train student + recognition. Tính `paired t-test` (raw +
Bonferroni), **Cohen's d ghép cặp**, **Wilcoxon signed-rank** (robustness
check phi tham số, không giả định normality như t-test), **CI95%** cho
mean_diff, và **MDES** (Minimum Detectable Effect Size, dùng noncentral-t
chính xác — định lượng cụ thể "n=3 + Bonferroni thiếu power tới mức nào",
thay vì chỉ nói định tính). Kết quả:
`results/lambda_sweep/lambda_sweep_summary.csv`.

**QUY TRÌNH ĐÚNG sau khi có kết quả — [SỬA hướng dẫn cũ SAI, phát hiện qua
review Q1]**: KHÔNG chỉ đổi `lambda_identity` trong `configs/config.yaml`
rồi chạy thẳng Bước 10.3 (multi-seed) — `pipeline/run_multi_seed.sh` KHÔNG
train lại SR, chỉ tái sử dụng ảnh `splits/sr_improved` đã sinh sẵn (từ Bước
6, vốn PIN CỨNG `lambda_identity=0` qua CLI, không đọc default config.yaml).
Đổi config.yaml một mình KHÔNG có tác dụng gì. Quy trình đúng: (1) sửa
`lambda_identity` trong `configs/config.yaml`, (2) sửa CÙNG giá trị vào dòng
pin CLI trong `pipeline/06_improve_span.sh`, (3) chạy lại Bước 6 (train lại
SR + sinh lại `splits/sr_improved`), (4) rồi MỚI chạy Bước 10.3 trên ảnh SR
MỚI.

### 10.3 Kiểm chứng độ ổn định — multi-seed cho bảng kết quả chính

```bash
bash pipeline/run_multi_seed.sh
bash pipeline/run_multi_seed_extra_seeds.sh   # mở rộng n=3 -> n=5 (seed 44, 999)
```

Chạy lại 3 domain chính (`lr`, `sr_baseline`, `sr_improved`) với 5 seed độc
lập (42/123/2024/44/999), **đúng theo chuỗi fine-tune** (`hr -> lr ->
sr_baseline`/`sr_improved` — KHÔNG dùng chung 1 checkpoint `hr` cho cả 3
domain, phải đi qua đúng domain `lr` trước). 5 backbone x 5 seed x 3 domain.
Kết quả: `results/multi_seed/multi_seed_summary.csv` +
`multi_seed_summary_pairwise.csv` (paired t-test THẬT + Cohen's d + Wilcoxon
+ CI95% + MDES cho mọi cặp domain cùng backbone — file quan trọng nhất để
viết bảng kết quả có ý nghĩa thống kê).

**Lưu ý quan trọng về phạm vi phương sai đo được**: cả chuỗi multi-seed này
chỉ đổi seed ở bước RECOGNITION downstream — checkpoint SR (`span_tiny`) vẫn
CỐ ĐỊNH ở 1 seed duy nhất (42, xem Bước 10.5), và checkpoint `recognition_hr`
gốc (điểm khởi đầu của mọi domain) cũng chỉ train 1 lần (n=1) — phương sai đo
được ở đây là phương sai downstream, KHÔNG phải phương sai toàn chuỗi
end-to-end. Riêng so sánh CẶP domain (ví dụ `lr` vs `sr_improved`, cùng seed)
vẫn hợp lệ vì phần đầu chuỗi (hr) giống nhau, triệt tiêu trong phép trừ.

### 10.4 Ablation kiến trúc Tier-1 (`span_large`) — và 3 LOẠI so sánh cần phân biệt khi viết Methods

```bash
bash RUN_ALL_span_large_ablation.sh
bash RUN_ALL_span_large_ablation_extra_seeds.sh   # mở rộng n=5
```

**Mục đích `span_large`**: `span_official` có reparameterization (Conv3XC) +
concat fusion + `n_blocks=6`; `span_tiny` bỏ cả REP lẫn fusion VÀ cắt xuống
`n_blocks=3` — đổi 2 biến cùng lúc. `span_large` (`feat=48, n_blocks=6`, bỏ
REP+fusion) là điểm trung gian, tách bạch tác động từng thay đổi.

**3 LOẠI so sánh cần phân biệt** (tránh diễn giải nhân quả sai):

1. **`span_large` vs `span_tiny` — SẠCH, đúng 1 biến (độ sâu)**: cả 2 train
   qua `train_sr_distill.py`, CÙNG teacher/identity-loss/lambda — chỉ khác
   `n_blocks` (6 vs 3). Kết luận nhân quả về độ sâu RÚT RA ĐƯỢC.
2. **`span_official` vs `span_large`/`span_tiny` — KHÔNG sạch, mô tả/động
   lực, KHÔNG phải ablation nhân quả**: đổi CẢ kiến trúc (REP+fusion) LẪN
   recipe (pixel-loss thuần vs distillation). KHÔNG dùng để quy kết riêng cho
   REP/fusion — chỉ dùng làm số liệu tham chiếu/bối cảnh.
3. **`span_tiny` vs Track B (mục 10.6) — SẠCH, đúng 1 biến (họ kiến
   trúc)**: cùng recipe distillation/teacher/lambda, chỉ khác kiến trúc nền.

**Khi viết bài báo**: chỉ dùng cặp 1 và 3 cho phát biểu nhân quả "X tốt
hơn/kém hơn Y vì Z". Cặp 2 chỉ nên xuất hiện như tham chiếu/bối cảnh, kèm 1
câu giải thích không phải ablation kiểm soát biến.

### 10.5 [MỚI — phát hiện qua review Q1] Đo phương sai do CHÍNH seed train SR

Toàn bộ Bước 1-10.4 chỉ train SR (`span_tiny`/`span_official`/mọi baseline)
**đúng 1 lần** (seed=42 mặc định), rồi tái sử dụng ảnh SR cho MỌI seed
downstream — phương sai do CHÍNH seed train SR chưa từng được đo. Đây là
điểm reviewer Q1 rất dễ bắt bẻ. Script sau đo trực tiếp phương sai đó cho
`span_tiny` (1 backbone đại diện `mobilenet_v2`, giữ seed downstream cố định
để cô lập đúng biến SR-seed, PIN CỨNG recipe khớp Bước 6):

```bash
bash pipeline/run_sr_seed_variance.sh
```

Train thêm 2 checkpoint SR mới (seed 123, 2024 — seed 42 tái sử dụng
checkpoint sẵn có), fine-tune recognition tương ứng, so sánh trực tiếp `std`
do SR-seed với `std` do downstream-seed đã có ở `multi_seed_summary.csv`.
Kết quả: `results/sr_seed_variance/sr_seed_variance_summary.csv` (+
`_sr_quality.csv` — phương sai PSNR/SSIM của chính checkpoint SR theo seed,
bằng chứng trực tiếp nhất) — dùng để viết rõ hơn phần Limitations về giới
hạn n=1 cho SR training thay vì chỉ nêu suông.

### 10.6 So sánh với các kiến trúc SR nhẹ khác đã công bố (RLFN/ECBSR/SAFMN/SMFANet)

Ngoài so với chính "họ SPAN", để chứng minh `span_tiny` cạnh tranh được với
SR nhẹ nói chung, bổ sung baseline BÊN NGOÀI. Cả 5 đều port trực tiếp từ mã
nguồn/bài báo gốc (không đoán công thức, xem chú thích chi tiết trong
`models/sr_models.py`):

| Kiến trúc | Nguồn | Cấu hình dùng | ~Tham số |
|---|---|---|---|
| `rlfn` | Kong et al., NTIRE22 runtime-track winner, arXiv:2205.07514 | RLFN-cut nguyên văn: feat=48, n_blocks=4, esa_channels=16, ESA-pool kernel=7/stride=3 | 0.317M |
| `rlfn_adapted` | (dựa trên `rlfn`) | Y hệt `rlfn`, chỉ đổi ESA-pool kernel=3/stride=2 (tránh suy biến ở 20x20, xem dưới) | 0.317M |
| `ecbsr` | Zhang et al., ACM MM 2021 | m4c16: 4 khối ECB, 16 kênh, with_idt, prelu (RGB thay Y-channel gốc) | ~0.1M (deploy) |
| `safmn` | Sun et al., ICCV 2023, arXiv:2302.13800 | dim=36, n_blocks=8 (cấu hình demo mặc định repo chính thức) | 0.228M |
| `smfanet` | Zheng et al., **ECCV 2024**, DOI 10.1007/978-3-031-72973-7_21 | dim=24, n_blocks=8, ffn_scale=1.5 (khớp file nộp NTIRE2024_ESR) | ~0.1-0.2M (ước tính) |

**`rlfn` vs `rlfn_adapted`**: module ESA trong RLFB dùng `max_pool2d(kernel=7,
stride=3)` — ở ảnh LR 20x20 của project, attention map suy biến về 1x1
(truy vết: conv2 stride2 20x20→9x9, max_pool k7/s3 → 1x1), mất tính
"spatial", hoạt động gần giống channel-attention (Squeeze-and-Excitation).
`rlfn_adapted` (kernel=3/stride=2 → 9x9→4x4, không suy biến) khôi phục đúng
chức năng — **dùng làm baseline CHÍNH**. `rlfn` nguyên bản giữ song song để
đối chiếu literature + làm bằng chứng suy biến, không dùng làm baseline
chính. Nêu 1 câu trong Limitations giải thích 2 biến thể — thể hiện rà soát
cẩn thận. `ecbsr`/`safmn`/`smfanet` không gặp vấn đề suy biến tương tự
(`safmn` pool tối thiểu 20//8=2x2, `smfanet` cũng 2x2).

**Track A vs Track B — KHÔNG trộn lẫn (đổi khác 2 biến nếu dùng sai)**:
- **Track A** (`RUN_ALL_extra_sr_baseline.sh <arch>`): train qua
  `train_sr.py` (pixel loss L1 thuần) — ĐÚNG recipe chính RLFN/ECBSR/SAFMN
  tự dùng (không distillation) — đối chiếu literature, KHÔNG so sánh công
  bằng trực tiếp với `span_tiny` (khác recipe huấn luyện).
- **Track B** (`RUN_ALL_extra_sr_baseline_distilled.sh <arch>`, khuyến
  nghị): train qua `train_sr_distill.py --student_arch <arch>` — CÙNG
  teacher/identity-loss/lambda với `span_tiny` — cô lập đúng biến kiến trúc.
  **Dùng Track B cho bảng kết quả CHÍNH.**

```bash
# Track A (đối chiếu literature)
bash RUN_ALL_extra_sr_baseline.sh rlfn
bash RUN_ALL_extra_sr_baseline.sh rlfn_adapted
bash RUN_ALL_extra_sr_baseline.sh ecbsr
bash RUN_ALL_extra_sr_baseline.sh safmn
bash RUN_ALL_extra_sr_baseline.sh smfanet
bash RUN_ALL_extra_sr_baseline_extra_seeds.sh <arch>          # mở rộng n=5, mỗi kiến trúc

# Track B (bảng kết quả CHÍNH khi so sánh với span_tiny)
bash RUN_ALL_extra_sr_baseline_distilled.sh rlfn_adapted      # khuyến nghị — baseline chính
bash RUN_ALL_extra_sr_baseline_distilled.sh ecbsr
bash RUN_ALL_extra_sr_baseline_distilled.sh safmn
bash RUN_ALL_extra_sr_baseline_distilled.sh smfanet
bash RUN_ALL_extra_sr_baseline_distilled_extra_seeds.sh <arch>  # mở rộng n=5
```

Trước khi chạy: `python test_new_sr_archs.py` (vài giây, không cần dataset)
để tự xác nhận mọi kiến trúc SR chạy được.

---

## 11. LPIPS, verification AUC/EER + confusion matrix, Cohen's d/Wilcoxon/CI/MDES

Đã tích hợp sẵn vào mọi lần chạy `eval_sr_quality.py`/`eval_recognition.py`/
các script `aggregate_*.py` — không cần lệnh riêng:

- **LPIPS** (`utils/metrics.py::compute_lpips()`, backbone AlexNet): cột
  `lpips` trong `sr_quality.csv`, đo trên ROI thật. **LPIPS càng THẤP càng
  tốt** (ngược hướng PSNR/SSIM). Cần `pip install lpips`; nếu chưa cài, điền
  `NA`, không chặn pipeline.
- **AUC/EER (verification-setting)**:
  `compute_pairwise_genuine_impostor_scores()` lấy embedding toàn bộ test
  set, tính cosine similarity mọi cặp, tách genuine/impostor, rồi
  `compute_roc_auc_eer()` tính AUC (rank-based, chính xác) + EER (nội suy
  giao điểm FPR=FNR) — đã kiểm chứng bằng test số học tổng hợp độc lập
  trước khi merge. Cột `identity_auc`/`identity_eer` trong `summary.csv`,
  cảnh báo tự động nếu <30 cặp genuine.
- **Confusion matrix**: `compute_confusion_matrix()` — ma trận 164x164, lưu
  `results/<config_name>_confusion.csv` mỗi lần chạy `eval_recognition.py`.
- **Cohen's d ghép cặp (paired dz)**, **Wilcoxon signed-rank**, **CI95%**,
  **MDES** (noncentral-t chính xác): bổ sung bên cạnh p-value trong
  `aggregate_lambda_sweep.py`/`aggregate_multi_seed_results.py`/
  `aggregate_saliency_sweep.py` — p-value chỉ nói "có ý nghĩa hay không"
  (phụ thuộc cỡ mẫu), Cohen's d nói độ lớn thực tế, Wilcoxon là robustness
  check phi tham số (không giả định normality — quan trọng với n<=5 seed),
  CI95% cho người đọc tự đánh giá độ chắc chắn, MDES định lượng "thiết kế
  này CÓ THỂ phát hiện chắc chắn hiệu ứng lớn cỡ nào" ở power=80%.

---

## 12. Multi-Judge Ensemble Identity Loss + Feature-level KD (KD v2)

Bổ sung sau khi phát hiện `span_tiny` (Bước 6) cải thiện accuracy KHÔNG đồng
nhất giữa 5 backbone recognition — có ý nghĩa ở 3/5, không có bằng chứng ở
`resnet18`. 2 cơ chế mới trong `train_sr_distill.py` (tương thích ngược
100% — config cũ không có các khoá này tự fallback về hành vi trước đây):

1. **Multi-judge identity loss** (`sr_improve.identity_judges`): dùng đồng
   thời nhiều backbone recognition (mặc định mobilenet_v2+resnet18+
   ghostnet_100) làm giám khảo thay vì 1 model duy nhất — nhắm giả thuyết
   "identity loss cũ overfit vào gu của đúng 1 kiến trúc".
2. **Feature-level KD** (`sr_improve.lambda_feat`): khớp feature map trước
   pixel-shuffle giữa student/teacher qua `models/sr_models.py::SRFeatureHook`
   (forward hook theo TYPE module, hoạt động với cả `span_official` lẫn
   kiến trúc tự viết).

```bash
bash pipeline/run_ablation_kd_v2.sh              # 1) sàng lọc nhanh, 1 backbone/1 seed
# sửa LAMBDA_FEAT/LAMBDA_IDENTITY đầu file dưới đây theo cấu hình thắng ở trên
bash pipeline/run_multi_seed_kdv2.sh             # 2) validate n=3, 5 backbone
bash pipeline/run_multi_seed_kdv2_extra_seeds.sh # 3) mở rộng n=3 -> n=5 (seed 44,999)
```

**Trạng thái thực nghiệm (số liệu CSV thật đã chạy ở n=3, TRƯỚC khi mở rộng
n=5)**: `multi_seed_kdv2_summary_pairwise.csv` — so `sr_improved` (recipe
cũ) vs `sr_improved_kdv2` (mới) qua 5 backbone: chỉ 1/5 (`ghostnet_100`) có ý
nghĩa, và chỉ ở ngưỡng
lỏng α=0.10 (p_bonferroni=0.0807, không p<0.05). 4/5 backbone
p_bonferroni≈1.0 (nhiễu thuần); `mobilenet_v3_small` trend ÂM
(mean_diff=-0.0094). n=3 seed — chưa đủ power, chưa chạy n=5
(`run_multi_seed_kdv2_extra_seeds.sh`). **Kết luận hiện tại: KD v2 chưa
chứng minh được — nên đưa vào Discussion/Limitations dạng "đã thử, chưa có
bằng chứng rõ ràng" thay vì headline novelty**, trừ khi n=5 đổi kết quả.

---

## 13. Saliency-Weighted Identity-Critical Loss + Learned Block Pruning

Trả lời câu hỏi "novelty của `span_tiny` có phải chỉ là cắt bớt số khối SPAN
tay?" — 2 cơ chế không có trong bất kỳ công trình nào trích ở Related Work.

**13.1 Saliency-Weighted Identity-Critical Loss**
(`sr_improve.lambda_saliency`) — dùng gradient của hội đồng judge theo từng
pixel HR (`train_sr_distill.py::compute_multi_judge_saliency`) làm trọng số
không gian cho 1 pixel loss bổ sung, ép model ưu tiên tái tạo đúng vùng ảnh
hưởng nhiều nhất đến đặc trưng nhận dạng (nhắm phát hiện "PSNR cao do tái
tạo tốt tóc/nền chứ không phải tai"). Không cần nhãn segmentation mới, không
ảnh hưởng chi phí deploy.

```bash
bash pipeline/run_lambda_saliency_sweep.sh   # chạy SAU khi đã chốt KD v2 (mục 12)
```

**Trạng thái thực nghiệm**: `saliency_sweep_summary.csv` — mọi mức
`lambda_saliency` (0.15/0.3/0.6/1.0) đều có mean accuracy THẤP HƠN baseline
0.0, Cohen's d ÂM ở cả 4 điểm (-0.36 đến -2.13), tuy chưa có ý nghĩa thống
kê ở n=3. Cùng dạng thất bại như phát hiện `lambda_identity` đơn-judge
trước đó. **Kết luận hiện tại: trend âm, chưa chứng minh được** — cùng
khuyến nghị Discussion/Limitations như KD v2.

**13.2 Differentiable/Learned Block Pruning**
(`models/sr_models.py::SPANLearnedPrune`, script `train_sr_learned_prune.py`)
— thay vì chọn tay "giữ khối 1-3, bỏ khối 4-6" như `span_tiny`, mỗi khối
SPAB có 1 gate liên tục học được, huấn luyện CÙNG loss downstream + 1
sparsity penalty — khối không đóng góp bị tự động đẩy gate về 0. Sau train,
`harden_and_export()` xoá hẳn khối bị pruning, xuất model SPAN gọn thật sự.

```bash
bash pipeline/run_prune_sparsity_screen.sh        # 1) sàng lọc nhanh lambda_sparsity
bash pipeline/run_multi_seed_learned_prune.sh     # 2) validate đầy đủ, 5 backbone x n seed
```

**Trạng thái thực nghiệm (số liệu CSV thật, sweep đầy đủ với lambda_feat=0.5,
lambda_identity=0.1 pin theo cấu hình thắng KD v2)**: ngưỡng cắt khối thực
tế nằm ở ~0.13 (KHÔNG phải 0.1 như ước tính ban đầu) — `lambda_sparsity`
0.0-0.1 đều giữ nguyên 6/6 khối (PSNR 26.15-26.28, gate hoàn toàn trơ trong
dải này); 0.13→5 khối (26.165); 0.16→4 khối (25.004 — **sụt ~1.16dB chỉ
trong 1 bước**, lớn bất thường so với các bước khác ~0.1-0.4dB, đáng kiểm
tra khối nào bị cắt qua `prune_metadata.json`); 0.2→3 khối (24.629, **khớp
CHÍNH XÁC 0.230M tham số của `span_tiny`** — điểm so sánh công bằng nhất
cùng ngân sách). **Việc cần làm trước khi kết luận**: so trực tiếp
PSNR/SSIM của `span_tiny` (0.230M) với `prune_lsp0.2_nblocks3` (cũng
0.230M, cùng kích thước) — nếu learned pruning cho chất lượng THẤP hơn
đáng kể so với `span_tiny` ở cùng kích thước, đây là bằng chứng learned
pruning (có thêm feat-KD+identity loss) chưa vượt được chọn-tay ở cùng
ngân sách, cần xem lại trước khi coi đây là novelty chính. Lưu ý: so sánh
này có 2 biến (phương pháp chọn khối: học vs tay; VÀ recipe: có/không
feat-KD+identity) — không hoàn toàn sạch 1-biến, cần nêu rõ khi diễn giải.

---

## 14. Danh sách lỗi đã phát hiện và vá (lịch sử phát triển)

Ghi lại để người sau hiểu vì sao code có dạng như hiện tại:

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
| 10 | Thiếu chuẩn hoá ImageNet cho backbone pretrained | Giảm hiệu quả transfer learning, không lỗi runtime | `models/recognition_model.py` |
| 11 | Kernel nội suy domain `lr` không khớp nơi khác (BILINEAR ngầm định) | Không nhất quán tiền xử lý ảnh giữa domain | `datasets/ear_dataset.py` |
| 12 | PSNR/SSIM đo cả viền đệm đen letterbox | Số liệu bị thổi phồng, không so sánh được literature | `utils/metrics.py`, `eval_sr_quality.py`, `datasets/hrlr_pair_dataset.py` |
| 13 | Quét lambda_identity: 5 so sánh không hiệu chỉnh multiple-comparison | Tăng rủi ro báo sai "có ý nghĩa thống kê" | `data/aggregate_lambda_sweep.py` |
| 14 | `splits/hr`,`splits/lr` trên đĩa không khớp code split đã sửa (0% chồng lấp identity) | Mọi checkpoint recognition dựa trên split lỗi | `data/prepare_splits.py` + regenerate |
| 15 | `build_lr.py`/`build_sr.py` không xoá thư mục đích cũ trước khi ghi | Ảnh cũ lẫn ảnh mới, sai số liệu kiểm tra vật lý | `data/build_lr.py`, `data/build_sr.py` |
| 16 | NaN batch vẫn cập nhật trọng số (backward/step chạy trước khi lọc NaN) | Model nhiễm NaN vĩnh viễn (Adam m/v), không cảnh báo rõ | `train_sr_distill.py`, `train_sr_learned_prune.py` |
| 17 | `RandomHorizontalFlip` trên ảnh tai (không đối xứng như mặt) | Rủi ro phản biện biometrics | `datasets/ear_dataset.py` (đã bỏ — **cần train lại mọi checkpoint recognition**) |
| 18 | `count_flops()` trả MACs nhưng gọi nhầm "FLOPs" (lệch 2x) | Sai khi đối chiếu literature NTIRE | `utils/metrics.py`, `eval_sr_quality.py` |
| 19 | Accuracy macro-average theo batch thay vì micro-average đúng | Số tuyệt đối lệch nhẹ (so sánh tương đối vẫn ổn) | `eval_recognition.py`, `eval_real_lr_holdout.py` |
| 20 | DataLoader `num_workers>0` không seed cố định (`worker_init_fn`/`generator`) | Không tái lập tuyệt đối dù đã set_seed + cudnn.deterministic | `utils/seed.py`, `train_recognition.py`, `train_sr.py`, `train_sr_distill.py` |
| 21 | `scipy.stats.nct.cdf` tràn số ở đuôi phân phối cực nhỏ trong `mdes_paired()` | Crash `ValueError` giữa chừng khi tổng hợp kết quả | `data/aggregate_multi_seed_results.py`, `data/aggregate_lambda_sweep.py`, `data/aggregate_saliency_sweep.py` (đổi sang `nct.sf()` + coi NaN đuôi là 0) |
| 22 | `eval_sr_quality.py` ghi CSV kiểu append; nhiều script sweep gọi lại (do bị dừng giữa chừng) không có chốt chặn | Nhãn trùng lặp trong CSV, số liệu 2 lần chạy khác nhau lẫn vào nhau không cảnh báo | `pipeline/run_lambda_saliency_sweep.sh` (thêm chốt + logic bỏ qua tổ hợp đã xong) |
| 23 | Hardcode "cấu hình KD v2 thắng" (feat=0.5/identity=0.1) làm nền cố định ở 3 script khác nhau, dựa trên kết luận SAU NÀY bị bác bỏ bằng multi-seed thật | Sweep saliency/learned-pruning bị nhiễu (confound) bởi 1 cơ chế đã biết không giúp ích | `pipeline/run_lambda_saliency_sweep.sh`, `pipeline/run_prune_sparsity_screen.sh`, `pipeline/run_multi_seed_learned_prune.sh` (đổi về feat=0/identity=0) |
| 24 | `EarlyStopping` không có ngưỡng cải thiện tối thiểu (`min_delta`) | Với loss giảm đơn điệu cực nhỏ dần (diminishing returns), early-stop không bao giờ kích hoạt thật, chạy tới tận `max_epochs` | `utils/early_stopping.py`, `train_sr_distill.py` (thêm `--min_delta`, mặc định 0.0, không đổi hành vi script khác) |
| 25 | `kill <PID>` tiến trình bash cha KHÔNG tự động giết tiến trình con Python đang chạy bên trong | Tiến trình mồ côi tích luỹ qua nhiều lần restart, nhiều `train_sr_distill.py` tranh chấp cùng 1 GPU, có cả bản dùng recipe đã bị bác bỏ | Quy trình vận hành: dùng `pkill -f <tên_script>` thay vì `kill <PID>`, xác nhận sạch bằng cả `ps aux` lẫn `nvidia-smi` |
| 26 | `check_duplicate_labels.py`/`check_nan_in_results.py` quét đệ quy `results/` dính luôn các thư mục đã cố tình cách ly dữ liệu hỏng (hậu tố `_CONTAMINATED...`) | Cổng chặn trước khi tổng hợp báo cáo bị chặn nhầm dù dữ liệu đang hoạt động thật đã sạch | `data/check_duplicate_labels.py`, `data/check_nan_in_results.py` (loại trừ đường dẫn chứa `CONTAMINATED`) |

**Mục 14, 16, 17 làm THAY ĐỔI input/gradient thật** — mọi checkpoint train
trước các bản vá này cần **train lại từ đầu** (Bước 1 trở đi cho mục 14;
Bước 3 trở đi cho mục 17; bất kỳ checkpoint nào qua `train_sr_distill.py`/
`train_sr_learned_prune.py` cho mục 16). Mục 15, 18-20 chỉ đổi cách
đo/tính/tái lập — không bắt buộc train lại, chỉ cần eval lại hoặc lưu ý khi
đối chiếu số liệu cũ.

**[MỚI — điểm 3, review Q1 tiếp theo]** `train_sr.py`/`train_recognition.py`
cũng đã được thêm cùng cơ chế chặn NaN trước `backward()`/`optimizer.step()`
ở nhánh KHÔNG dùng GradScaler (fallback CPU, hoặc `span_official` vốn tắt
AMP) — rủi ro thấp hơn `train_sr_distill.py` (L1/CrossEntropy ổn định hơn
identity/saliency loss nhiều) nên KHÔNG bắt buộc train lại checkpoint đã có
từ Bước 3-8 chỉ vì bản vá này, nhưng áp dụng cho MỌI lần train tiếp theo.

### Ba điểm cần ghi rõ trong Methods/Limitations khi viết bài báo

1. **Multi-seed (mục 10.3) chỉ đo phương sai ở bước RECOGNITION downstream**
   trên MỘT checkpoint SR cố định (seed=42) — không phải phương sai
   end-to-end toàn chuỗi. Mục 10.5 (`run_sr_seed_variance.sh`) đo thêm trục
   SR-seed, nhưng chỉ 1 backbone/n=3 — **không được viết "ổn định với seed
   train SR"** nếu chưa có đủ số liệu từ đó.
2. **`real_lr_holdout` không tách identity khỏi train** — cùng 164 người đã
   thấy trong train, chỉ khác ảnh (nhỏ hơn thật ngoài đời thay vì downsample
   nhân tạo). Đây là kiểm chứng "tổng quát hóa sang suy giảm thật", KHÔNG
   phải kiểm chứng "tổng quát hóa sang người chưa từng thấy" — nêu rõ để
   tránh reviewer hiểu nhầm phạm vi của bảng này.
3. **Recognition training không có augmentation không gian nào** (đã bỏ
   `RandomHorizontalFlip` — mục 17 — và từ trước đến giờ không có
   crop/rotate/color-jitter khác) — dễ overfit hơn so với recipe recognition
   tiêu chuẩn trong literature (thường có ít nhất flip+crop). Cần nêu rõ
   trong Methods đây là lựa chọn có chủ đích (ảnh tai không đối xứng, tránh
   augmentation làm méo đặc trưng sinh trắc học) chứ không phải thiếu sót,
   và cân nhắc thêm 1 câu Limitations về nguy cơ overfit tăng nhẹ do thiếu
   augmentation không gian.
4. **Ngưỡng "có ý nghĩa" dùng α=0.10, LỎNG HƠN chuẩn α=0.05 truyền thống**
   (áp dụng cho mọi p-value/Bonferroni/MDES trong mục 10.2, 10.3, 12, 13.1) —
   lựa chọn có chủ đích cho bước sàng lọc n=3-5 seed (bậc tự do rất nhỏ), NHƯNG
   phải nêu rõ tường minh trong Methods, không để ngầm định — reviewer Q1 có
   thể yêu cầu giải thích tại sao không dùng 0.05 chuẩn.

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
    run_multi_seed.sh, run_multi_seed_extra_seeds.sh  # Bước 10.3
    run_sr_seed_variance.sh         # Bước 10.5
    run_ablation_kd_v2.sh, run_multi_seed_kdv2.sh     # Mục 12
    run_lambda_saliency_sweep.sh    # Mục 13.1
    run_prune_sparsity_screen.sh, run_multi_seed_learned_prune.sh  # Mục 13.2
    check_prerequisites.sh, _update_config_from_report.py

  data/
    prepare_splits.py               # khảo sát + chọn ngưỡng + letterbox + chia split (theo ẢNH)
    build_lr.py                     # downsample HR -> LR
    build_sr.py                     # chạy model SR: LR -> SR
    aggregate_results.py, aggregate_ablation_results.py, aggregate_lambda_sweep.py,
    aggregate_multi_seed_results.py, aggregate_ablation_kd_v2_results.py,
    aggregate_saliency_sweep.py, aggregate_sr_seed_variance.py  # gộp JSON -> CSV + kiểm định
    export_training_log_summary.py  # trích epoch/OOM/thời gian từ train.log -> CSV

  datasets/
    ear_dataset.py                  # Dataset đọc theo domain (hr/lr/sr_baseline/sr_improved)
    hrlr_pair_dataset.py            # Dataset đọc cặp (HR, LR), dùng chung cho train_sr*/eval_sr_quality

  models/
    sr_models.py                    # SPAN tự viết lại (span, span_tiny, span_large) + EDSR + SPANLearnedPrune, SRFeatureHook + RLFN/ECBSR/SAFMN/SMFANet
    span_official_wrapper.py        # import SPAN CHÍNH THỨC, vá lỗi chuẩn hóa (SPANWithRescale)
    recognition_model.py            # factory 5 backbone, đo feat_dim động, chuẩn hoá ImageNet

  scripts/setup_span_official.sh    # clone repo SPAN + hướng dẫn tải checkpoint

  utils/
    letterbox.py                    # resize giữ tỷ lệ, không méo hình tai
    metrics.py                      # accuracy, PSNR, SSIM, params, MACs/FLOPs, latency
    logger.py, early_stopping.py, device_manager.py, seed.py

  train_sr.py                       # train SR thường (pixel loss) — SPAN baseline, EDSR, Track A
  train_sr_distill.py               # train span_tiny/span_large/Track B: distillation + feature-KD + saliency-weighted + identity loss (fp32, --seed)
  train_sr_learned_prune.py         # học pruning độ sâu có giám sát (gate học được, harden_and_export)
  train_recognition.py              # train recognition 1 domain + 1 backbone (--seed, --init_ckpt, --freeze_backbone)
  eval_recognition.py               # test checkpoint, hỗ trợ cross-domain
  eval_sr_quality.py                # đo PSNR/SSIM/MACs/FLOPs/latency cho 1 model SR
  eval_real_lr_holdout.py           # eval trên real_lr_holdout.json (LR thật)
  export_sr_comparison_images.py    # xuất ảnh so sánh trực quan (Bước 9)
```

---

## Yêu cầu phần cứng

- GPU khuyến nghị (CPU vẫn chạy được nhờ `DeviceManager` tự fallback, chậm
  hơn nhiều).
- Dung lượng: `splits/hr` + `splits/lr` ~180MB, checkpoint mỗi model
  recognition/SR vài MB đến vài chục MB.
- Thời gian: Bước 3 (10 model) và Bước 10.3 (multi-seed x 5 backbone) là các
  bước tốn thời gian nhất trong pipeline chính. Toàn bộ Bước 10 (ablation +
  lambda sweep 18 lần + multi-seed 45-75 lần + span_large ablation +
  Track A/B 5+4 kiến trúc x 15 lần) có thể mất nhiều giờ đến vài ngày tùy
  phần cứng — chạy qua đêm / `tmux`/`screen`/`nohup`.

## Tổng hợp báo cáo cuối cùng

```bash
python data/generate_final_report.py --config configs/config.yaml \
    --results_dir results --out_dir results/final_report
```

Đọc `results/final_report/REPORT.md` trước tiên — đã gộp sẵn toàn bộ bảng:
PSNR/SSIM/LPIPS, accuracy (top-1/rank-5/AUC/EER/gender) theo backbone x
domain, real-LR holdout, training summary.
