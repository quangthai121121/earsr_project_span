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
reparameterization (Conv3XC — lưu song song 2 nhánh trọng số train/eval),
dùng `feat=48, n_blocks=3` (so với SPAN baseline `feat=48, n_blocks=6` +
Conv3XC) → **[SỬA — đợt 7, số cũ "0.875M, gain=2" trong bản trước SAI, đã
đối chiếu trực tiếp `results/sr_quality.csv` thật] 0.230M tham số, giảm
46.0% so với SPAN baseline đo ở chế độ DEPLOY (0.4263M — đã khớp 99.93% với
số liệu SPAN-S tác giả gốc công bố trong Table 1 của arXiv:2311.12770)** —
so sánh CÔNG BẰNG theo chế độ deploy, không phải so với tổng tham số bao gồm
nhánh train dư thừa của SPAN baseline (2.237M).

**Cách train**: distillation từ **SPAN baseline** (không phải EDSR) làm
teacher, kết hợp identity-aware loss từ model recognition đã train trên
domain `hr` làm "giám khảo":

```
L_total = lambda_pixel . L1(SR, HR) + lambda_distill . L1(SR, teacher) + lambda_identity . (1 - cos(f(SR), f(HR)))
```

Trọng số hiện tại trong `configs/config.yaml`: `lambda_pixel=1.0,
lambda_distill=1.0, lambda_identity=0.0` — **[SỬA — đợt 7, số cũ "0.1, đang
kiểm định" trong bản trước đã LỖI THỜI, không khớp config thật]**. Sau khi
chạy `pipeline/run_lambda_sweep.sh` (mục "Thí nghiệm bổ sung" bên dưới) và
kiểm định thống kê (paired t-test + hiệu chỉnh Bonferroni cho so sánh bội —
xem mục 10.2), kết luận: **có bằng chứng trọng số identity loss > 0 gây hại
chất lượng SR** — đã hạ `lambda_identity` về 0.0 (chỉ còn pixel + distillation
loss) trong config chính thức.

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
| `results/summary.csv` | Bảng chính: accuracy (identity top-1/rank-5, AUC/EER verification, gender) theo backbone x domain — [MỚI, xem mục 13] |
| `results/sr_quality.csv` | PSNR/SSIM/LPIPS/params/FLOPs/latency của các model SR — [MỚI: thêm cột `lpips`, xem mục 13] |
| `results/real_lr_holdout.csv` | Accuracy trên ảnh LR thật (không phải downsample nhân tạo) |
| `results/training_summary.csv` | Epoch dừng, early-stop, OOM, thời gian train mỗi lần chạy |
| `results/<backbone>_<domain>_confusion.csv` | [MỚI] Ma trận nhầm lẫn identity (164x164) — 1 file/lần chạy `eval_recognition.py`, xem mục 13 |

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
2024) = 18 lần train student + recognition. Tự động tính `paired t-test`
(raw + Bonferroni) VÀ **Cohen's d ghép cặp** [MỚI, xem mục 13] so với mức
`lambda_identity=0` (chỉ distillation). Kết quả:
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
`results/multi_seed/multi_seed_summary.csv` (trung bình/độ lệch chuẩn theo
backbone x domain, gồm cả rank-5/gender — [MỚI, xem mục 13]).

**[MỚI — đợt bổ sung journal Q1, xem mục 13]** Kèm thêm file
`results/multi_seed/multi_seed_summary_pairwise.csv`: kiểm định **paired
t-test THẬT + Cohen's d** cho MỌI cặp domain (cùng backbone) — thay heuristic
cũ "chênh lệch so với độ lệch chuẩn" (chỉ là ước lượng thô, không phải kiểm
định thống kê). Đây là file quan trọng nhất để viết bảng kết quả có ý nghĩa
thống kê trong phần Results.

**[SỬA — đợt 7, ghi chú cũ đã lỗi thời]**: `pipeline/run_multi_seed.sh` ĐÃ
chạy đủ cả 5 backbone x 3 seed x 3 domain (45 lần train recognition) — không
còn giới hạn "1 backbone đại diện" như ghi chú trước đây. Muốn tăng độ tin
cậy hơn nữa có thể tăng `SEEDS` lên 5 nếu đủ thời gian tính toán.

### 10.4 [MỚI — đợt 7] Ablation kiến trúc Tier-1 (`span_large`) — và 3 LOẠI so sánh cần phân biệt khi viết Methods

```bash
bash RUN_ALL_span_large_ablation.sh
```

**Mục đích thiết kế `span_large`**: `span_official` (baseline gốc) có
reparameterization (Conv3XC) + concat fusion + `n_blocks=6`; `span_tiny` bỏ
cả REP lẫn fusion VÀ cắt xuống `n_blocks=3` — tức đổi 2 biến cùng lúc, không
biết phần sụt/tăng chất lượng đến từ thay đổi nào nếu chỉ so trực tiếp
`span_official` với `span_tiny`. `span_large` (`feat=48, n_blocks=6`, bỏ
REP+fusion — xem `models/sr_models.py::build_sr_model()`) là điểm trung
gian: giữ nguyên độ sâu (6 khối) như baseline, nhưng đã bỏ REP+fusion giống
`span_tiny` — cho phép tách bạch tác động của từng thay đổi.

**[QUAN TRỌNG — làm rõ sau khi rà soát lại]** Không phải mọi cặp trong chuỗi
`span_official -> span_large -> span_tiny` đều là ablation "đổi đúng 1 biến"
theo đúng nghĩa khoa học. Cần phân biệt RÕ 3 loại so sánh sau khi viết
Methods/Results, tránh diễn giải nhân quả sai:

1. **`span_large` vs `span_tiny` — SẠCH, đúng 1 biến (độ sâu)**: cả hai đều
   train qua `train_sr_distill.py`, CÙNG teacher `span_official`, CÙNG
   identity loss, CÙNG mọi lambda trong `configs/config.yaml` — chỉ khác
   `n_blocks` (6 vs 3). Kết luận nhân quả về ảnh hưởng của độ sâu RÚT RA
   ĐƯỢC từ cặp này.

2. **`span_official` vs `span_large` (hoặc `span_tiny`) — KHÔNG sạch, mang
   tính mô tả/động lực, KHÔNG phải ablation nhân quả**: cặp này đổi CẢ 2 thứ
   cùng lúc — (a) kiến trúc (REP+fusion) VÀ (b) recipe huấn luyện.
   `span_official` fine-tune bằng pixel-loss thuần (`train_sr.py`, từ
   checkpoint pretrained gốc tác giả) — trong khi `span_large`/`span_tiny`
   được DISTILL TỪ CHÍNH `span_official` làm teacher (`train_sr_distill.py`)
   — tức `span_large`/`span_tiny` có thêm tín hiệu giám sát (khớp output
   teacher + identity loss) mà `span_official` không có khi tự train. KHÔNG
   dùng cặp này để quy kết riêng cho REP/fusion — chỉ nên dùng để mô tả
   "SPAN nén + recipe distillation cải thiện/thay đổi ra sao so với baseline
   gốc đã fine-tune", không phải phép thử nhân quả 1-biến.

3. **`span_tiny` vs Track B (`rlfn_adapted`/`ecbsr`/`safmn` distilled) —
   SẠCH, đúng 1 biến (họ kiến trúc)** [xem mục 11 bên dưới]: cùng recipe
   distillation, cùng teacher, cùng lambda với `span_tiny` — chỉ khác kiến
   trúc nền. Kết luận nhân quả về "kiến trúc SPAN nén tốt hơn/kém hơn kiến
   trúc khác cùng quy mô" RÚT RA ĐƯỢC từ cặp này.

**Khi viết bài báo**: chỉ dùng cặp 1 và 3 để đưa ra phát biểu nhân quả kiểu
"X tốt hơn Y vì lý do Z". Cặp 2 chỉ nên xuất hiện như số liệu tham chiếu/bối
cảnh (ví dụ trong bảng tổng hợp `sr_quality.csv`), kèm câu giải thích rõ
đây không phải ablation kiểm soát biến.

---

## 11. [MỚI — đợt 7, mở rộng đợt bổ sung journal Q1] So sánh với các kiến trúc SR nhẹ khác đã công bố (RLFN/ECBSR/SAFMN/SMFANet)

Ngoài việc so với chính "họ SPAN" (SPAN baseline, span_large), để chứng minh
`span_tiny` cạnh tranh được với SR nhẹ nói chung, project bổ sung baseline
BÊN NGOÀI (RLFN/RLFN-adapted/ECBSR/SAFMN/SMFANet — **`smfanet` là baseline
MỚI**, xem mục 13.3).

**[SỬA — lỗi tài liệu phát hiện qua review, đã sai ở 4 chỗ trong bản trước]**
Bản trước ghi các baseline này "train from-scratch pixel loss L1 thuần túy —
ĐÚNG cách EDSR/span_large đang được đối xử" — **SAI**: `span_large` (xem
`RUN_ALL_span_large_ablation.sh`, dòng gọi
`train_sr_distill.py --student_arch span_large`) thực ra CŨNG là một "học
sinh" distillation giống hệt `span_tiny` (cùng teacher=`span_official`, cùng
identity loss, cùng recipe `train_sr_distill.py`) — KHÔNG train pixel-loss-
thuần như EDSR. Chỉ EDSR mới thật sự dùng `train_sr.py`. Hệ quả: so sánh ban
đầu giữa `span_tiny` (có distillation) và rlfn/ecbsr/safmn (không có, chỉ
pixel loss) lẫn lộn 2 biến số (kiến trúc VÀ recipe huấn luyện) cùng lúc,
không cô lập được riêng "kiến trúc nào tốt hơn".

**Đã sửa bằng cách cung cấp 2 track riêng biệt, không trộn lẫn:**
- **Track A** (`RUN_ALL_extra_sr_baseline.sh`) — train qua `train_sr.py`
  (pixel loss L1 thuần), ĐÚNG recipe chính các bài báo gốc RLFN/ECBSR/SAFMN
  tự dùng (cả 3 bài đều không dùng distillation) — có ý nghĩa đối chiếu với
  số liệu literature, nhưng KHÔNG so sánh công bằng trực tiếp với
  `span_tiny` (khác recipe).
- **Track B** (`RUN_ALL_extra_sr_baseline_distilled.sh`, **[MỚI]**) — train
  qua `train_sr_distill.py --student_arch <arch>`, CÙNG teacher/identity-
  loss/lambda với `span_tiny`/`span_large` — cô lập đúng biến kiến trúc, so
  sánh công bằng trực tiếp với `span_tiny`/`span_large`. **Dùng Track B cho
  bảng kết quả CHÍNH của bài báo khi so sánh với span_tiny.** Track A chỉ
  nên dùng làm bảng phụ/đối chiếu literature, kèm 1 câu trong
  Limitations/Discussion giải thích 2 track khác recipe.

| Kiến trúc | Nguồn | Cấu hình dùng | ~Tham số |
|---|---|---|---|
| `rlfn` | Kong et al., NTIRE22 runtime-track winner, arXiv:2205.07514 | RLFN-cut NGUYÊN VĂN: feat=48, n_blocks=4, esa_channels=16, ESA-pool kernel=7/stride=3 (đúng tác giả) | 0.317M |
| `rlfn_adapted` | (dựa trên RLFN-cut ở trên) | Y HỆT `rlfn`, CHỈ đổi ESA-pool thành kernel=3/stride=2 — tránh suy biến ở ảnh 20x20 (xem giới hạn bên dưới) | 0.317M (như nhau) |
| `ecbsr` | Zhang et al., ACM MM 2021 | m4c16: 4 khối ECB, 16 kênh, with_idt, prelu (RGB thay vì Y-channel gốc) | ~0.1M (deploy) |
| `safmn` | Sun et al., ICCV 2023, arXiv:2302.13800 | dim=36, n_blocks=8 (cấu hình demo mặc định của repo chính thức) | 0.228M |
| `smfanet` | [MỚI] Zheng et al., **ECCV 2024** (Springer LNCS, DOI 10.1007/978-3-031-72973-7_21) | dim=24, n_blocks=8, ffn_scale=1.5 (khớp file nộp chính thức NTIRE2024_ESR/team24_smfan.py) | ~0.1-0.2M (ước tính, chưa đo thật — xem giới hạn mục 13.3) |

Cả 5 đều port trực tiếp từ mã nguồn/bài báo gốc (đã đối chiếu trước khi
viết, xem chú thích chi tiết trong `models/sr_models.py`), KHÔNG đoán công
thức. `ecbsr` dùng cùng kỹ thuật structural reparameterization như SPAN
chính thức (Conv3XC) — tái sử dụng nguyên vẹn
`utils/metrics.py::freeze_reparam_modules()`/`count_params_deploy_mode()` đã
có sẵn, không cần code path riêng. Cả 6 kiến trúc so sánh (span_tiny và 5
baseline ngoài) đều dùng CHUNG `cfg["image"]["scale"]=4` từ 1 nơi duy nhất
trong `configs/config.yaml`, qua cùng hàm `build_sr_model(arch, scale, ...)`
— không kiến trúc nào bị test ở hệ số phóng đại khác các kiến trúc còn lại.
`smfanet` là baseline **CÔNG BỐ GẦN NHẤT (2024)** trong toàn bộ so sánh, giải
quyết đúng khoảng trống "baseline hơi cũ" (RLFN 2022/ECBSR 2021/SAFMN 2023)
nêu trong review journal Q1 — xem mục 13.3.

**[SỬA — quyết định lại sau review tiếp theo] `rlfn` vs `rlfn_adapted`**:
module ESA bên trong RLFB dùng `max_pool2d(kernel=7, stride=3)` — ở ảnh LR
20x20 của project này (nhỏ hơn nhiều so với LR patch chuẩn trong literature
SR, thường ≥32-64px), attention map suy biến về 1x1 rồi nhân bản đều ra toàn
bộ ảnh, mất tính "spatial" như tên gọi/thiết kế gốc (hoạt động gần giống 1
khối channel-attention kiểu Squeeze-and-Excitation hơn). Ban đầu định chỉ
ghi nhận đây là giới hạn tài liệu, giữ nguyên cấu hình gốc — cân nhắc lại
thấy rủi ro lớn hơn: để `rlfn` suy biến trong bảng so sánh CHÍNH nghĩa là 1
phần kết luận "span_tiny tốt hơn baseline X" dựa trên 1 baseline đã mất chức
năng thiết kế — reviewer SR giỏi có thể không chỉ hỏi mà bác cả phép so
sánh. Xử lý bằng cách cung cấp CẢ HAI:
- **`rlfn`** (kernel=7/stride=3, nguyên văn tác giả): giữ để đối chiếu số
  liệu trực tiếp với literature + làm bằng chứng định lượng cho hiện tượng
  suy biến — báo cáo như 1 dòng phụ/ablation kèm giải thích, KHÔNG dùng làm
  baseline chính.
- **`rlfn_adapted`** (kernel=3/stride=2 — CHỈ đổi đúng 2 số này, không đổi
  gì khác trong toàn kiến trúc): giữ v_max=4x4 thay vì suy biến về 1x1, ESA
  hoạt động đúng chức năng không gian. **Dùng làm baseline CHÍNH** để so
  sánh công bằng với `span_tiny` trong bảng kết quả của bài báo.

Xem số liệu truy vết chi tiết (cả 2 biến thể) trong `models/sr_models.py::ESA`
và `CHANGELOG_v7.md`, Phần B1. **Vẫn nên nêu 1 câu trong Limitations/Discussion**
giải thích vì sao có 2 biến thể `rlfn`/`rlfn_adapted` — đây là 1 phát hiện
nhỏ đáng đưa vào bài báo (naive port suy biến; adapted port thì không), thể
hiện sự rà soát cẩn thận hơn là một điểm yếu bị ẩn đi. `ecbsr` và `safmn`
không gặp vấn đề tương tự, không cần biến thể "adapted".

**Track A — recipe pixel-loss thuần (đối chiếu literature, script CHUNG cho
cả 5 kiến trúc, tham số hoá theo tên kiến trúc + config):**

```bash
bash RUN_ALL_extra_sr_baseline.sh rlfn                    # mặc định configs/config.yaml
bash RUN_ALL_extra_sr_baseline.sh rlfn_adapted
bash RUN_ALL_extra_sr_baseline.sh ecbsr
bash RUN_ALL_extra_sr_baseline.sh safmn
bash RUN_ALL_extra_sr_baseline.sh smfanet                 # [MỚI]
# hoặc cho dataset thứ 2 (ví dụ AWE): bash RUN_ALL_extra_sr_baseline.sh rlfn_adapted configs/config_awe.yaml
```

**Track B — CÙNG recipe distillation với span_tiny/span_large [MỚI, dùng
cho bảng kết quả CHÍNH khi so sánh với span_tiny]:**

```bash
bash RUN_ALL_extra_sr_baseline_distilled.sh rlfn_adapted   # khuyến nghị — baseline chính
bash RUN_ALL_extra_sr_baseline_distilled.sh ecbsr
bash RUN_ALL_extra_sr_baseline_distilled.sh safmn
bash RUN_ALL_extra_sr_baseline_distilled.sh smfanet        # [MỚI]
```

**Track A** mỗi lần chạy: (1) train from-scratch bằng `train_sr.py --sr_arch
<arch>` (pixel loss thuần), (2) sinh ảnh `splits/sr_<arch>/`, (3) đo
PSNR/SSIM/params/FLOPs/latency -> append vào `results/sr_quality.csv`
(nhãn `<arch>`), (4) multi-seed recognition (5 backbone x 3 seed = 15 lần,
tái dùng checkpoint domain `lr` đã có sẵn).

**Track B** mỗi lần chạy: (1) train qua `train_sr_distill.py --student_arch
<arch>` (CÙNG teacher/identity-loss/lambda với span_tiny), (2) sinh ảnh
`splits/sr_improved_<arch>/`, (3) đo chất lượng -> append vào
`results/sr_quality.csv` (nhãn `<arch>_distilled`), (4) multi-seed
recognition y hệt Track A. Cả 2 track đều tái dùng checkpoint domain `lr` —
ĐÚNG cùng protocol downstream recognition đã áp dụng cho `span_large`, đảm
bảo "EarVN1.0 làm sao thì làm y hệt" áp dụng luôn giữa các kiến trúc so sánh
trong CÙNG 1 dataset.

---

## 12. [MỚI — đợt 7] Sáu điểm sửa lỗi rà soát lại (đã áp dụng cho EarVN1.0)

Một đợt code review chi tiết (bên ngoài, độc lập) chỉ ra 6 điểm — cả 6 đã
được xác minh đúng/sai bằng cách đọc trực tiếp mã nguồn thật (không đoán)
trước khi quyết định sửa. Bảng dưới đây tóm tắt, xem chi tiết từng điểm +
mức ảnh hưởng tới số liệu đã có trong `CHANGELOG_v7.md`:

| # | Vấn đề | Kết luận sau khi kiểm tra | Đã sửa ở đâu |
|---|---|---|---|
| 1 | Nhãn giới tính suy từ số thứ tự thư mục — có phải suy đoán chưa kiểm chứng? | **KHÔNG phải bug** — khớp đúng bài báo gốc EarVN1.0 (đã xác minh qua Data in Brief 2019). Không ảnh hưởng `identity_accuracy`. | `data/prepare_splits.py`, `docs/01_data_preparation.md` (chỉ cập nhật ghi chú, không đổi logic) |
| 2 | PSNR/SSIM đo cả vùng viền đệm đen (letterbox), không chỉ vùng tai thật | **Đúng, đã sửa** — đo trên ROI thật, giữ song song số cũ để đối chiếu | `utils/letterbox.py`, `utils/metrics.py`, `datasets/hrlr_pair_dataset.py`, `eval_sr_quality.py` |
| 3 | README lệch config thật (param count span_tiny, giá trị lambda_identity) | **Đúng, lỗi thời** — đã cập nhật khớp số liệu/config thật | `README.md` |
| 4 | Thiếu chuẩn hoá ImageNet dù dùng backbone pretrained=True | **Đúng, đã sửa** — chuẩn hoá đặt trong model (buffer), áp dụng nhất quán cho mọi đường vào (forward + embed) | `models/recognition_model.py` |
| 5 | Kernel nội suy không nhất quán (Resize mặc định BILINEAR, nơi khác dùng BICUBIC) | **Đúng, đã sửa** — chỉ định tường minh BICUBIC | `datasets/ear_dataset.py` |
| 6 | Quét lambda_identity: 5 so sánh đồng thời không hiệu chỉnh multiple-comparison | **Đúng, đã sửa** — thêm hiệu chỉnh Bonferroni, giữ song song p-value thô | `data/aggregate_lambda_sweep.py` |

**Điểm 4 và 5 làm THAY ĐỔI input thật đưa vào recognition model** — mọi
checkpoint recognition đã train trước đợt 7 (EarVN1.0 lẫn AWE) cần **train
lại từ đầu** để số liệu accuracy phản ánh đúng bản sửa. Điểm 1, 3, 6 không
đổi số liệu đã có (chỉ đổi tài liệu/cách diễn giải thống kê). Điểm 2 chỉ đổi
cách ĐO (không đổi trọng số model SR đã train) — không bắt buộc train lại,
chỉ cần eval lại.

---

## 13. [MỚI — đợt bổ sung journal Q1] LPIPS, verification AUC/EER + confusion matrix, SMFANet (2024), Cohen's d

Bổ sung sau khi review kỹ liệu bộ số liệu đã có có đủ mạnh cho journal Q1
khó hay chưa — 4 khoảng trống cụ thể được chỉ ra, cả 4 đã implement (KHÔNG
chỉ ghi vào Limitations). **Toàn bộ mục này CHỈ áp dụng cho EarVN1.0 trong
lần chạy này** — dataset thứ 2 sẽ tự động có đủ 4 chỉ số/baseline mới này
(dùng chung code), không cần sửa gì thêm khi chuyển dataset.

### 13.1 LPIPS — chỉ số cảm nhận, bổ sung cho PSNR/SSIM

`utils/metrics.py::load_lpips_model()`/`compute_lpips()` (backbone AlexNet,
chuẩn phổ biến nhất trong literature SR). Tích hợp vào `eval_sr_quality.py`
— cột MỚI `lpips` trong `results/sr_quality.csv`, tính trên ROI thật (đồng bộ
với cách đo PSNR/SSIM đã sửa ở đợt 7, không tính viền đệm đen letterbox).
**LPIPS càng THẤP càng tốt** (ngược hướng PSNR/SSIM — chú ý khi viết bảng so
sánh). Cần cài thêm: `pip install lpips --break-system-packages` (đã thêm
vào `requirements.txt`). Nếu chưa cài, `eval_sr_quality.py` vẫn chạy được
bình thường (điền `NA` cho cột lpips, in cảnh báo) — không chặn pipeline.

**Giới hạn minh bạch**: sandbox phát triển code này bị chặn cài package mới
(proxy 403) nên KHÔNG thể tự chạy thực nghiệm kiểm tra `compute_lpips()`.
Cách viết dựa trên API ổn định, có tài liệu chính thức của package `lpips`
(không đổi signature nhiều năm nay). Đã kiểm tra kỹ phần xử lý ROI quá nhỏ
(auto-resize lên `min_size=64px` trước khi đưa vào LPIPS, tránh feature map
co về 0 chiều với ảnh crop cực nhỏ) — **người dùng cần tự chạy thử 1 lần nhỏ
trước khi tin số liệu LPIPS đầu ra**, ví dụ:
```bash
python eval_sr_quality.py --config configs/config.yaml --arch span_tiny \
    --ckpt runs/sr_improved_span_tiny/best.pt --label span_tiny --out_csv /tmp/test_lpips.csv
```

### 13.2 Verification-setting: AUC/EER + confusion matrix (phía nhận diện)

Trước đây `eval_recognition.py` chỉ báo `identity_accuracy` (top-1) +
`identity_accuracy_rank5` (rank-5 đã có sẵn từ trước, nhưng **bị rớt khỏi
`results/summary.csv`** — lỗi phát hiện qua rà soát lại lần này, đã sửa
trong `data/aggregate_results.py`). Bổ sung:

- **AUC + EER (verification setting)**: `utils/metrics.py::compute_pairwise_genuine_impostor_scores()`
  lấy embedding TOÀN BỘ test set, tính cosine similarity cho MỌI cặp ảnh,
  tách genuine (cùng identity)/impostor (khác identity), rồi
  `compute_roc_auc_eer()` tính AUC (rank-based, chính xác — không xấp xỉ qua
  lưới ngưỡng) và EER (giao điểm FPR=FNR, nội suy tuyến tính). **Đã kiểm tra
  bằng test số học tổng hợp độc lập** (2 phân phối tách biệt hoàn toàn ->
  AUC=1.0 EER~0; trùng hoàn toàn -> AUC~0.5 EER~0.5; 2 Gaussian chồng lấn ->
  AUC khớp công thức lý thuyết Φ(d'/√2) sai số <0.02) trước khi đưa vào code
  chính — không chỉ tin vào code mà không kiểm chứng độc lập.
  Kết quả nằm trong cột MỚI `identity_auc`/`identity_eer` của
  `results/summary.csv`, kèm cảnh báo tự động nếu số cặp genuine quá ít
  (<30 cặp — verification-setting cần nhiều ảnh/identity mới đủ tin cậy).
- **Confusion matrix**: `utils/metrics.py::compute_confusion_matrix()` — ma
  trận 164x164 (số identity EarVN1.0), lưu riêng file
  `results/<config_name>_confusion.csv` mỗi lần chạy `eval_recognition.py`
  (hàng=nhãn thật, cột=nhãn dự đoán) — dùng để phân tích lỗi có dồn vào 1 số
  ít identity (ví dụ trùng góc chụp/ánh sáng) hay dàn đều, viết phần Error
  Analysis cho bài báo.

Cả 2 không cần thêm dependency mới (chỉ dùng numpy/torch sẵn có trong
project) — tránh rủi ro không cài được package như tình huống LPIPS.

### 13.3 SMFANet — baseline SR công bố GẦN NHẤT (ECCV 2024)

Giải quyết khoảng trống "RLFN 2022/ECBSR 2021/SAFMN 2023 đã hơi cũ tính đến
2026". Thêm `smfanet` (Zheng, Sun, Dong, Pan, "SMFANet: A Lightweight
Self-Modulation Feature Aggregation Network for Efficient Image
Super-Resolution", **ECCV 2024**, Springer LNCS, DOI
10.1007/978-3-031-72973-7_21) — port trực tiếp từ code CHÍNH THỨC nộp cho
NTIRE2024_ESR (`Amazingren/NTIRE2024_ESR`, file
`models/team24_smfan.py`) — repo tác giả chính thức `Zheng-MJ/SMFANet` khớp
cùng kiến trúc. Cấu hình dùng (dim=24, n_blocks=8, ffn_scale=1.5) là biến thể
nhẹ nhất tác giả tự nộp thi (cùng quy ước với `safmn` dùng cấu hình demo nhẹ
nhất, không phải bản đầy đủ trong bài).

Đã kiểm tra suy biến không gian ở LR 20x20 của project này (rút kinh nghiệm
từ lỗi ESA/RLFN phát hiện trước đó): module SMFA pool xuống lưới `(20//8,
20//8) = (2,2)` — **KHÔNG suy biến về 1x1**, không cần biến thể "adapted".
Xem docstring đầy đủ trong `models/sr_models.py::SMFANet`/`_SMFA`.

Wiring đầy đủ: `build_sr_model("smfanet", ...)`,
`RUN_ALL_extra_sr_baseline.sh smfanet` (Track A),
`RUN_ALL_extra_sr_baseline_distilled.sh smfanet` (Track B) — dùng y hệt cú
pháp các baseline khác, xem mục 11.

**Giới hạn minh bạch**: cũng như 3 baseline trước, sandbox không cài được
torch nên KHÔNG tự chạy được forward pass để đo params/FLOPs thật — số
~0.1-0.2M trong bảng mục 11 là ƯỚC TÍNH, cần tự xác nhận bằng
`test_new_sr_archs.py` trước khi dùng trong bài báo.

### 13.4 Cohen's d — effect size bên cạnh p-value

`data/aggregate_lambda_sweep.py` và `data/aggregate_multi_seed_results.py`
đều bổ sung Cohen's d ghép cặp (paired dz = trung bình hiệu số / độ lệch
chuẩn hiệu số) bên cạnh p-value đã có — p-value chỉ nói "có ý nghĩa thống kê
hay không" (phụ thuộc cỡ mẫu, với n=3 seed rất nhạy), Cohen's d nói "chênh
lệch lớn hay nhỏ về THỰC TẾ", độc lập cỡ mẫu.

**2 lỗi phát hiện VÀ SỬA trong lúc implement (không phải chỉ ghi nhận rồi để
đó):**
1. **Ghép cặp sai (tiềm ẩn) trong `aggregate_lambda_sweep.py`**: bản trước
   ghép seed theo VỊ TRÍ trong list (`list(dict.values())`), chỉ đúng NGẪU
   NHIÊN nếu mọi mức lambda dùng đúng cùng 1 bộ seed. Sửa: ghép qua khoá seed
   tường minh, an toàn dù bộ seed lệch nhau giữa các mức.
2. **`aggregate_multi_seed_results.py` cũ hoàn toàn KHÔNG lưu khoá seed**
   (chỉ gom accuracy vào 1 list phẳng theo thứ tự file trong glob) — không
   thể ghép cặp đúng theo seed để làm paired t-test/Cohen's d thật. Viết lại
   để đọc seed từ tên file (`*_seed<N>.json`), đồng thời **thay heuristic cũ**
   ("chênh lệch > độ lệch chuẩn trung bình" — không phải kiểm định thống kê
   thật) **bằng paired t-test + Cohen's d thật**, ghi ra file mới
   `<out>_pairwise.csv` (Bonferroni theo số cặp domain so sánh trong CÙNG 1
   backbone).
3. **Bug dấu phẩy động bắt được qua functional test** (chạy thử với dữ liệu
   tổng hợp trước khi giao code): khi hiệu số giữa các seed gần như hằng số
   (ví dụ đều +0.20), độ lệch chuẩn của hiệu số có thể ra một số CỰC NHỎ khác
   0 do sai số làm tròn float64 (không phải đúng 0 tuyệt đối) — `Cohen's d =
   mean/std` khi đó NỔ thành số vô nghĩa (quan sát được ~3.1x10^15 trong 1
   test tổng hợp) thay vì báo "không xác định". Sửa: đổi điều kiện `std == 0`
   thành `std < 1e-9` ở cả 2 file.

Vì `aggregate_multi_seed_results.py` là script TỔNG HỢP DÙNG CHUNG cho MỌI
thí nghiệm multi-seed trong project (pipeline chính, span_large ablation,
Track A/B của RLFN/ECBSR/SAFMN/SMFANet, transfer learning dataset thứ 2) —
bản sửa này có tác dụng ở TẤT CẢ các nơi gọi, không cần sửa từng chỗ riêng lẻ.

**Đã functional-test cả 2 script** (không chỉ `py_compile`) bằng dữ liệu
JSON tổng hợp mô phỏng đúng định dạng/tên file thật, đối chiếu Cohen's d ra
bằng tay — khớp chính xác (ví dụ 1.500 vs 1.500) trước khi coi là xong.

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
| 10 | Thiếu chuẩn hoá ImageNet cho backbone pretrained | Giảm hiệu quả transfer learning từ pretrained, không lỗi runtime | `models/recognition_model.py` |
| 11 | Kernel nội suy domain `lr` không khớp nơi khác (BILINEAR ngầm định thay vì BICUBIC) | Không nhất quán tiền xử lý ảnh giữa các domain | `datasets/ear_dataset.py` |
| 12 | PSNR/SSIM đo cả viền đệm đen letterbox | Số liệu bị thổi phồng, không so sánh được với literature SR chuẩn | `utils/metrics.py`, `eval_sr_quality.py`, `datasets/hrlr_pair_dataset.py` |
| 13 | Quét lambda_identity: 5 so sánh không hiệu chỉnh multiple-comparison | Tăng rủi ro báo sai "có ý nghĩa thống kê" | `data/aggregate_lambda_sweep.py` |

---

## Yêu cầu phần cứng

- GPU khuyến nghị (CPU vẫn chạy được nhờ `DeviceManager` tự fallback, nhưng
  chậm hơn nhiều).
- Dung lượng: `splits/hr` + `splits/lr` ~180MB (ảnh 80x80 + 20x20 số lượng
  lớn), checkpoint mỗi model recognition/SR vài MB đến vài chục MB.
- Thời gian: Bước 3 (10 model) và Bước 10.3 (multi-seed x 5 backbone nếu mở
  rộng) là các bước tốn thời gian nhất trong toàn bộ pipeline.
