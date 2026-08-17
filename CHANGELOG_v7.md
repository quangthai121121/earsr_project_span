# CHANGELOG v7 — Sửa lỗi rà soát lại (đợt 7) + thêm baseline SR mới + dọn pipeline

Nối tiếp `CHANGELOG_v5.md`/`CHANGELOG_v6.md` (giữ nguyên nội dung các đợt
trước). File này ghi **đợt 7** — yêu cầu của bạn: "sửa code cho đúng hết cho
EarVN1.0, train lại từ đầu để xác nhận kết quả, coi như làm lại thí nghiệm
từ đầu, dọn pipeline cho chuẩn chỉnh, thêm RLFN/ECBSR/SAFMN để so sánh".

Toàn bộ 6 điểm dưới đây đến từ 1 đợt code review chi tiết (bên ngoài) mà bạn
gửi — **đã xác minh từng điểm bằng cách đọc trực tiếp mã nguồn thật** trước
khi quyết định sửa (không đoán), xem cột "Kết luận".

---

## Phần A — 6 điểm sửa lỗi

### A1. Nhãn giới tính (`data/prepare_splits.py::gender_from_person_id`)

**Câu hỏi đặt ra**: "001-098=nam, 099-164=nữ" có phải suy đoán chưa kiểm
chứng không?

**Kết luận sau khi xác minh**: KHÔNG phải bug. Đã tra cứu bài báo gốc công
bố dataset — Hoang, V.N., *"EarVN1.0: A new large-scale ear images dataset
in the wild"*, Data in Brief, 2019 (PMC6831707 / ScienceDirect
S2352340919309850): *"98 males and 66 females... the first 98 folders (from
01 to 98) belong to male class and the rest (from 99 to 164) are female."*
— khớp CHÍNH XÁC với code hiện tại (98+66=164, đúng ranh giới 98/99).

**Ảnh hưởng tới kết quả EarVN1.0 đã có**: KHÔNG — logic không đổi, chỉ cập
nhật comment/docs để không còn ghi "cần kiểm tra lại" (đã kiểm tra xong, kết
quả: đúng). `identity_accuracy` (chỉ số chính của Paper 1) độc lập hoàn toàn
với hàm này (`build_label_map()` trong `datasets/ear_dataset.py` không dùng
gender). Chỉ `gender_accuracy` phụ thuộc — và cột này vốn đã ngoài phạm vi
Paper 1 theo quyết định trước đó của bạn.

**File đổi**: `data/prepare_splits.py`, `docs/01_data_preparation.md` (chỉ
comment/docs).

---

### A2. PSNR/SSIM đo cả vùng đệm đen (letterbox padding)

**Kết luận**: ĐÚNG, đã sửa. `eval_sr_quality.py` trước đây đo PSNR/SSIM trên
TOÀN BỘ canvas 80x80 (bao gồm viền đen do letterbox pad, trung vị ~27-31%
diện tích canvas là viền đen tùy ảnh) — viền đen giống hệt nhau giữa SR và
HR (cả 2 đều bị pad cùng công thức) nên luôn "đoán đúng" ở đó, làm PSNR/SSIM
bị THỔI PHỒNG so với chất lượng phục hồi chi tiết tai thật, và không so sánh
được với cách đo chuẩn trong literature SR (thường không có viền đệm).

**Cách sửa**: KHÔNG sinh lại ảnh HR/LR/SR nào — vùng ROI thật được suy
NGƯỢC từ (width, height) ảnh gốc đã lưu sẵn trong `splits.json`, qua công
thức letterbox y hệt lúc tạo ảnh (`utils/letterbox.py::compute_letterbox_geometry()`,
tách ra dùng CHUNG cho cả việc tạo ảnh lẫn việc suy ROI — một nguồn công
thức duy nhất, tránh lệch nhau). `eval_sr_quality.py` giờ ghi CẢ 2 bộ số vào
CSV: `psnr_db`/`ssim` (ROI-only, **dùng cho bài báo**) và
`psnr_db_full_canvas`/`ssim_full_canvas` (số cũ, giữ lại để đối chiếu minh
bạch — không âm thầm xoá).

**QUAN TRỌNG — phạm vi cố ý giới hạn**: chỉ sửa cách ĐO (evaluation), KHÔNG
đổi loss lúc TRAIN (`train_sr.py`/`train_sr_distill.py` vẫn dùng L1 trên
toàn canvas như cũ) — đây là lựa chọn có chủ đích: sửa loss lúc train là
một thay đổi thiết kế khác (có thể cải thiện chất lượng nhưng cũng có thể
ảnh hưởng không lường trước), không phải sửa lỗi. Vì vậy: **KHÔNG cần train
lại bất kỳ model SR nào cho điểm A2** — chỉ cần chạy lại `eval_sr_quality.py`
(hoặc `pipeline/08_benchmark_and_aggregate.sh` bước tương ứng) trên
checkpoint đã có.

**File đổi**: `utils/letterbox.py`, `utils/metrics.py`
(`compute_psnr_roi`/`compute_ssim_roi`), `datasets/hrlr_pair_dataset.py`
(`return_bbox=True`, mặc định vẫn `False` — KHÔNG đổi hành vi
`train_sr.py`/`train_sr_distill.py`), `eval_sr_quality.py`.

**[SỬA THÊM — phát hiện qua review tiếp theo] Bug lệch cột CSV khi eval lại
trên file cũ**: `result` dict trong `eval_sr_quality.py` đổi từ 9 cột (bản
trước đợt 7) sang 11 cột (thêm `psnr_db_full_canvas`/`ssim_full_canvas`).
Code ghi CSV gốc chỉ kiểm `write_header = not out_path.exists()` — nếu
`results/sr_quality.csv` đã tồn tại từ lần chạy TRƯỚC đợt 7 (header 9 cột
cũ) và ta chỉ "eval lại" đúng như hướng dẫn ở mục A2 phía trên (không train
lại, không xoá file cũ), script sẽ append dòng 11-cột vào dưới header 9-cột
— LỆCH CỘT toàn bộ, không có exception nào báo (CSV vẫn hợp lệ về cú pháp).
Đã sửa: trước khi append vào file đã tồn tại, đọc header thật trên đĩa và so
với `fieldnames` hiện tại — nếu KHÁC, dừng lại và báo lỗi rõ ràng (yêu cầu
đổi tên/xoá file cũ) thay vì âm thầm ghi sai vị trí.

---

### A3. README lệch config thật

**Kết luận**: ĐÚNG, lỗi thời. 2 chỗ sai:
- `span_tiny`: README ghi "0.875M tham số, gain=2" — SAI, số thật (đối chiếu
  trực tiếp `results/sr_quality.csv` + `models/sr_models.py`) là **0.230M**
  tham số (feat=48, n_blocks=3, không có "gain"). Đã sửa.
- `lambda_identity`: README ghi "0.1, đang kiểm định" — SAI, `configs/config.yaml`
  thật đã chốt **0.0** từ trước (kết luận sau lambda sweep: identity loss >
  0 gây hại chất lượng SR). Đã sửa.
- Ngoài ra: README ghi `run_multi_seed.sh` "chỉ chạy 1 backbone đại diện" —
  SAI, script thật đã chạy đủ 5 backbone x 3 seed. Đã sửa luôn.

**Ảnh hưởng tới kết quả EarVN1.0 đã có**: KHÔNG — chỉ là văn bản tài liệu,
mọi kết quả đã có được sinh ra với config/code THẬT (đúng), README chỉ mô
tả sai bằng chữ.

**File đổi**: `README.md`.

---

### A4. Thiếu chuẩn hoá ImageNet cho backbone pretrained

**Kết luận**: ĐÚNG, đã sửa. Cả 5 backbone (`mobilenet_v2`,
`mobilenet_v3_small`, `resnet18`, `efficientnet_b0`, `ghostnet_100`) đều
`pretrained=True` (trọng số ImageNet) nhưng `datasets/ear_dataset.py` trước
đây chỉ `Resize` + `ToTensor()` (ảnh về [0,1]), KHÔNG chuẩn hoá theo
mean/std ImageNet mà các trọng số pretrained này được học trên đó — làm
giảm hiệu quả transfer learning từ pretrained (không gây lỗi runtime, chỉ
âm thầm kém hơn).

**Cách sửa**: chuẩn hoá đặt NGAY TRONG `models/recognition_model.py`
(buffer `_imagenet_mean`/`_imagenet_std`, gọi ở đầu CẢ `forward()` LẪN
`embed()`) — KHÔNG đặt trong Dataset transform. Lý do đặt trong model: đảm
bảo MỌI đường vào model đều được chuẩn hoá nhất quán qua đúng 1 chỗ, kể cả
đường `train_sr_distill.py` gọi `recognition_model.embed()` trực tiếp trên
ảnh SR/HR thô [0,1] từ `HRLRPairDataset` (dataset này KHÔNG normalize) — nếu
đặt chuẩn hoá trong `EarDataset` thay vì trong model, đường `embed()` này sẽ
bị bỏ sót (đúng kiểu lỗi "sửa 1 nơi quên 1 nơi" project từng gặp).

**Ảnh hưởng tới kết quả EarVN1.0 đã có**: CÓ, đáng kể — input đưa vào mọi
recognition model đã đổi phân phối giá trị. **BẮT BUỘC train lại TOÀN BỘ
recognition model** (mọi domain hr/lr/sr_baseline/sr_improved, mọi backbone,
mọi seed) để accuracy phản ánh đúng bản đã sửa. Đây là điểm tốn công sức
nhất trong 6 điểm.

**File đổi**: `models/recognition_model.py`.

---

### A5. Kernel nội suy không nhất quán

**Kết luận**: ĐÚNG, đã sửa. `data/build_lr.py` và `utils/letterbox.py` dùng
tường minh `Image.BICUBIC`; nhưng `datasets/ear_dataset.py`'s
`transforms.Resize((image_size, image_size))` không truyền `interpolation`,
mặc định của torchvision là BILINEAR — khác kernel so với nơi khác trong
pipeline.

**Cách sửa**: thêm tường minh `interpolation=InterpolationMode.BICUBIC`.

**Phạm vi ảnh hưởng THỰC TẾ** (quan trọng, tránh hiểu lầm phải train lại
toàn bộ): domain `hr`/`sr_baseline`/`sr_improved` lưu ảnh trên đĩa đã ĐÚNG
sẵn kích thước `image_size x image_size` — `Resize()` ở đó là NO-OP (không
đổi gì) bất kể kernel nào. **Chỉ domain `lr` bị ảnh hưởng thật** (ảnh gốc
20x20 phóng ngược lên 80x80 để đưa vào recognition network).

**Ảnh hưởng tới kết quả EarVN1.0 đã có**: CÓ, nhưng phạm vi HẸP hơn A4 —
chỉ cần train lại recognition model trên domain `lr` (5 backbone x N seed),
KHÔNG cần đụng tới domain hr/sr_baseline/sr_improved.

**File đổi**: `datasets/ear_dataset.py`.

---

### A6. Lambda sweep: 5 so sánh không hiệu chỉnh multiple-comparison

**Kết luận**: ĐÚNG, đã sửa. `data/aggregate_lambda_sweep.py` chạy 5 phép
`paired t-test` độc lập (lambda_identity 0.05/0.1/0.2/0.3/0.5 so với baseline
0.0) ở alpha=0.10 mà không hiệu chỉnh — tăng tỷ lệ báo sai dương tính so với
alpha danh nghĩa.

**Cách sửa**: thêm hiệu chỉnh Bonferroni (`p_corrected = min(1, p_raw x
n_comparisons)`) — BẢO THỦ nhất, phù hợp bước sàng lọc. Giữ NGUYÊN p-value
thô để đối chiếu (không xoá số cũ). **Bổ sung thêm**: bản trước p-value chỉ
in ra console (dễ mất) — giờ lưu luôn cả `p_value_vs_baseline_raw` và
`p_value_vs_baseline_bonferroni` vào `lambda_sweep_summary.csv`.

**Ảnh hưởng tới kết quả EarVN1.0 đã có**: KHÔNG đổi số liệu/checkpoint nào
— chỉ đổi cách DIỄN GIẢI ý nghĩa thống kê của bước sàng lọc lambda (đã có từ
trước khi chốt `lambda_identity=0.0`). Không cần train/chạy lại gì, chỉ cần
biết p-value nào còn "có ý nghĩa" sau hiệu chỉnh khi viết phần Method.

**File đổi**: `data/aggregate_lambda_sweep.py`.

---

## Tổng kết mức ảnh hưởng — CẦN TRAIN LẠI GÌ

| Điểm | Cần train lại? | Phạm vi |
|---|---|---|
| A1 (gender) | Không | — (chỉ docs) |
| A2 (PSNR/SSIM ROI) | Không (chỉ eval lại) | Chạy lại `eval_sr_quality.py` trên checkpoint SR đã có |
| A3 (README) | Không | — (chỉ docs) |
| A4 (ImageNet norm) | **CÓ — toàn bộ** | Mọi domain x mọi backbone x mọi seed recognition |
| A5 (kernel nội suy) | **CÓ — 1 phần** | Chỉ domain `lr`, mọi backbone x mọi seed |
| A6 (Bonferroni) | Không | — (chỉ cách diễn giải p-value) |

Vì A4 đã bắt buộc train lại TOÀN BỘ recognition (bao trùm cả phạm vi của
A5), và bạn đã quyết định "train lại từ đầu, coi như làm lại thí nghiệm" —
khuyến nghị: **xoá sạch `runs/`, `splits/` (trừ `raw_data/`), `results/`
hiện có và chạy lại toàn bộ pipeline từ Bước 1** thay vì train lại chọn lọc
từng phần — vừa đơn giản hơn, vừa tránh sai sót do trộn checkpoint cũ/mới.

---

## Phần B — Kiến trúc SR nhẹ bổ sung (RLFN + RLFN-adapted / ECBSR / SAFMN)

Thêm để so sánh `span_tiny` với SR nhẹ BÊN NGOÀI họ SPAN (không chỉ so với
"bản cắt gọn của chính SPAN"). Tất cả đều port/thiết kế lại dựa TRỰC TIẾP
trên bài báo gốc + mã nguồn chính thức (đã tải/đọc thật trước khi viết,
không đoán công thức — xem chú thích chi tiết ngay trong
`models/sr_models.py`).

### B0. [SỬA — lỗi tài liệu phát hiện qua review, sai ở 4 chỗ trong bản đầu]

Bản đầu ghi các baseline này "train from-scratch pixel loss L1 thuần túy —
ĐÚNG cách EDSR/span_large đang được đối xử, KHÔNG phải học sinh distillation
của span_tiny" — **SAI**. Kiểm tra trực tiếp `RUN_ALL_span_large_ablation.sh`:

```bash
python train_sr_distill.py --config "$CONFIG" --student_arch span_large
```

`span_large` THỰC RA train qua `train_sr_distill.py` — CÙNG một "học sinh"
distillation như `span_tiny` (cùng teacher=`span_official`, cùng identity
loss, cùng recipe) — KHÔNG train pixel-loss-thuần như EDSR. Chỉ EDSR mới
thật sự dùng `train_sr.py`. Câu "EDSR/span_large đối xử giống nhau" tự nó
đã sai — 2 model này dùng 2 recipe hoàn toàn khác nhau.

**Hệ quả của lỗi tài liệu này**: so sánh ban đầu giữa `span_tiny` (CÓ
distillation) và `rlfn`/`ecbsr`/`safmn` (pixel-loss-thuần, KHÔNG có
distillation) lẫn lộn 2 biến số cùng lúc (kiến trúc VÀ recipe huấn luyện) —
không cô lập được riêng "kiến trúc nào tốt hơn". Nếu `span_tiny` thắng trong
bảng đó, phần thắng có thể một phần đến từ việc CÓ distillation mà đối thủ
KHÔNG có, chứ không hẳn do kiến trúc SPAN nén ưu việt hơn.

**Đã sửa bằng cách tách 2 track riêng biệt, không trộn lẫn (4 file cùng
được cập nhật: `models/sr_models.py`, `README.md`,
`RUN_ALL_extra_sr_baseline.sh`, file CHANGELOG này):**
- **Track A** (`RUN_ALL_extra_sr_baseline.sh`, không đổi tên/logic, chỉ sửa
  comment cho đúng): train qua `train_sr.py` (pixel loss thuần) — ĐÚNG
  recipe chính các bài báo gốc RLFN/ECBSR/SAFMN tự dùng (không bài nào dùng
  distillation) — hữu ích đối chiếu với literature, KHÔNG so sánh công bằng
  trực tiếp với `span_tiny`.
- **Track B** (`RUN_ALL_extra_sr_baseline_distilled.sh`, **file MỚI**):
  train qua `train_sr_distill.py --student_arch <arch>` — CÙNG
  teacher/identity-loss/lambda với `span_tiny`/`span_large`, cô lập đúng
  biến kiến trúc. **Dùng Track B cho bảng kết quả CHÍNH khi so sánh với
  `span_tiny`.** Không cần sửa `build_sr_model()`/`train_sr_distill.py` —
  cả 4 kiến trúc mới đã tương thích sẵn với `--student_arch` vì
  `train_sr_distill.py` gọi `build_sr_model(student_arch, ...)` y hệt
  `train_sr.py`, chỉ cần thêm script điều phối mới.

Track A vẫn giữ nguyên, KHÔNG xoá — có giá trị riêng (đối chiếu literature)
nếu được gắn nhãn đúng. Track B là bổ sung, không thay thế.

| Kiến trúc | Nguồn | Cấu hình dùng | ~Tham số |
|---|---|---|---|
| `rlfn` | Kong et al., NTIRE 2022 runtime-track winner, arXiv:2205.07514 | RLFN-cut NGUYÊN VĂN (Sec 4.4 bài báo): feat=48, n_blocks=4, esa_channels=16, ESA-pool kernel=7/stride=3 | 0.317M |
| `rlfn_adapted` | Dựa trên RLFN-cut ở trên | Y HỆT `rlfn`, CHỈ đổi ESA-pool thành kernel=3/stride=2 (xem mục B1 — tránh suy biến ở ảnh 20x20) | 0.317M (như nhau) |
| `ecbsr` | Zhang, Zeng, Zhang — ACM MM 2021 | Port trực tiếp từ BasicSR chính thức (`ecbsr_arch.py`). Cấu hình m4c16 (repo gốc): 4 khối ECB, 16 kênh, with_idt, prelu. Dùng RGB (khác bản gốc dùng kênh Y) để nhất quán với project | nhỏ (deploy mode, nhờ reparameterization) |
| `safmn` | Sun et al., ICCV 2023, arXiv:2302.13800 | Port trực tiếp từ repo chính thức (`sunny2109/SAFMN`, `safmn_arch.py`). dim=36, n_blocks=8 (cấu hình demo mặc định của chính repo) | 0.228M |

**`ecbsr` dùng chung cơ chế `eval_conv`/`update_params()` với Conv3XC (SPAN
chính thức)** đã có sẵn trong `utils/metrics.py::freeze_reparam_modules()`/
`count_params_deploy_mode()` — tái sử dụng nguyên vẹn, không thêm code path
riêng (tránh đúng kiểu lỗi "quên xử lý 1 kiến trúc mới" đã từng gặp).

### B1. [CẬP NHẬT sau review tiếp theo] `rlfn`: ESA suy biến ở scale ảnh 20x20

Bản đầu của mục này chỉ ghi "đã kiểm tra kỹ, không bị lỗi kích thước 0" —
ĐÚNG nhưng THIẾU: "không lỗi kích thước 0" khác với "hoạt động đúng như
thiết kế". Review tiếp theo (bạn) chỉ ra chính xác: module `ESA` bên trong
mỗi `RLFB` (toàn bộ 4 khối của `RLFN`) dùng `max_pool2d(kernel=7, stride=3)`
— truy vết qua từng lớp với input 20x20 (LR thật của project):

```
conv1 (1x1):                  20x20 -> 20x20
conv2 (3x3, stride=2, pad=0): 20x20 -> floor((20-3)/2)+1 = 9x9
max_pool2d(k=7, s=3):          9x9  -> floor((9-7)/3)+1  = 1x1   <- suy biến
conv3 (3x3, pad=1) trên 1x1:   1x1  -> 1x1
interpolate lên lại 20x20:     nhân bản 1 giá trị ra toàn bộ ảnh
```

**Hệ quả**: attention map cuối cùng của ESA không còn thay đổi theo vị trí
không gian — mỗi kênh chỉ còn 1 giá trị scalar broadcast đều ra mọi pixel.
Ở scale 20x20 của project này, ESA hoạt động gần như 1 khối channel-attention
kiểu Squeeze-and-Excitation, KHÔNG còn giữ tính "Enhanced SPATIAL Attention"
như tên gọi/thiết kế gốc (vốn nhắm vào LR patch chuẩn ≥32-64px, nơi
max_pool2d không suy biến về 1 điểm).

**Đây KHÔNG phải lỗi implementation** — công thức đã port đúng 100% theo bài
báo/mã nguồn gốc. Đây là hạn chế CỐ HỮU khi áp cấu hình gốc tác giả vào bài
toán ảnh tai độ phân giải cực thấp.

**Quyết định [SỬA — đảo ngược sau khi bạn hỏi lại lần nữa, cân nhắc kỹ hơn]**:
Quyết định ban đầu (đợt review trước) là "giữ nguyên kernel/stride, chỉ ghi
tài liệu" — với lý do tránh tạo 1 quyết định thiết kế mới chưa kiểm chứng và
tránh thiên vị `rlfn`. Cân nhắc lại: rủi ro để `rlfn` suy biến trong bảng so
sánh CHÍNH của luận án LỚN HƠN rủi ro thêm 1 biến thể có kiểm soát — nếu
`span_tiny` thắng `rlfn` trong bảng chính, kết luận "span_tiny tốt hơn SR
nhẹ khác" dựa một phần trên 1 baseline đã mất chức năng thiết kế, đây là
điểm reviewer SR giỏi có thể dùng để bác bỏ CẢ bảng so sánh, không chỉ đặt
câu hỏi.

**Cách xử lý cuối cùng**: cung cấp CẢ HAI biến thể qua tham số
`esa_pool_kernel`/`esa_pool_stride` (thread qua `ESA` -> `RLFB` -> `RLFN`):
- **`rlfn`** (kernel=7, stride=3 — Y NGUYÊN tác giả công bố): giữ để đối
  chiếu số liệu trực tiếp với literature, và làm bằng chứng định lượng cho
  hiện tượng suy biến — báo cáo như 1 dòng phụ/ablation, KHÔNG dùng làm
  baseline chính trong bảng so sánh.
- **`rlfn_adapted`** (kernel=3, stride=2 — CHỈ đổi đúng 2 số này, không đổi
  gì khác: không đổi conv2 của ESA, không đổi RLFB, không đổi số khối/kênh):
  giữ `v_max` ở 4x4 thay vì suy biến về 1x1 — truy vết:
  `conv2(k=3,s=2,pad=0): 20x20->9x9 (không đổi) -> max_pool2d(k=3,s=2): 9x9->floor((9-3)/2)+1=4x4`
  — ESA khôi phục đúng chức năng "spatial attention". **Dùng làm baseline
  CHÍNH** để so sánh công bằng với `span_tiny` trong bảng kết quả bài báo.

Đây KHÔNG phải "chọn số để span_tiny thắng" — `rlfn_adapted` là thay đổi TỐI
THIỂU (đúng 2 tham số), áp dụng để khôi phục chức năng kiến trúc như tác giả
MÔ TẢ (không phải như tác giả CÔNG BỐ SỐ), không phải để tối ưu kết quả có
lợi cho `span_tiny` — và vẫn giữ `rlfn` (nguyên bản) song song để minh bạch,
không xoá/thay thế.

**Việc bắt buộc phải làm khi viết bài báo**: dù đã có `rlfn_adapted`, vẫn
nên nêu 1-2 câu trong Limitations/Discussion giải thích vì sao có 2 biến thể
`rlfn`/`rlfn_adapted` — đây là 1 phát hiện nhỏ đáng đưa vào bài báo (cho
thấy sự rà soát cẩn thận: "áp thẳng cấu hình gốc của RLFN vào ảnh 20x20 làm
ESA suy biến; chúng tôi phát hiện và điều chỉnh tối thiểu để khôi phục chức
năng, dùng bản đã điều chỉnh làm baseline so sánh chính") thay vì một điểm
yếu bị ẩn đi.

**`ecbsr` và `safmn` đã kiểm tra riêng, KHÔNG gặp vấn đề tương tự**: `ecbsr`
toàn conv thường (kể cả 5 nhánh reparameterization của ECB), không có bước
downsample/pooling nội bộ nào. `safmn`'s module SAFM có pool xuống nhưng chỉ
tới tối thiểu `20 // 8 = 2x2` (không suy biến về 1 điểm) ở nhánh sâu nhất
trong 4 nhánh đa tỷ lệ.

Xem thêm chú thích chi tiết + số liệu ngay trong code:
`models/sr_models.py::ESA` (docstring) và `models/sr_models.py::RLFN`
(docstring).

### B2. Giới hạn minh bạch khác đã biết trước

Môi trường soạn thảo code này KHÔNG có mạng để cài `torch` (giống hạn chế
đã gặp ở đợt review trước) — các kiến trúc mới được xác nhận đúng qua (a) đối
chiếu trực tiếp văn bản bài báo/mã nguồn gốc, (b) truy vết thủ công shape
qua từng lớp bằng tay (đã tính chi tiết ở mục B1 cho trường hợp `rlfn`).
**CHƯA chạy thực nghiệm forward pass thật**. Đã kèm sẵn
`test_new_sr_archs.py` — chạy file này ĐẦU TIÊN (vài giây, không cần
dataset) để tự xác nhận trước khi tin tưởng chạy full pipeline:

```bash
python test_new_sr_archs.py
```

**Cách chạy các kiến trúc mới** (sau khi đã có `splits/lr`, `splits/hr` và
`pipeline/run_multi_seed.sh` đã chạy xong — script tự kiểm tra tiền đề):

Track A (đối chiếu literature, recipe pixel-loss thuần — xem B0):
```bash
bash RUN_ALL_extra_sr_baseline.sh rlfn            # ESA suy biến ở 20x20, xem B1 — chỉ để đối chiếu
bash RUN_ALL_extra_sr_baseline.sh rlfn_adapted
bash RUN_ALL_extra_sr_baseline.sh ecbsr
bash RUN_ALL_extra_sr_baseline.sh safmn
```

Track B (CÙNG recipe distillation với span_tiny — dùng cho bảng kết quả
CHÍNH khi so sánh với span_tiny, xem B0):
```bash
bash RUN_ALL_extra_sr_baseline_distilled.sh rlfn_adapted   # khuyến nghị — baseline chính
bash RUN_ALL_extra_sr_baseline_distilled.sh ecbsr
bash RUN_ALL_extra_sr_baseline_distilled.sh safmn
```

Xem chi tiết mục "11." trong `README.md`.

### B3. [MỚI — làm rõ sau câu hỏi tiếp theo] Ba LOẠI so sánh cần phân biệt — không phải mọi cặp đều là ablation 1-biến

Câu hỏi đặt ra: "cái Track B mới sửa có đúng tinh thần cô lập-1-biến như
`span_large` được thiết kế ban đầu không?" — đúng cho MỘT cặp cụ thể,
nhưng cần làm rõ để không nhầm sang cặp khác. Toàn bộ project hiện có 3 loại
so sánh, KHÔNG phải cùng 1 loại:

1. **`span_large` vs `span_tiny` — SẠCH, đúng 1 biến (độ sâu)**: cả 2 train
   qua `train_sr_distill.py`, CÙNG teacher `span_official`, CÙNG identity
   loss, CÙNG lambda — chỉ khác `n_blocks` (6 vs 3). Ablation gốc của
   project (`RUN_ALL_span_large_ablation.sh`) vốn đã đúng ở cặp NÀY.

2. **`span_official` vs `span_large`/`span_tiny` — KHÔNG sạch, mang tính mô
   tả/động lực, KHÔNG phải ablation nhân quả** [phát hiện MỚI khi trả lời
   câu hỏi tiếp theo, chưa từng nêu rõ trước đây trong README/CHANGELOG]:
   cặp này đổi CẢ kiến trúc (REP+fusion) LẪN recipe huấn luyện cùng lúc —
   `span_official` fine-tune pixel-loss thuần (`train_sr.py`, từ checkpoint
   pretrained gốc), còn `span_large`/`span_tiny` được DISTILL TỪ CHÍNH
   `span_official` làm teacher (`train_sr_distill.py`) — có thêm tín hiệu
   giám sát mà `span_official` không có lúc tự train. KHÔNG dùng cặp này để
   quy kết nhân quả riêng cho REP/fusion — chỉ dùng làm số liệu mô
   tả/tham chiếu bối cảnh.

3. **`span_tiny` vs Track B (`rlfn_adapted`/`ecbsr`/`safmn` distilled) —
   SẠCH, đúng 1 biến (họ kiến trúc)**: đây chính là điều vừa sửa ở mục B0 —
   cùng recipe distillation, cùng teacher, cùng lambda với `span_tiny`, chỉ
   khác kiến trúc nền. ĐÚNG tinh thần cô lập-1-biến như cặp 1, chỉ khác biến
   được cô lập là "họ kiến trúc" thay vì "độ sâu".

**Kết luận cho việc viết bài báo**: chỉ dùng cặp 1 và 3 để phát biểu nhân
quả ("X tốt hơn Y vì lý do Z"). Cặp 2 chỉ nên xuất hiện như số liệu tham
chiếu/bối cảnh trong bảng tổng hợp, kèm 1 câu giải thích rõ đây không phải
ablation kiểm soát biến — tránh reviewer đọc bảng rồi tự suy luận nhân quả
sai cho cặp 2. Đã thêm mục README "10.4" giải thích chi tiết 3 loại so sánh
này, đặt cạnh phần giới thiệu `RUN_ALL_span_large_ablation.sh` (trước đây
script này chưa từng được giải thích mục đích thiết kế trong README, chỉ
xuất hiện rải rác trong hướng dẫn chạy lệnh — đây cũng là 1 lỗ hổng tài liệu
được lấp lại nhân dịp này).

---

## Phần C — Dọn dẹp project

- **Xoá** `scripts/RUN_ALL_span_large_ablation.sh` — bản trùng lặp 100%
  (đã `diff` xác nhận identical) với `./RUN_ALL_span_large_ablation.sh` ở
  thư mục gốc (bản gốc mới là canonical — `RUN_ALL_NEW_DATASET.sh` tham
  chiếu đúng đường dẫn này).
- **Áp bản sửa còn thiếu**: `scripts/make_finetune_config.py` — đối chiếu
  với project thật của bạn, đây là file DUY NHẤT còn sót bản lỗi cũ (chỉ
  giảm LR/epoch cho mục `recognition`/`sr_improve`, quên mục `sr` — khiến
  fine-tune `span_baseline`/`edsr`/mọi arch mới qua `train_sr.py` sẽ chạy
  với LR/epoch ĐẦY ĐỦ như from-scratch). Đã sửa để loop qua cả 3 mục.
- Xoá `.DS_Store`, `__pycache__` rải rác trong gói giao (không ảnh hưởng
  code, chỉ dọn sạch).
- **Không đổi**: `pipeline/run_all_after_fix.sh` (script backup/rerun riêng
  cho lần sửa lỗi công thức SPAB trước đó — vẫn hữu ích làm mẫu tham khảo,
  không xung đột với đợt 7), `data/generate_final_report.py` (đã tự động
  tương thích với cột `psnr_db`/`ssim` mới — không cần sửa vì tên cột không
  đổi, chỉ Ý NGHĨA của `psnr_db`/`ssim` đã đúng hơn).

---

## Phần D — [MỚI] Bổ sung đợt rà soát "đủ mạnh cho journal Q1 khó chưa?"

Sau khi giao gói v7f, bạn hỏi thẳng: bộ số liệu đã có đủ mạnh để accept ở
journal Q1/Scopus khó (xử lý ảnh/deep learning) chưa. Trả lời: chưa, còn 4
khoảng trống cụ thể — cả 4 đã implement thật (không chỉ ghi vào Limitations),
xem chi tiết đầy đủ trong `README.md` mục 13. Tóm tắt ở đây:

### D1. LPIPS (chỉ số cảm nhận)

`utils/metrics.py::load_lpips_model()`/`compute_lpips()` (AlexNet backbone),
tích hợp vào `eval_sr_quality.py` — cột mới `lpips` trong `sr_quality.csv`,
đo trên ROI thật (đồng bộ ROI-fix đợt 7), auto-resize ROI quá nhỏ (<64px) để
tránh feature map co về 0 chiều. Cần `pip install lpips --break-system-packages`
(đã thêm `requirements.txt`) — nếu chưa cài, script vẫn chạy, điền `NA`.

### D2. Verification AUC/EER + confusion matrix (nhận diện)

`compute_pairwise_genuine_impostor_scores()` + `compute_roc_auc_eer()` trong
`utils/metrics.py` — AUC (rank-based, chính xác) + EER (nội suy giao điểm
FPR/FNR) từ cosine similarity mọi cặp ảnh trong test set. **Đã kiểm chứng
bằng test số học tổng hợp độc lập** trước khi đưa vào: 2 phân phối tách biệt
-> AUC=1.0/EER~0; trùng hoàn toàn -> AUC~0.5/EER~0.5; 2 Gaussian chồng lấn ->
AUC khớp công thức lý thuyết Phi(d'/sqrt(2)) sai số <0.02 — 3 test đều pass
trước khi merge vào `eval_recognition.py`. Confusion matrix (164x164) lưu
riêng `results/<config_name>_confusion.csv` mỗi lần chạy.

**Bug tiện thể bắt được**: `identity_accuracy_rank5` đã có sẵn trong JSON từ
trước (do `eval_recognition.py` tính) nhưng bị **rớt âm thầm** khỏi
`results/summary.csv` — `data/aggregate_results.py` không liệt kê cột này
trong `fieldnames`. Đã sửa, thêm cả `identity_auc`/`identity_eer` mới.

### D3. SMFANet (ECCV 2024) — baseline SR công bố gần nhất

Zheng et al., ECCV 2024 (Springer LNCS, DOI 10.1007/978-3-031-72973-7_21) —
port trực tiếp từ code CHÍNH THỨC nộp NTIRE2024_ESR
(`Amazingren/NTIRE2024_ESR/models/team24_smfan.py`, đối chiếu khớp với repo
tác giả `Zheng-MJ/SMFANet`). dim=24, n_blocks=8, ffn_scale=1.5. Kiểm tra suy
biến không gian ở LR 20x20 (rút kinh nghiệm lỗi ESA/RLFN trước đó): pool
`(20//8,20//8)=(2,2)` — không suy biến về 1x1, không cần biến thể "adapted".
Wiring đầy đủ: `build_sr_model("smfanet",...)`, cả 2 script
`RUN_ALL_extra_sr_baseline*.sh smfanet` (Track A/B).

### D4. Cohen's d (effect size) — và 3 lỗi bắt được khi implement

Thêm Cohen's d ghép cặp (paired dz) vào `aggregate_lambda_sweep.py` và
`aggregate_multi_seed_results.py`, bên cạnh p-value đã có. Trong lúc làm,
phát hiện và sửa luôn 3 vấn đề (không chỉ thêm tính năng mới rồi bỏ qua code
cũ):

1. `aggregate_lambda_sweep.py` ghép cặp seed theo VỊ TRÍ trong list — chỉ
   đúng ngẫu nhiên nếu mọi mức lambda dùng đúng cùng 1 bộ seed. Sửa: ghép
   qua khoá seed tường minh.
2. `aggregate_multi_seed_results.py` (script tổng hợp DÙNG CHUNG cho MỌI
   thí nghiệm multi-seed trong project — pipeline chính, span_large
   ablation, Track A/B của 5 baseline SR, transfer learning dataset thứ 2)
   **không hề lưu khoá seed** — chỉ gom vào 1 list phẳng theo thứ tự file
   glob, không thể ghép cặp đúng. Viết lại: đọc seed từ tên file, thay
   heuristic "chênh lệch > độ lệch chuẩn" (không phải kiểm định thật) bằng
   paired t-test + Cohen's d thật, ghi ra file mới `<out>_pairwise.csv`
   (Bonferroni theo số cặp domain so sánh trong cùng backbone). Vì dùng
   chung, sửa 1 chỗ có tác dụng ở TẤT CẢ nơi gọi.
3. **Bug dấu phẩy động bắt được qua functional test** (chạy thử với dữ liệu
   JSON tổng hợp mô phỏng đúng tên file/schema thật, KHÔNG chỉ `py_compile`)
   trước khi giao: khi hiệu số giữa seed gần như hằng số (ví dụ đều +0.20),
   độ lệch chuẩn hiệu số ra một số cực nhỏ khác 0 (sai số float64, không
   phải đúng 0) — Cohen's d = mean/std nổ thành số vô nghĩa (quan sát được
   ~3.1x10^15 trong 1 lần chạy thử trước khi sửa). Sửa: đổi điều kiện
   `std == 0` thành `std < 1e-9` ở cả 2 file, chạy lại functional test xác
   nhận ra `NA` đúng như kỳ vọng thay vì số vô nghĩa.

### Kiểm chứng đã làm cho Phần D (trước khi giao)

- `py_compile` toàn bộ file `.py` đã sửa/thêm — PASS.
- Test số học độc lập (numpy thuần, ngoài project) cho AUC/EER — 3 kịch bản
  đều khớp giá trị lý thuyết trước khi đưa logic vào `utils/metrics.py`.
- **Functional test end-to-end** (không chỉ syntax) cho cả
  `aggregate_lambda_sweep.py`, `aggregate_multi_seed_results.py`,
  `aggregate_results.py`, `generate_final_report.py`: tạo file JSON tổng
  hợp đúng định dạng/tên thật, chạy script thật (dùng scipy stub cho phần
  không cài được scipy trong sandbox), đối chiếu output bằng tay — bắt được
  và sửa lỗi Cohen's d nổ số ở mục D4.3 nhờ chính bước này (không phải suy
  luận trước, mà chạy thử ra mới thấy).
- **Giới hạn còn lại (đã ghi rõ, không giấu)**: sandbox không cài được
  `torch`/`lpips` (mạng bị chặn) nên KHÔNG tự chạy được forward pass thật
  của SMFANet hay 1 lần gọi LPIPS thật — 2 phần này dựa trên đọc kỹ code/API
  chính thức + truy vết thủ công shape, CẦN bạn tự xác nhận bằng
  `test_new_sr_archs.py` (đã thêm `smfanet`) và 1 lần chạy nhỏ
  `eval_sr_quality.py` trước khi tin số liệu để viết vào bài báo.

---

## Cách chạy lại từ đầu cho EarVN1.0 (đúng yêu cầu "làm lại thí nghiệm")

```bash
# 0) Đặt code đã sửa (toàn bộ nội dung gói này) vào đúng vị trí, ghi đè code cũ.
pip install -r requirements.txt --break-system-packages   # [MỚI] gồm cả `lpips`

# 1) Xoá sạch kết quả/checkpoint CŨ (raw_data/ giữ nguyên, không đụng tới):
rm -rf runs splits results

# 2) Xác nhận nhanh các kiến trúc SR mới chạy được (vài giây, gồm cả smfanet):
python test_new_sr_archs.py

# 3) Chạy lại toàn bộ pipeline chính (Bước 1-8):
bash pipeline/01_survey_and_prepare_data.sh
bash pipeline/02_setup_span_official.sh   # + tải checkpoint pretrained thủ công như README
bash pipeline/03_train_baseline_recognition.sh
bash pipeline/04_train_teacher_and_span_baseline.sh
bash pipeline/05_train_recognition_sr_baseline.sh
bash pipeline/06_improve_span.sh
bash pipeline/07_train_recognition_sr_improved.sh
bash pipeline/08_benchmark_and_aggregate.sh

# 4) Thí nghiệm bổ sung (bắt buộc cho journal Q1):
bash pipeline/run_ablation.sh
bash pipeline/run_lambda_sweep.sh
bash pipeline/run_multi_seed.sh
bash RUN_ALL_span_large_ablation.sh                 # Tier-1 ablation kiến trúc

# 5) [MỚI] Baseline SR ngoài họ SPAN — Track A (đối chiếu literature):
bash RUN_ALL_extra_sr_baseline.sh rlfn
bash RUN_ALL_extra_sr_baseline.sh rlfn_adapted
bash RUN_ALL_extra_sr_baseline.sh ecbsr
bash RUN_ALL_extra_sr_baseline.sh safmn
bash RUN_ALL_extra_sr_baseline.sh smfanet           # [MỚI — Phần D3, ECCV 2024]

# 5b) [MỚI] Track B — CÙNG recipe với span_tiny (dùng cho bảng kết quả CHÍNH):
bash RUN_ALL_extra_sr_baseline_distilled.sh rlfn_adapted
bash RUN_ALL_extra_sr_baseline_distilled.sh ecbsr
bash RUN_ALL_extra_sr_baseline_distilled.sh safmn
bash RUN_ALL_extra_sr_baseline_distilled.sh smfanet # [MỚI — Phần D3]

# 6) Tổng hợp báo cáo cuối:
python data/generate_final_report.py --config configs/config.yaml \
    --results_dir results --out_dir results/final_report
```

**Cảnh báo thời gian**: đây là toàn bộ khối lượng công việc lớn nhất của dự
án (8 bước chính + ablation + lambda sweep 18 lần + multi-seed 45 lần +
span_large ablation 15 lần + Track A (5 baseline x 15 lần = 75, đã gồm
smfanet MỚI) + Track B (4 baseline x 15 lần = 60, đã gồm smfanet MỚI) = 135
lần train recognition nữa cho riêng phần SR mở rộng) — có thể mất nhiều giờ
đến vài ngày tùy phần cứng. Chạy qua đêm / `tmux`/`screen`/`nohup`. Nếu thời
gian hạn chế, ưu tiên Track B trước (dùng cho bảng kết quả chính), Track A
có thể chạy sau hoặc lược bớt nếu chỉ cần đối chiếu literature ở mức tham
khảo. LPIPS/AUC/EER/confusion matrix/Cohen's d (Phần D) tự động có trong mọi
lần chạy `eval_sr_quality.py`/`eval_recognition.py`/2 script aggregate ở
trên — KHÔNG cần thêm lệnh riêng.

---

## Kiểm chứng đã làm trước khi giao (không đoán)

- Đọc trực tiếp TOÀN BỘ mã nguồn thật liên quan trước khi sửa: `models/*.py`,
  `datasets/*.py`, `data/*.py`, `utils/*.py`, `eval_*.py`, `train_*.py`,
  `pipeline/*.sh`, `configs/config.yaml`, `README.md`, `docs/*.md` — không
  sửa file nào chỉ dựa trên suy đoán/tóm tắt.
- Tra cứu web xác minh trực tiếp: bài báo gốc EarVN1.0 (gender), bài báo +
  mã nguồn chính thức RLFN/ECBSR/SAFMN (kiến trúc mới).
- `py_compile` (Python) + `bash -n` (Shell) cho **TOÀN BỘ** file `.py`/`.sh`
  trong gói giao — tất cả PASS.
- Truy vết thủ công shape qua từng lớp của các kiến trúc SR mới với input
  20x20 (kích thước LR thật của project) — không phát hiện lỗi kích thước.
- Diff/grep xác nhận: 6 điểm sửa lỗi + kiến trúc SR mới (rlfn/rlfn_adapted/ecbsr/safmn) + dọn dẹp đều CÓ MẶT
  trong gói giao (không chỉ có trong bản nháp).
- **Giới hạn cần bạn tự xác nhận thêm** (môi trường soạn thảo không có
  mạng để cài `torch`): chạy `python test_new_sr_archs.py` trước khi chạy
  full pipeline — xem Phần B.

---

## Phần E — [MỚI, đợt 8] 3 script bổ sung: mở rộng seed, kiểm định KD thật, chẩn đoán SMFANet

Bối cảnh: sau khi chạy xong toàn bộ Phần A-D (đã có `results.zip` đầy đủ),
2 lỗ hổng còn lại được xác định:
1. n=3 seed cho multi-seed chính (`pipeline/run_multi_seed.sh`) có lực kiểm
   định (statistical power) còn yếu.
2. Câu hỏi "distillation (KD) có thật sự giúp span_tiny không?" chỉ có bằng
   chứng n=1 (`results/ablation.csv` gốc), xu hướng quan sát được
   (`pixel_only` > `pixel_distill`) KHÔNG có ý nghĩa thống kê ở n=1 — không
   đủ căn cứ để khẳng định theo hướng nào.
3. SMFANet (baseline 2024 mới thêm ở Phần D) có PSNR đo được ~20-25dB, thấp
   bất thường so với mọi kiến trúc khác (~26.25-27.5dB) — chưa rõ do lỗi
   train (bất ổn định số học) hay do kiến trúc thật sự không hợp với ảnh LR
   20x20 cực nhỏ của dataset — KHÔNG được đưa số liệu này vào bài khi chưa
   chẩn đoán.

**E1. `pipeline/run_multi_seed_extra_seeds.sh`** — chạy thêm seed 44, 999
cho ĐÚNG 3 domain (lr/sr_baseline/sr_improved) x 5 backbone như
`run_multi_seed.sh` gốc, ghi vào CÙNG `results/multi_seed/` (không tạo thư
mục riêng) để bước tổng hợp đọc đủ 5 seed (42,123,2024,44,999) cùng lúc.
Chạy SAU `pipeline/run_multi_seed.sh`.

**E2. `pipeline/run_ablation_multiseed.sh`** — phiên bản multi-seed (5 seed)
CỦA RIÊNG bước train_recognition trong `pipeline/run_ablation.sh` (4 cấu
hình pixel_only/pixel_distill/pixel_identity/full), DÙNG LẠI checkpoint SR
đã có (không train lại SR — đúng quy ước "SR train 1 lần, chỉ multi-seed
recognition" đã dùng xuyên suốt project). Kết quả:
`results/ablation_multiseed/ablation_multiseed_summary_pairwise.csv` — có
ĐỦ paired t-test (raw + Bonferroni theo 6 cặp) + Cohen's d cho cả 6 cặp so
sánh trong C(4,2), gồm ĐÚNG cặp `pixel_only` vs `pixel_distill` cần để trả
lời câu hỏi KD có tác dụng hay không BẰNG SỐ LIỆU THẬT thay vì cảm tính.
GIỚI HẠN cần nêu trong bài: multi-seed ở đây chỉ áp dụng cho bước
recognition, SR vẫn cố định ở seed=42 — không phải multi-seed toàn chuỗi.
Chạy SAU `pipeline/run_ablation.sh` VÀ sau khi đã có đủ 5 seed checkpoint
`recognition_lr_mobilenet_v2_seed*` (tức sau cả E1).

**E3. `debug_smfanet.py`** — script CHẨN ĐOÁN (không tự sửa/train lại gì).
Kiểm tra theo thứ tự: (0) quét `train.log` tìm cảnh báo batch bị NaN/Inf
GradScaler âm thầm bỏ qua; (1) NaN/Inf + độ lớn trọng số trong checkpoint;
(2) forward thật trên ảnh LR test thật — thống kê biên độ output trước/sau
clamp, tỷ lệ pixel bão hoà, độ lớn đặc trưng qua từng khối (`feats[i]`,
dùng forward hook) để phát hiện nổ/triệt tiêu qua chiều sâu — nghi vấn hàng
đầu là `F.normalize` không có tham số affine, lặp 2 lần/khối x 8 khối, có
thể làm mất ổn định biên độ đặc trưng ở ảnh cực nhỏ 20x20 (khác hẳn ảnh
DIV2K gốc dùng để thiết kế kiến trúc); (3) tuỳ chọn so sánh trực tiếp với 1
kiến trúc tham chiếu đang hoạt động bình thường (`--compare_ckpt`, mặc định
gợi ý `rlfn_adapted`) trên CÙNG ảnh LR. In kết luận gợi ý (KHÔNG tự động kết
luận thay người dùng) + lưu `results/debug_smfanet/report.txt` +
`sample_*.png` (ảnh LR/SR/HR cạnh nhau) để xem bằng mắt.

**Kiểm chứng đã làm cho Phần E**: `bash -n` PASS cho cả 2 file `.sh`,
`python -m py_compile` PASS cho `debug_smfanet.py`. Đối chiếu tên biến/tên
thuộc tính (`to_feat`/`feats`/`to_img`, `head`/`body`/`upsample`) trực tiếp
với `models/sr_models.py` để đảm bảo hook/forward thủ công trong
`debug_smfanet.py` khớp đúng kiến trúc thật — phát hiện RLFN dùng tên thuộc
tính khác SMFANet, đã thêm nhánh fallback + cảnh báo minh bạch trong output
thay vì báo số liệu "trước clamp" sai cho kiến trúc không hỗ trợ.
**Giới hạn**: môi trường soạn thảo KHÔNG có `torch` cài sẵn và không có
mạng để cài — chưa chạy được functional test thật (forward pass thật) cho 3
script này, CHỈ kiểm chứng qua đọc code đối chiếu trực tiếp với
`models/sr_models.py`/`utils/metrics.py`/`datasets/hrlr_pair_dataset.py`
thật + syntax check. Bạn nên chạy thử `debug_smfanet.py` trên 1 checkpoint
nhỏ trước khi tin tưởng hoàn toàn báo cáo của nó.

---

## Phần F — [MỚI, đợt 9] Đồng bộ n=5 seed cho Track A, Track B, span_large-ablation

Sau Phần E, bảng so sánh CHÍNH (lr/sr_baseline/sr_improved) và ablation KD đã
lên n=5 seed, nhưng Track A/Track B/span_large-ablation vẫn n=3 — lực kiểm
định KHÔNG đồng đều giữa các phần của bài báo. 3 script dưới đây mở rộng
từng phần lên n=5, theo ĐÚNG khuôn mẫu script gốc tương ứng (chỉ thêm seed
44,999 cho bước train_recognition.py, KHÔNG train lại SR — SR các phần này
vốn đã chỉ train 1 lần ở seed=42 theo quy ước toàn project):

- **`RUN_ALL_extra_sr_baseline_extra_seeds.sh <arch> [config]`** — mở rộng
  Track A. Chạy SAU `RUN_ALL_extra_sr_baseline.sh <arch>` gốc VÀ sau
  `pipeline/run_multi_seed_extra_seeds.sh`.
- **`RUN_ALL_extra_sr_baseline_distilled_extra_seeds.sh <arch> [config]`** —
  mở rộng Track B (bảng so sánh CHÍNH với span_tiny). Cùng tiền đề như trên,
  đổi sang `RUN_ALL_extra_sr_baseline_distilled.sh` gốc.
- **`RUN_ALL_span_large_ablation_extra_seeds.sh [config]`** — mở rộng
  ablation span_tiny vs span_large. Cùng tiền đề, đổi sang
  `RUN_ALL_span_large_ablation.sh` gốc.

Cả 3 đều: (1) kiểm tra tiền đề ở BƯỚC 0 (dừng sớm nếu thiếu checkpoint seed
44/999 của domain `lr` — cần chạy `run_multi_seed_extra_seeds.sh` trước);
(2) thêm đúng 2 seed cho recognition trên domain đã có; (3) xoá sạch thư mục
`combined/` cũ rồi build lại từ đầu (tránh trộn lẫn 3-seed cũ với 5-seed
mới) trước khi gọi lại `data/aggregate_multi_seed_results.py` — kết quả file
CSV tổng hợp giữ NGUYÊN TÊN như bản gốc (ghi đè, vì giờ là bản đầy đủ hơn,
không phải kết quả khác).

**Kiểm chứng**: `bash -n` PASS cho cả 3 file. Đối chiếu trực tiếp biến
`RESULTS_DIR`/tên checkpoint/tên domain với đúng 3 script gốc tương ứng
(`RUN_ALL_extra_sr_baseline.sh`, `RUN_ALL_extra_sr_baseline_distilled.sh`,
`RUN_ALL_span_large_ablation.sh`) để đảm bảo khớp quy ước đặt tên, không suy
đoán. Cùng giới hạn môi trường như Phần E: chưa chạy được functional test
thật (không có `torch` trong môi trường soạn thảo).

---

## Phần G — [SỬA, đợt 10] 2 lỗi thật trong debug_smfanet.py (phát hiện qua code review)

Người dùng đối chiếu trực tiếp code với `models/sr_models.py` và phát hiện
đúng 2 lỗi làm giảm giá trị chẩn đoán (không phải lỗi crash, lỗi ÂM THẦM cho
kết quả thiếu/gây hiểu nhầm):

1. **BƯỚC 1 (`check_checkpoint_weights`) không có breakdown theo khối như
   header hứa.** Header in "feats.0..feats.7 = 8 khối" nhưng code dùng
   `model.named_children()` — chỉ liệt kê 3 con trực tiếp (to_feat/feats/
   to_img), `module.parameters()` trên "feats" lại GỘP CHUNG tham số của cả
   8 khối thành 1 số trung bình duy nhất. Nếu 1 khối cụ thể bị nổ trọng số
   còn 7 khối kia bình thường, số trung bình gộp che mất dấu hiệu đó hoàn
   toàn — chẩn đoán sai lệch (false negative).
2. **`--compare_arch rlfn_adapted` (default) không có breakdown theo khối
   khi so sánh.** Điều kiện cũ `hasattr(model,"feats") and isinstance(...,
   Sequential)` chỉ đúng cho SMFANet/SAFMN — RLFN/RLFN_adapted dùng thuộc
   tính `.body` (ModuleList), không phải `.feats`, nên hook không được gắn,
   toàn bộ phần so sánh "quỹ đạo đặc trưng qua từng khối" với baseline chính
   của bài báo (rlfn_adapted) TRỐNG RỖNG khi chạy đúng lệnh mẫu trong
   docstring cũ.

**Sửa (không chỉ vá triệu chứng, sửa gốc)**:
- `_find_block_container(model)`: tự động chọn Sequential/ModuleList NHIỀU
  PHẦN TỬ NHẤT trong các con trực tiếp — phân biệt được "thân chứa khối lặp"
  (feats/body/backbone) với các Sequential ngắn khác (to_img/upsample, chỉ
  2 lớp). Đã xác nhận đúng cho SMFANet/SAFMN ("feats", len=8), RLFN/
  RLFN_adapted ("body", len=4), ECBSR ("backbone", len=6) bằng cách đọc trực
  tiếp `models/sr_models.py`, không suy đoán.
- `_forward_pre_clamp(model, lr_img)`: thay nhánh if/else cũ (chỉ hỗ trợ 1
  khuôn mẫu) bằng 3 khuôn mẫu forward tường minh (backbone/upsampler cho
  ECBSR; head/body/body_tail/upsample cho RLFN/RLFN_adapted — VÀ tình cờ
  cũng đúng cho SPAN/span_tiny/span_large/EDSR do cùng cấu trúc; to_feat/
  feats/to_img cho SMFANet/SAFMN) — cả BƯỚC 1 (trọng số) lẫn BƯỚC 2/3
  (activation) giờ dùng chung 1 nguồn phát hiện container.
- `check_checkpoint_weights()`: thêm breakdown theo TỪNG KHỐI thật (dùng
  `_find_block_container`) bên cạnh breakdown theo con trực tiếp cũ (giữ
  lại, đổi header cho ĐÚNG với những gì nó thực sự in ra).
- Đổi `_mean_abs_weight()` sang trung bình CÓ TRỌNG SỐ THEO SỐ PHẦN TỬ (thay
  vì trung bình của các trung bình từng tensor) — tránh thiên lệch khi 1
  tensor rất lớn bị đánh đồng trọng số ngang với nhiều tensor nhỏ.
- GIỮ NGUYÊN default `--compare_arch=rlfn_adapted` (không đổi sang `safmn`
  như gợi ý ban đầu) vì rlfn_adapted mới là baseline THẬT SỰ dùng trong bảng
  so sánh chính của bài báo (xem `build_sr_model()` docstring) — sửa gốc
  công cụ để hỗ trợ đúng kiến trúc cần dùng, thay vì đổi khuyến nghị sang
  kiến trúc khác chỉ vì công cụ từng có giới hạn.

**Kiểm chứng**: `py_compile` PASS. Đối chiếu TỪNG khuôn mẫu forward
(backbone/upsampler, head/body/body_tail/upsample, to_feat/feats/to_img)
trực tiếp với source thật của SPAN/EDSR/RLFN/ECBSR/SAFMN/SMFANet trong
`models/sr_models.py` — không suy đoán tên thuộc tính. Cùng giới hạn môi
trường như Phần E/F: chưa chạy được functional test thật (không có `torch`).
