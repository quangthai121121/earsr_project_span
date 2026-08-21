# 03 — Cải tiến SPAN (đóng góp phương pháp chính)

## Khởi tạo Student từ checkpoint chính thức

Student SPAN được khởi tạo từ checkpoint **pretrained chính thức** của tác
giả (train trên DIV2K + LSDIR, phát hành theo giấy phép Apache 2.0 tại
`github.com/hongyuanyu/SPAN`), sau đó fine-tune trên EarVN1.0 — không train
from-scratch. Câu văn methodology tham khảo:

> "SPAN was initialized from the officially released pretrained checkpoint
> (trained on DIV2K and LSDIR) and fine-tuned on EarVN1.0 using our proposed
> composite loss (Section X.X)."

*(Xem `scripts/setup_span_official.sh` và `models/span_official_wrapper.py`
trong repo code — quy trình tải checkpoint và đối chiếu kiến trúc gốc)*

### Lưu ý kỹ thuật đã phát hiện và vá khi tích hợp

Sau khi đối chiếu trực tiếp với source code gốc (`span_arch.py`), phát hiện
`forward()` của model chính thức biến đổi input theo
`x = (x - mean) * img_range` (mean theo thang [0,1], img_range=255) nhưng
**không có bước biến đổi ngược ở output** — output trả về nằm ở thang giá trị
đã nhân `img_range`, không phải `[0,1]` như quy ước ảnh chuẩn hóa qua
`ToTensor()` trong toàn bộ pipeline. Đây là loại lỗi tương thích "âm thầm"
(không gây crash, chỉ làm loss/PSNR sai lệch hoàn toàn) — đã vá bằng cách bọc
thêm lớp `SPANWithRescale` (`models/span_official_wrapper.py`) tự động thực
hiện phép biến đổi ngược (`output / img_range + mean`) và `clamp(0, 1)` trước
khi trả về, để tương thích với phần còn lại của pipeline.

## Động lực

Kết quả Giai đoạn baseline (docs/02) cho thấy SPAN gốc — dù nhẹ và nhanh —
train bằng pixel loss thuần túy nên không được tối ưu trực tiếp cho tín hiệu
mà bài toán downstream (nhận diện danh tính/giới tính) thực sự cần. Mục tiêu
của giai đoạn này: cải thiện accuracy downstream **mà không thay đổi kiến
trúc SPAN** — tức không đánh đổi tốc độ/kích thước model đã đạt được.

## Phương pháp: kết hợp Knowledge Distillation + Identity-aware loss

### Thành phần loss

```
L_total = λ_pixel   · L1(SR_student, HR)
        + λ_distill · L1(SR_student, SR_teacher)
        + λ_identity· (1 − cos_sim(f(SR_student), f(HR)))
```

| Ký hiệu | Ý nghĩa |
|---|---|
| `SR_student` | Output của SPAN (model sẽ deploy) |
| `SR_teacher` | Output của EDSR đã train sẵn (Giai đoạn baseline), **đóng băng** |
| `f(·)` | Embedding từ recognition model đã train trên domain HR, **đóng băng**, dùng làm "giám khảo" |
| `HR` | Ảnh ground truth |

Giá trị `λ` mặc định: `λ_pixel=1.0, λ_distill=0.5, λ_identity=0.5`
(`configs/config.yaml → sr_improve`). Cần tinh chỉnh qua val set — ghi lại
mọi tổ hợp đã thử vào bảng ablation bên dưới.

### Tại sao 3 thành phần này, không phải chỉ 1

- **L1 với HR (pixel loss)**: giữ output không "chạy lung tung" khỏi ground
  truth thật — nếu chỉ tối ưu distillation/identity, model có thể học cách
  "đánh lừa" recognition model mà tạo ra ảnh không giống ear thật.
- **Distillation với teacher (EDSR)**: EDSR tuy không tối ưu cho bài toán
  downstream, nhưng chứa nhiều chi tiết hợp lý hơn những gì SPAN tự học được
  chỉ với pixel loss — cung cấp tín hiệu bổ sung để SPAN "học nhanh hơn, tốt
  hơn" trong cùng dung lượng tham số nhỏ.
- **Cosine loss ở tầng embedding (identity-aware)**: đây là thành phần trực
  tiếp gắn với mục tiêu cuối cùng — ép SPAN giữ lại đúng loại chi tiết mà
  recognition model cần để nhận ra cùng một người, thay vì chỉ tối ưu độ
  giống nhau ở mức pixel (PSNR/SSIM).

### Quy trình huấn luyện tuần tự (không train chung 3 mạng)

Dựa trên phát hiện của SR4IR (Task-Driven Perceptual Loss for Super-Resolution):
train tuần tự (SR trước, task-model sau) cho kết quả tốt hơn train chung
(joint end-to-end). Áp dụng:

1. Train Teacher (EDSR) trên domain HR — độc lập, đã có ở Giai đoạn baseline.
2. Train recognition model trên domain HR — độc lập, đã có ở Giai đoạn
   baseline, dùng làm giám khảo, **đóng băng hoàn toàn** trong bước 3.
3. Train Student (SPAN) với `L_total`, chỉ backprop qua Student
   (`train_sr_distill.py`).
4. Sau khi Student hội tụ, sinh lại toàn bộ `sr_improved/{train,val,test}`,
   rồi train lại recognition model (từng backbone) trên domain mới này —
   tách biệt hoàn toàn bước train SR và bước train recognition.

### Chi phí tính toán — chỉ ảnh hưởng lúc TRAIN, không ảnh hưởng lúc DEPLOY

Teacher (EDSR) và recognition-giám khảo chỉ cần cho **quá trình train Student**.
Lúc triển khai thực tế, chỉ chạy riêng SPAN đã cải tiến — params/FLOPs/latency
**giữ nguyên** so với SPAN baseline (kiến trúc không đổi). Đây là điểm mạnh
cốt lõi cần nhấn trong phần Discussion của bài báo: "cải thiện accuracy mà
không tốn thêm chi phí suy luận."

## Tiêu chí thành công (3 điều kiện, phải đạt đồng thời)

1. `acc(sr_improved) > acc(sr_baseline)` — chứng minh phương pháp cải tiến có
   tác dụng thật, không phải nhiễu.
2. `acc(sr_improved) > acc(lr_lr)` — mục tiêu tối thượng: SR nhẹ + cải tiến
   đáng để triển khai thay vì bỏ qua SR hoàn toàn.
3. `params(SPAN improved) == params(SPAN baseline)` và
   `latency(SPAN improved) ≈ latency(SPAN baseline)` — xác nhận cải tiến chỉ
   nằm ở cách train, không "ăn gian" bằng cách làm model to hơn.

## Ablation cần chạy (ghi kết quả vào đây để đưa vào bài báo)

| Cấu hình loss | λ_pixel | λ_distill | λ_identity | acc_id (trung bình 5 backbone) | acc_gender | Ghi chú |
|---|---|---|---|---|---|---|
| Chỉ pixel (= SPAN baseline) | 1.0 | 0 | 0 | | | Đối chứng |
| Pixel + distill | 1.0 | 0.5 | 0 | | | Cô lập tác dụng của distillation |
| Pixel + identity | 1.0 | 0 | 0.5 | | | Cô lập tác dụng của identity loss |
| Pixel + distill + identity (đề xuất) | 1.0 | 0.5 | 0.5 | | | Cấu hình đầy đủ |
| *(thử thêm các tổ hợp λ khác nếu cần)* | | | | | | |

## Kết quả cuối cùng so với baseline (điền sau khi chạy đầy đủ)

| Backbone | acc_lr_lr | acc_sr_baseline | acc_sr_improved | Đạt tiêu chí 1&2? |
|---|---|---|---|---|
| mobilenet_v2 | | | | |
| mobilenet_v3_small | | | | |
| resnet18 | | | | |
| efficientnet_b0 | | | | |
| ghostnet_100 | | | | |

| Chỉ số hiệu năng | SPAN baseline | SPAN improved | Chênh lệch |
|---|---|---|---|
| Params (M) | | | (phải ~bằng nhau) |
| FLOPs (G) | | | (phải ~bằng nhau) |
| Latency (ms, thiết bị: ______) | | | (phải ~bằng nhau) |

**Nếu bảng trên cho thấy accuracy tăng nhất quán trên đa số/toàn bộ 5
backbone, trong khi params/FLOPs/latency không đổi — đây chính là bằng chứng
đủ mạnh để viết thành bài báo journal**, với thông điệp: "SPAN cải tiến bằng
distillation kết hợp identity-aware loss cải thiện accuracy nhận diện nhất
quán trên nhiều kiến trúc recognition khác nhau, mà không tốn thêm chi phí
suy luận."

## [MỚI] Multi-Judge Ensemble Identity-Aware Distillation + Feature-level KD

### Động lực — chẩn đoán tại sao recipe cũ không tổng quát hoá tốt qua backbone

Bảng kết quả chính (bản thảo bài báo) cho thấy `span_tiny` cải thiện accuracy
so với `lr` (no-SR) KHÔNG đồng nhất giữa 5 backbone recognition — có ý nghĩa
thống kê ở 3/5 (`ghostnet_100`, `mobilenet_v2`, `efficientnet_b0`), chỉ ở
mức trend ở 1/5 (`mobilenet_v3_small`), và KHÔNG có bằng chứng nào ở
`resnet18`. Trên AWEx from-scratch, tình trạng còn rõ hơn: `span_tiny` có
lúc còn CAO HƠN `span_baseline` (`ghostnet_100`), có lúc thấp hơn có ý nghĩa
thống kê so với `lr` xét theo baseline (`resnet18`).

Chẩn đoán nguyên nhân hợp lý nhất: identity loss ở recipe cũ chỉ dùng **1
model giám khảo duy nhất** (`mobilenet_v2` domain HR). Khi tăng
`lambda_identity`, kết quả xấu đi có ý nghĩa thống kê (Cohen's d=-10.09) —
tức là SR bị ép tối ưu theo đúng "gu" của 1 kiến trúc, tạo ra đặc trưng ăn
khớp riêng với backbone đó nhưng KHÔNG tổng quát hoá được sang backbone
khác. Đồng thời, distillation loss cũ chỉ so khớp OUTPUT PIXEL cuối cùng —
dạng KD yếu nhất trong literature, không truyền được tín hiệu về cách
teacher tổ chức đặc trưng nội bộ.

### Phương pháp đề xuất (2 cơ chế, độc lập, bật/tắt qua config)

**(A) Multi-Judge Ensemble Identity Loss** — thay vì 1 recognition model giám
khảo, dùng đồng thời `K` backbone khác họ kiến trúc (mặc định
`mobilenet_v2` + `resnet18` + `ghostnet_100`, đều đã train sẵn ở domain HR,
đóng băng hoàn toàn):

```
L_identity = (1/K) * Σ_k (1 − cos(f_k(SR), f_k(HR)))
```

Cấu hình qua `configs/config.yaml -> sr_improve.identity_judges` (danh sách
`{backbone, ckpt}`). Nếu để trống, tự fallback về hành vi single-judge cũ
(tương thích ngược 100%, xem `train_sr_distill.py::build_judges()`).

**(B) Feature-level Knowledge Distillation (hint-based)** — bổ sung BÊN
CẠNH distillation output-level cũ (không thay thế), khớp feature map ngay
trước pixel-shuffle của student (qua 1 adapter Conv1x1 học được, chỉ tồn tại
lúc train) với feature map tương ứng của teacher:

```
L_feat = || Adapter(F_student) − F_teacher ||_1
L_total = λ_pixel·L1(SR,HR) + λ_distill·L1(SR,SR_teacher)
        + λ_feat·L_feat + λ_identity·L_identity(multi-judge)
```

Trích feature qua `models/sr_models.py::SRFeatureHook` — dùng forward hook
đăng ký theo TYPE (`nn.PixelShuffle`), không theo tên thuộc tính nội bộ, nên
hoạt động được với `span_official` (code ngoài từ `external/SPAN/`, không
kiểm soát được tên thuộc tính) lẫn mọi kiến trúc tự viết trong
`models/sr_models.py`. Cấu hình qua `sr_improve.lambda_feat` (mặc định 0.0
= tắt hoàn toàn, không tốn thêm chi phí — bật bằng cách đặt >0).

Cả 2 cơ chế đã được **functional-test** (forward + backward + optimizer
step qua PyTorch thật, không chỉ kiểm tra cú pháp) trước khi coi là hoàn
tất — xác nhận: gradient chảy đúng về cả student lẫn feature-adapter, phép
trung bình multi-judge thu gọn đúng về công thức single-judge cũ khi
`identity_judges` rỗng, và đường tắt `lambda_feat=0` không tạo chi phí thừa.

### Không ảnh hưởng chi phí lúc DEPLOY

Cả feature adapter (Conv1x1) lẫn toàn bộ judge model CHỈ tồn tại trong quá
trình train — checkpoint lưu ra (`student.state_dict()`) không đổi cấu trúc
so với recipe cũ, params/FLOPs/latency lúc triển khai **giữ nguyên**.

### Ablation cần chạy trước khi đưa vào bài báo

```bash
bash pipeline/run_ablation_kd_v2.sh
```

Chạy 2x2 factorial (`kdv2_baseline`, `kdv2_feat`, `kdv2_multijudge`,
`kdv2_full`) trên 1 backbone đại diện (`mobilenet_v2`, giống quy mô
`run_ablation.sh` cũ) — CHỈ là tín hiệu sàng lọc nhanh.

### Validate đầy đủ (multi-seed x 5-backbone) — BẮT BUỘC trước khi công bố

Sau khi xác định cấu hình thắng ở ablation trên, sửa 2 biến `LAMBDA_FEAT`/
`LAMBDA_IDENTITY` đầu file `pipeline/run_multi_seed_kdv2.sh` cho khớp, rồi:

```bash
bash pipeline/run_multi_seed_kdv2.sh
```

Script train lại SR **1 lần** (seed cố định, đúng protocol thống kê của
project — xem mục 3.5 bản thảo bài báo: chỉ downstream recognition lặp qua
seed để đo phương sai, không train lại SR nhiều seed vì tốn kém), sau đó
train + eval recognition qua **5 backbone x 3 seed** (mặc định `42 123
2024`, có thể thêm `44 999` để khớp n=5 như bản thảo bài báo) trên domain
mới `sr_improved_kdv2`, tái sử dụng ĐÚNG checkpoint `recognition_lr_*_seed*`
đã có từ `pipeline/run_multi_seed.sh` (yêu cầu chạy trước) — giữ đúng chuỗi
fine-tune `hr -> lr -> sr_improved_kdv2`, không dùng chung 1 checkpoint `hr`.

Kết quả: `results/multi_seed_kdv2/multi_seed_kdv2_summary.csv`. Để so sánh
trực tiếp (paired t-test + Cohen's d) với `lr` (no-SR) và `sr_improved`
(recipe `span_tiny` CŨ) đã có sẵn, copy 2 bộ JSON đó vào cùng thư mục rồi
chạy lại `data/aggregate_multi_seed_results.py` — script tự gộp theo domain
đọc từ nội dung JSON (xem hướng dẫn in ra cuối `run_multi_seed_kdv2.sh`), tự
sinh bảng kiểm định cho MỌI cặp domain kể cả `sr_improved` (cũ) vs
`sr_improved_kdv2` (mới) — tránh lặp lại giới hạn "chỉ 1 backbone" đã bị nêu
ra khi review bản thảo hiện tại (Table 3, mục Limitations #4).

| Cấu hình | λ_pixel | λ_distill | λ_feat | λ_identity | Giả thuyết kiểm định |
|---|---|---|---|---|---|
| `kdv2_baseline` | 1.0 | 1.0 | 0.0 | 0.0 | Đối chứng — đúng recipe cũ |
| `kdv2_feat` | 1.0 | 1.0 | 0.5 | 0.0 | Feature-KD một mình có giúp không? |
| `kdv2_multijudge` | 1.0 | 1.0 | 0.0 | 0.1 | Multi-judge một mình có giúp không (khác single-judge cũ đã thất bại)? |
| `kdv2_full` | 1.0 | 1.0 | 0.5 | 0.1 | Kết hợp cả 2 — cấu hình đề xuất |

**Lưu ý quan trọng**: `lambda_feat=0.5` và `lambda_identity=0.1` (multi-judge)
ở bảng trên là giá trị KHỞI ĐIỂM hợp lý, CHƯA qua sweep — coi bảng ablation
2x2 này là bước đầu xác định "cơ chế nào có tác dụng theo hướng nào", còn
GIÁ TRỊ lambda tối ưu cần 1 lần sweep riêng (tương tự
`pipeline/run_lambda_sweep.sh` đã làm cho `lambda_identity` bản cũ) trước
khi chốt số liệu chính thức.

### Kết quả ablation KD v2 (đã chạy, 1 backbone `mobilenet_v2`, 1 seed — chỉ tín hiệu sàng lọc)

| Cấu hình | identity_accuracy (mobilenet_v2) | Chênh lệch so với `kdv2_baseline` |
|---|---|---|
| `kdv2_baseline` | 0.5299 | — |
| `kdv2_feat` | 0.5236 | −0.0063 (kém hơn) |
| `kdv2_multijudge` | 0.5338 | +0.0039 |
| `kdv2_full` | **0.5397** | **+0.0098 (thắng, dùng cho multi-seed)** |

### Kết quả validate đầy đủ (multi-seed x 5 backbone, n=3 seed: 42/123/2024 — ĐÃ CHẠY)

So `sr_improved` (recipe cũ, chỉ pixel+distill) vs `sr_improved_kdv2` (recipe
mới, `kdv2_full`: λ_feat=0.5, λ_identity=0.1 multi-judge):

| Backbone | mean_diff (kdv2 − cũ) | Cohen's d | p_bonferroni | Kết luận |
|---|---|---|---|---|
| EfficientNet-B0 | +0.0031 | 0.39 | 1.0 | không có ý nghĩa |
| GhostNet-100 | +0.0122 | 3.45 | 0.0807 | **có ý nghĩa** (ngưỡng 0.10) |
| MobileNetV2 | +0.0040 | 0.72 | 1.0 | không có ý nghĩa |
| MobileNetV3-Small | **−0.0094** | −0.96 | 1.0 | không có ý nghĩa, **hướng âm** |
| ResNet-18 | +0.0063 | 0.55 | 1.0 | không có ý nghĩa |

**Đánh giá theo tiêu chí thành công đã đặt ra ở trên:**

1. ❌ Tỷ lệ backbone đạt ý nghĩa cho "cải tiến > recipe cũ": chỉ **1/5**
   (`ghostnet_100`), ở ngưỡng α=0.10 (không đạt α=0.05 chuẩn).
2. ⚠️ Có cải thiện `resnet18` theo hướng đúng (recipe cũ: `tiny vs lr` KHÔNG
   có ý nghĩa với p_bonf=0.1768; recipe KDv2: CÓ ý nghĩa với p_bonf=0.0799)
   — nhưng đồng thời làm **`mobilenet_v3_small` từ có ý nghĩa (recipe cũ,
   p_bonf=0.0566) chuyển sang không có ý nghĩa (recipe KDv2, p_bonf=0.671,
   diff âm)**.
3. ❌ Có "được chỗ này mất chỗ kia" — vi phạm tiêu chí (3) đã đặt ra.
4. Chưa chạy trên AWEx.

**Kết luận trung thực**: recipe KDv2 (multi-judge + feature-KD) có tác dụng
thật nhưng **nhỏ và không nhất quán** — nó **đổi backbone nào là điểm yếu**
(từ resnet18 sang mobilenet_v3_small) thay vì xoá bỏ sự không nhất quán giữa
backbone. n=3 seed có power thống kê yếu; xem
`pipeline/run_multi_seed_kdv2_extra_seeds.sh` (thêm seed 44, 999 → n=5) để
kiểm tra lại trước khi kết luận cuối cùng liệu đây có phải hiệu ứng thật nhỏ
hay chỉ do thiếu mẫu.

## [MỚI, ĐÃ THỬ NHƯNG THẤT BẠI] Saliency-Weighted Identity-Critical Loss

> **Kết luận sau khi chạy sweep thật (đợt 9)**: cơ chế này KHÔNG cải thiện
> accuracy ở bất kỳ mức λ nào đã thử — xem bảng kết quả cuối mục này.
> **KHÔNG đưa vào danh sách đóng góp chính của bài báo.** Giữ lại mục này
> trong docs như 1 negative result đã được kiểm định nghiêm túc (không phải
> chỉ là ý tưởng chưa thử), có thể nhắc ngắn trong phần Discussion/Limitations
> nếu muốn thể hiện đã khảo sát hướng này. `configs/config.yaml` giữ
> `lambda_saliency` mặc định = 0.0 (tắt hoàn toàn) — không dùng trong bất kỳ
> pipeline chính thức nào (`run_multi_seed_kdv2.sh`,
> `run_multi_seed_learned_prune.sh` đều pin cứng `lambda_saliency=0`).

### Động lực

Phân tích định tính trong bản thảo bài báo (Figure 2b) chỉ ra 1 điểm mù của
recipe hiện tại: PSNR gần như hoà giữa 2 cấu hình (30.92 vs 30.89 dB) hoá ra
là do model tái tạo TỐT phần TÓC/NỀN, chứ không phải phần TAI (bị che khuất
trong mẫu đó) — tức là pixel loss L1 đồng đều trên toàn ảnh đang "phí" một
phần dung lượng học vào vùng KHÔNG quan trọng cho tác vụ nhận dạng downstream.
Đây là lỗ hổng khá cơ bản: loss huấn luyện SR không hề biết "vùng nào của
ảnh mới thực sự quyết định danh tính".

### Phương pháp

Dùng CHÍNH hội đồng judge (multi-judge, mục trên) để suy ra 1 saliency map
KHÔNG CẦN NHÃN SEGMENTATION TAI MỚI (EarVN1.0 không có, annotate tay tốn kém):
với mỗi judge, tính gradient của "năng lượng embedding" `||f(x)||²` theo
TỪNG PIXEL ảnh HR đầu vào — pixel nào có gradient lớn là pixel ảnh hưởng
nhiều nhất đến đặc trưng nhận dạng theo góc nhìn của judge đó. Trung bình
saliency qua tất cả judge, chuẩn hoá về [0.3, 1.0] (không triệt tiêu hoàn
toàn vùng "ít quan trọng", tránh model bỏ mặc hẳn 1 phần ảnh), rồi dùng làm
trọng số không gian cho 1 pixel loss BỔ SUNG (`lambda_saliency`, cộng thêm
bên cạnh `lambda_pixel` đồng đều, không thay thế):

```
saliency(x) = trung_bình_qua_judge( |∂||judge.embed(x)||² / ∂x| )   # chuẩn hoá [0.3, 1.0]
loss_saliency = mean( saliency ⊙ |SR - HR| )
```

Xem cài đặt đầy đủ trong `train_sr_distill.py::compute_multi_judge_saliency()`
và cách nối vào tổng loss trong `compute_total_loss()`.

**Vì sao đây là novelty thật (không chỉ là "thêm 1 hệ số")**: các công trình
SR-cho-nhận-dạng được trích trong Related Work (Ataer-Cansizoglu et al.,
Nguyen et al., Ribeiro et al.) đều chỉ so khớp EMBEDDING TOÀN CỤC (identity
loss dạng cosine/L2 trên vector đặc trưng) — KHÔNG có công trình nào trong
nhóm này dùng chính gradient của recognition model để tạo trọng số KHÔNG
GIAN cho pixel loss. Đây là 1 dạng "self-supervised region-of-interest" suy
ra hoàn toàn từ model đã có, không cần thêm dữ liệu/nhãn/model nào mới.

### Chi phí — chỉ ảnh hưởng lúc TRAIN

`compute_multi_judge_saliency()` cần 1 lần forward+backward bổ sung QUA TỪNG
judge (chỉ khi `lambda_saliency > 0`), KHÔNG ảnh hưởng gì đến checkpoint/
params/FLOPs/latency lúc deploy (giống hệt các judge khác, chỉ tồn tại lúc
train).

### Cần sweep trước khi đưa vào bài báo

```bash
bash pipeline/run_lambda_saliency_sweep.sh
```

Quét `lambda_saliency` (0.0, 0.15, 0.3, 0.6, 1.0) x 3 seed, cố định
`lambda_feat`/`lambda_identity` ở cấu hình đã thắng từ `run_ablation_kd_v2.sh`
(sửa 2 biến đầu file trước khi chạy). Kết quả:
`results/lambda_saliency_sweep/saliency_sweep_summary.csv` (đã gồm paired
t-test + Cohen's d so với `lambda_saliency=0.0`).

### Kết quả sweep (ĐÃ CHẠY, 1 backbone `mobilenet_v2`, n=3 seed) — kết quả ÂM

| λ_saliency | mean_identity_accuracy | Cohen's d vs baseline (λ=0) | p_bonferroni | Kết luận |
|---|---|---|---|---|
| 0.0 (đối chứng) | 0.5362 | — | — | — |
| 0.15 | 0.5261 | −0.78 | 1.0 | kém hơn, không có ý nghĩa |
| 0.3 | 0.5338 | −0.36 | 1.0 | kém hơn, không có ý nghĩa |
| 0.6 | 0.5197 | **−2.13** | 0.265 | kém hơn, GẦN đạt ý nghĩa (sai hướng) |
| 1.0 | 0.5314 | −0.62 | 1.0 | kém hơn, không có ý nghĩa |

**Mọi mức λ_saliency được thử đều cho Cohen's d ÂM** (accuracy thấp hơn đối
chứng), không mức nào cho tín hiệu dương dù chỉ là trend nhẹ. Ở mức λ=0.6,
hiệu ứng xấu đi gần đạt ý nghĩa thống kê (p_raw=0.066) — nếu có ý nghĩa thì
lại là "λ_saliency làm accuracy xấu đi có ý nghĩa", không phải cải thiện.

**Diễn giải khả dĩ** (chưa kiểm chứng thêm, ghi lại để tham khảo nếu muốn thử
tiếp): (a) gradient-saliency từ judge có thể nhiễu/không ổn định ở ảnh nhỏ độ
phân giải thấp (LR 20×20, HR 80×80 theo scale=4 của project này), khiến
trọng số không gian suy ra không đáng tin; (b) cộng thêm 1 loss trọng số theo
không gian có thể xung đột gradient với pixel loss đồng đều + distill loss đã
có, thay vì bổ trợ; (c) dải λ đã thử có thể chưa đúng — chưa thử λ rất nhỏ
(<0.15) là khoảng có khả năng ít gây hại nhất nhưng cũng ít khả năng đủ mạnh
để thấy hiệu ứng dương. Không tiếp tục sweep thêm trừ khi có thời gian
GPU dư và một giả thuyết cụ thể hơn để kiểm chứng.

## [MỚI] Differentiable/Learned Block Pruning

### Động lực

Bản thảo bài báo tự nhận việc `span_tiny` "giữ khối 1-3, bỏ khối 4-6" là
"a choice made for implementation convenience rather than validated against
alternatives" (mục 5.7-ii — hạn chế đã tự nêu, chưa giải quyết). Đây cũng là
nguồn gốc hợp lý nhất của hiện tượng "`span_tiny` không đồng nhất giữa các
backbone/dataset": cắt cố định theo VỊ TRÍ, không theo MỨC ĐỘ ĐÓNG GÓP thực
tế — khối bị cắt có thể quan trọng với backbone/dataset này nhưng không
quan trọng với backbone/dataset khác.

### Phương pháp

`models/sr_models.py::SPANLearnedPrune` — SPAN với 1 gate liên tục học được
cho MỖI khối SPAB (`gate_i = sigmoid(alpha_i)`, `alpha_i` là 1 scalar):

```
O_i = O_{i-1} + gate_i * (SPAB_i(O_{i-1}) - O_{i-1})
```

`gate_i=0` → bỏ hoàn toàn khối i; `gate_i=1` → áp dụng đầy đủ. Khởi tạo TẤT
CẢ gate gần 1 (hành vi ban đầu giống `span_large`, ngân sách đầy đủ 6 khối),
huấn luyện với ĐÚNG loss downstream đã có (pixel + distill + feature-KD +
identity, tái sử dụng 100% `compute_total_loss()` từ `train_sr_distill.py`
— **KHÔNG gồm saliency-weighted loss**, xem mục phía trên: sweep cho kết quả
âm nên bị loại khỏi recipe mặc định, `lambda_saliency` luôn pin=0 ở cả 2
script sàng lọc/validate pruning) cộng thêm 1 sparsity penalty:

```
loss_total = compute_total_loss(...) + lambda_sparsity * mean(gate)
```

Khối nào bị loss downstream "bảo vệ" (gỡ ra làm loss tăng nhiều hơn phần
tiết kiệm từ sparsity penalty) giữ gate cao; khối không đóng góp nhiều bị
đẩy gate về gần 0. Sau khi train xong, `harden_and_export()` XOÁ HẲN khối có
gate dưới ngưỡng (`gate_harden_threshold`, mặc định 0.5) khỏi `ModuleList`,
trả về 1 model **SPAN thường** (không còn gate/tham số phụ nào) — tái sử
dụng nguyên vẹn hạ tầng đo params/FLOPs/latency/eval đã có cho SPAN thường,
không cần code path riêng.

**Vì sao đây là novelty thật**: đây KHÔNG phải magnitude-based pruning kinh
điển (dựa vào độ lớn trọng số) — quyết định giữ/bỏ khối được dẫn dắt bởi TOÀN
BỘ loss downstream (bao gồm cả identity-aware loss + feature-KD ở trên), tức
là pruning "biết" tác vụ cuối cùng là nhận dạng vành tai, không chỉ là tái
tạo pixel. Đây cũng là câu trả lời NGUYÊN TẮC (tự động, tái lập được) cho
câu hỏi "vị trí cắt khối có quan trọng không?" mà bài báo mới chỉ đặt ra như
1 hạn chế/ablation dự kiến.

### Huấn luyện + cứng hoá

```bash
python train_sr_learned_prune.py --config configs/config.yaml
```

Script tự động: (1) train `SPANLearnedPrune` (ngân sách `n_blocks_budget=6`)
tới early-stop; (2) cứng hoá bằng `harden_and_export()`; (3) lưu checkpoint
deploy-ready tương thích `build_sr_model("span_pruned", scale, n_blocks=K)`
vào `best.pt` + metadata (`prune_metadata.json`, ghi lại khối nào bị
giữ/bỏ và giá trị gate cuối — dùng vẽ hình minh hoạ "learned pruning
pattern" cho bài báo).

### Cần sweep + validate trước khi đưa vào bài báo

```bash
bash pipeline/run_prune_sparsity_screen.sh          # bước 1: sàng lọc nhanh lambda_sparsity (1 backbone, 1 seed/mức)
bash pipeline/run_multi_seed_learned_prune.sh       # bước 2: validate multi-seed x 5-backbone (sau khi chọn xong lambda_sparsity)
bash pipeline/run_multi_seed_learned_prune_extra_seeds.sh  # bước 3 (tuỳ chọn): thêm seed 44,999 -> n=5
```

> **⚠️ Lần chạy đầu (đợt 8) BỊ LỖI THIẾT KẾ — kết quả KHÔNG dùng được, đã sửa
> lại script, cần chạy lại toàn bộ (đợt 9)**: cả 2 script trên đã chạy với
> `LAMBDA_FEAT=LAMBDA_SALIENCY=LAMBDA_IDENTITY=0.0` (giá trị mặc định lúc đó,
> chưa được sửa tay theo hướng dẫn đầu file) VÀ `LAMBDA_SPARSITY=0.05`. Xem
> `prune_metadata.json` của lần chạy đó: `"n_blocks_kept": 6` (bằng đúng
> `n_blocks_budget`) và `"identity_aware": false, "uses_feature_kd": false`.
> Nghĩa là (1) mô hình **không cắt bỏ khối nào cả** — so sánh
> `sr_improved` (span_tiny, 3 khối) vs `sr_learned_prune` (6 khối, KHÔNG nén)
> trong `results/multi_seed_learned_prune/` chỉ phản ánh "model to hơn nhẹ
> khá hơn", KHÔNG liên quan gì đến cơ chế learned-pruning; (2) do cả 3 λ
> identity-liên-quan = 0, đây chỉ là "reconstruction-aware pruning" thuần,
> KHÔNG PHẢI "identity-aware learned pruning" như tên gọi/novelty claim của
> mục này. **Đã sửa 2 script trên** (đợt 9): pin
> `LAMBDA_FEAT=0.5, LAMBDA_IDENTITY=0.1` (khớp `kdv2_full` — cấu hình thắng ở
> KDv2) và mở rộng dải `LAMBDA_SPARSITY_VALUES` sang khoảng (0.05, 0.2) để dò
> mốc cho `n_blocks_kept` gần 3 nhất (dải cũ 0/0.01/0.05 đều cho 6 khối, 0.1
> cho 5 khối, 0.2 cho 2 khối — không có mốc nào đúng 3). **Phải chạy lại từ
> đầu Bước 1 (`run_prune_sparsity_screen.sh`)** rồi cập nhật `LAMBDA_SPARSITY`
> trong `run_multi_seed_learned_prune.sh` theo kết quả mới, TRƯỚC khi có thể
> đưa số liệu learned-pruning vào bài báo.

### Dùng checkpoint đã cứng hoá ở các bước sau

Y HỆT cách dùng `span_tiny`, chỉ khác cờ kiến trúc:

```bash
python data/build_sr.py --arch span_pruned --n_blocks <K> --sr_ckpt runs/sr_learned_prune*/best.pt ...
python eval_sr_quality.py --arch span_pruned --n_blocks <K> --ckpt runs/sr_learned_prune*/best.pt ...
```

(`<K>` = `n_blocks_kept` đọc từ `prune_metadata.json` ghi kèm checkpoint đó.)

## Script liên quan

- `train_sr_distill.py` — huấn luyện SPAN cải tiến (multi-judge identity loss
  + feature-level KD + saliency-weighted identity-critical loss + distillation
  output-level + pixel loss)
- `train_sr_learned_prune.py` — [MỚI] học pruning độ sâu có giám sát (gate học
  được cho từng khối SPAB, tái sử dụng toàn bộ loss ở trên + sparsity penalty)
- `data/build_sr.py` — sinh `splits/sr_improved`/`splits/sr_pruned` từ
  checkpoint đã cải tiến (hỗ trợ `--arch span_pruned --n_blocks K`)
- `models/sr_models.py` — định nghĩa SPAN, SPAB, EDSR, `SRFeatureHook`,
  `SPANLearnedPrune` ([MỚI])
- `pipeline/run_ablation_kd_v2.sh` + `data/aggregate_ablation_kd_v2_results.py`
  — ablation 2x2 cho feature-KD + multi-judge
- `pipeline/run_lambda_saliency_sweep.sh` + `data/aggregate_saliency_sweep.py`
  — [MỚI] sweep lambda_saliency
- `pipeline/run_prune_sparsity_screen.sh` — [MỚI] sàng lọc lambda_sparsity cho
  learned pruning
- `pipeline/run_multi_seed_learned_prune.sh` — [MỚI] validate đầy đủ learned
  pruning (multi-seed x 5-backbone)
- `pipeline/run_multi_seed_kdv2_extra_seeds.sh` — [MỚI, đợt 9] thêm seed
  44,999 cho domain `sr_improved_kdv2` (n=3 -> n=5, không train lại SR)
- `pipeline/run_multi_seed_learned_prune_extra_seeds.sh` — [MỚI, đợt 9] thêm
  seed 44,999 cho domain `sr_learned_prune` (n=3 -> n=5, không train lại SR)
