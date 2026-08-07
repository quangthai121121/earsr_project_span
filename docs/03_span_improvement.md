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

## Script liên quan

- `train_sr_distill.py` — huấn luyện SPAN cải tiến
- `data/build_sr.py` — sinh `splits/sr_improved` từ checkpoint đã cải tiến
- `models/sr_models.py` — định nghĩa SPAN, SPAB, EDSR
