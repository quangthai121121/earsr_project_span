# 02 — Thí nghiệm baseline

## Mục tiêu

Trước khi đầu tư vào việc cải tiến SPAN, cần xác nhận: (a) hướng SR có tiềm
năng thật cho bài toán ear recognition hay không (trần lý thuyết), và (b) mức
độ SR nhẹ (SPAN gốc) hiện đang thua/thắng baseline không dùng SR bao nhiêu —
đây là "khoảng trống" mà Giai đoạn cải tiến (docs/03) cần lấp.

## Ma trận thí nghiệm

| Cấu hình | Train trên | Test trên | Vai trò |
|---|---|---|---|
| `hr_hr` | HR | HR | Trần lý thuyết (upper bound tuyệt đối) |
| `lr_lr` | LR | LR | **Baseline chính (không SR)** — mục tiêu cần vượt qua |
| `sr_baseline` (EDSR teacher) | HR (train EDSR) | — | Trần lý thuyết cho *hướng SR* (không phải cho recognition) |
| `sr_baseline` (SPAN gốc) | SR (SPAN, pixel loss thuần) | SR | SR nhẹ chưa cải tiến — điểm khởi đầu |

Toàn bộ ma trận được nhân với **5 backbone recognition** để đảm bảo kết luận
không phụ thuộc vào một kiến trúc cụ thể: `mobilenet_v2`, `mobilenet_v3_small`,
`resnet18`, `efficientnet_b0`, `ghostnet_100` (dòng kiến trúc chuyên cho
lightweight face/biometric recognition, nền tảng của GhostFaceNet).

## Quyết định huấn luyện: fine-tune, không train from-scratch

Recognition model trên các domain `lr`, `sr_baseline`, `sr_improved` được
**fine-tune** từ checkpoint đã train trên domain `hr` (hoặc `lr`) cùng
backbone, thay vì train from-scratch. Lý do: EarVN1.0 có 164 identity — không
đủ lớn để train ổn định from-scratch nhiều backbone khác nhau trong thời gian
hạn chế của luận án; fine-tune hội tụ nhanh hơn và ổn định hơn.

Checkpoint dùng để chọn mô hình tốt nhất luôn dựa trên **val set cùng domain**
với domain đang train — không dùng val của domain khác để chọn checkpoint
(tránh mismatch giữa tiêu chí chọn model và domain triển khai thực tế).

## Giả thuyết cần kiểm chứng

> SPAN gốc (train bằng pixel loss L1 thuần túy, tối ưu PSNR) sẽ cho
> `identity_accuracy` **thấp hơn** baseline `lr_lr` trên ít nhất một vài
> backbone — do pixel loss có xu hướng làm mượt (over-smooth) mất chi tiết
> tần số cao (texture da, nếp gấp sụn tai) vốn quan trọng cho nhận diện danh
> tính, dù ảnh SR "nhìn đẹp hơn" bằng mắt thường / có PSNR cao hơn.

Nếu giả thuyết đúng → đây chính là động lực khoa học cho Giai đoạn cải tiến
(docs/03_span_improvement.md). Nếu giả thuyết sai (SPAN gốc đã thắng `lr_lr`)
→ vẫn tiếp tục Giai đoạn cải tiến, nhưng đổi khung diễn giải sang "cải thiện
thêm biên độ, chuyên biệt hóa SPAN cho task ear recognition cụ thể" thay vì
"sửa từ thua thành thắng" — vẫn là đóng góp hợp lệ.

## Kết quả (điền sau khi chạy `run_full_pipeline.sh` các bước 3–8)

| Backbone | acc_hr_hr | acc_lr_lr | acc_sr_baseline | Δ (sr_baseline − lr_lr) |
|---|---|---|---|---|
| mobilenet_v2 | | | | |
| mobilenet_v3_small | | | | |
| resnet18 | | | | |
| efficientnet_b0 | | | | |
| ghostnet_100 | | | | |

*(cột Δ âm ở càng nhiều backbone → giả thuyết càng được củng cố, và là bằng
chứng định lượng mạnh cho phần Introduction/Motivation của bài báo)*

## Script liên quan

- `train_sr.py --sr_arch edsr` — train teacher nặng
- `train_sr.py --sr_arch span` — train SPAN baseline (pixel loss thuần)
- `data/build_sr.py` — sinh ảnh SR từ checkpoint SPAN baseline
- `train_recognition.py`, `eval_recognition.py` — train/test recognition theo domain+backbone
