# Thứ tự chạy các script

Chạy **lần lượt từng script theo số thứ tự**, mỗi script in rõ tiến trình
(BƯỚC x/8) và ghi log vào `runs/<tên_run>/train.log`. Có thể dừng giữa chừng
và chạy tiếp script sau — không cần chạy lại từ đầu.

| # | Script | Việc làm | Thời gian ước tính |
|---|---|---|---|
| 0 | *(thủ công)* | Copy dữ liệu EarVN1.0 vào `raw_data/EarVN1.0_raw/` | — |
| 1 | `01_survey_and_prepare_data.sh` | Khảo sát, **tự động chọn ngưỡng**, cập nhật config, tạo HR/LR | Vài phút |
| 2 | `02_setup_span_official.sh` | Tải code + checkpoint SPAN chính thức | Vài phút (+ 1 lần cần bạn xác nhận thủ công) |
| 3 | `03_train_baseline_recognition.sh` | Train recognition domain HR + LR, 5 backbone | Vài giờ |
| 4 | `04_train_teacher_and_span_baseline.sh` | Train EDSR (teacher) + fine-tune SPAN baseline | Vài giờ |
| 5 | `05_train_recognition_sr_baseline.sh` | Train recognition domain sr_baseline, 5 backbone | Vài giờ |
| 6 | `06_improve_span.sh` | Cải tiến SPAN (distillation + identity loss) | 1-2 giờ |
| 7 | `07_train_recognition_sr_improved.sh` | Train recognition domain sr_improved, 5 backbone | Vài giờ |
| 8 | `08_benchmark_and_aggregate.sh` | Test toàn bộ, tổng hợp `results/summary.csv` | Vài phút |

## Chạy nhanh (tất cả 1 lệnh, sau khi đã copy data vào Bước 0)

```bash
bash pipeline/run_all.sh
```

Script này sẽ **tự dừng lại** sau Bước 2 nếu bạn chưa xác nhận checkpoint SPAN
thủ công (xem hướng dẫn in ra khi chạy) — chạy lại `bash pipeline/run_all.sh`
sau khi xác nhận xong để tiếp tục từ Bước 3.

## Chạy ablation (tùy chọn, sau khi xong Bước 5)

```bash
bash pipeline/run_ablation.sh
```

Chạy 4 cấu hình loss (pixel_only / pixel_distill / pixel_identity / full) để
cô lập tác dụng từng thành phần, xuất `results/ablation.csv`. Mặc định chỉ
dùng 1 backbone (`mobilenet_v2`) để tiết kiệm thời gian.

## Khuyến nghị cho lần chạy đầu tiên

Trước khi chạy full cả 5 backbone (tốn nhiều giờ), nên sửa tạm biến
`BACKBONES` trong các script 03/05/07 chỉ còn `("mobilenet_v2")` để chạy thử
toàn bộ 8 bước với 1 backbone duy nhất trước — xác nhận không lỗi runtime,
rồi mới chạy lại full 5 backbone.
