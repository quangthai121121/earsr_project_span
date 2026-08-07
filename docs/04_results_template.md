# 04 — Bảng kết quả tổng hợp (điền sau khi chạy `run_full_pipeline.sh`)

Nguồn dữ liệu: `results/summary.csv` (sinh tự động bởi `data/aggregate_results.py`).

## Bảng chính — Accuracy theo backbone × domain

| Backbone | Params (M) | acc_hr_hr | acc_lr_lr | acc_sr_baseline | acc_sr_improved | Δ so với lr_lr | Δ so với sr_baseline |
|---|---|---|---|---|---|---|---|
| mobilenet_v2 | | | | | | | |
| mobilenet_v3_small | | | | | | | |
| resnet18 | | | | | | | |
| efficientnet_b0 | | | | | | | |
| ghostnet_100 | | | | | | | |
| **Trung bình** | | | | | | | |

## Bảng phụ — Gender accuracy theo backbone × domain

| Backbone | acc_hr_hr | acc_lr_lr | acc_sr_baseline | acc_sr_improved |
|---|---|---|---|---|
| mobilenet_v2 | | | | |
| mobilenet_v3_small | | | | |
| resnet18 | | | | |
| efficientnet_b0 | | | | |
| ghostnet_100 | | | | |

## Bảng hiệu năng tính toán — SR models (Pareto: accuracy vs chi phí)

| Model SR | Params (M) | FLOPs (G, @32×32→128×128) | Latency (ms, GPU dev) | Latency (ms, thiết bị edge: ______) |
|---|---|---|---|---|
| EDSR (teacher) | | | | |
| SPAN baseline | | | | |
| SPAN improved | | | *(phải ≈ SPAN baseline)* | *(phải ≈ SPAN baseline)* |

## Bảng LR thật (real_lr_holdout) — kiểm chứng tổng quát hóa

So sánh SPAN baseline vs SPAN improved khi áp dụng lên `real_lr_holdout.json`
(ảnh vốn dĩ đã nhỏ ngoài đời, không phải downsample bicubic nhân tạo) — dùng
recognition model tốt nhất từ bảng chính.

| | acc trên real_lr_holdout | Nhận xét |
|---|---|---|
| Không SR (đưa thẳng ảnh real-LR vào recognition) | | |
| SR baseline | | |
| SR improved | | |

*(Bảng này quan trọng cho phần Discussion: chứng minh phương pháp không chỉ
hiệu quả trên suy giảm bicubic tự tạo mà còn tổng quát hóa sang suy giảm thật)*

## Checklist trước khi viết bản thảo journal

- [ ] Đã điền đủ số liệu percentile độ phân giải thật (docs/01)
- [ ] Đã xác nhận/phủ nhận giả thuyết SPAN baseline thua lr_lr (docs/02)
- [ ] Đã chạy đủ ablation 4 cấu hình loss (docs/03)
- [ ] `acc_sr_improved > acc_lr_lr` đạt trên ít nhất 4/5 backbone
- [ ] Params/FLOPs/latency của SPAN improved ≈ SPAN baseline (chênh lệch < 5%)
- [ ] Đã đo latency thật trên thiết bị edge đích (không chỉ GPU train)
- [ ] Đã chạy kiểm chứng trên `real_lr_holdout` (LR thật, không phải downsample)
- [ ] Đã vẽ biểu đồ Pareto (accuracy vs latency) so sánh SPAN baseline/improved
      với các SR khác đã khảo sát (nếu có)
