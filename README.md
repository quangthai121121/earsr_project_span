# EarSR-Recognition: SPAN nhẹ + benchmark đa backbone cho nhận diện danh tính/giới tính qua ảnh tai

Khung thí nghiệm đầy đủ cho luận án: cải tiến **SPAN** (Swift Parameter-free
Attention Network — vô địch NTIRE 2024 Efficient SR Challenge, baseline chính
thức NTIRE 2026) để tăng accuracy nhận diện trên ảnh tai độ phân giải thấp
(EarVN1.0), trong khi giữ nguyên tốc độ/kích thước nhẹ — dùng knowledge
distillation từ teacher EDSR + identity-aware loss.

## Tài liệu phương pháp (đọc theo thứ tự, dùng viết journal)

| File | Nội dung |
|---|---|
| `docs/01_data_preparation.md` | Cách chuẩn bị EarVN1.0: khảo sát, phân nhóm, letterbox |
| `docs/02_baseline_experiments.md` | Thí nghiệm baseline: hr_hr, lr_lr, sr_baseline (SPAN gốc) |
| `docs/03_span_improvement.md` | Phương pháp cải tiến SPAN (distillation + identity loss), ablation |
| `docs/04_results_template.md` | Template bảng kết quả cuối cùng, checklist trước khi viết bản thảo |

**Mọi kết quả số liệu (accuracy, params, latency...) trong 4 file trên đều để
trống — điền vào sau khi bạn chạy pipeline trên dữ liệu và phần cứng thật.**

## Cách chạy — theo thứ tự script trong `pipeline/`

**Xem chi tiết đầy đủ tại `pipeline/README.md`** — bảng liệt kê rõ 8 script
theo đúng thứ tự chạy, việc mỗi script làm, và thời gian ước tính.

Tóm tắt nhanh:

```bash
pip install -r requirements.txt --break-system-packages

# 1. Copy dữ liệu EarVN1.0 vào raw_data/EarVN1.0_raw/
#    (xem raw_data/EarVN1.0_raw/README_ĐẶT_DATA_VÀO_ĐÂY.md)

# 2. Chạy toàn bộ pipeline (8 bước, tự dừng nếu cần xác nhận checkpoint SPAN thủ công)
bash pipeline/run_all.sh

# Hoặc chạy tay từng bước để dễ theo dõi (khuyến nghị cho lần chạy đầu):
bash pipeline/01_survey_and_prepare_data.sh
bash pipeline/02_setup_span_official.sh
bash pipeline/03_train_baseline_recognition.sh
bash pipeline/04_train_teacher_and_span_baseline.sh
bash pipeline/05_train_recognition_sr_baseline.sh
bash pipeline/06_improve_span.sh
bash pipeline/07_train_recognition_sr_improved.sh
bash pipeline/08_benchmark_and_aggregate.sh
```

**Điểm quan trọng**: Bước 1 (`01_survey_and_prepare_data.sh`) **tự động khảo
sát và chọn ngưỡng lọc dữ liệu** dựa trên số liệu thật (không cần bạn tự đoán
`hr_source_min`/`hr_size`) — thuật toán ưu tiên giữ 80% dữ liệu, tự động hạ
xuống mức an toàn hơn (90%) nếu phát hiện có identity bị mất trắng. Xem
`data/prepare_splits.py::auto_select_threshold()`.

## Toàn bộ file CSV kết quả (sau khi chạy xong pipeline)

| File | Sinh bởi | Nội dung |
|---|---|---|
| `splits/dataset_stats.csv` | Bước 1 | Percentile độ phân giải, ngưỡng đã chọn, số ảnh/identity theo split |
| `results/summary.csv` | Bước 8 | Identity/gender accuracy, params, latency — theo backbone × domain (ma trận chính) |
| `results/hr_sr_baseline_*.json`, `hr_sr_improved_*.json` | Bước 8 | Chẩn đoán domain gap (train=hr, test=sr_*) — chưa gộp CSV, đọc JSON trực tiếp |
| `results/sr_quality.csv` | Bước 8 | PSNR, SSIM, params, FLOPs, latency của EDSR/SPAN baseline/SPAN improved |
| `results/real_lr_holdout.csv` | Bước 8 | Accuracy trên LR thật (không phải downsample nhân tạo) — kiểm chứng tổng quát hóa |
| `results/training_summary.csv` | Bước 8 | Epoch dừng, early-stop, số lần OOM fallback, thời gian train, best val — mọi run |
| `results/ablation.csv` | `pipeline/run_ablation.sh` (chạy riêng) | 4 cấu hình loss (pixel/distill/identity) so sánh accuracy |

Tất cả các CSV này đủ để điền vào toàn bộ bảng trong `docs/01-04_*.md`.

## Cấu trúc thư mục

```
earsr_project/
  raw_data/EarVN1.0_raw/       # COPY dữ liệu EarVN1.0 gốc vào đây
  pipeline/                    # 8 script chạy lần lượt (xem pipeline/README.md)
    01_survey_and_prepare_data.sh
    02_setup_span_official.sh
    03_train_baseline_recognition.sh
    04_train_teacher_and_span_baseline.sh
    05_train_recognition_sr_baseline.sh
    06_improve_span.sh
    07_train_recognition_sr_improved.sh
    08_benchmark_and_aggregate.sh
    run_all.sh                 # chạy toàn bộ 8 bước liên tiếp
    run_ablation.sh            # chạy riêng: 4 cấu hình loss, xuất results/ablation.csv
    _update_config_from_report.py
  configs/config.yaml          # toàn bộ hyperparameter, đường dẫn, ngưỡng lọc
  docs/                        # tài liệu phương pháp — điền kết quả vào đây cho journal
    01_data_preparation.md
    02_baseline_experiments.md
    03_span_improvement.md
    04_results_template.md
  data/
    prepare_splits.py          # khảo sát + TỰ ĐỘNG chọn ngưỡng + chia train/val/test + dataset_stats.csv
    build_lr.py                # letterbox HR + downsample LR
    build_sr.py                # chạy SR model: LR -> SR (dùng cho cả sr_baseline và sr_improved)
    aggregate_results.py       # gộp JSON kết quả ma trận chính thành summary.csv
    aggregate_ablation_results.py  # gộp JSON kết quả ablation thành ablation.csv
    export_training_log_summary.py # trích xuất epoch/OOM/thời gian từ train.log -> CSV
  datasets/
    ear_dataset.py              # PyTorch Dataset đọc theo domain (hr/lr/sr_baseline/sr_improved)
    hrlr_pair_dataset.py         # Dataset đọc cặp (HR, LR) dùng chung cho train_sr*/eval_sr_quality
  models/
    sr_models.py                # SPAN tự viết lại + EDSR (teacher)
    span_official_wrapper.py    # import kiến trúc SPAN CHÍNH THỨC từ external/SPAN
    recognition_model.py        # factory 5 backbone: mobilenet_v2/v3_small, resnet18,
                                 # efficientnet_b0, ghostnet_100
  scripts/setup_span_official.sh  # clone repo SPAN thật + tải checkpoint
  utils/
    letterbox.py                # resize giữ tỷ lệ, không méo hình tai
    metrics.py                  # accuracy, PSNR, SSIM, params, FLOPs, latency
    logger.py                   # log ra file + màn hình
    early_stopping.py           # dừng sớm nếu val không cải thiện sau patience epoch
    device_manager.py           # tự fallback CPU khi GPU OOM, tự quay lại GPU sau
    seed.py
  train_sr.py                   # train SR thường (pixel loss) — dùng cho span_official và edsr
  train_sr_distill.py           # cải tiến SPAN: pixel + distillation(EDSR) + identity-aware loss
                                 # (hỗ trợ --lambda_* override + --run_suffix cho ablation)
  train_recognition.py          # train recognition trên 1 domain + 1 backbone
  eval_recognition.py           # test checkpoint, hỗ trợ cross-domain, ghi JSON kết quả
  eval_sr_quality.py            # đo PSNR/SSIM/FLOPs cho 1 model SR, append vào sr_quality.csv
  eval_real_lr_holdout.py       # eval trên real_lr_holdout.json (LR thật), xuất real_lr_holdout.csv
```

## Lưu ý quan trọng khi dùng

- `prepare_splits.py` chia theo **identity**, không theo ảnh — không có
  identity nào xuất hiện ở cả train và test.
- Nhãn giới tính suy ra từ số thứ tự thư mục người (001-098 = nam, 099-164 =
  nữ) theo mô tả gốc EarVN1.0 — **kiểm tra lại** với bản dataset bạn tải về.
- `train_recognition.py --domain sr_baseline/sr_improved` bắt buộc val set
  dùng chọn checkpoint cũng phải cùng domain đó, không dùng val của domain
  khác.
- `train_sr_distill.py` chỉ backprop qua SPAN (student) — EDSR (teacher) và
  recognition model (giám khảo) luôn đóng băng, chỉ dùng để forward.
- Backbone `ghostnet_100` cần cài `timm` (`pip install timm --break-system-packages`).
- Đo `latency_ms` trong `results/summary.csv` là đo trên máy chạy training —
  **cần đo lại trên thiết bị edge đích thật** (Jetson/mobile) trước khi đưa
  vào kết quả cuối cùng của luận án/bài báo.
