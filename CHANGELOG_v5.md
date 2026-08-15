# CHANGELOG v5 — Tổng hợp toàn bộ code đã sửa (2 đợt gộp lại)

File này thay thế mọi ghi chú rời rạc trước đó (`TONG_HOP_CUOI_CUNG.md`,
`CHANGELOG_transfer_learning.md`). Đây là bản DUY NHẤT cần đọc để biết code
nào đã đổi, vì sao, và đặt vào đâu.

## Danh sách file trong gói v5, đặt vào đâu

| File trong gói | Đặt vào (từ thư mục gốc project) | Trạng thái |
|---|---|---|
| `train_sr_distill.py` | `./train_sr_distill.py` | Đè file gốc |
| `train_recognition.py` | `./train_recognition.py` | Đè file gốc |
| `eval_recognition.py` | `./eval_recognition.py` | Đè file gốc |
| `utils/metrics.py` | `./utils/metrics.py` | Đè file gốc |
| `data/convert_generic_dataset_to_project_format.py` | `./data/convert_generic_dataset_to_project_format.py` | File mới |
| `scripts/setup_second_dataset_pipeline.sh` | `./scripts/setup_second_dataset_pipeline.sh` | File mới (đè bản lỗi nếu có) |
| `scripts/make_finetune_config.py` | `./scripts/make_finetune_config.py` | File mới |
| `scripts/run_transfer_learning.sh` | `./scripts/run_transfer_learning.sh` | File mới |

**KHÔNG có trong gói này** (đã gửi ở lượt trước, không đổi, giữ nguyên bản cũ
trên máy bạn — không cần thay): `RUN_ALL_span_large_ablation.sh`,
`RUN_ALL_NEW_DATASET.sh`. Hai file này không bị sửa trong 2 đợt vá vừa rồi.

## Đợt 1 — Sửa lỗi hardcode đường dẫn khi thêm dataset thứ 2 (AWE)

- `eval_recognition.py`, `utils/metrics.py`: thêm `compute_topk_accuracy`
  (Rank-5/CMC accuracy) — bổ sung chỉ số chuẩn biometrics.
- `train_sr_distill.py`: thêm `--student_arch` (đã có từ trước đợt 1, giữ
  nguyên) để chạy ablation kiến trúc (span_large).
- `data/convert_generic_dataset_to_project_format.py`: chuyển đổi cấu trúc
  thư mục dataset gốc (1 thư mục/người) sang định dạng project cần.
- `scripts/setup_second_dataset_pipeline.sh`: **sửa lỗi nghiêm trọng** — bản
  trước chỉ sed-thay biến `CONFIG=`/`RAW_DIR=`, bỏ sót các đường dẫn
  `runs/`, `splits/`, `results/`, `pipeline/` viết cứng rải rác trong thân
  các script `pipeline/*.sh`, khiến bước 1/8 của dataset thứ 2 (AWE) ghi đè
  nhầm vào `splits/` của EarVN1.0. Bản trong gói v5 đã sed-thay TOÀN BỘ các
  đường dẫn này (đối chiếu tay với nội dung thật của cả 8 script), có bước
  tự kiểm tra cuối cùng in cảnh báo nếu còn sót.

## Đợt 2 — Transfer learning (fine-tune từ EarVN1.0 sang AWE)

Lý do: AWE train-from-scratch (568 ảnh/100 người) cho accuracy thấp, phương
sai giữa seed rất lớn, và `span_large` train-from-scratch trên AWE bị PSNR
sụt xuống dưới cả `span_tiny` — dấu hiệu overfitting do thiếu dữ liệu.

- `train_sr_distill.py`: thêm `--init_ckpt` (khởi tạo student từ checkpoint
  có sẵn — an toàn `strict=True` vì SR model không có tầng phụ thuộc
  dataset).
- `train_recognition.py`: thêm `--init_ckpt_transfer` + hàm
  `load_transfer_checkpoint()` — chỉ nạp tensor CÙNG SHAPE (backbone +
  embedding + gender_head), tự bỏ qua `identity_head` (vì num_identities
  khác nhau giữa 2 dataset: 164 vs 100 — không thể "chuyển" trọng số phân
  loại của người này sang người khác). KHÔNG dùng lại `--init_ckpt` cũ cho
  việc này vì nó nạp `strict=True` toàn bộ, sẽ crash khi lệch shape.
- `scripts/make_finetune_config.py` (mới): tạo config fine-tune, tự tính
  learning rate (/10) và max_epochs (/3) từ giá trị THẬT trong config đích,
  không hardcode số.
- `scripts/run_transfer_learning.sh` (mới): điều phối 4 bước — (1) zero-shot
  SR (áp checkpoint nguồn thẳng lên ảnh đích, không train lại), (2) fine-tune
  SR, (3) fine-tune nhận diện 3 domain x 5 backbone x 3 seed (ghép cặp đúng
  seed với checkpoint nguồn để so sánh công bằng), (4) tổng hợp. Có kiểm tra
  đầy đủ tiền đề trước khi chạy — **đã rà soát lại 1 lần sau khi đối chiếu
  trực tiếp với `eval_sr_quality.py`/`config.yaml` thật**, bổ sung 2 kiểm
  tra còn thiếu (`recognition_hr_mobilenet_v2/best.pt` và
  `splits_awe/splits.json`) để không crash giữa chừng.

**Vì sao KHÔNG có "zero-shot" cho recognition** (chỉ SR mới có): 164 người
EarVN1.0 và 100 người AWE là hai tập người hoàn toàn khác nhau, không giao
nhau — không có ý nghĩa khi "áp thẳng" bộ phân loại của tập người này lên
tập người khác. Identity_head bắt buộc phải học lại (fine-tune).

## Cách chạy (từ thư mục gốc project, sau khi đặt xong 8 file ở bảng trên)

```bash
# 1. Kiểm tra đã patch đúng
grep -q "init_ckpt_transfer" train_recognition.py && echo "OK: train_recognition.py"
grep -q "TRANSFER LEARNING" train_sr_distill.py && echo "OK: train_sr_distill.py"
grep -q "compute_topk_accuracy" eval_recognition.py && echo "OK: eval_recognition.py"
grep -q "compute_topk_accuracy" utils/metrics.py && echo "OK: utils/metrics.py"
grep -q "BẢN SỬA LỖI" scripts/setup_second_dataset_pipeline.sh && echo "OK: setup_second_dataset_pipeline.sh"

# 2. Chạy transfer learning cho AWE (nguồn = EarVN1.0, đã có sẵn kết quả)
bash scripts/run_transfer_learning.sh awe earvn1
```

Không cần chạy lại `setup_second_dataset_pipeline.sh` hay pipeline 01-08 của
AWE — dữ liệu/checkpoint AWE from-scratch đã có sẵn từ trước, script transfer
learning tái sử dụng trực tiếp. `setup_second_dataset_pipeline.sh` trong gói
chỉ để dự phòng khi bạn thêm dataset thứ 3 (ví dụ IIT Delhi) sau này.

## File kết quả thu được sau khi chạy — cần cho tổng hợp bảng bài báo

| File | Nội dung | Đã có từ trước hay mới |
|---|---|---|
| `results_awe/sr_quality_transfer.csv` | 2 dòng: `span_tiny_zeroshot_from_earvn1`, `span_tiny_finetuned_from_earvn1` (PSNR/SSIM/params/latency) | **MỚI** |
| `results_awe/multi_seed_transfer/multi_seed_summary_transfer.csv` | Accuracy nhận diện fine-tuned, 3 domain x 5 backbone, trung bình±std 3 seed | **MỚI** |
| `results_awe/sr_quality.csv` | 4 dòng from-scratch (edsr/baseline/tiny/large) — dùng để so sánh với dòng zero-shot/finetuned ở trên | Đã gửi trước |
| `results_awe/multi_seed/multi_seed_summary.csv` | Accuracy from-scratch — dùng để so sánh với bản fine-tuned | Đã gửi trước |

Gửi lại đủ **2 file MỚI** (`sr_quality_transfer.csv` và
`multi_seed_summary_transfer.csv`) — tôi đã có sẵn 2 file from-scratch để
đối chiếu, sẽ dựng bảng so sánh đầy đủ 3 protocol (from-scratch / zero-shot /
fine-tuned) cho mục Results của bài báo.
