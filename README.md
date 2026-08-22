# EarSR-Recognition: SPAN nén nhẹ cho nhận diện danh tính/giới tính qua ảnh tai

Khung thí nghiệm cho luận án: **nén** kiến trúc SPAN (Swift Parameter-free
Attention Network, Wan et al. 2023, arXiv:2311.12770 — đoạt giải Nhất NTIRE
2024 Efficient SR Challenge) để giảm kích thước/tăng tốc độ xử lý, áp dụng
lên ảnh tai độ phân giải thấp, đo tác động lên accuracy nhận diện danh
tính/giới tính — kiểm chứng trên 2 dataset:

- **EarVN1.0** (28.412 ảnh, 164 người) — dataset chính, xem
  **[`RUNBOOK_EarVN1.0.md`](RUNBOOK_EarVN1.0.md)** để chạy đầy đủ từ đầu.
- **AWEx** (336 người, gộp AWE+CVLE+New images) — dataset thứ 2, kiểm chứng
  tổng quát hóa qua transfer learning, xem
  **[`RUNBOOK_AWEx.md`](RUNBOOK_AWEx.md)**.

Mỗi runbook là tài liệu **DUY NHẤT** cần đọc để tái lập kết quả cho đúng
dataset đó — mô tả đúng trạng thái code hiện tại (bao gồm phương pháp luận,
thứ tự chạy, danh sách lỗi đã phát hiện/vá, và những điểm cần lưu ý khi viết
bài báo).

## Cài đặt môi trường

```bash
git clone <repo_url> earsr_project
cd earsr_project
pip install -r requirements.txt --break-system-packages
```

Xem chi tiết môi trường (xung đột package `datasets`, v.v.) ở đầu
`RUNBOOK_EarVN1.0.md`.

## Cấu trúc thư mục & yêu cầu phần cứng

Xem mục "Cấu trúc thư mục" và "Yêu cầu phần cứng" trong `RUNBOOK_EarVN1.0.md`
(áp dụng chung cho cả 2 dataset — `pipeline_awex/` chỉ là bản sao đường dẫn
của `pipeline/`).
