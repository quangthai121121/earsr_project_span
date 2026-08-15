# 01 — Chuẩn bị dữ liệu

## Nguồn dữ liệu

EarVN1.0: 28.412 ảnh vành tai màu RGB, 164 người (98 nam, 66 nữ), thu thập năm
2018, crop tự động từ ảnh chân dung/toàn cảnh chụp ngoài trời (điều kiện không
kiểm soát). Mỗi người có 107–300 ảnh.

**Đặc điểm quan trọng ảnh hưởng đến toàn bộ thiết kế thí nghiệm**:
- Độ phân giải ảnh **không đồng nhất**, dao động rất rộng (một số ảnh dưới
  25×25px do vùng tai trong ảnh gốc quá nhỏ).
- Vành tai **không luôn nằm giữa khung ảnh** (do crop tự động).
- Không có cặp ảnh HR-LR thật đi kèm — dataset chỉ gồm một tập ảnh duy nhất ở
  nhiều độ phân giải khác nhau.

## Quyết định phương pháp luận và lý do

### 1. Coi ảnh gốc là "HR tương đối", không phải HR tuyệt đối

Vì không có cặp HR-LR thật, chúng tôi áp dụng quy ước phổ biến trong nghiên
cứu SR cho biometric khi thiếu ground truth: chọn tập con ảnh có độ phân giải
đủ lớn làm nguồn "HR", tự tạo LR bằng downsample bicubic. Đây là "HR tương
đối" — cần nêu rõ giả định này trong phần Limitation của bài báo.

### 2. Phân nhóm theo ngưỡng kích thước thay vì dùng toàn bộ dataset làm HR

Ảnh quá nhỏ (< `too_small_max`, mặc định 25px cạnh ngắn) không đủ tin cậy làm
ground truth — phóng to chúng lên kích thước chuẩn sẽ tạo ra chi tiết do nội
suy bịa ra, không phải chi tiết thật, khiến việc dùng chúng làm nhãn HR để
train SR học sai lệch.

Quy trình (`data/prepare_splits.py`):
1. Khảo sát phân phối kích thước toàn bộ dataset (`--survey_only`), in ra
   percentile p10/p25/p50/p75/p90 của cạnh ngắn.
2. Dựa vào số liệu thật, chọn `hr_source_min` (ảnh ≥ ngưỡng này → dùng làm
   nguồn HR cho toàn bộ pipeline chính) và `too_small_max` (ảnh < ngưỡng này
   → gác riêng, lưu vào `real_lr_holdout.json`).
3. Nhóm ảnh "quá nhỏ" **không bị xóa** — đây là LR *thật ngoài đời* (không
   phải downsample nhân tạo), dùng làm tập kiểm chứng bổ sung để đánh giá khả
   năng tổng quát hóa của SR sang suy giảm thật (khác với suy giảm bicubic
   sạch sẽ do ta tự tạo).

### 3. Letterbox thay vì resize thẳng (tránh méo hình)

Vì vành tai không nằm giữa khung và tỷ lệ khung hình ảnh rất đa dạng, resize
thẳng về NxN sẽ kéo dãn/méo hình dạng vành tai — một đặc trưng quan trọng cho
nhận diện danh tính. `utils/letterbox.py` thực hiện: giữ nguyên tỷ lệ khung
hình, resize sao cho cạnh dài nhất = kích thước mục tiêu, pad thêm nền cho đủ
hình vuông. Áp dụng nhất quán cho **mọi** ảnh trước khi downsample tạo LR.

### 4. Chia train/val/test theo identity, không theo ảnh

Đảm bảo không có identity nào xuất hiện ở cả train và test (tránh data
leakage khi đánh giá recognition). Tỷ lệ mặc định 70/15/15 theo số người.

### 5. Nhãn giới tính

Suy ra tự động từ số thứ tự thư mục người (001–098 = nam, 099–164 = nữ).
**[ĐÃ XÁC MINH — đợt 7]**: khớp đúng mô tả chính thức trong bài báo gốc công
bố dataset (Hoang, V.N., *"EarVN1.0: A new large-scale ear images dataset in
the wild"*, Data in Brief, 2019 — "98 males and 66 females... the first 98
folders (from 01 to 98) belong to male class and the rest (from 99 to 164)
are female") — không phải suy đoán chưa kiểm chứng như ghi chú ở bản trước.
**Phát hiện khi kiểm tra dữ liệu thật**: tên thư mục không phải chỉ số
(`001`) như giả định ban đầu, mà có định dạng `NNN.Tên` (ví dụ `001.ALI_HD`)
— `data/prepare_splits.py::extract_person_id()` đã xử lý việc tách số thứ tự
từ định dạng này.

## Tham số đã dùng (số liệu thật, khảo sát trên toàn bộ 28.412 ảnh EarVN1.0)

| Tham số | Giá trị |
|---|---|
| `hr_source_min` | **41px** (giữ 80.9% dữ liệu, đủ 164/164 identity, không identity nào dưới 10 ảnh) |
| `too_small_max` | **20px** (173 ảnh, dùng làm `real_lr_holdout`) |
| `hr_size` (kích thước HR sau letterbox) | **80×80** (chọn để tỷ lệ phóng đại tối đa ~1.95x, tránh tạo chi tiết giả) |
| `scale` (hệ số downsample HR→LR) | 4 (LR = 20×20) |
| Tỷ lệ train/val/test | 70/15/15 theo identity |
| Seed | 42 |
| **Số ảnh nhóm HR-source** | **22.972 ảnh** |
| **Số ảnh nhóm quá nhỏ (real LR holdout)** | **173 ảnh** |
| **Số ảnh nhóm giữa (không dùng)** | 5.267 ảnh |
| **Số identity train/val/test** | **114 / 24 / 26** (tổng 164) |
| **Số ảnh train/val/test** | **15.730 / 3.237 / 4.005** |
| Percentile cạnh ngắn toàn dataset | min=13, p5=27, p10=32, p20=41, p25=46, p50=77, p75=122, p90=161, max=472 |

**Lưu ý về trade-off đã cân nhắc**: thử cả phương án `hr_source_min=77` (=median,
giữ 50.6% dữ liệu, `hr_size=128`) — vẫn đủ 164/164 identity nhưng có 18 identity
chỉ còn dưới 10 ảnh. Chọn `hr_source_min=41` vì giữ được nhiều dữ liệu hơn đáng
kể và phân bố đồng đều hơn giữa các identity — quan trọng khi dataset chỉ có
164 người, không dư dả để hy sinh một nửa dữ liệu.

## Script liên quan

- `data/prepare_splits.py` — khảo sát, phân nhóm, chia split
- `data/build_lr.py` — tạo HR (letterbox) + LR (downsample bicubic)
- `utils/letterbox.py` — hàm letterbox dùng chung
