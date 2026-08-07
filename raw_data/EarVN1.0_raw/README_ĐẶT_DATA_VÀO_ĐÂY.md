# Copy dữ liệu EarVN1.0 (raw) vào đây

Copy toàn bộ các thư mục người (định dạng `NNN.Tên`, ví dụ `001.ALI_HD`,
`002.LeDuong_BL`, ...) vào **trực tiếp** trong thư mục này, KHÔNG qua thư mục
con trung gian nào khác.

Cấu trúc đúng (đã xác nhận khớp với dataset thật):

```
raw_data/EarVN1.0_raw/
├── 001.ALI_HD/
│   ├── 001 (1).jpg
│   ├── 001 (2).jpg
│   └── ...
├── 002.LeDuong_BL/
│   └── ...
├── ...
└── 164.Yen_Nhi_H/
    └── ...
```

Nếu bạn tải dataset dạng file `.zip` có cấu trúc `images/Images/001.ALI_HD/...`
(một lớp thư mục `Images` trung gian như bản gốc tác giả phát hành), hãy giải
nén rồi copy **nội dung bên trong thư mục `Images`** vào đây — không copy
nguyên cả thư mục `Images` lồng thêm 1 cấp.

**Lưu ý**: project này đã đi kèm sẵn `splits/` (HR/LR đã build sẵn từ một lần
chạy trước trên đúng bộ EarVN1.0 thật) — nếu bạn dùng đúng bộ dữ liệu đó,
KHÔNG cần copy lại vào đây hay chạy lại `pipeline/01_survey_and_prepare_data.sh`,
có thể bỏ qua thẳng sang `pipeline/02_setup_span_official.sh`. Chỉ cần copy
data vào đây nếu bạn muốn chạy lại từ đầu (ví dụ dùng bản dataset khác, hoặc
muốn tự chọn lại ngưỡng thủ công).

Sau khi copy xong (nếu chạy lại), chạy `bash pipeline/01_survey_and_prepare_data.sh`
(xem `pipeline/README.md` để biết thứ tự chạy đầy đủ).
