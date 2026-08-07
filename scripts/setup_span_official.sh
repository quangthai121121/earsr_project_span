#!/bin/bash
# Tải code + checkpoint CHÍNH THỨC của SPAN (hongyuanyu/SPAN, Apache 2.0 license).
# Chạy script này trên máy CÓ MẠNG (sandbox làm project này không có mạng nên
# không tự clone/tải được — đã thử và bị chặn bởi robots.txt của GitHub khi
# crawl, và không có kết nối để git clone/tải Google Drive).
#
# Sau khi chạy xong, thư mục external/SPAN sẽ chứa source code gốc, và
# checkpoints/span_pretrained.pth sẽ là checkpoint chính thức.
#
# Chạy:
#   bash scripts/setup_span_official.sh

set -e

EXTERNAL_DIR="external"
CKPT_DIR="checkpoints"
mkdir -p "$EXTERNAL_DIR" "$CKPT_DIR"

echo "== Bước 1: clone repo SPAN chính thức =="
if [ -d "$EXTERNAL_DIR/SPAN" ]; then
    echo "Đã tồn tại $EXTERNAL_DIR/SPAN, bỏ qua clone. Xóa thư mục này nếu muốn clone lại."
else
    git clone https://github.com/hongyuanyu/SPAN.git "$EXTERNAL_DIR/SPAN"
fi

echo "== Bước 2: cài gdown để tải checkpoint từ Google Drive =="
pip install gdown --break-system-packages -q

echo "== Bước 3: tải checkpoint pretrained chính thức =="
# ID file lấy trực tiếp từ link Google Drive trong README chính thức của repo:
# https://drive.google.com/file/d/1iYUA2TzKuxI0vzmA-UXr_nB43XgPOXUg/view
# Đây là file zip chứa NHIỀU checkpoint (các biến thể/scale khác nhau) —
# sau khi tải xong, giải nén và TỰ XEM bên trong để chọn đúng checkpoint scale x4.
gdown "https://drive.google.com/uc?id=1iYUA2TzKuxI0vzmA-UXr_nB43XgPOXUg" \
    -O "$CKPT_DIR/span_official_release.zip"

echo "== Bước 4: giải nén =="
unzip -o "$CKPT_DIR/span_official_release.zip" -d "$CKPT_DIR/span_official_release"

echo ""
echo "HOÀN TẤT. Các việc bạn cần làm tiếp (KHÔNG thể tự động hoàn toàn):"
echo "1. Mở $CKPT_DIR/span_official_release/ , tìm file .pth ứng với scale x4"
echo "   (thường có tên kiểu spanx4.pth hoặc tương tự — tên chính xác có thể"
echo "   đổi theo thời điểm tác giả cập nhật release, tự kiểm tra tên file)."
echo "2. Copy file đó thành: $CKPT_DIR/span_pretrained_x4.pth"
echo "3. Mở $EXTERNAL_DIR/SPAN/basicsr/archs/span_arch.py, xác nhận:"
echo "   - Tên class chính (thường là 'SPAN')"
echo "   - Danh sách tham số constructor (num_in_ch, num_feat, upscale, ...)"
echo "   rồi đối chiếu với models/span_official_wrapper.py — CHỈNH LẠI nếu"
echo "   tên tham số không khớp (mình chưa fetch được nội dung file này do"
echo "   GitHub chặn crawl, wrapper hiện viết theo cấu trúc BasicSR phổ biến"
echo "   nhưng CẦN BẠN XÁC NHẬN LẠI TRƯỚC KHI DÙNG CHO KẾT QUẢ CHÍNH THỨC)."
