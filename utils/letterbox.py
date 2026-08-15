"""
Letterbox resize: giữ nguyên tỷ lệ khung hình, pad thêm nền cho đủ kích thước
vuông mục tiêu — tránh méo/kéo dãn hình dạng vành tai (đã thảo luận: EarVN1.0
có ear không nằm giữa khung và tỷ lệ khung hình rất đa dạng, resize thẳng NxN
sẽ làm méo đặc trưng hình dạng quan trọng cho nhận diện).

[MỚI — đợt 7] Tách hàm compute_letterbox_geometry() ra riêng — MỘT nguồn
công thức DUY NHẤT dùng chung cho cả việc tạo ảnh (letterbox_resize, dùng ở
data/build_lr.py) LẪN việc tính vùng ROI thật (không phải vùng đệm đen) sau
này khi đo PSNR/SSIM (utils/metrics.py::compute_psnr_roi/compute_ssim_roi,
datasets/hrlr_pair_dataset.py) — tránh 2 nơi tính công thức letterbox khác
nhau rồi lệch nhau (đúng kiểu lỗi "quên đồng bộ 1 chỗ" project này đã từng
gặp nhiều lần, xem CHANGELOG). Vùng ROI được suy ngược LẠI từ (width, height)
gốc đã lưu sẵn trong splits.json — KHÔNG cần sinh lại ảnh HR/LR nào, chỉ cần
sửa cách đo ở bước eval.
"""
from PIL import Image


def compute_letterbox_geometry(orig_w: int, orig_h: int, target_size: int):
    """Trả về (new_w, new_h, paste_x, paste_y): kích thước ảnh sau khi resize
    giữ tỷ lệ (cạnh dài nhất = target_size) và vị trí dán vào canvas vuông
    target_size x target_size — ĐÚNG công thức letterbox_resize() bên dưới,
    dùng để suy ra vùng ROI thật (không phải viền đệm đen) mà KHÔNG cần mở
    lại ảnh gốc."""
    scale = target_size / max(orig_w, orig_h)
    new_w, new_h = round(orig_w * scale), round(orig_h * scale)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    return new_w, new_h, paste_x, paste_y


def letterbox_resize(img: Image.Image, target_size: int, fill_color=(0, 0, 0)) -> Image.Image:
    """Resize ảnh giữ tỷ lệ khung hình sao cho cạnh dài nhất = target_size,
    sau đó pad thêm nền để ra đúng hình vuông target_size x target_size."""
    w, h = img.size
    new_w, new_h, paste_x, paste_y = compute_letterbox_geometry(w, h, target_size)

    resized = img.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (target_size, target_size), fill_color)
    canvas.paste(resized, (paste_x, paste_y))
    return canvas
