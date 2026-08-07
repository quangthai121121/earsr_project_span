"""
Letterbox resize: giữ nguyên tỷ lệ khung hình, pad thêm nền cho đủ kích thước
vuông mục tiêu — tránh méo/kéo dãn hình dạng vành tai (đã thảo luận: EarVN1.0
có ear không nằm giữa khung và tỷ lệ khung hình rất đa dạng, resize thẳng NxN
sẽ làm méo đặc trưng hình dạng quan trọng cho nhận diện).
"""
from PIL import Image


def letterbox_resize(img: Image.Image, target_size: int, fill_color=(0, 0, 0)) -> Image.Image:
    """Resize ảnh giữ tỷ lệ khung hình sao cho cạnh dài nhất = target_size,
    sau đó pad thêm nền để ra đúng hình vuông target_size x target_size."""
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)

    resized = img.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (target_size, target_size), fill_color)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas
