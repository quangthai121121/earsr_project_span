"""
[MỚI — Mục 5.9 bài báo, giải pháp cho "Boundary Condition" real_lr_holdout]
Pipeline suy giảm THỰC TẾ (blur + noise + JPEG), dùng để tăng cường train SR
model chống chịu tốt hơn với suy giảm THẬT — giải quyết đúng nguyên nhân gốc
đã xác định ở Section res-main: model chỉ học ĐÚNG 1 kiểu suy giảm bicubic
sạch sẽ, gặp ảnh suy giảm thật (blur quang học, nhiễu cảm biến, nén ảnh) sẽ
hallucinate chi tiết tần số cao SAI, gây hại downstream recognition.

[LỰA CHỌN THIẾT KẾ] Dùng mô hình "classical degradation" (1 vòng blur ->
downsample -> noise -> JPEG, theo tinh thần BSRGAN~\\cite{zhang2021designing})
thay vì mô hình "2 vòng lặp lại" đầy đủ của Real-ESRGAN — lý do: (a) đơn giản
hơn đáng kể để cài đặt và kiểm chứng đúng, giảm rủi ro bug; (b) đủ để kiểm
định trực tiếp giả thuyết "đa dạng hoá degradation lúc train có giúp real_lr_
holdout không", là câu hỏi cụ thể bài báo cần trả lời, không phải mục tiêu
tổng quát "khớp Real-ESRGAN publication"; (c) chi phí GPU/thời gian phù hợp
với quy mô ablation bổ sung, không phải 1 đóng góp SR tổng quát mới.

Áp dụng NGẪU NHIÊN mỗi lần gọi (mỗi sample, mỗi epoch khác nhau) -- KHÔNG bake
cứng 1 lần vào 1 file LR tĩnh trên đĩa, vì mục tiêu là cho model thấy ĐA DẠNG
suy giảm qua nhiều epoch, đúng tinh thần data augmentation thật, không phải
chỉ đổi 1 phiên bản suy giảm cố định khác.
"""
import io
import random

import numpy as np
from PIL import Image, ImageFilter


def random_degrade(hr_img: Image.Image, scale: int,
                    blur_sigma_range=(0.2, 2.0),
                    noise_sigma_range=(0.0, 10.0),
                    jpeg_quality_range=(30, 95)) -> Image.Image:
    """HR (PIL RGB, đã letterbox vuông) -> LR suy giảm THẬT ngẫu nhiên (PIL
    RGB), kích thước = HR size / scale. Áp dụng theo đúng thứ tự vật lý của
    quá trình chụp+xử lý ảnh thật: blur quang học -> mất độ phân giải (cảm
    biến/zoom) -> nhiễu cảm biến -> nén lưu trữ/truyền tải.

    blur_sigma_range/noise_sigma_range/jpeg_quality_range: khoảng lấy mẫu
    ngẫu nhiên đều (uniform) mỗi lần gọi -- KHÔNG cố định 1 giá trị, để mỗi
    sample mỗi epoch thấy 1 tổ hợp suy giảm khác nhau (đúng tinh thần
    augmentation, xem docstring module)."""
    w, h = hr_img.size
    lr_w, lr_h = max(1, w // scale), max(1, h // scale)

    # 1. Blur ngẫu nhiên (mô phỏng out-of-focus/rung tay nhẹ) -- áp dụng
    # TRƯỚC downsample, đúng thứ tự vật lý thật (ống kính làm mờ trước khi
    # cảm biến lấy mẫu ở độ phân giải thấp hơn).
    sigma = random.uniform(*blur_sigma_range)
    blurred = hr_img.filter(ImageFilter.GaussianBlur(radius=sigma))

    # 2. Downsample bicubic (giữ NGUYÊN cơ chế suy giảm chính đã dùng xuyên
    # suốt project -- không thay thế, chỉ bổ sung thêm blur/noise/JPEG xung
    # quanh nó).
    lr = blurred.resize((lr_w, lr_h), Image.BICUBIC)

    # 3. Nhiễu ngẫu nhiên (mô phỏng nhiễu cảm biến/thiếu sáng) -- Gaussian
    # additive, đơn giản và phổ biến nhất trong literature degradation-aware
    # SR (BSRGAN/Real-ESRGAN đều dùng biến thể của nhiễu Gaussian làm thành
    # phần chính).
    arr = np.asarray(lr, dtype=np.float32)
    noise_sigma = random.uniform(*noise_sigma_range)
    if noise_sigma > 0:
        noise = np.random.normal(0.0, noise_sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
    lr = Image.fromarray(arr.astype(np.uint8))

    # 4. Nén JPEG ngẫu nhiên (mô phỏng nén lưu trữ/truyền tải thật -- nguồn
    # suy giảm phổ biến khác hẳn bicubic sạch, đặc biệt liên quan trực tiếp
    # tới ảnh camera an ninh/giám sát -- đúng use-case chính bài báo nhắm
    # tới, xem Introduction).
    quality = random.randint(*jpeg_quality_range)
    buf = io.BytesIO()
    lr.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    lr = Image.open(buf).convert("RGB")

    return lr
