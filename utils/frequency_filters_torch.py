"""
[MỚI — thử nghiệm frequency-weighted training loss, hướng novelty mới]
Bản khả vi (differentiable), theo batch, của utils/frequency_filters.py --
CHỈ hỗ trợ lowpass (đủ dùng cho loss training; numpy/PIL version vẫn giữ
nguyên, dùng cho tiền xử lý dataset ngoại tuyến + eval, không đổi).

Động lực: eval_frequency_ablation.py (Section~sec:freq-ablation trong bài)
đã chứng minh bằng THỰC NGHIỆM rằng tín hiệu nhận dạng trong ảnh tai độ phân
giải thấp tập trung ở dải tần số THẤP (giữ 10% tần số thấp nhất vẫn giữ được
22--34% accuracy; bỏ 10% đó thì accuracy sập gần về ngẫu nhiên). Module này
biến phát hiện đó thành một THÀNH PHẦN LOSS khi train SR: phạt thêm sai khác
ở đúng dải tần số thấp đó, cộng thêm (không thay thế) loss_pixel/loss_distill
hiện có trong train_sr_distill.py -- theo đúng triết lý "additive, opt-in,
lambda=0 mặc định" mà mọi cơ chế MỚI khác trong file đó (lambda_feat,
lambda_saliency, lambda_position) đều tuân theo.

QUY ƯỚC cutoff_frac giống HỆT utils/frequency_filters.py (xem docstring file
đó để hiểu đầy đủ lý do): tỉ lệ so với r_max = bán kính tại GÓC phổ sau khi
fftshift (sqrt((H/2)^2+(W/2)^2)), KHÔNG phải bán kính Nyquist trục ngắn --
cutoff_frac=1.0 phải là no-op tuyệt đối (test dưới xác nhận), cutoff_frac=0.0
chỉ còn thành phần DC.

Vì sao KHÔNG dùng lại thẳng utils/frequency_filters.py: hàm đó nhận PIL Image
+ numpy, không có gradient, không chạy theo batch trên GPU -- không dùng
được làm loss training. Module này viết lại đúng công thức đó bằng
torch.fft (có gradient, chạy batch, chạy GPU), giữ NGUYÊN convention để
diễn giải nhất quán với thí nghiệm frequency-ablation đã có trong bài
(cùng ý nghĩa "cutoff_frac=0.1" ở cả hai nơi).

FFT bị ép chạy fp32 tường minh (autocast tắt), giống đúng cách
compute_total_loss() trong train_sr_distill.py đã làm cho identity loss
(cosine_similarity) -- lý do tương tự: các phép biến đổi này kém ổn định số
học dưới fp16/autocast, quan sát thấy gây NaN ở identity loss trước đây.
"""
import torch


def _lowpass_mask(h: int, w: int, cutoff_frac: float, device) -> torch.Tensor:
    """Mask lowpass hình tròn cứng, shape (h, w), dtype float32, giá trị 0/1.
    Giống hệt _radius_grid()/_max_radius() của utils/frequency_filters.py,
    viết lại bằng torch để chạy trên GPU."""
    if not (0.0 <= cutoff_frac <= 1.0):
        raise ValueError(f"cutoff_frac phải trong [0, 1], nhận {cutoff_frac}")
    cy, cx = h / 2.0, w / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    radius = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = float((cy ** 2 + cx ** 2) ** 0.5)
    cutoff_r = cutoff_frac * r_max
    return (radius <= cutoff_r).to(torch.float32)


def lowpass_filter(x: torch.Tensor, cutoff_frac: float) -> torch.Tensor:
    """Lọc lowpass khả vi trên batch ảnh (B, C, H, W). Trả về tensor cùng
    shape/device/dtype đầu vào (nội bộ ép fp32 để tính FFT, cast lại dtype
    gốc ở cuối). cutoff_frac=1.0 là no-op tuyệt đối; cutoff_frac=0.0 chỉ còn
    thành phần DC (giá trị trung bình mỗi kênh, lặp lại khắp ảnh).

    Có gradient (torch.fft khả vi) -- dùng được trực tiếp trong loss training.
    """
    if x.dim() != 4:
        raise ValueError(f"x phải có shape (B, C, H, W), nhận shape {tuple(x.shape)}")
    h, w = x.shape[-2], x.shape[-1]
    orig_dtype = x.dtype

    with torch.autocast(device_type=x.device.type, enabled=False):
        x_f32 = x.float()
        mask = _lowpass_mask(h, w, cutoff_frac, x.device)  # (h, w), broadcasts over (B, C, h, w)
        spectrum = torch.fft.fftshift(torch.fft.fft2(x_f32), dim=(-2, -1))
        spectrum_filtered = spectrum * mask
        recon = torch.fft.ifft2(torch.fft.ifftshift(spectrum_filtered, dim=(-2, -1)))
        out = recon.real

    return out.to(orig_dtype)
