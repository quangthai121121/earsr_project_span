"""
Chỉ số cho cả 2 trục đánh giá: accuracy (identity/gender) và hiệu năng tính toán
(params, FLOPs, latency) — dùng để vẽ Pareto frontier ở Giai đoạn 4. Ngoài ra
có PSNR/SSIM để đánh giá chất lượng ảnh SR (độc lập với accuracy downstream).
"""
import math
import time

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


@torch.no_grad()
def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """img1, img2: tensor cùng shape, giá trị trong [0, max_val]."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse <= 1e-10:
        return 100.0  # coi như giống hệt, tránh log(0)
    return 10 * math.log10((max_val ** 2) / mse)


@torch.no_grad()
def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11,
                  sigma: float = 1.5, max_val: float = 1.0) -> float:
    """
    SSIM tự cài đặt bằng conv2d gaussian window (không phụ thuộc skimage).
    img1, img2: (B,C,H,W), giá trị trong [0, max_val].
    """
    device = img1.device
    channels = img1.shape[1]

    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    return ssim_map.mean().item()


def count_params(model: torch.nn.Module) -> float:
    """Trả về số triệu tham số (M)."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_flops(model: torch.nn.Module, input_size, device: str = "cpu") -> float:
    """
    Trả về số GFLOPs cho 1 lần forward với kích thước input_size (ví dụ (1,3,32,32)).
    Dùng thư viện thop — cần `pip install thop`.
    """
    try:
        from thop import profile
    except ImportError:
        raise ImportError(
            "Cần cài thop để tính FLOPs: pip install thop --break-system-packages"
        )
    model = model.to(device).eval()
    dummy = torch.randn(*input_size).to(device)
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    return flops / 1e9  # GFLOPs


@torch.no_grad()
def measure_latency(model: torch.nn.Module, input_size, device: str,
                     n_warmup: int = 10, n_iters: int = 100) -> float:
    """
    Đo latency trung bình (ms/ảnh) trên thiết bị chỉ định.
    input_size: ví dụ (1, 3, 32, 32) cho SR, hoặc (1, 3, 128, 128) cho recognition.
    Lưu ý: để đo latency thực tế trên thiết bị edge (Jetson/mobile), hãy chạy
    script này trực tiếp trên thiết bị đó, không chỉ trên máy train.
    """
    model.eval().to(device)
    dummy = torch.randn(*input_size).to(device)

    for _ in range(n_warmup):
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_iters):
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    return (elapsed / n_iters) * 1000  # ms/ảnh
