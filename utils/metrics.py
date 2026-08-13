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
def compute_topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Top-k accuracy — chuẩn Rank-N/CMC trong biometrics: tỷ lệ mẫu có nhãn
    đúng nằm trong k giá trị logit cao nhất (không nhất thiết đứng #1)."""
    k = min(k, logits.size(1))  # phòng trường hợp num_identities < k (an toàn, không xảy ra ở đây)
    topk_preds = logits.topk(k, dim=1).indices
    correct = (topk_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
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


def freeze_reparam_modules(model: torch.nn.Module) -> int:
    """
    SPAN chính thức dùng kỹ thuật 'structural reparameterization' (module
    Conv3XC): nhiều nhánh conv lúc train, hợp nhất thành 1 conv duy nhất lúc
    suy luận. ĐÚNG cách dùng chuẩn: hợp nhất 1 LẦN DUY NHẤT trước khi triển
    khai/đo tốc độ. Nhưng forward() của code gốc lại gọi update_params() (tính
    lại phép hợp nhất) MỖI LẦN forward, kể cả ở eval mode — với 100 lần gọi
    lặp lại trong measure_latency(), điều này làm SPAN baseline bị đo CHẬM GIẢ
    TẠO (tính dư thừa 100 lần cho cùng 1 kết quả), không phản ánh đúng tốc độ
    suy luận thật. Hàm này gọi update_params() một lần duy nhất rồi vô hiệu
    hóa việc tính lại — cho kết quả đo latency công bằng, đúng cách deploy
    chuẩn của các kiến trúc reparameterization (RepVGG-style).

    Trả về số module đã được "đóng băng" (0 nếu model không dùng kỹ thuật này,
    ví dụ span_tiny — khi đó hàm này không làm gì cả, an toàn để gọi luôn).
    """
    count = 0
    for module in model.modules():
        if hasattr(module, "update_params") and hasattr(module, "eval_conv"):
            module.update_params()          # tính hợp nhất lần cuối, đúng trọng số hiện tại
            module.update_params = lambda: None   # từ giờ không tính lại nữa
            count += 1
    return count


def count_params_deploy_mode(model: torch.nn.Module) -> float:
    """
    Đo số tham số Ở CHẾ ĐỘ DEPLOY (đã hợp nhất reparameterization) — chỉ tính
    nhánh eval_conv của mỗi Conv3XC, BỎ QUA nhánh train (self.conv, self.sk)
    vì nhánh đó KHÔNG được dùng khi suy luận thực tế, chỉ tồn tại để hỗ trợ
    học trong lúc train.

    QUAN TRỌNG: đây RẤT CÓ THỂ là cách các bài báo SR gốc (bao gồm SPAN, và
    nhiều kiến trúc dùng structural reparameterization khác) báo cáo số tham
    số trong bảng kết quả — KHÁC với count_params() thông thường (tính cả 2
    nhánh, tổng luôn LỚN HƠN). Dùng hàm này để so sánh công bằng với số liệu
    tác giả công bố, tránh nhầm lẫn "model của mình nặng hơn" trong khi thực
    ra chỉ là khác quy ước đếm.

    Cách làm: đệ quy qua cây module. Khi gặp Conv3XC (nhận diện qua có cả
    thuộc tính eval_conv lẫn update_params), CHỈ đếm tham số của eval_conv,
    KHÔNG đệ quy tiếp vào các con khác của nó (self.conv, self.sk — nhánh
    train). Với module thường, đếm tham số riêng rồi đệ quy tiếp vào con.
    """
    def recurse(module):
        if hasattr(module, "eval_conv") and hasattr(module, "update_params"):
            # Đây là Conv3XC: chỉ đếm eval_conv, KHÔNG đệ quy vào self.conv/self.sk
            return sum(p.numel() for p in module.eval_conv.parameters())
        total = sum(p.numel() for p in module.parameters(recurse=False))
        for child in module.children():
            total += recurse(child)
        return total

    return recurse(model) / 1e6


@torch.no_grad()
def measure_latency(model: torch.nn.Module, input_size, device: str,
                     n_warmup: int = 10, n_iters: int = 100) -> float:
    """
    Đo latency trung bình (ms/ảnh) trên thiết bị chỉ định.
    input_size: ví dụ (1, 3, 32, 32) cho SR, hoặc (1, 3, 128, 128) cho recognition.
    Tự động "đóng băng" các module reparameterization (nếu có, ví dụ Conv3XC
    của SPAN chính thức) để đo công bằng — xem freeze_reparam_modules().
    Lưu ý: để đo latency thực tế trên thiết bị edge (Jetson/mobile), hãy chạy
    script này trực tiếp trên thiết bị đó, không chỉ trên máy train.
    """
    model.eval().to(device)
    n_frozen = freeze_reparam_modules(model)

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
