"""
SPAN — Swift Parameter-free Attention Network (Wan et al., CVPR 2024 NTIRE Workshop).
Vô địch NTIRE 2024 Efficient SR Challenge, sau đó trở thành baseline chính thức
của NTIRE 2026 Efficient SR Challenge — hiện là chuẩn tham chiếu mới nhất trong
dòng SR nhẹ/real-time. Đây là model NỀN mà luận án sẽ cải tiến (Giai đoạn 3).

Ý tưởng cốt lõi: "parameter-free attention" — dùng hàm kích hoạt đối xứng
(symmetric activation, ví dụ sigmoid dịch tâm) áp trực tiếp lên feature map để
tạo attention map, KHÔNG cần thêm tham số học được (không có conv/linear layer
riêng cho attention như SE-block hay CBAM) — vừa tăng chất lượng vừa giữ nhẹ.

Cài đặt này là bản rút gọn, trung thành với ý tưởng gốc (SPAB - Swift
Parameter-free Attention Block), đủ để làm nền tảng thực nghiệm cho luận án.
"""
import torch
import torch.nn as nn


class SymmetricAttention(nn.Module):
    """Attention không tham số: dùng hàm kích hoạt đối xứng quanh gốc tọa độ
    (ví dụ sigmoid dịch tâm) áp trực tiếp lên feature map làm attention map."""

    def forward(self, x):
        # sigmoid dịch tâm: 2*sigmoid(x) - 1, đối xứng qua gốc, không có tham số học
        attn = 2 * torch.sigmoid(x) - 1
        return x * attn


class SPAB(nn.Module):
    """Swift Parameter-free Attention Block — khối cơ bản của SPAN.
    3 lớp conv 3x3 + parameter-free attention + residual connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.attn = SymmetricAttention()

    def forward(self, x):
        identity = x
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out = self.conv3(out)
        out = self.attn(out)
        return identity + out


class SPAN(nn.Module):
    """
    SPAN rút gọn: feature extraction -> N khối SPAB xếp chồng -> pixel shuffle upsample.
    n_blocks/channels mặc định theo tinh thần "cực nhẹ" của bản NTIRE 2026 baseline
    (~0.15M tham số cho ảnh 3 kênh, scale=4); có thể tăng n_blocks nếu cần thêm sức
    biểu diễn, đổi lại nặng hơn.
    """

    def __init__(self, scale: int = 4, channels_in: int = 3, feat: int = 28, n_blocks: int = 4):
        super().__init__()
        self.head = nn.Conv2d(channels_in, feat, kernel_size=3, padding=1)
        self.body = nn.ModuleList([SPAB(feat) for _ in range(n_blocks)])
        self.body_tail = nn.Conv2d(feat, feat, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(feat, channels_in * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        feat = self.head(x)
        body_in = feat
        for block in self.body:
            feat = block(feat)
        feat = self.body_tail(feat) + body_in  # long-range residual, ổn định training
        out = self.upsample(feat)
        return torch.clamp(out, 0.0, 1.0)


class ResBlock(nn.Module):
    """Residual block chuẩn của EDSR (Lim et al., 2017), không dùng BatchNorm."""

    def __init__(self, channels: int, res_scale: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.res_scale = res_scale

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))
        return x + out * self.res_scale


class EDSR(nn.Module):
    """
    EDSR rút gọn — dùng làm TEACHER nặng (Giai đoạn 1: xác lập trần lý thuyết,
    và Giai đoạn 3: distillation). KHÔNG phải model dùng để triển khai.
    n_resblocks/feat mặc định nhỏ hơn bản gốc trong paper (vốn 32 block/256 kênh)
    để việc train giáo viên trên EarVN1.0 (dataset không quá lớn) khả thi về
    thời gian; tăng lên nếu tài nguyên cho phép và muốn trần lý thuyết cao hơn.
    """

    def __init__(self, scale: int = 4, channels_in: int = 3, feat: int = 128,
                 n_resblocks: int = 16):
        super().__init__()
        self.head = nn.Conv2d(channels_in, feat, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResBlock(feat) for _ in range(n_resblocks)])
        self.body_tail = nn.Conv2d(feat, feat, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(feat, channels_in * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        feat = self.head(x)
        body_out = self.body_tail(self.body(feat)) + feat
        out = self.upsample(body_out)
        return torch.clamp(out, 0.0, 1.0)


def build_sr_model(arch: str, scale: int, pretrained_path: str = None) -> nn.Module:
    """
    arch:
      - "span"          : bản tự viết lại (reimplementation), feat=28, n_blocks=4 —
                           dùng khi KHÔNG cần khớp checkpoint chính thức.
      - "span_tiny"      : bản NÉN, feat=96, n_blocks=3 -> ~0.875M tham số
                           (~39% kích thước SPAN baseline 2.237M, tính chính
                           xác theo công thức kiến trúc, xem cách tính trong
                           docs/03_span_improvement.md). Dùng làm student để
                           distill từ SPAN baseline (đã chứng minh chất lượng
                           tốt), mục tiêu: giảm kích thước/tăng tốc so với
                           SPAN baseline, chấp nhận đánh đổi accuracy nhẹ,
                           miễn còn hơn no-SR.
      - "span_official"  : bản CHÍNH THỨC, import trực tiếp từ repo đã clone
                           qua scripts/setup_span_official.sh — dùng khi cần
                           load checkpoint pretrained thật để fine-tune.
                           Xem models/span_official_wrapper.py.
      - "span_large", "edsr": như cũ.
    """
    arch = arch.lower()
    if arch == "span":
        return SPAN(scale=scale)
    if arch == "span_tiny":
        return SPAN(scale=scale, feat=96, n_blocks=3)
    if arch == "span_large":
        # biến thể lớn hơn — chỉ dùng để so sánh/khảo sát, KHÔNG phải mục tiêu triển khai
        return SPAN(scale=scale, feat=48, n_blocks=6)
    if arch == "span_official":
        from models.span_official_wrapper import build_official_span
        return build_official_span(scale=scale, pretrained_path=pretrained_path)
    if arch == "edsr":
        return EDSR(scale=scale)  # teacher nặng, Giai đoạn 1 + Giai đoạn 3
    raise ValueError(f"Kiến trúc SR không hỗ trợ: {arch}")
