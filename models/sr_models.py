"""
SPAN — Swift Parameter-free Attention Network (Wan et al., 2023,
arXiv:2311.12770) — đoạt giải Nhất NTIRE 2024 Efficient SR Challenge.

Ý tưởng cốt lõi: "parameter-free attention" — dùng hàm kích hoạt đối xứng
(symmetric activation: sigmoid(x) - 0.5) áp trực tiếp lên feature map để tạo
attention map, KHÔNG cần thêm tham số học được (không có conv/linear layer
riêng cho attention như SE-block hay CBAM) — vừa tăng chất lượng vừa giữ nhẹ.

Công thức SPAB gốc (đã đối chiếu trực tiếp với bài báo + source code chính
thức github.com/hongyuanyu/SPAN, xác nhận khớp):
    H_i = conv_layers(O_{i-1})           # đặc trưng thô qua các lớp conv
    U_i = O_{i-1} + H_i                  # residual CỘNG TRƯỚC
    V_i = sigmoid(H_i) - 0.5             # attention map tính từ H_i thô
    O_i = U_i * V_i                      # NHÂN SAU CÙNG (nhân U_i đã cộng residual)

Cài đặt này là bản tự viết lại (reimplementation), dùng conv 3x3 thường
(KHÔNG dùng kỹ thuật reparameterization Conv3XC của bản chính thức) — đúng
với thiết kế span_tiny của luận án (xem build_sr_model() bên dưới).
"""
import torch
import torch.nn as nn


class SymmetricAttention(nn.Module):
    """Attention không tham số: sigmoid(x) - 0.5, đối xứng qua gốc tọa độ,
    không có tham số học. Chỉ trả về attention map — việc nhân với U_i (đã
    cộng residual) được thực hiện ở SPAB.forward(), không phải ở đây, để
    đúng đúng thứ tự phép tính của công thức gốc (residual cộng trước khi
    nhân attention, không phải nhân H_i thô rồi mới cộng residual)."""

    def forward(self, h):
        return torch.sigmoid(h) - 0.5


class SPAB(nn.Module):
    """Swift Parameter-free Attention Block — khối cơ bản của SPAN.
    3 lớp conv 3x3 + parameter-free attention + residual connection, ĐÚNG
    thứ tự công thức gốc: cộng residual TRƯỚC, nhân attention SAU."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.attn = SymmetricAttention()

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.conv3(h)              # H_i — đặc trưng thô, CHƯA cộng residual
        u = x + h                      # U_i = O_{i-1} + H_i — cộng residual TRƯỚC
        v = self.attn(h)               # V_i = attn(H_i) — tính từ H_i thô, riêng biệt
        return u * v                   # O_i = U_i * V_i — NHÂN SAU CÙNG


class SPAN(nn.Module):
    """
    SPAN rút gọn: feature extraction -> N khối SPAB xếp chồng -> pixel shuffle upsample.
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
    EDSR rút gọn — dùng làm TEACHER nặng (tham chiếu chất lượng, không dùng
    làm teacher cho span_tiny — xem sr_improve.teacher_arch trong config.yaml,
    dùng span_official). KHÔNG phải model dùng để triển khai.
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


def build_sr_model(arch: str, scale: int, pretrained_path: str = None,
                    feature_channels: int = None) -> nn.Module:
    """
    arch:
      - "span"          : bản tự viết lại (reimplementation), feat=28, n_blocks=4 —
                           dùng khi KHÔNG cần khớp checkpoint chính thức.
      - "span_tiny"      : bản NÉN (THIẾT KẾ LẠI), feat=48, n_blocks=3 ->
                           0.230M tham số. Cùng số kênh (48) với SPAN baseline
                           nhưng chỉ 3/6 khối SPAB, KHÔNG dùng reparameterization
                           (Conv3XC) -> giảm 46.0% so với SPAN baseline đo ở
                           chế độ DEPLOY (0.4263M, đã kiểm chứng khớp 99.93%
                           với số liệu SPAN-S tác giả gốc công bố trong
                           arXiv:2311.12770 Table 1) — so sánh CÔNG BẰNG,
                           không phải so với tổng params bao gồm nhánh train
                           dư thừa (2.237M).
      - "span_official"  : bản CHÍNH THỨC, import trực tiếp từ repo đã clone
                           qua scripts/setup_span_official.sh — dùng khi cần
                           load checkpoint pretrained thật để fine-tune, HOẶC
                           train from-scratch ở kích thước tùy ý qua
                           feature_channels. Xem models/span_official_wrapper.py.
      - "span_large", "edsr": như cũ.

    feature_channels: chỉ áp dụng cho arch="span_official". None -> dùng mặc
    định 48 (khớp checkpoint pretrained chuẩn). Đặt khác 48 sẽ KHÔNG load
    được checkpoint pretrained (shape mismatch) -> phải train from-scratch.
    """
    arch = arch.lower()
    if arch == "span":
        return SPAN(scale=scale)
    if arch == "span_tiny":
        return SPAN(scale=scale, feat=48, n_blocks=3)
    if arch == "span_large":
        # biến thể lớn hơn — chỉ dùng để so sánh/khảo sát, KHÔNG phải mục tiêu triển khai
        return SPAN(scale=scale, feat=48, n_blocks=6)
    if arch == "span_official":
        from models.span_official_wrapper import build_official_span
        kwargs = {"scale": scale, "pretrained_path": pretrained_path}
        if feature_channels is not None:
            kwargs["feature_channels"] = feature_channels
        return build_official_span(**kwargs)
    if arch == "edsr":
        return EDSR(scale=scale)
    raise ValueError(f"Kiến trúc SR không hỗ trợ: {arch}")
