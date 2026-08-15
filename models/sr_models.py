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
import torch.nn.functional as F


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


"""
=====================================================================
[MỚI — đợt 7] Kiến trúc SR nhẹ BỔ SUNG để so sánh với span_tiny — mục đích:
chứng minh span_tiny cạnh tranh được với các kiến trúc SR nhẹ đã công bố
khác, không chỉ so với "phiên bản cắt gọn của chính SPAN".

[SỬA — lỗi tài liệu phát hiện qua review, đã sai ở 4 chỗ] Bản trước ghi các
kiến trúc này "train from-scratch bằng train_sr.py, pixel loss L1 thuần
túy — ĐÚNG cách EDSR/span_large đang được đối xử" — SAI: `span_large` (xem
`RUN_ALL_span_large_ablation.sh`, dòng gọi
`train_sr_distill.py --student_arch span_large`) THỰC RA cũng là một "học
sinh" distillation giống hệt `span_tiny` (cùng teacher=span_official, cùng
identity loss, cùng recipe train_sr_distill.py) — KHÔNG train pixel-loss-
thuần như EDSR. Chỉ EDSR mới thật sự train qua train_sr.py (pixel loss
thuần). Hệ quả của lỗi tài liệu này: so sánh ban đầu giữa span_tiny (có
distillation) và rlfn/ecbsr/safmn (không có, chỉ pixel loss) là so sánh LẪN
LỘN 2 biến số cùng lúc (kiến trúc VÀ recipe huấn luyện), không cô lập được
riêng "kiến trúc nào tốt hơn".

Đã sửa bằng cách cung cấp 2 track so sánh riêng biệt, KHÔNG trộn lẫn:
  - **Track A — kiến trúc thuần, recipe chuẩn của chính bài báo gốc**: train
    từ đầu bằng train_sr.py (pixel loss L1 thuần), qua build_sr_model() với
    arch="rlfn"/"rlfn_adapted"/"ecbsr"/"safmn"/"edsr". Recipe này khớp với
    cách các bài báo GỐC của RLFN/ECBSR/SAFMN tự train và báo cáo số liệu
    (không có bài nào trong 3 bài dùng distillation) — có ý nghĩa để đối
    chiếu với số liệu literature, nhưng KHÔNG so sánh công bằng trực tiếp
    với span_tiny (vì span_tiny CÓ distillation, các kiến trúc này ở track
    này KHÔNG có).
  - **Track B — CÙNG recipe distillation với span_tiny/span_large** [MỚI]:
    train qua train_sr_distill.py --student_arch rlfn_adapted/ecbsr/safmn
    (CÙNG teacher=span_official, CÙNG identity loss, CÙNG lambda trong
    config.yaml mục sr_improve) — cô lập được đúng biến "kiến trúc", vì mọi
    kiến trúc ở track này đều được lợi từ distillation NHƯ NHAU. Đây là so
    sánh công bằng trực tiếp với `span_tiny`/`span_large`. Xem
    `RUN_ALL_extra_sr_baseline_distilled.sh` (script mới) + README mục 11.

CẢ 4 kiến trúc mới (rlfn/rlfn_adapted/ecbsr/safmn) dưới đây đã được đối
chiếu TRỰC TIẾP với mã nguồn/bài báo gốc trước khi viết (không đoán công
thức) — xem chú thích riêng từng lớp. build_sr_model() dùng chung cho CẢ 2
track (train_sr.py và train_sr_distill.py đều gọi build_sr_model(arch,
scale, ...) giống hệt nhau) — không cần code kiến trúc riêng cho từng track.
=====================================================================
"""


class ESA(nn.Module):
    """Enhanced Spatial Attention — dùng trong RLFB (RLFN). Cấu trúc chuẩn
    kế thừa từ RFDN (Liu et al., 2020) mà bài báo RLFN tái sử dụng, CHỈ giảm
    số lớp conv trong "ConvGroups" từ 3 xuống 1 (xem RLFN paper, Section
    "Residual Local Feature Block" + Table 3: "RLFB uses just one [conv
    layer in ConvGroups]... does not sacrifice performance, but accelerates
    inference" — đã đối chiếu trực tiếp với văn bản bài báo arXiv:2205.07514).

    !!! GIỚI HẠN ĐÃ BIẾT (đợt 7, phát hiện qua rà soát lại — không phải lỗi
    code, là hệ quả của áp kiến trúc gốc vào ảnh LR RẤT nhỏ 20x20 của project
    này, nhỏ hơn nhiều so với LR patch chuẩn trong literature SR, thường
    ≥32-64px). Truy vết kích thước qua từng lớp với input 20x20 (H=W=20,
    esa_channels=16 mặc định — xem RLFN.__init__):
        conv1 (1x1, pad=0):                 20x20 -> 20x20
        conv2 (3x3, stride=2, pad=0):        20x20 -> floor((20-3)/2)+1 = 9x9
        max_pool2d(kernel=7, stride=3):       9x9  -> floor((9-7)/3)+1  = 1x1   <- SUY BIẾN
        conv3 (3x3, pad=1) trên input 1x1:    1x1  -> 1x1 (chỉ còn phép biến đổi
                                                        affine của 1 điểm)
        interpolate(nearest/bilinear) từ 1x1 lên lại 20x20: NHÂN BẢN 1 giá trị
                                                        ra toàn bộ không gian
    Hệ quả: attention map cuối cùng (biến `m`) không còn thay đổi theo VỊ TRÍ
    KHÔNG GIAN trong ảnh — mỗi kênh chỉ còn đúng 1 giá trị scalar, broadcast
    đều ra mọi pixel. Về bản chất, ESA ở scale ảnh 20x20 của project này hoạt
    động gần như 1 khối channel-attention kiểu Squeeze-and-Excitation, KHÔNG
    còn giữ được tính chất "Enhanced SPATIAL Attention" như tên gọi/thiết kế
    gốc (vốn nhắm vào ảnh/patch SR chuẩn, nơi max_pool2d(k=7,s=3) áp lên
    feature map lớn hơn nhiều, không suy biến về 1 điểm).

    QUYẾT ĐỊNH THIẾT KẾ [SỬA — cập nhật sau khi cân nhắc lại]: ban đầu định
    chỉ ghi nhận đây là giới hạn tài liệu (giữ nguyên kernel=7/stride=3 gốc,
    không sửa code). Cân nhắc lại: để `rlfn` suy biến trong bảng so sánh
    CHÍNH của luận án là rủi ro lớn hơn — nếu `span_tiny` thắng `rlfn`, kết
    luận đó dựa một phần trên 1 baseline đã mất chức năng thiết kế, reviewer
    SR giỏi sẽ không chỉ hỏi mà có thể bác cả bảng so sánh. Xử lý bằng cách
    cung cấp CẢ HAI biến thể qua tham số `pool_kernel_size`/`pool_stride`:
      - `esa_pool_kernel=7, esa_pool_stride=3` (mặc định — arch "rlfn"):
        ĐÚNG NGUYÊN VĂN cấu hình tác giả công bố — dùng để đối chiếu số liệu
        trực tiếp với bài báo gốc/literature, và làm bằng chứng định lượng
        cho hiện tượng suy biến (báo cáo trong ablation/Limitations).
      - `esa_pool_kernel=3, esa_pool_stride=2` (arch "rlfn_adapted"): CHỈ đổi
        đúng 2 số này trong `ESA`, KHÔNG đổi gì khác trong toàn bộ kiến trúc
        (không đổi conv2 của ESA, không đổi RLFB, không đổi số khối/kênh) —
        giữ v_max ở 4x4 thay vì suy biến về 1x1 (xem truy vết số bên dưới),
        khôi phục lại tính chất "spatial" thật cho attention map. Đây là
        thay đổi TỐI THIỂU, có chủ đích, áp dụng ĐỒNG NHẤT (không phải tinh
        chỉnh để "thắng" — chỉ để không suy biến toán học), và được ghi nhận
        RÕ RÀNG là biến thể riêng (không lặng lẽ thay thế "rlfn" gốc).

    Truy vết số cho "rlfn_adapted" (esa_pool_kernel=3, esa_pool_stride=2),
    input 20x20, conv2 GIỮ NGUYÊN (kernel=3, stride=2, pad=0 — không đổi):
        conv1 (1x1):                          20x20 -> 20x20
        conv2 (3x3, stride=2, pad=0, KHÔNG đổi): 20x20 -> 9x9 (như cũ)
        max_pool2d(kernel=3, stride=2):        9x9  -> floor((9-3)/2)+1 = 4x4  <- KHÔNG suy biến
        conv3 (3x3, pad=1) trên 4x4:            4x4  -> 4x4 (còn biến thiên không gian thật)
        interpolate 4x4 -> 20x20: nội suy từ lưới 4x4 thật, không phải nhân bản 1 điểm

    **Khuyến nghị dùng trong bài báo**: coi `rlfn_adapted` là baseline CHÍNH
    để so sánh công bằng với `span_tiny` trong bảng kết quả chính (vì đây là
    phiên bản THỰC SỰ triển khai đúng ý tưởng "spatial attention" ở scale
    ảnh của bài toán này); báo cáo `rlfn` (nguyên bản) như 1 dòng phụ/ablation
    kèm 1 câu giải thích hiện tượng suy biến — biến giới hạn này thành 1 phát
    hiện nhỏ đáng đưa vào Discussion thay vì một điểm yếu bị ẩn đi.

    `ecbsr` (không downsample nội bộ) và `safmn` (SAFM chỉ pool xuống tối
    thiểu 2x2 ở scale này, không suy biến về 1 điểm) KHÔNG gặp vấn đề tương
    tự — đã kiểm tra riêng, không cần biến thể "adapted".
    """

    def __init__(self, n_feats: int, esa_channels: int = 16,
                 pool_kernel_size: int = 7, pool_stride: int = 3):
        super().__init__()
        f = esa_channels
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)  # ConvGroups rút còn 1 lớp (đúng RLFN)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        # arch="rlfn" (mặc định pool_kernel_size=7, pool_stride=3, ĐÚNG bài
        # báo gốc): ở input 20x20 (LR thật của project), v_max suy biến về
        # 1x1 — xem ghi chú "GIỚI HẠN ĐÃ BIẾT"/"QUYẾT ĐỊNH THIẾT KẾ" ở
        # docstring lớp phía trên. arch="rlfn_adapted"
        # (pool_kernel_size=3, pool_stride=2): v_max giữ 4x4, không suy biến.
        v_max = F.max_pool2d(c1, kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        c3 = self.relu(self.conv3(v_max))
        c3 = F.interpolate(c3, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class RLFB(nn.Module):
    """Residual Local Feature Block — khối cơ bản của RLFN (Kong et al.,
    NTIRE 2022 winner runtime track, arXiv:2205.07514). Công thức đã đối
    chiếu trực tiếp với văn bản bài báo (Eq. 6-7 + đoạn mô tả 1x1 conv + ESA
    ngay sau):
        F_r1 = ReLU(Conv3x3(F_in));  F_r2 = ReLU(Conv3x3(F_r1));  F_r3 = ReLU(Conv3x3(F_r2))
        F_refined = F_in + F_r3                       # residual TRƯỚC
        out = ESA(Conv1x1(F_refined))                  # chiếu kênh rồi attention
    """

    def __init__(self, n_feats: int, esa_channels: int = 16,
                 esa_pool_kernel: int = 7, esa_pool_stride: int = 3):
        super().__init__()
        self.c1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.c5 = nn.Conv2d(n_feats, n_feats, kernel_size=1)  # chiếu kênh (Conv-1 trong bài báo)
        self.esa = ESA(n_feats, esa_channels, pool_kernel_size=esa_pool_kernel, pool_stride=esa_pool_stride)

    def forward(self, x):
        r = self.act(self.c1(x))
        r = self.act(self.c2(r))
        r = self.act(self.c3(r))
        r = x + r
        out = self.esa(self.c5(r))
        return out


class RLFN(nn.Module):
    """Residual Local Feature Network (Kong et al. 2022, arXiv:2205.07514).
    Cấu hình dùng ở đây là "RLFN-cut" — phiên bản đã giành GIẢI NHẤT hạng mục
    Runtime của NTIRE 2022 Efficient SR Challenge (bài báo, Section 4.4:
    "The proposed RLFN-cut has 4 RLFBs, in which the number of feature
    channels is set to 48 while the channel number of ESA is set to 16" —
    ~0.317M tham số, quy mô gần với span_tiny 0.230M, phù hợp so sánh công
    bằng ở project này).

    LƯU Ý QUAN TRỌNG [SỬA] — có 2 biến thể qua `build_sr_model()`:
      - arch="rlfn": ESA-pool kernel=7/stride=3 ĐÚNG NGUYÊN VĂN tác giả —
        SUY BIẾN thành gần như channel-attention (mất tính không gian) ở độ
        phân giải LR 20x20 của project này (xem docstring lớp `ESA` để biết
        chi tiết/số liệu truy vết). Dùng để đối chiếu với số liệu literature,
        báo cáo như 1 dòng ablation phụ + giải thích hiện tượng suy biến —
        KHÔNG dùng làm baseline chính.
      - arch="rlfn_adapted": ESA-pool kernel=3/stride=2 — không suy biến,
        giữ được tính chất "spatial attention" thật. Dùng làm baseline
        CHÍNH để so sánh công bằng với span_tiny trong bảng kết quả.
    Đây KHÔNG phải lỗi implementation ở "rlfn" — port đúng bài báo gốc; hạn
    chế cố hữu khi áp cấu hình gốc tác giả (thiết kế cho LR ≥32-64px) vào
    bài toán ảnh tai độ phân giải cực thấp.

    Xác nhận công bằng: `scale` (hệ số x4) truyền vào đây GIỐNG HỆT mọi kiến
    trúc SR khác trong project (span_tiny/span_baseline/span_large/edsr/
    ecbsr/safmn) — đều lấy từ `cfg["image"]["scale"]` DUY NHẤT trong
    `configs/config.yaml`, qua cùng 1 hàm `build_sr_model(arch, scale, ...)`
    — không có kiến trúc nào bị test ở scale khác các kiến trúc còn lại.
    """

    def __init__(self, scale: int = 4, channels_in: int = 3, feat: int = 48,
                 n_blocks: int = 4, esa_channels: int = 16,
                 esa_pool_kernel: int = 7, esa_pool_stride: int = 3):
        super().__init__()
        self.head = nn.Conv2d(channels_in, feat, kernel_size=3, padding=1)
        self.body = nn.ModuleList([
            RLFB(feat, esa_channels, esa_pool_kernel=esa_pool_kernel, esa_pool_stride=esa_pool_stride)
            for _ in range(n_blocks)
        ])
        self.body_tail = nn.Conv2d(feat, feat, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(feat, channels_in * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        feat = self.head(x)
        f0 = feat
        for block in self.body:
            feat = block(feat)
        feat = self.body_tail(feat) + f0
        out = self.upsample(feat)
        return torch.clamp(out, 0.0, 1.0)


class _SeqConv3x3(nn.Module):
    """Nhánh tái tham số hoá (re-parameterizable) dùng trong ECB — PORT TRỰC
    TIẾP từ mã nguồn chính thức BasicSR (basicsr/archs/ecbsr_arch.py, đã tải
    trực tiếp từ github.com/XPixelGroup/BasicSR để đối chiếu trước khi viết,
    không đoán công thức), chỉ đổi tên biến cho khớp style project. Gốc từ
    ECBSR (Zhang, Zeng, Zhang — ACM Multimedia 2021).

    4 loại nhánh: 'conv1x1-conv3x3' (conv thường 2 lớp), 'conv1x1-sobelx',
    'conv1x1-sobely' (đạo hàm bậc 1 theo 2 hướng), 'conv1x1-laplacian'
    (đạo hàm bậc 2) — mỗi nhánh trích đặc trưng biên (edge) khác nhau, cộng
    lại lúc train; lúc deploy hợp nhất (rep_params) thành 1 conv 3x3 duy nhất.
    """

    def __init__(self, seq_type, in_channels, out_channels, depth_multiplier=1.0):
        super().__init__()
        self.seq_type = seq_type
        self.in_channels = in_channels
        self.out_channels = out_channels

        if seq_type == "conv1x1-conv3x3":
            self.mid_planes = int(out_channels * depth_multiplier)
            conv0 = nn.Conv2d(in_channels, self.mid_planes, kernel_size=1)
            self.k0, self.b0 = conv0.weight, conv0.bias
            conv1 = nn.Conv2d(self.mid_planes, out_channels, kernel_size=3)
            self.k1, self.b1 = conv1.weight, conv1.bias
        else:
            conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.k0, self.b0 = conv0.weight, conv0.bias
            scale = torch.randn(size=(out_channels, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(out_channels) * 1e-3
            self.bias = nn.Parameter(bias.reshape(out_channels))

            mask = torch.zeros((out_channels, 1, 3, 3), dtype=torch.float32)
            if seq_type == "conv1x1-sobelx":
                for i in range(out_channels):
                    mask[i, 0, 0, 0], mask[i, 0, 1, 0], mask[i, 0, 2, 0] = 1.0, 2.0, 1.0
                    mask[i, 0, 0, 2], mask[i, 0, 1, 2], mask[i, 0, 2, 2] = -1.0, -2.0, -1.0
            elif seq_type == "conv1x1-sobely":
                for i in range(out_channels):
                    mask[i, 0, 0, 0], mask[i, 0, 0, 1], mask[i, 0, 0, 2] = 1.0, 2.0, 1.0
                    mask[i, 0, 2, 0], mask[i, 0, 2, 1], mask[i, 0, 2, 2] = -1.0, -2.0, -1.0
            elif seq_type == "conv1x1-laplacian":
                for i in range(out_channels):
                    mask[i, 0, 0, 1] = 1.0
                    mask[i, 0, 1, 0] = 1.0
                    mask[i, 0, 1, 2] = 1.0
                    mask[i, 0, 2, 1] = 1.0
                    mask[i, 0, 1, 1] = -4.0
            else:
                raise ValueError(f"seq_type không hỗ trợ: {seq_type}")
            self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, x):
        if self.seq_type == "conv1x1-conv3x3":
            y0 = F.conv2d(x, self.k0, self.b0, stride=1)
            y0 = F.pad(y0, (1, 1, 1, 1), "constant", 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            return F.conv2d(y0, self.k1, self.b1, stride=1)
        else:
            y0 = F.conv2d(x, self.k0, self.b0, stride=1)
            y0 = F.pad(y0, (1, 1, 1, 1), "constant", 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            return F.conv2d(y0, self.scale * self.mask, self.bias, stride=1, groups=self.out_channels)

    def rep_params(self):
        if self.seq_type == "conv1x1-conv3x3":
            rep_weight = F.conv2d(self.k1, self.k0.permute(1, 0, 2, 3))
            rep_bias_map = torch.ones(1, self.mid_planes, 3, 3, device=self.k0.device) * self.b0.view(1, -1, 1, 1)
            rep_bias = F.conv2d(rep_bias_map, self.k1).view(-1) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_channels, self.out_channels, 3, 3), device=self.k0.device)
            for i in range(self.out_channels):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            rep_weight = F.conv2d(k1, self.k0.permute(1, 0, 2, 3))
            rep_bias_map = torch.ones(1, self.out_channels, 3, 3, device=self.k0.device) * self.b0.view(1, -1, 1, 1)
            rep_bias = F.conv2d(rep_bias_map, k1).view(-1) + self.bias
        return rep_weight, rep_bias


class ECB(nn.Module):
    """Edge-oriented Convolution Block (ECBSR, Zhang et al. 2021, ACM
    Multimedia — https://github.com/xindongzhang/ECBSR). Port trực tiếp từ
    BasicSR chính thức (đã tải nguồn thật để đối chiếu). Train: cộng 5 nhánh
    (conv3x3 + 4 nhánh conv1x1->{conv3x3,sobel_x,sobel_y,laplacian}). Eval:
    5 nhánh hợp nhất thành 1 conv 3x3 duy nhất (structural
    reparameterization) — CÙNG kiểu kỹ thuật với Conv3XC của SPAN chính thức
    trong project này, nên cố ý expose đúng 2 thuộc tính `eval_conv` +
    phương thức `update_params()` mà `utils/metrics.py::freeze_reparam_modules()`
    / `count_params_deploy_mode()` đã tìm kiếm sẵn — TÁI SỬ DỤNG được 2 hàm
    đó nguyên vẹn cho ECBSR, không cần thêm code path riêng (tránh đúng kiểu
    lỗi "quên xử lý 1 nhánh mới" đã từng gặp trong project này, xem CHANGELOG).

    QUAN TRỌNG về tính đúng: mặc định (chưa gọi freeze_reparam_modules()),
    forward() ở eval mode LUÔN tính lại hợp nhất (đúng nhưng chậm hơn) —
    giống hệt hành vi đã ghi nhận của Conv3XC gốc (utils/metrics.py, mục lỗi
    #7 trong README) — tránh vấn đề "eval_conv cũ, dùng nhầm trọng số cũ
    trong lúc validation xen kẽ train/eval mỗi epoch" nếu cache vĩnh viễn.
    """

    def __init__(self, in_channels: int, out_channels: int, depth_multiplier: float = 2.0,
                 act_type: str = "prelu", with_idt: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_type = act_type
        self.with_idt = with_idt and (in_channels == out_channels)

        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1x1_3x3 = _SeqConv3x3("conv1x1-conv3x3", in_channels, out_channels, depth_multiplier)
        self.conv1x1_sbx = _SeqConv3x3("conv1x1-sobelx", in_channels, out_channels)
        self.conv1x1_sby = _SeqConv3x3("conv1x1-sobely", in_channels, out_channels)
        self.conv1x1_lpl = _SeqConv3x3("conv1x1-laplacian", in_channels, out_channels)

        # [Giao diện dùng chung với Conv3XC — xem docstring lớp] "eval_conv"
        # KHÔNG phải tham số học độc lập (chỉ là bản hợp nhất từ 5 nhánh
        # trên), nên khoá requires_grad=False — tránh optimizer lãng phí cấp
        # phát/tối ưu 1 bản sao vô nghĩa (giá trị của nó luôn bị ghi đè lại
        # bởi update_params() mỗi lần eval, không tham gia đồ thị gradient).
        self.eval_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False

        if act_type == "prelu":
            self.act = nn.PReLU(num_parameters=out_channels)
        elif act_type == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act_type == "linear":
            self.act = None
        else:
            raise ValueError(f"act_type không hỗ trợ: {act_type}")

    def update_params(self):
        """Tính hợp nhất 5 nhánh thành 1 conv 3x3 (rep_params), ghi vào
        eval_conv. Gọi lại hàm này bất cứ khi nào trọng số 5 nhánh train đã
        thay đổi (mặc định forward() tự gọi lại mỗi lần ở eval mode — xem
        forward() bên dưới); `freeze_reparam_modules()` trong utils/metrics.py
        gọi 1 LẦN DUY NHẤT rồi vô hiệu hoá việc gọi lại, dùng khi đo latency."""
        w0, b0 = self.conv3x3.weight, self.conv3x3.bias
        w1, b1 = self.conv1x1_3x3.rep_params()
        w2, b2 = self.conv1x1_sbx.rep_params()
        w3, b3 = self.conv1x1_sby.rep_params()
        w4, b4 = self.conv1x1_lpl.rep_params()
        rep_weight, rep_bias = (w0 + w1 + w2 + w3 + w4), (b0 + b1 + b2 + b3 + b4)

        if self.with_idt:
            weight_idt = torch.zeros(self.out_channels, self.out_channels, 3, 3, device=rep_weight.device)
            for i in range(self.out_channels):
                weight_idt[i, i, 1, 1] = 1.0
            rep_weight = rep_weight + weight_idt

        self.eval_conv.weight.data = rep_weight
        self.eval_conv.bias.data = rep_bias

    def forward(self, x):
        if self.training:
            y = (self.conv3x3(x) + self.conv1x1_3x3(x) + self.conv1x1_sbx(x)
                 + self.conv1x1_sby(x) + self.conv1x1_lpl(x))
            if self.with_idt:
                y = y + x
        else:
            self.update_params()
            y = self.eval_conv(x)
        if self.act is not None:
            y = self.act(y)
        return y


class ECBSR(nn.Module):
    """ECBSR (Zhang et al. 2021). Cấu hình dùng ở đây tương ứng biến thể
    "m4c16" (4 khối ECB, 16 kênh) trong repo chính thức
    (configs/ecbsr_x4_m4c16_prelu.yml — 1 trong các cấu hình nhẹ nhất tác
    giả công bố sẵn, phù hợp so sánh với span_tiny/RLFN quy mô nhỏ).

    LƯU Ý khác biệt có chủ đích với bài báo gốc: bản gốc train/test trên
    kênh Y (YCbCr, ảnh xám) — ở đây dùng RGB 3 kênh (num_in_ch=num_out_ch=3)
    để NHẤT QUÁN với mọi kiến trúc SR khác trong project này (đều nhận/trả
    RGB) — không phải lỗi, là lựa chọn có chủ đích, ghi rõ ở đây.
    """

    def __init__(self, scale: int = 4, channels_in: int = 3, num_block: int = 4,
                 num_channel: int = 16, with_idt: bool = True, act_type: str = "prelu"):
        super().__init__()
        self.channels_in = channels_in
        self.scale = scale

        backbone = [ECB(channels_in, num_channel, depth_multiplier=2.0, act_type=act_type, with_idt=with_idt)]
        for _ in range(num_block):
            backbone.append(ECB(num_channel, num_channel, depth_multiplier=2.0, act_type=act_type, with_idt=with_idt))
        backbone.append(ECB(num_channel, channels_in * scale * scale, depth_multiplier=2.0,
                             act_type="linear", with_idt=with_idt))
        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, x):
        shortcut = torch.repeat_interleave(x, self.scale * self.scale, dim=1)
        y = self.backbone(x) + shortcut
        y = self.upsampler(y)
        return torch.clamp(y, 0.0, 1.0)


class _SAFMLayerNorm(nn.Module):
    """LayerNorm kiểu 'channels_first' (chuẩn hoá theo chiều kênh tại mỗi vị
    trí không gian) — port trực tiếp từ basicsr/archs/safmn_arch.py chính
    thức (sunny2109/SAFMN, đã tải nguồn thật để đối chiếu)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class _SAFMCCM(nn.Module):
    """Convolutional Channel Mixer — feed-forward của SAFMN (conv3x3 ->
    GELU -> conv1x1), port trực tiếp từ nguồn chính thức."""

    def __init__(self, dim: int, growth_rate: float = 2.0):
        super().__init__()
        hidden = int(dim * growth_rate)
        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.ccm(x)


class _SAFM(nn.Module):
    """Spatially-Adaptive Feature Modulation — module lõi của SAFMN (Sun et
    al., ICCV 2023, arXiv 2302.13800). Chia kênh thành n_levels nhóm, mỗi
    nhóm xử lý ở 1 độ phân giải không gian khác nhau (giảm dần theo cấp số
    nhân qua adaptive_max_pool2d) rồi nội suy về lại kích thước gốc — port
    trực tiếp từ basicsr/archs/safmn_arch.py chính thức."""

    def __init__(self, dim: int, n_levels: int = 4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels
        self.mfr = nn.ModuleList([
            nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for _ in range(n_levels)
        ])
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (max(1, h // 2 ** i), max(1, w // 2 ** i))
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode="nearest")
            else:
                s = self.mfr[i](xc[i])
            out.append(s)
        out = self.aggr(torch.cat(out, dim=1))
        return self.act(out) * x


class _SAFMAttBlock(nn.Module):
    """Feature Mixing Module (FMM) = SAFM + CCM, mỗi nhánh có LayerNorm +
    residual riêng — port trực tiếp từ nguồn chính thức."""

    def __init__(self, dim: int, ffn_scale: float = 2.0):
        super().__init__()
        self.norm1 = _SAFMLayerNorm(dim)
        self.norm2 = _SAFMLayerNorm(dim)
        self.safm = _SAFM(dim)
        self.ccm = _SAFMCCM(dim, ffn_scale)

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x


class SAFMN(nn.Module):
    """SAFMN (Sun et al., ICCV 2023 — "Spatially-Adaptive Feature Modulation
    for Efficient Image Super-Resolution"). Cấu hình dùng ở đây (dim=36,
    n_blocks=8) KHỚP với cấu hình mặc định trong chính script demo của repo
    chính thức (basicsr/archs/safmn_arch.py, khối `if __name__ == "__main__"`)
    — biến thể nhẹ nhất tác giả tự dùng để đo độ phức tạp, ~0.228M tham số,
    quy mô gần với span_tiny (0.230M)."""

    def __init__(self, scale: int = 4, channels_in: int = 3, dim: int = 36,
                 n_blocks: int = 8, ffn_scale: float = 2.0):
        super().__init__()
        self.to_feat = nn.Conv2d(channels_in, dim, 3, 1, 1)
        self.feats = nn.Sequential(*[_SAFMAttBlock(dim, ffn_scale) for _ in range(n_blocks)])
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, channels_in * scale ** 2, 3, 1, 1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        feat = self.to_feat(x)
        feat = self.feats(feat) + feat
        out = self.to_img(feat)
        return torch.clamp(out, 0.0, 1.0)


class _SMFADMlp(nn.Module):
    """[MỚI — bổ sung journal Q1] Nhánh MLP cục bộ (depthwise 3x3 -> 1x1 ->
    GELU -> 1x1) dùng trong SMFA — port TRỰC TIẾP từ code chính thức nộp cho
    NTIRE2024_ESR (models/team24_smfan.py trong repo Amazingren/NTIRE2024_ESR,
    tải ngày viết code này), khớp với kiến trúc công bố trong bài báo gốc:
    Zheng, Sun, Dong, Pan, "SMFANet: A Lightweight Self-Modulation Feature
    Aggregation Network for Efficient Image Super-Resolution", ECCV 2024
    (Springer LNCS, DOI 10.1007/978-3-031-72973-7_21). Repo tác giả chính
    thức: github.com/Zheng-MJ/SMFANet — 2 file khớp nhau (đã đối chiếu)."""

    def __init__(self, dim, growth_rate=2.0, bias=True):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim, bias=bias),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=bias),
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=bias)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x


class _SMFAPCFN(nn.Module):
    """[MỚI] Partial-Conv Feed-Forward Network — chỉ áp conv 3x3 lên MỘT
    PHẦN kênh (p_rate=0.25) để giảm chi phí tính toán (kỹ thuật "partial
    convolution" kiểu FasterNet). QUAN TRỌNG: có 2 nhánh code KHÁC NHAU giữa
    train và eval trong code CHÍNH THỨC (không phải lỗi, port nguyên văn):
      - train: split kênh ra 2 phần bằng torch.split (an toàn cho autograd)
      - eval: gán trực tiếp qua slice in-place x[:, :p_dim, ...] = ... (nhanh
        hơn nhưng KHÔNG an toàn cho autograd nếu dùng lúc train) — chỉ đúng
        vì đã bọc @torch.no_grad() ở toàn bộ pha eval trong eval_sr_quality.py.
    Giữ nguyên logic này (không "dọn" về 1 nhánh) để khớp đúng hành vi tác
    giả đã công bố/đo đạc."""

    def __init__(self, dim, growth_rate=2.0, p_rate=0.25, bias=True):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=bias)
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1, bias=bias)
        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=bias)
        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1, x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x = x.clone()  # [SỬA nhỏ so với bản gốc] tránh sửa in-place lên tensor
            # có thể đang được autograd/checkpoint khác tham chiếu; an toàn hơn
            # bản gốc mà KHÔNG đổi kết quả số học (chỉ khác ở chỗ không ghi đè
            # trực tiếp lên activation của conv_0, dùng bản sao) — vẫn @torch.no_grad()
            # nên không ảnh hưởng hiệu năng tính toán đáng kể ở quy mô ảnh 20x20.
            x[:, :self.p_dim, :, :] = self.act(self.conv_1(x[:, :self.p_dim, :, :]))
            x = self.conv_2(x)
        return x


class _SMFA(nn.Module):
    """[MỚI] Self-Modulation Feature Aggregation — module chính của SMFANet.
    Nhánh "x" (EASA — Efficient Approximation of Self-Attention): pool
    xuống lưới thô (h//down_scale, w//down_scale) bằng adaptive_max_pool2d,
    dw-conv, cộng với phương sai không gian (torch.var) rồi upsample lại
    bằng nearest — xấp xỉ self-attention KHÔNG cần tính ma trận attention
    NxN đầy đủ (nhẹ hơn transformer thật). Nhánh "y" (LDE — Local Detail
    Estimation): qua _SMFADMlp, giữ chi tiết cục bộ độ phân giải đầy đủ.

    KIỂM TRA SUY BIẾN Ở ĐỘ PHÂN GIẢI LR CỦA PROJECT NÀY (20x20, xem
    lr_size=hr_size//scale=80//4=20 trong configs/config.yaml) — ĐÃ RÚT KINH
    NGHIỆM từ lỗi suy biến ESA/RLFN phát hiện trước đó (xem class ESA phía
    trên): down_scale mặc định = 8 -> adaptive_max_pool2d(x, (20//8, 20//8))
    = (2, 2). KHÔNG suy biến về 1x1 (khác hẳn tình huống RLFN) — vẫn giữ
    được lưới không gian 2x2 rời rạc, "self-modulation" vẫn có thông tin
    không gian (dù thô) để hoạt động đúng chức năng thiết kế, KHÔNG suy biến
    thành pure channel-attention. Không cần biến thể "adapted" cho kiến trúc
    này."""

    def __init__(self, dim=36, bias=True, down_scale=8):
        super().__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0, bias=bias)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
        self.lde = _SMFADMlp(dim, 2, bias)
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias)
        self.gelu = nn.GELU()
        self.down_scale = down_scale

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        pool_h = max(1, h // self.down_scale)
        pool_w = max(1, w // self.down_scale)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (pool_h, pool_w)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s + x_v)), size=(h, w), mode="nearest")
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)


class _SMFAFMB(nn.Module):
    """[MỚI] Feature Modulation Block = 1 khối _SMFA + 1 khối _SMFAPCFN, mỗi
    khối có residual riêng, chuẩn hoá bằng F.normalize (khác LayerNorm/BatchNorm
    thường gặp — port nguyên văn theo code chính thức, KHÔNG tự đổi sang
    chuẩn hoá khác)."""

    def __init__(self, dim, ffn_scale=2.0, bias=True):
        super().__init__()
        self.smfa = _SMFA(dim, bias)
        self.pcfn = _SMFAPCFN(dim, ffn_scale, bias=bias)

    def forward(self, x):
        x = self.smfa(F.normalize(x)) + x
        x = self.pcfn(F.normalize(x)) + x
        return x


class SMFANet(nn.Module):
    """[MỚI — bổ sung journal Q1, baseline SR công bố 2024, gần đây nhất
    trong toàn bộ project] SMFANet (Zheng et al., ECCV 2024) — xem docstring
    _SMFADMlp phía trên để biết nguồn/DOI chi tiết. Cấu hình mặc định ở đây
    (dim=24, n_blocks=8, ffn_scale=1.5, bias=False) KHỚP NGUYÊN VĂN với khối
    `if __name__ == "__main__"` của file nộp NTIRE2024_ESR (team24_smfan.py)
    — đây CHÍNH LÀ cấu hình đã dùng để dự thi NTIRE2024 (biến thể nhẹ hơn bản
    "SMFANet" đầy đủ trong bài báo, giống cách "safmn" ở trên dùng cấu hình
    demo nhẹ nhất của SAFMN thay vì bản đầy đủ trong bài — nhất quán quy ước
    trong project này: ưu tiên biến thể NHẸ NHẤT có căn cứ chính thức, cùng
    tầm với span_tiny).

    [SỬA so với code gốc] Thêm torch.clamp(out, 0, 1) ở cuối forward() —
    code chính thức KHÔNG clamp, nhưng MỌI model SR khác trong project này
    (SPAN, EDSR, RLFN, ECBSR, SAFMN) đều clamp trước khi trả về, để so sánh
    PSNR/SSIM/LPIPS công bằng (không model nào được lợi/thiệt vì đôi khi ra
    giá trị hơi ngoài [0,1] rồi bị đo sai) — ĐÂY LÀ QUYẾT ĐỊNH NHẤT QUÁN HOÁ
    PROTOCOL nội bộ, không phải sửa lỗi của tác giả gốc."""

    def __init__(self, scale: int = 4, channels_in: int = 3, dim: int = 24,
                 n_blocks: int = 8, ffn_scale: float = 1.5, bias: bool = False):
        super().__init__()
        self.to_feat = nn.Conv2d(channels_in, dim, 3, 1, 1, bias=bias)
        self.feats = nn.Sequential(*[_SMFAFMB(dim, ffn_scale) for _ in range(n_blocks)])
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, channels_in * scale ** 2, 3, 1, 1, bias=bias),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        feat = self.to_feat(x)
        feat = self.feats(feat) + feat
        out = self.to_img(feat)
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
      - "rlfn"    : [MỚI] RLFN-cut (Kong et al. 2022, NTIRE22 runtime-track
                    winner) — baseline SR nhẹ BÊN NGOÀI họ SPAN, train
                    from-scratch bằng train_sr.py (pixel loss L1), KHÔNG
                    phải học sinh distillation. feat=48, n_blocks=4,
                    esa_channels=16 -> ~0.317M tham số. CẤU HÌNH ESA-pool
                    ĐÚNG NGUYÊN VĂN tác giả (kernel=7,stride=3) — SUY BIẾN
                    thành gần-channel-attention ở scale ảnh 20x20 của project
                    này (xem docstring lớp ESA) — dùng để đối chiếu số liệu
                    với literature, KHÔNG dùng làm baseline chính trong bảng
                    so sánh chính, xem "rlfn_adapted".
      - "rlfn_adapted": [MỚI] Y HỆT "rlfn" nhưng ESA-pool dùng
                    kernel=3,stride=2 (chỉ đổi đúng 2 số này) để tránh suy
                    biến về 1x1 ở scale ảnh của project — giữ v_max=4x4,
                    ESA hoạt động đúng chức năng "spatial attention". Đây là
                    baseline SR nhẹ ngoài họ SPAN dùng CHÍNH trong bảng so
                    sánh kết quả với span_tiny (xem README mục 11).
      - "ecbsr"   : [MỚI] ECBSR (Zhang et al. 2021, ACM MM) — cấu hình
                    m4c16 (4 khối ECB, 16 kênh), train from-scratch. Dùng
                    RGB 3 kênh (khác bài gốc dùng Y-channel) để nhất quán
                    với phần còn lại của project.
      - "safmn"   : [MỚI] SAFMN (Sun et al. 2023, ICCV) — dim=36, n_blocks=8
                    (cấu hình demo mặc định của repo chính thức), train
                    from-scratch, ~0.228M tham số.
      - "smfanet" : [MỚI — bổ sung journal Q1] SMFANet (Zheng et al., ECCV
                    2024, Springer LNCS) — công bố GẦN NHẤT (2024) trong toàn
                    bộ các baseline SR của project, giải quyết đúng khoảng
                    trống "baseline hơi cũ" nêu trong review. dim=24,
                    n_blocks=8, ffn_scale=1.5 (khớp file nộp chính thức
                    NTIRE2024_ESR/models/team24_smfan.py). Không suy biến
                    không gian ở LR 20x20 (đã kiểm tra, xem docstring _SMFA).

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
    if arch == "rlfn":
        return RLFN(scale=scale, feat=48, n_blocks=4, esa_channels=16,
                    esa_pool_kernel=7, esa_pool_stride=3)  # nguyên văn tác giả (suy biến ở 20x20)
    if arch == "rlfn_adapted":
        return RLFN(scale=scale, feat=48, n_blocks=4, esa_channels=16,
                    esa_pool_kernel=3, esa_pool_stride=2)  # tránh suy biến, xem docstring ESA
    if arch == "ecbsr":
        return ECBSR(scale=scale, num_block=4, num_channel=16, with_idt=True, act_type="prelu")
    if arch == "safmn":
        return SAFMN(scale=scale, dim=36, n_blocks=8, ffn_scale=2.0)
    if arch == "smfanet":
        return SMFANet(scale=scale, dim=24, n_blocks=8, ffn_scale=1.5, bias=False)
    raise ValueError(f"Kiến trúc SR không hỗ trợ: {arch}")
