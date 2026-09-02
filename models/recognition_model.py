"""
Backbone dùng chung, tách thành 2 head: identity + gender.

Hỗ trợ nhiều backbone để chạy BENCHMARK đa mô hình (Bước benchmark baseline
trong lộ trình) — mục đích: kiểm chứng SPAN cải tiến có giúp ích nhất quán
trên NHIỀU kiến trúc recognition khác nhau, không chỉ ăn may với 1 model.

Backbone có sẵn (đều load được qua torchvision/timm, không cần cài đặt tay):
  - mobilenet_v2      : nhẹ, đã dùng làm mặc định ban đầu
  - mobilenet_v3_small: nhẹ hơn V2, kiến trúc mới hơn (2019, NAS-based)
  - resnet18          : chuẩn tham chiếu kinh điển, không quá nhẹ
  - efficientnet_b0   : cân bằng tốt accuracy/hiệu năng, phổ biến trong benchmark biometric
  - ghostnet_100      : dòng kiến trúc chuyên cho lightweight face recognition
                        (nền tảng của GhostFaceNet) — cần cài `timm`

[MỚI — mở rộng 7 backbone mới, ưu tiên chọn kiến trúc NHẸ, mỗi backbone đại
diện 1 họ kiến trúc khác với 5 backbone gốc ở trên VÀ khác nhau giữa 7
backbone mới này (tránh trùng họ với mobilenet_v2/v3_small đã có), đã kiểm
chứng thực nghiệm (dummy forward 128x128 -> feature map 4D hợp lệ) trước khi
thêm, không đoán tên hàm/tham số:
  - shufflenet_v2_x1_0 : 2.28M tham số — nhẹ nhất trong 7 backbone mới (trừ
                        squeezenet1_1), họ channel-shuffle + group convolution
                        (khác hẳn depthwise-separable của họ MobileNet)
  - squeezenet1_1      : 1.24M tham số — nhẹ nhất trong TOÀN BỘ 12 backbone
                        (kể cả 5 backbone gốc), họ Fire-module (squeeze +
                        expand 1x1/3x3), kiến trúc cổ điển cho edge device
  - mnasnet1_0         : 4.38M tham số — họ kiến trúc tìm bằng NAS (Neural
                        Architecture Search) tối ưu latency trên di động,
                        khác cách tìm kiến trúc của MobileNetV3 (cũng NAS
                        nhưng search space/reward khác)
  - mobilenet_v3_large : 5.48M tham số — cùng họ MobileNetV3 với
                        mobilenet_v3_small đã có, nhưng cấu hình lớn hơn —
                        cho phép so sánh trực tiếp ảnh hưởng của việc mở
                        rộng cấu hình TRONG CÙNG 1 họ kiến trúc
  - regnet_y_400mf     : 4.34M tham số — họ RegNet (thiết kế bằng search
                        không gian thiết kế mạng, không phải NAS từng
                        kiến trúc lẻ), có khối Squeeze-and-Excitation
  - mobileone_s0       : 4.27M tham số (qua timm) — họ reparameterization
                        (Apple): train với nhiều nhánh song song, gộp lại
                        thành 1 nhánh conv đơn lúc deploy — cơ chế hoàn toàn
                        khác 6 backbone mới còn lại, tối ưu riêng cho độ trễ
                        suy luận trên thiết bị di động
  - lcnet_100          : 1.67M tham số (qua timm, PP-LCNet của Baidu) — thiết
                        kế THỦ CÔNG (không phải NAS) tối ưu riêng cho suy luận
                        trên CPU/thiết bị cấu hình thấp, họ kiến trúc riêng
                        biệt không trùng với bất kỳ backbone nào khác
  (Đã cân nhắc ghostnetv2_100 thay cho 1 trong 2 backbone cuối, nhưng loại bỏ
  vì quá giống ghostnet_100 đã có — cùng cơ chế Ghost module cốt lõi, chỉ
  thêm attention DFC, không đại diện họ kiến trúc mới.)
Tất cả 7 đều dưới 5.5M tham số (span_tiny+recognition vẫn nhẹ hơn nhiều so
với resnet18's 11.7M — backbone "không nhẹ" duy nhất trong 5 backbone gốc,
giữ nguyên vai trò tham chiếu, không phải mục tiêu mở rộng). mobileone_s0 và
lcnet_100 cần cài `timm` (đã là dependency có sẵn từ ghostnet_100).
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


SUPPORTED_BACKBONES = [
    "mobilenet_v2",
    "mobilenet_v3_small",
    "resnet18",
    "efficientnet_b0",
    "ghostnet_100",
    # [MỚI — 7 backbone mở rộng, ưu tiên nhẹ, xem docstring đầu file]
    "shufflenet_v2_x1_0",
    "squeezenet1_1",
    "mnasnet1_0",
    "mobilenet_v3_large",
    "regnet_y_400mf",
    "mobileone_s0",
    "lcnet_100",
]

# [MỚI — đợt 7, sửa lỗi phát hiện qua code review] Cả 5 backbone trên đều
# dùng pretrained=True (trọng số ImageNet) nhưng toàn bộ pipeline hiện tại
# (datasets/ear_dataset.py) chỉ ToTensor() ảnh về [0,1], KHÔNG chuẩn hoá theo
# mean/std ImageNet — ảnh đưa vào model bị lệch phân phối so với dữ liệu
# ImageNet gốc mà các trọng số pretrained này được học trên đó, làm giảm
# hiệu quả transfer learning từ pretrained (không gây lỗi runtime nào, chỉ
# âm thầm làm kém hơn — dạng lỗi khó phát hiện nếu không rà soát kỹ).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_backbone(name: str, pretrained: bool):
    """Trả về (feature_extractor, feat_dim). feature_extractor nhận ảnh, trả về
    feature map 4D (B, C, H, W) trước global pooling.

    feat_dim được ĐO THẬT bằng 1 lần forward giả (dummy input), KHÔNG tin vào
    metadata như base.num_features/last_channel/classifier[...].in_features —
    các giá trị này có thể KHÔNG khớp thực tế tùy kiến trúc/phiên bản thư viện
    (đã xảy ra với ghostnet_100 qua timm: num_features báo 960 nhưng output
    thật khi global_pool="" là 1280 kênh, gây lỗi RuntimeError shape mismatch)."""

    if name == "mobilenet_v2":
        base = tv_models.mobilenet_v2(
            weights=tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        feature_extractor = base.features

    elif name == "mobilenet_v3_small":
        base = tv_models.mobilenet_v3_small(
            weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        feature_extractor = base.features

    elif name == "resnet18":
        base = tv_models.resnet18(
            weights=tv_models.ResNet18_Weights.DEFAULT if pretrained else None)
        feature_extractor = nn.Sequential(*list(base.children())[:-2])

    elif name == "efficientnet_b0":
        base = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        feature_extractor = base.features

    elif name == "ghostnet_100":
        if not _HAS_TIMM:
            raise ImportError(
                "Backbone 'ghostnet_100' cần thư viện timm. Cài: "
                "pip install timm --break-system-packages")
        feature_extractor = timm.create_model("ghostnet_100", pretrained=pretrained,
                                               num_classes=0, global_pool="")

    elif name == "shufflenet_v2_x1_0":
        base = tv_models.shufflenet_v2_x1_0(
            weights=tv_models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None)
        # Không có thuộc tính .features gộp sẵn như MobileNet — ghép thủ công
        # mọi tầng TRỪ .fc (đã kiểm chứng bằng dummy forward: ra 4D hợp lệ).
        feature_extractor = nn.Sequential(
            base.conv1, base.maxpool, base.stage2, base.stage3, base.stage4, base.conv5)

    elif name == "squeezenet1_1":
        base = tv_models.squeezenet1_1(
            weights=tv_models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None)
        feature_extractor = base.features

    elif name == "mnasnet1_0":
        base = tv_models.mnasnet1_0(
            weights=tv_models.MNASNet1_0_Weights.DEFAULT if pretrained else None)
        feature_extractor = base.layers

    elif name == "mobilenet_v3_large":
        base = tv_models.mobilenet_v3_large(
            weights=tv_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        feature_extractor = base.features

    elif name == "regnet_y_400mf":
        base = tv_models.regnet_y_400mf(
            weights=tv_models.RegNet_Y_400MF_Weights.DEFAULT if pretrained else None)
        # Không có .features gộp sẵn — ghép .stem + .trunk_output, loại bỏ
        # .avgpool/.fc (đã kiểm chứng bằng dummy forward: ra 4D hợp lệ).
        feature_extractor = nn.Sequential(base.stem, base.trunk_output)

    elif name == "mobileone_s0":
        if not _HAS_TIMM:
            raise ImportError(
                "Backbone 'mobileone_s0' cần thư viện timm. Cài: "
                "pip install timm --break-system-packages")
        feature_extractor = timm.create_model("mobileone_s0", pretrained=pretrained,
                                               num_classes=0, global_pool="")

    elif name == "lcnet_100":
        if not _HAS_TIMM:
            raise ImportError(
                "Backbone 'lcnet_100' cần thư viện timm. Cài: "
                "pip install timm --break-system-packages")
        feature_extractor = timm.create_model("lcnet_100", pretrained=pretrained,
                                               num_classes=0, global_pool="")

    else:
        raise ValueError(f"Backbone không hỗ trợ: {name}. Chọn trong {SUPPORTED_BACKBONES}")

    # Đo feat_dim thật — dùng ảnh 128x128 (đủ lớn để tránh mọi rủi ro co lại
    # về 0 chiều không gian ở tầng cuối, kênh output không phụ thuộc kích
    # thước không gian đầu vào với các kiến trúc dùng global pooling này).
    feature_extractor.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 128, 128)
        dummy_out = feature_extractor(dummy)
    feat_dim = dummy_out.shape[1]
    feature_extractor.train()

    return feature_extractor, feat_dim


class EarRecognitionNet(nn.Module):
    def __init__(self, num_identities: int, num_genders: int = 2,
                 embedding_dim: int = 256, backbone: str = "mobilenet_v2",
                 pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self.features, feat_dim = _build_backbone(backbone, pretrained)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.identity_head = nn.Linear(embedding_dim, num_identities)
        self.gender_head = nn.Linear(embedding_dim, num_genders)

        # [MỚI — đợt 7] Chuẩn hoá ImageNet đặt NGAY TRONG model (buffer, không
        # phải tham số học) thay vì trong Dataset transform — CỐ Ý: đảm bảo
        # MỌI đường vào model (forward() qua EarDataset [0,1], LẪN embed() gọi
        # trực tiếp từ train_sr_distill.py với ảnh SR/HR thô [0,1] từ
        # HRLRPairDataset) đều được chuẩn hoá NHẤT QUÁN qua đúng 1 chỗ duy
        # nhất — tránh đúng kiểu lỗi "sửa 1 nơi, quên nơi khác" đã từng gặp
        # trong project này (ví dụ scripts/make_finetune_config.py quên mục
        # "sr", xem CHANGELOG). Nếu chuẩn hoá trong Dataset thay vì ở đây,
        # ảnh HR/SR đưa vào recognition_model.embed() lúc train span_tiny sẽ
        # KHÔNG được chuẩn hoá (HRLRPairDataset chỉ ToTensor()), gây lệch
        # phân phối input so với lúc train recognition — sai mà không có lỗi
        # runtime nào báo hiệu.
        self.register_buffer("_imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def _normalize(self, x):
        mean = self._imagenet_mean.to(dtype=x.dtype)
        std = self._imagenet_std.to(dtype=x.dtype)
        return (x - mean) / std

    def forward(self, x):
        x = self._normalize(x)
        feat = self.features(x)
        feat = self.pool(feat)
        emb = self.embedding(feat)
        identity_logits = self.identity_head(emb)
        gender_logits = self.gender_head(emb)
        return identity_logits, gender_logits, emb

    def embed(self, x):
        """Chỉ trích embedding — dùng làm 'giám khảo' cho identity-aware loss khi train SR."""
        x = self._normalize(x)
        feat = self.features(x)
        feat = self.pool(feat)
        return self.embedding(feat)
