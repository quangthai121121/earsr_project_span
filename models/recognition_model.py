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
