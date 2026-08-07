"""
Wrapper để dùng ĐÚNG kiến trúc SPAN chính thức (không tự viết lại), đảm bảo
tương thích 100% với checkpoint pretrained tải từ hongyuanyu/SPAN.

ĐÃ ĐỐI CHIẾU VỚI SOURCE THẬT (external/SPAN/basicsr/archs/span_arch.py, người
dùng cung cấp trực tiếp) — constructor khớp 100% với suy đoán ban đầu:
    SPAN(num_in_ch, num_out_ch, feature_channels=48, upscale=4, bias=True,
         img_range=255., rgb_mean=(0.4488, 0.4371, 0.4040))

PHÁT HIỆN QUAN TRỌNG cần vá: forward() của model gốc biến đổi input theo
    x = (x - mean) * img_range
nhưng KHÔNG biến đổi ngược lại ở output — output trả về nằm ở thang giá trị
đã nhân img_range, KHÔNG phải [0,1] như ảnh HR ground truth (ToTensor()).
Nếu dùng thẳng output này để tính loss/PSNR/SSIM so với ảnh [0,1], kết quả sẽ
sai hoàn toàn (loss cực lớn ngay từ đầu) mà KHÔNG có lỗi runtime nào báo hiệu.

Class SPANWithRescale bên dưới bọc model gốc, tự động thực hiện phép biến đổi
ngược (output / img_range + mean) và clamp về [0,1] — để tương thích với toàn
bộ pipeline train/eval hiện tại (vốn giả định output SR luôn ở [0,1]).

Cách dùng sau khi đã setup xong:
    from models.span_official_wrapper import build_official_span
    model = build_official_span(scale=4, pretrained_path="checkpoints/span_pretrained_x4.pth")
"""
import sys
import types
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

_EXTERNAL_SPAN_PATH = Path(__file__).resolve().parents[1] / "external" / "SPAN"


def _stub_basicsr_registry():
    """
    span_arch.py chỉ cần `from basicsr.utils.registry import ARCH_REGISTRY`
    làm decorator (@ARCH_REGISTRY.register()) — không cần logic thật.
    Đăng ký sẵn module giả trong sys.modules để tránh phải chạy
    basicsr/__init__.py thật (vốn cascade import cả pipeline data augmentation
    không liên quan, và bị lỗi do torchvision mới đã xóa
    torchvision.transforms.functional_tensor mà basicsr/data/degradations.py
    còn dùng — lỗi tương thích phổ biến của BasicSR cũ, không liên quan gì
    đến kiến trúc SPAN mà ta thực sự cần).
    """
    if "basicsr" not in sys.modules:
        sys.modules["basicsr"] = types.ModuleType("basicsr")
    if "basicsr.utils" not in sys.modules:
        sys.modules["basicsr.utils"] = types.ModuleType("basicsr.utils")
    if "basicsr.utils.registry" not in sys.modules:
        registry_module = types.ModuleType("basicsr.utils.registry")

        class _DummyRegistry:
            def register(self, cls=None, **kwargs):
                if cls is not None:
                    return cls

                def decorator(c):
                    return c
                return decorator

        registry_module.ARCH_REGISTRY = _DummyRegistry()
        sys.modules["basicsr.utils.registry"] = registry_module


def _import_official_span_class():
    if not _EXTERNAL_SPAN_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {_EXTERNAL_SPAN_PATH}. Chạy trước: "
            "bash scripts/setup_span_official.sh"
        )

    arch_file = _EXTERNAL_SPAN_PATH / "basicsr" / "archs" / "span_arch.py"
    if not arch_file.exists():
        raise FileNotFoundError(f"Không tìm thấy {arch_file}")

    _stub_basicsr_registry()

    # Load trực tiếp file span_arch.py, KHÔNG đi qua basicsr/__init__.py hay
    # basicsr/archs/__init__.py thật (cả 2 đều có thể cascade import các file
    # khác không liên quan và bị lỗi, như đã giải thích ở trên).
    spec = importlib.util.spec_from_file_location("span_arch_official", arch_file)
    span_arch_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(span_arch_module)

    return span_arch_module.SPAN


class SPANWithRescale(nn.Module):
    """Bọc SPAN chính thức, tự động vá lỗi thiếu biến đổi ngược output -> [0,1]."""

    def __init__(self, official_model: nn.Module, img_range: float, rgb_mean):
        super().__init__()
        self.model = official_model
        self.img_range = img_range
        self.register_buffer("mean", torch.tensor(rgb_mean).view(1, 3, 1, 1))

    def forward(self, x):
        raw_out = self.model(x)
        mean = self.mean.type_as(raw_out)
        # Phép biến đổi NGƯỢC lại đúng với phép biến đổi ở đầu forward() gốc:
        # gốc làm: x = (x - mean) * img_range  =>  ngược lại: out/img_range + mean
        out = raw_out / self.img_range + mean
        return torch.clamp(out, 0.0, 1.0)

    def load_state_dict(self, state_dict, strict=True):
        # checkpoint gốc lưu trọng số của model bên trong (không có tiền tố "model."),
        # nên cần nạp vào self.model, không phải nạp thẳng vào self (SPANWithRescale)
        return self.model.load_state_dict(state_dict, strict=strict)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)


def build_official_span(scale: int = 4, pretrained_path: str = None,
                         num_in_ch: int = 3, num_out_ch: int = 3,
                         feature_channels: int = 48,
                         img_range: float = 255.,
                         rgb_mean=(0.4488, 0.4371, 0.4040)):
    """Khởi tạo SPAN chính thức, bọc sẵn lớp biến đổi ngược output."""
    OfficialSPAN = _import_official_span_class()

    official_model = OfficialSPAN(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        feature_channels=feature_channels,
        upscale=scale,
        img_range=img_range,
        rgb_mean=rgb_mean,
    )

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location="cpu")
        if isinstance(state_dict, dict) and "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif isinstance(state_dict, dict) and "params" in state_dict:
            state_dict = state_dict["params"]
        official_model.load_state_dict(state_dict, strict=True)
        print(f"Đã load checkpoint pretrained chính thức: {pretrained_path}")

    return SPANWithRescale(official_model, img_range=img_range, rgb_mean=rgb_mean)
