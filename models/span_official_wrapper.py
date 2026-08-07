"""
Wrapper để dùng ĐÚNG kiến trúc SPAN chính thức (không tự viết lại), đảm bảo
tương thích 100% với checkpoint pretrained tải từ hongyuanyu/SPAN.

QUAN TRỌNG — cần bạn xác nhận trước khi dùng cho kết quả chính thức:
Mình KHÔNG fetch được nội dung file `span_arch.py` thật (GitHub chặn crawl
sâu vào repo qua robots.txt trong môi trường của mình), nên đoạn import bên
dưới viết theo cấu trúc BasicSR phổ biến (class đăng ký qua ARCH_REGISTRY,
tham số kiểu num_in_ch/num_feat/upscale) — đây là suy luận hợp lý dựa trên
quy ước chung của các arch trong BasicSR, KHÔNG phải chép nguyên văn từ file
gốc. Sau khi chạy `scripts/setup_span_official.sh`, hãy tự mở:
    external/SPAN/basicsr/archs/span_arch.py
đối chiếu tên class và tên tham số constructor với đoạn `try/except` bên
dưới, sửa lại cho khớp nếu cần trước khi tin tưởng dùng cho kết quả cuối.

Cách dùng sau khi đã setup xong:
    from models.span_official_wrapper import build_official_span
    model = build_official_span(scale=4, pretrained_path="checkpoints/span_pretrained_x4.pth")
"""
import sys
from pathlib import Path

import torch

_EXTERNAL_SPAN_PATH = Path(__file__).resolve().parents[1] / "external" / "SPAN"


def _import_official_span_class():
    if not _EXTERNAL_SPAN_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {_EXTERNAL_SPAN_PATH}. Chạy trước: "
            "bash scripts/setup_span_official.sh"
        )

    sys.path.insert(0, str(_EXTERNAL_SPAN_PATH))

    # Cách import phổ biến trong BasicSR: module basicsr.archs.span_arch,
    # class thường tên là SPAN. NẾU cấu trúc thật khác, sửa dòng import này.
    try:
        from basicsr.archs.span_arch import SPAN as OfficialSPAN  # noqa: E402
        return OfficialSPAN
    except ImportError as e:
        raise ImportError(
            "Không import được class SPAN từ external/SPAN/basicsr/archs/span_arch.py. "
            "Hãy mở file này, kiểm tra tên module/class thật, rồi sửa lại hàm "
            "_import_official_span_class() trong models/span_official_wrapper.py "
            f"cho khớp. Lỗi gốc: {e}"
        )


def build_official_span(scale: int = 4, pretrained_path: str = None,
                         num_in_ch: int = 3, num_out_ch: int = 3,
                         feature_channels: int = 48):
    """
    Khởi tạo SPAN chính thức. Các tham số num_in_ch/num_out_ch/feature_channels
    là GIÁ TRỊ SUY ĐOÁN theo quy ước phổ biến của BasicSR — đối chiếu lại với
    constructor thật trong span_arch.py (xem docstring module này) và sửa
    lại lời gọi bên dưới nếu tên/số lượng tham số khác.
    """
    OfficialSPAN = _import_official_span_class()

    try:
        model = OfficialSPAN(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            feature_channels=feature_channels,
            upscale=scale,
        )
    except TypeError as e:
        raise TypeError(
            "Constructor của SPAN chính thức không khớp tham số đã đoán "
            f"(num_in_ch/num_out_ch/feature_channels/upscale). Lỗi: {e}\n"
            "Mở external/SPAN/basicsr/archs/span_arch.py để xem đúng tên "
            "tham số, rồi sửa lại hàm build_official_span()."
        )

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location="cpu")
        # Một số checkpoint BasicSR bọc trong key 'params' hoặc 'params_ema'
        if isinstance(state_dict, dict) and "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif isinstance(state_dict, dict) and "params" in state_dict:
            state_dict = state_dict["params"]
        model.load_state_dict(state_dict, strict=True)
        print(f"Đã load checkpoint pretrained chính thức: {pretrained_path}")

    return model
