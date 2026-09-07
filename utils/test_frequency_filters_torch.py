"""
Test cho utils/frequency_filters_torch.py trước khi dùng làm loss training
thật. Chạy: python -m utils.test_frequency_filters_torch (từ thư mục gốc
project) hoặc python utils/test_frequency_filters_torch.py.

KHÔNG dùng pytest (project này không có sẵn pytest trong requirements.txt,
theo đúng quy ước các file test_*.py khác trong utils/ nếu có) -- assert
trần + traceback rõ ràng khi fail, in "TẤT CẢ TEST PASS" khi xong.
"""
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")
from utils.frequency_filters import apply_frequency_filter
from utils.frequency_filters_torch import lowpass_filter, _lowpass_mask


def test_cutoff_1_is_noop():
    """cutoff_frac=1.0 phải là no-op tuyệt đối -- đúng bug đã phát hiện và
    sửa ở bản numpy (_max_radius, xem docstring frequency_filters.py)."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 20, 20)  # kích thước thật của ảnh tai trong project
    out = lowpass_filter(x, cutoff_frac=1.0)
    max_diff = (out - x).abs().max().item()
    assert max_diff < 1e-3, f"cutoff=1.0 không phải no-op, max_diff={max_diff}"
    print(f"  OK: cutoff=1.0 no-op, max_diff={max_diff:.2e}")


def test_cutoff_0_is_dc_only():
    """cutoff_frac=0.0 chỉ giữ thành phần DC -- ảnh ra phải gần như hằng số
    (mọi pixel bằng nhau, = giá trị trung bình ảnh gốc)."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 20, 20)
    out = lowpass_filter(x, cutoff_frac=0.0)
    # phương sai theo từng ảnh/kênh phải xấp xỉ 0 (hằng số)
    std_per_channel = out.reshape(2, 3, -1).std(dim=-1)
    assert std_per_channel.max().item() < 1e-3, \
        f"cutoff=0.0 không phải hằng số, std max={std_per_channel.max().item()}"
    # giá trị hằng số đó phải khớp trung bình ảnh gốc
    mean_orig = x.reshape(2, 3, -1).mean(dim=-1)
    mean_out = out.reshape(2, 3, -1).mean(dim=-1)
    max_mean_diff = (mean_orig - mean_out).abs().max().item()
    assert max_mean_diff < 1e-3, f"DC-only không khớp mean gốc, diff={max_mean_diff}"
    print(f"  OK: cutoff=0.0 la hang so (std={std_per_channel.max().item():.2e}), "
          f"khop mean goc (diff={max_mean_diff:.2e})")


def test_matches_numpy_reference():
    """So sánh trực tiếp với bản numpy/PIL đã dùng cho frequency-ablation
    (utils/frequency_filters.py, đã verify + dùng thật trong bài báo) --
    đây là test QUAN TRỌNG NHẤT: nếu convention radius/r_max lệch nhau,
    ý nghĩa "cutoff_frac=0.1" giữa loss training mới và thí nghiệm
    frequency-ablation cũ sẽ KHÔNG nhất quán, phá vỡ toàn bộ lập luận
    "loss mới bắt nguồn trực tiếp từ phát hiện đã có"."""
    rng = np.random.RandomState(42)
    for h, w in [(20, 20), (21, 21), (20, 24)]:  # vuông chẵn, vuông lẻ, chữ nhật
        arr = rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        for cutoff in [0.1, 0.3, 0.5, 0.9]:
            ref = np.asarray(apply_frequency_filter(img, "lowpass", cutoff), dtype=np.float64)
            x = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)  # (1,3,h,w)
            out = lowpass_filter(x, cutoff).squeeze(0).permute(1, 2, 0).numpy()
            out_clipped = np.clip(out, 0, 255)
            # numpy reference làm tròn về uint8 (mất <=0.5 mỗi pixel do làm
            # tròn) -- cho phép sai số nhỏ, không so bằng tuyệt đối
            max_diff = np.abs(ref - out_clipped).max()
            assert max_diff < 1.0, \
                f"lệch numpy reference tại h={h},w={w},cutoff={cutoff}: max_diff={max_diff}"
    print("  OK: khop ban numpy reference (frequency_filters.py) tren moi kich thuoc/cutoff test")


def test_differentiable():
    """Gradient phải chảy được qua lowpass_filter -- đây là lý do DUY NHẤT
    module này tồn tại thay vì dùng thẳng bản numpy."""
    x = torch.randn(2, 3, 20, 20, requires_grad=True)
    out = lowpass_filter(x, cutoff_frac=0.1)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "gradient không chảy qua lowpass_filter"
    assert torch.isfinite(x.grad).all(), "gradient chứa NaN/Inf"
    assert x.grad.abs().sum().item() > 0, "gradient toàn số 0 (đáng ngờ)"
    print(f"  OK: gradient chay duoc, grad abs sum={x.grad.abs().sum().item():.4f}")


def test_batch_and_channel_shapes_preserved():
    for b, c, h, w in [(1, 1, 20, 20), (4, 3, 20, 20), (2, 3, 32, 20)]:
        x = torch.randn(b, c, h, w)
        out = lowpass_filter(x, cutoff_frac=0.2)
        assert out.shape == x.shape, f"shape thay doi: vao {x.shape}, ra {out.shape}"
    print("  OK: giu nguyen shape cho moi (B,C,H,W) thu")


def test_dtype_preserved():
    x = torch.randn(1, 3, 20, 20, dtype=torch.float64)
    out = lowpass_filter(x, cutoff_frac=0.2)
    assert out.dtype == torch.float64, f"dtype doi tu float64 sang {out.dtype}"
    print("  OK: giu nguyen dtype dau vao (test voi float64)")


def test_invalid_cutoff_raises():
    x = torch.randn(1, 3, 20, 20)
    for bad in [-0.1, 1.1, 2.0]:
        try:
            lowpass_filter(x, cutoff_frac=bad)
            raise AssertionError(f"cutoff={bad} phai raise ValueError nhung khong")
        except ValueError:
            pass
    print("  OK: cutoff_frac ngoai [0,1] raise ValueError dung nhu ky vong")


def test_invalid_shape_raises():
    for bad_shape in [(3, 20, 20), (20, 20)]:
        x = torch.randn(*bad_shape)
        try:
            lowpass_filter(x, cutoff_frac=0.1)
            raise AssertionError(f"shape {bad_shape} phai raise ValueError nhung khong")
        except ValueError:
            pass
    print("  OK: shape khong phai (B,C,H,W) raise ValueError dung nhu ky vong")


def test_mask_no_nan_and_correct_dc_center():
    """Mask ở cutoff nhỏ nhất > 0 phải giữ ít nhất điểm DC (tâm phổ) --
    nếu không, mọi cutoff nhỏ sẽ triệt tiêu toàn bộ tín hiệu về 0, vô nghĩa."""
    mask = _lowpass_mask(20, 20, cutoff_frac=0.01, device=torch.device("cpu"))
    assert not torch.isnan(mask).any(), "mask chứa NaN"
    cy, cx = 10, 10  # tâm phổ sau fftshift, h=w=20
    assert mask[cy, cx].item() == 1.0, "diem DC (tam pho) phai luon duoc giu du cutoff nho"
    print("  OK: mask khong NaN, diem DC luon duoc giu")


def test_zero_input_gives_zero_output():
    """Edge case số học: ảnh toàn số 0 phải cho ra toàn số 0 (không NaN do
    chia 0 ở đâu đó trong FFT/mask)."""
    x = torch.zeros(1, 3, 20, 20)
    out = lowpass_filter(x, cutoff_frac=0.3)
    assert torch.isfinite(out).all(), "input toan 0 nhung output chua NaN/Inf"
    assert out.abs().max().item() < 1e-5, "input toan 0 nhung output khac 0"
    print("  OK: input toan 0 -> output toan 0, khong NaN")


def test_constant_input_unchanged_by_lowpass():
    """Ảnh hằng số (chỉ có thành phần DC) phải không đổi qua BẤT KỲ cutoff
    lowpass nào > 0 (vì toàn bộ năng lượng đã nằm ở DC, trong mọi mask)."""
    x = torch.full((1, 3, 20, 20), 5.0)
    for cutoff in [0.05, 0.3, 0.7, 1.0]:
        out = lowpass_filter(x, cutoff_frac=cutoff)
        max_diff = (out - x).abs().max().item()
        assert max_diff < 1e-3, f"anh hang so bi doi o cutoff={cutoff}, diff={max_diff}"
    print("  OK: anh hang so khong doi qua moi cutoff lowpass")


if __name__ == "__main__":
    tests = [
        test_cutoff_1_is_noop,
        test_cutoff_0_is_dc_only,
        test_matches_numpy_reference,
        test_differentiable,
        test_batch_and_channel_shapes_preserved,
        test_dtype_preserved,
        test_invalid_cutoff_raises,
        test_invalid_shape_raises,
        test_mask_no_nan_and_correct_dc_center,
        test_zero_input_gives_zero_output,
        test_constant_input_unchanged_by_lowpass,
    ]
    for t in tests:
        print(f"{t.__name__}...")
        t()
    print("\nTAT CA TEST PASS")
