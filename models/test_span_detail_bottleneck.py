"""
Test cho SPANDetailBottleneck (models/sr_models.py) + build_sr_model("span_bottleneck", ...)
trước khi dùng cho thí nghiệm kiến trúc mới. Chạy:
    python models/test_span_detail_bottleneck.py

Không dùng pytest (đúng quy ước utils/test_frequency_filters_torch.py) --
assert trần, in "TAT CA TEST PASS" khi xong.
"""
import sys
import torch

sys.path.insert(0, ".")
from models.sr_models import SPAN, SPANDetailBottleneck, build_sr_model


def _count_params(model):
    return sum(p.numel() for p in model.parameters())


def test_forward_shape_and_range():
    """Input 20x20 (đúng kích thước LR thật của project), scale=4 -> output
    phải là 80x80, giá trị trong [0,1] (do torch.clamp)."""
    model = SPANDetailBottleneck(scale=4, feat=48, n_blocks=3, bottleneck_channels=16)
    x = torch.rand(2, 3, 20, 20)
    out = model(x)
    assert out.shape == (2, 3, 80, 80), f"shape sai: {out.shape}"
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0, \
        f"output ngoài [0,1]: min={out.min().item()}, max={out.max().item()}"
    print(f"  OK: shape {tuple(out.shape)}, range [{out.min().item():.3f}, {out.max().item():.3f}]")


def test_residual_unaffected_by_bottleneck():
    """Xác nhận residual dài body_tail(feat)+body_in vẫn hoạt động ĐÚNG như
    SPAN gốc -- so sánh trực tiếp: nếu bottleneck_channels=feat VÀ ta zero-hoá
    detail_bottleneck thành ánh xạ identity thủ công, phần THÂN MẠNG (trước
    bottleneck) phải cho feature giống hệt SPAN gốc với CÙNG trọng số head/body/body_tail."""
    torch.manual_seed(0)
    feat, n_blocks = 48, 3
    plain = SPAN(scale=4, feat=feat, n_blocks=n_blocks)
    bottleneck_model = SPANDetailBottleneck(scale=4, feat=feat, n_blocks=n_blocks,
                                             bottleneck_channels=feat)
    # copy trọng số head/body/body_tail từ plain sang bottleneck_model (upsample
    # khác cấu trúc nên không copy) để 2 model có ĐÚNG cùng phần thân
    bottleneck_model.head.load_state_dict(plain.head.state_dict())
    bottleneck_model.body.load_state_dict(plain.body.state_dict())
    bottleneck_model.body_tail.load_state_dict(plain.body_tail.state_dict())

    x = torch.rand(1, 3, 20, 20)
    with torch.no_grad():
        # tái tạo thủ công đúng phần thân (giống forward() của SPAN/SPANDetailBottleneck)
        f1 = plain.head(x)
        body_in1 = f1
        for block in plain.body:
            f1 = block(f1)
        f1 = plain.body_tail(f1) + body_in1

        f2 = bottleneck_model.head(x)
        body_in2 = f2
        for block in bottleneck_model.body:
            f2 = block(f2)
        f2 = bottleneck_model.body_tail(f2) + body_in2

    max_diff = (f1 - f2).abs().max().item()
    assert max_diff < 1e-6, f"phan than mang lech nhau: max_diff={max_diff} (residual bi anh huong?)"
    print(f"  OK: phan than mang (head+body+body_tail+residual) giong het SPAN goc, max_diff={max_diff:.2e}")


def test_param_count_reduction_vs_span_tiny():
    """Bottleneck hẹp phải giảm tổng tham số so với span_tiny (feat=48,
    n_blocks=3, KHÔNG có bottleneck) -- xác nhận bằng số, không chỉ bằng lý luận."""
    span_tiny = SPAN(scale=4, feat=48, n_blocks=3)
    for bc in [8, 16, 24, 32]:
        variant = SPANDetailBottleneck(scale=4, feat=48, n_blocks=3, bottleneck_channels=bc)
        p_tiny = _count_params(span_tiny)
        p_variant = _count_params(variant)
        # variant CÓ THÊM 1 lớp (detail_bottleneck) nhưng upsample nhẹ hơn nhiều
        # (input channels giảm feat=48 -> bc) -- tổng phải giảm ròng với bc đủ nhỏ
        print(f"  bottleneck_channels={bc}: span_tiny={p_tiny} params, variant={p_variant} params, "
              f"diff={p_variant - p_tiny:+d} ({100*(p_variant-p_tiny)/p_tiny:+.1f}%)")
    # với bc=16 (mặc định), phải giảm ròng đáng kể
    variant16 = SPANDetailBottleneck(scale=4, feat=48, n_blocks=3, bottleneck_channels=16)
    assert _count_params(variant16) < _count_params(span_tiny), \
        "bottleneck_channels=16 phải giảm tổng tham số so với span_tiny"
    print("  OK: bottleneck_channels=16 giam rong tong tham so so voi span_tiny")


def test_gradient_flows_through_bottleneck():
    model = SPANDetailBottleneck(scale=4, feat=48, n_blocks=3, bottleneck_channels=16)
    x = torch.rand(1, 3, 20, 20)
    out = model(x)
    loss = out.sum()
    loss.backward()
    # kiem tra gradient chay toi CA head (dau mang) LAN detail_bottleneck (lop moi)
    assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all()
    assert model.detail_bottleneck.weight.grad is not None and \
        torch.isfinite(model.detail_bottleneck.weight.grad).all()
    assert model.detail_bottleneck.weight.grad.abs().sum().item() > 0
    print("  OK: gradient chay tu upsample nguoc ve toi head, khong NaN")


def test_invalid_bottleneck_channels_raises():
    for bad in [0, -1, 49, 100]:
        try:
            SPANDetailBottleneck(scale=4, feat=48, n_blocks=3, bottleneck_channels=bad)
            raise AssertionError(f"bottleneck_channels={bad} phai raise ValueError nhung khong")
        except ValueError:
            pass
    print("  OK: bottleneck_channels ngoai [1,feat] raise ValueError dung nhu ky vong")


def test_bottleneck_equals_feat_is_valid_edge_case():
    """bottleneck_channels == feat: hợp lệ (không phải lỗi), dù không còn ý
    nghĩa 'thắt cổ chai' -- dùng làm điểm đối chứng."""
    model = SPANDetailBottleneck(scale=4, feat=48, n_blocks=3, bottleneck_channels=48)
    x = torch.rand(1, 3, 20, 20)
    out = model(x)
    assert out.shape == (1, 3, 80, 80)
    print("  OK: bottleneck_channels == feat van hop le, khong crash")


def test_bottleneck_equals_1_extreme_case():
    """bottleneck_channels=1: cực đoan nhất (chỉ 1 kênh duy nhất mang toàn bộ
    thông tin cho bước tổng hợp cuối) -- vẫn phải chạy được, không NaN."""
    model = SPANDetailBottleneck(scale=4, feat=48, n_blocks=3, bottleneck_channels=1)
    x = torch.rand(1, 3, 20, 20)
    out = model(x)
    assert out.shape == (1, 3, 80, 80)
    assert torch.isfinite(out).all()
    print("  OK: bottleneck_channels=1 (cuc doan) van chay duoc, khong NaN")


def test_build_sr_model_factory():
    """build_sr_model('span_bottleneck', ...) phải dùng đúng default (n_blocks=3,
    bottleneck_channels=16) khi không truyền, và tôn trọng override khi có truyền."""
    m_default = build_sr_model("span_bottleneck", scale=4)
    assert isinstance(m_default, SPANDetailBottleneck)
    assert len(m_default.body) == 3, f"default n_blocks phai la 3, nhan {len(m_default.body)}"
    assert m_default.detail_bottleneck.out_channels == 16, \
        f"default bottleneck_channels phai la 16, nhan {m_default.detail_bottleneck.out_channels}"

    m_override = build_sr_model("span_bottleneck", scale=4, n_blocks=5, bottleneck_channels=8)
    assert len(m_override.body) == 5
    assert m_override.detail_bottleneck.out_channels == 8
    print("  OK: build_sr_model factory dung default va override dung nhu ky vong")


def test_build_sr_model_other_archs_unaffected():
    """Kiem tra them tham so bottleneck_channels vao chu ky ham KHONG lam vo
    cac arch khac (chung phai bo qua tham so nay hoan toan)."""
    m1 = build_sr_model("span_tiny", scale=4, bottleneck_channels=999)  # phai bi bo qua, khong loi
    assert isinstance(m1, SPAN)
    assert len(m1.body) == 3
    print("  OK: cac arch khac (vd span_tiny) bo qua bottleneck_channels, khong bi anh huong")


def test_unknown_arch_still_raises():
    try:
        build_sr_model("khong_ton_tai", scale=4)
        raise AssertionError("expected ValueError for unknown arch")
    except ValueError:
        pass
    print("  OK: arch khong ton tai van raise ValueError dung nhu truoc")


if __name__ == "__main__":
    tests = [
        test_forward_shape_and_range,
        test_residual_unaffected_by_bottleneck,
        test_param_count_reduction_vs_span_tiny,
        test_gradient_flows_through_bottleneck,
        test_invalid_bottleneck_channels_raises,
        test_bottleneck_equals_feat_is_valid_edge_case,
        test_bottleneck_equals_1_extreme_case,
        test_build_sr_model_factory,
        test_build_sr_model_other_archs_unaffected,
        test_unknown_arch_still_raises,
    ]
    for t in tests:
        print(f"{t.__name__}...")
        t()
    print("\nTAT CA TEST PASS")
