"""
[MỚI — đợt 7] Smoke test nhanh cho 3 kiến trúc SR mới (rlfn/ecbsr/safmn) +
rà soát lại toàn bộ kiến trúc SR đã có — kiểm tra forward pass chạy được,
shape output đúng, và đếm tham số — TRƯỚC khi chạy train thật (tốn thời
gian). Không cần dataset thật, dùng tensor ngẫu nhiên.

Lưu ý: các kiến trúc mới đã được đối chiếu kỹ với bài báo/mã nguồn gốc bằng
cách đọc trực tiếp (xem chú thích trong models/sr_models.py + RUNBOOK_EarVN1.0.md)
nhưng KHÔNG chạy thực nghiệm được trong môi trường soạn thảo (không có
torch) — script này để BẠN tự xác nhận lại trước khi tin tưởng chạy full
pipeline.

Chạy:
    python test_new_sr_archs.py
"""
import torch

from models.sr_models import build_sr_model
from utils.metrics import count_params, count_params_deploy_mode


def check_arch(arch: str, scale: int = 4, lr_size: int = 20):
    model = build_sr_model(arch, scale)
    model.eval()
    dummy = torch.randn(2, 3, lr_size, lr_size)
    with torch.no_grad():
        out = model(dummy)
    expected_shape = (2, 3, lr_size * scale, lr_size * scale)
    ok_shape = tuple(out.shape) == expected_shape
    ok_range = bool((out.min() >= 0.0) and (out.max() <= 1.0))
    params = count_params(model)
    params_deploy = count_params_deploy_mode(model)

    # Kiểm tra thêm ở train mode (một số kiến trúc — vd ecbsr — có forward
    # KHÁC nhau giữa train/eval do structural reparameterization)
    model.train()
    with torch.no_grad():
        out_train = model(dummy)
    ok_train_shape = tuple(out_train.shape) == expected_shape

    status = "OK" if (ok_shape and ok_range and ok_train_shape) else "LỖI"
    print(f"[{status}] arch={arch:<10} out_shape={tuple(out.shape)} "
          f"(mong đợi {expected_shape}) | range=[{out.min():.3f},{out.max():.3f}] | "
          f"params={params:.4f}M | params_deploy={params_deploy:.4f}M | "
          f"train_mode_shape_ok={ok_train_shape}")
    return status == "OK"


if __name__ == "__main__":
    archs = ["span", "span_tiny", "span_large", "edsr", "rlfn", "rlfn_adapted", "ecbsr", "safmn", "smfanet"]
    print("== Smoke test kiến trúc SR (tensor ngẫu nhiên, không cần dataset) ==\n")
    results = {a: check_arch(a) for a in archs}
    print("\n== Tổng kết ==")
    n_fail = sum(1 for ok in results.values() if not ok)
    if n_fail == 0:
        print("Tất cả kiến trúc OK — an toàn để chạy pipeline thật.")
    else:
        print(f"{n_fail} kiến trúc LỖI — xem chi tiết ở trên, SỬA TRƯỚC khi chạy pipeline thật.")
