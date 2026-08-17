"""
[MỚI — bổ sung journal Q1, đợt 8, SỬA đợt 10 sau code review] Chẩn đoán lỗi
chất lượng bất thường của SMFANet (PSNR ROI đo được ~20-25dB, thấp hẳn so
với MỌI kiến trúc SR khác trong project ~26.25-27.5dB — xem results/sr_quality.csv).

KHÔNG tự ý "sửa" hay huấn luyện lại gì cả — đây CHỈ LÀ SCRIPT CHẨN ĐOÁN, in
ra bằng chứng cụ thể để xác định NGUYÊN NHÂN THẬT trước khi quyết định: (a)
sửa hyperparameter rồi train lại, (b) chấp nhận số liệu và giải thích trong
bài, hoặc (c) loại SMFANet khỏi bảng so sánh chính vì không đủ tin cậy.
KHÔNG được đưa số liệu SMFANet vào bài nếu chưa chạy qua chẩn đoán này.

[SỬA đợt 10 — 2 lỗi phát hiện qua code review]
  1. BƯỚC 1 (kiểm tra trọng số) trước đây in header hứa "feats.0..feats.7 =
     8 khối" nhưng code dùng model.named_children() — CHỈ liệt kê các con
     TRỰC TIẾP (to_feat/feats/to_img), module.parameters() bên trong lại
     GỘP CHUNG tất cả tham số của cả 8 khối thành 1 số trung bình duy nhất
     — không có breakdown 8 khối như header hứa. ĐÃ SỬA: thêm
     _find_block_container() tự động tìm đúng container chứa các khối lặp
     lại (Sequential/ModuleList NHIỀU PHẦN TỬ NHẤT trong các con trực tiếp),
     in breakdown TỪNG KHỐI thật cho cả BƯỚC 1 (trọng số) lẫn BƯỚC 2/3
     (activation, đã đúng từ trước).
  2. probe_forward() trước đây CHỈ nhận diện kiến trúc kiểu to_feat/feats/to_img
     (SMFANet, SAFMN) để hook theo khối + lấy output trước-clamp — kiến trúc
     RLFN/RLFN_adapted (dùng head/body/body_tail/upsample, --compare_arch mặc
     định gợi ý trong docstring cũ) bị BỎ SÓT hoàn toàn, không có breakdown
     theo khối, không có số liệu trước-clamp thật. ĐÃ SỬA: _forward_pre_clamp()
     hỗ trợ tường minh CẢ 3 khuôn mẫu kiến trúc thật đã xác nhận qua đọc trực
     tiếp models/sr_models.py (to_feat/feats/to_img | head/body/body_tail/
     upsample | backbone/upsampler) — rlfn_adapted GIỜ có đầy đủ breakdown
     như smfanet/safmn, giữ nguyên default --compare_arch=rlfn_adapted (kiến
     trúc THẬT SỰ dùng làm baseline chính trong bảng so sánh của bài báo,
     xem models/sr_models.py::build_sr_model() docstring) thay vì đổi sang
     kiến trúc khác chỉ vì công cụ chẩn đoán từng có giới hạn.

Chạy (ví dụ, sau khi đã train SMFANet ở Track A hoặc B):
    python debug_smfanet.py --config configs/config.yaml \
        --ckpt runs/sr_smfanet/best.pt \
        --compare_ckpt runs/sr_rlfn_adapted/best.pt --compare_arch rlfn_adapted \
        --out_dir results/debug_smfanet

Các bước kiểm tra (in ra console + lưu results/debug_smfanet/report.txt):
  0. Quét log train (train.log trong thư mục runs/ tương ứng) tìm cảnh báo
     "loss NaN/Inf" đã bị GradScaler âm thầm bỏ qua (xem train_sr.py/
     train_sr_distill.py::run_epoch()) — nếu tỷ lệ batch bị bỏ cao, mô hình
     hiệu quả được train với ÍT bước cập nhật thật hơn số epoch báo cáo.
  1. Kiểm tra checkpoint: NaN/Inf trong trọng số; độ lớn |weight| trung bình
     THEO TỪNG KHỐI thật (không còn gộp chung — xem sửa lỗi #1 ở trên).
  2. Forward thật trên ảnh LR test thật (không phải input ngẫu nhiên):
     - Thống kê output TRƯỚC clamp (min/max/mean/std) — nếu vượt xa [0,1]
       nhiều, mô hình đang "đoán bừa" biên độ, bị clamp cắt cụt sau đó.
     - Tỷ lệ pixel bị bão hoà (== 0.0 hoặc == 1.0 sau clamp) — nếu cao bất
       thường, ảnh SR bị "cháy sáng/tối" nhiều vùng, giải thích PSNR thấp.
     - Độ lớn trung bình đặc trưng SAU TỪNG khối (qua forward hook, tự động
       tìm đúng container — xem sửa lỗi #2) — xem đặc trưng có nổ/triệt tiêu
       dần qua các khối không (nghi vấn hàng đầu cho SMFANet: F.normalize
       không có affine/scale học được, lặp 2 lần/khối x 8 khối = 16 lần
       chuẩn hoá "cứng", có thể phá hỏng thông tin biên độ nếu đặc trưng đầu
       vào yếu — xem models/sr_models.py::_SMFAFMB).
     - PSNR/SSIM ROI tính LẠI tại chỗ (dùng đúng hàm compute_psnr_roi/
       compute_ssim_roi như eval_sr_quality.py) — đối chiếu xem số liệu thấp
       có phải lỗi ĐO (khác) hay đúng là mô hình sinh ảnh kém thật.
     - Lưu vài ảnh LR/SR/HR cạnh nhau để xem bằng mắt.
  3. (Nếu --compare_ckpt được truyền) Chạy lại toàn bộ bước 2 cho 1 kiến
     trúc THAM CHIẾU đang hoạt động bình thường (mặc định rlfn_adapted —
     baseline chính của bài báo, GIỜ có đầy đủ breakdown theo khối) trên
     CÙNG ảnh LR — so sánh trực tiếp quỹ đạo độ lớn đặc trưng/tỷ lệ bão hoà,
     để biết SMFANet có thật sự BẤT THƯỜNG so với kiến trúc khác hay không.
"""
import argparse
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets.hrlr_pair_dataset import HRLRPairDataset
from models.sr_models import build_sr_model
from utils.metrics import compute_psnr_roi, compute_ssim_roi, count_params


def scan_train_log(ckpt_path: str, report_lines: list):
    """Bước 0 — quét train.log cùng thư mục với ckpt tìm cảnh báo NaN/Inf bị bỏ qua."""
    log_path = Path(ckpt_path).parent / "train.log"
    header = f"\n--- BƯỚC 0: quét log train ({log_path}) ---"
    print(header); report_lines.append(header)
    if not log_path.exists():
        msg = f"  (không tìm thấy {log_path} — bỏ qua bước này, không phải lỗi)"
        print(msg); report_lines.append(msg)
        return
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    nan_matches = re.findall(r"(\d+) batch có loss NaN/Inf", text)
    total_nan_batches = sum(int(m) for m in nan_matches)
    msg = (f"  Số dòng cảnh báo NaN/Inf tìm thấy: {len(nan_matches)} lần "
           f"(tổng {total_nan_batches} batch bị bỏ qua cộng dồn qua các epoch).")
    print(msg); report_lines.append(msg)
    if total_nan_batches > 0:
        msg = ("  -> CÓ dấu hiệu bất ổn định số học khi train (GradScaler đã tự bỏ qua "
               "các batch này nên KHÔNG hỏng trọng số, nhưng mô hình hiệu quả được cập nhật "
               "ÍT hơn số batch danh nghĩa). Đây là NGHI VẤN HÀNG ĐẦU cho chất lượng thấp — "
               "cân nhắc giảm learning rate riêng cho smfanet hoặc thêm warmup vài epoch.")
        print(msg); report_lines.append(msg)
    else:
        msg = "  -> Không thấy cảnh báo NaN/Inf trong log (hoặc log không còn giữ đủ lịch sử)."
        print(msg); report_lines.append(msg)


def _find_block_container(model):
    """[MỚI — sửa lỗi #1] Tự động tìm thuộc tính con dạng Sequential/ModuleList
    CÓ NHIỀU PHẦN TỬ NHẤT trong các con TRỰC TIẾP của model — đây chính là
    "thân" chứa các khối lặp lại (residual blocks), phân biệt với các
    Sequential/ModuleList ngắn khác (ví dụ to_img/upsample chỉ có 2 lớp:
    conv + pixelshuffle, không phải "khối" theo nghĩa kiến trúc).

    Đã xác nhận đúng cho mọi kiến trúc dùng trong script này bằng cách đối
    chiếu trực tiếp models/sr_models.py (không suy đoán):
      - SMFANet/SAFMN: "feats" (Sequential, len=n_blocks=8) thắng "to_img" (len=2)
      - RLFN/RLFN_adapted: "body" (ModuleList, len=n_blocks=4) thắng "upsample" (len=2)
      - ECBSR: "backbone" (Sequential, len=num_block+2=6), không có Sequential/
        ModuleList nào khác cạnh tranh

    Trả về (tên_thuộc_tính, container) hoặc (None, None) nếu model không có
    Sequential/ModuleList nào ở cấp con trực tiếp."""
    best_name, best_container, best_len = None, None, 0
    for name, module in model.named_children():
        if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)) and len(module) > best_len:
            best_name, best_container, best_len = name, module, len(module)
    return best_name, best_container


def _mean_abs_weight(module) -> float:
    """Trung bình |weight| CÓ TRỌNG SỐ THEO SỐ PHẦN TỬ (không phải trung bình
    của các trung bình từng tensor — tránh thiên lệch khi 1 tensor rất lớn
    và nhiều tensor nhỏ khác bị đánh đồng trọng số ngang nhau)."""
    total, count = 0.0, 0
    for p in module.parameters():
        total += p.detach().abs().sum().item()
        count += p.numel()
    return total / count if count > 0 else float("nan")


def check_checkpoint_weights(model, report_lines: list):
    """Bước 1 — NaN/Inf + thống kê độ lớn trọng số, CẢ theo con trực tiếp
    LẪN theo TỪNG KHỐI thật bên trong container lặp lại (sửa lỗi #1)."""
    header = "\n--- BƯỚC 1: kiểm tra trọng số checkpoint ---"
    print(header); report_lines.append(header)
    any_bad = False
    for name, p in model.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            any_bad = True
            msg = f"  !!! NaN/Inf trong tham số: {name} (shape={tuple(p.shape)})"
            print(msg); report_lines.append(msg)
    if not any_bad:
        msg = "  Không có NaN/Inf trong bất kỳ tham số nào — checkpoint không bị hỏng."
        print(msg); report_lines.append(msg)

    msg = "  Độ lớn |weight| trung bình theo từng CON TRỰC TIẾP của model:"
    print(msg); report_lines.append(msg)
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        if n_params > 0:
            line = f"    {name:<12} mean|w|={_mean_abs_weight(module):.6f}  (n_params={n_params})"
            print(line); report_lines.append(line)

    block_name, block_container = _find_block_container(model)
    if block_container is not None and len(block_container) > 1:
        msg = (f"  Độ lớn |weight| trung bình THEO TỪNG KHỐI thật bên trong "
               f"'{block_name}' ({len(block_container)} khối):")
        print(msg); report_lines.append(msg)
        block_means = []
        for i, block in enumerate(block_container):
            m = _mean_abs_weight(block)
            block_means.append(m)
            line = f"    {block_name}[{i}]  mean|w|={m:.6f}"
            print(line); report_lines.append(line)
        if len(block_means) >= 2 and max(block_means) > 0 and (
                max(block_means) > min(block_means) * 10 or min(block_means) < max(block_means) * 0.1):
            msg = (f"  -> CẢNH BÁO: độ lớn trọng số chênh lệch >10x giữa các khối trong "
                   f"'{block_name}' — nghi vấn 1 khối cụ thể bị nổ/triệt tiêu trong khi các "
                   f"khối khác bình thường (số trung bình gộp ở BƯỚC 1 cũ sẽ CHE MẤT dấu hiệu này).")
            print(msg); report_lines.append(msg)
    else:
        msg = f"  (Không tìm thấy container nhiều khối để tách riêng — model chỉ có 1 khối duy nhất.)"
        print(msg); report_lines.append(msg)


def _forward_pre_clamp(model, lr_img):
    """[SỬA — lỗi #2] Forward THỦ CÔNG để lấy output TRƯỚC clamp (mọi model SR
    trong project đều clamp(0,1) ở bước cuối, xem models/sr_models.py). Hỗ
    trợ TƯỜNG MINH cả 3 khuôn mẫu kiến trúc đã xác nhận qua đọc trực tiếp
    models/sr_models.py (không suy đoán/genericize mù):
      - backbone/upsampler (+ shortcut repeat_interleave)     : ECBSR
      - head/body/body_tail/upsample (residual body_tail+f0)  : RLFN, RLFN_adapted
        (đã đối chiếu thêm: SPAN/span_tiny/span_large VÀ EDSR cũng dùng ĐÚNG
        khuôn mẫu này — head->loop body->body_tail(feat)+body_in->upsample —
        nên nhánh này cũng hoạt động đúng nếu --compare_arch là 1 trong các
        kiến trúc đó, dù không phải mục tiêu thiết kế ban đầu)
      - to_feat/feats/to_img (residual feats(feat)+feat)      : SMFANet, SAFMN
    Kiến trúc khác (span*, edsr, ...) -> fallback model(lr_img), trả về
    has_pre_clamp_access=False (ĐÃ bị clamp sẵn bên trong, số liệu sau đó chỉ
    mang tính tham khảo, không phản ánh biên độ thật trước clamp)."""
    if hasattr(model, "backbone") and hasattr(model, "upsampler") and hasattr(model, "scale"):
        shortcut = torch.repeat_interleave(lr_img, model.scale * model.scale, dim=1)
        y = model.backbone(lr_img) + shortcut
        return model.upsampler(y), True
    if (hasattr(model, "head") and hasattr(model, "body") and hasattr(model, "body_tail")
            and hasattr(model, "upsample")):
        feat = model.head(lr_img)
        f0 = feat
        for block in model.body:
            feat = block(feat)
        feat = model.body_tail(feat) + f0
        return model.upsample(feat), True
    if hasattr(model, "to_feat") and hasattr(model, "feats") and hasattr(model, "to_img"):
        feat = model.to_feat(lr_img)
        feat = model.feats(feat) + feat
        return model.to_img(feat), True
    return model(lr_img), False


@torch.no_grad()
def probe_forward(arch: str, ckpt_path: str, cfg: dict, device: str, n_samples: int,
                   out_dir: Path, tag: str, report_lines: list):
    """Bước 2/3 — forward thật trên ảnh LR test, thống kê biên độ + PSNR/SSIM tại chỗ."""
    header = f"\n--- BƯỚC 2/3 ({tag}, arch={arch}): forward trên {n_samples} ảnh LR test thật ---"
    print(header); report_lines.append(header)

    scale = cfg["image"]["scale"]
    splits_root = cfg["paths"]["splits_root"]
    splits_json = f"{splits_root}/splits.json"
    test_set = HRLRPairDataset(f"{splits_root}/hr", f"{splits_root}/lr", "test",
                                return_bbox=True, splits_json=splits_json)
    loader = DataLoader(test_set, batch_size=n_samples, shuffle=False, num_workers=0)
    lr_img, hr_img, bbox = next(iter(loader))
    lr_img, hr_img = lr_img.to(device), hr_img.to(device)

    model = build_sr_model(arch, scale)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    msg = f"  params_M = {count_params(model):.4f}"
    print(msg); report_lines.append(msg)

    # [SỬA — lỗi #1/#2] Hook độ lớn đặc trưng sau TỪNG KHỐI, dùng
    # _find_block_container() thay vì hardcode "feats" — giờ hoạt động cho
    # CẢ RLFN/RLFN_adapted (.body) lẫn SMFANet/SAFMN (.feats)/ECBSR (.backbone).
    block_stats = []
    hooks = []
    block_name, block_container = _find_block_container(model)
    if block_container is not None and len(block_container) > 1:
        def make_hook(idx):
            def hook(module, inp, out):
                block_stats.append((idx, out.abs().mean().item(), out.std().item()))
            return hook
        for i, block in enumerate(block_container):
            hooks.append(block.register_forward_hook(make_hook(i)))

    pre_clamp, has_pre_clamp_access = _forward_pre_clamp(model, lr_img)

    for h in hooks:
        h.remove()

    sr_img = torch.clamp(pre_clamp, 0.0, 1.0)

    if not has_pre_clamp_access:
        msg = (f"  (Lưu ý: kiến trúc '{arch}' không khớp 1 trong 3 khuôn mẫu forward đã biết "
               f"— số liệu 'TRƯỚC clamp' bên dưới THỰC RA đã bị clamp sẵn trong forward(), chỉ "
               f"mang tính đối chiếu tham khảo, không dùng để kết luận biên độ thật trước clamp.)")
        print(msg); report_lines.append(msg)

    msg = (f"  Output TRƯỚC clamp: min={pre_clamp.min().item():.4f} max={pre_clamp.max().item():.4f} "
           f"mean={pre_clamp.mean().item():.4f} std={pre_clamp.std().item():.4f}")
    print(msg); report_lines.append(msg)

    n_below0 = (pre_clamp < 0.0).float().mean().item() * 100
    n_above1 = (pre_clamp > 1.0).float().mean().item() * 100
    msg = f"  % pixel < 0.0 (bị clamp lên): {n_below0:.2f}%  |  % pixel > 1.0 (bị clamp xuống): {n_above1:.2f}%"
    print(msg); report_lines.append(msg)
    if has_pre_clamp_access and n_below0 + n_above1 > 30.0:
        msg = ("  -> CẢNH BÁO: hơn 30% pixel nằm ngoài [0,1] trước clamp — mô hình đang sinh "
               "biên độ SAI LỆCH LỚN, không chỉ lỗi tinh chỉnh nhỏ. Nghi vấn (nếu arch=smfanet): "
               "thiếu bias (bias=False toàn bộ SMFANet) + F.normalize không affine khiến mạng "
               "khó học đúng mức xám tuyệt đối, chỉ học được HƯỚNG tương đối của vector đặc trưng.")
        print(msg); report_lines.append(msg)

    n_sat0 = (sr_img == 0.0).float().mean().item() * 100
    n_sat1 = (sr_img == 1.0).float().mean().item() * 100
    msg = f"  SAU clamp — % pixel bão hoà tại 0.0: {n_sat0:.2f}%  |  tại 1.0: {n_sat1:.2f}%"
    print(msg); report_lines.append(msg)

    if block_stats:
        msg = f"  Độ lớn đặc trưng (mean|.|, std) sau mỗi khối '{block_name}[i]':"
        print(msg); report_lines.append(msg)
        for idx, m, s in sorted(block_stats):
            line = f"    {block_name}[{idx}]: mean|feat|={m:.4f}  std={s:.4f}"
            print(line); report_lines.append(line)
        means = [m for _, m, _ in sorted(block_stats)]
        if len(means) >= 2 and means[0] > 0 and (means[-1] > means[0] * 10 or means[-1] < means[0] * 0.1):
            msg = ("  -> CẢNH BÁO: độ lớn đặc trưng thay đổi >10x giữa khối đầu và khối cuối "
                   "(nổ hoặc triệt tiêu dần qua các khối).")
            print(msg); report_lines.append(msg)
    else:
        msg = "  (Không tìm được container nhiều khối để hook — bỏ qua breakdown theo khối.)"
        print(msg); report_lines.append(msg)

    # PSNR/SSIM ROI tại chỗ, đúng công thức eval_sr_quality.py — đối chiếu
    # với số liệu đã có trong results/sr_quality.csv.
    x0_b, y0_b, w_b, h_b = bbox
    psnrs, ssims = [], []
    for i in range(sr_img.size(0)):
        bbox_i = (int(x0_b[i]), int(y0_b[i]), int(w_b[i]), int(h_b[i]))
        psnrs.append(compute_psnr_roi(sr_img[i:i + 1], hr_img[i:i + 1], bbox_i))
        ssims.append(compute_ssim_roi(sr_img[i:i + 1], hr_img[i:i + 1], bbox_i))
    msg = (f"  PSNR ROI trung bình ({n_samples} ảnh): {sum(psnrs)/len(psnrs):.3f} dB  |  "
           f"SSIM ROI trung bình: {sum(ssims)/len(ssims):.4f}")
    print(msg); report_lines.append(msg)

    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / f"sample_{tag}.png"
    n_show = min(4, sr_img.size(0))
    rows = []
    for i in range(n_show):
        lr_up = F.interpolate(lr_img[i:i + 1], size=hr_img.shape[-2:], mode="nearest")
        rows.extend([lr_up[0], sr_img[i], hr_img[i]])
    save_image(rows, str(grid_path), nrow=3)
    msg = f"  Đã lưu ảnh mẫu (LR phóng to thô | SR | HR), mỗi hàng 1 ảnh: {grid_path}"
    print(msg); report_lines.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint SMFANet cần chẩn đoán")
    ap.add_argument("--arch", default="smfanet")
    ap.add_argument("--compare_ckpt", default=None,
                     help="[tuỳ chọn] checkpoint kiến trúc THAM CHIẾU đang hoạt động bình thường, "
                          "để so sánh quỹ đạo độ lớn đặc trưng trên CÙNG ảnh LR")
    ap.add_argument("--compare_arch", default="rlfn_adapted",
                     help="mặc định rlfn_adapted — baseline SR nhẹ CHÍNH của bài báo (xem "
                          "models/sr_models.py::build_sr_model() docstring); từ đợt 10, "
                          "_forward_pre_clamp()/_find_block_container() đã hỗ trợ đầy đủ "
                          "khuôn mẫu head/body/body_tail/upsample của RLFN nên breakdown theo "
                          "khối hoạt động bình thường với default này (không cần đổi sang safmn).")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--out_dir", default="results/debug_smfanet")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)

    report_lines = [f"CHẨN ĐOÁN SMFANet — ckpt={args.ckpt} arch={args.arch}"]

    scan_train_log(args.ckpt, report_lines)

    model = build_sr_model(args.arch, cfg["image"]["scale"])
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    check_checkpoint_weights(model, report_lines)
    del model

    probe_forward(args.arch, args.ckpt, cfg, device, args.n_samples, out_dir, "smfanet", report_lines)

    if args.compare_ckpt:
        scan_train_log(args.compare_ckpt, report_lines)
        model_cmp = build_sr_model(args.compare_arch, cfg["image"]["scale"])
        state_cmp = torch.load(args.compare_ckpt, map_location=device)
        model_cmp.load_state_dict(state_cmp)
        check_checkpoint_weights(model_cmp, report_lines)
        del model_cmp
        probe_forward(args.compare_arch, args.compare_ckpt, cfg, device, args.n_samples,
                      out_dir, args.compare_arch, report_lines)
        report_lines.append(
            f"\n--- So sánh nhanh: đối chiếu 2 khối 'BƯỚC 2/3' ở trên (smfanet vs "
            f"{args.compare_arch}) trên CÙNG {args.n_samples} ảnh LR test đầu tiên ---")

    report_lines.append(
        "\n--- KẾT LUẬN GỢI Ý (tự đánh giá dựa trên các cảnh báo '!!!'/'CẢNH BÁO' ở trên) ---\n"
        "  - Nếu KHÔNG có cảnh báo nào: số liệu PSNR thấp có thể là THẬT (kiến trúc này "
        "    thực sự không phù hợp tốt với ảnh LR rất nhỏ 20x20 của dataset) — có thể báo "
        "    cáo trung thực kèm ghi chú, KHÔNG cần loại bỏ.\n"
        "  - Nếu có cảnh báo NaN/Inf batch bị bỏ qua nhiều VÀ/HOẶC % pixel ngoài [0,1] lớn: "
        "    thử train lại với learning rate thấp hơn CHO RIÊNG smfanet (ví dụ 1e-4 -> 3e-5 "
        "    hoặc 5e-5) trước khi kết luận kiến trúc kém — KHÔNG kết luận vội kiến trúc dở "
        "    nếu nguyên nhân thực chất là bất ổn định tối ưu hoá.\n"
        "  - Nếu độ lớn trọng số/đặc trưng nổ/triệt tiêu rõ rệt ở 1 khối cụ thể (không phải "
        "    toàn bộ): đây là dấu hiệu cụ thể hơn (không phải đặc tính chung của kiến trúc) — "
        "    ưu tiên kiểm tra lại quá trình train (learning rate, gradient clipping) trước.\n"
        "  - Nếu độ lớn nổ/triệt tiêu ĐỀU qua mọi khối: đây là dấu hiệu kiến trúc (F.normalize "
        "    không affine, nếu arch=smfanet) không hợp với chế độ ảnh cực nhỏ của project — "
        "    cân nhắc NÊU RÕ giới hạn này trong bài thay vì cố sửa kiến trúc gốc (đã port "
        "    nguyên văn tác giả, tự ý đổi sẽ mất tính đối chiếu với công bố gốc)."
    )
    print(report_lines[-1])

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nĐã lưu toàn bộ báo cáo: {report_path}")


if __name__ == "__main__":
    main()
