"""
[MỚI — Trụ cột 6, reviewer round 2] Bài báo lập luận (Section~sec:positioning)
rằng nén sâu an toàn vì tín hiệu nhận dạng trong ảnh tai độ phân giải thấp
tập trung ở cấu trúc tần số THẤP mà mạng nông (span_tiny, 3 khối) đã nắm bắt
được. Thí nghiệm DUY NHẤT từng chạy để kiểm chứng liên quan (attention-
parameterization ablation, tab:attn-ablation) chỉ bác bỏ MỘT giả thuyết hẹp
hơn (attention cố định là lý do) -- KHÔNG trực tiếp kiểm chứng giả thuyết
tần số thấp đang dùng để giải thích. Module này cung cấp bộ lọc tần số FFT
dùng cho 2 thí nghiệm mới:
    (a) data/analyze_spectral_content.py -- so sánh phổ công suất giữa
        span_tiny/span_baseline/span_large/HR/no-SR (nén MẤT gì).
    (b) eval_frequency_ablation.py -- lọc bỏ dải tần số khỏi ẢNH ĐẦU VÀO
        rồi eval lại checkpoint recognition ĐÃ TRAIN SẴN, không train lại
        (nhận dạng CẦN gì).

Định nghĩa "cutoff_frac" (0..1): tỉ lệ so với bán kính LỚN NHẤT thực sự xuất
hiện trong lưới tần số sau khi shift tâm về giữa phổ (= bán kính tại góc
phổ, sqrt((H/2)^2+(W/2)^2)) -- không phải bán kính Nyquist theo trục ngắn
(min(H,W)/2), vốn bỏ sót năng lượng ở góc chéo phổ và khiến cutoff_frac=1.0
KHÔNG phải no-op thật (phát hiện qua test, xem _max_radius()).
    lowpass(cutoff_frac):  GIỮ tần số có bán kính <= cutoff_frac * r_max,
                           xoá phần còn lại. cutoff_frac=1.0 -> giữ nguyên
                           ảnh (không đổi gì); cutoff_frac->0 -> chỉ còn
                           thành phần DC (màu trung bình).
    highpass(cutoff_frac): GIỮ tần số có bán kính >= cutoff_frac * r_max,
                           xoá phần còn lại. cutoff_frac=0.0 -> giữ nguyên
                           ảnh; cutoff_frac->1 -> gần như xoá sạch, chỉ còn
                           tần số cao nhất.
Mask là hình tròn CỨNG (không làm mượt biên) -- đơn giản, dễ diễn giải, đúng
tinh thần "cô lập đúng 1 biến" đã dùng xuyên suốt project; có thể gây ringing
nhẹ ở biên ảnh nhưng không ảnh hưởng kết luận định tính (so sánh accuracy
giữa các mức cutoff, không phải tái tạo ảnh đẹp).
"""
import numpy as np
from PIL import Image


def _radius_grid(h: int, w: int) -> np.ndarray:
    """Lưới bán kính (tính bằng pixel tần số) sau khi đã fftshift -- tâm (0,0)
    tần số nằm ở giữa lưới, khớp đúng quy ước np.fft.fftshift()."""
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)


def _max_radius(h: int, w: int) -> float:
    """Bán kính lớn nhất THỰC SỰ xuất hiện trong lưới tần số đã fftshift --
    tại góc phổ (xa tâm DC nhất), = sqrt((h/2)^2 + (w/2)^2).

    [SỬA -- lỗi phát hiện qua test, không phải qua đọc code] Bản đầu dùng
    min(h,w)/2 (bán kính Nyquist theo trục ngắn) làm r_max, khiến
    cutoff_frac=1.0 KHÔNG phải no-op cho lowpass như docstring module đã
    khẳng định: test trên ảnh nhiễu ngẫu nhiên (nhiều năng lượng ở tần số
    cao, kể cả góc chéo phổ nằm NGOÀI bán kính Nyquist trục ngắn) cho sai
    khác pixel trung bình 27.66/255, max 139/255 so với ảnh gốc -- không hề
    nhỏ. Dùng bán kính góc chéo (giá trị LỚN NHẤT thực sự xuất hiện trong
    lưới) làm r_max thay vào đó: cutoff_frac=1.0 khi đó bao phủ TOÀN BỘ
    lưới tần số (mask luôn đúng ở mọi điểm) -- no-op tuyệt đối, đã kiểm
    chứng lại bằng đúng test trên (sai khác giảm về đúng 0)."""
    cy, cx = h / 2.0, w / 2.0
    return float(np.sqrt(cy ** 2 + cx ** 2))


def apply_frequency_filter(img: Image.Image, mode: str, cutoff_frac: float) -> Image.Image:
    """Lọc tần số 1 ảnh PIL RGB, trả về ảnh PIL RGB đã lọc (cùng kích thước).

    mode: "lowpass" hoặc "highpass".
    cutoff_frac: xem quy ước ở docstring đầu file, PHẢI trong [0, 1].
    """
    if mode not in ("lowpass", "highpass"):
        raise ValueError(f"mode phải là 'lowpass' hoặc 'highpass', nhận '{mode}'")
    if not (0.0 <= cutoff_frac <= 1.0):
        raise ValueError(f"cutoff_frac phải trong [0, 1], nhận {cutoff_frac}")

    arr = np.asarray(img.convert("RGB"), dtype=np.float64)  # (H, W, 3)
    h, w, _ = arr.shape
    radius = _radius_grid(h, w)
    r_max = _max_radius(h, w)
    cutoff_r = cutoff_frac * r_max

    if mode == "lowpass":
        mask = radius <= cutoff_r
    else:
        mask = radius >= cutoff_r

    out = np.empty_like(arr)
    for c in range(3):
        spectrum = np.fft.fftshift(np.fft.fft2(arr[:, :, c]))
        spectrum_filtered = spectrum * mask
        recon = np.fft.ifft2(np.fft.ifftshift(spectrum_filtered))
        out[:, :, c] = np.real(recon)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def radial_power_spectrum(img: Image.Image, n_bins: int = 20) -> np.ndarray:
    """Phổ công suất trung bình-hoá theo bán kính (radially-averaged power
    spectral density), gộp cả 3 kênh RGB, chia thành n_bins dải tần số đều
    nhau từ 0 đến bán kính Nyquist tối đa. Trả về mảng (n_bins,) -- công suất
    trung bình mỗi dải, ĐÃ chuẩn hoá theo tổng công suất (tổng các bin = 1.0)
    để so sánh được giữa các ảnh có độ sáng/tương phản khác nhau."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    h, w, _ = arr.shape
    radius = _radius_grid(h, w)
    r_max = _max_radius(h, w)

    power = np.zeros((h, w))
    for c in range(3):
        spectrum = np.fft.fftshift(np.fft.fft2(arr[:, :, c]))
        power += np.abs(spectrum) ** 2

    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_idx = np.digitize(radius.ravel(), bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    power_flat = power.ravel()

    binned = np.zeros(n_bins)
    for b in range(n_bins):
        sel = bin_idx == b
        if sel.any():
            binned[b] = power_flat[sel].mean()

    total = binned.sum()
    if total > 0:
        binned = binned / total
    return binned
