"""
Chỉ số cho cả 2 trục đánh giá: accuracy (identity/gender) và hiệu năng tính toán
(params, FLOPs, latency) — dùng để vẽ Pareto frontier ở Giai đoạn 4. Ngoài ra
có PSNR/SSIM/LPIPS để đánh giá chất lượng ảnh SR (độc lập với accuracy
downstream), và ROC/AUC/EER + confusion matrix cho phía nhận diện (bổ sung
đợt review journal Q1 — xem RUNBOOK_EarVN1.0.md mục 11).
"""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


@torch.no_grad()
def compute_topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Top-k accuracy — chuẩn Rank-N/CMC trong biometrics: tỷ lệ mẫu có nhãn
    đúng nằm trong k giá trị logit cao nhất (không nhất thiết đứng #1)."""
    k = min(k, logits.size(1))  # phòng trường hợp num_identities < k (an toàn, không xảy ra ở đây)
    topk_preds = logits.topk(k, dim=1).indices
    correct = (topk_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
    return correct / labels.size(0)


@torch.no_grad()
def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """img1, img2: tensor cùng shape, giá trị trong [0, max_val]."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse <= 1e-10:
        return 100.0  # coi như giống hệt, tránh log(0)
    return 10 * math.log10((max_val ** 2) / mse)


@torch.no_grad()
def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11,
                  sigma: float = 1.5, max_val: float = 1.0) -> float:
    """
    SSIM tự cài đặt bằng conv2d gaussian window (không phụ thuộc skimage).
    img1, img2: (B,C,H,W), giá trị trong [0, max_val].
    """
    device = img1.device
    channels = img1.shape[1]

    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    return ssim_map.mean().item()


@torch.no_grad()
def compute_psnr_roi(img1: torch.Tensor, img2: torch.Tensor, bbox, max_val: float = 1.0) -> float:
    """[MỚI — đợt 7] PSNR chỉ tính trên vùng ROI thật (không phải viền đệm
    đen do letterbox), sửa lỗi phát hiện qua code review: đo PSNR/SSIM trên
    CẢ canvas (bao gồm viền đen) làm số liệu bị THỔI PHỒNG (viền đen giống hệt
    nhau giữa SR/HR nên luôn "đoán đúng" ở đó, không phản ánh chất lượng phục
    hồi chi tiết tai thật) — không so sánh được với benchmark SR chuẩn trong
    literature. bbox = (x0, y0, w, h) từ
    utils/letterbox.py::compute_letterbox_geometry(), suy từ (width, height)
    ảnh gốc lưu sẵn trong splits.json — KHÔNG cần sinh lại ảnh nào."""
    x0, y0, w, h = bbox
    if w <= 0 or h <= 0:
        return float("nan")
    return compute_psnr(img1[:, :, y0:y0 + h, x0:x0 + w], img2[:, :, y0:y0 + h, x0:x0 + w], max_val)


@torch.no_grad()
def compute_ssim_roi(img1: torch.Tensor, img2: torch.Tensor, bbox, window_size: int = 11,
                      sigma: float = 1.5, max_val: float = 1.0) -> float:
    """[MỚI — đợt 7] SSIM chỉ tính trên vùng ROI thật, xem compute_psnr_roi().
    Tự thu nhỏ window_size nếu ROI nhỏ hơn 11px (có thể xảy ra với ảnh gốc tỷ
    lệ khung hình rất lệch — ví dụ ảnh 41x300px sau letterbox về 80x80 chỉ
    còn ROI cao ~11px) — tránh lỗi kích thước conv window > kích thước ảnh."""
    x0, y0, w, h = bbox
    if w <= 0 or h <= 0:
        return float("nan")
    c1 = img1[:, :, y0:y0 + h, x0:x0 + w]
    c2 = img2[:, :, y0:y0 + h, x0:x0 + w]
    ws = min(window_size, c1.shape[-1], c1.shape[-2])
    if ws % 2 == 0:
        ws -= 1
    ws = max(ws, 1)
    return compute_ssim(c1, c2, window_size=ws, sigma=sigma, max_val=max_val)


_LPIPS_MODEL_CACHE = {}


def load_lpips_model(device: str = "cpu", net: str = "alex"):
    """
    [MỚI — bổ sung journal Q1] Tải model LPIPS (Learned Perceptual Image Patch
    Similarity — Zhang et al., CVPR 2018, "The Unreasonable Effectiveness of
    Deep Features as a Perceptual Metric") — thước đo tương đồng cảm nhận,
    tương quan với đánh giá của con người TỐT HƠN PSNR/SSIM (PSNR/SSIM chỉ so
    khớp pixel/cấu trúc cục bộ, không phản ánh chất lượng cảm nhận thật).
    Phần lớn reviewer SR hiện nay mặc định hỏi LPIPS bên cạnh PSNR/SSIM.

    Dùng backbone AlexNet (net="alex") — chuẩn phổ biến nhất trong literature
    SR để so sánh (nhẹ hơn VGG, là lựa chọn mặc định trong hầu hết bài SR
    dùng LPIPS, bao gồm chính bài gốc).

    QUAN TRỌNG: gọi hàm này 1 LẦN DUY NHẤT trước vòng lặp đánh giá (giống
    cách build_sr_model() chỉ gọi 1 lần) — KHÔNG gọi lại trong loop, vì tải
    trọng số AlexNet mỗi lần rất tốn thời gian. Cache theo (device, net) để
    an toàn nếu lỡ gọi nhiều lần.

    Cần cài: pip install lpips --break-system-packages (xem requirements.txt).

    GIỚI HẠN MINH BẠCH: sandbox phát triển code này KHÔNG có quyền cài package
    mới (chặn qua proxy) nên KHÔNG thể chạy thực nghiệm để tự kiểm tra hàm
    này — cài đặt dựa trên API ổn định, được ghi trong tài liệu chính thức
    của package `lpips` (ổn định nhiều năm nay, không đổi signature). Người
    dùng cần tự chạy thử (ví dụ qua test_new_sr_archs.py hoặc 1 lần chạy
    eval_sr_quality.py nhỏ) trước khi tin tưởng số liệu LPIPS đầu ra.
    """
    key = (device, net)
    if key in _LPIPS_MODEL_CACHE:
        return _LPIPS_MODEL_CACHE[key]
    try:
        import lpips
    except ImportError:
        raise ImportError(
            "Cần cài package `lpips` để tính LPIPS: pip install lpips --break-system-packages"
        )
    model = lpips.LPIPS(net=net).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _LPIPS_MODEL_CACHE[key] = model
    return model


@torch.no_grad()
def compute_lpips(lpips_model, img1: torch.Tensor, img2: torch.Tensor,
                   min_size: int = 64) -> float:
    """
    [MỚI — bổ sung journal Q1] Tính LPIPS giữa 2 ảnh (B,3,H,W), giá trị trong
    [0,1] (dùng normalize=True của package `lpips` — nhận thẳng ảnh [0,1],
    KHÔNG cần tự scale về [-1,1], tránh lỗi quên scale). LPIPS CÀNG THẤP CÀNG
    TỐT (ngược hướng với PSNR/SSIM) — 0 = giống hệt.

    min_size: LPIPS dùng mạng CNN sâu (AlexNet) bên trong — nếu ảnh đầu vào
    quá nhỏ (ví dụ crop ROI vài chục px với ảnh gốc tỷ lệ khung hình rất lệch,
    xem compute_ssim_roi()), các tầng stride/pool liên tiếp có thể làm feature
    map co về kích thước 0, gây lỗi runtime. Cách xử lý: nếu min(H,W) <
    min_size, resize (bilinear) ảnh lên min_size trước khi đưa vào LPIPS —
    giống cách compute_ssim_roi() tự thu nhỏ window_size, nhưng theo hướng
    ngược lại (phóng to thay vì thu nhỏ), vì LPIPS không có tham số kiểu
    window_size để tự co giãn theo input. Đây là cách xử lý phổ biến trong
    literature khi dùng LPIPS trên ảnh/crop rất nhỏ — có làm mờ nhẹ chi tiết,
    nhưng LPIPS là thước đo TƯƠNG ĐỒNG chứ không phải phép đo pixel-chính-xác
    nên chấp nhận được, và áp dụng ĐỒNG NHẤT cho mọi model so sánh (không
    thiên vị model nào).
    """
    h, w = img1.shape[-2], img1.shape[-1]
    if min(h, w) < min_size:
        img1 = F.interpolate(img1, size=(min_size, min_size), mode="bilinear", align_corners=False)
        img2 = F.interpolate(img2, size=(min_size, min_size), mode="bilinear", align_corners=False)
    device = next(lpips_model.parameters()).device
    d = lpips_model(img1.to(device), img2.to(device), normalize=True)
    return d.mean().item()


@torch.no_grad()
def compute_pairwise_genuine_impostor_scores(embeddings: torch.Tensor, labels: torch.Tensor):
    """
    [MỚI — bổ sung journal Q1] Từ tập embedding (N, D) + nhãn identity (N,)
    của TOÀN BỘ test set, tính cosine similarity cho MỌI cặp (i<j, không lặp,
    không tự so với chính mình), tách thành 2 nhóm:
      - genuine: cặp CÙNG identity (đúng phải giống nhau — bài toán verification)
      - impostor: cặp KHÁC identity (đúng phải khác nhau)
    Đây là cách chuẩn trong biometrics để đánh giá ở "verification setting"
    (khác với "identification setting" của identity_accuracy/rank-k — đo khả
    năng phân biệt CẶP ảnh có cùng người hay không, không phụ thuộc số lượng
    identity cố định trong tập train, thường được review yêu cầu bên cạnh
    accuracy khi bài báo có đóng góp chính ở phía recognition).

    Trả về (genuine_scores, impostor_scores) — 2 tensor 1D trên cùng device
    với embeddings.

    LƯU Ý: số cặp genuine phụ thuộc số ảnh/identity trong test set — nếu quá
    ít (ví dụ nhiều identity chỉ có 1 ảnh test), EER/AUC ước lượng sẽ kém tin
    cậy. Hàm gọi (eval_recognition.py) in cảnh báo nếu n_genuine_pairs quá nhỏ.
    """
    emb_norm = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = emb_norm @ emb_norm.t()  # (N, N) cosine similarity
    n = labels.size(0)
    same_id = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N) bool
    # chỉ lấy tam giác trên, KHÔNG gồm đường chéo (i!=j)
    upper_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=labels.device), diagonal=1)
    genuine_mask = same_id & upper_mask
    impostor_mask = (~same_id) & upper_mask
    genuine_scores = sim_matrix[genuine_mask]
    impostor_scores = sim_matrix[impostor_mask]
    return genuine_scores, impostor_scores


def compute_roc_auc_eer(genuine_scores, impostor_scores, n_thresholds: int = 1000) -> dict:
    """
    [MỚI — bổ sung journal Q1] AUC (Area Under ROC Curve) + EER (Equal Error
    Rate — điểm mà FPR = FNR, thước đo chuẩn trong verification/biometrics,
    độc lập với việc chọn ngưỡng quyết định) từ 2 tập điểm số genuine/impostor.
    Chấp nhận torch.Tensor hoặc array-like (tự chuyển numpy nội bộ).

    Cách làm:
    - AUC: tính qua thống kê hạng (rank-based, tương đương Mann-Whitney U) —
      CHÍNH XÁC (không xấp xỉ qua lưới ngưỡng), xử lý đúng trường hợp có giá
      trị trùng (tie) bằng rank trung bình.
    - EER: quét lưới `n_thresholds` ngưỡng từ cao xuống thấp, tính FPR/FNR
      tại mỗi ngưỡng, tìm giao điểm đường FPR và FNR bằng nội suy tuyến tính
      giữa 2 điểm lưới liền kề đổi dấu hiệu (dấu của FNR-FPR).

    Đã kiểm tra bằng test số học tổng hợp (numpy, ngoài project, xem quá
    trình review): 2 phân phối tách biệt hoàn toàn -> AUC=1.0, EER~0; 2 phân
    phối trùng hoàn toàn -> AUC~0.5, EER~0.5; 2 Gaussian chồng lấn 1 phần ->
    AUC khớp công thức lý thuyết Phi(d'/sqrt(2)) trong khoảng sai số < 0.02.

    Trả về dict {"auc", "eer", "eer_threshold", "n_genuine_pairs", "n_impostor_pairs"}.
    """
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float64)
        return np.asarray(x, dtype=np.float64)

    genuine = _to_numpy(genuine_scores)
    impostor = _to_numpy(impostor_scores)
    n_g, n_i = len(genuine), len(impostor)
    if n_g == 0 or n_i == 0:
        return {"auc": float("nan"), "eer": float("nan"), "eer_threshold": float("nan"),
                "n_genuine_pairs": n_g, "n_impostor_pairs": n_i}

    # --- AUC qua rank-based (Mann-Whitney U), chính xác, không cần chọn ngưỡng ---
    all_scores = np.concatenate([genuine, impostor])
    order = np.argsort(all_scores, kind="mergesort")
    sorted_scores = all_scores[order]
    sorted_ranks = np.arange(1, len(all_scores) + 1, dtype=np.float64)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        sorted_ranks[i:j + 1] = (sorted_ranks[i] + sorted_ranks[j]) / 2.0  # rank trung bình cho tie
        i = j + 1
    ranks = np.empty(len(all_scores), dtype=np.float64)
    ranks[order] = sorted_ranks
    genuine_ranks_sum = ranks[:n_g].sum()
    auc = (genuine_ranks_sum - n_g * (n_g + 1) / 2.0) / (n_g * n_i)

    # --- EER qua quét ngưỡng + nội suy giao điểm FPR/FNR ---
    lo = min(genuine.min(), impostor.min())
    hi = max(genuine.max(), impostor.max())
    thresholds = np.linspace(hi, lo, n_thresholds)  # giảm dần
    tpr = np.array([(genuine >= t).mean() for t in thresholds])
    fpr = np.array([(impostor >= t).mean() for t in thresholds])
    fnr = 1.0 - tpr
    diff = fnr - fpr
    idx = np.where(np.diff(np.sign(diff)))[0]
    if len(idx) == 0:
        eer_idx = int(np.argmin(np.abs(diff)))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
        eer_threshold = thresholds[eer_idx]
    else:
        k = idx[0]
        x0, x1 = fpr[k], fpr[k + 1]
        y0, y1 = fnr[k], fnr[k + 1]
        t0, t1 = thresholds[k], thresholds[k + 1]
        if (y0 - x0) == (y1 - x1):
            eer, eer_threshold = (x0 + y0) / 2.0, t0
        else:
            alpha = (y0 - x0) / ((y0 - x0) - (y1 - x1))
            eer = x0 + alpha * (x1 - x0)
            eer_threshold = t0 + alpha * (t1 - t0)

    return {"auc": float(auc), "eer": float(eer), "eer_threshold": float(eer_threshold),
            "n_genuine_pairs": n_g, "n_impostor_pairs": n_i}


@torch.no_grad()
def compute_confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> np.ndarray:
    """
    [MỚI — bổ sung journal Q1] Ma trận nhầm lẫn (confusion matrix) NxN cho
    bài toán identity classification — hàng = nhãn thật, cột = nhãn dự đoán.
    Dùng để phân tích lỗi theo lớp (identity nào hay bị nhầm với identity
    nào) — bổ sung cho identity_accuracy tổng, vốn không cho biết lỗi có dồn
    vào 1 số ít identity (ví dụ trùng nhau về góc chụp/ánh sáng) hay dàn đều.

    Cách làm: mã hoá mỗi cặp (label, pred) thành 1 số nguyên duy nhất
    label*num_classes+pred rồi đếm tần suất bằng bincount — nhanh, không cần
    vòng lặp Python, không cần sklearn (giữ đúng tinh thần dependency tối
    thiểu của project — chỉ numpy/torch sẵn có).
    """
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    idx = labels * num_classes + preds
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).numpy()


def count_params(model: torch.nn.Module) -> float:
    """Trả về số triệu tham số (M)."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_flops(model: torch.nn.Module, input_size, device: str = "cpu") -> tuple:
    """
    [SỬA — lỗi nhãn phát hiện qua review Q1] Trả về (GMACs, GFLOPs) cho 1 lần
    forward với kích thước input_size (ví dụ (1,3,32,32)).

    LỖI TRƯỚC ĐÂY: hàm này trả về `flops/1e9` và gọi thẳng là "GFLOPs", nhưng
    `thop.profile()` thực ra trả về MACs (multiply-accumulate operations),
    KHÔNG PHẢI FLOPs — đã kiểm chứng độc lập bằng tay: Conv2d(1,1,kernel=1)
    trên input 1x1x2x2 có MACs=Cout*Cin*K*K*H*W=4 (đúng hand-calc), trong khi
    quy ước FLOPs=2xMACs (1 phép nhân + 1 phép cộng/vị trí) sẽ ra 8 —
    `thop.profile()` trả về đúng 4.0, xác nhận nó trả MACs chứ không phải
    FLOPs. Với 1 bài báo về "efficient SR" trích dẫn NTIRE (nơi các baseline
    RLFN/ECBSR/SAFMN/SMFANet công bố số liệu riêng của họ), lệch 2x giữa
    MACs/FLOPs khi đối chiếu trực tiếp với literature là sai số nghiêm trọng,
    không phải sai số làm tròn.

    QUY ƯỚC CỘNG ĐỒNG SR/NTIRE KHÔNG THỐNG NHẤT (một số bài tự gọi MACs là
    "FLOPs" một cách không chính xác) — nên hàm này trả về CẢ HAI, gọi tên rõ
    ràng, để người dùng tự đối chiếu đúng với quy ước mà từng baseline được
    trích dẫn thực sự dùng, thay vì đoán.

    Dùng thư viện thop — cần `pip install thop`.
    """
    try:
        from thop import profile
    except ImportError:
        raise ImportError(
            "Cần cài thop để tính FLOPs: pip install thop --break-system-packages"
        )
    model = model.to(device).eval()
    dummy = torch.randn(*input_size).to(device)
    macs, _ = profile(model, inputs=(dummy,), verbose=False)
    macs_g = macs / 1e9
    return macs_g, macs_g * 2  # (GMACs, GFLOPs=2xGMACs)


def freeze_reparam_modules(model: torch.nn.Module) -> int:
    """
    SPAN chính thức dùng kỹ thuật 'structural reparameterization' (module
    Conv3XC): nhiều nhánh conv lúc train, hợp nhất thành 1 conv duy nhất lúc
    suy luận. ĐÚNG cách dùng chuẩn: hợp nhất 1 LẦN DUY NHẤT trước khi triển
    khai/đo tốc độ. Nhưng forward() của code gốc lại gọi update_params() (tính
    lại phép hợp nhất) MỖI LẦN forward, kể cả ở eval mode — với 100 lần gọi
    lặp lại trong measure_latency(), điều này làm SPAN baseline bị đo CHẬM GIẢ
    TẠO (tính dư thừa 100 lần cho cùng 1 kết quả), không phản ánh đúng tốc độ
    suy luận thật. Hàm này gọi update_params() một lần duy nhất rồi vô hiệu
    hóa việc tính lại — cho kết quả đo latency công bằng, đúng cách deploy
    chuẩn của các kiến trúc reparameterization (RepVGG-style).

    Trả về số module đã được "đóng băng" (0 nếu model không dùng kỹ thuật này,
    ví dụ span_tiny — khi đó hàm này không làm gì cả, an toàn để gọi luôn).
    """
    count = 0
    for module in model.modules():
        if hasattr(module, "update_params") and hasattr(module, "eval_conv"):
            module.update_params()          # tính hợp nhất lần cuối, đúng trọng số hiện tại
            module.update_params = lambda: None   # từ giờ không tính lại nữa
            count += 1
    return count


def count_params_deploy_mode(model: torch.nn.Module) -> float:
    """
    Đo số tham số Ở CHẾ ĐỘ DEPLOY (đã hợp nhất reparameterization) — chỉ tính
    nhánh eval_conv của mỗi Conv3XC, BỎ QUA nhánh train (self.conv, self.sk)
    vì nhánh đó KHÔNG được dùng khi suy luận thực tế, chỉ tồn tại để hỗ trợ
    học trong lúc train.

    QUAN TRỌNG: đây RẤT CÓ THỂ là cách các bài báo SR gốc (bao gồm SPAN, và
    nhiều kiến trúc dùng structural reparameterization khác) báo cáo số tham
    số trong bảng kết quả — KHÁC với count_params() thông thường (tính cả 2
    nhánh, tổng luôn LỚN HƠN). Dùng hàm này để so sánh công bằng với số liệu
    tác giả công bố, tránh nhầm lẫn "model của mình nặng hơn" trong khi thực
    ra chỉ là khác quy ước đếm.

    Cách làm: đệ quy qua cây module. Khi gặp Conv3XC (nhận diện qua có cả
    thuộc tính eval_conv lẫn update_params), CHỈ đếm tham số của eval_conv,
    KHÔNG đệ quy tiếp vào các con khác của nó (self.conv, self.sk — nhánh
    train). Với module thường, đếm tham số riêng rồi đệ quy tiếp vào con.
    """
    def recurse(module):
        if hasattr(module, "eval_conv") and hasattr(module, "update_params"):
            # Đây là Conv3XC: chỉ đếm eval_conv, KHÔNG đệ quy vào self.conv/self.sk
            return sum(p.numel() for p in module.eval_conv.parameters())
        total = sum(p.numel() for p in module.parameters(recurse=False))
        for child in module.children():
            total += recurse(child)
        return total

    return recurse(model) / 1e6


@torch.no_grad()
def measure_latency(model: torch.nn.Module, input_size, device: str,
                     n_warmup: int = 10, n_iters: int = 100) -> float:
    """
    Đo latency trung bình (ms/ảnh) trên thiết bị chỉ định.
    input_size: ví dụ (1, 3, 32, 32) cho SR, hoặc (1, 3, 128, 128) cho recognition.
    Tự động "đóng băng" các module reparameterization (nếu có, ví dụ Conv3XC
    của SPAN chính thức) để đo công bằng — xem freeze_reparam_modules().
    Lưu ý: để đo latency thực tế trên thiết bị edge (Jetson/mobile), hãy chạy
    script này trực tiếp trên thiết bị đó, không chỉ trên máy train.
    """
    model.eval().to(device)
    n_frozen = freeze_reparam_modules(model)

    dummy = torch.randn(*input_size).to(device)

    for _ in range(n_warmup):
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_iters):
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    return (elapsed / n_iters) * 1000  # ms/ảnh
