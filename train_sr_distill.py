"""
Giai đoạn 3 — cải tiến SPAN: train SPAN (student, nhẹ) với loss tổng hợp gồm
4 thành phần TÙY CHỌN (BẬT/TẮT riêng qua lambda tương ứng — xem "MẶC ĐỊNH"
bên dưới), dùng nhiều "giám khảo" đã train sẵn và đóng băng:
  - Teacher SR nặng (mặc định span_official) — cung cấp 2 dạng distillation:
      (a) output-level: L1 giữa ảnh SR student/teacher (như bản gốc)
      (b) [MỚI] feature-level (hint-based, SRFeatureHook trong
          models/sr_models.py): L1 giữa feature map NỘI BỘ (kênh `feat`,
          TRƯỚC conv chiếu cuối feat->3*scale², KHÔNG PHẢI tensor đã đóng gói
          RGB đi vào PixelShuffle — xem docstring SRFeatureHook để rõ khác
          biệt và lý do bản vá code review đã sửa lại điểm hook này) của
          student (qua 1 adapter Conv1x1 học được, cả 2 phía đều được chuẩn
          hoá zero-mean/unit-std TRƯỚC khi so khớp) và của teacher.
  - [MỚI] Hội đồng NHIỀU recognition model đã train trên domain HR (thay vì
    1 model duy nhất) — cung cấp identity-aware loss, trung bình cosine loss
    qua từng judge.
  - [MỚI] Saliency-weighted identity-critical pixel loss
    (compute_multi_judge_saliency): trọng số không gian suy từ gradient CỦA
    ĐÚNG LOGIT IDENTITY THẬT (không phải năng lượng embedding) theo từng
    pixel ảnh HR, min-max chuẩn hoá CHỈ trên vùng ROI (không bị viền đen
    letterbox làm nhiễu) — ép model ưu tiên tái tạo đúng vùng ảnh hưởng
    nhiều nhất đến đặc trưng nhận dạng (ví dụ nếp sụn tai) thay vì tối ưu
    đồng đều cả ảnh (bao gồm cả tóc/nền).

MẶC ĐỊNH [SỬA — mô tả sai phát hiện qua code review]: `lambda_feat` và
`lambda_saliency` mặc định = 0.0 trong configs/config.yaml (opt-in, KHÔNG
tự bật) — để không làm lệch các script ablation/sweep cũ vốn chỉ so sánh
pixel/distill/identity. Muốn dùng feature-KD/saliency-loss, truyền tường
minh qua CLI (`--lambda_feat`, `--lambda_saliency`) hoặc sửa config, xem
pipeline/run_ablation_kd_v2.sh và pipeline/run_lambda_saliency_sweep.sh.

ĐỘNG LỰC bản vá multi-judge + feature-KD này (xem RUNBOOK_EarVN1.0.md mục 12
"Multi-Judge Ensemble Identity Loss + Feature-level KD" để biết đầy đủ):
  - Bản gốc chỉ dùng 1 recognition model (mobilenet_v2) làm giám khảo cho
    identity loss. Khi tăng lambda_identity, kết quả downstream accuracy XẤU
    ĐI có ý nghĩa thống kê (Cohen's d=-10.09) — cách diễn giải hợp lý nhất:
    SR bị ép tối ưu theo đúng "gu" của 1 backbone, tạo đặc trưng ăn khớp
    riêng với backbone đó nhưng KHÔNG tổng quát hoá được sang backbone khác
    — khớp đúng hiện tượng "accuracy không đồng nhất giữa backbone" quan sát
    được trong Table 1/6 của bản thảo bài báo. Multi-judge (nhiều backbone
    khác họ kiến trúc cùng lúc) nhắm thẳng vào cơ chế này.
  - Bản gốc distillation chỉ so khớp OUTPUT PIXEL cuối cùng (dạng KD yếu nhất
    trong literature) — feature-level (hint-based, kiểu Lee et al. ECCV 2020
    đã trích trong Related Work của bài báo) là tín hiệu mạnh hơn, CHƯA từng
    thử trong project này trước bản vá này.
  - Phân tích định tính trong bản thảo bài báo (Figure 2b) chỉ ra PSNR đôi
    khi cao là do tái tạo tốt TÓC/NỀN chứ không phải TAI (vùng bị che khuất
    trong ảnh mẫu) — pixel loss L1 đồng đều hiện tại không phân biệt được
    điều này. Saliency-weighted loss nhắm thẳng vào khoảng trống này, tự suy
    trọng số từ chính judge đã có (không cần nhãn segmentation tai mới).

TƯƠNG THÍCH NGƯỢC: nếu config không có "identity_judges" (hoặc để rỗng []),
tự động fallback về hành vi single-judge cũ (đọc "frozen_recognition_ckpt",
backbone cố định "mobilenet_v2" — ĐÚNG hệt hành vi trước bản vá này). Nếu
"lambda_feat" không có trong config hoặc =0, feature-level distillation tự
tắt hoàn toàn (không tính hook, không tốn thêm chi phí) — script chạy giống
hệt bản gốc với config cũ, không cần sửa gì thêm để tái lập kết quả cũ.

QUAN TRỌNG [SỬA — mô tả sai phát hiện qua code review]: chỉ Student SPAN (+
feature adapter, chỉ tồn tại lúc train) được CẬP NHẬT trọng số (nằm trong
optimizer, requires_grad=True). Teacher SR và mọi recognition judge bị ĐÓNG
BĂNG (requires_grad=False, không có trong optimizer) — NHƯNG điều đó KHÔNG
có nghĩa là chúng "chỉ forward torch.no_grad()":
  - Nhánh HR-side của identity loss (judge.embed(hr_img), giá trị neo cố
    định) dùng torch.no_grad() thật, vì hr_img không phải biến cần tối ưu.
  - Nhánh student-side của identity loss (judge.embed(student_out)) và toàn
    bộ saliency (judge(hr_req) lấy logits) BẮT BUỘC chạy CÓ gradient — cần
    lan truyền ngược QUA judge (tham số đóng băng, không tích luỹ gradient
    vào judge.parameters() vì requires_grad=False) để tới student_out/hr_req.
    Nếu bọc torch.no_grad() ở đây, gradient của loss_identity/loss_saliency
    sẽ KHÔNG bao giờ tới được student — bug im lặng, không có lỗi runtime.

Kiến trúc/tốc độ của student KHÔNG đổi so với bản baseline của chính nó —
feature adapter + nhiều judge chỉ tồn tại lúc TRAIN, không ảnh hưởng đến
latency/params lúc deploy (chỉ deploy riêng student.state_dict()).

Chạy (giống hệt cú pháp cũ — [SỬA lệch mô tả phát hiện qua code review, vòng
3] MẶC ĐỊNH configs/config.yaml hiện KHÔNG tự bật multi-judge/feature-KD/
saliency, lambda_feat=lambda_saliency=lambda_identity=0.0, xem mục "MẶC ĐỊNH"
ở trên — lệnh dưới đây CHỈ chạy pixel+distill thuần, giống hệt bản gốc trước
mọi bản vá multi-judge/feature-KD; muốn bật các cơ chế mới PHẢI truyền tường
minh --lambda_feat/--lambda_saliency/--lambda_identity, xem
pipeline/run_ablation_kd_v2.sh):
    python train_sr_distill.py --config configs/config.yaml

Chạy với lambda tùy chỉnh (dùng cho ablation, xem pipeline/run_ablation.sh
và pipeline/run_ablation_kd_v2.sh):
    python train_sr_distill.py --config configs/config.yaml \
        --lambda_pixel 1.0 --lambda_distill 1.0 --lambda_feat 0.0 \
        --lambda_identity 0.0 --run_suffix _ablation_pixel_distill

Chạy với kiến trúc student khác (dùng cho ablation kiến trúc, ví dụ
span_large — xem pipeline/run_span_large_ablation.sh):
    python train_sr_distill.py --config configs/config.yaml \
        --student_arch span_large

Chạy FINE-TUNE / TRANSFER LEARNING xuyên dataset (ví dụ: khởi tạo từ
checkpoint span_tiny đã train trên EarVN1.0, fine-tune tiếp trên AWE — xem
scripts/run_transfer_learning.sh):
    python train_sr_distill.py --config configs/config_awe_finetune.yaml \
        --init_ckpt runs/sr_improved_span_tiny/best.pt \
        --run_suffix _finetuned_from_earvn1
    (an toàn dùng strict=True vì model SR không có tầng phụ thuộc số lượng
    identity — kiến trúc giống hệt nhau giữa mọi dataset, không có rủi ro
    lệch shape như bên recognition.)
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.hrlr_pair_dataset import HRLRPairDataset
from models.sr_models import build_sr_model, SRFeatureHook
from models.recognition_model import EarRecognitionNet
from utils.device_manager import DeviceManager, move_optimizer_state
from utils.early_stopping import (EarlyStopping, save_state_dict as _save_state_dict,
                                   save_last_if_missing as _save_last_if_missing)
from utils.logger import setup_logger
from utils.seed import set_seed, seed_worker, seeded_generator


def build_judges(cfg, device):
    """[MỚI] Xây dựng danh sách recognition model giám khảo (đóng băng).

    Đọc "sr_improve.identity_judges" (list các {backbone, ckpt}) nếu có và
    không rỗng -> multi-judge. Nếu không -> fallback TƯƠNG THÍCH NGƯỢC về
    đúng hành vi cũ: 1 judge duy nhất, backbone cố định "mobilenet_v2", đọc
    checkpoint từ "frozen_recognition_ckpt"."""
    ci = cfg["sr_improve"]
    judge_specs = ci.get("identity_judges") or None
    if not judge_specs:
        judge_specs = [{"backbone": "mobilenet_v2", "ckpt": ci["frozen_recognition_ckpt"]}]

    judges = []
    for spec in judge_specs:
        ckpt_path = Path(spec["ckpt"])
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint recognition giám khảo không tồn tại: {ckpt_path} "
                f"(backbone={spec['backbone']}). Cần khi lambda_identity>0 hoặc "
                f"lambda_saliency>0. Chạy pipeline/03_train_baseline_recognition.sh trước.")
        judge = EarRecognitionNet(
            num_identities=cfg["num_identities"],
            num_genders=cfg["num_genders"],
            embedding_dim=cfg["recognition"]["embedding_dim"],
            backbone=spec["backbone"],
            pretrained=False,
        ).to(device)
        judge.load_state_dict(torch.load(ckpt_path, map_location=device))
        judge.eval()
        for p in judge.parameters():
            p.requires_grad = False
        judges.append(judge)
    return judges, judge_specs


def _setup_feat_kd(student, teacher, lambda_feat, cfg, scale, device, logger, tag="",
                    adapter_init_state=None):
    """[MỚI — tách ra dùng chung cho train_sr_distill.py VÀ
    train_sr_learned_prune.py sau code review, tránh 2 bản copy-paste lệch
    nhau] Khởi tạo hook feature-KD cho 1 cặp student/teacher + adapter Conv1x1.
    Trả về (None, None, None, []) nếu lambda_feat<=0 (tắt hẳn, không tốn chi phí).

    [SỬA — LỖI PHÁT HIỆN QUA CODE REVIEW, vòng 2] Trước đây `SRFeatureHook`
    có cờ nội bộ `_is_fallback` nhưng KHÔNG NƠI NÀO kiểm tra/log — nếu vì lý
    do nào đó (kiến trúc lạ, hoặc thay đổi cấu trúc SPAN sau này) hook rơi về
    fallback (hook thẳng PixelShuffle, gần như trùng loss_distill, KHÔNG PHẢI
    hint-based KD thật), quá trình train vẫn chạy êm re và log
    "student_feat_ch=48 teacher_feat_ch=48" TRÔNG HOÀN TOÀN BÌNH THƯỜNG (với
    SPAN/span_tiny ở scale=4, kênh feature nội bộ VÀ kênh RGB đã đóng gói
    TRÙNG NGẪU NHIÊN đều =48 = 3*scale²) — không thể phát hiện qua log số
    kênh. Giờ kiểm tra tường minh `hook.is_fallback` và cảnh báo to nếu True.

    [SỬA — bổ sung, điểm 5 review vòng 2] `adapter_init_state`: nếu truyền
    vào (state_dict của 1 adapter Conv1x1 đã học từ pha trước, ví dụ pha
    "gated" trước khi harden), NẠP LUÔN vào adapter mới thay vì để random init
    — áp dụng cho pha fine-tune sau harden_and_export() trong
    train_sr_learned_prune.py: model export ra vẫn giữ nguyên feat=48 kênh
    (chỉ đổi số khối), nên adapter (student_ch=48 -> teacher_ch cố định) có
    CÙNG shape giữa 2 pha — không có lý do gì bỏ phí trọng số đã học, để
    student phải "làm quen lại từ đầu" ngay khi vừa bị harden (đã đủ nhiễu)."""
    if lambda_feat <= 0:
        return None, None, None, []
    student_hook = SRFeatureHook(student)
    teacher_hook = SRFeatureHook(teacher)
    lr_size = cfg["image"]["hr_size"] // scale
    with torch.no_grad():
        dummy = torch.zeros(1, 3, lr_size, lr_size, device=device)
        student(dummy)
        teacher(dummy)
    if student_hook.feat is None or teacher_hook.feat is None:
        raise RuntimeError(
            f"SRFeatureHook không bắt được feature sau dummy forward "
            f"(student={'ok' if student_hook.feat is not None else 'None'}, "
            f"teacher={'ok' if teacher_hook.feat is not None else 'None'}). "
            f"Kiến trúc student/teacher có thể không đi qua module đã hook."
        )
    if student_hook.feat.shape[2:] != teacher_hook.feat.shape[2:]:
        raise RuntimeError(
            f"Feature spatial mismatch: student {tuple(student_hook.feat.shape)} vs "
            f"teacher {tuple(teacher_hook.feat.shape)}. Adapter 1x1 không nội suy "
            f"không gian — L1 sẽ crash ở batch đầu. Kiểm tra scale student/teacher."
        )
    student_ch = student_hook.feat.shape[1]
    teacher_ch = teacher_hook.feat.shape[1]
    feat_adapter = nn.Conv2d(student_ch, teacher_ch, kernel_size=1).to(device)
    if adapter_init_state is not None:
        feat_adapter.load_state_dict(adapter_init_state)
        if logger:
            logger.info(f"[Feature KD{tag}] Đã copy trọng số feat_adapter đã học từ pha "
                        f"trước (KHÔNG khởi tạo random) — tránh phải học lại từ đầu ngay "
                        f"sau harden/thay đổi kiến trúc.")
    if logger:
        logger.info(f"[Feature KD{tag}] student_feat_ch={student_ch} teacher_feat_ch={teacher_ch} "
                    f"-> adapter Conv1x1({student_ch}->{teacher_ch}) (chỉ tồn tại lúc train, "
                    f"không lưu vào checkpoint deploy)")
        if student_hook.is_fallback or teacher_hook.is_fallback:
            logger.warning(
                f"[Feature KD{tag}] CẢNH BÁO: "
                f"{'student' if student_hook.is_fallback else ''}"
                f"{' và ' if student_hook.is_fallback and teacher_hook.is_fallback else ''}"
                f"{'teacher' if teacher_hook.is_fallback else ''} hook đã FALLBACK về hook "
                f"thẳng PixelShuffle (không tìm thấy pattern Sequential[Conv2d, PixelShuffle] "
                f"mong đợi) — feature bắt được ở đây là ẢNH ĐÃ ĐÓNG GÓI KÊNH (packed RGB), "
                f"KHÔNG PHẢI feature nội bộ thật, loss_feat sẽ gần trùng loss_distill "
                f"(output-level KD), KHÔNG PHẢI hint-based KD như thiết kế. Số kênh log ở "
                f"trên (student_feat_ch/teacher_feat_ch) KHÔNG đủ để phát hiện lỗi này nếu "
                f"trùng ngẫu nhiên với 3*scale² — xem models/sr_models.py::SRFeatureHook.")
    return student_hook, teacher_hook, feat_adapter, list(feat_adapter.parameters())


def compute_multi_judge_saliency(judges, hr_img, identity_label, bbox=None,
                                  blur_kernel=5, floor=0.3):
    """[MỚI — SỬA sau code review] Saliency map trung bình qua nhiều judge —
    cường độ gradient của ĐÚNG logit identity thật (∂logit[y_true]/∂I) theo
    từng pixel input HR, cho biết pixel nào ảnh hưởng nhiều nhất đến việc
    PHÂN LOẠI ĐÚNG danh tính theo góc nhìn của judge đó. Dùng làm TRỌNG SỐ
    KHÔNG GIAN cho pixel loss (xem loss_saliency trong compute_total_loss) —
    ép model ưu tiên tái tạo đúng vùng quan trọng cho nhận dạng (ví dụ nếp
    sụn tai) hơn vùng không quan trọng (tóc/nền).

    [SỬA — LỖI PHÁT HIỆN QUA CODE REVIEW] Bản đầu tiên dùng
    score=(emb**2).sum() (năng lượng embedding SAU BatchNorm1d) làm hàm mục
    tiêu lấy gradient — đây KHÔNG phải tín hiệu "identity-critical" thật: nó
    không phụ thuộc identity_label, chỉ đo "embedding này có norm lớn hay
    nhỏ", hoàn toàn có thể lớn ở vùng KHÔNG liên quan phân loại (ví dụ
    artefact/nhiễu). Bản vá này dùng ĐÚNG logit của lớp identity thật
    (identity_head, gather theo identity_label) — CHUẨN gradient-based
    saliency map cho classification (Simonyan et al. 2014), đúng tinh thần
    "identity-CRITICAL" như tên gọi.

    ĐỘNG LỰC: nhắm trực tiếp vào phát hiện định tính trong bản thảo bài báo
    (Figure 2b): PSNR gần như hoà (30.92 vs 30.89 dB) hoá ra là do tái tạo
    tốt phần TÓC/NỀN chứ không phải phần tai (bị che khuất trong mẫu đó) —
    tức là L1 pixel loss đồng đều hiện tại "phí" một phần dung lượng học vào
    vùng không quan trọng cho nhận dạng. KHÔNG cần nhãn segmentation tai mới
    (EarVN1.0 không có) — saliency tự suy ra từ gradient của CHÍNH model
    giám khảo đã có sẵn (multi-judge) cộng với nhãn person_id đã có sẵn
    trong dataset, không cần huấn luyện thêm model nào.

    [SỬA — bổ sung sau code review] bbox=(x0,y0,w,h) mỗi phần tử là tensor
    (B,) (xem HRLRPairDataset return_bbox) — nếu truyền vào, saliency BÊN
    NGOÀI vùng ROI thật (viền đệm đen letterbox) bị ép cứng về `floor`, bất
    kể gradient thật ở đó là bao nhiêu. LÝ DO: gradient-based saliency vốn
    thô (lan truyền qua nhiều tầng downsample của backbone rồi blur) và dễ
    "dính" biên tương phản mạnh giữa ảnh thật và viền đen letterbox dù vùng
    đó không mang thông tin nhận dạng nào — ép cứng loại bỏ hẳn rủi ro này
    thay vì chỉ hy vọng gradient tự nhiên thấp ở đó.

    [SỬA — LỖI PHÁT HIỆN QUA CODE REVIEW, vòng 2] Bản trước tính min-max
    chuẩn hoá (lo/hi) trên TOÀN BỘ canvas (kể cả viền đệm đen letterbox) RỒI
    MỚI gán vùng ngoài ROI = floor. Sai vì: viền đen/biên ảnh thường có
    gradient LỚN NHẤT trong toàn ảnh (tương phản đen-tuyệt-đối với ảnh thật)
    → `hi` bị viền đen "thổi" lên rất cao → khi chia (sal-lo)/(hi-lo), toàn
    bộ giá trị BÊN TRONG ROI (vốn nhỏ hơn hi nhiều) bị NÉN gần về 0 → sau
    remap [floor,1] gần như đồng nhất ~floor → loss_saliency suy biến gần
    như thành L1 đều theo không gian, MẤT đúng tín hiệu "identity-critical"
    mà cơ chế này nhắm tới. Sửa bằng cách: min-max PHẢI tính CHỈ trên các
    pixel BÊN TRONG ROI của từng sample (không đụng tới viền đệm đen), sau
    đó mới remap + gán cứng phần ngoài ROI = floor.
    HẠN CHẾ CÒN LẠI (chưa khắc phục, ghi nhận cho bài báo): đây vẫn là
    vanilla gradient saliency (Simonyan et al.), không phải Grad-CAM — có
    thể còn khá lan toả/nhiễu bên TRONG vùng ROI so với CAM dựa trên feature
    map tầng conv cuối. Hướng cải tiến tiếp theo nếu cần mượt hơn: Grad-CAM
    hoặc SmoothGrad (chi phí cao hơn — nhiều forward/backward hoặc cần chọn
    tầng conv cụ thể cho từng backbone).

    [SỬA — LỖI PHÁT HIỆN QUA CODE REVIEW, vòng 3, điểm "rất thấp"] Bản vòng 2
    blur (avg_pool2d) TRƯỚC rồi mới cắt ROI để tính min-max — nghĩa là vài
    pixel SÁT BIÊN ROI vẫn bị "rò" một phần giá trị viền đen/nền qua kernel
    blur trước khi min-max được tính (dù bug CHÍNH đã sửa ở vòng 2 — min-max
    không còn lấy `hi` từ viền đen NỮA, nhưng vài pixel biên vẫn lệch nhẹ do
    blur). Sửa: CHE (=0) toàn bộ vùng ngoài ROI của TỪNG SAMPLE TRƯỚC KHI
    blur, rồi mới blur + min-max trên ROI — đảm bảo blur chỉ trộn giá trị
    GIỮA CÁC PIXEL TRONG ROI với nhau (hoặc với 0 ở rìa ROI, giống zero-
    padding chuẩn của conv, KHÔNG PHẢI giá trị viền đen thật vốn có thể rất
    lớn). Ảnh hưởng nhỏ (chỉ ~blur_kernel//2 pixel viền ROI) nhưng làm sạch
    hoàn toàn, không còn phụ thuộc thứ tự blur/mask.

    [SỬA — LỖI PHÁT HIỆN QUA CODE REVIEW, vòng 3, điểm "thấp"] Nếu bbox có
    w<=0 hoặc h<=0 (ROI rỗng — hiếm với letterbox chuẩn nhưng KHÔNG được để
    crash toàn bộ batch training vì 1 sample lỗi dữ liệu), roi.min()/max()
    trên tensor rỗng sẽ ném RuntimeError. Sửa: phát hiện ROI rỗng, dùng
    lo=0/hi=1 an toàn (không NaN) — vì ROI rỗng nghĩa là bước gán `masked`
    bên dưới cũng sẽ là no-op (slice rỗng), nên giá trị lo/hi ở đây thực ra
    KHÔNG được dùng tới, chỉ cần không crash.

    KHÔNG lan truyền ngược qua CHÍNH phép tính saliency này (trả về đã
    .detach() — dùng làm trọng số CỐ ĐỊNH cho batch hiện tại). Gradient của
    judge lan truyền TỚI INPUT ảnh HR (không phải tới tham số judge — đã
    đóng băng, requires_grad=False, autograd vẫn lan truyền QUA được để tới
    input, chỉ không tích luỹ gradient VÀO tham số).

    Chi phí: thêm 1 lần forward+backward QUA TỪNG judge (chỉ khi
    lambda_saliency > 0) — không ảnh hưởng gì khi tắt (mặc định số liệu cũ
    không đổi)."""
    with torch.enable_grad(), torch.autocast(device_type=hr_img.device.type, enabled=False):
        hr_req = hr_img.clone().detach().float().requires_grad_(True)
        saliencies = []
        for judge in judges:
            identity_logits, _gender_logits, _emb = judge(hr_req)
            true_logit = identity_logits.gather(1, identity_label.view(-1, 1)).sum()
            grad = torch.autograd.grad(true_logit, hr_req)[0]
            sal = grad.abs().mean(dim=1, keepdim=True)  # (B,1,H,W) — gộp 3 kênh màu
            saliencies.append(sal)
        sal_avg = torch.stack(saliencies, dim=0).mean(dim=0)

        b = sal_avg.shape[0]

        # [SỬA — LỖI PHÁT HIỆN QUA CODE REVIEW, vòng 3] CHE (=0) vùng NGOÀI
        # ROI của TỪNG SAMPLE TRƯỚC KHI blur — nếu blur (avg_pool2d) TRƯỚC
        # rồi mới cắt ROI (thứ tự cũ ở vòng 2), vài pixel SÁT BIÊN ROI vẫn
        # trộn lẫn 1 phần giá trị viền đen/nền qua kernel blur, dù bug CHÍNH
        # (min-max lấy `hi` từ viền đen) đã sửa ở vòng 2. Che trước khi blur
        # đảm bảo blur chỉ trộn giá trị GIỮA CÁC PIXEL TRONG ROI (rìa ROI
        # trộn với 0, giống zero-padding chuẩn của conv) — sạch hơn.
        empty_roi = [False] * b
        boxes = [None] * b
        if bbox is not None:
            x0_b, y0_b, w_b, h_b = bbox
            H, W = sal_avg.shape[-2:]
            roi_mask = torch.zeros_like(sal_avg)
            for i in range(b):
                # Clamp ROI vào canvas — bbox lệch (splits.json không khớp
                # hr_size, hoặc toạ độ âm) nếu không clamp thì slice rỗng
                # nhưng empty_roi vẫn False -> roi.min() trên tensor rỗng crash.
                x0 = max(0, min(int(x0_b[i]), W))
                y0 = max(0, min(int(y0_b[i]), H))
                w = max(0, min(int(w_b[i]), W - x0))
                h = max(0, min(int(h_b[i]), H - y0))
                boxes[i] = (x0, y0, w, h)
                if w > 0 and h > 0:
                    roi_mask[i, :, y0:y0 + h, x0:x0 + w] = 1.0
                else:
                    # [SỬA — vòng 3, điểm "thấp"] ROI rỗng (w<=0 hoặc h<=0,
                    # hiếm gặp với letterbox chuẩn) — đánh dấu để dùng
                    # lo=0/hi=1 an toàn bên dưới, KHÔNG để roi.min()/max()
                    # trên slice rỗng ném RuntimeError làm sập cả batch.
                    empty_roi[i] = True
            sal_avg = sal_avg * roi_mask

        pad = blur_kernel // 2
        sal_avg = F.avg_pool2d(sal_avg, kernel_size=blur_kernel, stride=1, padding=pad)

        if bbox is not None:
            # [SỬA — vòng 2] min-max theo TỪNG SAMPLE chỉ trên pixel TRONG ROI
            # (không phải toàn canvas kể cả viền đen) — xem giải thích ở
            # docstring phía trên. Dùng vòng lặp theo batch vì mỗi sample có
            # ROI (x0,y0,w,h) khác nhau nên không thể vector hoá bằng slicing
            # đồng nhất; batch size của bước identity/saliency luôn nhỏ nên
            # chi phí vòng lặp Python này không đáng kể so với forward/backward
            # qua judge ở trên.
            lo = torch.empty(b, 1, 1, 1, device=sal_avg.device, dtype=sal_avg.dtype)
            hi = torch.empty(b, 1, 1, 1, device=sal_avg.device, dtype=sal_avg.dtype)
            for i in range(b):
                if empty_roi[i]:
                    lo[i] = 0.0
                    hi[i] = 1.0
                    continue
                x0, y0, w, h = boxes[i]
                roi = sal_avg[i, :, y0:y0 + h, x0:x0 + w]
                lo[i] = roi.min()
                hi[i] = roi.max()
        else:
            flat = sal_avg.view(b, -1)
            lo = flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
            hi = flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)

        sal_norm = (sal_avg - lo) / (hi - lo + 1e-8)

        # Neo về [floor, 1.0] — KHÔNG để trọng số chạm 0.0 tuyệt đối ở vùng
        # "ít quan trọng nhất" theo judge, tránh model học cách bỏ mặc hoàn
        # toàn 1 phần ảnh liên tục (rủi ro artefact ở biên vùng đó).
        sal_norm = floor + (1.0 - floor) * sal_norm

        if bbox is not None:
            # Giá trị sal_norm TÍNH BÊN NGOÀI ROI ở bước trên chỉ là "rác"
            # (bị chuẩn hoá theo lo/hi của ROI, không có ý nghĩa) — ép cứng
            # về floor như cũ, KHÔNG dùng giá trị đó.
            masked = torch.full_like(sal_norm, floor)
            for i in range(b):
                if empty_roi[i]:
                    continue
                x0, y0, w, h = boxes[i]
                masked[i, :, y0:y0 + h, x0:x0 + w] = sal_norm[i, :, y0:y0 + h, x0:x0 + w]
            sal_norm = masked
    return sal_norm.detach()


def compute_total_loss(student_out, hr_img, teacher_out, judges, cfg,
                        student_feat=None, teacher_feat=None, feat_adapter=None,
                        saliency_map=None):
    l1 = nn.L1Loss()

    loss_pixel = l1(student_out, hr_img)
    loss_distill = l1(student_out, teacher_out)

    # [MỚI] Saliency-weighted pixel loss — BỔ SUNG bên cạnh loss_pixel đồng
    # đều (không thay thế), xem compute_multi_judge_saliency() ở trên.
    if saliency_map is not None:
        loss_saliency = (saliency_map * (student_out - hr_img).abs()).mean()
    else:
        loss_saliency = torch.zeros((), device=student_out.device)

    # [MỚI] Feature-level distillation (hint-based KD) — BỔ SUNG bên cạnh
    # loss_distill (output-level), không thay thế. teacher_feat luôn được
    # trích trong lúc torch.no_grad() (xem _forward_step) nên đã KHÔNG có
    # đồ thị gradient sẵn — .detach() thêm ở đây chỉ để phòng vệ tường minh,
    # không đổi hành vi.
    #
    # [SỬA — LỖI NGHIÊM TRỌNG phát hiện qua code review] Teacher mặc định là
    # span_official: nhân img_range=255 NỘI BỘ (xem span_official_wrapper.py)
    # và KHÔNG rescale lại cho tới SAU pixel-shuffle — nghĩa là feature nội
    # bộ (trước conv chiếu cuối, nơi SRFeatureHook bắt) của teacher nằm ở
    # thang giá trị ~O(10^2), trong khi feature của student (SPAN tự viết,
    # không nhân img_range) nằm ở thang ~O(1). Nếu tính L1 trực tiếp trên 2
    # thang giá trị lệch nhau ~100 lần, loss_feat sẽ:
    #   (a) lúc đầu áp đảo loss_pixel/loss_distill (vốn chỉ ~0.02-0.05) dù
    #       lambda_feat vừa phải, làm méo hướng tối ưu ngay từ đầu training;
    #   (b) dồn hết việc "sửa scale" cho 1x1 adapter thay vì để nó học ĐÚNG
    #       vai trò của FitNets/hint-KD là ánh xạ KHÔNG GIAN ĐẶC TRƯNG.
    # Sửa bằng cách CHUẨN HOÁ (zero-mean, unit-std) riêng từng feature map
    # (student & teacher, tính riêng mỗi sample) NGAY TRƯỚC KHI ĐƯA VÀO L1 —
    # [SỬA lệch comment phát hiện qua code review, vòng 4] phía teacher (vốn
    # KHÔNG qua adapter, đã đúng số kênh teacher_ch sẵn) chuẩn hoá trực tiếp;
    # phía student, thứ tự THẬT trong code là adapter(student_feat) RỒI MỚI
    # chuẩn hoá (không phải chuẩn hoá TRƯỚC adapter như câu trên từng viết
    # nhầm) — KHÔNG đổi tính đúng đắn (vẫn loại bỏ được lệch scale trước khi
    # so L1, vì bước chuẩn hoá luôn là bước CUỐI trước L1 ở cả 2 phía), chỉ
    # là mô tả thứ tự bị viết sai trước đây. Kết quả: loại bỏ hoàn toàn ảnh
    # hưởng của quy ước thang giá trị nội bộ riêng của từng kiến trúc, chỉ
    # còn so khớp HÌNH DẠNG/CẤU TRÚC tương đối của feature map (đúng tinh
    # thần hint-based KD, không phụ thuộc implementation
    # detail như img_range của từng teacher cụ thể — tổng quát cho MỌI cặp
    # teacher/student có thể có trong project, không chỉ span_official).
    if feat_adapter is not None and student_feat is not None and teacher_feat is not None:
        def _instance_normalize(f):
            f = f.float()
            b = f.shape[0]
            flat = f.reshape(b, -1)
            mean = flat.mean(dim=1).view(b, 1, 1, 1)
            std = flat.std(dim=1).view(b, 1, 1, 1) + 1e-6
            return (f - mean) / std

        student_feat_n = _instance_normalize(feat_adapter(student_feat))
        teacher_feat_n = _instance_normalize(teacher_feat.detach())
        loss_feat = l1(student_feat_n, teacher_feat_n)
    else:
        loss_feat = torch.zeros((), device=student_out.device)

    # [SỬA — lỗi phát hiện qua code review] Trước đây LUÔN forward+backward
    # QUA TẤT CẢ judge để tính identity loss dù lambda_identity=0 — vừa tốn
    # chi phí vô ích, vừa rủi ro `0 * NaN = NaN`: nếu 1 judge cho NaN/Inf ở
    # 1 batch bất thường, NHÂN với lambda_identity=0 vẫn ra NaN (0*NaN=NaN
    # trong IEEE 754, không phải 0), làm hỏng CẢ total loss dù người dùng đã
    # tắt hẳn thành phần này. Giờ chỉ tính khi thực sự dùng đến (khớp đúng
    # cách compute_multi_judge_saliency/lambda_saliency đã làm ở _forward_step).
    ci = cfg["sr_improve"]
    lambda_identity = ci.get("lambda_identity", 0.0)
    if lambda_identity > 0 and judges:
        # Tính identity loss (cosine_similarity qua từng model giám khảo) ở fp32
        # TƯỜNG MINH, không để dưới autocast fp16 của khối bên ngoài —
        # cosine_similarity kém ổn định số học ở fp16 (dễ chia gần-0/gần-0 ra
        # NaN), đặc biệt khi chuỗi qua nhiều model liên tiếp. Đã quan sát thấy
        # gây NaN gần như toàn bộ batch khi để dưới autocast — ép fp32 riêng phần
        # này để khắc phục.
        with torch.autocast(device_type=student_out.device.type, enabled=False):
            student_out_f32 = student_out.float()
            hr_img_f32 = hr_img.float()
            identity_losses = []
            for judge in judges:
                with torch.no_grad():
                    hr_emb = judge.embed(hr_img_f32)
                student_emb = judge.embed(student_out_f32)
                identity_losses.append(
                    1 - F.cosine_similarity(student_emb, hr_emb, dim=1, eps=1e-4).mean())
            # [MỚI] Multi-judge: trung bình cosine loss qua TẤT CẢ judge — ép SR
            # phải làm hài lòng nhiều kiến trúc recognition khác nhau CÙNG LÚC,
            # thay vì tối ưu riêng theo "gu" của đúng 1 backbone (xem động lực ở
            # docstring đầu file). Với 1 judge duy nhất (fallback tương thích
            # ngược), phép trung bình này thu gọn về ĐÚNG công thức cũ.
            loss_identity = torch.stack(identity_losses).mean()
    else:
        loss_identity = torch.zeros((), device=student_out.device)

    total = (ci["lambda_pixel"] * loss_pixel +
             ci["lambda_distill"] * loss_distill +
             ci.get("lambda_feat", 0.0) * loss_feat +
             ci.get("lambda_saliency", 0.0) * loss_saliency +
             ci["lambda_identity"] * loss_identity)

    return total, {
        "pixel": loss_pixel.item(),
        "distill": loss_distill.item(),
        "feat": loss_feat.item() if torch.is_tensor(loss_feat) else float(loss_feat),
        "saliency": loss_saliency.item() if torch.is_tensor(loss_saliency) else float(loss_saliency),
        "identity": loss_identity.item(),
    }


def _move_all_to(device, student, teacher, judges, feat_adapter, optimizer):
    student.to(device)
    teacher.to(device)
    for judge in judges:
        judge.to(device)
    if feat_adapter is not None:
        feat_adapter.to(device)
    move_optimizer_state(optimizer, device)


def _forward_step(student, teacher, judges, lr_img, hr_img, identity_label, bbox, device,
                   cfg, optimizer, scaler, is_train,
                   student_hook=None, teacher_hook=None, feat_adapter=None):
    lr_img = lr_img.to(device, non_blocking=True)
    hr_img = hr_img.to(device, non_blocking=True)
    identity_label = identity_label.to(device, non_blocking=True)

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    # LƯU Ý: KHÔNG dùng AMP/autocast ở đây (khác với train_sr.py/
    # train_recognition.py). SPAN chính thức nhân giá trị nội bộ lên tới
    # img_range=255 (xem span_arch.py: x = (x-mean)*img_range), biên độ hoạt
    # động lớn dễ tràn số dưới fp16 — càng dễ tràn hơn khi chuỗi qua nhiều
    # model liên tiếp (student->teacher->judges) trong cùng 1 lần forward.
    # Đã quan sát thực tế: bật AMP ở đây làm ~100% batch NaN kể cả lúc
    # validation (không qua backward/GradScaler) -> lỗi tràn số ở forward,
    # không phải lỗi gradient. fp32 chậm hơn nhưng ổn định tuyệt đối.
    student_out = student(lr_img)
    with torch.no_grad():
        teacher_out = teacher(lr_img)

    # [MỚI] Feature hook đã tự động cập nhật .feat trong lúc gọi student()/
    # teacher() ở trên (forward_pre_hook đăng ký 1 lần lúc khởi tạo, xem
    # models/sr_models.py::SRFeatureHook) — chỉ cần đọc ra đây, không cần
    # gọi lại forward.
    student_feat = student_hook.feat if student_hook is not None else None
    teacher_feat = teacher_hook.feat if teacher_hook is not None else None

    # [MỚI] Chỉ tính saliency map (tốn 1 lần forward+backward/judge) khi thực
    # sự cần dùng — giữ chi phí = 0 khi lambda_saliency=0 (mặc định cũ không đổi).
    # bbox (ROI thật, loại viền đệm đen letterbox) chỉ có nếu dataset được
    # khởi tạo với return_bbox=True (xem main()) — None thì saliency bỏ qua
    # bước ép floor ngoài ROI.
    saliency_map = None
    if cfg["sr_improve"].get("lambda_saliency", 0.0) > 0 and judges:
        bbox_device = None
        if bbox is not None:
            bbox_device = tuple(t.to(device, non_blocking=True) for t in bbox)
        saliency_map = compute_multi_judge_saliency(judges, hr_img, identity_label, bbox=bbox_device)

    loss, parts = compute_total_loss(student_out, hr_img, teacher_out, judges, cfg,
                                      student_feat=student_feat, teacher_feat=teacher_feat,
                                      feat_adapter=feat_adapter, saliency_map=saliency_map)

    loss_val = loss.item()
    # [SỬA — bug NGHIÊM TRỌNG phát hiện qua review Q1] TRƯỚC ĐÂY backward()+
    # optimizer.step() chạy VÔ ĐIỀU KIỆN, kiểm tra NaN/Inf chỉ nằm ở
    # run_epoch() và CHỈ loại batch đó khỏi trung bình HIỂN THỊ — gradient
    # NaN/Inf đã ngấm vào trọng số TỪ TRƯỚC đó rồi (optimizer.step() đã chạy
    # xong). Với Adam, 1 lần cập nhật bằng gradient NaN làm nhiễm NaN vĩnh
    # viễn vào buffer m/v của đúng tham số đó (NaN lan qua mọi bước sau, vì
    # beta*NaN + (1-beta)*x = NaN với MỌI x) — hậu quả: model coi như hỏng từ
    # thời điểm đó, dù log chỉ nói "bỏ qua khi tính trung bình" (nghe như vô
    # hại). Không dùng AMP/GradScaler ở file này (xem lý do phía trên) nên
    # KHÔNG có lớp bảo vệ tự động nào như train_sr.py/train_recognition.py có
    # được trên CUDA (scaler.step() tự bỏ qua update khi phát hiện grad
    # không hữu hạn). Chặn ĐÚNG TRƯỚC backward()/optimizer.step() thay vì chỉ
    # lọc sau khi đã cập nhật xong.
    is_finite = loss_val == loss_val and loss_val not in (float("inf"), float("-inf"))
    if is_train and is_finite:
        loss.backward()
        params_to_clip = list(student.parameters())
        if feat_adapter is not None:
            params_to_clip += list(feat_adapter.parameters())
        torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
        optimizer.step()

    return loss_val, parts


def run_epoch(student, teacher, judges, loader, device_mgr, cfg,
              optimizer=None, scaler=None, logger=None,
              student_hook=None, teacher_hook=None, feat_adapter=None):
    is_train = optimizer is not None
    student.train() if is_train else student.eval()

    totals = {"pixel": 0.0, "distill": 0.0, "feat": 0.0, "saliency": 0.0, "identity": 0.0, "total": 0.0}
    n, nan_batches = 0, 0
    model_device = next(student.parameters()).device.type

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for lr_img, hr_img, bbox, identity_label in tqdm(loader, leave=False):
            device = device_mgr.current_device()

            if device != model_device:
                _move_all_to(device, student, teacher, judges, feat_adapter, optimizer)
                model_device = device

            try:
                loss_val, parts = _forward_step(
                    student, teacher, judges, lr_img, hr_img, identity_label, bbox, device,
                    cfg, optimizer, scaler if device == "cuda" else None, is_train,
                    student_hook=student_hook, teacher_hook=teacher_hook, feat_adapter=feat_adapter)
            except RuntimeError as e:
                if device != "cuda" or "out of memory" not in str(e).lower():
                    raise
                device_mgr.report_oom()
                device = "cpu"
                _move_all_to(device, student, teacher, judges, feat_adapter, optimizer)
                model_device = device
                loss_val, parts = _forward_step(
                    student, teacher, judges, lr_img, hr_img, identity_label, bbox, device,
                    cfg, optimizer, None, is_train,
                    student_hook=student_hook, teacher_hook=teacher_hook, feat_adapter=feat_adapter)

            if loss_val != loss_val or loss_val in (float("inf"), float("-inf")):
                nan_batches += 1
                continue

            for k, v in parts.items():
                totals[k] += v
            totals["total"] += loss_val
            n += 1

    if nan_batches > 0 and logger:
        logger.info(f"  (lưu ý: {nan_batches} batch có loss NaN/Inf, đã bỏ qua khi tính trung bình)")

    if n == 0:
        return {k: float("nan") for k in totals}
    return {k: v / n for k, v in totals.items()}


def _require_ckpt(path, what: str):
    """Fail-fast trước khi train hàng giờ, thay vì FileNotFoundError lúc torch.load."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {what}: {p}. Chạy bước pipeline tương ứng trước.")
    return p


def _make_hrlr_loaders(train_set, val_set, batch_size, num_workers=4, seed=42):
    """DataLoader an toàn: dataset rỗng + persistent_workers=True dễ treo/crash
    worker (đặc biệt macOS spawn). Báo lỗi ngay, không để job chạy rồi sập.

    [MỚI — phát hiện qua review Q1] worker_init_fn + generator cố định (seed):
    cudnn.deterministic (set_seed()) không đủ để tái lập tuyệt đối khi
    num_workers>0 — xem utils/seed.py::seed_worker/seeded_generator."""
    if len(train_set) == 0 or len(val_set) == 0:
        raise RuntimeError(
            f"Dataset rỗng: train={len(train_set)} val={len(val_set)}. "
            f"Kiểm tra splits/hr|lr/{{train,val}} và splits.json "
            f"(chạy data/prepare_splits.py nếu chưa có).")
    nw = num_workers
    kwargs = dict(
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        worker_init_fn=seed_worker if nw > 0 else None,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               generator=seeded_generator(seed), **kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader


def _validate_identity_labels(dataset, num_identities: int):
    """identity_label >= num_identities -> IndexError ở gather() giữa epoch."""
    if dataset.label_map is None:
        return
    max_idx = max(dataset.label_map.values())
    if max_idx >= num_identities:
        raise ValueError(
            f"identity_label lớn nhất={max_idx} >= num_identities={num_identities} "
            f"(configs/config.yaml). compute_multi_judge_saliency().gather() sẽ "
            f"IndexError giữa epoch. Chỉnh num_identities hoặc splits.json.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lambda_pixel", type=float, default=None,
                     help="ghi đè sr_improve.lambda_pixel trong config (dùng cho ablation)")
    ap.add_argument("--lambda_distill", type=float, default=None,
                     help="ghi đè sr_improve.lambda_distill trong config (dùng cho ablation)")
    ap.add_argument("--lambda_feat", type=float, default=None,
                     help="[MỚI] ghi đè sr_improve.lambda_feat (feature-level KD) trong config "
                          "(dùng cho ablation, xem pipeline/run_ablation_kd_v2.sh)")
    ap.add_argument("--lambda_saliency", type=float, default=None,
                     help="[MỚI] ghi đè sr_improve.lambda_saliency (saliency-weighted identity-"
                          "critical pixel loss, xem compute_multi_judge_saliency) trong config "
                          "(dùng cho ablation)")
    ap.add_argument("--lambda_identity", type=float, default=None,
                     help="ghi đè sr_improve.lambda_identity trong config (dùng cho ablation)")
    ap.add_argument("--min_delta", type=float, default=0.0,
                     help="[MỚI — 2026-08-25, sự cố thật] ngưỡng cải thiện tối thiểu trên "
                          "val_total để tính là 'tốt hơn' cho early-stopping (xem "
                          "utils/early_stopping.py::EarlyStopping). Mặc định 0.0 giữ NGUYÊN "
                          "hành vi cũ (bất kỳ cải thiện nào dù nhỏ cũng reset patience) — chỉ "
                          "cần đặt >0 khi loss có xu hướng giảm đơn điệu cực nhỏ dần không hồi "
                          "kết (quan sát thấy với lambda_saliency>0, khiến training chạy gần hết "
                          "max_epochs thay vì dừng sớm thật, xem pipeline/run_lambda_saliency_sweep.sh).")
    ap.add_argument("--student_arch", default=None,
                     help="ghi đè sr_improve.student_arch trong config (dùng cho ablation kiến "
                          "trúc, ví dụ --student_arch span_large)")
    ap.add_argument("--student_n_blocks", type=int, default=None,
                     help="[MỚI — Mục 5.7(i), ablation attention-parameterization] ghi đè số "
                          "khối (n_blocks) của student — CHỈ có tác dụng khi student_arch là "
                          "'safmn'/'smfanet' (build_sr_model() bỏ qua tham số này với các kiến "
                          "trúc khác). Dùng để train biến thể half-depth, ví dụ "
                          "--student_arch safmn --student_n_blocks 4 (mặc định SAFMN/SMFANet là "
                          "n_blocks=8, xem RUN_ALL_attention_param_ablation.sh). Không truyền "
                          "-> giữ nguyên hành vi cũ (n_blocks mặc định của kiến trúc).")
    ap.add_argument("--init_ckpt", default=None,
                     help="checkpoint SPAN student có sẵn để khởi tạo trọng số trước khi "
                          "train — dùng cho FINE-TUNE/TRANSFER LEARNING xuyên dataset (ví dụ "
                          "khởi tạo từ span_tiny đã train trên EarVN1.0 rồi fine-tune trên AWE). "
                          "Nạp strict=True (an toàn: model SR không có tầng phụ thuộc số lượng "
                          "identity/dataset, kiến trúc giống hệt nhau mọi dataset).")
    ap.add_argument("--run_suffix", default="",
                     help="hậu tố thêm vào tên thư mục runs/, tránh ghi đè checkpoint "
                          "khi chạy nhiều cấu hình ablation/fine-tune khác nhau")
    ap.add_argument("--seed", type=int, default=None,
                     help="ghi đè seed trong config — dùng để chạy multi-seed, đo độ ổn định")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["split"]["seed"] = args.seed

    set_seed(cfg["split"]["seed"])
    ci = cfg["sr_improve"]

    if args.lambda_pixel is not None:
        ci["lambda_pixel"] = args.lambda_pixel
    if args.lambda_distill is not None:
        ci["lambda_distill"] = args.lambda_distill
    if args.lambda_feat is not None:
        ci["lambda_feat"] = args.lambda_feat
    if args.lambda_saliency is not None:
        ci["lambda_saliency"] = args.lambda_saliency
    if args.lambda_identity is not None:
        ci["lambda_identity"] = args.lambda_identity
    if args.student_arch is not None:
        ci["student_arch"] = args.student_arch

    scale = cfg["image"]["scale"]
    splits_root = cfg["paths"]["splits_root"]
    splits_json = f"{splits_root}/splits.json"

    # [LƯU Ý — bổ sung sau code review, vòng 2, điểm "nhẹ"] return_bbox/
    # return_label=True LUÔN bật (kể cả khi lambda_saliency=0 và không dùng
    # identity_label) để _forward_step/run_epoch có chữ ký cố định (4 giá trị
    # mỗi batch) bất kể cấu hình lambda — chi phí thêm không đáng kể (chỉ đọc
    # thêm 2 số nhỏ mỗi sample, KHÔNG tốn thêm forward/backward nào, saliency
    # vẫn chỉ tính khi lambda_saliency>0, xem _forward_step). ĐÁNH ĐỔI: script
    # này giờ LUÔN yêu cầu splits.json tồn tại (trước bản vá multi-judge/
    # saliency thì KHÔNG cần) — với pipeline chuẩn của project (chạy
    # data/prepare_splits.py ở Bước 1) thì splits.json luôn có sẵn nên không
    # ảnh hưởng gì, nhưng báo lỗi rõ ràng ngay từ đầu nếu thiếu, thay vì để
    # HRLRPairDataset ném FileNotFoundError khó hiểu ở lượt đọc batch đầu tiên.
    if not Path(splits_json).exists():
        raise FileNotFoundError(
            f"Không tìm thấy {splits_json} — train_sr_distill.py (từ khi có multi-judge "
            f"saliency/identity_label) LUÔN cần file này để đọc bbox+identity_label cho mỗi "
            f"ảnh, kể cả khi bạn chỉ chạy pixel+distill thuần (lambda_saliency=lambda_identity=0). "
            f"Chạy `python data/prepare_splits.py` (Bước 1 pipeline) trước để sinh file này.")

    train_set = HRLRPairDataset(f"{splits_root}/hr", f"{splits_root}/lr", "train",
                                 return_bbox=True, return_label=True, splits_json=splits_json)
    val_set = HRLRPairDataset(f"{splits_root}/hr", f"{splits_root}/lr", "val",
                               return_bbox=True, return_label=True, splits_json=splits_json)
    _validate_identity_labels(train_set, cfg["num_identities"])
    _validate_identity_labels(val_set, cfg["num_identities"])
    train_loader, val_loader = _make_hrlr_loaders(train_set, val_set, ci["batch_size"],
                                                   seed=cfg["split"]["seed"])

    student_arch = ci.get("student_arch", cfg["sr"]["arch"])
    student_pretrained = ci.get("student_pretrained_path")
    if student_pretrained:
        _require_ckpt(student_pretrained, "student_pretrained_path")

    run_dir = Path(cfg["paths"]["runs_root"]) / f"sr_improved_{student_arch}{args.run_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir, name="train")
    lambda_feat = ci.get("lambda_feat", 0.0)
    lambda_saliency = ci.get("lambda_saliency", 0.0)
    logger.info(f"Student architecture: {student_arch} | Lambda: pixel={ci['lambda_pixel']} "
                f"distill={ci['lambda_distill']} feat={lambda_feat} saliency={lambda_saliency} "
                f"identity={ci['lambda_identity']} | early_stop_min_delta={args.min_delta}")
    device_mgr = DeviceManager(logger=logger)
    device = device_mgr.preferred
    logger.info(f"=== Bắt đầu cải tiến SPAN (distillation + feature-KD + saliency-weighted loss "
                f"+ identity-aware loss) ===")
    logger.info(f"Device ưu tiên: {device}")
    logger.info(f"Teacher: {ci['teacher_arch']} (output-level distill"
                f"{' + feature-level KD' if lambda_feat > 0 else ''}"
                f"{' + saliency-weighted identity-critical loss' if lambda_saliency > 0 else ''})")

    # --- Student: kiến trúc NÉN (student_arch, ví dụ span_tiny) — model sẽ được deploy ---
    # [MỚI — Mục 5.7(i)] truyền student_n_blocks xuống build_sr_model(); hàm
    # đó tự bỏ qua tham số này với mọi arch khác "safmn"/"smfanet" (an toàn
    # ngược, không đổi hành vi bất kỳ lời gọi cũ nào không dùng cờ mới).
    student = build_sr_model(student_arch, scale, pretrained_path=student_pretrained,
                              n_blocks=args.student_n_blocks).to(device)

    # Khởi tạo student từ checkpoint có sẵn — dùng cho fine-tune/transfer learning.
    # An toàn strict=True vì SPAN/span_tiny/span_large không có tầng nào phụ thuộc
    # num_identities hay bất kỳ thông tin riêng của dataset — kiến trúc giống hệt
    # nhau dù train trên dataset nào, nên không có rủi ro lệch shape.
    if args.init_ckpt:
        _require_ckpt(args.init_ckpt, "checkpoint khởi tạo student (--init_ckpt)")
        student.load_state_dict(torch.load(args.init_ckpt, map_location=device))
        logger.info(f"[TRANSFER LEARNING] Khởi tạo student từ checkpoint có sẵn: {args.init_ckpt} "
                    f"(nạp toàn bộ trọng số, strict=True — an toàn vì không có tầng phụ thuộc dataset)")

    # --- Teacher: SPAN baseline (đã chứng minh chất lượng tốt), ĐÓNG BĂNG ---
    _require_ckpt(ci["teacher_ckpt"], f"teacher ({ci['teacher_arch']})")
    teacher = build_sr_model(ci["teacher_arch"], scale).to(device)
    teacher.load_state_dict(torch.load(ci["teacher_ckpt"], map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # --- [MỚI] Hội đồng recognition model giám khảo (multi-judge), ĐÓNG BĂNG ---
    # [SỬA — lỗi phát hiện qua code review] Trước đây LUÔN load đủ 3 checkpoint
    # judge (mobilenet_v2/resnet18/ghostnet_100 theo config) kể cả khi
    # lambda_identity=0 VÀ lambda_saliency=0 (không dùng đến judge nào cả) —
    # vừa tốn thời gian/VRAM vô ích, vừa CRASH nếu người dùng chỉ có sẵn
    # checkpoint recognition của 1 backbone (ví dụ chạy pixel+distill thuần
    # trên máy chưa train xong đủ 3 recognition backbone). Giờ chỉ load khi
    # thực sự có cơ chế nào cần đến judge.
    needs_judges = ci.get("lambda_identity", 0.0) > 0 or lambda_saliency > 0
    if needs_judges:
        judges, judge_specs = build_judges(cfg, device)
        judge_desc = ", ".join(f"{s['backbone']} ({s['ckpt']})" for s in judge_specs)
        logger.info(f"Recognition giám khảo ({len(judges)}): {judge_desc}")
    else:
        judges, judge_specs = [], []
        logger.info("Recognition giám khảo: KHÔNG tải (lambda_identity=0 và lambda_saliency=0, "
                    "không cần checkpoint recognition nào).")

    # --- [MỚI] Feature-level distillation setup (chỉ bật nếu lambda_feat > 0) ---
    student_hook, teacher_hook, feat_adapter, extra_params = _setup_feat_kd(
        student, teacher, lambda_feat, cfg, scale, device, logger)

    optimizer = torch.optim.Adam(list(student.parameters()) + extra_params, lr=ci["lr"])
    # KHÔNG dùng GradScaler/AMP cho script này (xem ghi chú trong _forward_step
    # về lý do img_range=255 của SPAN dễ tràn số dưới fp16 khi chuỗi nhiều model).

    max_epochs = ci["max_epochs"]
    patience = ci["patience"]
    stopper = EarlyStopping(patience=patience, mode="min", min_delta=args.min_delta)  # total loss: càng thấp càng tốt

    for epoch in range(max_epochs):
        train_stats = run_epoch(student, teacher, judges, train_loader,
                                 device_mgr, cfg, optimizer=optimizer, scaler=None, logger=logger,
                                 student_hook=student_hook, teacher_hook=teacher_hook, feat_adapter=feat_adapter)
        val_stats = run_epoch(student, teacher, judges, val_loader,
                               device_mgr, cfg, optimizer=None, scaler=None, logger=logger,
                               student_hook=student_hook, teacher_hook=teacher_hook, feat_adapter=feat_adapter)

        oom_note = f" | OOM fallback: {device_mgr.total_oom_events} lần" \
            if device_mgr.total_oom_events > 0 else ""
        logger.info(
            f"[sr_improved] epoch {epoch + 1}/{max_epochs} "
            f"(early-stop counter: {stopper.counter}/{patience}){oom_note} | "
            f"train_total={train_stats['total']:.4f} pixel={train_stats['pixel']:.4f} "
            f"distill={train_stats['distill']:.4f} feat={train_stats['feat']:.4f} "
            f"saliency={train_stats['saliency']:.4f} identity={train_stats['identity']:.4f} | "
            f"VAL_TOTAL={val_stats['total']:.4f}"
        )

        is_best = stopper.step(val_stats["total"])
        if is_best:
            _save_state_dict(student, run_dir / "best.pt")
            logger.info(f"  -> checkpoint tốt nhất mới (val_total={val_stats['total']:.4f}), đã lưu.")

        if stopper.should_stop:
            logger.info(
                f"EARLY STOPPING tại epoch {epoch + 1}: val_total không cải thiện "
                f"sau {patience} epoch liên tiếp. Best val_total={stopper.best_str}"
            )
            break

    if student_hook is not None:
        student_hook.remove()
    if teacher_hook is not None:
        teacher_hook.remove()

    _save_last_if_missing(run_dir / "best.pt", student, logger, "best.pt")

    logger.info(f"=== Hoàn tất cải tiến SPAN. Best val_total={stopper.best_str}. "
                f"Tổng số lần fallback CPU do OOM: {device_mgr.total_oom_events}. "
                f"Checkpoint: {run_dir / 'best.pt'} ===")


if __name__ == "__main__":
    main()
