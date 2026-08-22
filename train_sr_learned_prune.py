"""
[MỚI] Học pruning độ sâu CÓ GIÁM SÁT (differentiable/learned block pruning)
cho SPAN — thay vì chọn TAY "giữ khối 1-3, bỏ khối 4-6" như span_tiny hiện
tại, để CHÍNH quá trình huấn luyện quyết định khối SPAB nào đáng giữ, dựa
trên toàn bộ loss downstream (pixel + output/feature distillation +
saliency-weighted identity-critical loss + multi-judge identity loss — TÁI
SỬ DỤNG NGUYÊN VẸN compute_total_loss()/build_judges()/
compute_multi_judge_saliency() từ train_sr_distill.py, chỉ thêm 2 thành phần
mới: sparsity penalty + binary/polarization penalty trên gate).

ĐỘNG LỰC (xem thêm RUNBOOK_EarVN1.0.md mục 13.2):
Bản thảo bài báo tự nhận việc chọn "giữ 3 khối đầu" của span_tiny là "a
choice made for implementation convenience rather than validated against
alternatives" (mục 5.7-ii) — và đây cũng là nguồn gốc hợp lý nhất của hiện
tượng "span_tiny không đồng nhất giữa các backbone/dataset": cắt cố định
theo VỊ TRÍ, không theo MỨC ĐỘ ĐÓNG GÓP thực tế của từng khối cho tác vụ
nhận dạng, nên khối bị cắt có thể quan trọng với backbone/dataset này nhưng
không quan trọng với backbone/dataset khác (hoặc ngược lại). Học pruning
thay thế lựa chọn tay này bằng 1 cơ chế NGUYÊN TẮC, tái tạo được, và (quan
trọng nhất cho novelty) CHƯA từng có trong bất kỳ công trình nén SPAN nào
được trích trong Related Work của bài báo.

CÔNG THỨC (xem đầy đủ trong models/sr_models.py::SPANLearnedPrune):
    gate_i = sigmoid(alpha_i)   (alpha_i: 1 scalar học được / khối)
    O_i = O_{i-1} + gate_i * (SPAB_i(O_{i-1}) - O_{i-1})
Loss = compute_total_loss(...) [pixel/distill/feat/saliency/identity, Y HỆT
train_sr_distill.py] + lambda_sparsity * mean(gate)
     + lambda_binary * mean(gate * (1-gate)).

QUY TRÌNH (ĐÃ SỬA sau code review — thêm bước 3 fine-tune, tách bước 5 chọn
checkpoint theo quality-loss thay vì loss có sparsity):
  (1) Train SPANLearnedPrune (ngân sách ĐẦY ĐỦ n_blocks_budget=6, bằng
      span_large) tới khi early-stop — CHỌN checkpoint "gated" tốt nhất theo
      quality_total (pixel+distill+feat+saliency+identity, KHÔNG gồm
      sparsity/binary) — [SỬA lỗi phát hiện qua review] trước đây chọn theo
      loss CÓ sparsity trộn vào, khiến checkpoint có thể "thắng" chỉ vì gate
      giảm (ít khối hơn) chứ không phải vì chất lượng SR tốt hơn.
  (2) "Cứng hoá" bằng harden_and_export() — khối có gate < gate_harden_threshold
      bị XOÁ HẲN, trả về model SPAN THƯỜNG (feat=48, n_blocks=số khối còn lại).
  (3) [MỚI] Fine-tune model đã cứng hoá thêm finetune_epochs_after_harden
      epoch (TÁI SỬ DỤNG train_sr_distill.py::run_epoch — không sparsity/gate
      nữa vì model export ra là SPAN thường) — [SỬA lỗi "thiếu fine-tune sau
      harden" phát hiện qua review] lúc train, khối được giữ chạy ở gate mềm
      (ví dụ 0.85), lúc export chạy ở gate=1.0 CỨNG — có mismatch, cần
      fine-tune ngắn để các khối còn lại thích nghi lại, đúng thông lệ chuẩn
      của pruning (prune rồi fine-tune, không chỉ prune suông).
  (4) [MỚI] Đánh giá lại (validation) model SAU KHI fine-tune — checkpoint
      cuối cùng lưu ra best.pt LÀ checkpoint đã fine-tune, không phải bản
      hardened thô — [SỬA lỗi "chưa eval lại sau harden" phát hiện qua review].
  (5) Lưu checkpoint cuối (tương thích build_sr_model("span_pruned", scale,
      n_blocks=K), xem models/sr_models.py) + prune_metadata.json (khối nào
      giữ/bỏ, gate cuối, quality loss trước/sau fine-tune — minh bạch cho bài
      báo, có thể vẽ Figure "learned pruning pattern").

Chạy:
    python train_sr_learned_prune.py --config configs/config.yaml

Chạy với sparsity tùy chỉnh (dùng cho ablation/sweep):
    python train_sr_learned_prune.py --config configs/config.yaml \
        --lambda_sparsity 0.1 --run_suffix _sparsity0.1

Sau khi train xong, dùng checkpoint đã cứng hoá+fine-tune (runs/sr_learned_prune*/best.pt)
y hệt cách dùng span_tiny ở các bước sau (data/build_sr.py, eval_sr_quality.py,
train_recognition.py, eval_recognition.py) — CHỈ khác: phải thêm
`--arch span_pruned --n_blocks K` (K in kept_indices ghi trong
prune_metadata.json) thay vì `--arch span_tiny`.

LƯU Ý SO SÁNH CÔNG BẰNG [bổ sung sau code review]: model xuất ra dùng feat=48
(giống span_tiny) nhưng SỐ KHỐI (K) do training tự quyết định. CHỈ khi K=3
(đúng bằng span_tiny) thì so sánh params/accuracy trực tiếp với span_tiny mới
công bằng (cùng kiến trúc). Nếu K khác 3, đây là 1 kiến trúc KHÁC kích thước
— main() bên dưới tự in cảnh báo rõ ràng theo giá trị K thực tế mỗi lần chạy.

LƯU Ý "IDENTITY-AWARE" [bổ sung sau code review, vòng 2; SỬA LẠI ở vòng 5 —
LỖI GỘP NHẦM lambda_feat vào "identity-aware"]: 3 lambda tùy chọn của
compute_total_loss() KHÔNG cùng bản chất, dù cả 3 đều có thể >0 độc lập:
  - lambda_feat (feature-level hint-KD): so khớp feature NỘI BỘ của SR
    TEACHER (span_official) — một model super-resolution THUẦN TÚY, KHÔNG hề
    liên quan gì tới nhận dạng/identity. Bật lambda_feat chỉ giúp gate "biết"
    khối nào cần giữ để BẮT CHƯỚC TỐT TEACHER SR hơn — đây là tín hiệu
    reconstruction/distillation, KHÔNG PHẢI tín hiệu nhận dạng.
  - lambda_saliency / lambda_identity: MỚI thực sự "identity-aware" — cả 2
    đều suy trực tiếp từ (các) recognition judge (gradient của logit identity
    thật, hoặc cosine loss giữa embedding SR/HR) — đây mới là tín hiệu khiến
    gate "biết" khối nào quan trọng cho NHẬN DẠNG.
Do đó: pruning này CHỈ thật sự "identity-aware" (đúng tên gọi) khi ít nhất 1
trong lambda_saliency/lambda_identity > 0 — bật RIÊNG lambda_feat (để 2 cái
kia = 0) vẫn CHỈ là "reconstruction-aware pruning" (mạnh hơn phiên bản chỉ có
pixel+distill, vì có thêm tín hiệu hint-KD, nhưng vẫn KHÔNG dùng tín hiệu
nhận dạng nào) — KHÔNG được ghi "identity-aware pruning" vào bài báo nếu chỉ
bật lambda_feat. main() tự in CẢNH BÁO phân biệt rõ 2 trường hợp này, và
prune_metadata.json ghi cờ "identity_aware": true/false CHỈ dựa trên
lambda_saliency/lambda_identity (KHÔNG tính lambda_feat) để không báo cáo
nhầm trong bài báo — đồng thời có thêm cờ riêng "uses_feature_kd" cho
lambda_feat để phân biệt tường minh 2 trục "dùng feature-KD" và "dùng tín
hiệu nhận dạng" (độc lập nhau, không loại trừ nhau).
pipeline/run_prune_sparsity_screen.sh và pipeline/run_multi_seed_learned_prune.sh
đã được cập nhật để PIN tường minh recipe KD-v2 đã thắng (SỬA biến
LAMBDA_FEAT/LAMBDA_SALIENCY/LAMBDA_IDENTITY ở đầu 2 script đó theo recipe của
bạn trước khi chạy) — LƯU Ý: nếu recipe KD-v2 thắng của bạn CHỈ có
lambda_feat>0 (saliency/identity=0), pin đúng giá trị đó vẫn KHÔNG biến pruning
thành "identity-aware", chỉ là "reconstruction-aware (có feature-KD)".

CÁC SỬA KHÁC sau code review vòng 2 (xem lịch sử/comment inline ở nơi liên
quan để biết chi tiết):
  - run_epoch_prune(): sửa lỗi cộng "total" 2 lần (không đổi checkpoint được
    chọn vì early-stop dùng "quality_total", chỉ sửa số LOG/metadata).
  - Fine-tune sau harden: COPY trọng số feat_adapter đã học từ pha gated
    (KHÔNG random init lại) + hạ LR xuống finetune_lr_scale x LR gốc (mặc
    định 0.1x, đúng thông lệ pruning chuẩn).
  - Đánh giá cuối cùng: KHÔNG còn báo cứng feat=0 — tái dùng adapter đã
    fine-tune (hoặc adapter pha gated nếu bỏ qua fine-tune) để đo feat loss
    có ý nghĩa thay vì một con số luôn sai.

CÁC SỬA KHÁC sau code review vòng 3:
  - [Nghiêm trọng nhất trong đợt này] `feat_adapter`/`ft_feat_adapter` giờ
    được CHỤP LẠI (state_dict, trong bộ nhớ) đúng lúc `is_best` ở CẢ 2 pha
    (gated và fine-tune) — trước đây chỉ student/exported được lưu theo
    is_best, còn adapter tiếp tục trôi tới epoch cuối cùng (sau patience),
    gây lệch epoch giữa model và adapter dùng để đo feat loss.
  - `finetune_lr` giờ tính từ `lp["lr"]` (sr_learned_prune.lr — LR THẬT của
    chính script này) thay vì `ci["lr"]` (sr_improve.lr — thuộc
    train_sr_distill.py, chỉ tình cờ đang cùng giá trị).
  - `compute_multi_judge_saliency`: che (=0) vùng ngoài ROI TRƯỚC khi blur
    (không phải blur cả canvas rồi mới cắt ROI) — tránh rò rỉ nhẹ giá trị
    viền đen vào vài pixel biên ROI qua kernel blur; đồng thời guard ROI
    rỗng (w<=0/h<=0) để không crash.
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from datasets.hrlr_pair_dataset import HRLRPairDataset
from models.sr_models import build_sr_model, SPANLearnedPrune
from train_sr_distill import (build_judges, compute_multi_judge_saliency,
                               compute_total_loss, run_epoch, _move_all_to,
                               _setup_feat_kd, _make_hrlr_loaders,
                               _validate_identity_labels, _require_ckpt)
from utils.device_manager import DeviceManager
from utils.early_stopping import (EarlyStopping, save_state_dict as _save_state_dict,
                                   save_last_if_missing as _save_last_if_missing)
from utils.logger import setup_logger
from utils.seed import set_seed


def _forward_step_prune(student, teacher, judges, lr_img, hr_img, identity_label, bbox,
                         device, cfg, lambda_sparsity, lambda_binary, optimizer, is_train,
                         student_hook=None, teacher_hook=None, feat_adapter=None):
    lr_img = lr_img.to(device, non_blocking=True)
    hr_img = hr_img.to(device, non_blocking=True)
    identity_label = identity_label.to(device, non_blocking=True)

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    # KHÔNG dùng AMP/autocast — xem lý do chi tiết trong
    # train_sr_distill.py::_forward_step (img_range=255 của SPAN dễ tràn số fp16).
    student_out = student(lr_img)
    with torch.no_grad():
        teacher_out = teacher(lr_img)

    student_feat = student_hook.feat if student_hook is not None else None
    teacher_feat = teacher_hook.feat if teacher_hook is not None else None

    saliency_map = None
    if cfg["sr_improve"].get("lambda_saliency", 0.0) > 0 and judges:
        bbox_device = None
        if bbox is not None:
            bbox_device = tuple(t.to(device, non_blocking=True) for t in bbox)
        saliency_map = compute_multi_judge_saliency(judges, hr_img, identity_label, bbox=bbox_device)

    loss_quality, parts = compute_total_loss(student_out, hr_img, teacher_out, judges, cfg,
                                              student_feat=student_feat, teacher_feat=teacher_feat,
                                              feat_adapter=feat_adapter, saliency_map=saliency_map)

    # [MỚI] Sparsity penalty — khuyến khích gate trung bình giảm dần.
    # [MỚI — sửa lỗi phát hiện qua review] Binary/polarization penalty — khuyến
    # khích TỪNG gate tụ về 0 HOẶC 1 (cực đại hoá g*(1-g) chính là g=0.5, cực
    # tiểu ở 2 đầu mút) thay vì lơ lửng quanh 0.5 — giảm mismatch giữa hành vi
    # train (gate mềm) và deploy (gate cứng 0/1 sau harden_and_export()).
    gates = torch.sigmoid(student.gate_logits)
    loss_sparsity = gates.mean()
    loss_binary = (gates * (1 - gates)).mean()
    loss_train = loss_quality + lambda_sparsity * loss_sparsity + lambda_binary * loss_binary

    # [SỬA — lỗi phát hiện qua review] Tách RÕ 2 chỉ số:
    #   - "total": loss THẬT dùng để backward (gồm sparsity+binary) — chỉ để LOG.
    #   - "quality_total": loss KHÔNG gồm sparsity/binary — DUY NHẤT chỉ số
    #     dùng để early-stop/chọn checkpoint (xem main()). Trước đây dùng
    #     "total" (có sparsity) để chọn checkpoint -> 1 checkpoint có thể
    #     "thắng" chỉ vì gate sập xuống (giảm sparsity penalty) dù chất lượng
    #     SR/identity thực tế XẤU ĐI — sai mục tiêu chọn model.
    parts["sparsity"] = loss_sparsity.item()
    parts["binary"] = loss_binary.item()
    parts["quality_total"] = loss_quality.item()
    # [SỬA — LỖI PHÁT HIỆN QUA CODE REVIEW, vòng 2] KHÔNG nhét "total" vào
    # `parts` ở đây — run_epoch_prune() bên dưới ĐÃ TỰ CỘNG DỒN
    # `totals["total"] += loss_val` (loss_val chính là loss_train.item() trả
    # về ở cuối hàm này) NGAY SAU vòng lặp `for k, v in parts.items():
    # totals[k] += v`. Nếu "total" cũng có mặt trong `parts`, giá trị đó bị
    # CỘNG 2 LẦN (1 lần qua vòng lặp parts, 1 lần qua dòng totals["total"] +=
    # loss_val riêng) -> val_total_với_sparsity bị GẤP ĐÔI giá trị thật trong
    # log/metadata (early-stopping/chọn checkpoint KHÔNG bị sai vì chúng dùng
    # "quality_total", không dùng "total" — nhưng số log/metadata gây hiểu
    # nhầm khi đọc lại để chỉnh lambda_sparsity/lambda_binary).

    loss_val = loss_train.item()
    # [SỬA — bug NGHIÊM TRỌNG phát hiện qua review Q1, cùng lỗi với
    # train_sr_distill.py::_forward_step] TRƯỚC ĐÂY backward()+optimizer.step()
    # chạy VÔ ĐIỀU KIỆN, kiểm tra NaN/Inf chỉ ở run_epoch_prune() và CHỈ loại
    # batch đó khỏi trung bình hiển thị — gradient NaN/Inf đã ngấm vào trọng
    # số (và cả gate_logits) TỪ TRƯỚC đó. Không dùng AMP ở file này nên không
    # có lớp bảo vệ tự động như GradScaler. Chặn ĐÚNG TRƯỚC backward().
    is_finite = loss_val == loss_val and loss_val not in (float("inf"), float("-inf"))
    if is_train and is_finite:
        loss_train.backward()
        params_to_clip = list(student.parameters())
        if feat_adapter is not None:
            params_to_clip += list(feat_adapter.parameters())
        torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
        optimizer.step()

    return loss_val, parts


def run_epoch_prune(student, teacher, judges, loader, device_mgr, cfg,
                     lambda_sparsity, lambda_binary, optimizer=None, logger=None,
                     student_hook=None, teacher_hook=None, feat_adapter=None):
    is_train = optimizer is not None
    student.train() if is_train else student.eval()

    totals = {"pixel": 0.0, "distill": 0.0, "feat": 0.0, "saliency": 0.0,
              "identity": 0.0, "sparsity": 0.0, "binary": 0.0,
              "quality_total": 0.0, "total": 0.0}
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
                loss_val, parts = _forward_step_prune(
                    student, teacher, judges, lr_img, hr_img, identity_label, bbox,
                    device, cfg, lambda_sparsity, lambda_binary, optimizer, is_train,
                    student_hook=student_hook, teacher_hook=teacher_hook, feat_adapter=feat_adapter)
            except RuntimeError as e:
                if device != "cuda" or "out of memory" not in str(e).lower():
                    raise
                device_mgr.report_oom()
                device = "cpu"
                _move_all_to(device, student, teacher, judges, feat_adapter, optimizer)
                model_device = device
                loss_val, parts = _forward_step_prune(
                    student, teacher, judges, lr_img, hr_img, identity_label, bbox,
                    device, cfg, lambda_sparsity, lambda_binary, optimizer, is_train,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lambda_pixel", type=float, default=None)
    ap.add_argument("--lambda_distill", type=float, default=None)
    ap.add_argument("--lambda_feat", type=float, default=None)
    ap.add_argument("--lambda_saliency", type=float, default=None)
    ap.add_argument("--lambda_identity", type=float, default=None)
    ap.add_argument("--lambda_sparsity", type=float, default=None,
                     help="[MỚI] ghi đè sr_learned_prune.lambda_sparsity trong config")
    ap.add_argument("--lambda_binary", type=float, default=None,
                     help="[MỚI] ghi đè sr_learned_prune.lambda_binary (regularizer phân cực "
                          "hoá gate) trong config")
    ap.add_argument("--gate_harden_threshold", type=float, default=None,
                     help="[MỚI] ghi đè sr_learned_prune.gate_harden_threshold trong config")
    ap.add_argument("--finetune_epochs_after_harden", type=int, default=None,
                     help="[MỚI] ghi đè sr_learned_prune.finetune_epochs_after_harden trong config "
                          "(0 để tắt hẳn bước fine-tune sau harden — không khuyến khích)")
    ap.add_argument("--finetune_lr_scale", type=float, default=None,
                     help="[MỚI — vòng 2] ghi đè sr_learned_prune.finetune_lr_scale (hệ số nhân "
                          "LR gốc dùng cho pha fine-tune sau harden, mặc định 0.1 = hạ 10x)")
    ap.add_argument("--run_suffix", default="")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["split"]["seed"] = args.seed
    set_seed(cfg["split"]["seed"])

    ci = cfg["sr_improve"]
    lp = cfg["sr_learned_prune"]

    for name, val in [("lambda_pixel", args.lambda_pixel), ("lambda_distill", args.lambda_distill),
                       ("lambda_feat", args.lambda_feat), ("lambda_saliency", args.lambda_saliency),
                       ("lambda_identity", args.lambda_identity)]:
        if val is not None:
            ci[name] = val
    if args.lambda_sparsity is not None:
        lp["lambda_sparsity"] = args.lambda_sparsity
    if args.lambda_binary is not None:
        lp["lambda_binary"] = args.lambda_binary
    if args.gate_harden_threshold is not None:
        lp["gate_harden_threshold"] = args.gate_harden_threshold
    if args.finetune_epochs_after_harden is not None:
        lp["finetune_epochs_after_harden"] = args.finetune_epochs_after_harden
    if args.finetune_lr_scale is not None:
        lp["finetune_lr_scale"] = args.finetune_lr_scale

    scale = cfg["image"]["scale"]
    splits_root = cfg["paths"]["splits_root"]
    splits_json = f"{splits_root}/splits.json"

    # Xem chú thích tương tự trong train_sr_distill.py::main() — script này
    # cũng LUÔN cần splits.json (đọc bbox+identity_label) dù lambda_saliency/
    # lambda_identity=0, để chữ ký _forward_step_prune/run_epoch_prune cố định.
    if not Path(splits_json).exists():
        raise FileNotFoundError(
            f"Không tìm thấy {splits_json} — train_sr_learned_prune.py LUÔN cần file này "
            f"để đọc bbox+identity_label cho mỗi ảnh. Chạy `python data/prepare_splits.py` "
            f"(Bước 1 pipeline) trước để sinh file này.")

    train_set = HRLRPairDataset(f"{splits_root}/hr", f"{splits_root}/lr", "train",
                                 return_bbox=True, return_label=True, splits_json=splits_json)
    val_set = HRLRPairDataset(f"{splits_root}/hr", f"{splits_root}/lr", "val",
                               return_bbox=True, return_label=True, splits_json=splits_json)
    _validate_identity_labels(train_set, cfg["num_identities"])
    _validate_identity_labels(val_set, cfg["num_identities"])
    train_loader, val_loader = _make_hrlr_loaders(train_set, val_set, lp["batch_size"],
                                                   seed=cfg["split"]["seed"])

    run_dir = Path(cfg["paths"]["runs_root"]) / f"sr_learned_prune{args.run_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir, name="train")
    lambda_feat = ci.get("lambda_feat", 0.0)
    lambda_saliency = ci.get("lambda_saliency", 0.0)
    lambda_sparsity = lp["lambda_sparsity"]
    lambda_binary = lp.get("lambda_binary", 0.0)
    finetune_epochs = lp.get("finetune_epochs_after_harden", 0)
    logger.info("=== [MỚI] Học pruning độ sâu SPAN (learned/differentiable block pruning) ===")
    logger.info(f"n_blocks_budget={lp['n_blocks_budget']} gate_init={lp['gate_init']} "
                f"gate_harden_threshold={lp['gate_harden_threshold']}")
    logger.info(f"Lambda: pixel={ci['lambda_pixel']} distill={ci['lambda_distill']} "
                f"feat={lambda_feat} saliency={lambda_saliency} identity={ci['lambda_identity']} "
                f"sparsity={lambda_sparsity} binary={lambda_binary} | "
                f"finetune_epochs_after_harden={finetune_epochs}")

    device_mgr = DeviceManager(logger=logger)
    device = device_mgr.preferred
    logger.info(f"Device ưu tiên: {device}")

    # --- Student: SPAN với gate học được, ngân sách ĐẦY ĐỦ ban đầu ---
    student = SPANLearnedPrune(scale=scale, feat=48, n_blocks=lp["n_blocks_budget"],
                                gate_init=lp["gate_init"]).to(device)

    # --- Teacher: SPAN baseline (đã chứng minh chất lượng tốt), ĐÓNG BĂNG ---
    _require_ckpt(ci["teacher_ckpt"], f"teacher ({ci['teacher_arch']})")
    teacher = build_sr_model(ci["teacher_arch"], scale).to(device)
    teacher.load_state_dict(torch.load(ci["teacher_ckpt"], map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # --- Hội đồng recognition model giám khảo (multi-judge), ĐÓNG BĂNG ---
    # [SỬA — lỗi phát hiện qua review] chỉ load judge khi thực sự cần (xem
    # train_sr_distill.py::main() — cùng lý do: tránh tốn thời gian/VRAM và
    # tránh crash nếu thiếu checkpoint recognition không dùng đến).
    needs_judges = ci.get("lambda_identity", 0.0) > 0 or lambda_saliency > 0
    if needs_judges:
        judges, judge_specs = build_judges(cfg, device)
        judge_desc = ", ".join(f"{s['backbone']} ({s['ckpt']})" for s in judge_specs)
        logger.info(f"Recognition giám khảo ({len(judges)}): {judge_desc}")
    else:
        judges, judge_specs = [], []
        logger.info("Recognition giám khảo: KHÔNG tải (lambda_identity=0 và lambda_saliency=0).")

    # [SỬA — LỖI GỘP NHẦM phát hiện qua code review, vòng 5] Bản trước coi
    # lambda_feat NGANG HÀNG với lambda_saliency/lambda_identity khi quyết định
    # "identity-aware" — SAI: lambda_feat (feature-KD) chỉ so khớp feature nội
    # bộ của SR TEACHER (span_official, không liên quan nhận dạng), KHÔNG dùng
    # bất kỳ recognition judge nào (xem needs_judges ở trên — hoàn toàn không
    # phụ thuộc lambda_feat). Cờ "identity-aware" ĐÚNG NGHĨA chỉ nên dựa trên
    # lambda_saliency/lambda_identity (2 lambda DUY NHẤT thật sự lấy tín hiệu
    # từ judge nhận dạng).
    identity_aware = bool(lambda_saliency > 0 or ci.get("lambda_identity", 0.0) > 0)
    if not identity_aware:
        feat_note = (
            f" (lambda_feat={lambda_feat} > 0 CÓ bật, nhưng đây là feature-KD so khớp "
            f"SR teacher, KHÔNG PHẢI tín hiệu nhận dạng — KHÔNG tính là 'identity-aware')"
            if lambda_feat > 0 else ""
        )
        logger.warning(
            f"[CẢNH BÁO — không 'identity-aware'] lambda_saliency=lambda_identity=0 trong lần "
            f"chạy này{feat_note} -> gate KHÔNG dùng bất kỳ tín hiệu nhận dạng nào để quyết định "
            "khối nào giữ/bỏ (chỉ pixel+distill+sparsity, cộng thêm feature-KD nếu lambda_feat>0). "
            "Đây vẫn là 1 cơ chế pruning hợp lệ ('reconstruction-aware pruning'), nhưng KHÔNG phải "
            "'identity-aware pruning' như mô tả trong docstring/bài báo — CHỈ bật lambda_feat KHÔNG "
            "ĐỦ để gọi là identity-aware. Muốn đúng tinh thần identity-aware, truyền tường minh "
            "--lambda_saliency và/hoặc --lambda_identity > 0 qua CLI (xem "
            "pipeline/run_prune_sparsity_screen.sh và run_multi_seed_learned_prune.sh — SỬA giá "
            "trị LAMBDA_SALIENCY/LAMBDA_IDENTITY ở đầu 2 script đó theo đúng recipe thắng của bạn "
            "từ KD-v2/saliency ablation trước khi chạy).")

    # --- Feature-level distillation setup (chỉ bật nếu lambda_feat > 0) ---
    student_hook, teacher_hook, feat_adapter, extra_params = _setup_feat_kd(
        student, teacher, lambda_feat, cfg, scale, device, logger)

    optimizer = torch.optim.Adam(list(student.parameters()) + extra_params, lr=lp["lr"])

    max_epochs = lp["max_epochs"]
    patience = lp["patience"]
    stopper = EarlyStopping(patience=patience, mode="min")

    best_gated_path = run_dir / "best_gated.pt"
    # [SỬA — LỖI PHÁT HIỆN QUA CODE REVIEW, vòng 3, điểm 1] Lưu SONG SONG
    # trọng số feat_adapter TẠI ĐÚNG epoch is_best (trong bộ nhớ, không ghi
    # file riêng vì chỉ cần dùng nội bộ ngay sau vòng lặp) — TRƯỚC ĐÂY chỉ
    # student.state_dict() được lưu vào best_gated_path mỗi lần is_best, còn
    # feat_adapter TIẾP TỤC được train đến hết vòng lặp (early-stop/hết
    # max_epochs) rồi mới bị đọc ra (SAU vòng lặp) làm "gated_adapter_state"
    # — nghĩa là student (epoch tốt nhất) bị ghép với feat_adapter (epoch
    # CUỐI CÙNG, có thể là epoch TỆ HƠN do early-stop patience) — 2 phần
    # KHÔNG cùng 1 thời điểm training, sai lệch nhẹ khi dùng adapter này làm
    # điểm khởi tạo cho pha fine-tune/eval cuối.
    best_gated_adapter_state = None
    for epoch in range(max_epochs):
        train_stats = run_epoch_prune(student, teacher, judges, train_loader, device_mgr, cfg,
                                       lambda_sparsity, lambda_binary, optimizer=optimizer, logger=logger,
                                       student_hook=student_hook, teacher_hook=teacher_hook,
                                       feat_adapter=feat_adapter)
        val_stats = run_epoch_prune(student, teacher, judges, val_loader, device_mgr, cfg,
                                     lambda_sparsity, lambda_binary, optimizer=None, logger=logger,
                                     student_hook=student_hook, teacher_hook=teacher_hook,
                                     feat_adapter=feat_adapter)

        gate_str = ", ".join(f"{g:.3f}" for g in student.gates().tolist())
        oom_note = f" | OOM fallback: {device_mgr.total_oom_events} lần" \
            if device_mgr.total_oom_events > 0 else ""
        logger.info(
            f"[sr_learned_prune] epoch {epoch + 1}/{max_epochs} "
            f"(early-stop counter: {stopper.counter}/{patience}){oom_note} | "
            f"train_quality={train_stats['quality_total']:.4f} pixel={train_stats['pixel']:.4f} "
            f"distill={train_stats['distill']:.4f} feat={train_stats['feat']:.4f} "
            f"saliency={train_stats['saliency']:.4f} identity={train_stats['identity']:.4f} "
            f"sparsity={train_stats['sparsity']:.4f} binary={train_stats['binary']:.4f} | "
            f"VAL_QUALITY={val_stats['quality_total']:.4f} (val_total_với_sparsity={val_stats['total']:.4f}) "
            f"| gates=[{gate_str}]"
        )

        # [SỬA — lỗi phát hiện qua review] Chọn/dừng sớm THEO quality_total
        # (KHÔNG gồm sparsity/binary) — trước đây dùng val_stats["total"] (có
        # sparsity), khiến checkpoint có thể "thắng" chỉ vì gate giảm.
        is_best = stopper.step(val_stats["quality_total"])
        if is_best:
            _save_state_dict(student, best_gated_path)
            if feat_adapter is not None:
                best_gated_adapter_state = {k: v.detach().cpu().clone()
                                             for k, v in feat_adapter.state_dict().items()}
            logger.info(f"  -> checkpoint (gated) tốt nhất mới "
                        f"(val_quality={val_stats['quality_total']:.4f}), đã lưu"
                        f"{' (kèm feat_adapter cùng epoch)' if feat_adapter is not None else ''}.")

        if stopper.should_stop:
            logger.info(f"EARLY STOPPING tại epoch {epoch + 1}: val_quality không cải thiện "
                        f"sau {patience} epoch liên tiếp.")
            break

    # --- Cứng hoá: nạp lại checkpoint (gated) tốt nhất, xoá hẳn khối có
    # gate thấp, xuất ra model SPAN thường (deploy-ready) ---
    logger.info("=== Cứng hoá (harden_and_export): xoá hẳn khối có gate thấp ===")
    _save_last_if_missing(best_gated_path, student, logger, "best_gated.pt")
    if feat_adapter is not None and best_gated_adapter_state is None:
        best_gated_adapter_state = {k: v.detach().cpu().clone()
                                     for k, v in feat_adapter.state_dict().items()}
    student.load_state_dict(torch.load(best_gated_path, map_location=device))
    threshold = lp["gate_harden_threshold"]
    exported, kept_indices, gate_values = student.harden_and_export(threshold=threshold)
    exported = exported.to(device)

    val_quality_before_ft = stopper.best
    logger.info(f"Gate cuối cùng (0-indexed, ngân sách {lp['n_blocks_budget']} khối): "
                f"{[round(g, 3) for g in gate_values]}")
    logger.info(f"-> GIỮ {len(kept_indices)}/{lp['n_blocks_budget']} khối (chỉ số: {kept_indices}), "
                f"ngưỡng={threshold}")

    # [SỬA — bổ sung sau code review, vòng 2, điểm 5; SỬA LẠI ở vòng 3, điểm 1]
    # Dùng `best_gated_adapter_state` (đã chụp ĐÚNG lúc is_best trong vòng
    # lặp trên, CÙNG epoch với student vừa nạp lại) làm điểm khởi tạo cho
    # adapter ở pha fine-tune/eval cuối bên dưới — THAY VÌ feat_adapter.
    # state_dict() đọc SAU vòng lặp (epoch CUỐI, có thể lệch epoch với
    # student.load_state_dict(best_gated_path) ngay phía trên, xem chú thích
    # tại nơi khai báo best_gated_adapter_state). `exported` (SPAN thường,
    # feat=48) và `student` (SPANLearnedPrune, feat=48) có CÙNG số kênh
    # feature tại điểm hook (harden_and_export() chỉ XOÁ HẲN khối bị cắt,
    # KHÔNG đổi feat width của các khối còn lại) nên state_dict của adapter
    # Conv1x1(48->teacher_ch) tương thích shape 100% giữa 2 pha.
    gated_adapter_state = best_gated_adapter_state

    if student_hook is not None:
        student_hook.remove()
    if teacher_hook is not None:
        teacher_hook.remove()

    # --- [MỚI — sửa lỗi "thiếu fine-tune sau harden" phát hiện qua review] ---
    # Model gated lúc train chạy khối giữ lại ở gate MỀM (ví dụ 0.85), model
    # export chạy CỨNG ở gate=1.0 và các khối bị cắt biến mất HẲN (không chỉ
    # gate≈0) — có mismatch train/deploy. Fine-tune ngắn giúp các khối còn lại
    # thích nghi lại, đúng thông lệ pruning chuẩn (prune rồi fine-tune).
    val_quality_after_ft = None
    ft_student_hook = ft_teacher_hook = ft_feat_adapter = None
    if finetune_epochs > 0:
        logger.info(f"=== Fine-tune model đã cứng hoá ({len(kept_indices)} khối) thêm "
                    f"{finetune_epochs} epoch ===")
        ft_student_hook, ft_teacher_hook, ft_feat_adapter, ft_extra_params = _setup_feat_kd(
            exported, teacher, lambda_feat, cfg, scale, device, logger, tag=" (fine-tune)",
            adapter_init_state=gated_adapter_state)
        # [SỬA — bổ sung sau code review, vòng 2, điểm "nhẹ"] Fine-tune sau
        # prune theo thông lệ chuẩn dùng LR THẤP HƠN NHIỀU (thường 5-10x) so
        # với LR train ban đầu — model đã hội tụ gần tối ưu, chỉ cần thích
        # nghi NHẸ với việc mất vài khối, LR gốc dễ làm trồi sụt/phá vỡ trọng
        # số đã học tốt của các khối còn lại.
        # [SỬA — LỖI PHÁT HIỆN QUA CODE REVIEW, vòng 3] "LR ban đầu" ở đây
        # PHẢI là lp["lr"] (sr_learned_prune.lr — LR THẬT đã dùng để train
        # pha gated ngay phía trên), KHÔNG PHẢI ci["lr"] (sr_improve.lr —
        # thuộc script train_sr_distill.py khác, chỉ TÌNH CỜ đang cùng =1e-4
        # trong config hiện tại nên chưa lộ ra sai lệch). Nếu sau này 2 LR
        # này được tách ra khác giá trị (rất có thể vì 2 script train 2 kiến
        # trúc/mục tiêu khác nhau), dùng nhầm ci["lr"] sẽ khiến fine-tune chạy
        # sai LR so với ý định (không phải X% của LR ĐÃ DÙNG để train model
        # này, mà là X% của LR của 1 script hoàn toàn khác).
        finetune_lr = lp["lr"] * lp.get("finetune_lr_scale", 0.1)
        logger.info(f"[fine-tune sau harden] LR={finetune_lr:.2e} "
                    f"({lp.get('finetune_lr_scale', 0.1)}x LR gốc (sr_learned_prune.lr) "
                    f"{lp['lr']:.2e})")
        ft_optimizer = torch.optim.Adam(list(exported.parameters()) + ft_extra_params, lr=finetune_lr)
        ft_stopper = EarlyStopping(patience=lp["patience"], mode="min")
        best_finetuned_path = run_dir / "best_finetuned.pt"
        # [SỬA — LỖI PHÁT HIỆN QUA CODE REVIEW, vòng 3, điểm 1] Y HỆT vấn đề
        # ở pha gated: chụp riêng trọng số ft_feat_adapter TẠI ĐÚNG epoch
        # is_best, KHÔNG để adapter tiếp tục trôi tới epoch cuối (sau
        # patience) trong khi exported đã bị nạp lại về epoch tốt nhất — nếu
        # không, "đánh giá CUỐI CÙNG" bên dưới sẽ ghép SAI 1 exported (epoch
        # tốt nhất) với 1 ft_feat_adapter (epoch cuối, tệ hơn) — 2 con số
        # val_quality_after_finetune (lúc is_best) và val_total_final_deploy_model
        # (đo lại sau) khi đó KHÔNG dùng cùng 1 adapter, không thể so sánh.
        best_ft_adapter_state = None

        for epoch in range(finetune_epochs):
            # [MỚI] Tái sử dụng NGUYÊN VẸN run_epoch() của train_sr_distill.py —
            # exported là SPAN thường (không gate/sparsity), nên công thức loss
            # ĐÚNG HỆT recipe sr_improve bình thường, không cần code riêng.
            train_stats = run_epoch(exported, teacher, judges, train_loader, device_mgr, cfg,
                                     optimizer=ft_optimizer, scaler=None, logger=logger,
                                     student_hook=ft_student_hook, teacher_hook=ft_teacher_hook,
                                     feat_adapter=ft_feat_adapter)
            val_stats = run_epoch(exported, teacher, judges, val_loader, device_mgr, cfg,
                                   optimizer=None, scaler=None, logger=logger,
                                   student_hook=ft_student_hook, teacher_hook=ft_teacher_hook,
                                   feat_adapter=ft_feat_adapter)
            logger.info(
                f"[fine-tune sau harden] epoch {epoch + 1}/{finetune_epochs} "
                f"(early-stop counter: {ft_stopper.counter}/{lp['patience']}) | "
                f"train_total={train_stats['total']:.4f} | VAL_TOTAL={val_stats['total']:.4f}"
            )
            is_best = ft_stopper.step(val_stats["total"])
            if is_best:
                _save_state_dict(exported, best_finetuned_path)
                if ft_feat_adapter is not None:
                    best_ft_adapter_state = {k: v.detach().cpu().clone()
                                              for k, v in ft_feat_adapter.state_dict().items()}
                logger.info(f"  -> checkpoint fine-tune tốt nhất mới "
                            f"(val_total={val_stats['total']:.4f}), đã lưu"
                            f"{' (kèm feat_adapter cùng epoch)' if ft_feat_adapter is not None else ''}.")
            if ft_stopper.should_stop:
                logger.info(f"EARLY STOPPING fine-tune tại epoch {epoch + 1}: val_total không cải "
                            f"thiện sau {lp['patience']} epoch liên tiếp.")
                break

        # KHÔNG remove ft_student_hook/ft_teacher_hook ở đây — TÁI SỬ DỤNG
        # nguyên hook+adapter đã fine-tune cho bước "đánh giá CUỐI CÙNG" bên
        # dưới (xem điểm "eval cuối feat=0" review vòng 2). Sẽ remove sau khi
        # eval cuối xong.

        # Nạp lại checkpoint fine-tune tốt nhất (theo val_total CHUẨN, không
        # sparsity) làm model deploy CUỐI CÙNG.
        _save_last_if_missing(best_finetuned_path, exported, logger, "best_finetuned.pt")
        if ft_feat_adapter is not None and best_ft_adapter_state is None:
            best_ft_adapter_state = {k: v.detach().cpu().clone()
                                      for k, v in ft_feat_adapter.state_dict().items()}
        exported.load_state_dict(torch.load(best_finetuned_path, map_location=device))
        # [SỬA — LỖI PHÁT HIỆN QUA CODE REVIEW, vòng 3, điểm 1] Nạp lại
        # ft_feat_adapter về ĐÚNG epoch is_best CÙNG LÚC với exported — trước
        # đây thiếu bước này nên ft_feat_adapter vẫn ở trạng thái epoch CUỐI
        # (sau patience), lệch epoch với exported vừa nạp lại, khiến "đánh
        # giá CUỐI CÙNG" bên dưới (dùng lại ft_feat_adapter) đo "feat" không
        # khớp với epoch đã chọn làm best.
        if ft_feat_adapter is not None and best_ft_adapter_state is not None:
            ft_feat_adapter.load_state_dict(best_ft_adapter_state)
        val_quality_after_ft = ft_stopper.best
        logger.info(f"Hoàn tất fine-tune. val_total trước fine-tune (ước lượng từ pha gated, "
                    f"KHÔNG hoàn toàn tương đương do khác kiến trúc)={stopper.best_str}, "
                    f"SAU fine-tune (model đã cứng hoá thật, so sánh trực tiếp được)="
                    f"{ft_stopper.best_str}.")
    else:
        logger.info("finetune_epochs_after_harden=0 -> BỎ QUA bước fine-tune sau harden "
                    "(KHÔNG khuyến khích — xem docstring đầu file về mismatch train/deploy).")

    # --- [MỚI — sửa lỗi "chưa eval lại sau harden" phát hiện qua review] ---
    # Luôn đo lại loss của model deploy-ready cuối cùng (dù có fine-tune hay
    # không) bằng đúng model đã lưu (best.pt) — đây là con số phản ánh ĐÚNG
    # chất lượng SR sẽ dùng ở các bước sau (build_sr.py/eval_sr_quality.py/
    # train_recognition.py), không suy diễn từ số liệu lúc train gated (kiến
    # trúc khác nhau, không thể so trực tiếp).
    #
    # [SỬA — LỖI PHÁT HIỆN QUA CODE REVIEW, vòng 2] Bản trước gọi run_epoch()
    # KHÔNG kèm hook/adapter nào (đã remove() ở trên) -> "feat" trong
    # final_val_stats LUÔN = 0 dù lambda_feat>0, MỘT CON SỐ SAI/GÂY HIỂU NHẦM
    # nếu vô tình đưa vào bài báo. Sửa: nếu đã fine-tune, TÁI SỬ DỤNG nguyên
    # ft_student_hook/ft_teacher_hook/ft_feat_adapter (đã học, đang sống) cho
    # eval này; nếu KHÔNG fine-tune (finetune_epochs=0), tạo hook mới + NẠP
    # LẠI adapter đã học từ pha gated (gated_adapter_state) làm xấp xỉ hợp lý
    # (harden chỉ xoá khối, không đổi feat width các khối còn lại nên adapter
    # gated vẫn ánh xạ đúng không gian đặc trưng) — luôn TỐT HƠN 1 adapter
    # random (sẽ cho ra "feat" là số ngẫu nhiên vô nghĩa, còn tệ hơn báo 0 rõ
    # ràng) hoặc báo cứng =0 (sai, không phản ánh gì).
    eval_student_hook, eval_teacher_hook, eval_feat_adapter = ft_student_hook, ft_teacher_hook, ft_feat_adapter
    created_eval_hook = False
    if finetune_epochs <= 0 and lambda_feat > 0:
        eval_student_hook, eval_teacher_hook, eval_feat_adapter, _ = _setup_feat_kd(
            exported, teacher, lambda_feat, cfg, scale, device, logger, tag=" (eval cuối)",
            adapter_init_state=gated_adapter_state)
        created_eval_hook = True

    final_val_stats = run_epoch(exported, teacher, judges, val_loader, device_mgr, cfg,
                                 optimizer=None, scaler=None, logger=logger,
                                 student_hook=eval_student_hook, teacher_hook=eval_teacher_hook,
                                 feat_adapter=eval_feat_adapter)
    logger.info(f"=== Đánh giá CUỐI CÙNG model deploy-ready ({len(kept_indices)} khối, sau "
                f"{'fine-tune' if finetune_epochs > 0 else 'harden thô, KHÔNG fine-tune'}): "
                f"val_total={final_val_stats['total']:.4f} pixel={final_val_stats['pixel']:.4f} "
                f"distill={final_val_stats['distill']:.4f} feat={final_val_stats['feat']:.4f} ===")

    if eval_student_hook is not None:
        eval_student_hook.remove()
    if eval_teacher_hook is not None:
        eval_teacher_hook.remove()
    # Nếu hook được tạo RIÊNG cho eval cuối (không phải tái dùng từ fine-tune),
    # nhắc rõ đây KHÔNG phải adapter đã fine-tune cùng exported hiện tại.
    if created_eval_hook:
        logger.info("(Lưu ý: adapter dùng để đo 'feat' ở trên được NẠP LẠI từ pha gated trước "
                    "harden, KHÔNG được fine-tune cùng model hiện tại vì finetune_epochs_after_harden=0 "
                    "— chỉ mang tính tham khảo xấp xỉ, không phải số liệu chính xác tuyệt đối.)")

    torch.save({k: v.cpu() for k, v in exported.state_dict().items()}, run_dir / "best.pt")

    def _json_float(x):
        if x is None:
            return None
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return None
        if xf != xf or xf in (float("inf"), float("-inf")):
            return None
        return xf

    metadata = {
        "n_blocks_budget": lp["n_blocks_budget"],
        "gate_harden_threshold": threshold,
        "gate_values_final": [
            None if _json_float(g) is None else round(g, 4) for g in gate_values
        ],
        "kept_block_indices": kept_indices,
        "n_blocks_kept": len(kept_indices),
        "lambda_feat": lambda_feat,
        "lambda_saliency": lambda_saliency,
        "lambda_identity": ci.get("lambda_identity", 0.0),
        "lambda_sparsity": lambda_sparsity,
        "lambda_binary": lambda_binary,
        # [SỬA — LỖI GỘP NHẦM phát hiện qua code review, vòng 5] identity_aware
        # CHỈ dựa trên lambda_saliency/lambda_identity (2 lambda thật sự dùng
        # tín hiệu recognition judge) — lambda_feat (feature-KD so khớp SR
        # teacher, không liên quan nhận dạng) KHÔNG được tính vào đây nữa,
        # xem giải thích đầy đủ trong docstring đầu file mục "IDENTITY-AWARE".
        "identity_aware": identity_aware,
        "uses_feature_kd": bool(lambda_feat > 0),
        "finetune_epochs_after_harden_configured": finetune_epochs,
        "finetune_lr_scale": lp.get("finetune_lr_scale", 0.1) if finetune_epochs > 0 else None,
        "val_quality_gated_before_harden": _json_float(val_quality_before_ft),
        "val_quality_after_finetune": _json_float(val_quality_after_ft),
        "val_total_final_deploy_model": _json_float(final_val_stats["total"]),
        "final_eval_feat_adapter_source": (
            "finetuned" if finetune_epochs > 0 else
            ("gated_phase_reused" if created_eval_hook else "n/a_lambda_feat_zero")
        ),
    }
    with open(run_dir / "prune_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, allow_nan=False)

    # [MỚI — bổ sung sau code review] Cảnh báo tường minh về khả năng so sánh
    # công bằng với span_tiny (feat=48, n_blocks=3 CỐ ĐỊNH) — chỉ khi K=3 thì
    # 2 model mới CÙNG kiến trúc/kích thước, so sánh params/accuracy trực tiếp
    # mới hợp lệ.
    if len(kept_indices) == 3:
        logger.info("[So sánh công bằng] Số khối giữ lại = 3, TRÙNG span_tiny -> so sánh "
                    "params/FLOPs/accuracy trực tiếp với span_tiny HỢP LỆ (cùng kiến trúc).")
    else:
        logger.warning(
            f"[CẢNH BÁO so sánh công bằng] Số khối giữ lại = {len(kept_indices)}, "
            f"KHÁC span_tiny (3 khối cố định) -> model này có params/FLOPs KHÁC span_tiny, "
            f"KHÔNG nên so sánh trực tiếp accuracy như 'cùng 1 model' trong bảng kết quả. "
            f"{'So sánh với span_large/SPAN baseline sẽ công bằng hơn (nhiều khối hơn span_tiny).' if len(kept_indices) > 3 else 'Đây là kiến trúc NHẸ HƠN CẢ span_tiny — báo cáo params/FLOPs riêng, không gộp chung bảng so sánh với span_tiny.'}"
        )

    logger.info(f"Đã lưu model deploy-ready (SPAN thường, {len(kept_indices)} khối): {run_dir / 'best.pt'}")
    logger.info(f"Đã lưu metadata pruning: {run_dir / 'prune_metadata.json'}")
    logger.info(f"BƯỚC TIẾP THEO: dùng checkpoint này ở data/build_sr.py, eval_sr_quality.py, "
                f"train_recognition.py Y HỆT span_tiny, nhưng thay "
                f"'--arch span_tiny' bằng '--arch span_pruned --n_blocks {len(kept_indices)}'.")


if __name__ == "__main__":
    main()
