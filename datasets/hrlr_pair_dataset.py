"""Dataset đọc cặp ảnh (HR, LR) cùng tên file từ hai thư mục song song.
Dùng chung cho train_sr.py, train_sr_distill.py, eval_sr_quality.py."""
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.degradation import random_degrade
from utils.letterbox import compute_letterbox_geometry


class HRLRPairDataset(Dataset):
    """
    return_bbox=False, return_label=False (mặc định, giữ nguyên hành vi cũ —
    KHÔNG đổi gì cho train_sr.py, vẫn trả về đúng 2 giá trị (lr_img, hr_img)
    như trước): loss lúc TRAIN vẫn tính trên toàn bộ canvas (bao gồm viền đệm
    đen letterbox) như thiết kế ban đầu — đây là lựa chọn có chủ đích (không
    phải bug), giữ nguyên để không cần train lại các model SR đã có chỉ vì
    sửa lỗi đo lường.

    return_bbox=True ([MỚI — đợt 7], splits_json bắt buộc đi kèm): trả về
    thêm bbox=(x0,y0,w,h) — vùng ROI thật (không phải viền đệm đen), suy từ
    (width, height) ảnh gốc lưu trong splits.json qua
    utils/letterbox.py::compute_letterbox_geometry() — dùng cho
    eval_sr_quality.py để đo PSNR/SSIM chỉ trên vùng tai thật, sửa lỗi phát
    hiện qua code review (đo trên cả viền đen làm số liệu bị thổi phồng, xem
    utils/metrics.py::compute_psnr_roi/compute_ssim_roi).

    return_label=True ([MỚI — sửa sau code review], splits_json bắt buộc đi
    kèm): trả về thêm identity_label (int, cùng ánh xạ person_id->index với
    datasets/ear_dataset.py::build_label_map(), đọc TỪ CÙNG splits_json nên
    NHẤT QUÁN 100% với chỉ số identity dùng lúc train recognition) — dùng cho
    train_sr_distill.py::compute_multi_judge_saliency() để lấy đúng
    ∂logit[y_true]/∂I thay vì ||emb||² (không phải tín hiệu identity thật, xem
    code review).

    Thứ tự trả về khi bật cả 2 cờ: (lr_img, hr_img, bbox, identity_label) —
    bbox LUÔN đứng trước label nếu cả 2 cùng bật.

    degradation_augment=False ([MỚI — Mục 5.9 bài báo, giải pháp cho
    "Boundary Condition" real_lr_holdout], mặc định giữ NGUYÊN hành vi cũ):
    bật True để sinh LR bằng suy giảm THẬT ngẫu nhiên (blur+noise+JPEG, xem
    utils/degradation.py::random_degrade()) NGAY TẠI __getitem__, một tổ hợp
    MỚI mỗi lần gọi (mỗi sample mỗi epoch khác nhau) -- THAY vì đọc file LR
    tĩnh đã sinh sẵn bằng bicubic sạch. Yêu cầu truyền kèm `scale`. CHỈ nên
    bật cho train_set (cần đa dạng suy giảm qua nhiều epoch để học tính
    robust); val_set nên giữ nguyên degradation_augment=False (LR tĩnh, sạch)
    để chỉ số early-stopping ổn định/so sánh được qua các epoch, đúng
    nguyên tắc "chỉ đổi training signal, không đổi protocol đánh giá" đã áp
    dụng cho mọi cơ chế khác trong project (lambda_feat, lambda_saliency...).
    """

    def __init__(self, hr_root: str, lr_root: str, split_name: str,
                 return_bbox: bool = False, return_label: bool = False,
                 splits_json: str = None,
                 degradation_augment: bool = False, scale: int = None):
        self.hr_root = Path(hr_root) / split_name
        self.lr_root = Path(lr_root) / split_name
        self.pairs = sorted(self.hr_root.glob("*/*.jpg"))
        self.to_tensor = transforms.ToTensor()

        # Fail-fast: HR có mà LR thiếu → FileNotFoundError giữa epoch (sau hàng
        # giờ GPU). Pipeline chuẩn (data/build_lr.py) luôn ghi cặp; kiểm tra
        # ở đây bắt lệch tay / copy thiếu thư mục lr.
        missing_lr = [p for p in self.pairs
                      if not (self.lr_root / p.parent.name / p.name).is_file()]
        if missing_lr:
            example = missing_lr[0]
            raise FileNotFoundError(
                f"{len(missing_lr)} ảnh {split_name} có HR nhưng không có cặp LR. "
                f"Ví dụ: {example} -> {self.lr_root / example.parent.name / example.name}. "
                f"Chạy lại data/build_lr.py.")

        self.return_bbox = return_bbox
        self.return_label = return_label
        self.roi_lookup = None
        self.label_map = None
        if return_bbox or return_label:
            if splits_json is None:
                raise ValueError("return_bbox/return_label=True cần truyền kèm splits_json "
                                  "(để lấy width/height ảnh gốc và/hoặc ánh xạ nhãn identity).")
            with open(splits_json, "r", encoding="utf-8") as f:
                splits = json.load(f)
            if return_bbox:
                self.roi_lookup = {
                    (e["person_id"], Path(e["path"]).name): (e["width"], e["height"])
                    for e in splits[split_name]
                }
            if return_label:
                # Dùng ĐÚNG hàm build_label_map() của datasets/ear_dataset.py (không
                # tự viết lại ánh xạ riêng) — đảm bảo identity index ở đây LUÔN khớp
                # với identity index dùng lúc train/eval recognition trên MỌI domain.
                from datasets.ear_dataset import build_label_map
                self.label_map = build_label_map(splits_json)

            # Fail-fast: thiếu key trong splits.json sẽ KeyError giữa epoch
            # (sau hàng giờ GPU). Kiểm tra hết ở __init__.
            if return_bbox:
                missing = [p for p in self.pairs
                           if (p.parent.name, p.name) not in self.roi_lookup]
                if missing:
                    raise KeyError(
                        f"{len(missing)} ảnh {split_name} không có trong splits.json "
                        f"(roi_lookup). Ví dụ: {missing[0]}. "
                        f"Chạy lại data/prepare_splits.py.")
            if return_label:
                missing_pid = sorted({p.parent.name for p in self.pairs} - set(self.label_map))
                if missing_pid:
                    raise KeyError(
                        f"{len(missing_pid)} person_id trong thư mục {split_name} không có "
                        f"trong label_map (splits.json). Ví dụ: {missing_pid[:5]}")

        self.degradation_augment = degradation_augment
        self.scale = scale
        if degradation_augment and scale is None:
            raise ValueError("degradation_augment=True cần truyền kèm scale "
                              "(để biết sinh LR kích thước bao nhiêu từ HR).")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hr_path = self.pairs[idx]
        hr_pil = Image.open(hr_path).convert("RGB")
        hr_img = self.to_tensor(hr_pil)
        if self.degradation_augment:
            # [MỚI — Mục 5.9] Sinh LR bằng suy giảm THẬT ngẫu nhiên NGAY TỪ
            # HR, một tổ hợp MỚI mỗi lần gọi -- KHÔNG đọc file LR tĩnh (bicubic
            # sạch) đã sinh sẵn. Xem utils/degradation.py::random_degrade().
            lr_img = self.to_tensor(random_degrade(hr_pil, self.scale))
        else:
            lr_path = self.lr_root / hr_path.parent.name / hr_path.name
            lr_img = self.to_tensor(Image.open(lr_path).convert("RGB"))
        pid = hr_path.parent.name

        result = (lr_img, hr_img)

        if self.return_bbox:
            orig_w, orig_h = self.roi_lookup[(pid, hr_path.name)]
            target_size = hr_img.shape[-1]  # ảnh HR đã letterbox về hình vuông target_size x target_size
            new_w, new_h, paste_x, paste_y = compute_letterbox_geometry(orig_w, orig_h, target_size)
            bbox = (paste_x, paste_y, new_w, new_h)
            result = result + (bbox,)

        if self.return_label:
            result = result + (torch.tensor(self.label_map[pid], dtype=torch.long),)

        return result
