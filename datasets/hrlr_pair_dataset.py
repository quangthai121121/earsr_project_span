"""Dataset đọc cặp ảnh (HR, LR) cùng tên file từ hai thư mục song song.
Dùng chung cho train_sr.py, train_sr_distill.py, eval_sr_quality.py."""
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.letterbox import compute_letterbox_geometry


class HRLRPairDataset(Dataset):
    """
    return_bbox=False (mặc định, giữ nguyên hành vi cũ — KHÔNG đổi gì cho
    train_sr.py/train_sr_distill.py, vẫn trả về đúng 2 giá trị (lr_img,
    hr_img) như trước): loss lúc TRAIN vẫn tính trên toàn bộ canvas (bao gồm
    viền đệm đen letterbox) như thiết kế ban đầu — đây là lựa chọn có chủ
    đích (không phải bug), giữ nguyên để không cần train lại các model SR đã
    có chỉ vì sửa lỗi đo lường.

    return_bbox=True ([MỚI — đợt 7], splits_json bắt buộc đi kèm): trả về
    thêm bbox=(x0,y0,w,h) — vùng ROI thật (không phải viền đệm đen), suy từ
    (width, height) ảnh gốc lưu trong splits.json qua
    utils/letterbox.py::compute_letterbox_geometry() — dùng cho
    eval_sr_quality.py để đo PSNR/SSIM chỉ trên vùng tai thật, sửa lỗi phát
    hiện qua code review (đo trên cả viền đen làm số liệu bị thổi phồng, xem
    utils/metrics.py::compute_psnr_roi/compute_ssim_roi).
    """

    def __init__(self, hr_root: str, lr_root: str, split_name: str,
                 return_bbox: bool = False, splits_json: str = None):
        self.hr_root = Path(hr_root) / split_name
        self.lr_root = Path(lr_root) / split_name
        self.pairs = sorted(self.hr_root.glob("*/*.jpg"))
        self.to_tensor = transforms.ToTensor()

        self.return_bbox = return_bbox
        self.roi_lookup = None
        if return_bbox:
            if splits_json is None:
                raise ValueError("return_bbox=True cần truyền kèm splits_json "
                                  "(để lấy width/height ảnh gốc, suy ra vùng ROI thật).")
            with open(splits_json, "r", encoding="utf-8") as f:
                splits = json.load(f)
            self.roi_lookup = {
                (e["person_id"], Path(e["path"]).name): (e["width"], e["height"])
                for e in splits[split_name]
            }

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hr_path = self.pairs[idx]
        lr_path = self.lr_root / hr_path.parent.name / hr_path.name
        hr_img = self.to_tensor(Image.open(hr_path).convert("RGB"))
        lr_img = self.to_tensor(Image.open(lr_path).convert("RGB"))

        if not self.return_bbox:
            return lr_img, hr_img

        pid = hr_path.parent.name
        orig_w, orig_h = self.roi_lookup[(pid, hr_path.name)]
        target_size = hr_img.shape[-1]  # ảnh HR đã letterbox về hình vuông target_size x target_size
        new_w, new_h, paste_x, paste_y = compute_letterbox_geometry(orig_w, orig_h, target_size)
        bbox = (paste_x, paste_y, new_w, new_h)
        return lr_img, hr_img, bbox
