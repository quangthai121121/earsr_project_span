"""Dataset đọc cặp ảnh (HR, LR) cùng tên file từ hai thư mục song song.
Dùng chung cho train_sr.py, train_sr_distill.py, eval_sr_quality.py."""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HRLRPairDataset(Dataset):
    def __init__(self, hr_root: str, lr_root: str, split_name: str):
        self.hr_root = Path(hr_root) / split_name
        self.lr_root = Path(lr_root) / split_name
        self.pairs = sorted(self.hr_root.glob("*/*.jpg"))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hr_path = self.pairs[idx]
        lr_path = self.lr_root / hr_path.parent.name / hr_path.name
        hr_img = self.to_tensor(Image.open(hr_path).convert("RGB"))
        lr_img = self.to_tensor(Image.open(lr_path).convert("RGB"))
        return lr_img, hr_img
