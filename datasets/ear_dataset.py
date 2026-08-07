"""
Dataset dùng chung cho mọi domain (hr/lr/sr). Ánh xạ person_id -> class index
được xây một lần và tái sử dụng nhất quán giữa các domain để nhãn không bị lệch.
"""
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_label_map(splits_json: str):
    with open(splits_json, "r", encoding="utf-8") as f:
        splits = json.load(f)
    all_ids = sorted({e["person_id"] for entries in splits.values() for e in entries})
    return {pid: idx for idx, pid in enumerate(all_ids)}


class EarDataset(Dataset):
    """
    domain_root: ví dụ splits/hr, splits/lr, hoặc splits/sr
    split_name: "train" | "val" | "test"
    splits_json: đường dẫn splits.json gốc (chỉ dùng để lấy danh sách file + nhãn,
                 ảnh thật được đọc từ domain_root/split_name/<person_id>/<filename>)
    """

    def __init__(self, domain_root: str, split_name: str, splits_json: str,
                 label_map: dict, image_size: int, train: bool):
        with open(splits_json, "r", encoding="utf-8") as f:
            splits = json.load(f)
        self.entries = splits[split_name]
        self.domain_root = Path(domain_root) / split_name
        self.label_map = label_map

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        pid = entry["person_id"]
        filename = Path(entry["path"]).name
        img_path = self.domain_root / pid / filename

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        identity_label = self.label_map[pid]
        gender_label = entry["gender"]

        return img, torch.tensor(identity_label), torch.tensor(gender_label)
