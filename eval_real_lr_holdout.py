"""
Đánh giá trên `splits/real_lr_holdout.json` — ảnh vốn dĩ đã nhỏ NGOÀI ĐỜI THẬT
(không phải downsample bicubic nhân tạo). Đây là bài kiểm tra tổng quát hóa
quan trọng: SPAN cải tiến có hoạt động tốt trên suy giảm thật, hay chỉ ăn may
trên suy giảm bicubic sạch sẽ mà chính pipeline tự tạo ra để train?

3 điều kiện so sánh, mỗi điều kiện dùng checkpoint recognition ĐÃ TRAIN TRÊN
ĐÚNG DOMAIN tương ứng (đúng nguyên tắc đã thống nhất — không dùng checkpoint
train trên domain khác để test):
  - no_sr:       ảnh raw -> letterbox thẳng lên hr_size (bicubic thường)
                 -> checkpoint recognition_lr_<backbone>
  - sr_baseline: ảnh raw -> letterbox xuống lr_size -> SPAN baseline -> hr_size
                 -> checkpoint recognition_sr_baseline_<backbone>
  - sr_improved: ảnh raw -> letterbox xuống lr_size -> SPAN improved -> hr_size
                 -> checkpoint recognition_sr_improved_<backbone>

LƯU Ý QUAN TRỌNG: một số identity trong real_lr_holdout có thể trùng với
identity đã xuất hiện ở tập TRAIN của pipeline chính (vì việc phân chia
train/val/test chỉ áp dụng cho nhóm HR-source, không áp dụng cho nhóm quá nhỏ
này). Đây là hạn chế đã biết — kết quả trên tập này chỉ mang tính chất kiểm
chứng bổ sung, không dùng làm số liệu accuracy chính của luận án.

Chạy:
    python eval_real_lr_holdout.py --config configs/config.yaml --backbone mobilenet_v2 \
        --sr_baseline_ckpt runs/sr_span_official/best.pt --sr_baseline_arch span_official \
        --sr_improved_ckpt runs/sr_improved_span_official/best.pt --sr_improved_arch span_official \
        --out_csv results/real_lr_holdout.csv
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from datasets.ear_dataset import build_label_map
from models.recognition_model import EarRecognitionNet
from models.sr_models import build_sr_model
from utils.letterbox import letterbox_resize
from utils.metrics import compute_accuracy


class RealLRHoldoutDataset(Dataset):
    """
    mode="no_sr": trả về ảnh raw letterbox thẳng lên hr_size.
    mode="sr": trả về ảnh raw letterbox xuống lr_size (đầu vào cho SR model).
    """

    def __init__(self, holdout_json: str, label_map: dict, hr_size: int, lr_size: int,
                 mode: str):
        with open(holdout_json, "r", encoding="utf-8") as f:
            self.entries = json.load(f)
        self.label_map = label_map
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.mode = mode
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img = Image.open(entry["path"]).convert("RGB")

        target_size = self.hr_size if self.mode == "no_sr" else self.lr_size
        img = letterbox_resize(img, target_size)
        img_tensor = self.to_tensor(img)

        identity_label = self.label_map[entry["person_id"]]
        gender_label = entry["gender"]
        return img_tensor, torch.tensor(identity_label), torch.tensor(gender_label)


def _run_recognition_eval(model, loader, device):
    id_acc_sum, gender_acc_sum, n_batches = 0.0, 0.0, 0
    for img, id_labels, gender_labels in loader:
        img = img.to(device)
        id_labels, gender_labels = id_labels.to(device), gender_labels.to(device)
        id_logits, gender_logits, _ = model(img)
        id_acc_sum += compute_accuracy(id_logits, id_labels)
        gender_acc_sum += compute_accuracy(gender_logits, gender_labels)
        n_batches += 1
    return id_acc_sum / n_batches, gender_acc_sum / n_batches


@torch.no_grad()
def eval_no_sr(cfg, label_map, backbone, device):
    splits_root = cfg["paths"]["splits_root"]
    hr_size = cfg["image"]["hr_size"]
    lr_size = hr_size // cfg["image"]["scale"]

    dataset = RealLRHoldoutDataset(f"{splits_root}/real_lr_holdout.json", label_map,
                                    hr_size, lr_size, mode="no_sr")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    ckpt_path = f"{cfg['paths']['runs_root']}/recognition_lr_{backbone}/best.pt"
    model = EarRecognitionNet(
        num_identities=cfg["num_identities"], num_genders=cfg["num_genders"],
        embedding_dim=cfg["recognition"]["embedding_dim"], backbone=backbone,
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    return _run_recognition_eval(model, loader, device), len(dataset), ckpt_path


@torch.no_grad()
def eval_with_sr(cfg, label_map, backbone, device, sr_arch, sr_ckpt, recognition_domain):
    splits_root = cfg["paths"]["splits_root"]
    hr_size = cfg["image"]["hr_size"]
    scale = cfg["image"]["scale"]
    lr_size = hr_size // scale

    dataset = RealLRHoldoutDataset(f"{splits_root}/real_lr_holdout.json", label_map,
                                    hr_size, lr_size, mode="sr")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    sr_model = build_sr_model(sr_arch, scale)
    sr_model.load_state_dict(torch.load(sr_ckpt, map_location=device))
    sr_model.to(device).eval()

    ckpt_path = f"{cfg['paths']['runs_root']}/recognition_{recognition_domain}_{backbone}/best.pt"
    rec_model = EarRecognitionNet(
        num_identities=cfg["num_identities"], num_genders=cfg["num_genders"],
        embedding_dim=cfg["recognition"]["embedding_dim"], backbone=backbone,
        pretrained=False,
    ).to(device)
    rec_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    rec_model.eval()

    id_acc_sum, gender_acc_sum, n_batches = 0.0, 0.0, 0
    for lr_img, id_labels, gender_labels in loader:
        lr_img = lr_img.to(device)
        id_labels, gender_labels = id_labels.to(device), gender_labels.to(device)

        sr_img = sr_model(lr_img)  # LR (nhỏ) -> HR (hr_size), đúng input cho recognition
        id_logits, gender_logits, _ = rec_model(sr_img)

        id_acc_sum += compute_accuracy(id_logits, id_labels)
        gender_acc_sum += compute_accuracy(gender_logits, gender_labels)
        n_batches += 1

    return (id_acc_sum / n_batches, gender_acc_sum / n_batches), len(dataset), ckpt_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--sr_baseline_ckpt", required=True)
    ap.add_argument("--sr_baseline_arch", required=True)
    ap.add_argument("--sr_improved_ckpt", required=True)
    ap.add_argument("--sr_improved_arch", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits_root = cfg["paths"]["splits_root"]
    label_map = build_label_map(f"{splits_root}/splits.json")

    rows = []

    print(">>> [no_sr] đánh giá trên real_lr_holdout...")
    (id_acc, gender_acc), n, ckpt = eval_no_sr(cfg, label_map, args.backbone, device)
    rows.append({"condition": "no_sr", "backbone": args.backbone,
                 "identity_accuracy": round(id_acc, 4), "gender_accuracy": round(gender_acc, 4),
                 "n_images": n, "recognition_ckpt": ckpt})

    print(">>> [sr_baseline] đánh giá trên real_lr_holdout...")
    (id_acc, gender_acc), n, ckpt = eval_with_sr(
        cfg, label_map, args.backbone, device,
        args.sr_baseline_arch, args.sr_baseline_ckpt, "sr_baseline")
    rows.append({"condition": "sr_baseline", "backbone": args.backbone,
                 "identity_accuracy": round(id_acc, 4), "gender_accuracy": round(gender_acc, 4),
                 "n_images": n, "recognition_ckpt": ckpt})

    print(">>> [sr_improved] đánh giá trên real_lr_holdout...")
    (id_acc, gender_acc), n, ckpt = eval_with_sr(
        cfg, label_map, args.backbone, device,
        args.sr_improved_arch, args.sr_improved_ckpt, "sr_improved")
    rows.append({"condition": "sr_improved", "backbone": args.backbone,
                 "identity_accuracy": round(id_acc, 4), "gender_accuracy": round(gender_acc, 4),
                 "n_images": n, "recognition_ckpt": ckpt})

    print(f"\n{'=' * 60}")
    print(f"KẾT QUẢ TRÊN REAL_LR_HOLDOUT (backbone={args.backbone})")
    print(f"{'=' * 60}")
    for r in rows:
        print(f"  {r['condition']:<14} identity_acc={r['identity_accuracy']:.4f}  "
              f"gender_acc={r['gender_accuracy']:.4f}")
    print(f"{'=' * 60}\n")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã ghi {out_path}")


if __name__ == "__main__":
    main()
