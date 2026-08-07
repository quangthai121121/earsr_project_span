"""
Bước 4: dùng SR model đã train (train_sr.py) để sinh tập SR từ tập LR, cho cả 3 split.

Chạy:
    python data/build_sr.py --lr_dir splits/lr --sr_ckpt runs/sr_fsrcnn/best.pt \
        --arch fsrcnn --scale 4 --out_dir splits/sr
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.sr_models import build_sr_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr_dir", required=True)
    ap.add_argument("--sr_ckpt", required=True)
    ap.add_argument("--arch", default="fsrcnn")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sr_model(args.arch, args.scale).to(device)
    model.load_state_dict(torch.load(args.sr_ckpt, map_location=device))
    model.eval()

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    for split_name in ["train", "val", "test"]:
        split_dir = Path(args.lr_dir) / split_name
        out_split_dir = Path(args.out_dir) / split_name
        img_paths = list(split_dir.glob("*/*.jpg"))

        for img_path in tqdm(img_paths, desc=f"[{split_name}] chạy SR"):
            pid = img_path.parent.name
            out_person_dir = out_split_dir / pid
            out_person_dir.mkdir(parents=True, exist_ok=True)

            with Image.open(img_path) as im:
                im = im.convert("RGB")
                lr_tensor = to_tensor(im).unsqueeze(0).to(device)

            with torch.no_grad():
                sr_tensor = model(lr_tensor).squeeze(0).cpu().clamp(0, 1)

            sr_img = to_pil(sr_tensor)
            sr_img.save(out_person_dir / img_path.name)

    print("Hoàn tất sinh tập SR cho train/val/test.")


if __name__ == "__main__":
    main()
