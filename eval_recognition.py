"""
Test một checkpoint (đã train ở train_domain, backbone cụ thể) trên test set
của test_domain bất kỳ. Ghi kết quả JSON — dùng để tổng hợp bảng benchmark.

Chạy ví dụ:
    python eval_recognition.py --config configs/config.yaml \
        --ckpt runs/recognition_hr_mobilenet_v2/best.pt \
        --backbone mobilenet_v2 --train_domain hr --test_domain hr \
        --out_json results/hr_hr_mobilenet_v2.json
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.ear_dataset import EarDataset, build_label_map
from models.recognition_model import EarRecognitionNet, SUPPORTED_BACKBONES
from utils.metrics import compute_accuracy, compute_topk_accuracy, count_params, measure_latency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", required=True, choices=SUPPORTED_BACKBONES)
    ap.add_argument("--train_domain", required=True,
                     help="hr | lr | sr_baseline | sr_improved | sr_ablation_<tên>")
    ap.add_argument("--test_domain", required=True,
                     help="hr | lr | sr_baseline | sr_improved | sr_ablation_<tên>")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits_root = cfg["paths"]["splits_root"]
    splits_json = f"{splits_root}/splits.json"
    label_map = build_label_map(splits_json)

    domain_root = f"{splits_root}/{args.test_domain}"
    image_size = cfg["image"]["hr_size"]

    test_set = EarDataset(domain_root, "test", splits_json, label_map, image_size, train=False)
    test_loader = DataLoader(test_set, batch_size=cfg["recognition"]["batch_size"],
                              shuffle=False, num_workers=4)

    model = EarRecognitionNet(
        num_identities=cfg["num_identities"],
        num_genders=cfg["num_genders"],
        embedding_dim=cfg["recognition"]["embedding_dim"],
        backbone=args.backbone,
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    id_acc_sum, id_rank5_sum, gender_acc_sum, n_batches = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, id_labels, gender_labels in test_loader:
            imgs = imgs.to(device)
            id_labels = id_labels.to(device)
            gender_labels = gender_labels.to(device)

            id_logits, gender_logits, _ = model(imgs)
            id_acc_sum += compute_accuracy(id_logits, id_labels)
            id_rank5_sum += compute_topk_accuracy(id_logits, id_labels, k=5)
            gender_acc_sum += compute_accuracy(gender_logits, gender_labels)
            n_batches += 1

    id_acc = id_acc_sum / n_batches
    id_rank5_acc = id_rank5_sum / n_batches
    gender_acc = gender_acc_sum / n_batches
    params_m = count_params(model)
    latency_ms = measure_latency(model, (1, 3, image_size, image_size), device)

    result = {
        "backbone": args.backbone,
        "train_domain": args.train_domain,
        "test_domain": args.test_domain,
        "config_name": f"{args.backbone}__{args.train_domain}_{args.test_domain}",
        "identity_accuracy": round(id_acc, 4),
        "identity_accuracy_rank5": round(id_rank5_acc, 4),
        "gender_accuracy": round(gender_acc, 4),
        "params_M": round(params_m, 3),
        "latency_ms": round(latency_ms, 3),
    }

    print(f"\n{'=' * 60}")
    print(f"KẾT QUẢ TEST | backbone={args.backbone} | "
          f"train_domain={args.train_domain} | test_domain={args.test_domain}")
    print(f"{'=' * 60}")
    print(f"  IDENTITY ACCURACY : {id_acc:.4f}")
    print(f"  IDENTITY RANK-5   : {id_rank5_acc:.4f}")
    print(f"  GENDER ACCURACY   : {gender_acc:.4f}")
    print(f"  Params (M)        : {params_m:.3f}")
    print(f"  Latency (ms)      : {latency_ms:.3f}")
    print(f"{'=' * 60}\n")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
