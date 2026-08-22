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
import csv
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.ear_dataset import EarDataset, build_label_map
from models.recognition_model import EarRecognitionNet, SUPPORTED_BACKBONES
from utils.metrics import (count_params, measure_latency,
                            compute_pairwise_genuine_impostor_scores, compute_roc_auc_eer,
                            compute_confusion_matrix)

# [MỚI — bổ sung journal Q1] Ngưỡng số cặp genuine tối thiểu để coi AUC/EER
# verification-setting là đủ tin cậy để báo cáo — dưới ngưỡng này in cảnh
# báo rõ ràng thay vì âm thầm đưa ra số liệu có thể nhiễu (ví dụ nếu phần lớn
# identity trong test set chỉ có 1 ảnh, hầu như không có cặp genuine nào).
MIN_GENUINE_PAIRS_RELIABLE = 30


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

    # [SỬA — bug phát hiện qua review Q1] TRƯỚC ĐÂY tính accuracy bằng
    # macro-average theo BATCH (cộng dồn accuracy TỪNG batch rồi chia cho
    # n_batches) — không phải micro-average đúng nghĩa top-1 (tổng số dự đoán
    # đúng / tổng số ảnh). Với test set không chia hết cho batch_size (ví dụ
    # 3480 % 64 = 24), batch CUỐI nhỏ hơn vẫn được tính TRỌNG SỐ NGANG BẰNG
    # mọi batch khác trong trung bình — sai lệch so với true top-1 accuracy
    # trên từng mẫu (không lớn, cùng hướng ở mọi domain/backbone nên so sánh
    # TƯƠNG ĐỐI giữa các domain vẫn hợp lệ, nhưng SỐ TUYỆT ĐỐI báo cáo không
    # đúng định nghĩa chuẩn "top-1 accuracy"). Code đã sẵn tích luỹ
    # all_id_preds/all_id_labels cho toàn bộ test set (dùng cho confusion
    # matrix) — tận dụng luôn để tính ĐÚNG micro-average, thay vì tính thêm 1
    # cột riêng gây nhầm giữa 2 định nghĩa.
    all_embeddings, all_id_labels, all_id_preds = [], [], []
    all_id_rank5_hits, all_gender_preds, all_gender_labels = [], [], []
    with torch.no_grad():
        for imgs, id_labels, gender_labels in test_loader:
            imgs = imgs.to(device)
            id_labels = id_labels.to(device)
            gender_labels = gender_labels.to(device)

            id_logits, gender_logits, emb = model(imgs)
            k = min(5, id_logits.size(1))
            rank5_hit = (id_logits.topk(k, dim=1).indices == id_labels.unsqueeze(1)).any(dim=1)

            all_embeddings.append(emb.cpu())
            all_id_labels.append(id_labels.cpu())
            all_id_preds.append(id_logits.argmax(dim=1).cpu())
            all_id_rank5_hits.append(rank5_hit.cpu())
            all_gender_preds.append(gender_logits.argmax(dim=1).cpu())
            all_gender_labels.append(gender_labels.cpu())

    # Cat 1 lần duy nhất, dùng lại cho cả accuracy (micro-average đúng) lẫn
    # AUC/EER/confusion matrix bên dưới — tránh cat trùng lặp.
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_id_labels = torch.cat(all_id_labels, dim=0)
    all_id_preds = torch.cat(all_id_preds, dim=0)
    all_gender_preds = torch.cat(all_gender_preds, dim=0)
    all_gender_labels = torch.cat(all_gender_labels, dim=0)

    id_acc = (all_id_preds == all_id_labels).float().mean().item()
    id_rank5_acc = torch.cat(all_id_rank5_hits, dim=0).float().mean().item()
    gender_acc = (all_gender_preds == all_gender_labels).float().mean().item()
    params_m = count_params(model)
    latency_ms = measure_latency(model, (1, 3, image_size, image_size), device)

    # --- [MỚI] Verification setting: AUC/EER qua toàn bộ cặp genuine/impostor ---
    genuine_scores, impostor_scores = compute_pairwise_genuine_impostor_scores(
        all_embeddings, all_id_labels)
    roc_result = compute_roc_auc_eer(genuine_scores, impostor_scores)
    if roc_result["n_genuine_pairs"] < MIN_GENUINE_PAIRS_RELIABLE:
        print(f"CẢNH BÁO: chỉ {roc_result['n_genuine_pairs']} cặp genuine trong test set "
              f"(< ngưỡng {MIN_GENUINE_PAIRS_RELIABLE}) — AUC/EER verification-setting bên dưới "
              f"CÓ THỂ KHÔNG ĐỦ TIN CẬY để báo cáo trong bài báo, chỉ nên dùng tham khảo.")

    # --- [MỚI] Confusion matrix (identity), lưu riêng ra CSV nếu có --out_json ---
    confusion_csv_path = None
    if args.out_json:
        cm = compute_confusion_matrix(all_id_preds, all_id_labels, num_classes=cfg["num_identities"])
        confusion_csv_path = str(Path(args.out_json).with_name(Path(args.out_json).stem + "_confusion.csv"))
        Path(confusion_csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(confusion_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true_label\\pred_label"] + list(range(cfg["num_identities"])))
            for row_idx in range(cfg["num_identities"]):
                writer.writerow([row_idx] + cm[row_idx].tolist())

    result = {
        "backbone": args.backbone,
        "train_domain": args.train_domain,
        "test_domain": args.test_domain,
        "config_name": f"{args.backbone}__{args.train_domain}_{args.test_domain}",
        "identity_accuracy": round(id_acc, 4),
        "identity_accuracy_rank5": round(id_rank5_acc, 4),
        "gender_accuracy": round(gender_acc, 4),
        "identity_auc": round(roc_result["auc"], 4),
        "identity_eer": round(roc_result["eer"], 4),
        "identity_eer_threshold": round(roc_result["eer_threshold"], 4),
        "n_genuine_pairs": roc_result["n_genuine_pairs"],
        "n_impostor_pairs": roc_result["n_impostor_pairs"],
        "params_M": round(params_m, 3),
        "latency_ms": round(latency_ms, 3),
    }

    print(f"\n{'=' * 60}")
    print(f"KẾT QUẢ TEST | backbone={args.backbone} | "
          f"train_domain={args.train_domain} | test_domain={args.test_domain}")
    print(f"{'=' * 60}")
    print(f"  IDENTITY ACCURACY (rank-1) : {id_acc:.4f}")
    print(f"  IDENTITY RANK-5            : {id_rank5_acc:.4f}")
    print(f"  GENDER ACCURACY            : {gender_acc:.4f}")
    print(f"  IDENTITY AUC (verification): {roc_result['auc']:.4f}  "
          f"(n_genuine={roc_result['n_genuine_pairs']}, n_impostor={roc_result['n_impostor_pairs']})")
    print(f"  IDENTITY EER (verification): {roc_result['eer']:.4f}  "
          f"(ngưỡng={roc_result['eer_threshold']:.4f})")
    print(f"  Params (M)                 : {params_m:.3f}")
    print(f"  Latency (ms)               : {latency_ms:.3f}")
    if confusion_csv_path:
        print(f"  Confusion matrix           : {confusion_csv_path}")
    print(f"{'=' * 60}\n")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
