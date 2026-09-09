"""
[MỚI — Trụ cột 6, reviewer round 2, Phần B] Bài báo lập luận (Section~
sec:positioning) rằng nén sâu (span_tiny) an toàn vì tín hiệu nhận dạng
trong ảnh tai độ phân giải thấp tập trung ở cấu trúc tần số THẤP. Thí
nghiệm DUY NHẤT đã chạy liên quan (attention-parameterization ablation) chỉ
bác bỏ một giả thuyết KHÁC (attention cố định), không trực tiếp kiểm chứng
giả thuyết tần số thấp. Script này kiểm chứng TRỰC TIẾP: xoá bỏ 1 dải tần
số khỏi ẢNH ĐẦU VÀO rồi đánh giá lại một checkpoint recognition ĐÃ TRAIN
SẴN -- KHÔNG train lại gì.

Logic: nếu identity accuracy hầu như KHÔNG đổi khi xoá tần số CAO (lowpass,
chỉ giữ tần số thấp) nhưng SỤP xuống gần mức ngẫu nhiên khi xoá tần số THẤP
(highpass, chỉ giữ tần số cao), đó là bằng chứng trực tiếp thông tin nhận
dạng nằm ở tần số thấp -- đúng giả thuyết bài báo đang dùng để giải thích.
Nếu ngược lại hoặc không có khác biệt rõ, giả thuyết không được ủng hộ (đúng
tinh thần "báo cáo trung thực" đã áp dụng cho attention-parameterization
ablation trước đó -- kết quả âm cũng được báo cáo).

[SỬA — hiệu năng, phát hiện qua tự review TRƯỚC khi giao] Bản đầu lọc tần số
TRỰC TIẾP mỗi lần script này chạy -- vì script này được gọi 55 lần độc lập
(11 backbone x 5 seed) và bộ lọc KHÔNG phụ thuộc backbone/seed, điều đó lọc
lại CÙNG 16 điều kiện x toàn bộ tập test 55 LẦN GIỐNG HỆT NHAU (đo thực tế
lãng phí ~17 phút). Giờ chỉ ĐỌC LẠI 16 domain ảnh đã lọc SẴN 1 LẦN DUY NHẤT
bởi data/build_frequency_filtered_domains.py -- PHẢI chạy script đó trước
(pipeline/run_frequency_ablation.sh tự làm điều này).

Chạy (ví dụ, 1 backbone/1 seed -- lặp qua nhiều backbone/seed bằng pipeline/
run_frequency_ablation.sh):
    python data/build_frequency_filtered_domains.py --config configs/config.yaml
    python eval_frequency_ablation.py --config configs/config.yaml \
        --ckpt runs/recognition_sr_improved_mobilenet_v2_seed42/best.pt \
        --backbone mobilenet_v2 --domain sr_improved --seed_label 42 \
        --out_csv results/frequency_ablation/mobilenet_v2_seed42.csv
"""
import argparse
import csv
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.ear_dataset import EarDataset, build_label_map
from models.recognition_model import EarRecognitionNet, SUPPORTED_BACKBONES

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.build_frequency_filtered_domains import domain_dir_name, LOWPASS_CUTOFFS, HIGHPASS_CUTOFFS  # noqa: E402


def evaluate_one_condition(model, freq_ablation_root, splits_json, label_map, image_size,
                            filter_mode, cutoff_frac, batch_size, num_workers, device):
    domain_root = Path(freq_ablation_root) / domain_dir_name(filter_mode, cutoff_frac)
    if not domain_root.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy {domain_root} -- chạy data/build_frequency_filtered_domains.py trước.")
    dataset = EarDataset(str(domain_root), "test", splits_json, label_map, image_size, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, _gender in loader:
            imgs = imgs.to(device)
            id_logits, _, _ = model(imgs)
            all_preds.append(id_logits.argmax(dim=1).cpu())
            all_labels.append(labels)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return (all_preds == all_labels).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True,
                     help="checkpoint recognition ĐÃ TRAIN SẴN (KHÔNG train lại) -- ví dụ "
                          "runs/recognition_sr_improved_<backbone>_seed<seed>/best.pt")
    ap.add_argument("--backbone", required=True, choices=SUPPORTED_BACKBONES)
    ap.add_argument("--domain", required=True,
                     help="domain NGUỒN đã dùng để build các domain đã lọc (vd sr_improved cho "
                          "span_tiny) -- PHẢI khớp domain --ckpt đã train trên VÀ khớp "
                          "--source_domain đã dùng khi chạy build_frequency_filtered_domains.py")
    ap.add_argument("--freq_ablation_root", default=None,
                     help="mặc định: <splits_root>/freq_ablation_<domain> (khớp mặc định của "
                          "data/build_frequency_filtered_domains.py)")
    ap.add_argument("--seed_label", default=None, help="chỉ ghi vào CSV để phân biệt, không ảnh hưởng logic")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--only_cutoff", type=float, default=None,
                     help="[MỚI — kiểm chứng tổng quát hoá trên SAFMN/SMFANet] Không truyền -> "
                          "chạy ĐỦ 16 điều kiện như cũ (KHÔNG đổi hành vi mặc định, an toàn cho "
                          "mọi lời gọi hiện có, kể cả kết quả span_tiny đã dùng trong bài). Truyền "
                          "1 giá trị (ví dụ 0.1) -> CHỈ chạy đúng 2 điều kiện matched-severity "
                          "(lowpass VÀ highpass tại cutoff đó) thay vì 16 -- dùng cho bản sàng lọc "
                          "nhanh trước khi chạy full sweep. Giá trị PHẢI có mặt trong CẢ HAI "
                          "LOWPASS_CUTOFFS và HIGHPASS_CUTOFFS (xem data/build_frequency_filtered_"
                          "domains.py) -- nếu không, báo lỗi rõ ràng thay vì âm thầm bỏ qua.")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits_root = cfg["paths"]["splits_root"]
    splits_json = f"{splits_root}/splits.json"
    label_map = build_label_map(splits_json)
    image_size = cfg["image"]["hr_size"]
    batch_size = cfg["recognition"]["batch_size"]
    freq_ablation_root = args.freq_ablation_root or f"{splits_root}/freq_ablation_{args.domain}"

    if not Path(freq_ablation_root).is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy {freq_ablation_root} -- chạy data/build_frequency_filtered_domains.py "
            f"--source_domain {args.domain} trước.")
    if not Path(args.ckpt).is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint {args.ckpt}.")

    # [MỚI] Chặn NGAY nếu --only_cutoff không hợp lệ, TRƯỚC KHI load model
    # (tốn thời gian) -- fail-fast, đúng triết lý xuyên suốt project.
    lowpass_cutoffs, highpass_cutoffs = LOWPASS_CUTOFFS, HIGHPASS_CUTOFFS
    if args.only_cutoff is not None:
        if args.only_cutoff not in LOWPASS_CUTOFFS or args.only_cutoff not in HIGHPASS_CUTOFFS:
            raise ValueError(
                f"--only_cutoff {args.only_cutoff} phải có mặt trong CẢ HAI LOWPASS_CUTOFFS "
                f"({LOWPASS_CUTOFFS}) và HIGHPASS_CUTOFFS ({HIGHPASS_CUTOFFS}) -- domain đã lọc "
                f"tương ứng phải tồn tại sẵn (build bởi data/build_frequency_filtered_domains.py) "
                f"để đọc lại, không tự lọc mới ở đây.")
        lowpass_cutoffs, highpass_cutoffs = [args.only_cutoff], [args.only_cutoff]

    model = EarRecognitionNet(
        num_identities=cfg["num_identities"], num_genders=cfg["num_genders"],
        embedding_dim=cfg["recognition"]["embedding_dim"], backbone=args.backbone,
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    rows = []
    for mode, cutoffs in [("lowpass", lowpass_cutoffs), ("highpass", highpass_cutoffs)]:
        for cutoff in cutoffs:
            acc = evaluate_one_condition(model, freq_ablation_root, splits_json, label_map, image_size,
                                          mode, cutoff, batch_size, args.num_workers, device)
            print(f"  mode={mode} cutoff={cutoff}: identity_accuracy={acc:.4f}")
            rows.append({
                "backbone": args.backbone, "domain": args.domain, "seed": args.seed_label,
                "mode": mode, "cutoff_frac": cutoff, "identity_accuracy": round(acc, 4),
            })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã ghi {out_path}")


if __name__ == "__main__":
    main()
