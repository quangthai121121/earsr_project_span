"""
[MỚI] Tạo config FINE-TUNE từ 1 config dataset đích có sẵn (ví dụ
configs/config_awe.yaml) — giảm learning rate và số epoch tối đa so với bản
gốc (chuẩn thực hành fine-tune/transfer learning: LR thấp hơn để không phá
hỏng trọng số đã học được từ dataset nguồn, epoch ít hơn vì chỉ cần thích
nghi chứ không học lại từ đầu).

QUAN TRỌNG: số liệu LR/epoch được TÍNH TỪ giá trị thật đang có trong config
đích (chia theo hệ số), KHÔNG hardcode số cố định — tránh áp sai công thức
nếu sau này config gốc thay đổi.

Chạy:
    python scripts/make_finetune_config.py \
        --in_config configs/config_awe.yaml \
        --out_config configs/config_awe_finetune.yaml
"""
import argparse

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_config", required=True)
    ap.add_argument("--out_config", required=True)
    ap.add_argument("--lr_divide", type=float, default=10.0,
                     help="chia learning rate gốc cho hệ số này (mặc định /10, chuẩn fine-tune)")
    ap.add_argument("--epoch_divide", type=float, default=3.0,
                     help="chia max_epochs gốc cho hệ số này (mặc định /3)")
    ap.add_argument("--min_epochs", type=int, default=10,
                     help="sàn tối thiểu cho max_epochs sau khi chia, tránh quá ít epoch")
    args = ap.parse_args()

    with open(args.in_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    old_rec_lr = cfg["recognition"]["lr"]
    old_sr_lr = cfg["sr_improve"]["lr"]
    old_rec_epochs = cfg["recognition"]["max_epochs"]
    old_sr_epochs = cfg["sr_improve"]["max_epochs"]
    old_rec_patience = cfg["recognition"]["patience"]
    old_sr_patience = cfg["sr_improve"]["patience"]

    cfg["recognition"]["lr"] = old_rec_lr / args.lr_divide
    cfg["sr_improve"]["lr"] = old_sr_lr / args.lr_divide
    cfg["recognition"]["max_epochs"] = max(args.min_epochs, int(round(old_rec_epochs / args.epoch_divide)))
    cfg["sr_improve"]["max_epochs"] = max(args.min_epochs, int(round(old_sr_epochs / args.epoch_divide)))
    # patience: giữ nguyên nếu vẫn nhỏ hơn max_epochs mới; nếu không thì co lại
    # còn nửa max_epochs mới (tối thiểu 5) để early-stopping vẫn hoạt động hợp lý.
    cfg["recognition"]["patience"] = min(old_rec_patience, max(5, cfg["recognition"]["max_epochs"] // 2))
    cfg["sr_improve"]["patience"] = min(old_sr_patience, max(5, cfg["sr_improve"]["max_epochs"] // 2))

    with open(args.out_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"Đã tạo {args.out_config} từ {args.in_config}:")
    print(f"  recognition.lr        : {old_rec_lr} -> {cfg['recognition']['lr']}")
    print(f"  sr_improve.lr         : {old_sr_lr} -> {cfg['sr_improve']['lr']}")
    print(f"  recognition.max_epochs: {old_rec_epochs} -> {cfg['recognition']['max_epochs']}")
    print(f"  sr_improve.max_epochs : {old_sr_epochs} -> {cfg['sr_improve']['max_epochs']}")
    print(f"  recognition.patience  : {old_rec_patience} -> {cfg['recognition']['patience']}")
    print(f"  sr_improve.patience   : {old_sr_patience} -> {cfg['sr_improve']['patience']}")
    print("  (mọi field khác giữ nguyên như config đích gốc — cùng splits_root/runs_root/"
          "results_root/num_identities/teacher_ckpt/frozen_recognition_ckpt)")


if __name__ == "__main__":
    main()
