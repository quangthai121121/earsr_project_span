"""
Tạo config FINE-TUNE từ 1 config dataset đích có sẵn (ví dụ
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

    # [SỬA] Bản trước chỉ giảm mục "recognition" và "sr_improve" (dùng cho
    # span_tiny), QUÊN mục "sr" (dùng cho span_baseline/edsr/rlfn/ecbsr/safmn
    # qua train_sr.py) — phát hiện khi mở rộng script để fine-tune CẢ
    # span_baseline (đợt 3). Nếu không sửa, span_baseline (và mọi baseline SR
    # khác train qua train_sr.py) sẽ fine-tune với LR/epoch ĐẦY ĐỦ như train
    # from-scratch (1e-3, 100 epoch) — quá mạnh, rủi ro phá nhanh trọng số
    # tốt đã học từ dataset nguồn. Giờ giảm đồng bộ cả 3 mục bằng vòng lặp
    # (không liệt kê tay từng field) — thêm kiến trúc SR mới nào sau này vẫn
    # tự động được xử lý đúng miễn dùng chung mục "sr" trong config.
    sections = ["recognition", "sr_improve", "sr"]
    old_vals = {}
    for sec in sections:
        old_vals[sec] = {
            "lr": cfg[sec]["lr"],
            "max_epochs": cfg[sec]["max_epochs"],
            "patience": cfg[sec]["patience"],
        }
        cfg[sec]["lr"] = old_vals[sec]["lr"] / args.lr_divide
        cfg[sec]["max_epochs"] = max(args.min_epochs,
                                      int(round(old_vals[sec]["max_epochs"] / args.epoch_divide)))
        # patience: giữ nguyên nếu vẫn nhỏ hơn max_epochs mới; nếu không thì co
        # lại còn nửa max_epochs mới (tối thiểu 5) để early-stopping vẫn hợp lý.
        cfg[sec]["patience"] = min(old_vals[sec]["patience"], max(5, cfg[sec]["max_epochs"] // 2))

    with open(args.out_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"Đã tạo {args.out_config} từ {args.in_config}:")
    for sec in sections:
        print(f"  {sec}.lr        : {old_vals[sec]['lr']} -> {cfg[sec]['lr']}")
        print(f"  {sec}.max_epochs: {old_vals[sec]['max_epochs']} -> {cfg[sec]['max_epochs']}")
        print(f"  {sec}.patience  : {old_vals[sec]['patience']} -> {cfg[sec]['patience']}")
    print("  (mọi field khác giữ nguyên như config đích gốc — cùng splits_root/runs_root/"
          "results_root/num_identities/teacher_ckpt/frozen_recognition_ckpt)")


if __name__ == "__main__":
    main()
