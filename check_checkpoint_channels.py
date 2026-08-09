"""Kiểm tra TRỰC TIẾP checkpoint span_pretrained_x4.pth đang dùng thật sự có
bao nhiêu kênh (feature_channels) — sự thật duy nhất đáng tin, không dựa vào
bảng tổng hợp thứ cấp có thể bị gán nhầm nhãn.

Chạy:
    python check_checkpoint_channels.py --ckpt checkpoints/span_pretrained_x4.pth
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/span_pretrained_x4.pth")
    args = ap.parse_args()

    state_dict = torch.load(args.ckpt, map_location="cpu")
    if isinstance(state_dict, dict) and "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif isinstance(state_dict, dict) and "params" in state_dict:
        state_dict = state_dict["params"]

    print(f"Checkpoint: {args.ckpt}")
    print(f"Tổng số tensor trong state_dict: {len(state_dict)}\n")

    # conv_1.sk.weight có shape (feature_channels, 3, 1, 1) -> kênh đầu ra = feature_channels
    total_params = 0
    for key, tensor in state_dict.items():
        total_params += tensor.numel()
        if "conv_1.sk.weight" in key or "conv_1.eval_conv.weight" in key:
            print(f"{key}: shape={tuple(tensor.shape)} -> feature_channels THẬT = {tensor.shape[0]}")

    print(f"\nTổng số tham số trong checkpoint (đúng những gì được LƯU, không suy đoán): "
          f"{total_params:,} ({total_params/1e6:.4f}M)")

    print("\nDanh sách 10 tensor đầu tiên (để đối chiếu cấu trúc):")
    for i, (key, tensor) in enumerate(state_dict.items()):
        if i >= 10:
            break
        print(f"  {key}: {tuple(tensor.shape)}")


if __name__ == "__main__":
    main()
