"""
[MỚI] Trích xuất thông tin môi trường training cho bài báo (Methods/
Reproducibility): phiên bản PyTorch/torchvision/timm và tên/số lượng GPU --
thay cho việc để trống \\open{} trong paper/main.tex.

Chạy trên server (nơi thực sự chạy training):
    python scripts/report_training_environment.py
"""


def main():
    print("=== Phiên bản thư viện ===")
    try:
        import torch
        print(f"torch: {torch.__version__}")
        print(f"CUDA (build với torch): {torch.version.cuda}")
        print(f"cuDNN: {torch.backends.cudnn.version()}")
    except ImportError:
        print("torch: KHÔNG cài đặt trong môi trường này")
        torch = None

    try:
        import torchvision
        print(f"torchvision: {torchvision.__version__}")
    except ImportError:
        print("torchvision: KHÔNG cài đặt trong môi trường này")

    try:
        import timm
        print(f"timm: {timm.__version__}")
    except ImportError:
        print("timm: KHÔNG cài đặt trong môi trường này")

    print("\n=== GPU ===")
    if torch is not None and torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print(f"Số GPU khả dụng: {n_gpu}")
        for i in range(n_gpu):
            prop = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {prop.name} | {prop.total_memory / 1e9:.1f} GB VRAM "
                  f"| compute capability {prop.major}.{prop.minor}")
    else:
        print("Không có GPU CUDA khả dụng trong môi trường này (hoặc torch chưa cài).")


if __name__ == "__main__":
    main()
