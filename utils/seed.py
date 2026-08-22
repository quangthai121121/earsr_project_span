import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Không ép cuDNN deterministic thì cùng 1 seed vẫn có thể ra kết quả khác
    # nhau giữa các lần chạy trên GPU (cuDNN tự chọn thuật toán không xác định).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """[MỚI — phát hiện qua review Q1] DataLoader worker subprocess (khi
    num_workers>0) không tự động thừa hưởng seed của process chính theo cách
    tường minh — nếu dataset có augmentation ngẫu nhiên (numpy/random, khác
    với torch RNG), mỗi worker có thể lệch nhau/lệch giữa các lần chạy dù đã
    gọi set_seed(). Công thức chuẩn của PyTorch: suy seed riêng cho mỗi
    worker từ torch.initial_seed() (đã được DataLoader gán xác định dựa trên
    seed chính + worker_id), rồi seed lại random/numpy TRONG worker đó.
    Dùng cùng với generator cố định (xem seeded_generator()) khi tạo
    DataLoader(..., worker_init_fn=seed_worker, generator=seeded_generator(seed))."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seeded_generator(seed: int) -> torch.Generator:
    """[MỚI] torch.Generator cố định seed — truyền vào DataLoader(generator=...)
    để thứ tự shuffle không phụ thuộc trạng thái RNG toàn cục ngầm định,
    tường minh và tái lập được cùng 1 seed."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
