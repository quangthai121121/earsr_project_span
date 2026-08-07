"""
Logger ghi đồng thời ra màn hình VÀ ra file trong thư mục run — để bạn biết
tiến trình chạy tới đâu ngay cả khi không ngồi theo dõi màn hình liên tục
(ví dụ chạy qua SSH, mất kết nối, hoặc muốn xem lại log sau khi train xong).
"""
import logging
from pathlib import Path


def setup_logger(run_dir, name: str = "train") -> logging.Logger:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{name}.log"

    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()   # tránh log trùng lặp nếu gọi lại nhiều lần

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    logger.info(f"Bắt đầu ghi log tại: {log_path}")
    return logger
