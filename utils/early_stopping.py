"""
Early stopping tiêu chuẩn: theo dõi 1 metric trên val (accuracy hoặc loss),
dừng train nếu không cải thiện sau `patience` epoch liên tiếp.
"""


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max"):
        """
        mode="max": metric càng cao càng tốt (ví dụ accuracy)
        mode="min": metric càng thấp càng tốt (ví dụ loss)
        """
        assert mode in ("max", "min")
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Gọi sau mỗi epoch với giá trị metric trên val. Trả về True nếu đây
        là giá trị tốt nhất từ trước đến giờ (dùng để quyết định lưu checkpoint)."""
        is_better = (
            self.best is None
            or (value > self.best if self.mode == "max" else value < self.best)
        )
        if is_better:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return is_better
