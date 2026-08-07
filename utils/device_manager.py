"""
DeviceManager: chạy ưu tiên GPU, nhưng nếu gặp CUDA out-of-memory ngay giữa
lúc train (ví dụ tiến trình khác trên cùng máy đang chiếm VRAM), TỰ ĐỘNG
chuyển batch đó sang CPU để không crash toàn bộ quá trình train, rồi tự động
thử lại GPU sau một số batch "nghỉ" (cooldown) — không cần bạn can thiệp tay.

Cách dùng trong vòng lặp train (xem train_recognition.py để có ví dụ đầy đủ):

    device_mgr = DeviceManager(logger)
    for batch in loader:
        device = device_mgr.current_device()
        try:
            ... forward/backward trên `device` ...
        except torch.cuda.OutOfMemoryError:
            device_mgr.report_oom()
            ... thử lại chính batch đó trên CPU ...
"""
import torch


class DeviceManager:
    def __init__(self, logger=None, cooldown_batches: int = 20):
        self.preferred = "cuda" if torch.cuda.is_available() else "cpu"
        self._current = self.preferred
        self._cooldown_remaining = 0
        self._cooldown_batches = cooldown_batches
        self.total_oom_events = 0
        self.logger = logger

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def current_device(self) -> str:
        """Gọi ở đầu mỗi batch. Nếu đang trong thời gian 'nghỉ' sau OOM thì vẫn
        dùng CPU; hết thời gian nghỉ thì tự động thử lại GPU."""
        if self.preferred == "cpu":
            return "cpu"

        if self._current == "cpu" and self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining == 0:
                self._log(f"[DeviceManager] Hết thời gian nghỉ, thử lại GPU...")
                self._current = "cuda"

        return self._current

    def report_oom(self):
        """Gọi khi bắt được lỗi CUDA OOM — chuyển sang CPU, đặt thời gian nghỉ
        trước khi tự thử lại GPU."""
        self.total_oom_events += 1
        self._current = "cpu"
        self._cooldown_remaining = self._cooldown_batches
        torch.cuda.empty_cache()
        self._log(
            f"[DeviceManager] !!! CUDA hết bộ nhớ (lần thứ {self.total_oom_events}). "
            f"Tạm chuyển sang CPU cho batch này, sẽ tự thử lại GPU sau "
            f"{self._cooldown_batches} batch."
        )

    def report_recovered(self):
        """Gọi khi 1 batch chạy thành công trên GPU sau khi từng OOM — xác nhận
        GPU đã rảnh trở lại."""
        if self.total_oom_events > 0 and self._current == "cuda" and self._cooldown_remaining == 0:
            pass  # đã log ở current_device(), không log lặp lại mỗi batch


def move_optimizer_state(optimizer, device: str):
    """Di chuyển toàn bộ tensor trạng thái của optimizer (momentum, exp_avg...)
    sang device mới — bắt buộc phải làm khi đổi device giữa chừng, nếu không
    optimizer.step() sẽ lỗi do param và state nằm khác device nhau."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
