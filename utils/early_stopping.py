"""
Early stopping tiêu chuẩn: theo dõi 1 metric trên val (accuracy hoặc loss),
dừng train nếu không cải thiện sau `patience` epoch liên tiếp.
"""

import torch


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
        là giá trị tốt nhất từ trước đến giờ (dùng để quyết định lưu checkpoint).

        [SỬA — LỖI PHÁT HIỆN QUA CODE REVIEW] Nếu `value` là NaN (ví dụ epoch
        đầu tiên val toàn batch NaN — hiếm với pixel/distill loss thuần,
        nhưng khả năng xảy ra tăng lên khi có thêm identity/saliency loss đi
        qua nhiều judge/cosine-similarity/autograd.grad), TRƯỚC ĐÂY:
        `self.best is None` khiến `is_better=True` bất kể `value` là gì ->
        `self.best` bị gán = NaN. MỌI so sánh SAU ĐÓ (`value > NaN` hay
        `value < NaN`) trong Python LUÔN trả về False (thuộc tính chuẩn của
        NaN, không phải bug so sánh) -> is_better VĨNH VIỄN = False từ đó về
        sau, dù các epoch tiếp theo có loss hợp lệ, tốt tới đâu -> KHÔNG BAO
        GIỜ lưu được checkpoint tốt hơn nữa, chỉ dừng lại (early-stop) sau
        đúng `patience` epoch với checkpoint duy nhất đã lưu (nếu có) là bản
        NaN của chính epoch đầu (không phản ánh gì về chất lượng model).
        Sửa: coi NaN là "không tốt hơn" (KHÔNG chạm vào self.best, để lượt
        gọi sau — kể cả khi self.best vẫn còn None — có cơ hội trở thành best
        hợp lệ đầu tiên), đồng thời vẫn tính vào bộ đếm patience (coi như 1
        epoch không cải thiện, nhất quán với epoch có giá trị hợp lệ nhưng
        tệ hơn)."""
        # NaN != NaN luôn True; inf/-inf cũng không phải metric hợp lệ để chọn ckpt.
        if value != value or value in (float("inf"), float("-inf")):
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

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

    @property
    def best_str(self) -> str:
        """In log an toàn khi chưa có epoch hợp lệ nào (best is None) — tránh
        TypeError `None:.4f` làm sập job ngay lúc early-stop/kết thúc train."""
        return "NA" if self.best is None else f"{self.best:.4f}"


def save_state_dict(module, path):
    """Lưu state_dict lên CPU (load lại được bất kể device lúc train)."""
    torch.save({k: v.cpu() for k, v in module.state_dict().items()}, path)


def save_last_if_missing(path, module, logger, what: str = "best.pt"):
    """[SỬA — bổ sung sau code review, vòng 4, điểm 3] Nếu `EarlyStopping`
    chưa từng ghi nhận is_best=True (val toàn NaN/Inf suốt training — sau khi
    step() coi NaN/Inf là "không tốt hơn" thay vì đầu độc self.best, kịch bản
    NÀY vẫn xảy ra được nếu MỌI epoch đều NaN/Inf), file `path` (thường là
    best.pt) sẽ KHÔNG BAO GIỜ được tạo ra qua đường `if is_best: save(...)`
    thông thường -> bước pipeline SAU (data/build_sr.py, eval_*.py) sẽ
    FileNotFoundError khi cố load nó.

    Hàm này gọi ở CUỐI vòng train (sau khi thoát for-loop) để đảm bảo LUÔN
    có 1 file tại `path`: nếu đã có (tức từng is_best >= 1 lần, best.pt hợp
    lệ) thì KHÔNG làm gì; nếu chưa có, lưu trọng số EPOCH CUỐI CÙNG vào đó
    (model gần như chắc chắn KHÔNG DÙNG ĐƯỢC — vì mọi epoch đều NaN/Inf, tức
    training đã hỏng ngay từ đầu, không phải epoch cuối "kém hơn epoch đầu")
    và LOG CẢNH BÁO rõ ràng ghi nhận điều này, để:
      (a) pipeline không bị crash giữa chừng vì thiếu file (dễ debug hơn khi
          chạy loạt lệnh tự động, ví dụ multi-seed/ablation),
      (b) người chạy KHÔNG NHẦM checkpoint này là "tốt nhất" thật sự — phải
          đọc log/quay lại kiểm tra nguyên nhân NaN (learning rate quá cao,
          identity/saliency loss chưa ổn định, v.v.) trước khi tin bất kỳ số
          liệu downstream nào tính từ checkpoint này.
    """
    from pathlib import Path
    path = Path(path)
    if path.exists():
        return
    if logger:
        logger.warning(
            f"[EarlyStopping] KHÔNG có checkpoint {what} hợp lệ (val toàn "
            f"NaN/Inf suốt training, chưa từng is_best=True). Đây LÀ DẤU HIỆU "
            f"TRAINING ĐÃ HỎNG (không phải chỉ epoch cuối tệ hơn epoch đầu) — "
            f"KHÔNG dùng số liệu downstream tính từ checkpoint {what} này để "
            f"kết luận bất cứ điều gì, chỉ dùng để pipeline không crash vì "
            f"thiếu file. Lưu tạm trọng số epoch cuối vào {path}.")
    save_state_dict(module, path)
