"""
Quét toàn bộ runs/*/train.log, trích xuất thông tin hội tụ thành 1 bảng CSV —
dùng cho phần phụ lục phương pháp (training details) của journal: epoch thực
tế đã chạy, có dừng sớm không, số lần fallback CPU do OOM, thời gian train,
best val metric đạt được.

Chạy:
    python data/export_training_log_summary.py --runs_root runs --out_csv results/training_summary.csv
"""
import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

EPOCH_RE = re.compile(r"epoch (\d+)/(\d+)")
VAL_ID_ACC_RE = re.compile(r"VAL_ID_ACC=([\d.]+)")
VAL_L1_RE = re.compile(r"VAL_L1=([\d.]+)")
VAL_TOTAL_RE = re.compile(r"VAL_TOTAL=([\d.]+)")
OOM_RE = re.compile(r"CUDA hết bộ nhớ")
EARLY_STOP_RE = re.compile(r"EARLY STOPPING")
TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def parse_log(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None

    first_ts, last_ts = None, None
    max_epoch, total_epoch_lines = 0, 0
    oom_count = 0
    early_stopped = False
    best_val_id_acc, best_val_l1, best_val_total = None, None, None

    for line in lines:
        m_ts = TIMESTAMP_RE.match(line)
        if m_ts:
            ts = datetime.strptime(m_ts.group(1), "%Y-%m-%d %H:%M:%S")
            if first_ts is None:
                first_ts = ts
            last_ts = ts

        m_epoch = EPOCH_RE.search(line)
        if m_epoch:
            total_epoch_lines += 1
            max_epoch = max(max_epoch, int(m_epoch.group(1)))

        if OOM_RE.search(line):
            oom_count += 1
        if EARLY_STOP_RE.search(line):
            early_stopped = True

        m = VAL_ID_ACC_RE.search(line)
        if m:
            val = float(m.group(1))
            if best_val_id_acc is None or val > best_val_id_acc:
                best_val_id_acc = val

        m = VAL_L1_RE.search(line)
        if m:
            val = float(m.group(1))
            if best_val_l1 is None or val < best_val_l1:
                best_val_l1 = val

        m = VAL_TOTAL_RE.search(line)
        if m:
            val = float(m.group(1))
            if best_val_total is None or val < best_val_total:
                best_val_total = val

    duration_min = None
    if first_ts and last_ts:
        duration_min = round((last_ts - first_ts).total_seconds() / 60, 1)

    return {
        "run_name": path.parent.name,
        "epochs_run": max_epoch,
        "early_stopped": early_stopped,
        "oom_fallback_count": oom_count,
        "duration_minutes": duration_min if duration_min is not None else "NA",
        "best_val_id_acc": best_val_id_acc if best_val_id_acc is not None else "NA",
        "best_val_l1": best_val_l1 if best_val_l1 is not None else "NA",
        "best_val_total": best_val_total if best_val_total is not None else "NA",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--out_csv", default="results/training_summary.csv")
    args = ap.parse_args()

    rows = []
    for log_path in sorted(Path(args.runs_root).glob("*/train.log")):
        row = parse_log(log_path)
        if row:
            rows.append(row)

    if not rows:
        print(f"Không tìm thấy file train.log nào trong {args.runs_root}/*/train.log")
        return

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["run_name", "epochs_run", "early_stopped", "oom_fallback_count",
                  "duration_minutes", "best_val_id_acc", "best_val_l1", "best_val_total"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Đã ghi {out_path} — {len(rows)} run.")


if __name__ == "__main__":
    main()
