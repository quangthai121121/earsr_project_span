"""
[MỚI — Trụ cột 6, reviewer round 2] Gộp n=5 seed cho eval_frequency_ablation.py
(pipeline/run_frequency_ablation.sh, Phần B) -- tính mean/std identity_accuracy
mỗi điều kiện lọc (mode, cutoff_frac) qua 5 file CSV per-seed.

Chạy:
    python data/aggregate_frequency_ablation.py --results_dir results/frequency_ablation \
        --backbone mobilenet_v2 --out_csv results/frequency_ablation/mobilenet_v2_summary.csv
"""
import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

SEED_RE = re.compile(r"_seed(\d+)\.csv$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--backbone", required=True,
                     help="chỉ gộp các file <backbone>_seed<seed>.csv, bỏ qua file khác trong "
                          "cùng thư mục (ví dụ spectral_content.csv của Phần A).")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    # key = (mode, cutoff_frac:str) -> {seed:int -> accuracy}
    by_key = defaultdict(dict)
    n_files = 0
    for f in sorted(Path(args.results_dir).glob(f"{args.backbone}_seed*.csv")):
        m = SEED_RE.search(f.name)
        if not m:
            continue
        seed = int(m.group(1))
        n_files += 1
        with open(f, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (row["mode"], row["cutoff_frac"])
                by_key[key][seed] = float(row["identity_accuracy"])

    if n_files == 0:
        print(f"Không tìm thấy file {args.backbone}_seed*.csv nào trong {args.results_dir}")
        return
    print(f"Đã đọc {n_files} file seed cho backbone={args.backbone}")

    rows = []
    for (mode, cutoff_frac), by_seed in by_key.items():
        seeds_present = sorted(by_seed.keys())
        values = [by_seed[s] for s in seeds_present]
        n = len(values)
        mean = statistics.mean(values)
        std = statistics.stdev(values) if n > 1 else 0.0
        rows.append({
            "backbone": args.backbone, "mode": mode, "cutoff_frac": float(cutoff_frac),
            "n_seeds": n, "mean_identity_accuracy": round(mean, 4),
            "std_identity_accuracy": round(std, 4),
            "seeds": ";".join(str(s) for s in seeds_present),
        })

    # [SỬA — phát hiện qua review lại lần 2] Trước đây so sánh r["n_seeds"] với
    # kết quả gọi lại glob() LẦN NỮA (không dùng n_files đã đếm ở trên) -- 2
    # lần đếm dùng 2 tiêu chí khác nhau (glob thô vs. regex SEED_RE đã lọc ở
    # vòng lặp trên), có thể lệch nhau trong trường hợp hiếm (file khớp glob
    # nhưng seed không phải số nguyên) khiến cảnh báo không kích hoạt đúng lúc
    # cần. Dùng lại đúng n_files đã đếm bằng CÙNG tiêu chí (regex) thay vì
    # đếm lại bằng tiêu chí khác.
    if any(r["n_seeds"] < n_files for r in rows):
        print("CẢNH BÁO: một số điều kiện lọc không có đủ seed như số file đọc được -- "
              "kiểm tra lại các file CSV per-seed có cùng danh sách (mode, cutoff_frac) không.")

    # Sắp theo mode rồi theo cutoff_frac tăng dần, dễ đọc thành đường cong.
    rows.sort(key=lambda r: (r["mode"], r["cutoff_frac"]))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã ghi {out_path}")

    print("\n== Đường cong accuracy-vs-cutoff (mean +- std qua seed) ==")
    for mode in ["lowpass", "highpass"]:
        print(f"-- {mode} --")
        for r in rows:
            if r["mode"] == mode:
                print(f"  cutoff={r['cutoff_frac']}: {r['mean_identity_accuracy']} +- "
                      f"{r['std_identity_accuracy']} (n={r['n_seeds']})")


if __name__ == "__main__":
    main()
