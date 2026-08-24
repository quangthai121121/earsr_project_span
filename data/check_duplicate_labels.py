"""Quet TOAN BO CSV do eval_sr_quality.py sinh ra (results/**/*.csv co cot
'label') de tim NHAN TRUNG LAP — dau hieu du lieu tu 2 lan chay KHAC NHAU
(vi du 2 recipe lambda khac nhau) bi LAN VAO CUNG 1 file do eval_sr_quality.py
ghi APPEND (khong overwrite) va nhieu script goi no khong co chot chan chay
lai (xem RUNBOOK_EarVN1.0.md, su co thuc te 2026-08-24 voi
results/lambda_saliency_sweep/sr_quality_sweep.csv).

Day la cong cu CHUNG, khong phu thuoc script nao sinh ra CSV do — thay vi va
tung diem goi eval_sr_quality.py rieng le (de sot), quet TRIEU CHUNG (nhan
trung lap) o BAT KY dau trong results/.

Cach dung:
    python3 data/check_duplicate_labels.py
    python3 data/check_duplicate_labels.py --root results

Exit code: 0 = sach (khong nhan nao trung), 1 = phat hien nhan trung lap.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def scan_csv(path: Path):
    """Tra ve dict {label: [row_indices]} cho cac label xuat hien >1 lan."""
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "label" not in reader.fieldnames:
                return None  # khong phai CSV kieu eval_sr_quality.py, bo qua
            label_rows = defaultdict(list)
            for idx, row in enumerate(reader):
                label_rows[row["label"]].append((idx, row))
    except Exception as e:
        return f"[LOI DOC FILE] {e}"

    return {label: rows for label, rows in label_rows.items() if len(rows) > 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", nargs="+", default=["results"],
                     help="Cac thu muc goc can quet CSV (mac dinh: results)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    csv_files = []
    for root_name in args.root:
        root = repo_root / root_name
        if root.exists():
            csv_files.extend(sorted(root.rglob("*.csv")))

    if not csv_files:
        print(f"[CANH BAO] Khong tim thay file CSV nao trong: {args.root}")
        sys.exit(0)

    print(f"Tim thay {len(csv_files)} file CSV. Dang quet cot 'label'...\n")

    total_bad_files = 0
    for path in csv_files:
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            rel = path
        result = scan_csv(path)
        if result is None:
            continue  # khong co cot 'label', khong phai loai CSV nay quan tam
        if isinstance(result, str):
            print(f"{result}: {rel}")
            total_bad_files += 1
            continue
        if not result:
            continue
        total_bad_files += 1
        print(f"[NHAN TRUNG LAP] {rel}")
        for label, rows in result.items():
            print(f"    label='{label}' xuat hien {len(rows)} lan:")
            for idx, row in rows:
                # in vai cot dau (bo 'label') de de doi chieu gia tri khac nhau the nao
                preview = {k: v for k, v in row.items() if k != "label"}
                preview_str = ", ".join(f"{k}={v}" for k, v in list(preview.items())[:4])
                print(f"        dong {idx}: {preview_str}")
        print()

    print("=" * 60)
    if total_bad_files == 0:
        print(f"KET QUA: SACH. Quet {len(csv_files)} file, khong co nhan trung lap nao.")
        sys.exit(0)
    else:
        print(f"KET QUA: PHAT HIEN LOI. {total_bad_files} file co nhan trung lap.")
        print("=> Rat co the do 1 script bi dung giua chung roi chay lai ma khong don")
        print("   thu muc ket qua cu (eval_sr_quality.py ghi APPEND, khong overwrite).")
        print("   Kiem tra thu cong tung dong trung de biet dong nao la du lieu THAT,")
        print("   dong nao la rac tu lan chay truoc, roi xoa dong rac truoc khi tong hop.")
        sys.exit(1)


if __name__ == "__main__":
    main()
