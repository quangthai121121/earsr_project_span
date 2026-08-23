"""Quet toan bo CSV ket qua that (results/, runs/) de tim NaN/Inf THAT trong du lieu,
tach biet voi NaN noi bo cua cong thuc MDES (da sua trong mdes_paired(), khong lien quan
den du lieu nay). Chay truoc khi tin tuong bat ky bang tong hop nao.

Cach dung:
    python3 data/check_nan_in_results.py
    python3 data/check_nan_in_results.py --root results runs   # tuy chinh thu muc quet

Exit code: 0 = sach (khong co NaN/Inf that), 1 = phat hien NaN/Inf that trong du lieu.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def scan_csv(path: Path):
    """Tra ve list (row_index, col_name, value) cho moi o NaN/Inf trong bat ky cot nao.

    Dung df.isna() tren toan bang thay vi tach nhanh theo dtype: pandas da tu dong
    nhan dien cac token pho bien ("nan", "NaN", "", "NULL", "N/A"...) thanh NaN THAT
    ngay khi doc CSV, bat ke cot do cuoi cung la kieu so hay kieu chuoi (da kiem chung:
    voi pandas ban moi, cot chuoi khong con luon la dtype "object" nhu truoc, tach
    nhanh theo dtype se bo sot). Kiem tra Inf rieng vi isna() khong bat Inf.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return None, f"khong doc duoc file ({e})"

    bad_cells = []
    na_mask = df.isna()
    for col in df.columns:
        for idx in df.index[na_mask[col]]:
            bad_cells.append((idx, col, df.loc[idx, col]))

    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        col_data = df[col]
        inf_mask = col_data.apply(lambda v: pd.notna(v) and abs(v) == float("inf"))
        for idx in df.index[inf_mask]:
            bad_cells.append((idx, col, col_data.loc[idx]))

    return df, bad_cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", nargs="+", default=["results", "runs"],
                     help="Cac thu muc goc can quet CSV (mac dinh: results runs)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    csv_files = []
    for root_name in args.root:
        root = repo_root / root_name
        if root.exists():
            csv_files.extend(sorted(root.rglob("*.csv")))

    if not csv_files:
        print(f"[CANH BAO] Khong tim thay file CSV nao trong: {args.root}")
        print("           (co the ket qua chua duoc dong bo tu server, hoac pipeline chua chay xong)")
        sys.exit(0)

    print(f"Tim thay {len(csv_files)} file CSV. Dang quet...\n")

    total_bad_files = 0
    total_bad_cells = 0
    for path in csv_files:
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            rel = path
        df, bad_cells = scan_csv(path)
        if df is None:
            print(f"[LOI DOC FILE] {rel}: {bad_cells}")
            total_bad_files += 1
            continue
        if not bad_cells:
            continue
        total_bad_files += 1
        total_bad_cells += len(bad_cells)
        print(f"[CO NaN/Inf THAT] {rel}  ({len(bad_cells)} o loi / {len(df)} dong)")
        # in toi da 10 dong loi dau tien kem cot dinh danh (backbone/seed/lambda...) neu co
        id_cols = [c for c in df.columns if c.lower() in
                   ("backbone", "seed", "lambda_sparsity", "lambda_saliency", "lambda_identity",
                    "config", "n_blocks", "run_name", "domain")]
        for i, (row_idx, col, val) in enumerate(bad_cells[:10]):
            id_info = ", ".join(f"{c}={df.loc[row_idx, c]}" for c in id_cols if c in df.columns)
            print(f"    dong {row_idx} | cot '{col}' = {val!r}" + (f"  [{id_info}]" if id_info else ""))
        if len(bad_cells) > 10:
            print(f"    ... con {len(bad_cells) - 10} o loi khac, khong liet ke het")
        print()

    print("=" * 60)
    if total_bad_files == 0:
        print(f"KET QUA: SACH. Quet {len(csv_files)} file, khong co NaN/Inf that nao trong du lieu.")
        sys.exit(0)
    else:
        print(f"KET QUA: PHAT HIEN LOI. {total_bad_files}/{len(csv_files)} file co NaN/Inf "
              f"(tong {total_bad_cells} o). Xem chi tiet o tren.")
        print("=> Day la NaN THAT trong du lieu training/eval (vi du: mot seed training bi loi,")
        print("   metric tinh khong ra so...), KHAC voi NaN noi bo cua cong thuc MDES da sua truoc do.")
        print("   Can kiem tra log cua tung run/seed tuong ung truoc khi dua vao bat ky bang tong hop nao.")
        sys.exit(1)


if __name__ == "__main__":
    main()
