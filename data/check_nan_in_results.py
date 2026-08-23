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


# training_summary.csv (data/export_training_log_summary.py) gop chung 1 bang cho 3 loai
# training script khac nhau, moi loai chi log DUNG 1 trong 3 metric nay trong train.log:
#   train_sr.py             -> chi log VAL_L1=...          (best_val_l1)
#   train_sr_distill.py /
#   train_sr_learned_prune.py -> chi log VAL_TOTAL=...     (best_val_total)
#   train_recognition.py    -> chi log VAL_ID_ACC=...      (best_val_id_acc)
# => moi dong DUNG RA phai co chinh xac 2/3 cot nay la "NA" (khong ap dung, khong phai
# loi), 1 cot con lai co gia tri that. Day KHONG phai suy doan: da grep truc tiep 4 file
# train_*.py de xac nhan tung script chi emit dung 1 pattern, va kiem chung bang so hoc
# tren du lieu that (23 dong x 2 cot NA/dong = 46 o, khop chinh xac voi so o NaN thuc te
# quan sat duoc). Chi dong nao LECH khoi dung 2/3 (vd ca 3 cot deu NaN -> log khong parse
# duoc metric nao, that su dang ngo; hoac >=2 cot co gia tri -> bat thuong) moi la loi that.
_TRAINING_SUMMARY_METRIC_COLS = ("best_val_id_acc", "best_val_l1", "best_val_total")


def filter_training_summary_false_positives(path: Path, df, bad_cells):
    """Bo cac o NaN o training_summary.csv la "NA co chu dich" (khac schema theo loai
    run), chi giu lai nhung dong that su bat thuong. Tra ve (bad_cells_that, ghi_chu)."""
    if path.name != "training_summary.csv":
        return bad_cells, []
    if not all(c in df.columns for c in _TRAINING_SUMMARY_METRIC_COLS):
        return bad_cells, []

    notes = []
    metric_bad = {c: set(idx for idx, col, _ in bad_cells if col == c)
                  for c in _TRAINING_SUMMARY_METRIC_COLS}
    other_bad = [(idx, col, val) for idx, col, val in bad_cells
                 if col not in _TRAINING_SUMMARY_METRIC_COLS]

    kept = list(other_bad)
    n_benign = 0
    for idx in df.index:
        nan_count = sum(idx in metric_bad[c] for c in _TRAINING_SUMMARY_METRIC_COLS)
        if nan_count == 2:
            n_benign += 1
            continue  # dung 1 metric ap dung, 2 metric con lai NA theo thiet ke -> bo qua
        for c in _TRAINING_SUMMARY_METRIC_COLS:
            if idx in metric_bad[c]:
                kept.append((idx, c, df.loc[idx, c]))
        if nan_count == 3:
            notes.append(f"dong {idx} (run_name={df.loc[idx, 'run_name'] if 'run_name' in df.columns else '?'}): "
                          f"CA 3 cot metric deu NaN -> log co the khong parse duoc metric nao, can kiem tra thu cong")
        elif nan_count <= 1:
            notes.append(f"dong {idx} (run_name={df.loc[idx, 'run_name'] if 'run_name' in df.columns else '?'}): "
                          f"co >=2 cot metric CUNG co gia tri -> bat thuong so voi thiet ke 1 script/1 metric, can kiem tra")

    if n_benign:
        notes.insert(0, f"[da loc {n_benign} dong co dung 2/3 cot NA theo thiet ke schema "
                         f"(SR-run / recognition-run chi log 1 trong 3 metric) — khong phai loi]")
    return kept, notes


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
        bad_cells, notes = filter_training_summary_false_positives(path, df, bad_cells)
        if notes:
            for n in notes:
                print(f"[GHI CHU] {rel}: {n}")
        if not bad_cells:
            if notes:
                print()
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
