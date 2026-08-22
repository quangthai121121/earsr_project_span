"""
Khảo sát độ phân giải, TỰ ĐỘNG chọn ngưỡng hr_source_min/hr_size dựa trên số
liệu thật, phân nhóm, chia train/val/test theo identity.

Chế độ AUTO (khuyến nghị, mặc định) — tự chọn ngưỡng, không cần đoán số tay:
    python data/prepare_splits.py --raw_dir raw_data/images --out_dir splits/ --auto

Chế độ thủ công (nếu muốn tự quyết định ngưỡng):
    python data/prepare_splits.py --raw_dir raw_data/images --out_dir splits/ \
        --hr_source_min 41 --too_small_max 20

Chỉ khảo sát, không lọc/chia gì (xem số liệu trước khi quyết định):
    python data/prepare_splits.py --raw_dir raw_data/images --out_dir splits/ --survey_only
"""
import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image


def extract_person_id(folder_name: str) -> str:
    """Thư mục thật của EarVN1.0 có định dạng 'NNN.Tên' (ví dụ '001.ALI_HD'),
    không phải chỉ 'NNN'. Lấy phần số 3 chữ số đầu làm person_id chuẩn hóa."""
    prefix = folder_name.split(".")[0].strip()
    return prefix.zfill(3)


def gender_from_person_id(person_id: str) -> int:
    """001-098 = nam (0), 099-164 = nữ (1).

    [ĐÃ XÁC MINH — đợt 7, trước đây ghi "kiểm tra lại quy ước này", giờ đã
    kiểm tra xong]: khớp ĐÚNG với mô tả chính thức của bài báo gốc công bố
    dataset EarVN1.0 — Hoang, V.N. "EarVN1.0: A new large-scale ear images
    dataset in the wild", Data in Brief, 2019 (PMC6831707/ScienceDirect
    S2352340919309850): "98 males and 66 females... the first 98 folders
    (from 01 to 98) belong to male class and the rest (from 99 to 164) are
    female." Số lượng khớp chính xác (98 nam + 66 nữ = 164), không phải suy
    đoán. Chỉ ảnh hưởng cột gender_accuracy (KHÔNG ảnh hưởng identity_accuracy
    — độc lập hoàn toàn với hàm này, xem build_label_map() trong
    datasets/ear_dataset.py)."""
    idx = int(person_id)
    return 0 if idx <= 98 else 1


def survey_resolutions(raw_dir: Path):
    records = []
    for person_dir in sorted(raw_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        person_id = extract_person_id(person_dir.name)
        # sorted(): glob() không đảm bảo thứ tự ổn định giữa các filesystem/OS
        # khác nhau — nếu không sort trước khi shuffle theo seed cố định bên
        # dưới, cùng 1 seed vẫn có thể ra split khác nhau giữa các máy.
        for img_path in sorted(person_dir.glob("*.jpg")):
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception as e:
                print(f"[bỏ qua] lỗi đọc ảnh {img_path}: {e}")
                continue
            records.append((str(img_path), person_id, w, h))
    return records


def compute_percentiles(records):
    sizes = sorted(min(w, h) for _, _, w, h in records)
    n = len(sizes)
    if n == 0:
        return None, None

    def pct(p):
        return sizes[int(p * (n - 1))]

    percentiles = {p: pct(p) for p in [0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 0.90]}
    return percentiles, pct


def print_resolution_stats(records, percentiles):
    sizes = sorted(min(w, h) for _, _, w, h in records)
    print(f"Tổng số ảnh: {len(sizes)}")
    print(f"Cạnh ngắn nhỏ nhất / lớn nhất: {sizes[0]} / {sizes[-1]}")
    print(f"Percentile cạnh ngắn — p5: {percentiles[0.05]}  p10: {percentiles[0.10]}  "
          f"p20: {percentiles[0.20]}  p25: {percentiles[0.25]}  p50: {percentiles[0.50]}  "
          f"p75: {percentiles[0.75]}  p90: {percentiles[0.90]}")


def check_identity_coverage(all_records, hr_source_records, min_images_per_identity: int = 10,
                             verbose: bool = True):
    """Trả về (missing_ids, low_count_ids). Cảnh báo nếu có identity bị mất
    trắng hoặc còn quá ít ảnh sau khi lọc theo ngưỡng kích thước."""
    all_ids = {r[1] for r in all_records}
    kept_counts = {}
    for r in hr_source_records:
        kept_counts[r[1]] = kept_counts.get(r[1], 0) + 1

    missing_ids = sorted(all_ids - set(kept_counts.keys()))
    low_count_ids = sorted(
        pid for pid, c in kept_counts.items() if c < min_images_per_identity
    )

    if verbose:
        print(f"Tổng identity gốc: {len(all_ids)} | Còn lại sau lọc: {len(kept_counts)}")
        if missing_ids:
            print(f"!!! {len(missing_ids)} identity MẤT TRẮNG: {missing_ids}")
        else:
            print("OK: không identity nào bị mất trắng.")
        if low_count_ids:
            print(f"Cảnh báo nhẹ: {len(low_count_ids)} identity còn dưới "
                  f"{min_images_per_identity} ảnh: {low_count_ids}")

    return missing_ids, low_count_ids


def group_by_size(records, hr_source_min: int, too_small_max: int, verbose: bool = True):
    hr_source, too_small, middle = [], [], []
    for r in records:
        short_side = min(r[2], r[3])
        if short_side >= hr_source_min:
            hr_source.append(r)
        elif short_side < too_small_max:
            too_small.append(r)
        else:
            middle.append(r)

    if verbose:
        print(f"Nhóm HR-source (>= {hr_source_min}px): {len(hr_source)} ảnh")
        print(f"Nhóm quá nhỏ (< {too_small_max}px): {len(too_small)} ảnh — LR thật")
        print(f"Nhóm giữa (không dùng): {len(middle)} ảnh")
        kept_pct = len(hr_source) / len(records) * 100 if records else 0
        print(f"Giữ lại: {kept_pct:.1f}% dữ liệu")

    return hr_source, too_small


def auto_select_threshold(records, percentiles, pct_fn, too_small_max: int,
                           min_images_per_identity: int = 10):
    """
    Tự động chọn hr_source_min theo thứ tự ưu tiên % dữ liệu giữ lại:
    80% (khuyến nghị mặc định) -> 90% (an toàn hơn nếu 80% làm mất identity)
    -> 70% -> 60% -> 50% (chỉ dùng khi các mức trên đều thất bại).
    Điều kiện CHỌN: không có identity nào bị mất trắng (missing_ids rỗng).
    """
    print("\n== TỰ ĐỘNG chọn ngưỡng hr_source_min ==")
    priority = [0.80, 0.90, 0.70, 0.60, 0.50]

    chosen = None
    for keep_pct in priority:
        threshold = pct_fn(1 - keep_pct)
        hr_source, _ = group_by_size(records, threshold, too_small_max, verbose=False)
        missing_ids, low_count_ids = check_identity_coverage(
            records, hr_source, min_images_per_identity, verbose=False)

        status = "OK" if not missing_ids else f"THẤT BẠI ({len(missing_ids)} identity mất trắng)"
        print(f"  Thử giữ {int(keep_pct*100)}% -> hr_source_min={threshold} -> {status}")

        if not missing_ids and chosen is None:
            chosen = (threshold, keep_pct, len(low_count_ids))

    if chosen is None:
        # Không mức nào sạch hoàn toàn -> lấy mức an toàn nhất (90%, threshold thấp nhất)
        threshold = pct_fn(1 - 0.90)
        print(f"\n!!! CẢNH BÁO: không tìm được ngưỡng nào hoàn toàn sạch identity. "
              f"Dùng mức an toàn nhất (giữ 90%, hr_source_min={threshold}) — "
              f"vẫn có thể còn identity thiếu ảnh, kiểm tra lại log chi tiết bên dưới.")
        chosen = (threshold, 0.90, None)

    hr_source_min, keep_pct, low_count = chosen
    print(f"\n>>> ĐÃ CHỌN: hr_source_min={hr_source_min} (giữ ~{int(keep_pct*100)}% dữ liệu"
          + (f", {low_count} identity hơi ít ảnh" if low_count else ", không identity nào thiếu ảnh")
          + ")")

    # hr_size: tỷ lệ phóng đại tối đa ~1.95x từ hr_source_min, làm tròn về bội số của 4 (scale)
    hr_size_raw = hr_source_min * 1.95
    hr_size = max(32, round(hr_size_raw / 4) * 4)
    print(f">>> ĐÃ CHỌN: hr_size={hr_size} (tỷ lệ phóng đại tối đa ~{hr_size/hr_source_min:.2f}x)")

    return hr_source_min, hr_size


def split_by_identity(records, train_ratio, val_ratio, seed):
    """
    QUAN TRỌNG: chia theo ẢNH của TỪNG người (stratified per-identity), KHÔNG
    chia nguyên người vào 1 split duy nhất.

    Lý do: bài toán ở đây là closed-set classification (softmax cố định N=164
    lớp, CrossEntropyLoss). Nếu một người chỉ xuất hiện ở val/test mà KHÔNG
    có ảnh nào ở train, neuron output ứng với người đó không bao giờ được học
    (gradient chỉ liên tục dìm nó xuống vì không bao giờ là target đúng) ->
    model KHÔNG THỂ đoán đúng người đó, VAL_ID_ACC sẽ luôn ~0% bất kể train
    bao lâu — đây là giới hạn toán học, không phải model kém. Phải đảm bảo
    MỌI người đều xuất hiện ở cả train/val/test, chỉ khác nhau ở ảnh nào.
    """
    from collections import defaultdict

    by_person = defaultdict(list)
    for r in records:
        by_person[r[1]].append(r)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    empty_split_warning = []

    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    val_share_of_remainder = (
        val_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0.5
    )

    for pid, imgs in by_person.items():
        imgs = imgs.copy()
        rng.shuffle(imgs)
        n = len(imgs)

        if n < 3:
            # Không đủ ảnh để chia cả 3 phần (cần tối thiểu 1 ảnh/phần) — toàn bộ
            # rơi vào train, val/test rỗng cho identity này. Đây là trường hợp
            # biên không thể tránh khỏi (không phải lỗi làm tròn), luôn được ghi
            # vào empty_split_warning bên dưới để không bị bỏ sót khi báo cáo.
            n_train, n_val = n, 0
        else:
            # [SỬA] n_train được CHỐT trước (không đổi sau khi tính n_val — bản
            # cũ tính n_val từ n_train GỐC rồi mới hạ n_train, khiến phần dư luôn
            # bị dồn hết vào test thay vì chia đều theo đúng val_ratio/test_ratio
            # -> lệch tỷ lệ có hệ thống, đo được ~19% trên dữ liệu thật).
            n_train = min(max(round(n * train_ratio), 1), n - 2)
            remaining = n - n_train  # luôn >= 2
            n_val = min(max(round(remaining * val_share_of_remainder), 1), remaining - 1)

        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train + n_val]
        test_imgs = imgs[n_train + n_val:]

        if len(train_imgs) == 0 or len(val_imgs) == 0 or len(test_imgs) == 0:
            empty_split_warning.append((pid, n))

        for split_name, img_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            for path, _, w, h in img_list:
                gender = gender_from_person_id(pid)
                splits[split_name].append(
                    {"path": path, "person_id": pid, "gender": gender, "width": w, "height": h})

    print(f"Số identity (giống nhau ở cả 3 split): {len(by_person)}")
    print(f"Số ảnh: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    if empty_split_warning:
        detail = ", ".join(f"{pid} (n={n})" for pid, n in empty_split_warning)
        print(f"!!! CẢNH BÁO: {len(empty_split_warning)} identity có split rỗng "
              f"(quá ít ảnh để chia đủ 3 phần): {detail}")

    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--survey_only", action="store_true",
                     help="chỉ in thống kê độ phân giải, không chia/xuất file gì cả")
    ap.add_argument("--auto", action="store_true",
                     help="TỰ ĐỘNG chọn hr_source_min/hr_size dựa trên số liệu thật "
                          "(khuyến nghị dùng thay vì tự đoán số tay)")
    ap.add_argument("--hr_source_min", type=int, default=None,
                     help="ngưỡng thủ công (bỏ qua nếu dùng --auto)")
    ap.add_argument("--too_small_max", type=int, default=20,
                     help="ngưỡng cạnh ngắn dưới mức này -> gác riêng làm LR thật")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report_out", default=None,
                     help="nếu chỉ định, ghi báo cáo ngưỡng đã chọn ra file text này "
                          "(dùng để điền vào bảng dữ liệu bài báo, xem RUNBOOK_EarVN1.0.md mục 1)")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    print("== Khảo sát phân phối độ phân giải ==")
    records = survey_resolutions(raw_dir)
    percentiles, pct_fn = compute_percentiles(records)
    if percentiles is None:
        print("Không có ảnh nào trong raw_dir, dừng lại.")
        return
    print_resolution_stats(records, percentiles)

    if args.survey_only:
        print("\n(--survey_only) Dừng ở đây.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.auto:
        hr_source_min, hr_size = auto_select_threshold(records, percentiles, pct_fn,
                                                         args.too_small_max)
    elif args.hr_source_min is not None:
        hr_source_min = args.hr_source_min
        hr_size = max(32, round(hr_source_min * 1.95 / 4) * 4)
        print(f"\n== Dùng ngưỡng thủ công: hr_source_min={hr_source_min}, "
              f"hr_size gợi ý={hr_size} ==")
    else:
        raise ValueError("Cần chỉ định --auto hoặc --hr_source_min.")

    print("\n== Phân nhóm theo ngưỡng đã chọn ==")
    hr_source, too_small = group_by_size(records, hr_source_min, args.too_small_max)
    check_identity_coverage(records, hr_source)

    print("\n== Chia train/val/test theo identity (chỉ trên nhóm HR-source) ==")
    splits = split_by_identity(hr_source, args.train_ratio, args.val_ratio, args.seed)

    out_path = out_dir / "splits.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {out_path} — "
          f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} ảnh")

    real_lr_path = out_dir / "real_lr_holdout.json"
    real_lr_entries = [
        {"path": p, "person_id": pid, "gender": gender_from_person_id(pid), "width": w, "height": h}
        for p, pid, w, h in too_small
    ]
    with open(real_lr_path, "w", encoding="utf-8") as f:
        json.dump(real_lr_entries, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {real_lr_path} — {len(real_lr_entries)} ảnh LR thật")

    if args.report_out:
        with open(args.report_out, "w", encoding="utf-8") as f:
            f.write(f"hr_source_min={hr_source_min}\n")
            f.write(f"hr_size={hr_size}\n")
            f.write(f"too_small_max={args.too_small_max}\n")
            f.write(f"n_hr_source={len(hr_source)}\n")
            f.write(f"n_too_small={len(too_small)}\n")
            f.write(f"n_train={len(splits['train'])}\n")
            f.write(f"n_val={len(splits['val'])}\n")
            f.write(f"n_test={len(splits['test'])}\n")
        print(f"Đã ghi báo cáo ngưỡng: {args.report_out}")

    # Xuất CSV thống kê dữ liệu — dùng cho Bảng 1 (Dataset) trong journal
    stats_csv_path = out_dir / "dataset_stats.csv"
    with open(stats_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_images_raw", len(records)])
        writer.writerow(["min_short_side_px", min(min(w, h) for _, _, w, h in records)])
        writer.writerow(["max_short_side_px", max(min(w, h) for _, _, w, h in records)])
        for p_key, p_val in percentiles.items():
            writer.writerow([f"p{int(p_key*100)}_short_side_px", p_val])
        writer.writerow(["hr_source_min_px", hr_source_min])
        writer.writerow(["too_small_max_px", args.too_small_max])
        writer.writerow(["hr_size_px", hr_size])
        writer.writerow(["scale_factor", "4 (cố định, xem configs/config.yaml)"])
        writer.writerow(["n_hr_source_images", len(hr_source)])
        writer.writerow(["n_too_small_images_real_lr", len(too_small)])
        writer.writerow(["pct_data_kept", round(len(hr_source) / len(records) * 100, 2)])
        writer.writerow(["n_identities_total", len({r[1] for r in records})])
        writer.writerow(["n_train_images", len(splits["train"])])
        writer.writerow(["n_val_images", len(splits["val"])])
        writer.writerow(["n_test_images", len(splits["test"])])
    print(f"Đã ghi thống kê dữ liệu: {stats_csv_path}")


if __name__ == "__main__":
    main()
