"""
[MỚI] Báo cáo chính xác phân bố gender theo SUBJECT (không phải theo ảnh) từ
splits.json đã sinh sẵn — thay cho việc ước lượng trong bài báo (Table dataset
stats). Đọc trực tiếp trường "gender" đã được prepare_splits.py gán sẵn cho
từng ảnh (qua gender_from_person_id(), xem prepare_splits.py để biết quy ước
và nguồn xác minh — CHỈ đúng cho EarVN1.0, xem cảnh báo dưới), rồi quy về theo
subject (person_id) để đếm số NGƯỜI theo từng gender, không phải số ẢNH.

QUAN TRỌNG: EarVN1.0 có gender ground-truth thật (khớp công bố gốc của tác giả
dataset, xem prepare_splits.py::gender_from_person_id()). AWEx KHÔNG có gender
ground-truth thật — quy ước "id<=98 -> nam" trong prepare_splits.py chỉ đúng ý
nghĩa cho EarVN1.0; khi áp dụng cho AWEx (qua convert_generic_dataset_to_
project_format.py, đánh số lại NNN theo alphabet tên thư mục gốc, xem docstring
file đó), nó KHÔNG phản ánh gender thật của AWEx. KHÔNG dùng script này để báo
cáo gender cho AWEx.

Chạy:
    python data/report_gender_distribution.py --splits_json splits/splits.json
"""
import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_json", required=True)
    args = ap.parse_args()

    with open(args.splits_json, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # person_id -> set các gender xuất hiện (phải luôn đúng 1 giá trị/subject
    # -- nếu >1, gender_from_person_id() hoặc dữ liệu splits.json có vấn đề).
    subject_gender = defaultdict(set)
    images_per_gender = defaultdict(int)
    images_per_split_gender = defaultdict(lambda: defaultdict(int))

    for split_name, entries in splits.items():
        for entry in entries:
            pid = entry["person_id"]
            gender = entry["gender"]
            subject_gender[pid].add(gender)
            images_per_gender[gender] += 1
            images_per_split_gender[split_name][gender] += 1

    inconsistent = {pid: g for pid, g in subject_gender.items() if len(g) > 1}
    if inconsistent:
        raise RuntimeError(
            f"LỖI DỮ LIỆU: {len(inconsistent)} subject có gender KHÔNG NHẤT QUÁN "
            f"giữa các ảnh (phải đúng 1 gender/subject): {inconsistent}")

    GENDER_NAME = {0: "nam (0)", 1: "nữ (1)"}
    subjects_per_gender = defaultdict(int)
    for pid, g in subject_gender.items():
        subjects_per_gender[next(iter(g))] += 1

    n_subjects = len(subject_gender)
    n_images = sum(images_per_gender.values())

    print(f"=== Phân bố gender — {args.splits_json} ===")
    print(f"Tổng số subject: {n_subjects} | Tổng số ảnh: {n_images}\n")

    print("-- Theo SUBJECT (dùng cho Table dataset stats trong bài báo) --")
    for g in sorted(subjects_per_gender.keys()):
        n = subjects_per_gender[g]
        print(f"  {GENDER_NAME.get(g, g)}: {n} subject ({100 * n / n_subjects:.2f}%)")

    print("\n-- Theo ẢNH (tổng train+val+test) --")
    for g in sorted(images_per_gender.keys()):
        n = images_per_gender[g]
        print(f"  {GENDER_NAME.get(g, g)}: {n} ảnh ({100 * n / n_images:.2f}%)")

    print("\n-- Theo ẢNH, tách riêng từng split --")
    for split_name in ["train", "val", "test"]:
        if split_name not in images_per_split_gender:
            continue
        counts = images_per_split_gender[split_name]
        total = sum(counts.values())
        parts = ", ".join(f"{GENDER_NAME.get(g, g)}={n} ({100*n/total:.2f}%)"
                           for g, n in sorted(counts.items()))
        print(f"  {split_name}: {parts}")


if __name__ == "__main__":
    main()
