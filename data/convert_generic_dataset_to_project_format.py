"""
Chuyển đổi cấu trúc thư mục 1 dataset ear-biometrics BẤT KỲ (miễn ảnh đã được
gộp theo TỪNG NGƯỜI trong 1 thư mục con — quy ước phổ biến của hầu hết
dataset công khai: AWE, AWEx, IIT Delhi, USTB, WPUT...) sang đúng định dạng
mà data/prepare_splits.py của project earsr_project_span yêu cầu:

    <out_dir>/NNN/<ten_file>.jpg    (NNN = số thứ tự 3 chữ số, bắt đầu 001)

(prepare_splits.py::extract_person_id() lấy phần trước dấu "." đầu tiên của
TÊN THƯ MỤC rồi zfill(3) -> thư mục "001" hay "001.AnyName" đều được, nhưng
PHẢI là số nguyên có thể ép kiểu int() vì gender_from_person_id() gọi
int(person_id) — thư mục không phải NNN thuần số sẽ làm prepare_splits.py
crash. Script này luôn đặt tên thư mục output là NNN thuần số, an toàn.)

GIẢ ĐỊNH CẤU TRÚC ĐẦU VÀO (LUÔN kiểm tra lại bằng --dry_run sau khi tải
dataset thật trước khi chạy thật — cấu trúc file nén tải về CÓ THỂ khác giả
định này, ví dụ có thêm 1 lớp thư mục trung gian, hoặc ảnh nằm lẫn không
theo thư mục):

    <in_dir>/<ten_hoac_id_nguoi_bat_ky>/<anh1>.<jpg|png|bmp|...>
    <in_dir>/<ten_hoac_id_nguoi_bat_ky>/<anh2>.<jpg|png|bmp|...>
    ...

Nếu cấu trúc thật KHÁC (ví dụ AWE bản tải về có thêm thư mục con
"images/" hoặc file .csv annotation riêng, không nhúng gender/split trong
tên file) — KHÔNG chạy thẳng script này, cần xem lại --in_dir trỏ đúng cấp
thư mục "mỗi người 1 thư mục con" chưa (thường chỉ cần trỏ --in_dir vào
đúng cấp đó, không cần sửa code).

Luôn dùng --dry_run TRƯỚC để xem kế hoạch chuyển đổi (số identity, số ảnh)
trước khi copy file thật — script không ghi đè out_dir đã tồn tại nội dung.

Chạy thử (không copy gì, chỉ xem kế hoạch):
    python convert_generic_dataset_to_project_format.py \
        --in_dir /path/to/AWE_downloaded --out_dir raw_data/awe_raw --dry_run

Chạy thật:
    python convert_generic_dataset_to_project_format.py \
        --in_dir /path/to/AWE_downloaded --out_dir raw_data/awe_raw
"""
import argparse
import shutil
from pathlib import Path

from PIL import Image

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True,
                     help="thư mục gốc dataset đã tải, mỗi người 1 thư mục con")
    ap.add_argument("--out_dir", required=True,
                     help="thư mục đích, đúng định dạng NNN/*.jpg cho prepare_splits.py")
    ap.add_argument("--min_images_per_subject", type=int, default=1,
                     help="bỏ qua người có ít hơn số ảnh này (mặc định: giữ tất cả, kể cả 1 ảnh)")
    ap.add_argument("--dry_run", action="store_true",
                     help="chỉ in kế hoạch chuyển đổi (số người, số ảnh mỗi người), "
                          "KHÔNG copy file thật — LUÔN chạy bước này trước")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if not in_dir.exists():
        raise SystemExit(f"LỖI: không tìm thấy --in_dir {in_dir}")

    if out_dir.exists() and any(out_dir.iterdir()) and not args.dry_run:
        raise SystemExit(
            f"LỖI: --out_dir {out_dir} đã tồn tại VÀ không rỗng. "
            f"Xóa/đổi tên trước khi chạy lại, tránh trộn lẫn dữ liệu 2 lần chạy khác nhau."
        )

    subject_dirs = sorted(p for p in in_dir.iterdir() if p.is_dir())
    if not subject_dirs:
        raise SystemExit(
            f"LỖI: không tìm thấy thư mục con nào trong {in_dir}. "
            f"Kiểm tra lại --in_dir có đúng là thư mục GỐC chứa các thư mục con theo người "
            f"không (xem phần 'GIẢ ĐỊNH CẤU TRÚC ĐẦU VÀO' trong docstring đầu file)."
        )

    print(f"Tìm thấy {len(subject_dirs)} thư mục con (ứng viên mỗi người 1 thư mục).")

    plan = []
    skipped_too_few = []
    for subject_dir in subject_dirs:
        imgs = sorted(p for p in subject_dir.rglob("*") if p.suffix.lower() in VALID_EXT)
        if len(imgs) < args.min_images_per_subject:
            skipped_too_few.append((subject_dir.name, len(imgs)))
            continue
        plan.append((subject_dir, imgs))

    if skipped_too_few:
        preview = skipped_too_few[:10]
        print(f"Bỏ qua {len(skipped_too_few)} thư mục có dưới "
              f"{args.min_images_per_subject} ảnh hợp lệ: "
              f"{preview}{' ...' if len(skipped_too_few) > 10 else ''}")

    if not plan:
        raise SystemExit(
            "LỖI: sau khi lọc, không còn identity nào để chuyển đổi. "
            "Kiểm tra lại đuôi file ảnh có nằm trong VALID_EXT không, hoặc "
            "--min_images_per_subject có đang đặt quá cao không."
        )

    total_imgs = sum(len(imgs) for _, imgs in plan)
    counts = sorted(len(imgs) for _, imgs in plan)
    print(f"\nSẽ tạo {len(plan)} identity (NNN từ 001 đến {len(plan):03d}), "
          f"tổng {total_imgs} ảnh.")
    print(f"Số ảnh/người: min={counts[0]} max={counts[-1]} "
          f"trung bình={total_imgs / len(plan):.1f}")

    if args.dry_run:
        print("\n5 thư mục đầu tiên (mẫu, kiểm tra xem có đúng 'mỗi thư mục = 1 người' không):")
        for subject_dir, imgs in plan[:5]:
            print(f"  {subject_dir.name}  ->  {len(imgs)} ảnh, ví dụ: "
                  f"{[p.name for p in imgs[:3]]}")
        print("\n(--dry_run) Dừng ở đây, CHƯA copy file nào. "
              "Kiểm tra kế hoạch trên đúng ý muốn rồi bỏ --dry_run để chạy thật.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    n_converted, n_failed = 0, 0
    for idx, (subject_dir, imgs) in enumerate(plan, start=1):
        person_id = f"{idx:03d}"
        person_out = out_dir / person_id
        person_out.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(imgs):
            out_name = f"{person_id}_{i:03d}.jpg"
            try:
                if img_path.suffix.lower() in (".jpg", ".jpeg"):
                    # copy trực tiếp nếu đã là JPEG (nhanh hơn, không mất chất lượng do nén lại)
                    shutil.copy(img_path, person_out / out_name)
                else:
                    # chuyển định dạng khác (.png/.bmp/...) sang .jpg — BẮT BUỘC vì
                    # prepare_splits.py/build_lr.py/build_sr.py chỉ glob đúng "*.jpg"
                    with Image.open(img_path) as im:
                        im.convert("RGB").save(person_out / out_name, "JPEG", quality=95)
                n_converted += 1
            except Exception as e:
                print(f"  [bỏ qua] lỗi xử lý {img_path}: {e}")
                n_failed += 1

    print(f"\nHoàn tất: {n_converted} ảnh đã chuyển vào {out_dir}/NNN/*.jpg "
          f"({len(plan)} identity). Lỗi/bỏ qua: {n_failed} ảnh.")
    print(f"\nQUAN TRỌNG: mở thử vài thư mục con trong {out_dir} để xác nhận bằng mắt "
          f"ảnh đúng người, đúng định dạng, TRƯỚC khi chạy pipeline chính thức "
          f"(sai ở bước này sẽ lan sang toàn bộ kết quả sau).")
    print(f"Tiếp theo: đếm --out_dir {out_dir} có đúng số identity/ảnh mong đợi của "
          f"dataset không (đối chiếu với trang chủ/paper gốc của dataset), rồi mới chạy "
          f"pipeline_<ten_dataset>/01_survey_and_prepare_data.sh")


if __name__ == "__main__":
    main()
