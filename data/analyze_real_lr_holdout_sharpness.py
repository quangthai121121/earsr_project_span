"""
[MỚI] Tìm bằng chứng khách quan cho giả thuyết "vì sao span_baseline tệ hơn
span_tiny trên real_lr_holdout": span_baseline (nhiều tham số hơn) có thể
"tự tin" tái tạo chi tiết tần số cao SAI (hallucinate) mạnh hơn span_tiny khi
gặp suy giảm ngoài phân phối đã train, trong khi span_tiny (ít tham số hơn)
buộc phải tái tạo "an toàn/mượt" hơn.

KHÔNG cần train lại gì -- chỉ chạy inference (đã có sẵn 4 checkpoint: span_tiny,
span_baseline, span_tiny_robust, span_baseline_robust) trên chính 173 ảnh
real_lr_holdout đã có, đo 1 proxy KHÁCH QUAN cho "mức độ chi tiết tần số cao
được thêm vào": phương sai Laplacian (Laplacian variance) -- thước đo độ sắc
nét chuẩn trong xử lý ảnh, giá trị càng cao càng nhiều cấu trúc tần số cao.

Vì không có ảnh HR gốc để so sánh (real_lr_holdout là ảnh suy giảm THẬT,
không có cặp HR sạch), KHÔNG thể đo "đúng/sai" trực tiếp -- chỉ đo LƯỢNG chi
tiết tần số cao được THÊM VÀO so với no-SR (bicubic thường, không tái tạo gì
thêm) làm đường tham chiếu an toàn. "sharpness_gain" = Laplacian(SR output) -
Laplacian(no-SR output) trên CÙNG 1 ảnh -- gain càng lớn, model càng "tự tin"
thêm cấu trúc mới. Nếu span_baseline có gain LỚN HƠN span_tiny một cách có ý
nghĩa thống kê (paired qua 173 ảnh), đó là bằng chứng khách quan ủng hộ giả
thuyết capacity/hallucination -- KHÔNG PHẢI bằng chứng "chi tiết đó đúng hay
sai" (không có ground truth để biết), chỉ là bằng chứng về LƯỢNG.

Chạy:
    python data/analyze_real_lr_holdout_sharpness.py --config configs/config.yaml \
        --span_tiny_ckpt runs/sr_improved_span_tiny/best.pt \
        --span_baseline_ckpt runs/sr_span_official/best.pt \
        --span_tiny_robust_ckpt runs/sr_improved_span_tiny_robust/best.pt \
        --span_baseline_robust_ckpt runs/sr_span_official_robust/best.pt \
        --out_csv results/degradation_robustness_lrholdout/sharpness_analysis.csv
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from scipy import ndimage, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_multi_seed_results import cohens_d_paired, paired_ci95, wilcoxon_paired_p  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.sr_models import build_sr_model  # noqa: E402
from utils.letterbox import letterbox_resize  # noqa: E402


def laplacian_variance(img: Image.Image) -> float:
    """Phương sai Laplacian trên kênh grayscale -- proxy độ sắc nét/chi tiết
    tần số cao chuẩn trong xử lý ảnh. Càng cao càng nhiều cấu trúc tần số cao
    (KHÔNG phân biệt được cấu trúc đó đúng hay bịa -- chỉ đo LƯỢNG)."""
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    lap = ndimage.laplace(gray)
    return float(lap.var())


@torch.no_grad()
def run_sr(model, lr_tensor, device):
    return model(lr_tensor.unsqueeze(0).to(device)).squeeze(0).cpu().clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--span_tiny_ckpt", required=True)
    ap.add_argument("--span_baseline_ckpt", required=True)
    ap.add_argument("--span_tiny_robust_ckpt", required=True)
    ap.add_argument("--span_baseline_robust_ckpt", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--identity_accuracy_csv", default=None,
                     help="[MỚI] đường dẫn results/real_lr_holdout_multiseed/"
                          "real_lr_holdout_multiseed_identity.csv (span_tiny/span_baseline gốc) -- "
                          "nếu truyền, script đối chiếu sharpness_gain với identity_accuracy ĐÃ ĐO "
                          "ĐƯỢC THẬT. Sharpness cao hơn KHÔNG tự nó chứng minh 'bịa sai' (có thể là "
                          "tái tạo chi tiết ĐÚNG, không có ground-truth HR để phân biệt) -- nhưng nếu "
                          "gain cao hơn lại ĐI KÈM accuracy THẤP hơn một cách nhất quán, đó là bằng "
                          "chứng mạnh hơn nhiều cho giả thuyết hallucination-có-hại so với chỉ nhìn "
                          "sharpness đơn lẻ.")
    ap.add_argument("--identity_accuracy_robust_csv", default=None,
                     help="[MỚI] đường dẫn results/degradation_robustness_lrholdout/"
                          "real_lr_holdout_robust_identity.csv (span_tiny_robust/span_baseline_robust) "
                          "-- dùng cùng với --identity_accuracy_csv.")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits_root = cfg["paths"]["splits_root"]
    hr_size = cfg["image"]["hr_size"]
    scale = cfg["image"]["scale"]
    lr_size = hr_size // scale

    with open(f"{splits_root}/real_lr_holdout.json", "r", encoding="utf-8") as f:
        entries = json.load(f)

    checkpoints = {
        "span_tiny": (args.span_tiny_ckpt, "span_tiny"),
        "span_baseline": (args.span_baseline_ckpt, "span_official"),
        "span_tiny_robust": (args.span_tiny_robust_ckpt, "span_tiny"),
        "span_baseline_robust": (args.span_baseline_robust_ckpt, "span_official"),
    }
    models = {}
    for name, (ckpt_path, arch) in checkpoints.items():
        m = build_sr_model(arch, scale)
        m.load_state_dict(torch.load(ckpt_path, map_location=device))
        m.to(device).eval()
        models[name] = m
        print(f"Đã nạp {name} <- {ckpt_path}")

    from torchvision import transforms
    to_tensor = transforms.ToTensor()

    # per-image: {condition: giá trị sharpness}
    rows = []
    for i, entry in enumerate(entries):
        img = Image.open(entry["path"]).convert("RGB")
        no_sr_img = letterbox_resize(img, hr_size)
        lr_img_pil = letterbox_resize(img, lr_size)
        lr_tensor = to_tensor(lr_img_pil)

        sharp_no_sr = laplacian_variance(no_sr_img)
        row = {"idx": i, "path": entry["path"], "sharpness_no_sr": sharp_no_sr}
        for name, model in models.items():
            sr_tensor = run_sr(model, lr_tensor, device)
            sr_img = transforms.ToPILImage()(sr_tensor)
            row[f"sharpness_{name}"] = laplacian_variance(sr_img)
            row[f"gain_{name}"] = row[f"sharpness_{name}"] - sharp_no_sr
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  ... đã xử lý {i + 1}/{len(entries)} ảnh")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nĐã ghi {out_path} ({len(rows)} ảnh)\n")

    print("=== Trung bình sharpness_gain (SR - no_sr), càng cao = càng 'tự tin' thêm chi tiết ===")
    gains = {}
    for name in models:
        vals = [r[f"gain_{name}"] for r in rows]
        gains[name] = vals
        print(f"  {name:<22} {statistics.mean(vals):+.4f} +- {statistics.stdev(vals):.4f}")

    print("\n=== So sánh cặp (paired qua đúng 173 ảnh): baseline có gain > tiny có ý nghĩa không? ===")
    pairwise_rows = []
    for pair_name, (a, b) in {
        "baseline vs tiny (recipe gốc)": ("span_tiny", "span_baseline"),
        "baseline_robust vs tiny_robust (recipe mới)": ("span_tiny_robust", "span_baseline_robust"),
    }.items():
        vals_a, vals_b = gains[a], gains[b]
        mean_diff = statistics.mean(vals_b) - statistics.mean(vals_a)
        _, p_raw = stats.ttest_rel(vals_b, vals_a)
        d = cohens_d_paired(vals_b, vals_a)
        ci_lo, ci_hi = paired_ci95(vals_b, vals_a)
        wilcoxon_p = wilcoxon_paired_p(vals_b, vals_a)
        # [SỬA — bug phát hiện qua review] d/wilcoxon_p CÓ THỂ là None (std
        # hiệu số ~0, xem cohens_d_paired/wilcoxon_paired_p) -- f"{None:.4f}"
        # crash ngay với TypeError. Và dùng `if d else` (truthy) thay vì
        # `if d is not None else` sẽ hiển thị NHẦM "NA" khi d/wilcoxon_p tính
        # ra ĐÚNG BẰNG 0.0 (giá trị hợp lệ -- không có hiệu ứng -- 0.0 là
        # falsy trong Python, không phải None). Format string an toàn cho cả
        # 2 trường hợp.
        d_str = f"{d:.4f}" if d is not None else "NA"
        wilcoxon_str = f"{wilcoxon_p:.4g}" if wilcoxon_p is not None else "NA"
        print(f"  {pair_name}: diff(b-a)={mean_diff:+.4f}  d={d_str}  "
              f"CI95%=[{ci_lo:.4f},{ci_hi:.4f}]  p_raw={p_raw:.4g}  wilcoxon_p={wilcoxon_str}")
        pairwise_rows.append({
            "comparison": pair_name, "condition_a": a, "condition_b": b,
            "mean_diff_b_minus_a": round(mean_diff, 4),
            "cohens_d": round(d, 4) if d is not None else "NA",
            "ci95_lower": round(ci_lo, 4), "ci95_upper": round(ci_hi, 4),
            "p_raw": round(p_raw, 6),
            "wilcoxon_p": round(wilcoxon_p, 6) if wilcoxon_p is not None else "NA",
        })

    pairwise_path = out_path.with_name(out_path.stem + "_pairwise.csv")
    with open(pairwise_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairwise_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pairwise_rows)
    print(f"\nĐã ghi {pairwise_path}")

    # ================= [MỚI] Đối chiếu sharpness_gain với accuracy ĐÃ ĐO ĐƯỢC THẬT =================
    # Sharpness cao hơn KHÔNG tự nó chứng minh "bịa sai" -- không có ground-
    # truth HR cho real_lr_holdout để phân biệt "tái tạo đúng" với "tự tin
    # bịa sai". Nhưng nếu gain cao hơn lại ĐI KÈM accuracy THẤP hơn một cách
    # NHẤT QUÁN qua cả 4 điều kiện, đó là bằng chứng gián tiếp mạnh hơn nhiều
    # cho giả thuyết hallucination-có-hại, vì tái tạo chi tiết ĐÚNG lẽ ra
    # phải TƯƠNG QUAN THUẬN với accuracy (giúp nhận dạng tốt hơn), không phải
    # nghịch.
    if args.identity_accuracy_csv and args.identity_accuracy_robust_csv:
        def _read_condition_accuracy(csv_path, condition):
            with open(csv_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["condition"] == condition:
                        return float(row["mean_identity_accuracy"])
            raise KeyError(f"Không tìm thấy condition='{condition}' trong {csv_path}")

        # span_tiny/span_baseline (gốc) <-> condition "sr_improved"/"sr_baseline" trong
        # file gốc; span_*_robust <-> CÙNG tên condition nhưng trong file robust (xem
        # thiết kế eval_real_lr_holdout.py::--domain_suffix -- cột "condition" KHÔNG đổi
        # theo suffix, chỉ đổi file nguồn).
        acc = {
            "span_tiny": _read_condition_accuracy(args.identity_accuracy_csv, "sr_improved"),
            "span_baseline": _read_condition_accuracy(args.identity_accuracy_csv, "sr_baseline"),
            "span_tiny_robust": _read_condition_accuracy(args.identity_accuracy_robust_csv, "sr_improved"),
            "span_baseline_robust": _read_condition_accuracy(args.identity_accuracy_robust_csv, "sr_baseline"),
        }
        print("\n=== Đối chiếu sharpness_gain vs.\\ identity_accuracy đã đo được (4 điều kiện) ===")
        correlation_rows = []
        for name in models:
            mean_gain = statistics.mean(gains[name])
            correlation_rows.append({"condition": name, "mean_sharpness_gain": round(mean_gain, 4),
                                      "mean_identity_accuracy": acc[name]})
            print(f"  {name:<22} sharpness_gain={mean_gain:+9.4f}   identity_accuracy={acc[name]:.4f}")

        # Spearman rank correlation (chỉ 4 điểm dữ liệu -- không đủ để kiểm định
        # thống kê có ý nghĩa, CHỈ dùng để mô tả xu hướng định tính, không suy
        # rộng thành kết luận thống kê).
        gains_list = [c["mean_sharpness_gain"] for c in correlation_rows]
        acc_list = [c["mean_identity_accuracy"] for c in correlation_rows]
        rho, _ = stats.spearmanr(gains_list, acc_list)
        print(f"\n  Spearman rho (gain vs accuracy, n=4 điều kiện -- MÔ TẢ xu hướng, "
              f"KHÔNG phải kiểm định thống kê có ý nghĩa với n nhỏ thế này): {rho:.3f}")
        print("  rho < 0: gain cao hơn đi kèm accuracy thấp hơn -- ủng hộ giả thuyết hallucination-có-hại.")
        print("  rho > 0: gain cao hơn đi kèm accuracy CAO hơn -- ngược lại, gợi ý tái tạo chi tiết ĐÚNG "
              "chứ không phải bịa, giả thuyết hallucination KHÔNG được ủng hộ.")

        correlation_path = out_path.with_name(out_path.stem + "_accuracy_correlation.csv")
        with open(correlation_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(correlation_rows[0].keys()))
            writer.writeheader()
            writer.writerows(correlation_rows)
        print(f"\nĐã ghi {correlation_path}")
    else:
        print("\n(Bỏ qua đối chiếu accuracy -- không truyền --identity_accuracy_csv/"
              "--identity_accuracy_robust_csv.)")


if __name__ == "__main__":
    main()
