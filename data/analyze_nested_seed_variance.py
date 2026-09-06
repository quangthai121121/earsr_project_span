"""
[MỚI — mở rộng theo phản biện] Gộp 3 nguồn dữ liệu đã thu thập (không train gì
thêm ở BƯỚC NÀY, chỉ đọc lại JSON đã có) thành 1 lưới (gần) đầy đủ SR-seed x
downstream-seed x backbone x condition cho 2 domain trung tâm của Table 3
(span_tiny, span_baseline), rồi fit mixed-effects model để tách TƯỜNG MINH
phương sai do SR-seed và phương sai do downstream-seed, thay vì chỉ so sánh
2 độ lệch chuẩn mô tả như Section~sec:sr-seed-variance đã làm.

Nguồn dữ liệu (mỗi ô (condition, backbone, sr_seed, downstream_seed) chỉ lấy
từ ĐÚNG 1 nguồn, không trùng lặp -- xem _load_* bên dưới):
    - results/multi_seed/*.json                      (sr_seed=42 CỐ ĐỊNH, mọi downstream_seed)
    - results/sr_seed_variance_<span_tiny|span_baseline>/*.json  (downstream_seed=42 CỐ ĐỊNH, sr_seed != 42)
    - results/nested_seed_grid/*.json                 (sr_seed != 42 VÀ downstream_seed != 42 -- ô mới train)

QUAN TRỌNG về random effects: seed=123 dùng cho span_tiny và seed=123 dùng
cho span_baseline là 2 LẦN RÚT MẪU NGẪU NHIÊN ĐỘC LẬP (chỉ trùng NHÃN seed,
không có cơ chế thống kê nào liên kết chúng -- kiến trúc khác nhau, dữ liệu
ảnh SR khác nhau). Vì vậy sr_seed/downstream_seed được LỒNG (nested) theo
từng condition khi đưa vào vc_formula, KHÔNG dùng chung 1 mức "seed=123" cho
cả 2 domain -- nếu dùng chung sẽ ngầm giả định 2 lần rút mẫu đó tương quan
với nhau, một giả định sai và không kiểm chứng được.

Chạy:
    python data/analyze_nested_seed_variance.py --config configs/config.yaml \
        --out_csv results/nested_seed_grid/mixed_effects_summary.csv
"""
import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml
from scipy.stats import norm

DOMAIN_TO_CONDITION = {"sr_improved": "span_tiny", "sr_baseline": "span_baseline"}

MULTISEED_RE = re.compile(r"^(sr_improved|sr_baseline)_(.+)_seed(\d+)\.json$")
SRSEED_RE = re.compile(r"^(.+)_srseed(\d+)\.json$")
NESTED_RE = re.compile(r"^(span_tiny|span_baseline)_(.+)_srseed(\d+)_dseed(\d+)\.json$")


def _load_multiseed(results_root):
    """sr_seed=42 (ngầm định, KHÔNG có trong tên file), downstream_seed từ '_seed<D>'."""
    rows = []
    d = Path(results_root) / "multi_seed"
    if not d.exists():
        return rows
    for f in d.glob("*_seed*.json"):
        m = MULTISEED_RE.match(f.name)
        if not m:
            continue
        domain, backbone, d_seed = m.group(1), m.group(2), int(m.group(3))
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        rows.append({
            "condition": DOMAIN_TO_CONDITION[domain], "backbone": backbone,
            "sr_seed": 42, "downstream_seed": d_seed,
            "identity_accuracy": data["identity_accuracy"], "source": "multi_seed",
        })
    return rows


def _load_sr_seed_variance(results_root):
    """downstream_seed=42 (ngầm định). BỎ QUA sr_seed=42 (trùng với multi_seed's (42,42) ô --
    lấy từ multi_seed làm nguồn CHUẨN duy nhất cho ô đó, tránh đếm 2 lần)."""
    rows = []
    for condition in ("span_tiny", "span_baseline"):
        d = Path(results_root) / f"sr_seed_variance_{condition}"
        if not d.exists():
            continue
        for f in d.glob("*_srseed*.json"):
            m = SRSEED_RE.match(f.name)
            if not m:
                continue
            backbone, sr_seed = m.group(1), int(m.group(2))
            if sr_seed == 42:
                continue  # đã có từ multi_seed, tránh trùng
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            rows.append({
                "condition": condition, "backbone": backbone,
                "sr_seed": sr_seed, "downstream_seed": 42,
                "identity_accuracy": data["identity_accuracy"], "source": "sr_seed_variance",
            })
    return rows


def _load_nested_grid(results_root):
    rows = []
    d = Path(results_root) / "nested_seed_grid"
    if not d.exists():
        return rows
    for f in d.glob("*.json"):
        m = NESTED_RE.match(f.name)
        if not m:
            continue
        condition, backbone, sr_seed, d_seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        rows.append({
            "condition": condition, "backbone": backbone,
            "sr_seed": sr_seed, "downstream_seed": d_seed,
            "identity_accuracy": data["identity_accuracy"], "source": "nested_seed_grid",
        })
    return rows


def build_dataset(results_root):
    rows = _load_multiseed(results_root) + _load_sr_seed_variance(results_root) + _load_nested_grid(results_root)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    dup_mask = df.duplicated(subset=["condition", "backbone", "sr_seed", "downstream_seed"], keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup:
        print(f"CẢNH BÁO: {n_dup} ô (condition,backbone,sr_seed,downstream_seed) trùng lặp giữa các "
              f"nguồn -- chỉ giữ dòng đầu tiên. Kiểm tra lại nếu số này > 0 (không nên xảy ra với "
              f"logic loại trừ hiện tại).")
        df = df[~dup_mask].reset_index(drop=True)
    # Lồng seed theo condition -- xem giải thích ở đầu file.
    df["sr_seed_id"] = df["condition"] + "_sr" + df["sr_seed"].astype(str)
    df["d_seed_id"] = df["condition"] + "_d" + df["downstream_seed"].astype(str)
    df["_all"] = "all"
    return df


def report_grid_coverage(df):
    print("== Độ phủ lưới (condition x backbone): số ô (sr_seed, downstream_seed) có dữ liệu ==")
    for condition in sorted(df["condition"].unique()):
        sub = df[df["condition"] == condition]
        for backbone in sorted(sub["backbone"].unique()):
            n_cells = sub[sub["backbone"] == backbone][["sr_seed", "downstream_seed"]].drop_duplicates().shape[0]
            print(f"  {condition:14s} {backbone:20s} n_cells={n_cells}")


def fit_mixed_effects(df):
    """accuracy ~ condition * backbone (fixed effects: khớp đúng câu hỏi Table 3 -- khoảng cách
    span_tiny/span_baseline ĐƯỢC PHÉP khác nhau theo từng backbone, có tương tác, không giả định
    khoảng cách cố định qua mọi backbone), với 2 nguồn phương sai ngẫu nhiên CHÉO (crossed) --
    sr_seed_id và d_seed_id -- tách riêng khỏi residual. groups='_all' (1 nhóm duy nhất, hằng số)
    để vc_formula ước lượng phương sai chéo toàn cục thay vì phân cấp (nested) theo groups -- đúng
    cách dùng vc_formula cho crossed random effects trong statsmodels khi không có cấu trúc lồng
    tự nhiên nào khác."""
    # [MỚI — phát hiện qua mô phỏng kiểm chứng riêng, không phải chạy thật ở
    # đây] Với số nhóm ngẫu nhiên (SR-seed, downstream-seed) còn khiêm tốn
    # (4-5/condition), REML fit của statsmodels HẦU NHƯ LUÔN in
    # ConvergenceWarning kiểu "MLE may be on the boundary of the parameter
    # space" -- mô phỏng riêng cho thấy với chỉ 2-3 mức SR-seed/condition,
    # cảnh báo này đi kèm ~13% khả năng ước lượng SAI HƯỚNG (không phải lỗi
    # code, mà là giới hạn cỡ mẫu thật); với 4-5 mức (thiết kế hiện tại),
    # cảnh báo vẫn xuất hiện nhưng KHÔNG còn quan sát thấy sai hướng qua 30
    # lần mô phỏng lặp lại. BẮT BUỘC bắt cảnh báo này và in ra rõ ràng --
    # KHÔNG được để âm thầm biến mất, người đọc số liệu phải biết fit có ở
    # biên tham số hay không trước khi tin con số phương sai báo ra.
    model = smf.mixedlm(
        "identity_accuracy ~ condition * backbone",
        data=df,
        groups="_all",
        vc_formula={
            "sr_seed": "0 + C(sr_seed_id)",
            "downstream_seed": "0 + C(d_seed_id)",
        },
    )
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        result = model.fit(reml=True)
        conv_warnings = [str(w.message) for w in wlist
                          if "Convergence" in w.category.__name__ or "boundary" in str(w.message).lower()
                          or "positive definite" in str(w.message).lower()]
    return result, conv_warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out_csv", default=None,
                     help="mặc định: <results_root>/nested_seed_grid/mixed_effects_summary.csv")
    ap.add_argument("--tost_margin", type=float, default=0.01,
                     help="[MỚI — trả lời phản biện M4] biên tương đương cho TOST trên phép kiểm "
                          "định gộp (đơn vị accuracy, mặc định 0.01 = 1 điểm %%) -- khớp ĐÚNG giá trị "
                          "mặc định của data/aggregate_multi_seed_results.py's --equivalence_margin "
                          "(Section~sec:stats), để phép kiểm định gộp này so sánh được trực tiếp với "
                          "TOST per-backbone đã có trong Table 3 (tên flag khác nhau có chủ đích -- "
                          "script đó gọi tost_paired() cho từng backbone riêng, ĐÚNG tên hàm nhưng "
                          "KHÔNG có flag --tost_margin; chỉ giá trị mặc định 0.01 là dùng chung).")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    results_root = cfg["paths"]["results_root"]

    df = build_dataset(results_root)
    if df.empty:
        print("Không tìm thấy dữ liệu nào -- chạy pipeline/run_sr_seed_variance.sh, "
              "run_sr_seed_variance_backbones.sh, và run_nested_seed_grid.sh trước.")
        return

    report_grid_coverage(df)
    print(f"\nTổng số quan sát: {len(df)}")

    result, conv_warnings = fit_mixed_effects(df)
    print("\n" + "=" * 70)
    print(result.summary())
    print("=" * 70)

    # [MỚI — trả lời trực tiếp phản biện "p-value ở Table 3 tính trên nguồn
    # phương sai sai (chỉ downstream-seed, bỏ qua SR-seed vốn lớn hơn nhiều
    # lần)"] Hệ số chính `condition[T.span_tiny]` KHÔNG phải hiệu ứng trung
    # bình gộp qua 12 backbone -- vì công thức có tương tác
    # (condition*backbone), hệ số đó chỉ là hiệu ứng của ĐÚNG backbone tham
    # chiếu (backbone đầu tiên theo thứ tự alphabet, bị gộp vào Intercept,
    # không xuất hiện riêng trong danh sách backbone[T.*]). Muốn có phép kiểm
    # định "tiny vs baseline, gộp cả 12 backbone, tính ĐÚNG cả 2 nguồn phương
    # sai" cần 1 tổ hợp tuyến tính (linear contrast): hệ số chính + trung bình
    # cộng của mọi hệ số tương tác condition:backbone, KHÔNG thể suy ra bằng
    # mắt từ bảng in phía trên -- phải tính qua result.t_test() với ma trận
    # contrast đúng trên object model đã fit.
    # [SỬA — bug phát hiện qua test end-to-end trên FILE THẬT, không chỉ bản
    # test scratch] result.params gồm CẢ hệ số cố định (fixed effects) LẪN
    # biến phương sai (downstream_seed Var, sr_seed Var) nối ở cuối -- dùng
    # nó làm contrast cho t_test() sai độ dài (result.t_test() chỉ nhận đúng
    # k_fe cột). Phải dùng result.fe_params (CHỈ hệ số cố định) để khớp đúng
    # số cột mà t_test() yêu cầu.
    param_names = list(result.fe_params.index)
    main_name = "condition[T.span_tiny]"
    if main_name not in param_names:
        print(f"\nCẢNH BÁO: không tìm thấy hệ số '{main_name}' -- bỏ qua phép kiểm định gộp.")
    else:
        interaction_prefix = "condition[T.span_tiny]:backbone[T."
        interaction_idx = [i for i, n in enumerate(param_names) if n.startswith(interaction_prefix)]
        n_backbones_total = df["backbone"].nunique()
        # [Guard] số hệ số tương tác PHẢI đúng bằng (số backbone - 1) -- 1
        # backbone tham chiếu không có hệ số tương tác riêng (gộp vào hệ số
        # chính). Sai số này thường do công thức model đổi mà quên cập nhật
        # chỗ này, hoặc do backbone bị lọc thiếu ở bước build_dataset.
        expected_n_interactions = n_backbones_total - 1
        if len(interaction_idx) != expected_n_interactions:
            print(f"\nCẢNH BÁO: kỳ vọng {expected_n_interactions} hệ số tương tác "
                  f"condition:backbone (= {n_backbones_total} backbone - 1 tham chiếu), "
                  f"nhưng tìm thấy {len(interaction_idx)} -- BỎ QUA phép kiểm định gộp vì "
                  f"contrast có thể sai. Kiểm tra lại công thức model / dữ liệu đầu vào.")
        else:
            contrast = np.zeros(len(param_names))
            contrast[param_names.index(main_name)] = 1.0
            weight = 1.0 / n_backbones_total
            for i in interaction_idx:
                contrast[i] = weight
            pooled = result.t_test(contrast.reshape(1, -1))
            # [SỬA — bug phát hiện qua test end-to-end] pooled.sd có shape
            # (1,1) (2 chiều) -- numpy hiện tại KHÔNG cho float() trên mảng
            # >0 chiều dù chỉ có đúng 1 phần tử (lỗi "only 0-dimensional
            # arrays can be converted to Python scalars"). Dùng .item() --
            # hoạt động đúng bất kể effect/sd/pvalue có shape (), (1,), hay
            # (1,1), miễn đúng 1 phần tử.
            pooled_est = np.asarray(pooled.effect).item()
            pooled_se = np.asarray(pooled.sd).item()
            pooled_p = np.asarray(pooled.pvalue).item()
            print("\n== Phép kiểm định gộp: span_tiny vs.\\ span_baseline, TRUNG BÌNH qua "
                  f"cả {n_backbones_total} backbone (tính đúng cả 2 nguồn phương sai) ==")
            print(f"  Ước lượng hiệu ứng (span_tiny - span_baseline) = {pooled_est:+.4f}")
            print(f"  Sai số chuẩn (SE)                             = {pooled_se:.4f}")
            print(f"  p-value                                       = {pooled_p:.4f}")
            print("  (Đây là phép kiểm định trả lời trực tiếp phản biện: p-value ở Table 3 chỉ")
            print("  dùng phương sai downstream-seed (cố định SR-seed=42); phép kiểm định này")
            print("  gộp CẢ HAI nguồn phương sai qua mô hình crossed random-effects ở trên.)")

            # [MỚI — trả lời trực tiếp phản biện M4] Không dừng lại ở "p=0.7961,
            # không có ý nghĩa" -- non-significance KHÔNG phải bằng chứng
            # equivalence (chính bài báo đã nhấn mạnh điều này ở Section~sec:stats,
            # nên áp dụng đúng quy tắc đó cho CHÍNH phép kiểm định này thay vì
            # tự mâu thuẫn). Chạy TOST thật trên contrast đã tính ở trên, dùng
            # ĐÚNG cùng phân phối (chuẩn/z, không phải t) mà result.t_test() tự
            # dùng nội bộ -- đã xác nhận qua test end-to-end với dữ liệu giả lập
            # (fit thật 1 model MixedLM cùng công thức, so p từ .t_test() với
            # 2*(1-norm.cdf(|t|)): khớp chính xác, xác nhận statsmodels dùng
            # phân phối chuẩn cho phép kiểm định này, KHÔNG dùng t-distribution).
            # Dùng lại pooled_est/pooled_se đã có (KHÔNG fit lại model, KHÔNG
            # tốn thêm thời gian đáng kể) để tính 2 kiểm định 1 phía kiểu
            # Schuirmann (1987), cùng biên ±tost_margin đã dùng cho mọi TOST
            # khác trong bài báo.
            #
            # LƯU Ý KHÁC BIỆT CÓ CHỦ ĐÍCH với tost_paired() trong
            # data/aggregate_multi_seed_results.py: hàm đó dùng t-distribution
            # (stats.t.cdf, df=n-1=4) vì test cặp trên ĐÚNG n=5 seed -- cỡ mẫu
            # nhỏ, hiệu chỉnh t-distribution là cần thiết. Ở ĐÂY n=600 quan sát
            # (12 backbone x 25 ô x 2 domain) và bản thân p=0.7961 đã báo cáo
            # trong bài (tính bằng result.t_test() mặc định) CŨNG dùng phân
            # phối chuẩn, không phải t -- dùng norm.cdf ở đây giữ TOST nhất
            # quán với ĐÚNG con số NHST đã có, không tự ý đổi giả định phân
            # phối giữa 2 phép tính trên CÙNG 1 ước lượng. Khác biệt z-vs-t
            # giữa 2 script là hợp lý do khác cỡ mẫu (asymptotic OK ở n=600,
            # cần hiệu chỉnh nhỏ-mẫu ở n=5), không phải sơ suất.
            margin = args.tost_margin
            # [MỚI — bổ sung sau review, phát hiện qua test edge-case trực
            # tiếp, không chỉ đọc code] Ba lỗ hổng chưa được chặn trước khi
            # thêm guard này: (1) pooled_se=0 (suy biến hoàn toàn) làm
            # ZeroDivisionError -- crash khó hiểu SAU KHI đã tốn thời gian fit
            # model; (2) pooled_se=NaN (có thể xảy ra nếu ma trận hiệp phương
            # sai của contrast bị suy biến ở 1 fit boundary/singular) KHÔNG
            # crash mà âm thầm in "z=nan, p=nan" -- verdict "KHÔNG xác nhận
            # tương đương" khi đó ĐÚNG theo nghĩa đen (nan<0.05 = False) nhưng
            # SAI LÝ DO (phép tính hỏng, không phải test thật sự thất bại);
            # (3) margin<=0 (nhập sai qua CLI) không được validate, ra kết quả
            # vô nghĩa mà không cảnh báo. Cùng tinh thần "thà crash to còn hơn
            # sai âm thầm" đã áp dụng ở eval_recognition.py/train_sr_distill.py
            # trong dự án -- chặn cả 3 tường minh ở đây thay vì để lan xuống
            # phép tính rồi mới phát hiện (hoặc không phát hiện) sau đó.
            if not (margin > 0):
                raise ValueError(
                    f"--tost_margin phải > 0 (nhận được {margin}) -- biên tương đương bằng 0 hoặc "
                    f"âm không có ý nghĩa thống kê cho TOST.")
            if pooled_se != pooled_se:  # NaN check không cần import thêm math.isnan
                raise RuntimeError(
                    "pooled_se = NaN -- ma trận hiệp phương sai của contrast có thể bị suy biến "
                    "(liên quan cảnh báo boundary/convergence bên dưới, nếu có). KHÔNG tính TOST "
                    "trên SE không hợp lệ; kiểm tra lại fit trước khi tin bất kỳ số liệu nào từ model này.")
            if abs(pooled_se) < 1e-12:
                raise RuntimeError(
                    f"pooled_se = {pooled_se:.2e} (~0) -- trường hợp suy biến, tương tự MnasNet-1.0's "
                    f"degenerate row ở Table 12 (std hiệu số =0). Không chia được cho SE~0; nếu điều "
                    f"này thật sự xảy ra, quyết định equivalence trực tiếp bằng |pooled_est| < margin "
                    f"thay vì qua z-test, giống cách tost_paired() xử lý trường hợp suy biến.")
            z_lower = (pooled_est - (-margin)) / pooled_se
            z_upper = (pooled_est - margin) / pooled_se
            p_lower = 1.0 - norm.cdf(z_lower)   # H0: hiệu ứng thật <= -margin
            p_upper = norm.cdf(z_upper)         # H0: hiệu ứng thật >= +margin
            p_tost = max(p_lower, p_upper)
            equivalent = p_tost < 0.05
            print(f"\n== TOST trên phép kiểm định gộp (biên tương đương +/-{margin:.4f}, "
                  f"tuong duong +/-{margin*100:.2g} diem %) ==")
            print(f"  H0 (hiệu ứng thật <= -{margin:.4f}): z={z_lower:+.3f}, p={p_lower:.5f}")
            print(f"  H0 (hiệu ứng thật >= +{margin:.4f}): z={z_upper:+.3f}, p={p_upper:.5f}")
            print(f"  p_TOST = max(p_lower, p_upper)                = {p_tost:.5f}")
            print(f"  => {'TƯƠNG ĐƯƠNG được xác nhận' if equivalent else 'KHÔNG xác nhận được tương đương'} "
                  f"(ngưỡng p_TOST < 0.05, khớp quy ước TOST dùng xuyên suốt bài báo)")
            if conv_warnings:
                print("  LƯU Ý QUAN TRỌNG: model này có cảnh báo boundary/convergence (in bên dưới) --")
                print("  SE=±{:.4f} dùng để tính TOST ở trên lấy từ CHÍNH model đó, nên kết quả TOST".format(pooled_se))
                print("  này nên được đọc với cùng mức thận trọng đã áp dụng cho tỉ lệ phương sai")
                print("  26:1 (xác nhận đúng hướng/độ lớn, không phải ước lượng chính xác tuyệt đối).")

    if conv_warnings:
        print("\n" + "!" * 70)
        print("CẢNH BÁO CONVERGENCE/BOUNDARY (đọc trước khi tin số liệu bên dưới):")
        for w in conv_warnings:
            print(f"  - {w}")
        print("Ý nghĩa: fit có thể nằm ở biên không gian tham số (thường là 1 thành phần")
        print("phương sai bị ép về gần 0). Mô phỏng kiểm chứng riêng (n=30 lần lặp, xem")
        print("comment trong fit_mixed_effects()) cho thấy với >=4 mức SR-seed/condition")
        print("(thiết kế hiện tại), cảnh báo này KHÔNG đi kèm sai hướng ước lượng qua 30/30")
        print("lần thử -- nhưng vẫn nên đọc điểm ước lượng bên dưới như một chỉ dấu định")
        print("hướng, không phải con số chính xác tuyệt đối.")
        print("!" * 70)
    else:
        print("\n(Không có cảnh báo convergence/boundary nào.)")

    # [SỬA — bug phát hiện qua test với dữ liệu biết trước đáp số] statsmodels
    # KHÔNG giữ thứ tự khai báo trong vc_formula (dict) -- exog_vc.names bị
    # sắp lại theo ALPHABET nội bộ ("downstream_seed" < "sr_seed"), nên
    # result.vcomp[0]/[1] theo thứ tự khai báo ban đầu (sr_seed trước) sẽ LẤY
    # NHẦM VỊ TRÍ -- đã tái lập bằng test: dữ liệu giả var_sr=4x var_d nhưng
    # code cũ báo ngược lại (var_sr < var_d). Tra tên tường minh qua
    # result.model.exog_vc.names thay vì đoán vị trí theo thứ tự khai báo.
    vc_names = list(result.model.exog_vc.names)
    var_sr_seed = float(result.vcomp[vc_names.index("sr_seed")])
    var_d_seed = float(result.vcomp[vc_names.index("downstream_seed")])
    var_resid = float(result.scale)
    total = var_sr_seed + var_d_seed + var_resid

    # [SỬA — bug phát hiện qua test với lưới đầy đủ 5x5] guard cũ chỉ kiểm
    # tra "var_d_seed > 0" -- khi REML ép 1 thành phần phương sai về gần biên
    # (không hẳn = 0 tuyệt đối, ví dụ 4.2e-14 do sai số số học của optimizer),
    # guard này vẫn qua nhưng tỷ lệ ra một con số vô nghĩa (đã tái lập:
    # 91317971.153 khi var_d_seed chỉ ~4e-14) -- cùng LOẠI lỗi "chia cho số
    # gần-không" mà aggregate_sr_seed_variance.py đã tránh bằng ngưỡng 1e-9,
    # nhưng bị bỏ sót ở đây. Dùng ngưỡng tuyệt đối MIN_VAR_EPS thay vì bare >0.
    MIN_VAR_EPS = 1e-9
    degenerate = var_d_seed <= MIN_VAR_EPS or var_sr_seed <= MIN_VAR_EPS
    ratio_str = "NA (phương sai gần 0 -- kết quả không đáng tin, xem cảnh báo boundary ở trên)" \
        if degenerate else f"{var_sr_seed/var_d_seed:.3f}"

    print("\n== Phương sai tách theo nguồn (mixed-effects, REML) ==")
    print(f"  Var(SR-seed)          = {var_sr_seed:.6f}  ({100*var_sr_seed/total:.1f}% tổng phương sai)")
    print(f"  Var(downstream-seed)  = {var_d_seed:.6f}  ({100*var_d_seed/total:.1f}% tổng phương sai)")
    print(f"  Var(residual)         = {var_resid:.6f}  ({100*var_resid/total:.1f}% tổng phương sai)")
    print(f"  Tỷ lệ Var(SR-seed)/Var(downstream-seed) = {ratio_str}")

    out_path = Path(args.out_csv) if args.out_csv else Path(results_root) / "nested_seed_grid" / "mixed_effects_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_row = {
        "n_obs": len(df), "var_sr_seed": var_sr_seed, "var_downstream_seed": var_d_seed,
        "var_residual": var_resid,
        "pct_sr_seed": 100 * var_sr_seed / total, "pct_downstream_seed": 100 * var_d_seed / total,
        "pct_residual": 100 * var_resid / total,
        "ratio_sr_over_downstream": "NA" if degenerate else var_sr_seed / var_d_seed,
        "degenerate_fit": degenerate,
        "n_convergence_warnings": len(conv_warnings),
        "convergence_warnings": " | ".join(conv_warnings) if conv_warnings else "",
    }
    pd.DataFrame([summary_row]).to_csv(out_path, index=False)
    print(f"\nĐã ghi {out_path}")

    df.to_csv(out_path.with_name("nested_seed_grid_raw_data.csv"), index=False)
    print(f"Đã ghi {out_path.with_name('nested_seed_grid_raw_data.csv')} (toàn bộ dữ liệu thô đã gộp)")


if __name__ == "__main__":
    main()
