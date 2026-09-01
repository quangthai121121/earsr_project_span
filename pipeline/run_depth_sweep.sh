#!/bin/bash
# [MỚI — trả lời phản biện] Tiêu đề/abstract đặt câu hỏi "how much further can
# [span] be compressed before accuracy degrades", nhưng thí nghiệm chính chỉ
# so sánh đúng 1 điểm (3 khối vs.\ 6 khối, Table~depth) -- không đủ để trả
# lời "nén được tới đâu", chỉ trả lời được "cắt còn một nửa thì có sao không".
# Reviewer đề xuất: thêm depth sweep (1,2,4,5 khối) trên 1 backbone đại diện,
# n=5 seed, để dựng đường cong accuracy-vs-depth đầy đủ (1..6 khối).
#
# HẠ TẦNG: KHÔNG cần model mới -- models/sr_models.py::build_sr_model()
# arch="span_pruned" đã nhận n_blocks TUỲ Ý, tạo đúng class SPAN giống hệt
# span_tiny (n_blocks=3)/span_large (n_blocks=6), chỉ khác số khối. Dùng
# CHUNG recipe PIN CỨNG của span_tiny (pipeline/06_improve_span.sh:
# lambda_pixel=1.0, lambda_distill=1.0, feat/saliency/identity=0) để cô lập
# ĐÚNG 1 biến là số khối -- không đổi bất kỳ siêu tham số nào khác.
#
# 2 điểm đã có sẵn (KHÔNG train lại): 3 khối = span_tiny (domain sr_improved),
# 6 khối = span_large (domain sr_span_large, CÙNG class SPAN tự viết với
# span_tiny -- xem ghi chú phương pháp trong Section~arch của bài báo, đây là
# so sánh SẠCH hơn span_baseline vì span_baseline dùng Conv3XC của tác giả
# gốc, khác implementation). Script này CHỈ train 4 điểm mới: 1,2,4,5 khối.
#
# DÙNG:
#   bash pipeline/run_depth_sweep.sh [đường_dẫn_config]
#
# TIỀN ĐỀ: đã chạy xong pipeline chính (span_tiny, domain sr_improved) VÀ
# RUN_ALL_span_large_ablation.sh (+ _extra_seeds.sh, domain sr_span_large,
# backbone mobilenet_v2, n=5 seed) cho CÙNG config.
#
# CẢNH BÁO THỜI GIAN: 4 điểm mới x (1 train SR + 5 lần fine-tune recognition)
# = 4 checkpoint SR mới + 20 lần train+eval recognition.

set -e

CONFIG="${1:-configs/config.yaml}"
BACKBONE="mobilenet_v2"        # đại diện, đúng đề xuất reviewer
DEPTHS=(1 2 4 5)                # điểm MỚI cần train; 3 và 6 đã có sẵn
SEEDS=(42 123 2024 44 999)      # n=5, khớp chuẩn project
NUM_WORKERS="${NUM_WORKERS:-4}"

if [ ! -f "$CONFIG" ]; then
    echo "LỖI: không thấy $CONFIG"
    exit 1
fi

RUNS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['runs_root'])")
SPLITS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['splits_root'])")
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

RESULTS_DIR="${RESULTS_ROOT}/depth_sweep"
mkdir -p "$RESULTS_DIR"

echo "Kiểm tra tiền đề..."
MISSING=0
if [ ! -d "${SPLITS_ROOT}/sr_improved" ]; then
    echo "LỖI: thiếu ${SPLITS_ROOT}/sr_improved (chạy xong pipeline chính -- span_tiny, "
    echo "     domain sr_improved -- trước khi chạy depth sweep)."
    MISSING=1
fi
for SEED in "${SEEDS[@]}"; do
    CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
    if [ ! -f "$CKPT" ]; then
        echo "LỖI: thiếu $CKPT -> chạy pipeline/run_multi_seed.sh (+ _extra_seeds.sh) trước."
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo "DỪNG LẠI — thiếu tiền đề ở trên."
    exit 1
fi
echo "OK — bắt đầu chạy."

for DEPTH in "${DEPTHS[@]}"; do
    RUN_SUFFIX="_depth${DEPTH}"
    DOMAIN_NAME="sr_depth${DEPTH}"
    SR_CKPT="${RUNS_ROOT}/sr_improved_span_pruned${RUN_SUFFIX}/best.pt"

    echo ""
    echo "################################################################"
    echo "# [1/3] Train SR — n_blocks=$DEPTH (checkpoint MỚI)              #"
    echo "################################################################"
    if [ -f "$SR_CKPT" ]; then
        echo ">>> Bỏ qua (đã có checkpoint từ lần chạy trước)"
    else
        # [PIN CỨNG — khớp CHÍNH XÁC pipeline/06_improve_span.sh, recipe của
        # span_tiny] cô lập ĐÚNG 1 biến là n_blocks -- không được để lambda
        # khác đi giữa các điểm trong sweep, nếu không đường cong đo được là
        # đường cong hỗn hợp (depth + recipe) chứ không phải đường cong depth
        # thuần.
        python train_sr_distill.py --config "$CONFIG" --student_arch span_pruned \
            --student_n_blocks "$DEPTH" \
            --lambda_pixel 1.0 --lambda_distill 1.0 --lambda_feat 0 --lambda_saliency 0 --lambda_identity 0 \
            --run_suffix "$RUN_SUFFIX" --num_workers "$NUM_WORKERS"
    fi

    echo ""
    echo "################################################################"
    echo "# [2/3] Sinh ảnh SR + đo chất lượng — n_blocks=$DEPTH            #"
    echo "################################################################"
    # [SỬA — phát hiện qua review] KHÔNG skip theo kiểu "-d out_dir tồn tại
    # thì bỏ qua" như bản trước -- build_sr.py CHỈ tự xoá-và-ghi-lại sạch
    # (shutil.rmtree rồi build lại từ đầu) khi THỰC SỰ được gọi; nếu script
    # này bị ngắt giữa chừng lúc build_sr.py đang chạy dở (đã từng xảy ra thật
    # với train_recognition.py, xem lịch sử Ctrl+C), thư mục out_dir đã tồn
    # tại nhưng CHỈ CÓ MỘT PHẦN ảnh -- check "-d tồn tại" sẽ bỏ qua nhầm,
    # không bao giờ gọi lại build_sr.py để nó tự dọn/ghi lại đầy đủ, để lại
    # domain ảnh SR không đầy đủ mà recognition vẫn dùng để train mà không hề
    # biết. Khớp đúng quy ước AN TOÀN đã dùng ở MỌI script khác trong project
    # (run_sr_seed_variance.sh, run_sr_seed_variance_extra_seeds.sh,
    # run_degradation_robustness_ablation.sh) -- LUÔN gọi build_sr.py vô điều
    # kiện mỗi lần, dựa vào cơ chế tự xoá-và-ghi-lại của chính nó để đảm bảo
    # đúng, chấp nhận chi phí sinh lại ảnh dù có thể đã đủ từ trước.
    python data/build_sr.py --lr_dir "${SPLITS_ROOT}/lr" --sr_ckpt "$SR_CKPT" \
        --arch span_pruned --n_blocks "$DEPTH" --scale "$SCALE" --out_dir "${SPLITS_ROOT}/${DOMAIN_NAME}"
    # [SỬA — phát hiện qua review, chưa chạy thử] eval_sr_quality.py LUÔN
    # append (không dedup, xem giải thích đầy đủ trong
    # data/aggregate_sr_seed_variance.py::aggregate_sr_quality) -- nếu script
    # này bị ngắt giữa chừng (đã từng xảy ra thật, xem lịch sử Ctrl+C) rồi
    # chạy lại, 2 bước train/build ở trên tự bỏ qua đúng (đã có sẵn) nhưng
    # eval_sr_quality KHÔNG có guard nào sẽ chạy lại, sinh dòng "depth${DEPTH}"
    # trùng lặp trong CSV. Guard bằng cách kiểm tra label đã có trong CSV
    # chưa trước khi gọi lại.
    # "label" LUÔN là cột đầu tiên trong CSV này (xem eval_sr_quality.py:186,
    # result = {"label": ..., ...} rồi fieldnames = list(result.keys())) --
    # match chính xác đầu dòng, không cần đoán vị trí cột.
    QUALITY_CSV="${RESULTS_DIR}/sr_quality_depth_sweep.csv"
    if [ -f "$QUALITY_CSV" ] && grep -q "^depth${DEPTH}," "$QUALITY_CSV"; then
        echo ">>> Bỏ qua eval_sr_quality (đã có dòng depth${DEPTH} trong $QUALITY_CSV)"
    else
        python eval_sr_quality.py --config "$CONFIG" --arch span_pruned --n_blocks "$DEPTH" --ckpt "$SR_CKPT" \
            --label "depth${DEPTH}" --out_csv "$QUALITY_CSV"
    fi

    echo ""
    echo "################################################################"
    echo "# [3/3] Fine-tune + eval recognition — n_blocks=$DEPTH (n=5 seed) #"
    echo "################################################################"
    for SEED in "${SEEDS[@]}"; do
        OUT_JSON="${RESULTS_DIR}/${DOMAIN_NAME}_${BACKBONE}_seed${SEED}.json"
        if [ -f "$OUT_JSON" ]; then
            echo ">>> Bỏ qua seed=$SEED (đã có JSON từ lần chạy trước)"
            continue
        fi
        INIT_CKPT="${RUNS_ROOT}/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        RUN_SUFFIX_REC="_seed${SEED}"
        python train_recognition.py --config "$CONFIG" --domain "$DOMAIN_NAME" --backbone "$BACKBONE" \
            --init_ckpt "$INIT_CKPT" \
            --seed "$SEED" --run_suffix "$RUN_SUFFIX_REC" --num_workers "$NUM_WORKERS"
        python eval_recognition.py --config "$CONFIG" \
            --ckpt "${RUNS_ROOT}/recognition_${DOMAIN_NAME}_${BACKBONE}${RUN_SUFFIX_REC}/best.pt" \
            --backbone "$BACKBONE" --train_domain "$DOMAIN_NAME" --test_domain "$DOMAIN_NAME" \
            --out_json "$OUT_JSON" --num_workers "$NUM_WORKERS"
    done
done

echo ""
echo ">>> Tổng hợp đường cong accuracy-vs-depth (1..6 khối)..."
python data/aggregate_depth_sweep.py --config "$CONFIG" \
    --depth_sweep_dir "$RESULTS_DIR" \
    --out_csv "${RESULTS_DIR}/depth_sweep_summary.csv"

echo ""
echo "HOÀN TẤT. Kết quả: ${RESULTS_DIR}/depth_sweep_summary.csv"
