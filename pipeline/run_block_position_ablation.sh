#!/bin/bash
# [MỚI] Mục 5.7.2 bài báo — "block-removal position ablation": span_tiny hiện
# tại giữ khối 1-3 (bỏ khối 4-6) của SPAN 6-khối, một lựa chọn thực hiện vì
# tiện lợi (implementation convenience), chưa từng kiểm chứng so với các cách
# chọn khác cùng giữ ĐÚNG 3 khối. Script này train 3 biến thể, MỖI biến thể
# đều dùng feature-hint distillation nhắm vào ĐÚNG 1 khối cụ thể của teacher
# (xem models/sr_models.py::SPANBlockHook và train_sr_distill.py::
# _setup_position_kd để hiểu ĐỘNG LỰC đầy đủ vì sao cần hint này — tóm tắt:
# không có hint, 1 model 3-khối train từ đầu KHÔNG có khái niệm nội tại nào
# về "tôi đang giữ vai trò khối số mấy", nên 3 biến thể sẽ giống hệt nhau về
# mặt kỳ vọng nếu không có tín hiệu giám sát riêng cho từng vị trí).
#
# QUAN TRỌNG — TRÁNH CONFOUND: cả 3 biến thể dưới đây (kể cả biến thể lặp lại
# đúng lựa chọn hiện tại của span_tiny, "keepfirst") đều dùng CÙNG cơ chế
# hint (lambda_position=0.5), chỉ khác teacher_block_idx — so sánh ĐÚNG "hint
# ở vị trí nào tốt hơn", KHÔNG lẫn với câu hỏi "có hint hay không" (KHÔNG so
# sánh với sr_improved gốc — domain đó không có hint theo-vị-trí).
#
#   keepfirst    (span_tiny hiện tại, giữ khối 1-3) -> teacher_block_idx=2
#   keeplast     (giữ khối 4-6)                     -> teacher_block_idx=5 (cuối)
#   interleaved  (giữ khối 1,3,5)                   -> teacher_block_idx=4
#
# [SỬA — lỗi phương pháp luận phát hiện qua code review vòng 3] Bản đầu tiên
# của script này train 1 SR MODEL MỚI CHO TỪNG SEED (--seed "$SEED" truyền
# vào train_sr_distill.py trong vòng lặp seed) — SAI quy ước đã thống nhất
# xuyên suốt project và ghi rõ trong Methodology của bài báo: "SR models are
# trained once per configuration, at a fixed seed of 42; only the downstream
# recognition training is repeated across seeds" (xem pipeline/
# run_multi_seed_saliency.sh — ĐÚNG mẫu cần theo). Hệ quả kép của lỗi đó:
# (a) tốn gấp 3 lần GPU-giờ một cách vô ích, (b) domain ảnh SR bị đặt tên
# kèm seed (sr_position_X_seedN) nhưng bước eval lại dùng tên KHÔNG kèm seed
# — bước eval_recognition.py sẽ tìm nhầm thư mục splits/ KHÔNG TỒN TẠI (xem
# eval_recognition.py dòng "domain_root = f'{splits_root}/{args.test_domain}'"
# — --test_domain QUYẾT ĐỊNH TRỰC TIẾP thư mục đọc dữ liệu test, không phải
# chỉ là nhãn ghi JSON như giả định ban đầu) và crash NGAY BƯỚC EVAL ĐẦU
# TIÊN. Bản này sửa lại ĐÚNG mẫu run_multi_seed_saliency.sh: train SR 1 LẦN
# mỗi biến thể (seed cố định SR_SEED=42), domain ảnh SR CHUNG cho cả 3 seed,
# CHỈ lặp seed ở bước train_recognition.py (dùng checkpoint LR CÓ seed suffix
# để giữ đúng chuỗi fine-tune hr->lr->sr_position_X theo TỪNG seed, không
# dùng chung 1 checkpoint hr/lr cho mọi seed).
#
# [SỬA — mở rộng theo yêu cầu] GIAI ĐOẠN SÀNG LỌC: n=3 seed x ĐỦ 5 BACKBONE
# (không chỉ mobilenet_v2 như bản đầu) — vì bước train SR KHÔNG phụ thuộc
# backbone (chỉ ảnh hưởng bước recognition phía sau), mở rộng sang 5 backbone
# KHÔNG làm tăng chi phí train SR (vẫn đúng 3 lần, 1 lần/biến thể), chỉ tăng
# chi phí ở bước recognition (rẻ hơn nhiều, vì fine-tune từ checkpoint LR có
# sẵn). Nếu thấy khác biệt đáng chú ý giữa 3 biến thể, mở rộng tiếp sang n=5
# seed theo đúng mẫu đã dùng cho KD v2/saliency/learned-pruning (xem
# RUNBOOK_EarVN1.0.md).
#
# lambda_position=0.5: giá trị KHỞI ĐIỂM hợp lý (cùng bậc độ lớn với
# lambda_feat đã dùng ở KD v2), CHƯA sweep riêng — nếu kết quả sàng lọc cho
# thấy hướng đi này đáng theo đuổi, nên sweep lambda_position trước khi validate
# đầy đủ, giống cách đã làm với lambda_saliency.
#
# [SỬA — lỗi THIẾT KẾ phát hiện qua code review vòng 2] TEACHER dùng ở đây
# PHẢI là span_large, KHÔNG phải span_official (mặc định của toàn bộ
# project). Lý do: span_official được bọc qua SPANWithRescale (models/
# span_official_wrapper.py, import kiến trúc NGOÀI dự án từ external/SPAN/)
# — object đó KHÔNG có thuộc tính `.body` mà SPANBlockHook cần (xem
# models/sr_models.py::SPANBlockHook, đã tự ghi chú rõ hạn chế này).
# span_large dùng CHUNG class SPAN tự viết với span_tiny (có `.body` tương
# thích) — đây CŨNG là phép so sánh "sạch" hơn theo đúng lý do đã nêu trong
# Section~arch của bài báo (span_tiny/span_large cùng 1 class, chỉ khác
# n_blocks; span_baseline/span_official là implementation khác, dùng
# Conv3XC).

set -e

CONFIG="configs/config.yaml"
RESULTS_DIR="results/block_position_screen"
BACKBONES=("mobilenet_v2" "mobilenet_v3_small" "resnet18" "efficientnet_b0" "ghostnet_100")
SEEDS=(42 123 2024)
SR_SEED=42
LAMBDA_POSITION=0.5
TEACHER_ARCH="span_large"
TEACHER_CKPT="runs/sr_improved_span_large/best.pt"
# [MỚI — 2026-08-30, sự cố thật] số DataLoader worker, đọc từ biến môi trường
# nếu có, mặc định 4 (giữ NGUYÊN hành vi cũ). Đặt 0 bằng cách chạy
# "NUM_WORKERS=0 bash pipeline/run_block_position_ablation.sh" khi server dùng
# chung bị job khác chiếm gần hết /dev/shm (đã quan sát thực tế: tmpfs 32G bị
# 1 job khác chiếm tới 90% trong vài phút, gây crash "unable to allocate
# shared memory" giữa lúc train — xem --num_workers trong train_sr_distill.py
# và train_recognition.py để hiểu cơ chế đầy đủ).
NUM_WORKERS="${NUM_WORKERS:-4}"

declare -A VARIANT_TEACHER_IDX=(
    [keepfirst]=2
    [keeplast]=5
    [interleaved]=4
)

mkdir -p "$RESULTS_DIR"

STUDENT_ARCH=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['sr_improve'].get('student_arch', cfg['sr']['arch']))")
SCALE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['image']['scale'])")

# Kiểm tra ĐỦ CẢ 5 backbone x 3 seed = 15 checkpoint recognition_lr_<backbone>
# _seed<seed> NGAY TỪ ĐẦU (giống run_multi_seed_saliency.sh) — fail sớm
# trước khi tốn hàng giờ train SR, thay vì crash giữa chừng ở bước
# train_recognition.py (có thể sau khi đã train xong 1-2 biến thể).
echo "Kiểm tra tiền đề: ${#BACKBONES[@]} backbone x ${#SEEDS[@]} seed = "\
"$((${#BACKBONES[@]} * ${#SEEDS[@]})) checkpoint recognition_lr_*_seed*..."
MISSING=0
for BACKBONE in "${BACKBONES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "LỖI: thiếu $CKPT" >&2
            MISSING=1
        fi
    done
done
if [ "$MISSING" -eq 1 ]; then
    echo "-> chạy pipeline/run_multi_seed.sh (đủ 5 backbone x 3 seed, domain lr) trước." >&2
    exit 1
fi
if [ ! -f "$TEACHER_CKPT" ]; then
    echo "LỖI: thiếu $TEACHER_CKPT (teacher = span_large cho RIÊNG ablation này, xem giải thích"
    echo "     ở đầu file) — chạy 'python train_sr_distill.py --config $CONFIG --student_arch"
    echo "     span_large' trước (Section res-depth/Table depth)."
    exit 1
fi
echo "OK — đủ tiền đề, bắt đầu chạy."

for VARIANT in "${!VARIANT_TEACHER_IDX[@]}"; do
    TEACHER_IDX="${VARIANT_TEACHER_IDX[$VARIANT]}"
    DOMAIN="sr_position_${VARIANT}"
    SR_TAG="position_${VARIANT}"

    echo "################################################################"
    echo "# Bước 1/2: Train SR (position-hint, teacher_block_idx=$TEACHER_IDX) — 1 lần,"
    echo "# seed cố định=$SR_SEED, variant=$VARIANT"
    echo "################################################################"
    if [ -f "runs/sr_improved_${STUDENT_ARCH}_${SR_TAG}/best.pt" ]; then
        echo ">>> Bỏ qua train SR cho $VARIANT (checkpoint đã có từ lần chạy trước)"
    else
        # --min_delta 0.001: PHÒNG NGỪA — lambda_position là loss dạng L1 sau
        # chuẩn hoá instance-norm, CÙNG cấu trúc với lambda_saliency (đã từng
        # gây sự cố thật: EarlyStopping không có min_delta khiến training
        # chạy gần hết max_epochs=500 thay vì dừng sớm thật, xem
        # utils/early_stopping.py + pipeline/run_lambda_saliency_sweep.sh).
        python train_sr_distill.py --config "$CONFIG" \
            --teacher_arch "$TEACHER_ARCH" --teacher_ckpt "$TEACHER_CKPT" \
            --lambda_pixel 1.0 --lambda_distill 1.0 \
            --lambda_feat 0.0 --lambda_saliency 0.0 --lambda_identity 0.0 \
            --lambda_position "$LAMBDA_POSITION" --teacher_block_idx "$TEACHER_IDX" \
            --min_delta 0.001 --seed "$SR_SEED" --run_suffix "_${SR_TAG}" \
            --num_workers "$NUM_WORKERS"
    fi

    echo ""
    echo ">>> Sinh tập $DOMAIN (dùng chung cho cả ${#SEEDS[@]} seed bên dưới)..."
    # [SỬA — lỗi phát hiện qua code review vòng 6] KHÔNG dùng skip-check kiểu
    # "thư mục tồn tại + có >=1 file" ở đây (bản trước đã làm vậy) — nếu lần
    # chạy trước bị dừng giữa chừng lúc build_sr.py đang sinh ảnh (ví dụ xong
    # train/ nhưng chưa xong val/test/), skip-check đó sẽ THẤY CÓ FILE rồi
    # BỎ QUA NHẦM, để lại dataset THIẾU/HỎNG mà không báo lỗi gì — đúng loại
    # "silent data corruption" đã từng xảy ra thật trong project này (xem bài
    # học ở data/check_duplicate_labels.py). build_sr.py TỰ xoá sạch out_dir
    # cũ rồi sinh lại từ đầu mỗi lần gọi (đã đọc code xác nhận) nên AN TOÀN
    # để luôn gọi lại vô điều kiện — đúng mẫu run_multi_seed_saliency.sh
    # (script đó cũng KHÔNG có skip-check cho bước này, vì lý do tương tự).
    python data/build_sr.py --lr_dir splits/lr \
        --sr_ckpt "runs/sr_improved_${STUDENT_ARCH}_${SR_TAG}/best.pt" \
        --arch "$STUDENT_ARCH" --scale "$SCALE" --out_dir "splits/${DOMAIN}"

    echo ""
    echo "################################################################"
    echo "# Bước 2/2: Recognition multi-seed (n=${#SEEDS[@]}) x ${#BACKBONES[@]} backbone"
    echo "# trên domain $DOMAIN"
    echo "################################################################"
    for BACKBONE in "${BACKBONES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            # Bỏ qua tổ hợp ĐÃ CHẠY XONG THẬT SỰ.
            if [ -f "$RESULTS_DIR/${DOMAIN}_${BACKBONE}_seed${SEED}.json" ]; then
                echo ">>> Bỏ qua $DOMAIN backbone=$BACKBONE seed=$SEED (đã có JSON từ lần chạy trước)"
                continue
            fi

            echo "----------------------------------------------------------------"
            echo "variant=$VARIANT | backbone=$BACKBONE | seed=$SEED | domain=$DOMAIN"
            echo "----------------------------------------------------------------"
            # QUAN TRỌNG: fine-tune từ checkpoint LR CỦA CHÍNH SEED NÀY, đúng
            # chuỗi hr -> lr -> sr_position_X, giống hệt quy ước
            # run_multi_seed_saliency.sh — KHÔNG dùng chung 1 checkpoint LR cho
            # mọi seed.
            LR_CKPT="runs/recognition_lr_${BACKBONE}_seed${SEED}/best.pt"
            python train_recognition.py --config "$CONFIG" --domain "$DOMAIN" --backbone "$BACKBONE" \
                --init_ckpt "$LR_CKPT" --seed "$SEED" --run_suffix "_seed${SEED}" \
                --num_workers "$NUM_WORKERS"

            python eval_recognition.py --config "$CONFIG" \
                --ckpt "runs/recognition_${DOMAIN}_${BACKBONE}_seed${SEED}/best.pt" \
                --backbone "$BACKBONE" --train_domain "$DOMAIN" --test_domain "$DOMAIN" \
                --out_json "$RESULTS_DIR/${DOMAIN}_${BACKBONE}_seed${SEED}.json" \
                --num_workers "$NUM_WORKERS"
        done
    done
done

echo ""
echo ">>> Tổng hợp (tái sử dụng aggregator chung, xem data/aggregate_multi_seed_results.py)..."
python data/aggregate_multi_seed_results.py --results_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/block_position_screen_summary.csv"

echo ""
echo "HOÀN TẤT (sàng lọc n=3 x 5 backbone). Xem"
echo "$RESULTS_DIR/block_position_screen_summary_pairwise.csv để so sánh 3 biến thể"
echo "(keepfirst/keeplast/interleaved) trên cả 5 backbone. Nếu thấy khác biệt đáng chú ý,"
echo "mở rộng thêm 2 seed (44, 999) cho đủ n=5 theo đúng mẫu"
echo "pipeline/run_multi_seed_saliency_extra_seeds.sh trước khi đưa vào bảng kết quả chính thức."
