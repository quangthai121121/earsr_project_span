# Helper: fail-fast nếu thiếu checkpoint recognition domain HR dùng làm
# multi-judge (configs/config.yaml::sr_improve.identity_judges).
# Source từ script khác:  source "$(dirname "$0")/_check_hr_judges.sh"
# rồi gọi check_hr_judges. Cần biến $CONFIG đã được set TRƯỚC khi gọi (dùng
# để đọc yaml — mọi script gọi hàm này đều đã có "CONFIG=configs/config.yaml"
# ở đầu file). Không chạy độc lập.
#
# [SỬA — bổ sung sau code review, vòng 8, điểm 1] TRƯỚC ĐÂY hardcode CỨNG 3
# tên backbone (mobilenet_v2/resnet18/ghostnet_100) VÀ đường dẫn "runs/..." —
# nếu sau này đổi danh sách "identity_judges" trong config.yaml (thêm/bớt/đổi
# backbone, hoặc đổi đường dẫn ckpt), file bash này sẽ LỆCH khỏi những gì
# train_sr_distill.py::build_judges() thực sự đọc (Python luôn đọc động từ
# "sr_improve.identity_judges") — kiểm tra ở đây thành vô nghĩa hoặc báo sai.
# Hiện tại config.yaml đúng 3 backbone/đường dẫn đó nên chưa có triệu chứng gì
# — sửa để 2 bên (bash check + Python thật) LUÔN đọc từ CÙNG 1 nguồn sự thật
# (config.yaml), không phải để sửa 1 bug đang xảy ra.
check_hr_judges() {
    local missing_j=0
    local ckpts f
    echo "Kiểm tra checkpoint recognition HR (multi-judge, đọc từ $CONFIG::sr_improve.identity_judges)..."
    ckpts=$(python -c "
import yaml
ci = yaml.safe_load(open('$CONFIG'))['sr_improve']
judges = ci.get('identity_judges') or [{'backbone': 'mobilenet_v2', 'ckpt': ci['frozen_recognition_ckpt']}]
for j in judges:
    print(j['ckpt'])
")
    while IFS= read -r f; do
        if [ -n "$f" ] && [ ! -f "$f" ]; then
            echo "LỖI: thiếu $f" >&2
            missing_j=1
        fi
    done <<< "$ckpts"
    if [ "$missing_j" -eq 1 ]; then
        echo "     Cần khi lambda_identity>0 hoặc lambda_saliency>0 (kdv2_multijudge/" >&2
        echo "     kdv2_full, lambda sweep, saliency sweep). Chạy" >&2
        echo "     pipeline/03_train_baseline_recognition.sh cho backbone còn thiếu ở trên." >&2
        exit 1
    fi
    echo "OK — đủ checkpoint judge HR (khớp $CONFIG::sr_improve.identity_judges)."
}
