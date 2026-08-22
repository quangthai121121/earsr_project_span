# Ablation attention-parameterization (Mục 5.7(i)) — hướng dẫn áp dụng

Zip này giữ ĐÚNG cấu trúc thư mục con của repo `earsr_project_span` (thư mục
gốc bên trong zip đặt tên `earsr_span/` — đổi tên nếu repo bạn clone tên
khác) — chỉ cần giải nén rồi COPY ĐÈ trực tiếp vào repo thật, không cần
lệnh `patch`/`git apply`. Đã kiểm tra `py_compile`/`bash -n` và
functional-test logic thống kê bằng dữ liệu giả lập trước khi giao.

```
earsr_span/
  models/sr_models.py                              <- ĐÈ file cũ (đã thêm n_blocks cho safmn/smfanet)
  train_sr_distill.py                              <- ĐÈ file cũ (đã thêm --student_n_blocks)
  data/compare_depth_deltas.py                      <- file MỚI
  RUN_ALL_attention_param_ablation.sh               <- file MỚI
  RUN_ALL_attention_param_ablation_extra_seeds.sh   <- file MỚI
  HUONG_DAN_AP_DUNG.md                              <- chính là file này
```

**Cách áp dụng nhanh nhất** (từ thư mục gốc repo thật, ví dụ `earsr_project_span/`):
```bash
unzip earsr_span_attention_ablation_patch.zip
cp -r earsr_span/* /đường/dẫn/tới/earsr_project_span/
```
Vì `models/sr_models.py` và `train_sr_distill.py` là 2 file đã có sẵn trong
repo, lệnh `cp -r` sẽ **ghi đè** đúng 2 file đó — 2 thay đổi duy nhất trong
chúng là: (1) `build_sr_model()` expose thêm `n_blocks` cho `safmn`/`smfanet`
(mặc định vẫn 8 nếu không truyền — an toàn ngược 100% với mọi kết quả cũ đã
chạy), (2) `train_sr_distill.py` thêm cờ `--student_n_blocks` (chỉ có tác
dụng khi `--student_arch safmn`/`smfanet`). Nếu bạn đã tự sửa 2 file này
theo hướng khác từ lúc trao đổi, nên `diff` trước khi ghi đè để không mất
sửa đổi riêng của bạn.

Sau khi copy xong:

1. `RUN_ALL_attention_param_ablation.sh` — chạy từ thư mục gốc repo:
   ```bash
   bash RUN_ALL_attention_param_ablation.sh safmn 4      # n=3 seed đầu
   bash RUN_ALL_attention_param_ablation.sh smfanet 4
   ```
   Tiền đề: đã chạy `RUN_ALL_extra_sr_baseline_distilled.sh <arch>` (bản
   full-depth) và pipeline chính (Bước 1-8) từ trước — script này TÁI SỬ
   DỤNG số liệu đó, không train lại.

2. `RUN_ALL_attention_param_ablation_extra_seeds.sh` — chạy sau bản gốc để
   lên n=5 (khớp quy ước project):
   ```bash
   bash RUN_ALL_attention_param_ablation_extra_seeds.sh safmn 4
   bash RUN_ALL_attention_param_ablation_extra_seeds.sh smfanet 4
   ```

3. `data/compare_depth_deltas.py` — đã nằm sẵn trong `data/` sau khi copy.
   Sau khi đủ n=5 cả hai phía (full-depth có sẵn, half-depth vừa chạy),
   tính kiểm định:
   ```bash
   python data/compare_depth_deltas.py \
       --results_dir results/attention_param_ablation_safmn_half4/combined \
       --arch_full_domain sr_improved_safmn --arch_half_domain sr_improved_safmn_half4 \
       --span_full_domain sr_span_large --span_tiny_domain sr_improved \
       --out_csv results/attention_param_ablation_safmn_half4/depth_delta_comparison_safmn.csv
   ```
   Lặp lại tương tự cho `smfanet`.

   **[SỬA — lỗi confound phát hiện qua review]** Bản đầu của patch này dùng
   `--span_baseline_domain sr_baseline` làm mặc định/ví dụ — ĐÂY LÀ SAI: domain
   `sr_baseline` là teacher SPAN train bằng pixel-loss THUẦN (không cùng recipe
   distillation với `span_tiny`), dùng nó sẽ trộn lẫn biến recipe vào biến độ
   sâu, làm hỏng mục đích cô lập biến độ sâu của chính ablation này. Đã sửa
   thành `--span_full_domain sr_span_large` (domain từ
   `RUN_ALL_span_large_ablation.sh` — CÙNG recipe distillation, chỉ khác độ
   sâu 6 vs 3 khối). Cả `RUN_ALL_attention_param_ablation.sh` và bản
   extra_seeds cũng đã được sửa để copy thêm JSON domain `sr_span_large` từ
   `results/span_large_ablation/` vào `combined/` (bản đầu thiếu bước này) và
   kiểm tra tiền đề "đã chạy `RUN_ALL_span_large_ablation.sh`" ngay ở BƯỚC 0.

## Đọc kết quả thế nào

Cột `diff_of_diffs_mean` > 0 và `verdict = significant` ở một backbone nghĩa
là kiến trúc đó (SAFMN/SMFANet) mất accuracy NHIỀU HƠN SPAN khi cùng giảm
một nửa số khối — đây là bằng chứng ủng hộ giả thuyết Mục 6 ("attention
không tham số chịu nén tốt hơn attention có tham số học"). Ngưỡng ý nghĩa
dùng ĐÚNG quy ước đã có trong bản thảo: p_Bonf < 0.05 = significant,
0.05–0.10 = trend (không gọi significant), ≥0.10 = n.s. — script không tự
đổi ngưỡng.

Không dùng cho `ecbsr` — sau reparameterize kiến trúc này gần như suy biến
thành 1 conv đơn, không còn attention có tham số học thật để so sánh công
bằng (đã ghi rõ trong docstring đầu `RUN_ALL_attention_param_ablation.sh`).
