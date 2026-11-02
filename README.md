# Sony RAW AI — v0.3 (GitHub-first Colab MVP)

Pipeline xử lý ảnh **Sony A7 II `.ARW` → TIFF RGB 16-bit full-size**, ưu tiên giữ chi tiết thật, không "bịa" chi tiết.
A conservative **Sony ARW → full-resolution 16-bit RGB TIFF** pipeline that keeps real detail and does not invent it.

```
ARW (read-only) ─► LibRaw/rawpy demosaic + camera WB (16-bit, linear)
   ─► DPIR DRUNet AI denoise (pretrained, SHA256-verified, tiled on GPU)
   ─► single-exposure local tone mapping ─► luminance detail (halo guard)
   ─► skin retouch (MediaPipe skin segmentation, optional) ─► sRGB uint16 TIFF + ICC + JSON
```

> **Trạng thái / Status:** unit tests chạy trên CPU (CI). **Chưa** chạy end-to-end trên Colab T4 với ARW thật
> trong môi trường phát triển — xem [Smoke test trên Colab](#smoke-test-trên-colab--colab-smoke-test) và
> `docs/benchmark_protocol.md`. Not yet verified end-to-end on a real T4 + real ARW.

---

## 1. Chạy trên Google Colab / Run on Colab (khuyến nghị)

1. Mở notebook: `colab/Sony_RAW_AI70_GitHub_T4.ipynb`
   (Colab → *File → Open notebook → GitHub* → nhập `jayce-vu/photo-ai`, hoặc
   `https://colab.research.google.com/github/jayce-vu/photo-ai/blob/main/colab/Sony_RAW_AI70_GitHub_T4.ipynb`).
2. *Runtime → Change runtime type → **T4 GPU***.
3. Sửa **cell đầu tiên** nếu cần: `REPO_URL`, `REPO_REF` (nên dùng commit SHA để tái lập kết quả), `STRENGTH`, `TILE`,
   `OVERLAP`, `ENABLE_SKIN`, `MAX_FILES`.
4. Chạy lần lượt các cell. Notebook sẽ:
   * kiểm tra `REPO_URL`/`REPO_REF`, `git clone` vào `/content/sony-raw-ai`, in `git rev-parse HEAD`;
   * `pip install -e '.[colab,test]'` (giữ nguyên torch/CUDA có sẵn của Colab), chạy preflight GPU + unit tests;
   * mount Drive, tạo `MyDrive/SonyRAWAI/{input,output,models,reports}`;
   * tải model DRUNet từ release chính thức của KAIR, kiểm tra kích thước + SHA256 (cache vào `models/`);
   * xử lý ARW trong `input/` (giới hạn `MAX_FILES`, `SKIP_EXISTING`, lỗi từng file không làm dừng cả batch),
     dựng TIFF trong `/content/work` rồi copy nguyên tử (atomic) sang `output/`;
   * ghi JSON cho từng ảnh + `reports/summary-*.csv/json` (file lỗi, peak VRAM), hiển thị preview và crop 100%.

**Không** cần upload ZIP mã nguồn. Chỉ đặt file `.ARW` vào `MyDrive/SonyRAWAI/input/`.

**Repo private:** tạo fine-grained token chỉ đọc, lưu trong *Colab Secrets* (`GH_TOKEN`), đọc bằng
`google.colab.userdata.get` lúc chạy và truyền qua credential helper — không dán token vào notebook/URL, không in ra.

## 2. Cài đặt local / Local install

```bash
git clone https://github.com/jayce-vu/photo-ai.git sony-raw-ai && cd sony-raw-ai
python -m venv .venv && . .venv/bin/activate
pip install -e '.[ai,test]'          # + '.[skin]' for MediaPipe skin segmentation
sonyrawai doctor
sonyrawai models download --config configs/local_cpu.yaml
pytest -q
```

## 3. CLI

```bash
sonyrawai process /path/DSC00001.ARW --output /path/DSC00001.tif --config configs/colab_t4.yaml --strength 0.7 --device cuda
sonyrawai batch /path/input --output-dir /path/output --config configs/colab_t4.yaml --glob '*.ARW' --continue-on-error
sonyrawai doctor
sonyrawai models verify --config configs/colab_t4.yaml
```

Tuỳ chọn hữu ích: `--no-skin` (ảnh động vật), `--sigma 0.06` (ISO cao), `--tile/--overlap`, `--allow-cpu`,
`--denoise-backend bilateral|none`, `--work-dir`, `--max-files`, `--no-skip-existing`, `--log-format json`.
Exit codes: `0` ok · `1` processing failure · `2` usage/config error · `3` GPU unavailable · `4` model verification failed.

## 4. Ý nghĩa tham số / What the parameters mean

* **`strength = 0.70`** là *mức trộn* (blend) giữa ảnh gốc và kết quả mà **người dùng chọn** — **không** phải
  độ tin cậy của model hay "chất lượng 70%". Mỗi stage có `amount` riêng × strength.
* **`sigma`** (mặc định 10/255) là mức nhiễu giả định cho DRUNet — **ước lượng**, không hiệu chỉnh theo ISO.
* Mức thay đổi mỗi pixel bị chặn (`denoise.max_delta`, `detail.max_change`, `skin.max_delta`) để bảo vệ texture thật.

## 5. Giới hạn trung thực / Honest limitations

* **AI denoise trên RGB, không phải Bayer RAW.** DRUNet chạy sau demosaic, trên ảnh sRGB-encoded. Nhiễu thật sau
  demosaic có tương quan không gian, nên DRUNet (huấn luyện với AWGN) chỉ là xấp xỉ; có thể làm mịn lông thú/lông mi mảnh.
* **HDR một khung hình** (tone mapping cục bộ) — không phải HDR bracketing; vùng cháy sáng không khôi phục được.
* **Skin mask gần đúng**: MediaPipe Selfie Multiclass (256 px, phân lớp face-skin/body-skin) × ngưỡng màu da, loại tóc,
  rồi feather. Có thể sót người nhỏ/xa; lông thú *thường* bị loại nhưng không đảm bảo → `--no-skin` cho wildlife.
  Không có biến đổi hình học (không liquify/warp).
* **Không đảm bảo tuyệt đối không có chi tiết giả**; kiểm tra ở crop 100–200%.
* **Colab T4 không được đảm bảo** trên gói miễn phí; phiên có thể reset (model cache trong Drive, `SKIP_EXISTING`).
* TIFF nhúng ICC sRGB; EXIF gốc không được sao chép (metadata xử lý nằm trong JSON sidecar).
* RAM: ảnh 24 MP float32 ≈ 290 MB/bản sao; pipeline cần vài GB RAM hệ thống.

## 6. Xuất bản lên GitHub / Publishing to GitHub (copy-paste)

```bash
# Repo already exists at github.com/jayce-vu/photo-ai. For a new fork/copy:
git init && git add . && git commit -m "Sony RAW AI v0.3"
git branch -M main
git remote add origin https://github.com/OWNER/sony-raw-ai.git
git push -u origin main
# Then in the notebook set REPO_URL = 'https://github.com/OWNER/sony-raw-ai.git'
# For reproducibility pin REPO_REF to the commit SHA printed by: git rev-parse HEAD
```

Notebook cần repo **public** để clone ẩn danh (hoặc dùng token như mục 1). Notebook mặc định `REPO_REF='main'`:
hãy merge nhánh phát triển vào `main`, hoặc đặt `REPO_REF` thành tên nhánh / SHA.

## Smoke test trên Colab / Colab smoke test

1. Đặt 1 file ARW (ví dụ ISO 1600) vào `MyDrive/SonyRAWAI/input/`.
2. Cell 0: `MAX_FILES = 1`, `REPO_REF = '<commit SHA>'`; chạy tất cả cell; đặt `RUN_GPU_TEST = True` ở cell 8.
3. Kỳ vọng: preflight in tên GPU "Tesla T4"; unit tests pass; `drunet_color: ... sha256=479abe3c…`;
   `ok=1 failed=0`; TIFF `(4000, 6000, 3) uint16` (A7 II, khi LibRaw decode ở 6000×4000); JSON có `repo_commit`,
   `model.sha256`, `gpu_peak_memory_mb`; GPU smoke test `1 passed`.
4. Ghi lại thời gian/ảnh, peak VRAM, crop 100% theo `docs/benchmark_protocol.md`.

## Tài liệu / Docs

* `docs/architecture.md` — stage, miền màu (linear vs encoded), bộ nhớ.
* `docs/model_provenance.md` — URL, SHA256, license của model.
* `docs/benchmark_protocol.md` — quy trình QA ảnh thật (bắt buộc trước khi gọi là production-ready).
* `CHANGELOG.md` — thay đổi so với v0.2 (bao gồm breaking changes).

License: MIT (xem `LICENSE`, `THIRD_PARTY_NOTICES.md`).
