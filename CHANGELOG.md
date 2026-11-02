# Changelog

## 0.3.0 — GitHub-first Colab MVP

### Added
* Colab notebook `colab/Sony_RAW_AI70_GitHub_T4.ipynb`: clones the GitHub repo (validated `REPO_URL`/`REPO_REF`,
  argument-list `subprocess`), installs with `pip install -e`, GPU preflight, verified model download, batch with
  per-file isolation, `SKIP_EXISTING`, `MAX_FILES`, Drive atomic copy, summary CSV/JSON, 100% crops.
* Modules: `raw_io`, `tiling`, `denoise`, `tone`, `detail`, `skin`, `export`, `quality`, `models`, `config`.
* Verified DRUNet checkpoint (URL + SHA256 + size, HTML/404 rejection, strict loading, `weights_only=True`).
* MediaPipe Tasks *Selfie Multiclass* skin segmentation (face/body-skin classes) replacing person ∩ chroma.
* sRGB ICC embedded in TIFF; TIFF readback validation; JSON sidecar with repo commit, model hash, timings, GPU.
* CLI subcommands `process`, `batch`, `doctor`, `models verify|download`; exit codes; JSON logs.
* YAML configs `configs/colab_t4.yaml`, `configs/local_cpu.yaml` with strict validation.
* GitHub Actions CI (Ubuntu, Python 3.10–3.12, ruff + pytest), GPU tests marked and skipped.

### Fixed (v0.2 blockers)
* DRUNet built with `avgpool/upconv` (does not match the released checkpoint) → `strideconv/convtranspose`.
* First weight URL returned 404; now the official KAIR release asset.
* `mediapipe==0.10.21` pin forced `numpy<2` on Colab → `mediapipe>=0.10.30`, Tasks API, optional extra.
* Tiled inference failed on 1-pixel-thin images (`BORDER_REFLECT_101`) → symmetric padding for any size.
* Notebook required uploading a source ZIP to Drive → GitHub clone.

### Breaking changes
* CLI: canonical form is `sonyrawai process IN -o OUT`. The v0.2 form `sonyrawai IN -o OUT` still works (warning);
  `--disable STAGE` replaced by `--no-tone/--no-detail/--no-skin/--denoise-backend none`; `--skin-mask` (manual mask)
  and the ONNX adapter (`--onnx-model`, non-overlapping tiles) were removed.
* `sonyrawai.colab_ai` removed: use `sonyrawai.tiling.tiled_predict`, `sonyrawai.models.load_drunet(checkpoint, device)`
  (no DPIR repo path), `sonyrawai.pipeline.Pipeline`.
* `sonyrawai.pipeline.Settings/process/export` replaced by `PipelineConfig` + `Pipeline.process_array/process_file`.
* `configs/default.json` replaced by YAML configs.
* Default output naming in batch: `<stem>_sonyrawai.tif` (+ `.json`, `.preview.jpg`).
