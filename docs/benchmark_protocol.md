# Real-image QA / benchmark protocol

Required before claiming "production-ready". **No results are reported in this repository yet.**

## Test set (minimum 5 Sony A7 II ARW, not committed to git)

1. ISO 100 daylight landscape/architecture (fine detail, highlights)
2. ISO 1600 indoor
3. ISO 6400 low light
4. Animal fur at ~200 mm (run with `--no-skin`)
5. Portrait (skin, eyelashes, hair)

## Variants (same WB, exposure, crop)

* **Baseline:** neutral decode only — `sonyrawai process X.ARW -o base.tif --denoise-backend none --no-tone --no-detail --no-skin`
* **v0.2:** previous release output (if reproducible)
* **v0.3:** `configs/colab_t4.yaml`, `strength 0.70` (also try `sigma` 15/255 and 25/255 for ISO 6400)

## Record per image

| Metric | Source |
|---|---|
| Time per image (s), per-stage timings | sidecar JSON `duration_s`, `timings` |
| Peak VRAM (MB) | sidecar `gpu_peak_memory_mb` |
| Output dimensions / dtype | sidecar `output.shape`, `output.dtype` (must equal decoded) |
| Clipped highlight ratio in/out | sidecar `quality.clipped_ratio_input/output` |
| Colour drift (mean / p95 Δa\*b\*, mean ΔL\*) | sidecar `quality` |
| Subjective 100% crop: fur, eyelashes, skin pores, edges | human review, side-by-side crops |
| Seams (look at multiples of 208 px = tile−overlap), halos, false texture, banding | human review at 100–200% |

PSNR/SSIM only if a genuine clean reference exists (e.g. ISO 100 tripod shot vs ISO 6400 of the same scene, aligned);
otherwise do not report them.

## Deliverables

* Table of the metrics above for each image and variant.
* Before/after 100% crops (PNG, stored outside git or in a release asset).
* Explicit caveats: model limitations, failed cases, any seams/halos observed.
