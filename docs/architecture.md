# Architecture

## Stages and colour domains

| # | Stage | Module | Domain in → out | Notes |
|---|-------|--------|-----------------|-------|
| 1 | RAW decode | `raw_io.py` | ARW → **scene-linear** RGB (sRGB primaries), float32 [0,1] | rawpy/LibRaw `postprocess(output_bps=16, gamma=(1,1), no_auto_bright=True, use_camera_wb=True, output_color=sRGB, highlight_mode=clip, half_size=False)`. Orientation (flip flag) applied by LibRaw. Source file opened read-only; SHA256 checked before/after. |
| 2 | AI denoise | `denoise.py`, `tiling.py`, `models.py` | linear → **sRGB-encoded** → DRUNet → encoded → linear | DRUNet was trained on sRGB-encoded images; input = RGB + constant noise map `sigma`. Runs on demosaiced RGB, **not** Bayer RAW. Output = `enc + strength * clip(pred - enc, ±max_delta)`. |
| 3 | Tone ("HDR look") | `tone.py` | linear → linear | Single-exposure. Luminance gain from a blurred base layer: lift shadows / compress highlights; equal gain on R,G,B (ratios preserved); ratio-preserving limit instead of per-channel clipping. |
| 4 | Detail | `detail.py` | encoded luminance | Unsharp residual with noise threshold, change cap and 3×3 local min/max halo clamp; same offset on R,G,B. |
| 5 | Skin | `skin.py` | encoded, low-frequency layer | Mask = MediaPipe Selfie Multiclass (face+body skin) × (1−hair) × wide skin-chroma gate, feathered; area gate → strict no-op. Retouch smooths only the Gaussian(σ=1) low layer with a bilateral filter, re-adds fine detail, bounded by `max_delta`. |
| 6 | Export | `export.py` | linear → sRGB-encoded uint16 | NaN/range check before quantisation; TIFF RGB uint16 exact decoded H×W, deflate, embedded sRGB ICC; atomic write + readback (dtype, shape, photometric, pixel SHA256). Optional 8-bit JPEG preview; JSON sidecar. |

`strength` (default 0.70) scales every stage: denoise blend = strength; tone/detail/skin amount = stage `amount` × strength.

## Tiling

Tiles are always exactly `tile × tile` (default 256, multiple of 8 for DRUNet's three stride-2 levels). Images smaller
than a tile, or thin images, are padded by symmetric reflection and cropped back. Tile origins cover the image with
stride `tile − overlap`, the last tile is aligned to the border. Each prediction is weighted by a separable linear ramp
that is strictly positive (never zero), accumulated, and normalised by the summed weights — no seams, no black
edges. Peak VRAM for DRUNet at 256² on a T4 is expected to be well under 2 GB (to be confirmed by the benchmark).

## Devices

* GPU (CUDA) only for DRUNet inference (`torch.inference_mode()`, tensors freed per tile, peak memory recorded).
* CPU: RAW decode, colour, tone, detail, skin, export.
* `device: cuda` fails with exit code 3 if CUDA is unavailable unless `allow_cpu_fallback` / `--allow-cpu`.

## Memory

A 6000×4000 float32 RGB frame is ~288 MB; the pipeline keeps a handful of copies (input, stage output, tiling
accumulators) → plan for 2–4 GB system RAM per image. Colab free tier provides ~12 GB.

## Failure policy

No silent fallbacks: missing/invalid checkpoint, SHA256 mismatch, architecture mismatch (strict `load_state_dict`),
MediaPipe API mismatch, NaN/Inf, dimension change, and TIFF readback mismatch all raise. Batch mode isolates
failures per file and reports them in `summary-*.csv/json`.
