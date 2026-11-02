# Model provenance

## DRUNet colour denoiser (DPIR)

| Field | Value |
|---|---|
| Paper | K. Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior", IEEE TPAMI 2021 |
| Code | https://github.com/cszn/DPIR (MIT), https://github.com/cszn/KAIR (MIT) |
| Checkpoint URL | https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth |
| SHA256 | `479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4` |
| Size | 130,579,305 bytes |
| Architecture | `UNetRes(in_nc=4, out_nc=3, nc=[64,128,256,512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose')`, bias-free |
| Input | sRGB-encoded RGB in [0,1] + 1 channel noise level map (σ/255) |
| Loading | `torch.load(weights_only=True)`, `load_state_dict(strict=True)` |

The SHA256/size were computed on 2026-11-02 from the release asset. The architecture is re-implemented in
`src/sonyrawai/models.py` with the same parameter names (see `THIRD_PARTY_NOTICES.md`) so no upstream code is
executed at runtime.

**v0.2 issues found during audit**
* v0.2 built the network with `downsample_mode='avgpool'`, `upsample_mode='upconv'` → parameter names/shapes do not
  match the checkpoint; strict loading would fail.
* v0.2's first download URL (`raw.githubusercontent.com/cszn/DPIR/master/model_zoo/drunet_color.pth`) returns
  HTTP 404 (14-byte body); `model_zoo/` in the DPIR repo contains only a README.
* v0.2 executed `models/network_unet.py` from a fresh, unpinned `git clone` of DPIR.

**License review:** KAIR and DPIR code are MIT (copyright Kai Zhang). The release weights are distributed alongside
the MIT code without a separate license; training data (DIV2K, Flickr2K, BSD, WED) has its own terms. Review before
commercial use. Weights are never committed to this repository.

## MediaPipe Selfie Multiclass segmenter (optional, skin)

| Field | Value |
|---|---|
| URL | https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite |
| SHA256 | `c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0` |
| Size | 16,371,837 bytes |
| Classes | 0 background, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 others |
| Runtime | `mediapipe>=0.10.30`, Tasks API `vision.ImageSegmenter` (confidence masks) |
| License | MediaPipe is Apache-2.0; see the model card linked from https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter |

**v0.2 issue:** pinned `mediapipe==0.10.21`, which requires `numpy<2`, `protobuf<5` and `jax` — installing it on a
current Colab image downgrades NumPy and can break preinstalled packages. It also used the legacy
`mp.solutions.selfie_segmentation` API (person mask only, not skin).

## Allowed download hosts

`github.com`, `objects.githubusercontent.com`, `release-assets.githubusercontent.com`, `storage.googleapis.com`
(https only). A mirror URL can be configured (`models.drunet_url`) but must be on the allow-list and match the pinned
SHA256.
