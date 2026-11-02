# Third-party notices

## DPIR / KAIR — DRUNet architecture (`src/sonyrawai/models.py: UNetRes`)

The `UNetRes` network definition is a re-implementation of `models/network_unet.py`
and the relevant blocks of `models/basicblock.py` from
<https://github.com/cszn/DPIR> / <https://github.com/cszn/KAIR>, kept
parameter-name-compatible with the official `drunet_color.pth` checkpoint.

```
MIT License

Copyright (c) 2019 Kai Zhang (KAIR) / Copyright (c) 2020 Kai Zhang (DPIR)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Model weights are **not** redistributed in this repository; they are downloaded at
runtime from the upstream KAIR GitHub release. See `docs/model_provenance.md`.

## Runtime dependencies (not vendored)

| Package | License | Notes |
|---|---|---|
| rawpy / LibRaw | MIT (rawpy); LibRaw LGPL-2.1 or CDDL-1.0 | Dynamically linked via the rawpy wheel |
| MediaPipe | Apache-2.0 | Optional `skin` extra; Selfie Multiclass model downloaded at runtime (see model card) |
| PyTorch | BSD-3-Clause | |
| OpenCV | Apache-2.0 | |
| tifffile | BSD-3-Clause | |
| NumPy, Pillow, PyYAML | BSD / HPND / MIT | |
