#!/usr/bin/env python3
"""Generate colab/Sony_RAW_AI70_GitHub_T4.ipynb (edit this file, then re-run it)."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "colab" / "Sony_RAW_AI70_GitHub_T4.ipynb"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


code(r'''
# ==== 0. CONFIGURATION (edit this cell only) ====
REPO_URL = 'https://github.com/jayce-vu/photo-ai.git'  # fork? replace with https://github.com/OWNER/REPO.git
REPO_REF = 'main'  # branch, tag, or full commit SHA (use a SHA for reproducible results)
DRIVE_ROOT = '/content/drive/MyDrive/SonyRAWAI'
INPUT_GLOB = '*.ARW'
STRENGTH = 0.70        # blend intensity 0..1 ("70%"), NOT a quality score
TILE = 256             # multiple of 8
OVERLAP = 48           # < TILE/2
ENABLE_SKIN = True     # False for wildlife / landscapes
SIGMA = 10 / 255       # approximate DRUNet noise level; ~15-25/255 for ISO 3200-6400
SINGLE_FILE = ''       # e.g. 'DSC01234.ARW' to process just one file; '' = batch
MAX_FILES = 3          # explicit batch limit per session (free Colab time is limited)
SKIP_EXISTING = True
UPLOAD_FROM_DEVICE = False  # True = pick RAW files from this phone/computer; saved to Drive input/
ALLOW_CPU = False      # True = continue without GPU (very slow)
RUN_TESTS = True       # fast CPU unit tests after install
''')

md(r'''
# Sony RAW AI 70% — GitHub-first Colab (T4)

**Luồng xử lý / Workflow:** GitHub repo → Colab `git clone` → `pip install -e` → tải & kiểm tra SHA256 model DRUNet → mount Google Drive → xử lý ARW → TIFF RGB 16-bit full-size + JSON lưu vào Drive.

* Không cần upload ZIP mã nguồn. / No source ZIP upload.
* **AI denoise thật**: DPIR DRUNet (pretrained, SHA256-verified). Chạy trên ảnh RGB đã demosaic (sRGB-encoded), **không** phải trên dữ liệu Bayer RAW.
* `STRENGTH = 0.70` là mức trộn (blend) do người dùng chọn, **không** phải "chất lượng 70%".
* HDR = tone mapping một khung hình (single exposure), **không** khôi phục vùng sáng bị cháy.
* Skin mask = MediaPipe Selfie Multiclass (face/body skin, 256 px) ∩ ngưỡng màu da — gần đúng.
* Bộ nhớ T4 không được đảm bảo trên Colab miễn phí. File ARW gốc không bị thay đổi.

**Runtime → Change runtime type → T4 GPU**, rồi chạy lần lượt từng cell.

**Private repo:** add a fine-grained, read-only GitHub token in Colab *Secrets* (🔑 panel) named `GH_TOKEN`, enable notebook access, and use a credential helper / `http.extraHeader` populated from `google.colab.userdata.get('GH_TOKEN')` **inside the cell at runtime**. Never paste a token into the notebook, the URL, or printed output.
''')

code(r'''
# ==== 1. Validate repo settings and clone from GitHub ====
import os, re, subprocess, sys
from pathlib import Path

REPO_URL_RE = re.compile(r'^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?(\.git)?$')
REPO_REF_RE = re.compile(r'^(?!-)[A-Za-z0-9._/-]{1,200}$')
if not REPO_URL_RE.match(REPO_URL):
    raise ValueError(f'REPO_URL must be https://github.com/OWNER/REPO(.git), got {REPO_URL!r}')
if not REPO_REF_RE.match(REPO_REF) or '..' in REPO_REF:
    raise ValueError(f'Invalid REPO_REF {REPO_REF!r}')

REPO_DIR = Path('/content/sony-raw-ai')

def git(*args, cwd=REPO_DIR):
    return subprocess.run(['git', *args], cwd=str(cwd), check=True, capture_output=True, text=True).stdout.strip()

if (REPO_DIR / '.git').exists():
    if git('remote', 'get-url', 'origin') != REPO_URL:
        raise RuntimeError(f'{REPO_DIR} is a clone of another repo; delete it or change REPO_URL')
    git('fetch', '--tags', '--prune', 'origin')
else:
    subprocess.run(['git', 'clone', '--', REPO_URL, str(REPO_DIR)], check=True)

resolved = None
for candidate in (f'origin/{REPO_REF}', REPO_REF):
    try:
        resolved = git('rev-parse', '--verify', '--end-of-options', candidate + '^{commit}')
        break
    except subprocess.CalledProcessError:
        pass
if resolved is None:
    raise ValueError(f'REPO_REF {REPO_REF!r} is not a branch, tag or commit in {REPO_URL}')
git('checkout', '--detach', resolved)
COMMIT = git('rev-parse', 'HEAD')
os.environ['SONYRAWAI_GIT_COMMIT'] = COMMIT
print('Repo:', REPO_URL, '| ref:', REPO_REF, '| commit:', COMMIT)
''')

code(r'''
# ==== 2. Install from the cloned repo (keeps Colab's preinstalled torch/CUDA) ====
import importlib, importlib.metadata as md
def _v(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None
before = {k: _v(k) for k in ('numpy', 'torch')}
extras = 'colab,test' if ENABLE_SKIN else 'ai,test'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', f'{REPO_DIR}[{extras}]'], check=True)
after = {k: _v(k) for k in ('numpy', 'torch')}
print('before:', before, '\nafter: ', after)
if before != after:
    print('WARNING: pip changed numpy/torch. Use Runtime > Restart session, then run cells from step 1 again.')
src = str(REPO_DIR / 'src')
if src not in sys.path:
    sys.path.insert(0, src)
importlib.invalidate_caches()
import sonyrawai
print('sonyrawai', sonyrawai.__version__, 'from', sonyrawai.__file__)
''')

code(r'''
# ==== 3. GPU / environment preflight + fast CPU tests ====
cmd = [sys.executable, str(REPO_DIR / 'scripts' / 'verify_environment.py')]
if not ALLOW_CPU:
    cmd.append('--require-cuda')
if ENABLE_SKIN:
    cmd.append('--require-skin')
subprocess.run(cmd, check=True)
if RUN_TESTS:
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-m', 'not gpu and not integration', 'tests'],
                       cwd=str(REPO_DIR), capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-1500:])
    if r.returncode != 0:
        raise RuntimeError('Unit tests failed; do not trust outputs from this commit')
''')

code(r'''
# ==== 4. Mount Google Drive and select inputs ====
from google.colab import drive
drive.mount('/content/drive')
ROOT = Path(DRIVE_ROOT)
INPUT, OUTPUT, MODELS, REPORTS = ROOT / 'input', ROOT / 'output', ROOT / 'models', ROOT / 'reports'
for d in (INPUT, OUTPUT, MODELS, REPORTS):
    d.mkdir(parents=True, exist_ok=True)
WORK = Path('/content/work')
WORK.mkdir(parents=True, exist_ok=True)

from sonyrawai.pipeline import find_inputs
from sonyrawai.raw_io import RAW_SUFFIXES

if UPLOAD_FROM_DEVICE:
    # Photos only (never source code). Large files go through the browser: on mobile data or
    # weak Wi-Fi, uploading via the Google Drive app is more reliable.
    from google.colab import files
    from sonyrawai.export import atomic_write_bytes
    uploaded = files.upload()
    for name, data in uploaded.items():
        safe = Path(name).name
        Path(name).unlink(missing_ok=True)  # files.upload also writes a copy to the working dir
        if Path(safe).suffix.lower() not in RAW_SUFFIXES:
            print(f'skip {safe}: not a RAW file ({", ".join(RAW_SUFFIXES)})')
            continue
        if len(data) < 1_000_000:
            print(f'skip {safe}: only {len(data)} bytes, upload probably incomplete')
            continue
        target = INPUT / safe
        if target.exists():
            print(f'already in Drive input: {safe}')
            continue
        atomic_write_bytes(target, data)
        print(f'uploaded {safe} ({len(data) / 1e6:.1f} MB) -> {target}')
    del uploaded

if SINGLE_FILE:
    if Path(SINGLE_FILE).name != SINGLE_FILE:
        raise ValueError('SINGLE_FILE must be a file name inside the input folder')
    files = [INPUT / SINGLE_FILE]
    if not files[0].is_file():
        raise FileNotFoundError(files[0])
else:
    files = find_inputs(INPUT, INPUT_GLOB)
if not files:
    raise FileNotFoundError(f'Put .ARW files in {INPUT} (Google Drive: MyDrive/SonyRAWAI/input)')
print(f'{len(files)} input file(s) found; up to MAX_FILES={MAX_FILES} not-yet-processed file(s) run per session')
''')

code(r'''
# ==== 5. Download + verify pretrained weights (cached on Drive) and build the pipeline ====
import dataclasses
from sonyrawai.config import load_config
from sonyrawai.models import DRUNET_COLOR, SELFIE_MULTICLASS, download_checkpoint
from sonyrawai.pipeline import Pipeline

for spec in ([DRUNET_COLOR] + ([SELFIE_MULTICLASS] if ENABLE_SKIN else [])):
    path = download_checkpoint(spec, MODELS)  # rejects HTML/404/truncated files and SHA256 mismatch
    print(f'{spec.name}: {path}\n  url={spec.url}\n  sha256={spec.sha256}\n  version={spec.version}\n  license={spec.license}')

cfg = load_config(REPO_DIR / 'configs' / 'colab_t4.yaml')
cfg = dataclasses.replace(
    cfg, strength=STRENGTH, allow_cpu_fallback=ALLOW_CPU, device='auto' if ALLOW_CPU else 'cuda',
    denoise=dataclasses.replace(cfg.denoise, tile=TILE, overlap=OVERLAP, sigma=SIGMA),
    skin=dataclasses.replace(cfg.skin, enabled=ENABLE_SKIN),
    models=dataclasses.replace(cfg.models, cache_dir=str(MODELS)))
cfg.validate()
pipe = Pipeline.from_config(cfg)  # strict state_dict load; no fallback to untrained weights
print('Pipeline ready on', pipe.device)
''')

code(r'''
# ==== 6. Process (per-file error isolation; TIFF built in /content/work, copied atomically to Drive) ====
import json, time
from sonyrawai.pipeline import run_batch
t0 = time.time()
summary = run_batch(pipe, files, OUTPUT, skip_existing=SKIP_EXISTING, continue_on_error=True,
                    max_files=MAX_FILES, work_dir=WORK, report_dir=REPORTS)
print(f"ok={summary['ok']} skipped={summary['skipped']} deferred={summary['deferred']} "
      f"failed={len(summary['failed'])} "
      f"time={time.time() - t0:.1f}s gpu_peak_mb={summary['gpu_peak_mb']}")
for row in summary['rows']:
    print(f"  {row['status']:8s} {Path(row['source']).name}  {row.get('seconds', '')}s  {row.get('error', '')}")
if summary['failed']:
    print('\nFAILED INPUTS:', *[f"{Path(r['source']).name}: {r['error']}" for r in summary['failed']], sep='\n  ')
print('Summary CSV/JSON written to', REPORTS)
''')

code(r'''
# ==== 7. Readback validation, preview and 100% crops (before = neutral decode, after = output) ====
import numpy as np, tifffile
from PIL import Image
from IPython.display import display
from sonyrawai.export import quantize_uint16, sidecar_path
from sonyrawai.raw_io import load_image

ok_rows = [r for r in summary['rows'] if r['status'] in ('ok', 'skipped')]
for row in ok_rows[:2]:
    out = Path(row['output'])
    meta = json.loads(sidecar_path(out).read_text())
    arr = tifffile.imread(out)
    assert arr.dtype == np.uint16 and arr.ndim == 3 and arr.shape[2] == 3
    assert list(arr.shape) == meta['dimensions']['decoded'], 'dimension mismatch'
    print(out.name, arr.shape, arr.dtype, '| model sha256', meta['model'].get('sha256', '')[:16],
          '| commit', meta['repo_commit'], '| peak VRAM MB', meta['gpu_peak_memory_mb'])
    prev = Image.fromarray((arr >> 8).astype(np.uint8)); prev.thumbnail((1200, 1200)); display(prev)
    before = quantize_uint16(load_image(Path(row['source']), cfg.raw).linear)
    h, w = arr.shape[:2]
    for name, (cy, cx) in {'center': (h // 2, w // 2), 'upper-third': (h // 3, w // 2)}.items():
        y0, x0 = max(0, cy - 256), max(0, cx - 256)
        pair = np.concatenate([before[y0:y0 + 512, x0:x0 + 512], arr[y0:y0 + 512, x0:x0 + 512]], axis=1)
        print(f'100% crop ({name}): left = neutral RAW decode, right = output')
        display(Image.fromarray((pair >> 8).astype(np.uint8)))
''')

code(r'''
# ==== 8. (Optional) GPU smoke test with the verified checkpoint ====
RUN_GPU_TEST = False
if RUN_GPU_TEST:
    env = dict(os.environ, SONYRAWAI_CHECKPOINT=str(MODELS / DRUNET_COLOR.filename))
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-m', 'gpu', 'tests'], cwd=str(REPO_DIR),
                       env=env, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-1000:])
''')

md(r'''
## Giới hạn đã biết / Known limitations

* AI denoise chạy trên RGB đã demosaic (sRGB-encoded), không phải Bayer RAW; `SIGMA` là ước lượng, không hiệu chỉnh theo ISO.
* DRUNet có thể làm mịn lông thú / lông mi rất mảnh — luôn kiểm tra crop 100%. Biên độ thay đổi bị giới hạn (`max_delta`).
* HDR một khung hình; vùng sáng đã cháy (clipped) không thể khôi phục.
* Skin mask gần đúng (model 256 px, upsample); có thể sót người nhỏ/xa hoặc nhận nhầm. Tắt `ENABLE_SKIN` cho ảnh động vật.
* TIFF: sRGB-encoded uint16, nhúng ICC sRGB; EXIF gốc không được sao chép (thông tin xử lý nằm trong JSON).
* Colab miễn phí không đảm bảo T4; phiên có thể bị reset — model được cache trong Drive, `SKIP_EXISTING` cho phép chạy tiếp.
* Không có kết quả benchmark nào được khẳng định khi chưa chạy với ARW thật (xem `docs/benchmark_protocol.md`).
''')

for i, c in enumerate(cells):
    c["id"] = f"cell-{i:02d}"
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
