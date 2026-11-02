"""Static validation of the Colab notebook (no execution)."""
import ast
import json
import re
from pathlib import Path

import pytest

NB = Path(__file__).resolve().parents[1] / "colab" / "Sony_RAW_AI70_GitHub_T4.ipynb"


@pytest.fixture(scope="module")
def nb():
    return json.loads(NB.read_text(encoding="utf-8"))


def _code(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _all(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def test_valid_ipynb(nb):
    assert nb["nbformat"] == 4 and nb["cells"]
    for c in nb["cells"]:
        assert c["cell_type"] in ("markdown", "code")
        if c["cell_type"] == "code":
            assert c.get("outputs") == [] and c.get("execution_count") is None


def test_code_cells_are_valid_python(nb):
    for src in _code(nb):
        lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%"))]
        ast.parse("\n".join(lines))


def test_first_cell_is_configuration(nb):
    first = _code(nb)[0]
    for name in ("REPO_URL", "REPO_REF", "DRIVE_ROOT", "INPUT_GLOB", "STRENGTH", "TILE", "OVERLAP", "ENABLE_SKIN"):
        assert re.search(rf"^{name}\s*=", first, re.M), name
    assert "/content/drive/MyDrive/SonyRAWAI" in first


def test_github_clone_and_editable_install(nb):
    text = "\n".join(_code(nb))
    assert "git('clone'" in text or "'git', 'clone'" in text
    assert "rev-parse" in text and "checkout" in text
    assert re.search(r"pip.*install.*-e", text)
    assert "subprocess.run(" in text and "shell=True" not in text


def test_no_zip_upload_workflow(nb):
    text = _all(nb).lower()
    assert "zipfile" not in text and "extractall" not in text and ".zip" not in text
    # Uploading *photos* is allowed, but only RAW files, saved into the Drive input folder.
    if "files.upload" in text:
        assert "raw_suffixes" in text and "upload_from_device" in text


def test_no_embedded_secrets(nb):
    text = _all(nb)
    for pat in (r"ghp_[A-Za-z0-9]{20,}", r"github_pat_[A-Za-z0-9_]{20,}", r"AKIA[0-9A-Z]{16}",
                r"https://[^/\s]+:[^@\s]+@github\.com", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"):
        assert not re.search(pat, text), pat


def test_no_source_path_assumptions(nb):
    text = _all(nb)
    assert "sony-raw-ai-mvp" not in text
    assert "/content/project" not in text
    assert "MyDrive/SonyRAWAI/sony" not in text


def test_safety_features_present(nb):
    text = "\n".join(_code(nb))
    for token in ("SKIP_EXISTING", "MAX_FILES", "download_checkpoint", "run_batch", "gpu_peak",
                  "summary", "REPO_URL_RE", "REPO_REF_RE"):
        assert token in text, token
