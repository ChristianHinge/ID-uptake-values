import json
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from id_uptake_values import sul, suv

MODULES = ["id_uptake_values.suv", "id_uptake_values.sul"]


def _run(module, *args):
    return subprocess.run([sys.executable, "-m", module, *args], capture_output=True, text=True)


@pytest.mark.parametrize("module", MODULES)
def test_help(module):
    result = _run(module, "--help")
    assert result.returncode == 0
    assert "--pet" in result.stdout


@pytest.mark.parametrize("module", MODULES)
def test_missing_required_args_fails(module):
    result = _run(module)
    assert result.returncode != 0
    assert "--pet" in result.stderr


@pytest.mark.parametrize("module", MODULES)
def test_requires_an_output(module, tmp_path):
    required = ["--pet", "p", "--ct", "c", "--totalseg", "t", "--bodyseg", "b"]
    if module.endswith("sul"):
        required += ["--tissueseg", "s"]
    result = _run(module, *required)
    assert result.returncode != 0
    assert "at least one of" in result.stderr


@pytest.mark.parametrize("module,extra", [(suv, []), (sul, ["--tissueseg", "s"])])
def test_main_writes_outputs(module, extra, monkeypatch, tmp_path):
    """main() wiring with the heavy pipeline stubbed out."""
    img = nib.Nifti1Image(np.ones((2, 2, 2), dtype="float32"), np.eye(4))
    monkeypatch.setattr(module.nib, "load", lambda path: img)
    pipeline = "suv_id" if module is suv else "sul_id"
    monkeypatch.setattr(module, pipeline, lambda *a: (img, {"estimated_activity_MBq": 1.0}))

    out_img, out_json = tmp_path / "sub" / "out.nii.gz", tmp_path / "sub" / "out.json"
    argv = ["prog", "--pet", "p", "--ct", "c", "--totalseg", "t", "--bodyseg", "b", *extra,
            "--output-image", str(out_img), "--output-json", str(out_json)]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()

    assert out_img.exists()
    assert json.loads(out_json.read_text()) == {"estimated_activity_MBq": 1.0}
