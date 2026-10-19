"""Optional real-runtime FITS replacement and sibling protection acceptance.

Set SHINOBI_FITS_TEST_RUNTIME and SHINOBI_FITS_TEST_IMAGE for another runtime.
No image is pulled: Docker/Podman require a cached image, Apptainer/Singularity
require a local SIF. The tool needs Python only; inspection runs on the host.
"""

from pathlib import Path
import os
import shutil
import subprocess

import pytest
from pydantic import BaseModel

from shinobi import Cab, FITSAccess, FITSImage
from shinobi.steps.dispatch import _dispatch
from shinobi.steps.schema import ParamMeta

fits = pytest.importorskip("astropy.io.fits")
np = pytest.importorskip("numpy")
RUNTIME = os.environ.get("SHINOBI_FITS_TEST_RUNTIME", "docker")
IMAGE = os.environ.get("SHINOBI_FITS_TEST_IMAGE", "ghcr.io/shinobi-dosho/breizorro:0.2.0-d0.1.0")


def _available():
    if not shutil.which(RUNTIME):
        return False
    if RUNTIME in ("apptainer", "singularity"):
        return Path(IMAGE).is_file()
    return subprocess.run([RUNTIME, "image", "inspect", IMAGE], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not _available(), reason="FITS live runtime or locally cached Python image unavailable")


class Inputs(BaseModel):
    script: str
    file: FITSImage
    sibling: FITSImage


class Outputs(BaseModel):
    file: FITSImage


class CreateInputs(BaseModel):
    script: str
    target: Path
    sibling: FITSImage


def _image(path):
    hdu = fits.PrimaryHDU(np.zeros((2, 2), dtype=np.int32))
    hdu.header["CTYPE1"] = "LINEAR"
    hdu.header["CTYPE2"] = "LINEAR"
    hdu.writeto(path)
    return path


def _cab(create=False):
    target = "target" if create else "file"
    return Cab(
        name="live-fits",
        command="python3 -c",
        image=IMAGE,
        backend=RUNTIME,
        inputs_model=CreateInputs if create else Inputs,
        outputs_model=Outputs,
        field_meta={
            "script": ParamMeta(positional_head=True),
            target: ParamMeta(positional=True),
            "sibling": ParamMeta(positional=True),
            **({"file": ParamMeta(implicit="{target}")} if create else {}),
        },
        fits_accesses=[FITSAccess(field="file", mode="create" if create else "write"), FITSAccess(field="sibling", mode="read")],
    )


@pytest.mark.parametrize("fail", [False, True])
def test_replace_and_protect_sibling(tmp_path, monkeypatch, fail):
    monkeypatch.chdir(tmp_path)
    target, sibling = _image(tmp_path / "target.fits"), _image(tmp_path / "sibling.fits")
    before, protected = target.read_bytes(), sibling.read_bytes()
    script = """from pathlib import Path
import sys, struct
p, sibling = map(Path, sys.argv[1:])
b = bytearray(p.read_bytes()); b[2880:2896] = struct.pack('>4i', 7, 7, 7, 7)
q = p.with_suffix('.replacement'); q.write_bytes(b); q.replace(p)
for operation in ('overwrite', 'replace'):
    try:
        if operation == 'overwrite': sibling.write_bytes(b'bad')
        else:
            q.write_bytes(b'bad'); q.replace(sibling)
    except OSError: pass
    else: raise AssertionError('read sibling was writable: ' + operation)
""" + ("raise SystemExit(7)\n" if fail else "")
    result = _dispatch(_cab(), None, script=script, file=target, sibling=sibling, cache_dir=str(tmp_path / "cache"))
    assert result.success is not fail
    assert sibling.read_bytes() == protected
    if fail:
        assert target.read_bytes() == before
    else:
        assert fits.getdata(target).tolist() == [[7, 7], [7, 7]]


def test_failed_create_restores_absence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sibling = _image(tmp_path / "sibling.fits")
    target = tmp_path / "created.fits"
    script = "from pathlib import Path;import sys;Path(sys.argv[1]).write_bytes(Path(sys.argv[2]).read_bytes());raise SystemExit(7)"
    result = _dispatch(_cab(create=True), None, script=script, target=target, sibling=sibling, cache_dir=str(tmp_path / "cache"))
    assert not result.success
    assert not target.exists()
