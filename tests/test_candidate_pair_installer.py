"""Candidate identity checks reject same-version artifacts and source shadowing."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

_INSTALLER = Path(__file__).resolve().parents[1] / ".github/scripts/install-candidate-pair.sh"


@pytest.mark.parametrize("mismatch", [None, "origin", "digest", "version", "source"])
def test_candidate_identity_verification(make_venv, tmp_path, mismatch):
    env = make_venv()
    python = env / "bin/python"
    site = Path(subprocess.check_output([str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
    sources, wheels = [], []
    for distribution, module in (("stimela-ninja", "shinobi"), ("dosho", "dosho")):
        source = tmp_path / module
        source.mkdir()
        (source / "pyproject.toml").write_text('[project]\nversion = "0.2.0b1"\n')
        wheel = tmp_path / f"{module}.whl"
        wheel.write_bytes(b"candidate artifact")
        package = site / module
        package.mkdir()
        (package / "__init__.py").write_text("")
        info = site / f"{distribution.replace('-', '_')}-0.2.0b1.dist-info"
        info.mkdir()
        version = "0.1.0" if mismatch == "version" and module == "dosho" else "0.2.0b1"
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n")
        direct = {
            "url": wheel.as_uri(),
            "archive_info": {"hashes": {"sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}},
        }
        if module == "dosho":
            if mismatch == "origin":
                direct["url"] = (tmp_path / "other-same-version.whl").as_uri()
            elif mismatch == "digest":
                direct["archive_info"]["hashes"]["sha256"] = "0" * 64
        (info / "direct_url.json").write_text(json.dumps(direct))
        sources.append(str(source))
        wheels.append(str(wheel))
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    if mismatch == "source":
        shadow = tmp_path / "shadow" / "dosho"
        shadow.mkdir(parents=True)
        (shadow / "__init__.py").write_text("")
        environment["PYTHONPATH"] = str(shadow.parent)
    # Run the installer's actual verifier against real interpreter metadata.
    verifier = _INSTALLER.read_text().split("<<'VERIFY'\n", 1)[1].split("\nVERIFY", 1)[0]
    result = subprocess.run([str(python), "-c", verifier, *sources, *wheels], cwd=tmp_path, env=environment, capture_output=True, text=True, check=False)
    if mismatch is None:
        assert result.returncode == 0, result.stderr
        assert "Verified dosho" in result.stdout
    else:
        assert result.returncode != 0
        assert "AssertionError" in result.stderr
