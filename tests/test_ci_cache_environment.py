"""CI helpers keep uv's configured cache available to setup-uv's cleanup."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("script, fallback", [("test-dosho-compat.sh", "compat"), ("install-candidate-pair.sh", "candidate-pair")])
@pytest.mark.parametrize("configured", [False, True])
def test_uv_calls_preserve_configured_cache(tmp_path, script, fallback, configured):
    repo = tmp_path / "ninja"
    scripts = repo / ".github/scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / ".github/scripts" / script
    shutil.copyfile(source, scripts / script)
    candidate = tmp_path / "dosho"
    candidate.mkdir()
    commands = tmp_path / "bin"
    commands.mkdir()
    # Stop at the first uv call, before any network or environment mutation.
    (commands / "uv").write_text('#!/bin/sh\nprintf "%s\\n" "$UV_CACHE_DIR" > "$TEST_UV_CACHE_FILE"\nexit 42\n')
    (commands / "git").write_text('#!/bin/sh\ncase "$*" in\n  *--show-toplevel*) printf "%s\\n" "$TEST_NINJA_REPO" ;;\n  *"rev-parse HEAD"*) printf "%040d\\n" 1 ;;\nesac\n')
    for command in commands.iterdir():
        command.chmod(0o755)
    observed = tmp_path / "observed-cache"
    environment = dict(os.environ, PATH=f"{commands}{os.pathsep}{os.defpath}", TEST_NINJA_REPO=str(repo), TEST_UV_CACHE_FILE=str(observed))
    environment.pop("UV_CACHE_DIR", None)
    expected = tmp_path / "setup uv cache" if configured else repo / ".work-cache" / fallback / "cache"
    if configured:
        environment["UV_CACHE_DIR"] = str(expected)
    argument = "main" if script == "test-dosho-compat.sh" else str(candidate)
    result = subprocess.run(["bash", str(scripts / script), argument], cwd=repo, env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 42, result.stderr
    assert observed.read_text().strip() == str(expected)
