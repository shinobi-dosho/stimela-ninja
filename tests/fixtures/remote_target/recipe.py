"""Fixture recipe file for tests/test_offload_ssh.py's find_cab_deps tests."""

from __future__ import annotations

from pathlib import Path

from shinobi.loaders.cultcargo import load_file

_CABS_DIR = Path(__file__).parent / "cabs"

tool = load_file(_CABS_DIR / "tool.yml")["tool"]
