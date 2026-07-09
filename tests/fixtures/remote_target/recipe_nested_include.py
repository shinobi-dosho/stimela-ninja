"""Fixture recipe whose cab file's `_include` is nested under `inputs:`
(mirrors real cult-cargo cubical.yml/quartical.yml, not just a top-level
`_include:`), for tests/test_offload_ssh.py.
"""

from __future__ import annotations

from pathlib import Path

from shinobi.loaders.cultcargo import load_file

_CABS_DIR = Path(__file__).parent / "cabs"

tool = load_file(_CABS_DIR / "nested_include_tool.yml")["nested_tool"]
