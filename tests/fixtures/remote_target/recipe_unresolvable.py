"""Fixture recipe whose load_file() argument can't be statically resolved
(a runtime-computed path), for tests/test_offload_ssh.py.
"""

from __future__ import annotations

import os

from shinobi.loaders.cultcargo import load_file

_TOOL_NAME = os.environ.get("TOOL_NAME", "tool") + ".yml"

tool = load_file(_TOOL_NAME)["tool"]
