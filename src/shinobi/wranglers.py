"""Output wranglers: extract structured outputs from a cab's stdout/stderr
by matching each line against the cab's configured regexes.

Only the ``PARSE_OUTPUT`` action is implemented for now -- enough to pull
named values out of a tool's console output. Display-oriented actions from
stimela2 (HIGHLIGHT, SUPPRESS, SEVERITY, ...) are deliberately left out of
this scaffold; add them to this module if/when a real cab needs them.
"""

from __future__ import annotations

import re
from typing import Any

_TYPES: dict[str, type] = {"str": str, "int": int, "float": float, "bool": bool}


def apply_wranglers(wranglers: dict[str, list[str]], lines: list[str]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    compiled = [(re.compile(pattern), actions) for pattern, actions in wranglers.items()]

    for line in lines:
        for pattern, actions in compiled:
            match = pattern.search(line)
            if not match:
                continue
            for action in actions:
                _apply_action(action, match, outputs)

    return outputs


def _apply_action(action: str, match: re.Match, outputs: dict[str, Any]) -> None:
    if not action.startswith("PARSE_OUTPUT"):
        return  # other action kinds not implemented in this scaffold

    _, groupname, typename = action.split(":")
    caster = _TYPES.get(typename, str)
    outputs[groupname] = caster(match.group(groupname))
