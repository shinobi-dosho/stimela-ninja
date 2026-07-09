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
    """Extract structured output values from a cab's console output.

    Args:
        wranglers: Mapping of regex pattern to the list of wrangler action
            strings to apply on each matching line (e.g.
            `"PARSE_OUTPUT:<groupname>:<typename>"`).
        lines: The cab's combined stdout/stderr lines, in order.

    Returns:
        A dict of extracted output field name to typed value, built by
        applying each matched line's actions in order.
    """
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


def parse_output_action(action: str) -> tuple[str, str] | None:
    """Parse one wrangler action string: `(groupname, typename)` for a
    `PARSE_OUTPUT:<groupname>:<typename>` action, or `None` for any other
    (unimplemented) action kind. The one place both `apply_wranglers`
    (which populates the field at run time, below) and
    `graph._wrangler_output_fields` (which needs the field name statically,
    for `check_offloadable`, without running anything) agree on this
    grammar -- so the two can't silently drift on what a wrangler action
    means the way they briefly did (one required exactly 3 colon-separated
    parts, the other tolerated 2+).
    """
    if not action.startswith("PARSE_OUTPUT"):
        return None
    parts = action.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"malformed PARSE_OUTPUT action {action!r} -- expected "
            "'PARSE_OUTPUT:<groupname>:<type>'"
        )
    return parts[1], parts[2]


def _apply_action(action: str, match: re.Match, outputs: dict[str, Any]) -> None:
    parsed = parse_output_action(action)
    if parsed is None:
        return  # other action kinds not implemented in this scaffold

    groupname, typename = parsed
    caster = _TYPES.get(typename, str)
    outputs[groupname] = caster(match.group(groupname))
