"""Load stimela-classic style `parameters.json` cab definitions into
shinobi `CabDef` objects. This is a *different* cab schema format from
cult-cargo's YAML (see `shinobi.loaders.cultcargo`) -- useful for exactly
the tools cult-cargo doesn't have a loadable definition for (several CASA
tasks, msutils -- see AGENTS.md/examples/ninja_selfcal.py for which ones
and why).

Classic's format: one JSON file per cab (e.g.
`stimela/cargo/cab/casa_mstransform/parameters.json`), a top-level
`task`/`binary`/`base`/`prefix`/`msdir` plus a flat `parameters` list --
unlike cult-cargo, there's no `_include`/`_use` composition to resolve;
each file is fully self-contained.

Field mapping to shinobi's `ParamSchema`:
* `name` -> the schema key, unchanged.
* `dtype` -> lowercased dtype string. A param can declare `dtype` as a
  *list* of alternatives (e.g. `["int", "str"]`) for a genuine type
  union; shinobi's `ParamSchema` has one dtype string, so the first
  alternative is used and the rest are dropped -- narrowing a real union
  to shinobi's simpler model, not a bug to fix.
* `io: "msfile"` forces dtype to `"MS"` (matching shinobi/cult-cargo
  convention for the main measurement-set parameter), regardless of
  whatever the raw `dtype` said (almost always `"file"` anyway). `io:
  "input"/"output"` have no separate shinobi concept -- dtype alone
  already drives bind-mounting via `is_file_like_dtype` -- so they're
  otherwise dropped.
* `required`, `default`, `info` -> direct mapping.
* `mapping` -> shinobi's `nom_de_guerre` (classic's own name for the same
  concept: what the underlying tool actually calls this parameter).
* `choices` -> shinobi has no enum/choices concept; appended to `info`
  as a parenthetical instead of inventing new schema machinery for one
  format's sake.

`flavour`: classic's CASA-task cabs (`base` containing `"casa"`) are
*not* real standalone executables -- `binary` there is a CASA task name
(mstransform/listobs/flagdata/...), invoked by wrapping it in a CASA
script, not `subprocess.run(["mstransform", ...])`. These load with
`flavour="casa-task"` (shinobi's existing non-executable flavour,
`UnsupportedFlavourError`-guarded in `shinobi.policies` -- see AGENTS.md's
"Never eval()/exec() a cab's command" section), not `"binary"`, so they
can't be silently misrun as if they were real binaries. Cabs with any
other `base` (msutils, wsclean, cubical, ...) are real CLI tools and load
as `flavour="binary"`.

`image`: classic's `base` (e.g. `"stimela/casa"`) is a base-image
*family* name, not a concrete pullable reference -- the real tag/version
lives in separate `tag`/`version` fields (arrays of compatible versions,
no single "the" version). `base` is used as a best-effort `image`
default; override it on the loaded `CabDef` if you need a specific real
image.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shinobi.schema import CabDef, ParamSchema


def load_file(path: str | Path) -> CabDef:
    return loads(Path(path).read_text())


def loads(text: str) -> CabDef:
    return _build_cabdef(json.loads(text))


def _build_cabdef(spec: dict[str, Any]) -> CabDef:
    base = spec.get("base") or ""
    flavour = "casa-task" if "casa" in base else "binary"

    inputs = {param["name"]: _build_param(param) for param in spec.get("parameters", [])}

    return CabDef(
        name=spec["task"],
        command=spec.get("binary", spec["task"]),
        info=spec.get("description"),
        image=base or None,
        flavour=flavour,
        inputs=inputs,
    )


def _build_param(param: dict[str, Any]) -> ParamSchema:
    dtype = param.get("dtype", "str")
    if isinstance(dtype, list):
        dtype = dtype[0] if dtype else "str"
    dtype = str(dtype)

    if param.get("io") == "msfile":
        dtype = "MS"

    info = param.get("info")
    choices = param.get("choices")
    if choices:
        choices_text = f"choices: {', '.join(str(c) for c in choices)}"
        info = f"{info} ({choices_text})" if info else choices_text

    return ParamSchema(
        dtype=dtype,
        required=bool(param.get("required", False)),
        default=param.get("default"),
        info=info,
        nom_de_guerre=param.get("mapping"),
    )
