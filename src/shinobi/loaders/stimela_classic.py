"""Load stimela-classic style `parameters.json` cab definitions into
shinobi `Cab` objects. This is a *different* cab schema format from
cult-cargo's YAML (see `shinobi.loaders.cultcargo`) -- useful for exactly
the tools cult-cargo doesn't have a loadable definition for (several CASA
tasks, msutils -- see AGENTS.md/examples/ninja_selfcal.py for which ones
and why).

Classic's format: one JSON file per cab (e.g.
`stimela/cargo/cab/casa_mstransform/parameters.json`), a top-level
`task`/`binary`/`base`/`prefix`/`msdir` plus a flat `parameters` list --
unlike cult-cargo, there's no `_include`/`_use` composition to resolve;
each file is fully self-contained.

Field mapping (into a generated `inputs_model` + `field_meta`):

* `name` -> the model field name, sanitised to a valid identifier if
  needed (the original kept as a `nom_de_guerre`).
* `dtype` -> a Python type on the generated model. A param can declare
  `dtype` as a *list* of alternatives (e.g. `["int", "str"]`) for a
  genuine type union; the first alternative is used and the rest are
  dropped -- narrowing a real union to shinobi's simpler model, not a bug.
* `io: "msfile"` forces dtype to `"MS"` (matching shinobi/cult-cargo
  convention for the main measurement-set parameter), regardless of
  whatever the raw `dtype` said (almost always `"file"` anyway). `io:
  "input"/"output"` have no separate shinobi concept -- a file-like type
  alone already drives bind-mounting via `path_fields` -- so they're
  otherwise dropped.
* `required`, `default`, `info` -> the model field / its `ParamMeta`.
* `mapping` -> `ParamMeta.nom_de_guerre` (classic's own name for the same
  concept: what the underlying tool actually calls this parameter).
* `choices` -> `ParamMeta.choices`, and the generated model field's real
  annotation is narrowed to `typing.Literal[*choices]` (see
  `_modelgen.narrow_choices`) -- an out-of-set value fails pydantic
  validation, not just a note in `info`. Also still appended to `info` as
  a human-readable parenthetical, for callers that only look at `info`.

`flavour`: classic's CASA-task cabs (`base` containing `"casa"`) are
*not* real standalone executables -- `binary` there is a CASA task name
(mstransform/listobs/flagdata/...), invoked by wrapping it in a CASA
script, not `subprocess.run(["mstransform", ...])`. These load with
`flavour="casa-task"` (shinobi's existing non-executable flavour,
`UnsupportedFlavourError`-guarded in `shinobi.policies` -- see SECURITY.md's
"Never eval()/exec() a cab's command" section), not `"binary"`, so they
can't be silently misrun as if they were real binaries. Cabs with any
other `base` (msutils, wsclean, cubical, ...) are real CLI tools and load
as `flavour="binary"`.

`image`: classic's `base` (e.g. `"stimela/casa"`) is a base-image
*family* name, not a concrete pullable reference -- the real tag/version
lives in separate `tag`/`version` fields (arrays of compatible versions,
no single "the" version). `base` is used as a best-effort `image`
default; override it on the loaded `Cab` if you need a specific real
image.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shinobi.exceptions import CabLoadError
from shinobi.loaders._modelgen import build_model, sanitize_unique, validate_choices
from shinobi.steps.schema import Cab, ParamMeta


def load_file(path: str | Path) -> Cab:
    """Load a stimela-classic cab definition (JSON) file into a `Cab`.

    Args:
        path: Path to the JSON cab definition file.

    Returns:
        The built `Cab` instance.
    """
    return loads(Path(path).read_text())


def loads(text: str) -> Cab:
    """Parse a stimela-classic cab definition from a JSON string.

    Args:
        text: JSON text of the cab definition.

    Returns:
        The built `Cab` instance.
    """
    return _build_cabdef(json.loads(text))


def _build_cabdef(spec: dict[str, Any]) -> Cab:
    base = spec.get("base") or ""
    flavour = "casa-task" if "casa" in base else "binary"

    name = spec["task"]
    fields: dict[str, tuple[str, bool, Any]] = {}
    metas: dict[str, ParamMeta] = {}
    seen: dict[str, str] = {}
    for param in spec.get("parameters", []):
        field = sanitize_unique(param["name"], seen)
        dtype, meta = _build_param(param, original=param["name"], field=field)
        fields[field] = dtype
        if meta is not None:
            metas[field] = meta

    choices = {field: meta.choices for field, meta in metas.items() if meta.choices}

    return Cab(
        name=name,
        command=spec.get("binary", name),
        info=spec.get("description"),
        image=base or None,
        flavour=flavour,
        inputs_model=build_model(f"{name}_Inputs", fields, choices=choices),
        outputs_model=build_model(f"{name}_Outputs", {}),
        field_meta=metas,
    )


def _build_param(
    param: dict[str, Any], *, original: str, field: str
) -> tuple[tuple[str, bool, Any], ParamMeta | None]:
    dtype = param.get("dtype", "str")
    if isinstance(dtype, list):
        dtype = dtype[0] if dtype else "str"
    dtype = str(dtype)

    if param.get("io") == "msfile":
        dtype = "MS"

    choices = validate_choices(param.get("choices"), error=CabLoadError)
    info = param.get("info")
    if choices:
        choices_text = f"choices: {', '.join(str(c) for c in choices)}"
        info = f"{info} ({choices_text})" if info else choices_text

    # the tool's real flag name: classic's `mapping`, else the original
    # param name if sanitising the field name changed it.
    nom = param.get("mapping") or (original if original != field else None)
    field_spec = (dtype, bool(param.get("required", False)), param.get("default"))
    meta = ParamMeta(nom_de_guerre=nom, info=info, choices=choices) if (nom or info or choices) else None
    return field_spec, meta
