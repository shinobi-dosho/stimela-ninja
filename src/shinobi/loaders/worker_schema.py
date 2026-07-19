"""Load scabha-dialect worker/config schema YAML (as used by caracal2's
`caracal/schemas/*_schema.yaml`) into a plain pydantic model -- without
depending on scabha itself.

This is a *config* schema, not a *cab* schema: there's no `command`,
`policies`, or `image` here, just nested `inputs:`/`outputs:` parameter
groups describing what a pipeline worker accepts in its config file. See
`shinobi.loaders.cultcargo` for the sibling loader that builds executable
`Cab`s from cult-cargo YAML -- this module deliberately does not reuse
`Scope`/`Cab` for the result, since a worker config is never dispatched as
a step.

Dialect, as actually used by caracal2 (see its `caracal/schemas/`):

* A param node is a dict. If it has a `dtype` key, it's a **leaf**
  parameter. Otherwise it's a **group** whose values are themselves
  leaves/groups, nested arbitrarily deep (e.g. crosscal's
  `rewind_flags.mode`) -- one rule, no special-casing per file.
* dtypes are `str`/`int`/`float`/`bool`/`File` and `List[<inner>]`
  (bracket syntax; see `_modelgen.dtype_to_type`).
* `choices` (a list) maps to `typing.Literal`.
* `implicit` is a template/expression string (`"{current.x}-y.json"` or
  `"=IFSET(...)"`) -- left as a raw, unevaluated string, matching
  `loaders.cultcargo`'s policy on cult-cargo's own expression language.
  A field with `implicit` set is never required from the caller, same
  rule as `loaders.cultcargo._collect`.
* `_include: "(module.path)filename.yaml"` -- a single package-scoped
  string (different from cult-cargo's list-of-plain-paths form), or a
  plain relative-path string, or a list of either. Resolved recursively
  (an included file's own `_include` resolves relative to *its* directory).
* `_use: dotted.path` or `_use: [dotted.path, ...]` -- deep-merges one or
  more dotted lookups (against the fully `_include`-resolved document)
  into the dict it appears in, with that dict's own sibling keys winning
  -- same convention as `loaders.cultcargo`, extended to accept a list.

`writable` (seen in caracal2's `caracal_base.yaml`) is carried onto the
generated field's `json_schema_extra`: a `writable: false` directory input is
bind-mounted read-only by the container backend (see `_leaf_field` and
`backends.container.bind_dirs`). `must_exist`/`path_policies` are still dropped
(path-behaviour hints with no consumer yet).
"""

from __future__ import annotations

import functools
import importlib
import re
import warnings
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, create_model

from shinobi.exceptions import ConfigLoadError
from shinobi.loaders._modelgen import (
    COMMON_LEAF_KEYS,
    dtype_to_type,
    narrow_choices,
    required_field_spec,
    resolve_directive,
    resolve_use,
    sanitize_unique,
    validate_choices,
)


class ConfigSchema(BaseModel):
    """A loaded worker/config schema: just enough to validate and
    introspect a config section -- name, human info, and the pydantic
    models for its `inputs`/`outputs`.
    """

    name: str
    info: str | None = None
    inputs_model: type[BaseModel]
    outputs_model: type[BaseModel]


_PKG_INCLUDE_RE = re.compile(r"^\((?P<module>[\w.]+)\)(?P<file>.+)$")


def load_worker_schema(path: str | Path) -> ConfigSchema:
    """Load a stimela-classic worker schema (YAML) file into a `ConfigSchema`.

    Args:
        path: Path to the YAML worker schema file.

    Returns:
        The built `ConfigSchema`, with `inputs_model`/`outputs_model`
        pydantic models generated from the schema's `inputs`/`outputs`.

    Raises:
        ConfigLoadError: If the file's top-level content isn't a mapping,
            or it has no top-level `name`.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    raw = _resolve_includes(raw, path.parent)
    resolved = resolve_use(raw, raw, error=ConfigLoadError)

    if not isinstance(resolved, dict):
        raise ConfigLoadError(f"worker schema '{path}' must be a mapping, got {resolved!r}")

    name = resolved.get("name")
    if not name:
        raise ConfigLoadError(f"worker schema '{path}' has no top-level 'name'")

    inputs_model = _build_group(f"{name}_Inputs", resolved.get("inputs") or {})
    outputs_model = _build_group(f"{name}_Outputs", resolved.get("outputs") or {})
    return ConfigSchema(
        name=name,
        info=resolved.get("info"),
        inputs_model=inputs_model,
        outputs_model=outputs_model,
    )


def _resolve_includes(node: Any, base_dir: Path) -> Any:
    def entry_to_dict(entry: Any) -> Any:
        """Resolve one `_include` entry to the dict it refers to.

        Args:
            entry: The `_include` entry -- a plain path or `(module)file`
                string; anything else is unsupported and skipped.

        Returns:
            The loaded include's dict content, or `{}` if `entry` is not
            a supported string form.
        """
        if not isinstance(entry, str):
            warnings.warn(
                f"skipping unsupported _include entry {entry!r} in {base_dir} "
                "(only plain-path or (module)file strings are supported)",
                stacklevel=2,
            )
            return {}
        return _load_include(entry, base_dir)

    return resolve_directive(node, "_include", entry_to_dict)


@functools.lru_cache(maxsize=None)
def _load_include_file(path: Path) -> dict[str, Any]:
    """Read, parse, and recursively `_include`-resolve one file, cached on
    its resolved absolute path -- a schema set commonly has many files all
    including the same shared base (e.g. caracal2's `caracal_base.yaml`),
    so without this every worker schema re-reads and re-parses it from disk.
    Safe to cache: `resolve_directive`/`deep_merge` never mutate their
    inputs, so the same returned dict can be reused (and further deep_merged
    from, which always builds a new dict) by every caller.
    """
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigLoadError(f"_include target '{path}' must be a mapping, got {data!r}")
    return _resolve_includes(data, path.parent)


def _load_include(entry: str, base_dir: Path) -> dict[str, Any]:
    if m := _PKG_INCLUDE_RE.match(entry):
        module = importlib.import_module(m.group("module"))
        if not module.__file__:
            raise ConfigLoadError(f"_include module {m.group('module')!r} has no file path")
        path = Path(module.__file__).parent / m.group("file")
    else:
        path = base_dir / entry
    return _load_include_file(path.resolve())


_LEAF_KEYS = COMMON_LEAF_KEYS


def _build_group(model_name: str, spec: dict[str, Any]) -> type[BaseModel]:
    """A key is a **leaf** parameter if its value dict has any recognised
    leaf-descriptor key (`dtype` is common but not required -- e.g. a param
    with only `info`/`required` and no `dtype` still means "a `str`", same
    as `dtype` simply being omitted). Anything else -- including an empty
    dict -- is a **group**: recurse and embed as a nested submodel.
    """
    if not isinstance(spec, dict):
        raise ConfigLoadError(f"expected a mapping for '{model_name}', got {spec!r}")

    definitions: dict[str, tuple[Any, Any]] = {}
    seen: dict[str, str] = {}
    for key, value in spec.items():
        if value is not None and not isinstance(value, dict):
            raise ConfigLoadError(
                f"expected a param/group mapping for '{key}' in '{model_name}', got {value!r}"
            )
        value = value or {}
        field = sanitize_unique(key, seen)
        if _LEAF_KEYS & value.keys():
            definitions[field] = _leaf_field(value)
        else:
            sub_model = _build_group(f"{model_name}_{field}", value)
            if any(f.is_required() for f in sub_model.model_fields.values()):
                # a group with its own required leaf (e.g. `cabs.name`) can't
                # default to `sub_model()` -- that call would itself fail --
                # so the group is required from the caller instead. This
                # also propagates transitively: a required *nested* group
                # already makes its own parent's fields "required" here.
                definitions[field] = (sub_model, Field(..., description=None))
            else:
                definitions[field] = (sub_model, Field(default_factory=sub_model))
    return create_model(model_name, **definitions)


def _leaf_field(value: dict[str, Any]) -> tuple[Any, Any]:
    py_type = dtype_to_type(value.get("dtype", "str"))
    py_type = narrow_choices(py_type, validate_choices(value.get("choices"), error=ConfigLoadError))

    implicit = value.get("implicit")
    required = bool(value.get("required", False)) and implicit is None
    default = value.get("default")

    # `writable` is carried onto the field (via json_schema_extra) so the
    # container backend can mount a `writable: false` directory input read-only
    # (`readonly_path_fields` + `bind_dirs`). It's the one path-behaviour hint
    # with a consumer; `must_exist`/`path_policies` are still dropped.
    # `abbreviation` rides the same channel so `clickutil.build_options` can
    # emit a `-<abbrev>` short flag (see `steps.schema.ParamMeta`).
    extra: dict[str, Any] = {}
    if "writable" in value:
        extra["writable"] = bool(value["writable"])
    if value.get("abbreviation"):
        extra["abbreviation"] = value["abbreviation"]

    annotation, field_default = required_field_spec(py_type, required, default)
    return (annotation, Field(field_default, description=value.get("info"), json_schema_extra=extra or None))
