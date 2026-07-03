"""Load cult-cargo style YAML cab definitions into shinobi CabDef objects.

The cult-cargo cab *schema* (inputs/outputs/policies/wranglers) is a good
design and is reused as-is here -- it's stimela2's recipe/alias layer that
shinobi drops, not this. This loader lets shinobi use the existing library
of cult-cargo tool wrappers without anyone having to rewrite them.

Real cult-cargo cab files, however, are not self-contained: they rely on
two composition mechanisms from stimela2's config system, which this
loader implements a deliberately minimal version of:

* ``_include: [file, ...]`` -- merges other YAML files in (relative to the
  including file), most often to pull in a shared ``vars:``/``lib:``
  namespace. Merging is a plain deep-merge; the including file's own keys
  win over included ones.

* ``_use: dotted.path`` -- deep-merges a dict looked up by dotted path in
  the fully-merged document (post-``_include``) into the dict it appears
  in, with that dict's own sibling keys taking precedence. Used both for
  small things (``image: {_use: vars.cult-cargo.images, name: breizorro}``)
  and to inherit a cab's entire command/flavour block.

Deliberately NOT implemented (this is the boundary -- see AGENTS.md):

* The ``=config.x.y``/``${...}`` expression language cult-cargo values
  can contain -- left as literal strings.
* The package-scoped include form ``_include: [{(cultcargo): [file,
  ...]}]`` (searches an installed package's data directory rather than a
  relative path), whether at the top level or nested inside ``inputs:``
  (as real cult-cargo's own ``cubical.yml`` does) -- skipped with a
  warning at the top level, raised as a clear CabLoadError if nested
  inside ``inputs:`` (since there's no sensible per-param schema to fall
  back to there).
* ``dynamic_schema: dotted.path`` -- a reference to a Python function
  that would need importing and *calling* to get a cab's real schema
  (real cult-cargo's ``wsclean.yml`` uses this). Resolving it is not just
  a parsing gap like the above: it means executing arbitrary code named
  by a cab file at load time. Not implemented; a cab using only this
  (no static ``inputs:``/``outputs:`` alongside it) loads with an empty
  schema, silently wrong unless you notice the warning this emits.

Building any of these out would mean re-deriving stimela2's config
engine, just relocated from recipes to cabs -- exactly what this project
exists to avoid unless a real cab actually needs it.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml

from shinobi.exceptions import CabLoadError
from shinobi.schema import CabDef, ParamSchema, Policies


def load_file(path: str | Path) -> dict[str, CabDef]:
    path = Path(path)
    raw = _load_raw(path)
    resolved = _resolve_use(raw, raw)
    cabs_section = resolved.get("cabs", resolved)
    return {name: _build_cabdef(name, spec) for name, spec in cabs_section.items()}


def loads(text: str) -> dict[str, CabDef]:
    """Parse cab defs from a YAML string. Supports ``_use`` (resolved
    against the document itself) but not ``_include``, since there's no
    base directory to resolve relative file paths against.
    """
    raw = yaml.safe_load(text) or {}
    resolved = _resolve_use(raw, raw)
    cabs_section = resolved.get("cabs", resolved)
    return {name: _build_cabdef(name, spec) for name, spec in cabs_section.items()}


def _load_raw(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    includes = data.pop("_include", None) or []

    merged: dict[str, Any] = {}
    for inc in includes:
        if not isinstance(inc, str):
            warnings.warn(
                f"skipping unsupported package-scoped _include entry {inc!r} in {path} "
                "(only plain relative-path includes are supported)",
                stacklevel=2,
            )
            continue
        merged = _deep_merge(merged, _load_raw((path.parent / inc).resolve()))

    return _deep_merge(merged, data)


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _deep_merge(merged[key], value) if key in merged else value
        return merged
    return override


def _get_path(root: dict[str, Any], dotted: str) -> Any:
    node: Any = root
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise CabLoadError(f"_use path '{dotted}' not found (stuck at '{part}')")
        node = node[part]
    return node


def _resolve_use(node: Any, root: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_resolve_use(item, root) for item in node]
    if not isinstance(node, dict):
        return node

    node = {key: _resolve_use(value, root) for key, value in node.items()}
    if "_use" in node:
        use_path = node.pop("_use")
        resolved = _resolve_use(_get_path(root, use_path), root)
        node = _deep_merge(resolved, node)
    return node


def _build_cabdef(name: str, spec: dict[str, Any]) -> CabDef:
    image = spec.get("image")
    if isinstance(image, dict):
        image = image.get("name")

    flavour = spec.get("flavour", "binary")
    if isinstance(flavour, dict):
        flavour = flavour.get("kind", "binary")

    if "command" not in spec:
        raise CabLoadError(f"cab '{name}' has no 'command' (check its _use references)")

    if spec.get("dynamic_schema"):
        warnings.warn(
            f"cab '{name}' uses dynamic_schema ({spec['dynamic_schema']!r}), which "
            "shinobi doesn't resolve -- it's a dotted reference to a Python function "
            "that would need importing and calling to get the real schema. Any static "
            "'inputs:'/'outputs:' present are used as-is, but may be incomplete "
            "relative to the tool's actual interface.",
            stacklevel=2,
        )

    policies_spec = spec.get("policies") or {}
    wranglers = ((spec.get("management") or {}).get("wranglers")) or {}

    return CabDef(
        name=name,
        command=spec["command"],
        info=spec.get("info"),
        image=image,
        flavour=flavour,
        policies=Policies(**policies_spec),
        inputs={k: _build_param(v) for k, v in (spec.get("inputs") or {}).items()},
        outputs={k: _build_param(v) for k, v in (spec.get("outputs") or {}).items()},
        wranglers=wranglers,
    )


def _build_param(spec: dict[str, Any] | None) -> ParamSchema:
    if spec is not None and not isinstance(spec, dict):
        raise CabLoadError(
            f"expected a param spec mapping, got {spec!r} -- this usually means an "
            "unsupported nested _include (e.g. 'inputs: {_include: (pkg)file}'), which "
            "shinobi doesn't resolve (see this module's docstring)"
        )
    spec = spec or {}
    return ParamSchema(
        dtype=str(spec.get("dtype", "str")),
        required=bool(spec.get("required", False)),
        default=spec.get("default"),
        info=spec.get("info"),
        implicit=spec.get("implicit"),
        nom_de_guerre=spec.get("nom_de_guerre"),
    )
