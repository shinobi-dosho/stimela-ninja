"""Load cult-cargo style YAML cab definitions into shinobi Cab objects.

The cult-cargo cab *schema* (inputs/outputs/policies/wranglers) is a good
design and is reused as-is here -- it's stimela2's recipe/alias layer that
shinobi drops, not this. This loader lets shinobi use the existing library
of cult-cargo tool wrappers without anyone having to rewrite them.

Real cult-cargo cab files, however, are not self-contained: they rely on
composition mechanisms from stimela2's config system, which this loader
implements a deliberately minimal version of:

* ``_include: [file, ...]`` -- merges other YAML files in (relative to the
  including file), most often to pull in a shared ``vars:``/``lib:``
  namespace. Merging is a plain deep-merge; the including file's own keys
  win over included ones. Resolved wherever it appears in the document
  (top level, or nested under ``inputs:``/``outputs:``, as real cult-cargo's
  ``cubical.yml``/``quartical.yml`` do) via the same tree-walking
  ``resolve_directive`` helper ``_use`` already relies on.

* ``_use: dotted.path`` -- deep-merges a dict looked up by dotted path in
  the fully-merged document (post-``_include``) into the dict it appears
  in, with that dict's own sibling keys taking precedence. Used both for
  small things (``image: {_use: vars.cult-cargo.images, name: breizorro}``)
  and to inherit a cab's entire command/flavour block.

* The package-scoped include form (``_include: (pkg.dotted.path)file.yaml``
  or ``_include: [{(pkg.dotted.path): [file, ...]}]``) -- searches an
  installed package's data directory rather than a relative path. Resolving
  a dotted package name to a filesystem directory would normally mean
  importing the package (``importlib``), but that risks executing arbitrary
  code from *any* ``__init__.py`` on the path -- shinobi never imports a
  cab package for any reason (see AGENTS.md's "never eval()/exec() a cab's
  command" boundary, which this extends to "never import a cab package").
  Instead, callers pass ``package_roots={"cultcargo": Path(...)}`` to
  ``load_file()``/``loads()``: an explicit, caller-supplied mapping from a
  dotted package prefix to its filesystem directory. A dotted name is
  resolved against the *longest* registered prefix, descending the
  remainder as subdirectories (``cultcargo.genesis.cubical`` against
  ``{"cultcargo": Path("/.../cultcargo")}`` -> ``Path("/.../cultcargo/genesis/cubical")``)
  -- the normal package/subpackage-is-a-subdirectory convention, without
  ever asking Python's import machinery to confirm it. A package-scoped
  ``_include`` naming a package with no registered root raises a clear
  ``CabLoadError``.

Deliberately NOT implemented (this is the boundary -- see AGENTS.md):

* The ``=config.x.y``/``${...}`` expression language cult-cargo values
  can contain -- left as literal strings.
* ``dynamic_schema: dotted.path`` -- a reference to a Python function that
  would need importing and *calling* to get a cab's real schema (real
  cult-cargo's ``wsclean.yml``/``cubical.yml``/``quartical.yml`` use this).
  Resolving it for real is not just a parsing gap like the above: it means
  executing arbitrary code named by a cab file at load time. Not
  implemented, and not worked around here either: a cab using
  ``dynamic_schema`` always loads with a warning and whatever static
  ``inputs:``/``outputs:`` are present -- silently incomplete unless you
  notice the warning. The hand-authored, cross-checked static schemas for
  the three real cabs that need this (wsclean, cubical, quartical) live in
  ``dosho`` (the native shinobi cab repository, a sibling project) instead
  of as a stopgap table in this loader -- this loader used to carry one
  (a small per-cab ``ParamPattern`` table read from each cab's own static
  *data* files, e.g. cubical's ``schema_JONES_TEMPLATE.yaml``), removed
  once dosho's real ports superseded it. See ``dosho/cabs/wsclean.py``/
  ``cubical.py``/``quartical.py`` for that knowledge now, and prefer
  porting a cab there over reintroducing a table here.

Building the expression language out, or actually executing a cab's own
``dynamic_schema``, would mean re-deriving stimela2's config engine (or
its code-execution trust model) -- exactly what this project exists to
avoid unless a real cab actually needs it.
"""

from __future__ import annotations

import functools
import re
import warnings
from pathlib import Path
from typing import Any

import yaml

from shinobi.exceptions import CabLoadError
from shinobi.loaders._modelgen import (
    COMMON_LEAF_KEYS,
    build_model,
    deep_merge,
    resolve_directive,
    resolve_use,
    sanitize_unique,
    validate_choices,
)
from shinobi.steps.schema import Cab, ParamMeta, Policies


def load_file(path: str | Path, *, package_roots: dict[str, Path] | None = None) -> dict[str, Cab]:
    """Load a cult-cargo cab definition file into `Cab` instances.

    Args:
        path: Path to the YAML cab definition file.
        package_roots: Mapping of package name to filesystem root, used to
            resolve `_include` directives that reference other packages.

    Returns:
        A dict mapping cab name to its built `Cab` instance.
    """
    path = Path(path)
    roots = package_roots or {}
    raw = _load_raw(path.resolve(), roots)
    resolved = resolve_use(raw, raw, error=CabLoadError)
    cabs_section = resolved.get("cabs", resolved)
    return {name: _build_cabdef(name, spec, roots) for name, spec in cabs_section.items()}


def loads(text: str, *, package_roots: dict[str, Path] | None = None) -> dict[str, Cab]:
    """Parse cab defs from a YAML string. Supports ``_use`` (resolved
    against the document itself) and package-scoped ``_include`` (resolved
    against `package_roots`), but not a plain relative-path ``_include``,
    since there's no base directory to resolve a relative file path against.
    """
    roots = package_roots or {}
    raw = yaml.safe_load(text) or {}
    raw = resolve_directive(raw, "_include", lambda entry: _include_entry_to_dict(entry, None, roots))
    resolved = resolve_use(raw, raw, error=CabLoadError)
    cabs_section = resolved.get("cabs", resolved)
    return {name: _build_cabdef(name, spec, roots) for name, spec in cabs_section.items()}


_PKG_INCLUDE_RE = re.compile(r"^\((?P<pkg>[\w.]+)\)(?P<rest>.*)$")


def _resolve_package_root(dotted: str, package_roots: dict[str, Path]) -> Path:
    """`dotted` (e.g. `cultcargo.genesis.cubical`) -> filesystem directory,
    resolved against the *longest* registered prefix in `package_roots`
    (descending the remainder as subdirectories) -- never via `importlib`.
    See this module's docstring for why.
    """
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in package_roots:
            return package_roots[prefix].joinpath(*parts[i:])
    raise CabLoadError(
        f"package-scoped _include references package {dotted!r}, but no filesystem "
        f"root was supplied for it (or a parent package of it) -- pass "
        f"package_roots={{{parts[0]!r}: Path(...)}} to load_file()/loads() "
        "(shinobi never imports a package to resolve this -- see this module's docstring)"
    )


def _include_entry_to_dict(entry: Any, base_dir: Path | None, package_roots: dict[str, Path]) -> dict[str, Any]:
    """One `_include` list entry -> its fully-loaded (and itself
    recursively `_include`-resolved) dict. Three real shapes:
    - plain relative path string (`"base.yml"`), only valid with a `base_dir`
    - combined package+path string (`"(cultcargo.genesis.cubical)schema.yaml"`)
    - package + file-list dict (`{"(cultcargo)": ["genesis/cult-cargo-base.yml"]}`)
    """
    if isinstance(entry, str):
        if m := _PKG_INCLUDE_RE.match(entry):
            if not m.group("rest"):
                raise CabLoadError(f"package-scoped _include {entry!r} has no filename")
            pkg_dir = _resolve_package_root(m.group("pkg"), package_roots)
            return _load_raw((pkg_dir / m.group("rest")).resolve(), package_roots)
        if base_dir is None:
            raise CabLoadError(
                f"relative-path _include {entry!r} has no base directory to resolve "
                "against (loads() only supports package-scoped _include entries)"
            )
        return _load_raw((base_dir / entry).resolve(), package_roots)
    if isinstance(entry, dict) and len(entry) == 1:
        ((key, files),) = entry.items()
        if (m := _PKG_INCLUDE_RE.match(key)) and not m.group("rest"):
            pkg_dir = _resolve_package_root(m.group("pkg"), package_roots)
            merged: dict[str, Any] = {}
            for f in files if isinstance(files, list) else [files]:
                merged = deep_merge(merged, _load_raw((pkg_dir / f).resolve(), package_roots))
            return merged
    raise CabLoadError(f"unsupported _include entry {entry!r}")


def _load_raw(path: Path, package_roots: dict[str, Path]) -> dict[str, Any]:
    """Read, parse, and recursively `_include`-resolve one file. Cached
    (keyed on the resolved path and `package_roots`) for the same reason as
    `worker_schema._load_include_file`: a cab library commonly has many
    files `_include`-ing the same shared base (cult-cargo's own
    `cult-cargo-base.yml`/`vars` files) or `_use`-ing each other, so without
    this every referencing file re-reads and re-parses it from disk. Safe
    to cache: `resolve_directive`/`deep_merge` never mutate their inputs, so
    the same returned dict can be reused (and further deep_merged from,
    which always builds a new dict) by every caller. `package_roots` is
    turned into a hashable, order-independent key since a plain dict can't
    be an `lru_cache` argument directly.
    """
    return _load_raw_cached(path, tuple(sorted(package_roots.items())))


@functools.lru_cache(maxsize=None)
def _load_raw_cached(path: Path, roots_key: tuple[tuple[str, Path], ...]) -> dict[str, Any]:
    package_roots = dict(roots_key)
    data = yaml.safe_load(path.read_text()) or {}
    return resolve_directive(
        data, "_include", lambda entry: _include_entry_to_dict(entry, path.parent, package_roots)
    )


_LEAF_SPEC_KEYS = COMMON_LEAF_KEYS | {"nom_de_guerre", "mkdir", "element_choices"}


def _build_cabdef(name: str, spec: dict[str, Any], package_roots: dict[str, Path]) -> Cab:
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
            "relative to the tool's actual interface. Check whether dosho (the native "
            "shinobi cab repository) already has a real port of this cab.",
            stacklevel=2,
        )

    policies_spec = spec.get("policies") or {}
    wranglers = ((spec.get("management") or {}).get("wranglers")) or {}

    in_fields, field_meta = _collect(spec.get("inputs") or {})
    out_fields, out_meta = _collect(spec.get("outputs") or {})

    in_choices = {field: meta.choices for field, meta in field_meta.items() if meta.choices}
    out_choices = {field: meta.choices for field, meta in out_meta.items() if meta.choices}

    return Cab(
        name=name,
        command=spec["command"],
        info=spec.get("info"),
        image=image,
        flavour=flavour,
        policies=Policies(**policies_spec),
        inputs_model=build_model(f"{name}_Inputs", in_fields, choices=in_choices),
        outputs_model=build_model(f"{name}_Outputs", out_fields, choices=out_choices),
        field_meta=field_meta,
        wranglers=wranglers,
    )


def _is_section(value: dict) -> bool:
    """A non-empty dict under `inputs:`/`outputs:` is a stimela2-style
    section (to be flattened into dotted `section.param` field names, e.g.
    cubical's `data: {ms: {...}, column: {...}}` -> `data.ms`/`data.column`)
    rather than a leaf param spec, when none of its own top-level keys look
    like a known param-spec key. An empty dict is always a (minimal) leaf
    spec, never an empty section -- this preserves the existing bare `key:`
    (implicit `{}`) leaf convention.
    """
    return bool(value) and not (set(value) & _LEAF_SPEC_KEYS)


def _collect(
    raw: dict[str, Any],
    *,
    _prefix: str = "",
    _seen: dict[str, str] | None = None,
) -> tuple[dict[str, tuple[str, bool, Any]], dict[str, ParamMeta]]:
    """Split a cult-cargo inputs/outputs mapping into modelgen field specs
    and per-field ParamMeta (nom_de_guerre/implicit/info/positional/
    repeat_as_tokens). Recurses into stimela2-style CLI-section nesting
    (`data: {ms: {...}}`), flattening into dotted field names (`data.ms`).
    """
    fields: dict[str, tuple[str, bool, Any]] = {}
    metas: dict[str, ParamMeta] = {}
    seen = _seen if _seen is not None else {}
    for key, value in raw.items():
        if value is not None and not isinstance(value, dict):
            raise CabLoadError(
                f"expected a param spec mapping, got {value!r} -- this usually means an "
                "unsupported nested _include, which shinobi doesn't resolve without a "
                "package_roots entry (see this module's docstring)"
            )
        value = value or {}
        dotted_key = f"{_prefix}.{key}" if _prefix else key
        if _is_section(value):
            sub_fields, sub_metas = _collect(value, _prefix=dotted_key, _seen=seen)
            fields.update(sub_fields)
            metas.update(sub_metas)
            continue
        field = sanitize_unique(dotted_key, seen)
        implicit = value.get("implicit")
        required = bool(value.get("required", False)) and implicit is None
        fields[field] = (str(value.get("dtype", "str")), required, value.get("default"))
        # the tool's real flag name: an explicit nom_de_guerre, else the
        # original (unsanitised) param name if sanitising changed it.
        nom = value.get("nom_de_guerre") or (dotted_key if dotted_key != field else None)
        param_policies = value.get("policies") or {}
        positional = bool(param_policies.get("positional", False))
        positional_head = bool(param_policies.get("positional_head", False))
        repeat_as_tokens = param_policies.get("repeat") == "list"
        choices = validate_choices(value.get("choices"), error=CabLoadError)
        if (
            nom
            or implicit is not None
            or value.get("info")
            or positional
            or positional_head
            or repeat_as_tokens
            or choices
        ):
            metas[field] = ParamMeta(
                nom_de_guerre=nom,
                implicit=implicit,
                info=value.get("info"),
                positional=positional,
                positional_head=positional_head,
                repeat_as_tokens=repeat_as_tokens,
                choices=choices,
            )
    return fields, metas
