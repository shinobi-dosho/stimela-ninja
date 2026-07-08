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
  implemented. Instead, `_DYNAMIC_INPUT_PATTERNS`/`_DYNAMIC_OUTPUT_PATTERNS`
  below give a small, explicit, per-cab table of *static* `ParamPattern`
  catch-alls for the two real dynamic shapes cult-cargo actually has
  (per-solver-term input families for cubical/quartical; a permissive
  dynamic-output-name catch-all for wsclean) -- built by reading the cab's
  own static *data* files (e.g. cubical's ``schema_JONES_TEMPLATE.yaml``)
  as plain YAML, never by importing/calling the cab's ``dynamic_schema``
  function. A cab using ``dynamic_schema`` that isn't in this table loads
  with a warning and whatever static ``inputs:``/``outputs:`` are present,
  same as before -- silently incomplete unless you notice the warning.

Building the expression language out, or actually executing a cab's own
``dynamic_schema``, would mean re-deriving stimela2's config engine (or
its code-execution trust model) -- exactly what this project exists to
avoid unless a real cab actually needs it.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import yaml

from shinobi.exceptions import CabLoadError
from shinobi.loaders._modelgen import (
    build_model,
    deep_merge,
    get_path,
    resolve_directive,
    sanitize_unique,
)
from shinobi.steps.schema import Cab, ParamMeta, ParamPattern, ParamSegment, Policies


def load_file(path: str | Path, *, package_roots: dict[str, Path] | None = None) -> dict[str, Cab]:
    path = Path(path)
    roots = package_roots or {}
    raw = _load_raw(path, roots)
    resolved = _resolve_use(raw, raw)
    cabs_section = resolved.get("cabs", resolved)
    dynamic_inputs = _dynamic_input_patterns(roots)
    return {name: _build_cabdef(name, spec, roots, dynamic_inputs) for name, spec in cabs_section.items()}


def loads(text: str, *, package_roots: dict[str, Path] | None = None) -> dict[str, Cab]:
    """Parse cab defs from a YAML string. Supports ``_use`` (resolved
    against the document itself) and package-scoped ``_include`` (resolved
    against `package_roots`), but not a plain relative-path ``_include``,
    since there's no base directory to resolve a relative file path against.
    """
    roots = package_roots or {}
    raw = yaml.safe_load(text) or {}
    raw = resolve_directive(raw, "_include", lambda entry: _include_entry_to_dict(entry, None, roots))
    resolved = _resolve_use(raw, raw)
    cabs_section = resolved.get("cabs", resolved)
    dynamic_inputs = _dynamic_input_patterns(roots)
    return {name: _build_cabdef(name, spec, roots, dynamic_inputs) for name, spec in cabs_section.items()}


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
    data = yaml.safe_load(path.read_text()) or {}
    return resolve_directive(
        data, "_include", lambda entry: _include_entry_to_dict(entry, path.parent, package_roots)
    )


def _resolve_use(node: Any, root: dict[str, Any]) -> Any:
    def entry_to_dict(dotted: str) -> Any:
        # recurse so a `_use` target that itself has a `_use` resolves too
        return resolve_directive(get_path(root, dotted, error=CabLoadError), "_use", entry_to_dict)

    return resolve_directive(node, "_use", entry_to_dict)


# -- static ParamPattern catch-alls for the real dynamic_schema cabs --------

# Closed-world assumption, deliberately: `_is_section` (below) treats a
# dict as a leaf param spec if it uses *any* of these keys, and as a
# section (to recurse/flatten) otherwise. Every real cult-cargo leaf spec
# key seen in this project's own vendored cab files is listed -- but a
# leaf spec using only some *other*, not-yet-seen key (and no key from
# this set) would be misclassified as a section, since there's no
# positive "this dict is definitely a leaf" signal, only the absence of
# a known negative one. Extend this set (never remove from it) if a real
# cab hits this.
_LEAF_SPEC_KEYS = {
    "info", "dtype", "required", "default", "implicit", "nom_de_guerre",
    "policies", "choices", "must_exist", "mkdir", "writable",
    "path_policies", "element_choices",
}


def _load_template_attrs(dotted_pkg: str, filename: str, top_key: str, package_roots: dict[str, Path]) -> dict[str, ParamMeta]:
    """Load a dynamic_schema cab's real per-term attrs template as static
    YAML *data* (never importing/calling the cab's own dynamic_schema
    Python function) and turn it into `{attr_name: ParamMeta}`. `top_key`
    is the one wrapping key the real template file uses (cubical's
    `JONES_TEMPLATE:`, quartical's `gain:`).

    Carries over every `ParamMeta` field the template's own per-attr spec
    can express -- `info`/`dtype` directly, `nom_de_guerre`/`implicit`
    directly, `positional`/`repeat_as_tokens` from a nested `policies:`
    block -- the same extraction `_collect` does for ordinary static
    fields below. `required`/`default`/`choices` (real keys template specs
    do carry, e.g. quartical's `gain.type.choices`) are deliberately not
    forwarded: `ParamMeta` has no fields for them at all, because a
    pattern-matched dynamic attr never becomes a declared pydantic model
    field the way a static one does (see `ParamPattern`'s own docstring)
    -- there's no per-field "required"/"default" slot to populate for an
    attr that's only ever validated by regex/dtype at match time, not by
    the model itself. This table is a soft validation catch-all, not a
    complete re-derivation of the tool's real schema (see this module's
    own top-level docstring on `dynamic_schema`).
    """
    pkg_dir = _resolve_package_root(dotted_pkg, package_roots)
    try:
        text = (pkg_dir / filename).read_text()
    except OSError as exc:
        raise CabLoadError(f"cannot read dynamic_schema template {pkg_dir / filename}: {exc}") from exc
    doc = yaml.safe_load(text) or {}
    attrs = doc.get(top_key) or {}
    result: dict[str, ParamMeta] = {}
    for name, spec in attrs.items():
        spec = spec or {}
        param_policies = spec.get("policies") or {}
        result[name] = ParamMeta(
            info=spec.get("info"),
            dtype=spec.get("dtype"),
            nom_de_guerre=spec.get("nom_de_guerre"),
            implicit=spec.get("implicit"),
            positional=bool(param_policies.get("positional", False)),
            repeat_as_tokens=param_policies.get("repeat") == "list",
        )
    return result


def _dynamic_input_patterns(package_roots: dict[str, Path]) -> dict[str, list[ParamPattern]]:
    """cab name -> extra `input_patterns` to attach after normal static
    loading, for the two real per-solver-term dynamic_schema cabs. Explicit
    per-cab, not structural inference: cubical and quartical don't share a
    clean structural signal to detect this shape from automatically, and
    only two real examples exist.

    A pattern's `separator="."` matches the cab's *internal* dotted field
    convention (`g1.solvable`, matching every other loader-generated field,
    e.g. `data.ms` -> field `data_ms`/`nom_de_guerre="data.ms"`) -- the
    cab's own `policies.replace: {'.': '-'}` then turns it into the real
    `--g1-solvable` CLI flag at `build_argv` time, exactly as it already
    does for every static field. No special-casing needed here.

    Gracefully returns no entry for a cab whose template file isn't
    resolvable (e.g. `package_roots` wasn't supplied) -- the cab still
    loads with just its static fields, as if this table didn't exist.

    Called once per `load_file`/`loads` (not once per cab defined in the
    loaded document) and the result threaded through to `_build_cabdef` --
    both real template files are only ever read from disk once per load
    call, however many cabs the document defines, rather than once per
    cab (which would have re-read *both* cubical's and quartical's
    template files for every cab in a multi-cab file, regardless of
    whether that cab even used `dynamic_schema`).
    """
    patterns: dict[str, list[ParamPattern]] = {}
    sources = {
        "cubical": ("cultcargo.genesis.cubical", "schema_JONES_TEMPLATE.yaml", "JONES_TEMPLATE"),
        "quartical": ("cultcargo.genesis.quartical", "gain_schema.yaml", "gain"),
    }
    for cab_name, (pkg, filename, top_key) in sources.items():
        try:
            attrs = _load_template_attrs(pkg, filename, top_key, package_roots)
        except CabLoadError:
            continue
        if not attrs:
            continue
        patterns[cab_name] = [
            ParamPattern(separator=".", segments=[ParamSegment(regex=r".+?"), ParamSegment(attrs=attrs)])
        ]
    return patterns


# wsclean's real dynamic output names are `<enumerable-imagetype>.<qualifiers>`
# (e.g. `dirty.per-band`, `restored.i.per-interval.mfs`) -- the imagetype is
# enumerable, the qualifier tail is open-ended/combinatorial (built from
# several resolved params: nchan/pol/intervals-out/niter/make-psf/...). This
# is validation-only (see task scope): it lets `recipe.outputs(step, name)`
# accept a real dynamic name without raising, but does not resolve `implicit`
# glob/placeholder templates into real file paths -- that stays unbuilt.
#
# Hand-transcribed (not read from cult-cargo at load time -- would mean
# importing cab package code, the exact thing this loader avoids) from
# cult-cargo's `cultcargo/genesis/wsclean/__init__.py`'s
# `make_stimela_schema()`: `imagetypes` only ever collects
# "psf"/"dirty"/"restored"/"residual"/"model" (the `outputs[f"{imagetype}...]`
# dict-key prefix -- this table's keys). "image" is *not* a second valid
# key prefix -- it's `img_output()`'s own real filename component for the
# "restored" key (`imagetype == "restored": imagetype = "image"`, then
# `{prefix}-image.fits`): "restored" is wsclean's nom_de_guerre-style
# alias for what the tool actually names "image" on disk, the same
# declared-name-vs-real-name split `ParamMeta.nom_de_guerre` exists for
# elsewhere in this loader. If cult-cargo's own `imagetypes` list ever
# changes, this table silently drifts out of sync (no runtime
# cross-check against the real package).
_WSCLEAN_IMAGETYPES: dict[str, ParamMeta] = {
    "dirty": ParamMeta(dtype="File", info="wsclean dynamic output (validation only)"),
    "restored": ParamMeta(
        dtype="File", info="wsclean dynamic output (validation only)", nom_de_guerre="image"
    ),
    "residual": ParamMeta(dtype="File", info="wsclean dynamic output (validation only)"),
    "model": ParamMeta(dtype="File", info="wsclean dynamic output (validation only)"),
    "psf": ParamMeta(dtype="File", info="wsclean dynamic output (validation only)"),
}


def _dynamic_output_patterns() -> dict[str, list[ParamPattern]]:
    return {
        "wsclean": [
            ParamPattern(
                separator=".",
                segments=[ParamSegment(attrs=_WSCLEAN_IMAGETYPES), ParamSegment(regex=r".+")],
            )
        ]
    }


def _build_cabdef(
    name: str, spec: dict[str, Any], package_roots: dict[str, Path], dynamic_inputs: dict[str, list[ParamPattern]]
) -> Cab:
    image = spec.get("image")
    if isinstance(image, dict):
        image = image.get("name")

    flavour = spec.get("flavour", "binary")
    if isinstance(flavour, dict):
        flavour = flavour.get("kind", "binary")

    if "command" not in spec:
        raise CabLoadError(f"cab '{name}' has no 'command' (check its _use references)")

    # dynamic_inputs is computed once per load_file()/loads() call and
    # passed in, not recomputed here -- see _dynamic_input_patterns's own
    # docstring for why (avoids re-reading cubical's/quartical's template
    # files from disk once per cab in a multi-cab document).
    extra_input_patterns = dynamic_inputs.get(name, [])
    extra_output_patterns = _dynamic_output_patterns().get(name, [])

    if spec.get("dynamic_schema") and not extra_input_patterns and not extra_output_patterns:
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

    in_fields, field_meta = _collect(spec.get("inputs") or {})
    out_fields, _ = _collect(spec.get("outputs") or {})

    return Cab(
        name=name,
        command=spec["command"],
        info=spec.get("info"),
        image=image,
        flavour=flavour,
        policies=Policies(**policies_spec),
        inputs_model=build_model(f"{name}_Inputs", in_fields, allow_extra=bool(extra_input_patterns)),
        outputs_model=build_model(f"{name}_Outputs", out_fields),
        field_meta=field_meta,
        input_patterns=extra_input_patterns,
        output_patterns=extra_output_patterns,
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

    Closed-world heuristic (see `_LEAF_SPEC_KEYS`'s own comment): this can
    misclassify a leaf spec as a section if it uses only param-spec keys
    this project hasn't seen yet. A misclassified leaf would recurse into
    `_collect` and get flattened as if it were a nested section instead
    of being treated as one field -- the fix, when that happens, is
    adding the missing key(s) to `_LEAF_SPEC_KEYS`, not rewriting this
    function.
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
        repeat_as_tokens = param_policies.get("repeat") == "list"
        if nom or implicit is not None or value.get("info") or positional or repeat_as_tokens:
            metas[field] = ParamMeta(
                nom_de_guerre=nom,
                implicit=implicit,
                info=value.get("info"),
                positional=positional,
                repeat_as_tokens=repeat_as_tokens,
            )
    return fields, metas
