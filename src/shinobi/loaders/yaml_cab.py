"""Load YAML cab definitions in the scabha dialect into shinobi `Cab` objects.

**Lineage.** shinobi's cab schema is borrowed from scabha, the schema library
underneath Stimela 2.0, and the vocabulary here is deliberately scabha's:
`inputs`/`outputs` with `dtype`/`required`/`default`/`info`/`choices`,
`policies`, `management.wranglers`, `image`, `flavour`, `command`. Reusing it
was a design decision, not an accident of history -- the cab schema is the part
of stimela2 that got it right, and shinobi's own `Cab` mirrors it closely
enough that loading a scabha cab is a translation rather than an
interpretation. What shinobi drops is the layer *above* the cab: stimela2's
recipe, alias and expression machinery (see stimela-ninja's `AGENTS.md`).

cult-cargo is the largest published library of cabs written in this dialect and
is what this loader is usually pointed at, but the dialect is scabha's and
nothing here is specific to that project. `shinobi.loaders.worker_schema` reads
a scabha-derived *config* dialect through the same shared helpers.

**shinobi-native keys.** shinobi's own `Cab` carries a few things scabha has
no vocabulary for, so the dialect accepts them as an extension rather than
inventing a second format for cabs authored against shinobi directly. A
document using none of them is a plain scabha document, and cult-cargo's own
files remain a readable subset.

Per field, alongside the scabha keys: ``write_path: true`` marks a
string-typed input naming a filesystem path the tool writes to -- a stem
products are built from, or a complete path written directly (see
`ParamMeta.write_path`) -- and ``mutable: true`` marks an input the step may
change in place (`Mutability.MUTABLE`). Both are registered in
`_LEAF_SPEC_KEYS`, which matters more than it looks: `_is_section` tells a
leaf param from a nested CLI section by whether the mapping has *any* known
param-spec key, so a spec carrying only an unregistered key is read as a
section and the field disappears without a word.

Per cab: ``sandbox``, ``harvest`` and ``scratch``, which mirror the `Scope`
fields of the same names.

``image:`` may also name a *key* rather than a reference, resolved through the
caller-supplied ``images`` mapping (see `loads`) -- the same shape as
``package_roots``, and for the same reason: shinobi has no manifest and does
not go looking for one. It lets a document say ``image: WSCLEAN`` and leave
which reference that is to the deployment that loads it.

Also per cab: ``input_patterns``/``output_patterns``, families of
dynamically-named params (`ParamPattern`). A pattern is a ``separator`` plus
ordered ``segments``; each segment is either a ``regex`` (a level that cannot
be enumerated ahead of time) or ``attrs`` (the known level, each attr a param
spec in its own right). ``ParamPattern``'s own validator enforces the real
rule -- exactly one segment carries ``attrs`` -- so the loader only produces
the shape and lets it object, naming the cab and key when it does.

An attr spec is read by the same `_param_meta` as a declared field, with one
asymmetry: an attr keeps its ``dtype``, a field does not. A declared field's
dtype is already its model annotation, and repeating it on `field_meta` would
make every field of every cab differ from its Python-authored equivalent; a
pattern attr has no model field at all, which is the reason
`ParamMeta.dtype` exists.

**Support is deliberately partial.** This reads the static, declarative subset
and refuses the parts that are a programming language wearing YAML. The
boundary is drawn once, here and in SECURITY.md, and the sections below say
exactly where it falls: composition mechanisms this implements, then the
scabha features it does not.

Composition mechanisms, implemented in a deliberately minimal form -- real
scabha cab files are not self-contained and rely on stimela2's config system
for these:

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
  cab package for any reason (see SECURITY.md's "never eval()/exec() a cab's
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

Deliberately NOT implemented (this is the boundary -- see SECURITY.md). Each of
these is a place where scabha stops describing a tool and starts computing
something, which is the line shinobi does not cross in a cab:

* **Expressions and substitutions.** The ``=config.x.y`` / ``=recipe.ms`` /
  ``${...}`` / ``=IFSET(...)`` language scabha values can contain. Left as
  literal strings, so a cab carrying one loads with that value verbatim rather
  than resolved -- visible in the built `Cab`, not silently dropped.
  `ParamMeta.implicit` is the one templating shinobi does resolve, and it is
  plain ``str.format`` against the step's own validated inputs: no name
  resolution across steps, no function calls, no conditionals.

* **Conditionals and control flow.** Anything whose value depends on evaluating
  a predicate at load or run time. A cab is a parameter table; branching over
  it belongs in the Python that calls the step, where it is visible to the
  reader and to the DAG.

* **Aliases and propagation.** stimela2 propagates parameter values up and down
  between recipe and step level, which is what forces its expression language
  to exist. shinobi wires steps with typed `InputRef`/`OutputRef` objects
  instead, so there is nothing to propagate.
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
    contain_include,
    deep_merge,
    resolve_directive,
    resolve_package_root,
    resolve_use,
    sanitize_unique,
    validate_choices,
)
from shinobi.steps.schema import Cab, Mutability, ParamMeta, ParamPattern, ParamSegment, Policies


def load_file(
    path: str | Path,
    *,
    package_roots: dict[str, Path] | None = None,
    images: dict[str, str] | None = None,
) -> dict[str, Cab]:
    """Load a YAML cab definition file into `Cab` instances.

    Args:
        path: Path to the YAML cab definition file.
        package_roots: Mapping of package name to filesystem root, used to
            resolve `_include` directives that reference other packages.
        images: Mapping of image *key* to full reference. See `loads`.

    Returns:
        A dict mapping cab name to its built `Cab` instance.
    """
    path = Path(path)
    roots = package_roots or {}
    raw = _load_raw(path.resolve(), roots)
    resolved = resolve_use(raw, raw, error=CabLoadError)
    cabs_section = resolved.get("cabs", resolved)
    return {name: _build_cabdef(name, spec, roots, images or {}) for name, spec in cabs_section.items()}


def loads(
    text: str,
    *,
    package_roots: dict[str, Path] | None = None,
    images: dict[str, str] | None = None,
) -> dict[str, Cab]:
    """Parse cab defs from a YAML string. Supports ``_use`` (resolved
    against the document itself) and package-scoped ``_include`` (resolved
    against `package_roots`), but not a plain relative-path ``_include``,
    since there's no base directory to resolve a relative file path against.

    ``images`` maps an image *key* to its full reference, for a document that
    names images symbolically (``image: WSCLEAN``) rather than by a baked-in
    reference. Caller-supplied for the same reason ``package_roots`` is:
    shinobi has no manifest of its own and will not go looking for one. A cab
    repository passes its own -- dosho's `images.yaml` is exactly this -- so a
    deployment's overrides still decide the reference at load time instead of
    it being fixed when the document was written.

    An image string absent from the mapping is left alone, because a
    literal reference is the older and still-valid form (cult-cargo's files
    carry ``quay.io/stimela2/...`` directly). A key that is simply misspelled
    therefore reaches the runtime as an image name and fails there -- loudly,
    at pull time, which is the safe direction: the alternative rejects every
    legitimate bare name (``ubuntu``) to catch a typo.
    """
    roots = package_roots or {}
    raw = yaml.safe_load(text) or {}
    raw = resolve_directive(raw, "_include", lambda entry: _include_entry_to_dict(entry, None, roots))
    resolved = resolve_use(raw, raw, error=CabLoadError)
    cabs_section = resolved.get("cabs", resolved)
    return {name: _build_cabdef(name, spec, roots, images or {}) for name, spec in cabs_section.items()}


_PKG_INCLUDE_RE = re.compile(r"^\((?P<pkg>[\w.]+)\)(?P<rest>.*)$")


def _resolve_package_root(dotted: str, package_roots: dict[str, Path]) -> Path:
    """This dialect's `CabLoadError`-flavoured `resolve_package_root`. See
    that helper (and this module's docstring) for why `importlib` is never
    involved.
    """
    return resolve_package_root(dotted, package_roots, error=CabLoadError)


def _include_entry_to_dict(
    entry: Any,
    base_dir: Path | None,
    package_roots: dict[str, Path],
    containment_root: Path | None = None,
) -> dict[str, Any]:
    """One `_include` list entry -> its fully-loaded (and itself
    recursively `_include`-resolved) dict. Three real shapes:
    - plain relative path string (`"base.yml"`), only valid with a `base_dir`
    - combined package+path string (`"(cultcargo.genesis.cubical)schema.yaml"`)
    - package + file-list dict (`{"(cultcargo)": ["genesis/cult-cargo-base.yml"]}`)

    `containment_root` is the package root the enclosing include chain
    entered through (`None` at the top level, where a plain relative include
    is unconstrained). Every package-scoped hop sets it to its own package
    root and every file below that hop is checked against it -- see
    `_modelgen.contain_include`.
    """
    if isinstance(entry, str):
        if m := _PKG_INCLUDE_RE.match(entry):
            if not m.group("rest"):
                raise CabLoadError(f"package-scoped _include {entry!r} has no filename")
            pkg_dir = _resolve_package_root(m.group("pkg"), package_roots)
            target = contain_include(pkg_dir / m.group("rest"), pkg_dir, entry=entry, error=CabLoadError)
            return _load_raw(target, package_roots, pkg_dir)
        if base_dir is None:
            raise CabLoadError(f"relative-path _include {entry!r} has no base directory to resolve against (loads() only supports package-scoped _include entries)")
        target = base_dir / entry
        if containment_root is not None:
            target = contain_include(target, containment_root, entry=entry, error=CabLoadError)
        return _load_raw(target.resolve(), package_roots, containment_root)
    if isinstance(entry, dict) and len(entry) == 1:
        ((key, files),) = entry.items()
        if (m := _PKG_INCLUDE_RE.match(key)) and not m.group("rest"):
            pkg_dir = _resolve_package_root(m.group("pkg"), package_roots)
            merged: dict[str, Any] = {}
            for f in files if isinstance(files, list) else [files]:
                target = contain_include(pkg_dir / f, pkg_dir, entry=f, error=CabLoadError)
                merged = deep_merge(merged, _load_raw(target, package_roots, pkg_dir))
            return merged
    raise CabLoadError(f"unsupported _include entry {entry!r}")


def _load_raw(path: Path, package_roots: dict[str, Path], containment_root: Path | None = None) -> dict[str, Any]:
    """Read, parse, and recursively `_include`-resolve one file. Cached
    (keyed on the resolved path, `package_roots`, and the active
    `containment_root`, all three of which change how the file's own nested
    includes resolve) for the same reason as
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
    return _load_raw_cached(path, tuple(sorted(package_roots.items())), containment_root)


@functools.lru_cache(maxsize=None)
def _load_raw_cached(path: Path, roots_key: tuple[tuple[str, Path], ...], containment_root: Path | None) -> dict[str, Any]:
    package_roots = dict(roots_key)
    data = yaml.safe_load(path.read_text()) or {}
    return resolve_directive(
        data,
        "_include",
        lambda entry: _include_entry_to_dict(entry, path.parent, package_roots, containment_root),
    )


# shinobi-native per-field keys. They must be here as well as read in
# `_collect`: `_is_section` decides leaf-vs-section by whether a mapping has
# *any* known param-spec key, so a spec carrying only a new key would
# otherwise be mistaken for a nested CLI section and vanish.
_SHINOBI_LEAF_KEYS = {"write_path", "mutable"}

_LEAF_SPEC_KEYS = COMMON_LEAF_KEYS | {"nom_de_guerre", "mkdir", "element_choices"} | _SHINOBI_LEAF_KEYS


def _build_cabdef(name: str, spec: dict[str, Any], package_roots: dict[str, Path], images: dict[str, str] | None = None) -> Cab:
    image = spec.get("image")
    if isinstance(image, dict):
        image = image.get("name")
    # A symbolic key resolves through the caller's mapping; anything else is
    # already a reference (see `loads`).
    if images and isinstance(image, str):
        image = images.get(image, image)

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

    in_fields, field_meta, input_mutability = _collect(spec.get("inputs") or {})
    out_fields, out_meta, _out_mutability = _collect(spec.get("outputs") or {})

    in_choices = {field: meta.choices for field, meta in field_meta.items() if meta.choices}
    out_choices = {field: meta.choices for field, meta in out_meta.items() if meta.choices}

    # `abbreviation` is a CLI-only alias -- carried onto the field's
    # json_schema_extra so `clickutil.build_options` can emit a `-<abbrev>`
    # short flag. Only meaningful on inputs (outputs aren't CLI options).
    in_extras = {field: {"abbreviation": meta.abbreviation} for field, meta in field_meta.items() if meta.abbreviation}

    return Cab(
        name=name,
        command=spec["command"],
        info=spec.get("info"),
        image=image,
        flavour=flavour,
        policies=Policies(**policies_spec),
        inputs_model=build_model(f"{name}_Inputs", in_fields, choices=in_choices, extras=in_extras),
        outputs_model=build_model(f"{name}_Outputs", out_fields, choices=out_choices),
        # Output metas merged over input ones, the same way
        # `dosho._builder.define_cab` composes them, so a cab built from a
        # document and the same cab built in Python agree. Without the output
        # half an `implicit` output template is silently dropped: nothing
        # resolves the output's value, and `declared_output_dirs` finds no
        # write directory to mount, which is how a tool's products end up
        # inside the container. The merge replaces whole `ParamMeta` objects,
        # so a name declared on both sides keeps the output's -- a sharp edge
        # inherited deliberately rather than diverging from dosho here.
        field_meta={**field_meta, **out_meta},
        wranglers=wranglers,
        input_mutability=input_mutability,
        input_patterns=_param_patterns(spec.get("input_patterns"), cab=name, key="input_patterns"),
        output_patterns=_param_patterns(spec.get("output_patterns"), cab=name, key="output_patterns"),
        sandbox=spec.get("sandbox"),
        harvest=list(spec.get("harvest") or []),
        scratch=list(spec.get("scratch") or []),
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


def _param_patterns(raw: Any, *, cab: str, key: str) -> list[ParamPattern]:
    """Read `input_patterns:`/`output_patterns:` into `ParamPattern`s.

    A pattern is a `separator` plus an ordered list of `segments`; each segment
    is either a `regex` (a level that cannot be enumerated) or `attrs` (the
    known level, each attr a param spec in its own right). `ParamPattern`'s own
    validator enforces the real rule -- exactly one segment carries `attrs` --
    so this only has to produce the shape and let it complain.

    Errors name the cab and the key, because a pattern is the one part of a cab
    a reader cannot check by eye against the tool's `--help`.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise CabLoadError(f"cab '{cab}': '{key}' must be a list of patterns, got {type(raw).__name__}")
    patterns: list[ParamPattern] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise CabLoadError(f"cab '{cab}': '{key}[{i}]' must be a mapping, got {type(entry).__name__}")
        segments_raw = entry.get("segments")
        if not isinstance(segments_raw, list) or not segments_raw:
            raise CabLoadError(f"cab '{cab}': '{key}[{i}]' needs a non-empty 'segments' list")
        segments: list[ParamSegment] = []
        for j, seg in enumerate(segments_raw):
            if not isinstance(seg, dict):
                raise CabLoadError(f"cab '{cab}': '{key}[{i}].segments[{j}]' must be a mapping, got {type(seg).__name__}")
            attrs_raw = seg.get("attrs")
            if attrs_raw is None:
                segments.append(ParamSegment(regex=seg.get("regex")))
                continue
            if not isinstance(attrs_raw, dict):
                raise CabLoadError(f"cab '{cab}': '{key}[{i}].segments[{j}].attrs' must be a mapping, got {type(attrs_raw).__name__}")
            # An attr always gets a ParamMeta, even an empty one. Unlike a
            # declared field, where an all-default meta carries no information
            # and is dropped, the *set of attr names* is what the pattern
            # matches on -- dropping an empty one would delete the attr.
            attrs = {name: _param_meta(spec or {}, with_dtype=True) for name, spec in attrs_raw.items()}
            segments.append(ParamSegment(attrs=attrs))
        try:
            patterns.append(ParamPattern(separator=entry.get("separator", "."), segments=segments))
        except ValueError as exc:
            raise CabLoadError(f"cab '{cab}': '{key}[{i}]' is not a valid pattern -- {exc}") from exc
    return patterns


_DEFAULT_PARAM_META = ParamMeta()


def _param_meta(value: dict[str, Any], *, nom_de_guerre: str | None = None, with_dtype: bool = False) -> ParamMeta:
    """Build a `ParamMeta` from a param-spec mapping.

    Shared by declared fields and by `ParamPattern` attrs, which are the same
    shape -- an attr is a param spec that happens to name part of a pattern
    rather than a whole field. Keeping one reader means a key added for one is
    understood by the other, which is the drift `AGENTS.md` warns about for
    these loaders.

    `with_dtype` is the one asymmetry, and it is not cosmetic. A declared
    field's dtype lives in its model annotation, so repeating it here would
    put something on `field_meta` that the Python-authored equivalent does not
    have -- every field of every cab would then differ on a round trip. A
    pattern attr has no model field at all, which is exactly why
    `ParamMeta.dtype` exists (see its docstring): it is the only way a backend
    can tell a dynamically-named input is file-like.
    """
    policies = value.get("policies") or {}
    return ParamMeta(
        nom_de_guerre=nom_de_guerre,
        implicit=value.get("implicit"),
        info=value.get("info"),
        positional=bool(policies.get("positional", False)),
        positional_head=bool(policies.get("positional_head", False)),
        repeat_as_tokens=policies.get("repeat") == "list",
        choices=validate_choices(value.get("choices"), error=CabLoadError),
        dtype=value.get("dtype") if with_dtype else None,
        write_path=bool(value.get("write_path", False)),
        abbreviation=value.get("abbreviation"),
    )


def _collect(
    raw: dict[str, Any],
    *,
    _prefix: str = "",
    _seen: dict[str, str] | None = None,
) -> tuple[dict[str, tuple[str, bool, Any]], dict[str, ParamMeta], dict[str, Mutability]]:
    """Split a cult-cargo inputs/outputs mapping into modelgen field specs
    and per-field ParamMeta (nom_de_guerre/implicit/info/positional/
    repeat_as_tokens). Recurses into stimela2-style CLI-section nesting
    (`data: {ms: {...}}`), flattening into dotted field names (`data.ms`).
    """
    fields: dict[str, tuple[str, bool, Any]] = {}
    metas: dict[str, ParamMeta] = {}
    mutability: dict[str, Mutability] = {}
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
            sub_fields, sub_metas, sub_mut = _collect(value, _prefix=dotted_key, _seen=seen)
            fields.update(sub_fields)
            metas.update(sub_metas)
            mutability.update(sub_mut)
            continue
        field = sanitize_unique(dotted_key, seen)
        implicit = value.get("implicit")
        required = bool(value.get("required", False)) and implicit is None
        fields[field] = (str(value.get("dtype", "str")), required, value.get("default"))
        # the tool's real flag name: an explicit nom_de_guerre, else the
        # original (unsanitised) param name if sanitising changed it.
        nom = value.get("nom_de_guerre") or (dotted_key if dotted_key != field else None)
        if value.get("mutable"):
            mutability[field] = Mutability.MUTABLE
        meta = _param_meta(value, nom_de_guerre=nom)
        # Only carry a meta that says something. Compared against the default
        # rather than testing each attribute: the old form was a long boolean
        # chain that had to be edited every time `ParamMeta` gained a field,
        # and was one edit behind more than once.
        if meta != _DEFAULT_PARAM_META:
            metas[field] = meta
    return fields, metas, mutability
