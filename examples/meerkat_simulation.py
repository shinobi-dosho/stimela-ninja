"""MeerKAT simulation -- a shinobi Recipe reimplementing the old
stimela-classic `meerkat_simulation.py` example on the step model
(`Cab`/`Recipe`/`add_step`, `ninja run`).

This is a genuinely *runnable* pipeline, not just a schema demo: make an
empty MS -> simulate visibilities from a sky model -> calibrate -> image
3x with different Briggs robust weightings. Every tool involved is real
and was exercised directly against this file while writing it.

Tool choices, differing from the original script:

* The old stimela-1.x `cab/simms` + the MeqTrees-based `cab/simulator` are
  replaced by a single tool, **simms 3.0** (https://github.com/wits-cfa/simms):
  `simms telsim` creates the empty MS, `simms skysim` simulates a sky model
  into it. Its cab schema is loaded from the vendored, authoritative
  `examples/simms/simms-cabs.yaml` (see `examples/simms/README.md`) via
  `shinobi.loaders.cultcargo` -- it's genuine cult-cargo-format YAML and is
  also the real source the `simms` CLI itself is generated from, so it
  can't drift out of sync the way a hand-declared cab could.
* `cab/calibrator` (also MeqTrees-based) is replaced by **cubical**
  (hand-declared here, same as `ninja_selfcal.py`'s `cubical` -- real
  docker image `quay.io/stimela2/cubical:latest`), calibrating against the
  same sky model used to simulate (a standard smoke-test pattern -- not
  scientifically meaningful, but structurally correct).
* `cab/casa_listobs`/`cab/casa_rmtables` are dropped entirely: both are
  CASA tasks, and shinobi deliberately never executes a non-"binary"
  flavour cab (`UnsupportedFlavourError` -- see AGENTS.md, "Never
  eval()/exec() a cab's command"). An MS-info listing and MS teardown
  aren't needed for a smoke test; see `ninja_selfcal.py`'s docstring for
  the same exclusion pattern applied to its own CASA cabs.
* I/O is plain path/string Recipe inputs wired via `InputRef`/`OutputRef`
  -- *not* stimela-classic's `indir`/`outdir`/`msdir` directory-staging
  convention.

Two real shinobi infra gaps surfaced while wiring this up for real (both
fixed in `src/shinobi/policies.py`/`src/shinobi/steps/schema.py`/
`src/shinobi/loaders/cultcargo.py`, and both already present -- just
previously unimplemented -- in real cult-cargo cab files this project
already vendors):

* `command: simms telsim` is a two-word subcommand invocation, not a
  single executable name -- `build_argv` now splits `cab.command` on
  whitespace.
* Both simms cabs' `ms` input, and wsclean's `ms`/`size`/`weight` inputs,
  use per-parameter `policies: {positional: true}` / `{repeat: list}` --
  a positional CLI arg, and a list value emitted as separate bare argv
  tokens after one flag (`-size 4096 4096`, not `-size 4096,4096`).
  `ParamMeta.positional`/`ParamMeta.repeat_as_tokens` + the cultcargo
  loader + `build_argv` now support both.

Run it:

    ninja run examples/meerkat_simulation.py:recipe --dryrun

A real run needs `simms` installed (it has no docker image yet):

    uv sync --group examples
    ninja run examples/meerkat_simulation.py:recipe --ms sim.ms
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from shinobi import Cab, Recipe
from shinobi.loaders import build_model, sanitize_unique
from shinobi.loaders.cultcargo import load_file
from shinobi.steps.schema import ParamMeta

_CULTCARGO_DIR = Path(__file__).parent / "cultcargo"
_SIMMS_DIR = Path(__file__).parent / "simms"


def _infer_dtype(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, (list, tuple)):
        return "list:str"
    return "str"


def cab_from_defaults(
    name: str,
    command: str,
    image: str,
    defaults: dict[str, Any],
    *,
    extra: dict[str, tuple[str, bool, Any]] | None = None,
    outputs: dict[str, tuple[str, bool, Any]] | None = None,
) -> Cab:
    """Build a Cab whose inputs_model comes from a `defaults` dict, instead
    of typing every parameter by hand. Hyphenated tool parameter names are
    sanitised to valid pydantic field names, with the original kept as a
    nom_de_guerre. Copied from `ninja_selfcal.py` -- each example stays
    self-contained rather than importing from another.
    """
    fields: dict[str, tuple[str, bool, Any]] = {}
    field_meta: dict[str, ParamMeta] = {}
    seen: dict[str, str] = {}
    for raw_name, value in defaults.items():
        field = sanitize_unique(raw_name, seen)
        fields[field] = (_infer_dtype(value), False, value)
        if field != raw_name:
            field_meta[field] = ParamMeta(nom_de_guerre=raw_name)
    for raw_name, spec in (extra or {}).items():
        field = sanitize_unique(raw_name, seen)
        fields[field] = spec
        if field != raw_name:
            field_meta[field] = ParamMeta(nom_de_guerre=raw_name)
    return Cab(
        name=name,
        command=command,
        image=image,
        inputs_model=build_model(f"{name}_Inputs", fields),
        outputs_model=build_model(f"{name}_Outputs", outputs or {}),
        field_meta=field_meta,
    )


# simms 3.0: real, authoritative schema (examples/simms/, vendored). Both
# cabs load with no `outputs:` at all (real upstream file), so each gets a
# `.model_copy` override adding a passthrough `ms` output -- same name as
# the (positional) `ms` input, so dispatch's same-named-input fallback
# carries the produced/updated MS path through with no wrangler needed.
# Neither has a docker image yet, so both run via NativeBackend.
_simms_cabs = load_file(_SIMMS_DIR / "simms-cabs.yaml")
_SIMMS_MS_OUTPUT = build_model("simms_Outputs", {"ms": ("MS", False, None)})
telsim = _simms_cabs["telsim"].model_copy(update={"outputs_model": _SIMMS_MS_OUTPUT, "backend": "native"})
skysim = _simms_cabs["skysim"].model_copy(update={"outputs_model": _SIMMS_MS_OUTPUT, "backend": "native"})

# Real cult-cargo schema (examples/cultcargo/, vendored, shared with
# ninja_selfcal.py). Its `inputs:` resolve fully via `_use`/`_include`
# (both implemented by the loader) even though the cab also references an
# unresolved `dynamic_schema` -- so `field_meta` (positional/repeat_as_tokens/
# nom_de_guerre) for every field below is genuine, loader-parsed metadata,
# not hand-typed. `inputs_model` still needs narrowing to what this recipe
# uses: real wsclean dtypes like `Union[int, Tuple[int, int]]` (`size`) or
# `List[MS]` (`ms`) use bracket syntax `_modelgen.dtype_to_type` doesn't
# parse (only its own `list:<inner>` form) -- generalising that would mean
# reimplementing stimela2's whole type-string grammar, well beyond what
# this recipe needs, so the handful of fields actually wired below are
# rebuilt with concrete dtypes shinobi does understand, keeping their real
# field_meta from the load. `outputs_model` needs an override outright:
# wsclean's real "implicit outputs" (dirty/restored/residual/model,
# per-band/per-interval) are structured by the same unresolved
# dynamic_schema, so there's no static `restored` field to inherit.
_wsclean_loaded = load_file(_CULTCARGO_DIR / "wsclean.yml")["wsclean"]
_WSCLEAN_FIELDS: dict[str, tuple[str, bool, Any]] = {
    "ms": ("MS", True, None),
    "column": ("str", False, None),
    "prefix": ("str", True, None),
    "size": ("list:int", True, None),
    "scale": ("str", True, None),
    "niter": ("int", False, None),
    "mgain": ("float", False, None),
    "pol": ("str", False, None),
    "multiscale": ("bool", False, None),
    "multiscale_scales": ("list:int", False, None),
    "weight": ("list:str", False, None),
}
wsclean = _wsclean_loaded.model_copy(
    update={
        "inputs_model": build_model("wsclean_Inputs", _WSCLEAN_FIELDS),
        "outputs_model": build_model("wsclean_Outputs", {"restored": ("File", False, None)}),
        "field_meta": {
            name: _wsclean_loaded.field_meta[name]
            for name in _WSCLEAN_FIELDS
            if name in _wsclean_loaded.field_meta
        },
    }
)

# cubical: hand-declared, same shape as ninja_selfcal.py's cubical, but
# with a `data_ms` passthrough output (not `corrected_ms`) -- cubical
# calibrates *in place* (writes CORRECTED_DATA into the same MS `data-ms`
# already points at), so the calibrated MS path *is* the `data-ms` path;
# naming the output the same lets imaging wire a genuine MS via the same
# same-named-input fallback used for telsim/skysim's `ms`.
cal_opts: dict[str, Any] = {
    "data-column": "DATA",
    "out-mode": "sc",
    "weight-column": "WEIGHT_SPECTRUM",
    "out-overwrite": True,
    "madmax-enable": True,
    "madmax-threshold": 7.0,
}
cubical = cab_from_defaults(
    "cubical",
    "gocubical",
    "quay.io/stimela2/cubical:latest",
    cal_opts,
    extra={
        "data-ms": ("MS", True, None),
        "out-name": ("str", True, None),
        "model-list": ("str", True, None),
    },
    outputs={"data_ms": ("MS", False, None)},
)


# ---------------------------------------------------------------- recipe ---


class SimInputs(BaseModel):
    ms: str = "meerkat_simulation.ms"
    telescope: str = "meerkat"
    skymodel: str = str(_SIMMS_DIR / "testsky.txt")
    prefix: str = "meerkat-sim"


class SimOutputs(BaseModel):
    image: str | None = None


def build_simulation(robust_values: tuple[int, ...] = (2, 0, -2)) -> Recipe:
    """Construct the simulation Recipe: make_ms -> simulate -> calibrate ->
    one wsclean image per entry in `robust_values` (matching the original
    script's `briggs_robust = [2, 0, -2]` loop).
    """
    recipe = Recipe(name="meerkat_simulation", inputs_model=SimInputs, outputs_model=SimOutputs)

    recipe.add_step(
        "make_ms",
        telsim,
        ms=recipe.inputs.ms,
        telescope=recipe.inputs.telescope,
        dtime=30.0,
        ntime=4,
        startfreq="750MHz",
        dfreq="1MHz",
        nchan=16,
    )
    recipe.add_step(
        "simulate",
        skysim,
        ms=recipe.outputs("make_ms", "ms"),
        ascii_sky=recipe.inputs.skymodel,
        column="DATA",
        sefd=831.0,
    )
    recipe.add_step(
        "calibrate",
        cubical,
        data_ms=recipe.outputs("simulate", "ms"),
        out_name=recipe.inputs.prefix,
        model_list="MODEL_DATA",
    )

    image_steps: list[str] = []
    for robust in robust_values:
        name = f"image_robust_{str(robust).replace('-', 'm')}"
        recipe.add_step(
            name,
            wsclean,
            ms=recipe.outputs("calibrate", "data_ms"),
            column="CORRECTED_DATA",
            weight=["briggs", str(robust)],
            prefix=f"meerkat-sim-robust{robust}",
            size=[4096, 4096],
            scale="2asec",
            niter=5000,
            mgain=0.85,
            pol="I",
            multiscale=True,
            multiscale_scales=[0, 2],
        )
        image_steps.append(name)

    recipe.set_output("image", recipe.outputs(image_steps[0], "restored"))
    return recipe


# The default target `ninja run` resolves.
recipe = build_simulation()


if __name__ == "__main__":
    from shinobi.dag import graph_nodes, render_dag

    print(render_dag(graph_nodes(recipe)))
