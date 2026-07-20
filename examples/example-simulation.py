"""End-to-End simulation -- a shinobi Recipe reimplementing the old
stimela-classic `example_simulation.py` example on the step model
(`Cab`/`Recipe`/`add_step`, `ninja run`).

This is a genuinely *runnable* pipeline, not just a schema demo: make an
empty MS -> simulate visibilities from a sky model -> image the simulated
data (to populate MODEL_DATA via deconvolution, since simms itself only
writes DATA + SEFD-derived noise, not a model column CubiCal can
calibrate against) -> calibrate -> image 3x with different Briggs robust
weightings. Every tool involved is real and was exercised directly
against this file while writing it.

Every cab here comes from **dosho** (https://github.com/SpheMakh/dosho),
the native shinobi cab repository -- not hand-loaded YAML, not a
hand-declared `Cab`. This is the intended way to use cabs now; see
`ninja_selfcal.py` for the older `shinobi.loaders.cultcargo`/hand-declared
path this example used to take (kept as a second example of that path,
not because it's still recommended).

Tool choices, differing from the original stimela-classic script:

* The old stimela-1.x `cab/simms` + the MeqTrees-based `cab/simulator` are
  replaced by a single tool, **simms 3.0** (https://github.com/wits-cfa/simms):
  `simms telsim` creates the empty MS, `simms skysim` simulates a sky model
  into it (`dosho.cabs.simms`).
* `cab/calibrator` (also MeqTrees-based) is replaced by **cubical**
  (`dosho.cabs.cubical` -- the real 135-parameter schema, docker image
  `quay.io/stimela2/cubical:...`), calibrating against the same sky model
  used to simulate (a standard smoke-test pattern -- not scientifically
  meaningful, but structurally correct). `sol_jones=["G"]` plus the
  pattern-matched `G-solvable`/`G-type` kwargs exercise
  `dosho.cabs.cubical`'s real per-Jones-term `ParamPattern`.
* `cab/casa_listobs`/`cab/casa_rmtables` are dropped entirely: both are
  CASA tasks, and shinobi deliberately never executes a non-"binary"
  flavour `Cab` (`UnsupportedFlavourError` -- see SECURITY.md, "Never
  eval()/exec() a cab's command"). An MS-info listing and MS teardown
  aren't needed for a smoke test. (dosho does have a real `listobs`
  pystep, `dosho.cabs.casatasks.listobs` -- a `StepRef`, not a `Cab` --
  left out here since it's genuinely optional for this pipeline, not
  because it's unavailable.)
* I/O is plain path/string Recipe inputs wired via `InputRef`/`OutputRef`
  -- *not* stimela-classic's `indir`/`outdir`/`msdir` directory-staging
  convention.

Run it:

    ninja run examples/example-simulation.py:recipe --dryrun

A real run needs `simms` installed (it has no docker image yet):

    uv sync --group examples
    ninja run examples/example-simulation.py:recipe --ms sim.ms
"""

from __future__ import annotations

from pathlib import Path

from dosho.cabs import cubical, wsclean
from dosho.cabs.simms import skysim, telsim
from pydantic import BaseModel, create_model

from shinobi import Recipe
from shinobi.loaders import build_model
from shinobi.steps.schema import ParamMeta

_INPUT_DIR = Path(__file__).parent / "input-dir"

# Neither real simms cab declares any outputs (matches cult-cargo's own
# schema -- a genuine gap in the tool's cab metadata, not a dosho
# omission), but this recipe needs to wire the MS each one produces into
# the next step -- add a same-named-as-input `ms` passthrough output
# locally, the same caller-side pattern used elsewhere for cabs with a
# real output-schema gap (e.g. caracal2's wsclean `input_ms` echo).
# Neither cab has a docker image yet either, so both run via
# NativeBackend regardless of the caller's own default backend.
_SIMMS_MS_OUTPUT = build_model("simms_Outputs", {"ms": ("MS", False, None)})
telsim = telsim.model_copy(update={"outputs_model": _SIMMS_MS_OUTPUT, "backend": "native"})
skysim = skysim.model_copy(update={"outputs_model": _SIMMS_MS_OUTPUT, "backend": "native"})

# wsclean images *in place* -- deconvolution writes the clean model back
# into the same MS's MODEL_DATA column, but dosho's real wsclean outputs
# are FITS image products (`image`/`image_mfs`/...), not the MS itself,
# so there's no real output to depend on "this imaging step has finished
# updating MODEL_DATA" from. Add an `input_ms` field to *outputs_model
# only* (not inputs_model!) -- adding it as an input too (the shape
# caracal2's own line/selfcal workers use for this same gap) would make
# build_argv emit it as a real (bogus) `-input-ms <path>` CLI flag, since
# it iterates every declared *input* field with a resolved value and
# real wsclean has no such option (the same class of bug just fixed for
# cubical's booleans, caught by actually running this against real
# wsclean). Instead, resolve it the same way `image`/`image_mfs` already
# are: an `implicit` template, `"{ms[0]}"` indexing into wsclean's own
# real (list) `ms` input -- this recipe always wires exactly one MS into
# it. Only `wsclean_with_model` (the pre-calibration imaging step) needs
# this; the final per-robust imaging steps below use the plain `wsclean`
# cab and only need its real `image` output.
wsclean_with_model = wsclean.model_copy(
    update={
        "outputs_model": create_model("wsclean_sim_Outputs", __base__=wsclean.outputs_model, input_ms=(Path | None, None)),
        "field_meta": {**wsclean.field_meta, "input_ms": ParamMeta(implicit="{ms[0]}")},
    }
)


class SimInputs(BaseModel):
    ms: str = "example-simulation.ms"
    telescope: str = "meerkat"
    skymodel: str = str(_INPUT_DIR / "testsky.txt")
    prefix: str = "example-sim"


class SimOutputs(BaseModel):
    # dosho's real wsclean cab resolves `image` via an implicit
    # `{prefix}-image.fits` template (see dosho/cabs/wsclean.py) -- a real
    # Path, not the always-empty placeholder the old hand-narrowed wsclean
    # cab's `restored` output was (it had no way to ever get populated).
    image: Path | None = None


def build_simulation(robust_values: tuple[int, ...] = (2, 0, -2)) -> Recipe:
    """Construct the simulation Recipe: make_ms -> simulate -> calibrate ->
    one wsclean image per entry in `robust_values` (matching the original
    script's `briggs_robust = [2, 0, -2]` loop).
    """
    recipe = Recipe(name="example_simulation", inputs_model=SimInputs, outputs_model=SimOutputs)

    recipe.add_step(
        "make_ms",
        telsim,
        ms=recipe.inputs.ms,
        telescope=recipe.inputs.telescope,
        direction="J2000,0h0m0s,-30d8m0s",
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
        "image_sim",
        wsclean_with_model,
        ms=[recipe.outputs("simulate", "ms")],
        column="DATA",
        weight=("briggs", 1.5),
        prefix="example-simdata",
        size=(1024, 1024),
        scale="4asec",
        niter=5000,
        mgain=0.9,
        pol="I",
    )

    recipe.add_step(
        "calibrate",
        cubical,
        data_ms=recipe.outputs("image_sim", "input_ms"),
        out_name=recipe.inputs.prefix,
        model_list="MODEL_DATA",
        data_column="DATA",
        out_mode="sc",
        weight_column="WEIGHT_SPECTRUM",
        out_overwrite=True,
        madmax_enable="True",  # dosho's cubical port has no explicit
        madmax_threshold="7.0",  # dtype for these -- both are real str fields
        sol_jones=["G"],
        # sol-jones' *value* uses the term's own label case ("G"), but
        # real CubiCal's per-term CLI flags are always lowercase
        # regardless -- confirmed against caracal 1.x's real production
        # usage (selfcal_worker.py/ddcal_worker.py: "g-solvable", never
        # "G-solvable"). shinobi's ParamPattern match is case-agnostic,
        # so "G-solvable" would build fine but gocubical itself rejects it.
        **{"g-solvable": True, "g-type": "complex-2x2"},
    )

    image_steps: list[str] = []
    for robust in robust_values:
        name = f"image_robust_{str(robust).replace('-', 'm')}"
        recipe.add_step(
            name,
            wsclean,
            ms=[recipe.outputs("calibrate", "ms")],
            column="CORRECTED_DATA",
            weight=("briggs", float(robust)),
            prefix=f"example-sim-robust{robust}",
            size=(2048, 2048),
            scale="2asec",
            niter=1000,
            mgain=0.9,
        )
        image_steps.append(name)

    recipe.set_output("image", recipe.outputs(image_steps[0], "image"))
    return recipe


# The default target `ninja run` resolves.
recipe = build_simulation()


if __name__ == "__main__":
    from shinobi.dag import graph_nodes, render_dag

    print(render_dag(graph_nodes(recipe)))
