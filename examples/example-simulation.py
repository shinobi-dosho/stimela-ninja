"""Simulate an MS, solve gains with QuartiCal, then image three weightings.

Run with ``ninja run examples/example-simulation.py:recipe --dryrun``.
Native simms steps require simms installed for a real run.
"""

from __future__ import annotations

from pathlib import Path

from dosho.cabs import quartical, wsclean
from dosho.cabs.simms import skysim, telsim
from pydantic import BaseModel, create_model

from shinobi import Recipe
from shinobi.loaders import build_model
from shinobi.steps.schema import ParamMeta

_INPUT_DIR = Path(__file__).parent / "input-dir"

# simms does not declare the MS it modifies as an output; expose it for wiring.
_SIMMS_MS_OUTPUT = build_model("simms_Outputs", {"ms": ("MS", False, None)})
telsim = telsim.model_copy(update={"outputs_model": _SIMMS_MS_OUTPUT, "backend": "native"})
skysim = skysim.model_copy(update={"outputs_model": _SIMMS_MS_OUTPUT, "backend": "native"})

# WSClean writes MODEL_DATA in place. An output-only MS reference orders calibration
# after imaging without emitting a nonexistent -input-ms CLI option.
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
    image: Path | None = None


def build_simulation(robust_values: tuple[int, ...] = (2, 0, -2)) -> Recipe:
    """Build the simulation and calibration recipe."""
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
        quartical,
        input_ms_path=recipe.outputs("image_sim", "input_ms"),
        input_model_recipe="MODEL_DATA",
        output_gain_directory=recipe.inputs.prefix,
        output_overwrite=True,
        output_products=["corrected_data"],
        output_columns=["CORRECTED_DATA"],
        solver_terms=["G"],
        **{"G.type": "diag_complex"},
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


recipe = build_simulation()


if __name__ == "__main__":
    from shinobi.dag import graph_nodes, render_dag

    print(render_dag(graph_nodes(recipe)))
