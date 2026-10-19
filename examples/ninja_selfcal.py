"""Two rounds of image, mask, and calibration in a declared Recipe.

This standalone loader example uses vendored cult-cargo and Stimela classic
schemas, plus hand-declared Breizorro and CubiCal cabs. The WSClean output is
a stub, so use ``ninja run examples/ninja_selfcal.py:selfcal --dryrun``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from shinobi import Cab, InputRef, OutputRef, Recipe
from shinobi.loaders import build_model, sanitize_unique
from shinobi.loaders.yaml_cab import load_file
from shinobi.loaders.stimela_classic import load_file as load_classic_cab
from shinobi.steps.schema import ParamMeta, ParamPattern, ParamSegment

_CULTCARGO_DIR = Path(__file__).parent / "cultcargo"
_STIMELA_CLASSIC_DIR = Path(__file__).parent / "stimela_classic"


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
    input_patterns: list[ParamPattern] | None = None,
    extra: dict[str, tuple[str, bool, Any]] | None = None,
    outputs: dict[str, tuple[str, bool, Any]] | None = None,
) -> Cab:
    """Build a cab from defaults, preserving hyphenated CLI names."""
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
        inputs_model=build_model(f"{name}_Inputs", fields, allow_extra=bool(input_patterns)),
        outputs_model=build_model(f"{name}_Outputs", outputs or {}),
        field_meta=field_meta,
        input_patterns=input_patterns or [],
    )


# Vendored Stimela classic schemas.
casa_mstransform = load_classic_cab(_STIMELA_CLASSIC_DIR / "casa_mstransform" / "parameters.json")
casa_listobs = load_classic_cab(_STIMELA_CLASSIC_DIR / "casa_listobs" / "parameters.json")
casa_flagdata = load_classic_cab(_STIMELA_CLASSIC_DIR / "casa_flagdata" / "parameters.json")
casa_flagmanager = load_classic_cab(_STIMELA_CLASSIC_DIR / "casa_flagmanager" / "parameters.json")
msutils = load_classic_cab(_STIMELA_CLASSIC_DIR / "msutils" / "parameters.json")


breizorro = Cab(
    name="breizorro",
    command="breizorro",
    image="breizorro:latest",
    inputs_model=build_model(
        "breizorro_Inputs",
        {
            "restored_image": ("File", True, None),
            "boxsize": ("int", False, None),
            "threshold": ("float", False, None),
            "outfile": ("File", False, None),
        },
    ),
    outputs_model=build_model("breizorro_Outputs", {"mask": ("File", False, None)}),
    field_meta={"restored_image": ParamMeta(nom_de_guerre="restored-image")},
)


# Calibration and imaging defaults from the original recipe.
cal_opts: dict[str, Any] = {
    "data-column": "DATA",
    "out-mode": "sc",
    "weight-column": "WEIGHT_SPECTRUM",
    "out-overwrite": True,
    "madmax-enable": True,
    "madmax-threshold": 7.0,
}

im_opts: dict[str, Any] = {
    "niter": 400000,
    "size": 6000,
    "scale": 1.3,
    "auto-threshold": 3,
    "nchan": 5,
    "multiscale": True,
    "multiscale-scales": [0, 1, 3, 5],
    "mgain": 0.8,
    "column": "CORRECTED_DATA",
}

cubical = cab_from_defaults(
    "cubical",
    "gocubical",
    "quay.io/stimela2/cubical:latest",
    cal_opts,
    input_patterns=[
        # Jones term names are selected per call.
        ParamPattern(
            separator="-",
            segments=[
                ParamSegment(regex=r".+?"),
                ParamSegment(
                    attrs={
                        "solvable": ParamMeta(),
                        "type": ParamMeta(),
                        "time-int": ParamMeta(),
                        "freq-int": ParamMeta(),
                    }
                ),
            ],
        )
    ],
    extra={
        "data-ms": ("MS", True, None),
        "out-name": ("str", True, None),
        "model-list": ("str", True, None),
        "sol-jones": ("str", False, None),
    },
    outputs={"corrected_ms": ("MS", False, None)},
)

# WSClean's dynamic schema is not loaded; declare only the wired fields here.
wsclean = load_file(_CULTCARGO_DIR / "wsclean.yml")["wsclean"].model_copy(
    update={
        "inputs_model": build_model("wsclean_Inputs", {"ms": ("MS", True, None), "prefix": ("str", False, None)}),
        "outputs_model": build_model("wsclean_Outputs", {"restored": ("File", False, None)}),
    }
)


class SelfcalInputs(BaseModel):
    ms: str = "line.ms"
    prefix: str = "selfcal"
    rounds: int = 2


class SelfcalOutputs(BaseModel):
    final_ms: str | None = None


def build_selfcal(rounds: int = 2) -> Recipe:
    """Build an image, mask, calibrate chain for each round."""
    recipe = Recipe(name="selfcal", inputs_model=SelfcalInputs, outputs_model=SelfcalOutputs)

    prev_ms: InputRef | OutputRef = recipe.inputs.ms
    for n in range(1, rounds + 1):
        image = f"image{n}"
        mask = f"mask{n}"
        cal = f"cal{n}"
        recipe.add_step(image, wsclean, ms=prev_ms, prefix=f"round{n}")
        recipe.add_step(mask, breizorro, restored_image=recipe.outputs(image, "restored"))
        recipe.add_step(
            cal,
            cubical,
            data_ms=prev_ms if n == 1 else recipe.outputs(f"cal{n - 1}", "corrected_ms"),
            out_name=f"round{n}",
            model_list=f"round{n}-model",
        )
        prev_ms = recipe.outputs(cal, "corrected_ms")

    recipe.set_output("final_ms", prev_ms)
    return recipe


selfcal = build_selfcal(rounds=2)


if __name__ == "__main__":
    from shinobi.dag import graph_nodes, render_dag

    print(render_dag(graph_nodes(selfcal)))
