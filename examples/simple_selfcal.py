"""A minimal @shinobi.step example: two free-standing decorated steps over
loaded cabs, plus a tiny two-step Recipe wiring one into the next.

Run a single step:      ninja run examples/simple_selfcal.py:image --ms obs.ms
Dry-run the recipe:     ninja run examples/simple_selfcal.py:selfcal --dryrun
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from shinobi import Cab, Recipe, step
from shinobi.loaders import build_model
from shinobi.steps import ParamMeta


class ImageInputs(BaseModel):
    ms: Path = Path("obs.ms")
    prefix: str = "img"


class PipelineInputs(ImageInputs):
    mask: Path = Path("mask.fits")


class ImageOutputs(BaseModel):
    restored: Path | None = None


wsclean = Cab(
    name="wsclean",
    command="wsclean",
    image="quay.io/stimela/wsclean:latest",
    inputs_model=ImageInputs,
    outputs_model=ImageOutputs,
    field_meta={"restored": ParamMeta(implicit="{prefix}-MFS-image.fits")},
)

breizorro = Cab(
    name="breizorro",
    command="breizorro",
    image="breizorro:latest",
    inputs_model=build_model(
        "MaskInputs",
        {
            "restored_image": ("File", True, None),
            "outfile": ("File", True, None),
        },
    ),
    outputs_model=build_model("MaskOutputs", {"mask": ("File", True, None)}),
    field_meta={
        "restored_image": ParamMeta(nom_de_guerre="restored-image"),
        "mask": ParamMeta(implicit="{outfile}"),
    },
)


@step(wsclean, backend="native")
def image(ctx):
    """Image the visibilities."""
    return ctx.run()


@step(breizorro, backend="native")
def make_mask(ctx):
    return ctx.run()


# A two-step recipe: image, then mask the image it produced.
selfcal = Recipe(
    name="selfcal",
    inputs_model=PipelineInputs,
    outputs_model=build_model("Out", {"mask": ("File", False, None)}),
)
selfcal.add_step("image", wsclean, ms=selfcal.inputs.ms, prefix=selfcal.inputs.prefix)
selfcal.add_step(
    "mask",
    breizorro,
    restored_image=selfcal.outputs.image.restored,
    outfile=selfcal.inputs.mask,
)
selfcal.set_output("mask", selfcal.outputs.mask.mask)
