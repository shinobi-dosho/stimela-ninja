"""A minimal @shinobi.step example: two free-standing decorated steps over
loaded cabs, plus a tiny two-step Recipe wiring one into the next.

Run a single step:      ninja run examples/simple_selfcal.py:image --ms obs.ms
Dry-run the recipe:     ninja run examples/simple_selfcal.py:selfcal --dryrun
"""

from __future__ import annotations

from pydantic import BaseModel

from shinobi import Cab, Recipe, step
from shinobi.loaders._modelgen import build_model


class ImageInputs(BaseModel):
    ms: str = "obs.ms"
    prefix: str = "img"


class ImageOutputs(BaseModel):
    restored: str | None = None


wsclean = Cab(
    name="wsclean",
    command="wsclean",
    image="quay.io/stimela/wsclean:latest",
    inputs_model=ImageInputs,
    outputs_model=ImageOutputs,
)

breizorro = Cab(
    name="breizorro",
    command="breizorro",
    image="breizorro:latest",
    inputs_model=build_model("MaskInputs", {"restored_image": ("File", True, None)}),
    outputs_model=build_model("MaskOutputs", {"mask": ("File", False, None)}),
)


@step(wsclean, backend="native")
def image(ctx):
    """Image the visibilities. A near-empty body auto-runs the cab."""
    return ctx.run()


@step(breizorro, backend="native")
def make_mask(ctx):
    return ctx.run()


# A two-step recipe: image, then mask the image it produced.
selfcal = Recipe(name="selfcal", inputs_model=ImageInputs, outputs_model=build_model("Out", {"mask": ("File", False, None)}))
selfcal.add_step("image", wsclean, ms=selfcal.inputs.ms, prefix=selfcal.inputs.prefix)
selfcal.add_step("mask", breizorro, restored_image=selfcal.outputs.image.restored)
selfcal.set_output("mask", selfcal.outputs.mask.mask)
