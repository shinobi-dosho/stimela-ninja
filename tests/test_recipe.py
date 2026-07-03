import pytest

from shinobi.exceptions import CabRunError
from shinobi.recipe import call
from shinobi.schema import CabDef, ParamSchema


def test_call_chains_outputs_as_plain_python_values(native):
    """The whole point of the recipe layer: no string substitution, no
    aliasing -- a step's output is just a Python value passed to the next
    call().
    """
    make_image = CabDef(
        name="make-image",
        command="/bin/echo",
        inputs={"text": ParamSchema(dtype="str", default="IMAGE: out.fits")},
        wranglers={r"IMAGE: (?P<image>\S+)": ["PARSE_OUTPUT:image:str"]},
    )
    use_image = CabDef(
        name="use-image",
        command="/bin/echo",
        inputs={"path": ParamSchema(dtype="str", required=True)},
    )

    image_result = call(make_image, native)
    assert image_result.image == "out.fits"

    final_result = call(use_image, native, path=image_result.image)
    assert "out.fits" in final_result.stdout


def test_call_raises_by_default_on_failure(native):
    cab = CabDef(name="fail", command="/bin/false")
    with pytest.raises(CabRunError):
        call(cab, native)


def test_call_check_false_suppresses_raise(native):
    cab = CabDef(name="fail", command="/bin/false")
    result = call(cab, native, check=False)
    assert not result.success
