"""String validation survives the closed model and recipe bundle codecs."""

import re
from typing import Annotated

import pytest
import yaml
from pydantic import Field, StringConstraints, ValidationError, create_model

from shinobi.config import AppConfig
from shinobi.loaders import yaml_cab
from shinobi.loaders._modelgen import build_model
from shinobi.loaders.worker_schema import load_worker_schema
from shinobi.offload._codec import BundleError, Constraint, ModelSpec, pack
from shinobi.offload.bundle import RecipeBundle, freeze_recipe
from shinobi.steps.schema import InputRef, OutputRef, Recipe, StepRef

_PATTERN = r"\A(?!.*None).+\Z"


def _round_trip(model):
    return ModelSpec.model_validate_json(ModelSpec.capture(model).model_dump_json()).restore()


def test_generated_model_patterns_round_trip_required_optional_choices_and_defaults():
    model = build_model(
        "Constrained",
        {"required": ("str", True, None), "optional": ("str", False, None), "defaulted": ("str", False, "None"), "choice": ("str", False, None)},
        choices={"choice": ["ok", "None"]},
        string_patterns=dict.fromkeys(("required", "optional", "defaulted", "choice"), _PATTERN),
    )
    restored = _round_trip(model)
    assert restored.model_json_schema() == model.model_json_schema()
    assert restored(required="solutions", defaulted="ok").optional is None
    for values in (
        {"required": "None", "defaulted": "ok"},
        {"required": "ok", "optional": "None", "defaulted": "ok"},
        {"required": "ok"},
        {"required": "ok", "defaulted": "ok", "choice": "None"},
        {"required": "ok", "defaulted": "ok", "choice": "other"},
    ):
        with pytest.raises(ValidationError):
            restored(**values)


def test_constrained_cab_can_be_frozen_and_restored(tmp_path):
    field = {"dtype": "str", "string_pattern": _PATTERN}
    cab = yaml_cab.loads(yaml.safe_dump({"cabs": {"solver": {"command": "solver", "inputs": {"sols": {**field, "required": True}}, "outputs": {"result": field}}}}))["solver"]
    recipe = Recipe(
        name="pipeline",
        inputs_model=cab.inputs_model,
        outputs_model=cab.outputs_model,
        steps=[StepRef(name="solve", step=cab, wiring={"sols": InputRef(field="sols")})],
        output_wiring={"result": OutputRef(step="solve", field="result")},
    )
    bundle = freeze_recipe(recipe, {"sols": "solutions"}, config=AppConfig(), workspace=tmp_path)
    restored = RecipeBundle.model_validate_json(bundle.model_dump_json()).declaration().steps[0].step
    assert restored.field_meta == cab.field_meta
    assert restored.inputs_model(sols="solutions").sols == "solutions"
    assert restored.outputs_model().result is None
    for model, values in ((restored.inputs_model, {"sols": "None"}), (restored.outputs_model, {"result": "None"})):
        with pytest.raises(ValidationError, match="pattern"):
            model(**values)


def test_worker_schema_pattern_round_trip(tmp_path):
    path = tmp_path / "worker.yaml"
    path.write_text(yaml.safe_dump({"name": "worker", "inputs": {"sols": {"dtype": "str", "string_pattern": _PATTERN}}}))
    restored = _round_trip(load_worker_schema(path).inputs_model)
    assert restored().sols is None
    with pytest.raises(ValidationError, match="pattern"):
        restored(sols="None")


@pytest.mark.parametrize("optional", [False, True])
def test_compiled_regex_preserves_python_engine_and_flags(optional):
    pattern = re.compile(r"\A(?!none$)[a-z]+\Z", re.IGNORECASE | re.ASCII)
    annotation = Annotated[str, StringConstraints(pattern=pattern)]
    model = create_model("PythonRegex", value=(annotation | None if optional else annotation, None if optional else ...))
    restored = _round_trip(model)
    assert restored(value="SOLUTIONS").value == "SOLUTIONS"
    for value in ("NONE", "é"):
        with pytest.raises(ValidationError, match="pattern"):
            restored(value=value)
    if optional:
        assert restored().value is None


def test_plain_pattern_keeps_string_representation():
    model = create_model("PlainRegex", value=(str, Field(pattern="^ok$")))
    spec = ModelSpec.capture(model)
    assert spec.fields[0].constraints[0].attributes["flags"] == pack(None)
    restored = _round_trip(model)
    assert restored(value="ok").value == "ok"
    with pytest.raises(ValidationError, match="pattern"):
        restored(value="bad")


@pytest.mark.parametrize(
    "attributes",
    [
        {"pattern": pack("^ok$")},
        {"pattern": pack("^ok$"), "flags": pack(None), "unknown": pack(None)},
        {"pattern": pack(3), "flags": pack(None)},
        {"pattern": pack("ok"), "flags": pack(True)},
        {"pattern": pack("ok"), "flags": pack(-1)},
        {"pattern": pack("["), "flags": pack(0)},
        {"pattern": pack("ok"), "flags": pack(int(re.LOCALE))},
        {"pattern": pack("ok"), "flags": pack(1 << 100)},
    ],
)
def test_malformed_wire_pattern_is_rejected(attributes):
    with pytest.raises(BundleError, match="string pattern"):
        Constraint(name="StringPattern", attributes=attributes).restore()


def test_other_general_metadata_and_string_options_are_still_refused():
    for metadata in (Field(coerce_numbers_to_str=True).metadata[0], StringConstraints(pattern="ok", strip_whitespace=True)):
        with pytest.raises(BundleError, match="unsupported string constraint options"):
            Constraint.capture(metadata)
    with pytest.raises(BundleError, match="text regex"):
        Constraint.capture(StringConstraints(pattern=re.compile(b"ok")))
