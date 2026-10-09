"""Both declarative schema loaders retain nullability in Pydantic and offload."""

import pytest
from pydantic import ValidationError

from shinobi.exceptions import CabLoadError, ConfigLoadError
from shinobi.loaders.worker_schema import load_worker_schema
from shinobi.loaders.yaml_cab import loads
from shinobi.offload._codec import ModelSpec


@pytest.fixture(params=["cab", "worker"])
def schema_loader(request, tmp_path):
    def load(spec):
        if request.param == "cab":
            cab = loads(f"cabs:\n  tool:\n    command: tool\n    inputs:\n      value: {{{spec}}}\n")["tool"]
            return cab.inputs_model
        path = tmp_path / "worker.yaml"
        path.write_text(f"name: worker\ninputs:\n  value: {{{spec}}}\n")
        return load_worker_schema(path).inputs_model

    return load


def test_nonnullable_default_choices_survive_codec(schema_loader):
    model = schema_loader("dtype: int, required: true, default: 0, choices: [0, 1], nullable: false")
    for candidate in (model, ModelSpec.capture(model).restore()):
        assert candidate().value == 0
        assert candidate(value=1).value == 1
        for invalid in (None, 2):
            with pytest.raises(ValidationError):
                candidate(value=invalid)
        assert "anyOf" not in candidate.model_json_schema()["properties"]["value"]


def test_nonnullable_default_is_validated(schema_loader):
    model = schema_loader("dtype: int, default: '3', nullable: false")
    for candidate in (model, ModelSpec.capture(model).restore()):
        assert candidate().value == 3
    model = schema_loader("dtype: int, default: 2, choices: [0, 1], nullable: false")
    with pytest.raises(ValidationError):
        model()


def test_required_nonnullable_no_default(schema_loader):
    model = schema_loader("dtype: int, required: true, nullable: false")
    for candidate in (model, ModelSpec.capture(model).restore()):
        for data in ({}, {"value": None}):
            with pytest.raises(ValidationError):
                candidate(**data)
        assert candidate(value=0).value == 0


def test_legacy_nullability_unchanged(schema_loader):
    model = schema_loader("dtype: int, required: true, default: 0, choices: [0, 1]")
    assert model().value == 0
    assert model(value=None).value is None


def test_explicit_nullable_required_field(schema_loader):
    model = schema_loader("dtype: int, required: true, nullable: true")
    assert model(value=None).value is None
    with pytest.raises(ValidationError):
        model()


@pytest.mark.parametrize(
    "spec, match",
    [
        ("dtype: int, nullable: false", "requires required: true"),
        ("dtype: int, required: true, default: null, nullable: false", "null default"),
        ("dtype: int, nullable: 'false'", "must be a boolean"),
        ("dtype: int, nullable: null", "must be a boolean"),
        ("dtype: int, choices: [0, null], required: true, nullable: false", "annotation accepting null"),
        ("dtype: 'Optional[int]', required: true, nullable: false", "nullable dtype"),
        ("dtype: 'Union[int, None]', required: true, nullable: false", "nullable dtype"),
    ],
)
def test_invalid_nullable_declarations_are_load_errors(schema_loader, spec, match):
    with pytest.raises((CabLoadError, ConfigLoadError), match=match):
        schema_loader(spec)


def test_nullable_metadata_preserved_on_cab_inputs_and_outputs():
    cab = loads("""cabs:
  tool:
    command: tool
    inputs:
      value: {dtype: int, default: 0, nullable: false}
    outputs:
      result: {dtype: File, required: true, nullable: false}
""")["tool"]
    assert cab.field_meta["value"].nullable is False
    assert cab.field_meta["result"].nullable is False
    with pytest.raises(ValidationError):
        cab.outputs_model(result=None)
