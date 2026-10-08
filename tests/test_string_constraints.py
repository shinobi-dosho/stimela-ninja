"""Declarative scalar string constraints are enforced by the generated models."""

import pytest
import yaml
from pydantic import ValidationError

from shinobi.exceptions import CabLoadError, ConfigLoadError, ParameterError
from shinobi.loaders import yaml_cab
from shinobi.loaders.worker_schema import load_worker_schema
from shinobi.loaders._modelgen import build_model
from shinobi.backends.recording import RecordingBackend
from shinobi.steps import register_step_backend
from shinobi.steps.dispatch import _dispatch

_PATTERN = r"\A(?!.*(?:None|True|False|[#\s\[\],])).+\Z"


def _document(**field):
    return {"cabs": {"solver": {"command": "solver", "inputs": {"sols": {"dtype": "str", "required": True, "string_pattern": _PATTERN, **field}}}}}


@pytest.mark.parametrize("invalid", ["", "None", "a/None/b", "a#b", "a b", "a\nb", "[a,b]", "True", "False"])
def test_cab_constraint_rejects_before_backend(tmp_path, monkeypatch, invalid):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "ownership.json"))
    backend = RecordingBackend()
    register_step_backend("string-check", backend)
    cab = yaml_cab.loads(yaml.safe_dump(_document()))["solver"].model_copy(update={"backend": "string-check"})
    with pytest.raises(ParameterError, match="pattern"):
        _dispatch(cab, None, sols=invalid)
    assert backend.calls == []


def test_cab_valid_string_and_json_schema():
    cab = yaml_cab.loads(yaml.safe_dump(_document()))["solver"]
    assert cab.inputs_model(sols="/data/solutions-1").sols == "/data/solutions-1"
    assert cab.inputs_model.model_json_schema()["properties"]["sols"]["pattern"] == _PATTERN
    assert cab.field_meta["sols"].string_pattern == _PATTERN


def test_optional_and_default_values_are_constrained():
    cab = yaml_cab.loads(yaml.safe_dump(_document(required=False)))["solver"]
    assert cab.inputs_model().sols is None
    cab = yaml_cab.loads(yaml.safe_dump(_document(default="None")))["solver"]
    with pytest.raises(ValidationError, match="pattern"):
        cab.inputs_model()


@pytest.mark.parametrize("field", [{"dtype": "Directory"}, {"dtype": "int"}, {"string_pattern": "["}, {"string_pattern": 3}])
def test_bad_constraint_declarations_are_rejected(field):
    with pytest.raises(ValueError, match="string_pattern"):
        yaml_cab.loads(yaml.safe_dump(_document(**field)))


def test_choices_and_pattern_both_apply():
    model = build_model("Choice", {"value": ("str", True, None)}, choices={"value": ["ok", "None"]}, string_patterns={"value": _PATTERN})
    assert model(value="ok").value == "ok"
    with pytest.raises(ValidationError):
        model(value="None")
    with pytest.raises(ValidationError):
        model(value="other")


def test_dynamic_attribute_constraint_is_refused():
    spec = _document()
    spec["cabs"]["solver"]["input_patterns"] = [{"segments": [{"regex": ".+"}, {"attrs": {"value": {"dtype": "str", "string_pattern": _PATTERN}}}]}]
    with pytest.raises(CabLoadError, match="literal fields"):
        yaml_cab.loads(yaml.safe_dump(spec))


def test_worker_constraints_share_the_same_model_behavior(tmp_path):
    path = tmp_path / "worker.yaml"
    path.write_text(yaml.safe_dump({"name": "worker", "inputs": {"sols": {"dtype": "str", "required": True, "string_pattern": _PATTERN}}}))
    model = load_worker_schema(path).inputs_model
    assert model(sols="solutions").sols == "solutions"
    with pytest.raises(ValidationError, match="pattern"):
        model(sols="None")
    assert model.model_json_schema()["properties"]["sols"]["pattern"] == _PATTERN
    path.write_text(yaml.safe_dump({"name": "worker", "inputs": {"sols": {"dtype": "Directory", "string_pattern": _PATTERN}}}))
    with pytest.raises(ConfigLoadError, match="scalar str"):
        load_worker_schema(path)


def test_string_pattern_rejects_non_string_choices():
    with pytest.raises(ValueError, match="choices must all be strings"):
        build_model("BadChoice", {"value": ("str", True, None)}, choices={"value": [1]}, string_patterns={"value": _PATTERN})


def test_constraint_only_spec_is_a_leaf_and_outputs_are_constrained():
    doc = {"cabs": {"app": {"command": "app", "inputs": {"value": {"string_pattern": "^ok$"}}, "outputs": {"result": {"dtype": "str", "string_pattern": "^ok$"}}}}}
    cab = yaml_cab.loads(yaml.safe_dump(doc))["app"]
    assert cab.inputs_model(value="ok").value == "ok"
    with pytest.raises(ValidationError):
        cab.outputs_model(result="bad")


def test_revised_constraint_is_checked_before_cache_lookup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = RecordingBackend()
    register_step_backend("string-cache", backend)
    document = _document(string_pattern="^s.*$")
    cab = yaml_cab.loads(yaml.safe_dump(document))["solver"].model_copy(update={"backend": "string-cache", "cache": True, "cache_dir": str(tmp_path / "cache")})
    first = _dispatch(cab, None, sols="sols")
    assert not first.cached
    assert _dispatch(cab, None, sols="sols").cached
    document["cabs"]["solver"]["inputs"]["sols"]["string_pattern"] = "^other$"
    revised = yaml_cab.loads(yaml.safe_dump(document))["solver"].model_copy(update={"backend": "string-cache", "cache": True, "cache_dir": str(tmp_path / "cache")})
    with pytest.raises(ParameterError, match="pattern"):
        _dispatch(revised, None, sols="sols")
    assert len(backend.calls) == 1
