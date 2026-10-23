"""Cross-package ownership and cache/transport contracts."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field, field_validator

from dosho.validation import compile_cab, parse_document, build_argv as format_argv
from dosho.validation.modelgen import dtype_to_type
from shinobi import Cab
from shinobi.cache import cab_semantic_identity, compute_cache_key
from shinobi.loaders.yaml_cab import loads
from shinobi.offload._codec import BundleError, ModelSpec
from shinobi.offload.bundle import ScopeSpec
from shinobi.policies import build_argv
from shinobi.steps.schema import ParamMeta, Policies


def test_standalone_compile_and_format():
    compiled = compile_cab(parse_document("tool", {"command": "tool", "inputs": {"some-name": {"dtype": "int", "required": True}}}))
    values = compiled.inputs_model(some_name="3").model_dump()
    assert format_argv(compiled, values) == ["tool", "--some-name", "3"]


def test_unbound_strict_dtype_never_falls_back():
    with pytest.raises(ValueError, match="binding"):
        dtype_to_type("List[Union[int, MSv2]]")
    assert dtype_to_type("legacy-unknown") is str


def test_framework_fragments_are_inert_then_authoritatively_validated():
    spec = parse_document("tool", {"command": "tool", "derived_reads": {"x": {"not_a_contract": True}}})
    compiled = compile_cab(spec)
    assert compiled.framework_declarations["derived_reads"]["x"]["not_a_contract"] is True
    with pytest.raises(ValueError):
        loads(json.dumps({"tool": spec.document}))


def test_cache_identity_changes_with_formatting_and_argument_order(monkeypatch):
    cab = loads("tool: {command: tool, inputs: {x: {dtype: int}, y: {dtype: int}}}")["tool"]
    key = compute_cache_key(cab, None, {"x": 1, "y": 2})
    assert compute_cache_key(cab.model_copy(update={"policies": Policies(prefix="-")}), None, {"x": 1, "y": 2}) != key
    assert compute_cache_key(cab.model_copy(update={"field_meta": {"x": ParamMeta(positional=True)}}), None, {"x": 1, "y": 2}) != key
    import dosho.validation

    monkeypatch.setattr(dosho.validation, "FORMATTER_CONTRACT_VERSION", 2)
    assert compute_cache_key(cab, None, {"x": 1, "y": 2}) != key


def test_trusted_local_validators_factories_and_implicit_objects_remain_supported():
    class Inputs(BaseModel):
        value: str = Field(default_factory=lambda: "value")

        @field_validator("value")
        @classmethod
        def normalize(cls, value):
            return value.upper()

    class Token:
        def __str__(self):
            return "fixed"

    cab = Cab(name="tool", command="tool", inputs_model=Inputs, outputs_model=BaseModel, field_meta={"value": ParamMeta(implicit=Token())})
    assert build_argv(cab, Inputs().model_dump()) == ["tool", "--value", "fixed"]
    assert compute_cache_key(cab, None, Inputs().model_dump())
    with pytest.raises(BundleError):
        ModelSpec.capture(Inputs)


def test_frozen_cab_semantics_are_verified():
    cab = loads("tool: {command: tool}")["tool"]
    frozen = ScopeSpec.capture(cab)
    assert type(frozen.restore()) is Cab
    assert frozen.cab_semantics == cab_semantic_identity(cab)
    changed = frozen.model_dump(mode="json")
    changed["cab_semantics"][2] += 1
    with pytest.raises(BundleError, match="semantics"):
        ScopeSpec.model_validate(changed).restore()


@pytest.mark.parametrize("container", [lambda t: t, lambda t: t | None, lambda t: list[t]])
def test_choices_cannot_erase_extended_strict_annotations(container):
    from typing import Annotated
    from shinobi.datasets import MSv2, DatasetDeclarationError
    from shinobi.loaders._modelgen import narrow_choices

    with pytest.raises(DatasetDeclarationError, match="erase dataset metadata"):
        narrow_choices(container(Annotated[MSv2, "trusted-marker"]), ["source.ms"])


def test_replacement_order_is_part_of_cab_identity():
    cab = loads("tool: {command: tool, inputs: {a: {dtype: str}}}")["tool"]
    first = cab.model_copy(update={"policies": Policies(replace={"a": "b", "b": "c"})})
    second = cab.model_copy(update={"policies": Policies(replace={"b": "c", "a": "b"})})
    assert build_argv(first, {"a": "value"}) != build_argv(second, {"a": "value"})
    assert cab_semantic_identity(first) != cab_semantic_identity(second)


def test_new_bundle_requires_cab_semantics_and_old_bundle_stays_readable(tmp_path):
    from shinobi import Recipe
    from shinobi.config import AppConfig
    from shinobi.offload.bundle import RecipeBundle, freeze_recipe

    cab = loads("tool: {command: tool}")["tool"]

    class Empty(BaseModel):
        pass

    recipe = Recipe(name="r", inputs_model=Empty, outputs_model=Empty)
    recipe.add_step("tool", cab)
    bundle = freeze_recipe(recipe, {}, config=AppConfig(), workspace=tmp_path)
    assert bundle.schema_version == 3
    payload = bundle.model_dump(mode="json")
    payload["steps"][0]["scope"].pop("cab_semantics")
    with pytest.raises(ValueError, match="requires frozen semantics"):
        RecipeBundle.model_validate(payload)
    payload["schema_version"] = 1
    assert RecipeBundle.model_validate(payload).declaration().steps[0].step.command == "tool"


@pytest.mark.parametrize("shape", ["mapping", "list", "tuple", "nested"])
def test_implicit_mapping_order_changes_cached_execution_and_frozen_semantics(tmp_path, monkeypatch, shape):
    from typing import Any

    from shinobi.backends.recording import RecordingBackend
    from shinobi.offload._codec import pack
    from shinobi.steps.dispatch import _STEP_BACKENDS, _dispatch

    class Inputs(BaseModel):
        value: Any = None

    class Empty(BaseModel):
        pass

    def implicit(reverse):
        value = dict(reversed([("a", 1), ("b", 2)])) if reverse else {"a": 1, "b": 2}
        return {"mapping": value, "list": [value], "tuple": (value,), "nested": {"outer": [value]}}[shape]

    recorder = RecordingBackend()
    monkeypatch.setitem(_STEP_BACKENDS, "implicit-record", recorder)
    cab = Cab(
        name="tool",
        command="tool",
        inputs_model=Inputs,
        outputs_model=Empty,
        field_meta={"value": ParamMeta(implicit=implicit(False))},
        backend="implicit-record",
        cache=True,
        cache_dir=str(tmp_path / "cache"),
    )
    changed = cab.model_copy(update={"field_meta": {"value": ParamMeta(implicit=implicit(True))}})
    assert not _dispatch(cab, None).cached
    assert _dispatch(cab, None).cached
    assert not _dispatch(changed, None).cached
    assert len(recorder.calls) == 2
    assert recorder.calls[0][1] != recorder.calls[1][1]
    assert build_argv(cab, {}) != build_argv(changed, {})

    frozen = ScopeSpec.capture(cab)
    assert build_argv(frozen.restore(), {}) == build_argv(cab, {})
    changed_settings = dict(frozen.settings)
    changed_settings["field_meta"] = pack({name: meta.model_dump(mode="python") for name, meta in changed.field_meta.items()})
    with pytest.raises(BundleError, match="semantics"):
        frozen.model_copy(update={"settings": changed_settings}).restore()


@pytest.mark.parametrize("head", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_implicit_field_boundaries_invalidate_cache_and_frozen_semantics(tmp_path, monkeypatch, head, empty):
    from shinobi.backends.recording import RecordingBackend
    from shinobi.offload._codec import pack
    from shinobi.steps.dispatch import _STEP_BACKENDS, _dispatch

    class Inputs(BaseModel):
        a: list[str] = []
        b: str = "B"
        c: list[str] = []

    class Empty(BaseModel):
        pass

    def metadata(changed):
        a, c = (([], ["x"]) if not changed else (["x"], [])) if empty else ((["x"], ["y", "z"]) if not changed else (["x", "y"], ["z"]))
        return {
            "a": ParamMeta(implicit=a, positional=True, positional_head=head, repeat_as_tokens=True),
            "b": ParamMeta(positional=True, positional_head=head),
            "c": ParamMeta(implicit=c, positional=True, positional_head=head, repeat_as_tokens=True),
        }

    recorder = RecordingBackend()
    monkeypatch.setitem(_STEP_BACKENDS, "boundary-record", recorder)
    cab = Cab(
        name="tool", command="tool", inputs_model=Inputs, outputs_model=Empty, field_meta=metadata(False), backend="boundary-record", cache=True, cache_dir=str(tmp_path / "cache")
    )
    changed = cab.model_copy(update={"field_meta": metadata(True)})
    # Without the resolved middle input, the concatenated implicit tokens match.
    assert build_argv(cab, {}) == build_argv(changed, {})
    assert build_argv(cab, {"b": "B"}) != build_argv(changed, {"b": "B"})
    assert not _dispatch(cab, None).cached
    assert _dispatch(cab, None).cached
    assert not _dispatch(changed, None).cached
    assert [call[1] for call in recorder.calls] == [build_argv(cab, {"b": "B"}), build_argv(changed, {"b": "B"})]

    frozen = ScopeSpec.capture(cab)
    assert build_argv(frozen.restore(), {"b": "B"}) == build_argv(cab, {"b": "B"})
    settings = dict(frozen.settings)
    settings["field_meta"] = pack({name: meta.model_dump(mode="python") for name, meta in changed.field_meta.items()})
    with pytest.raises(BundleError, match="semantics"):
        frozen.model_copy(update={"settings": settings}).restore()


def test_implicit_identity_preserves_boolean_mode_and_unsupported_flavour():
    cab = loads("tool: {command: tool, inputs: {value: {dtype: str}}}")["tool"]
    flag = cab.model_copy(update={"field_meta": {"value": ParamMeta(implicit=True)}})
    text = cab.model_copy(update={"field_meta": {"value": ParamMeta(implicit="true")}})
    assert build_argv(flag, {}) == ["tool", "--value"]
    assert build_argv(text, {}) == ["tool", "--value", "true"]
    assert compute_cache_key(flag, None, {}) != compute_cache_key(text, None, {})

    from shinobi.exceptions import UnsupportedFlavourError

    unsupported = flag.model_copy(update={"flavour": "python-code"})
    assert compute_cache_key(unsupported, None, {})
    with pytest.raises(UnsupportedFlavourError):
        build_argv(unsupported, {})


def test_decorated_cab_cache_includes_both_code_and_argument_semantics(tmp_path, monkeypatch):
    import shinobi
    from shinobi.backends.recording import RecordingBackend
    from shinobi.cache import ExecutionIdentity
    from shinobi.steps.dispatch import _STEP_BACKENDS, _dispatch

    cab = loads("tool: {command: tool, inputs: {value: {dtype: int, default: 1}}}")["tool"]
    recorder = RecordingBackend()
    monkeypatch.setitem(_STEP_BACKENDS, "decorated-record", recorder)
    cab = cab.model_copy(update={"backend": "decorated-record", "cache": True, "cache_dir": str(tmp_path)})

    @shinobi.step(scope=cab)
    def wrapped(ctx):
        return ctx.run()

    changed = cab.model_copy(update={"policies": Policies(prefix="-")})
    assert not _dispatch(cab, wrapped.func).cached
    assert _dispatch(cab, wrapped.func).cached
    assert not _dispatch(changed, wrapped.func).cached
    assert [call[1] for call in recorder.calls] == [["tool", "--value", "1"], ["tool", "-value", "1"]]
    execution = ExecutionIdentity(code_digest="same-trusted-code")
    assert compute_cache_key(cab, wrapped.func, {"value": 1}, execution=execution) != compute_cache_key(changed, wrapped.func, {"value": 1}, execution=execution)
