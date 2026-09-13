"""Independent-review reproductions for the experimental M1 protocol."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from shinobi.config import AppConfig
from shinobi.offload import bundle as bundle_module
from shinobi.offload._codec import BundleError, ModelSpec, pack, unpack
from shinobi.offload.bundle import RecipeBundle, ScopeSpec, Submission, freeze_recipe, write_new
from shinobi.offload.code import CodeBundle, capture_code
from shinobi.offload.records import AttemptRecord
from shinobi.resources import Resources
from shinobi.results import StepResult
from shinobi.steps.schema import Mutability, ParamMeta, Scope

from .test_offload_bundle import Empty, recipe, source_function


def identity():
    return {"workflow_id": uuid4(), "attempt_id": uuid4(), "step_path": "loop.2.step", "bundle_digest": "bundle-key"}


@pytest.mark.parametrize(
    "annotation,value,strict",
    [
        (Path, Path("relative.ms"), True),
        (tuple[Path, int], (Path("relative.ms"), 1), True),
        (Path | str, Path("relative.ms"), False),
        (tuple[int, ...] | list[int], (1, 2), False),
        (Any, {"nested": (Path("relative.ms"), [1, True])}, False),
    ],
)
@pytest.mark.parametrize("state", ["succeeded", "cached", "skipped", "failed"])
def test_record_io_survives_disk_round_trip(tmp_path, annotation, value, strict, state):
    model = create_model("Payload", __config__=ConfigDict(strict=strict), value=(annotation, ...))
    scope = Scope(name="payload", inputs_model=model, outputs_model=model)
    result = StepResult(
        name="payload", returncode=1 if state == "failed" else 0, inputs=model(value=value), outputs=model(value=value), cached=state == "cached", skipped=state == "skipped"
    )
    ids = identity()
    record = AttemptRecord.from_result(result, **ids)
    loaded = AttemptRecord.read(record.write(tmp_path), **ids)
    restored = loaded.result(scope)
    assert restored.inputs.value == value
    assert restored.outputs.value == value
    assert type(restored.inputs.value) is type(value)
    assert type(restored.outputs.value) is type(value)
    assert loaded.state == state


@pytest.mark.parametrize("value", [object(), {"nested": [object()]}, float("nan"), float("inf")])
@pytest.mark.parametrize("side", ["inputs", "outputs"])
def test_unsupported_record_values_fail_before_publication(tmp_path, value, side):
    model = create_model("Payload", value=(Any, ...))
    values = {"inputs": model(value=1), "outputs": model(value=1)}
    values[side] = model(value=value)
    result = StepResult(name="payload", returncode=0, **values)
    with pytest.raises(BundleError, match="cannot transport"):
        AttemptRecord.from_result(result, **identity()).write(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_reporting_exclusions_do_not_remove_executable_record_inputs(tmp_path):
    model = create_model("Hidden", value=(Path, Field(exclude=True)))
    scope = Scope(name="payload", inputs_model=model, outputs_model=model)
    result = StepResult(name="payload", returncode=0, inputs=model(value="input"), outputs=model(value="output"))
    ids = identity()
    record = AttemptRecord.from_result(result, **ids)
    restored = AttemptRecord.read(record.write(tmp_path), **ids).result(scope)
    assert restored.inputs.value == Path("input")
    assert restored.outputs.value == Path("output")


@pytest.mark.parametrize("annotation,wrap", [(Any, lambda v: v), (list[Any], lambda v: [v]), (dict[str, Any], lambda v: {"v": v})])
def test_models_hidden_under_any_are_rejected(annotation, wrap):
    class Child(BaseModel):
        n: int

    model = create_model("OpaqueModel", value=(annotation, ...))
    result = StepResult(name="payload", returncode=0, inputs=Empty(), outputs=model(value=wrap(Child(n=1))))
    with pytest.raises(BundleError, match="model values under Any"):
        AttemptRecord.from_result(result, **identity())


def test_explicit_nested_model_record_round_trip(tmp_path):
    class Child(BaseModel):
        path: Path

    model = create_model("NestedModel", child=(Child, ...))
    scope = Scope(name="payload", inputs_model=Empty, outputs_model=model)
    result = StepResult(name="payload", returncode=0, inputs=Empty(), outputs=model(child=Child(path=Path("child.ms"))))
    ids = identity()
    record = AttemptRecord.from_result(result, **ids)
    restored = AttemptRecord.read(record.write(tmp_path), **ids).result(scope)
    assert isinstance(restored.outputs.child, Child)
    assert restored.outputs.child.path == Path("child.ms")


@pytest.mark.parametrize("wrap", [lambda child: child, lambda child: [child], lambda child: (child,), lambda child: {"child": child}, lambda child: [{"nested": (child,)}]])
def test_model_defaults_rejected_at_any_container_depth(wrap):
    class Child(BaseModel):
        n: int

    model = create_model("Defaults", value=(Any, wrap(Child(n=3))))
    with pytest.raises(BundleError, match="model-instance defaults"):
        ModelSpec.capture(model)


def test_scope_and_config_use_finite_type_preserving_payloads(tmp_path):
    value = recipe()
    cab = value.steps[0].step
    cab.field_meta["ms"] = ParamMeta(implicit=Path("product.ms"))
    cab.field_meta["n"] = ParamMeta(implicit=(Path("path"), "tuple"))
    cab.input_mutability["ms"] = Mutability.MUTABLE
    config = AppConfig()
    frozen = freeze_recipe(value, {}, config=config, workspace=tmp_path)
    loaded = RecipeBundle.read(frozen.stage(tmp_path / "runs") / "bundle.json")
    restored = loaded.steps[0].scope.restore()
    assert restored.field_meta["ms"].implicit == Path("product.ms")
    assert restored.field_meta["n"].implicit == (Path("path"), "tuple")
    assert restored.input_mutability["ms"] is Mutability.MUTABLE
    assert {k: unpack(v) for k, v in loaded.config.items()} == config.model_dump(mode="python")
    assert loaded.digest == frozen.digest
    meta = ParamMeta(implicit=(Path("p"), 2))
    assert unpack(pack(meta)).implicit == meta.implicit


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("where", ["resources", "metadata", "config"])
def test_nonfinite_settings_fail_during_freeze(tmp_path, number, where):
    value, config = recipe(), AppConfig()
    if where == "resources":
        value.steps[0].step.resources = Resources.model_construct(cpus=number)
    elif where == "metadata":
        value.steps[0].step.field_meta["n"] = ParamMeta(implicit={"nested": [number]})
    else:
        config.execution.resources.cpus = number
    with pytest.raises(BundleError, match="cannot transport"):
        freeze_recipe(value, {}, config=config, workspace=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_nonfinite_persisted_settings_rejected(tmp_path):
    spec = ScopeSpec.capture(recipe().steps[0].step).model_dump()
    spec["settings"]["resources"] = ["dict", {"cpus": ["scalar", float("nan")]}]
    with pytest.raises(ValidationError):
        ScopeSpec.model_validate(spec)


@pytest.mark.parametrize("alias", ["runtime_alias", "__main__", "os.system;not_a_module"])
def test_alias_loaded_or_mutated_module_address_rejected(tmp_path, alias):
    func = source_function(tmp_path)
    func.__module__ = alias
    with pytest.raises(ValidationError, match="does not match source entry"):
        capture_code(func, roots=(tmp_path,))


@pytest.mark.parametrize("entry,module", [("pkg/mod.py", "pkg.mod"), ("pkg/__init__.py", "pkg")])
def test_package_entry_address_matches_materialized_module(tmp_path, entry, module):
    source = tmp_path / entry
    source.parent.mkdir()
    source.write_text("def compute():\n    return 42\n")
    (source.parent / "__init__.py").touch(exist_ok=True)
    spec = importlib.util.spec_from_file_location(module, source)
    imported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported)
    bundle = capture_code(imported.compute, roots=(tmp_path,))
    loaded = CodeBundle.model_validate_json(bundle.model_dump_json())
    assert loaded.module == module
    assert loaded.write(tmp_path / "staged").relative_to(tmp_path / "staged").as_posix() == entry
    data = loaded.model_dump()
    data["module"] = "runtime_alias"
    with pytest.raises(ValidationError, match="does not match source entry"):
        CodeBundle.model_validate(data)


@pytest.mark.parametrize("state", ["succeeded", "cached", "skipped", "failed"])
def test_final_observation_must_agree_with_envelope_on_read(tmp_path, state):
    result = StepResult(name="cab", returncode=1 if state == "failed" else 0, inputs=Empty(), outputs=Empty(), cached=state == "cached", skipped=state == "skipped")
    ids = identity()
    record = AttemptRecord.from_result(result, **ids)
    path = record.write(tmp_path)
    broken = record.model_copy(update={"observation": record.observation.model_copy(update={"name": "wrong-step"})})
    with pytest.raises(ValidationError, match="logical step_path"):
        broken.write(tmp_path)
    path.write_text(broken.model_dump_json())
    with pytest.raises(ValidationError, match="logical step_path"):
        AttemptRecord.read(path, **ids)


def track_syncs(monkeypatch):
    real_open, real_close, real_fsync, real_link = os.open, os.close, os.fsync, os.link
    directories, events = {}, []

    def opened(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if flags & os.O_DIRECTORY:
            directories[fd] = Path(path)
        return fd

    def synced(fd):
        events.append(("directory", directories[fd]) if fd in directories else ("file", None))
        real_fsync(fd)

    def closed(fd):
        directories.pop(fd, None)
        real_close(fd)

    def linked(source, destination, **kwargs):
        events.append(("link", Path(destination)))
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(os, "close", closed)
    monkeypatch.setattr(os, "fsync", synced)
    monkeypatch.setattr(os, "link", linked)
    return events


def test_publication_syncs_file_then_link_then_entire_parent_chain(tmp_path, monkeypatch):
    events = track_syncs(monkeypatch)
    path = tmp_path / "new" / "attempts" / "attempt" / "final.json"
    record = Submission(workflow_id=uuid4(), bundle_digest="key")
    write_new(path, record)
    assert events[:2] == [("file", None), ("link", path)]
    assert events[2:] == [("directory", parent) for parent in (path.parent, *path.parent.parents)]
    assert path.is_file()


def test_submission_directory_entries_are_synced(tmp_path, monkeypatch):
    value = freeze_recipe(recipe(), {}, config=AppConfig(), workspace=tmp_path)
    events = track_syncs(monkeypatch)
    root = tmp_path / "new" / "runs"
    directory = value.stage(root)
    synced = [path for kind, path in events if kind == "directory"]
    assert all(parent in synced for parent in (directory, *directory.parents))


def test_publication_syncs_both_symlink_and_target_ancestry(tmp_path, monkeypatch):
    target = tmp_path / "shared" / "target"
    target.mkdir(parents=True)
    alias = tmp_path / "links" / "alias"
    alias.parent.mkdir()
    alias.symlink_to(target, target_is_directory=True)
    events = track_syncs(monkeypatch)
    path = alias / "new" / "record.json"
    write_new(path, Submission(workflow_id=uuid4(), bundle_digest="key"))
    synced = [parent for kind, parent in events if kind == "directory"]
    assert target / "new" in synced
    assert target in synced
    assert target.parent in synced
    assert alias.parent in synced


def test_failed_parent_sync_is_reported_and_cannot_overwrite_record(tmp_path, monkeypatch):
    record = Submission(workflow_id=uuid4(), bundle_digest="key")
    path = tmp_path / "new" / "record.json"

    def failed_sync(parent):
        raise OSError("directory fsync failed")

    with monkeypatch.context() as patch:
        patch.setattr(bundle_module, "_sync_directory_chain", failed_sync)
        with pytest.raises(OSError, match="fsync failed"):
            write_new(path, record)
    # Post-link failure is uncertain durability, never a successful return.
    # A retry must not replace an already visible record.
    assert path.is_file()
    with pytest.raises(FileExistsError):
        write_new(path, record)


def test_failed_file_sync_never_publishes_final_record(tmp_path, monkeypatch):
    def failed_sync(fd):
        raise OSError("file fsync failed")

    monkeypatch.setattr(os, "fsync", failed_sync)
    path = tmp_path / "record.json"
    with pytest.raises(OSError, match="file fsync failed"):
        write_new(path, Submission(workflow_id=uuid4(), bundle_digest="key"))
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
