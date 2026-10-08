"""Finite derived reads share identity across planning and execution routes."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from shinobi.cache import compute_cache_key
from shinobi.derived import DerivedAddress, DerivedRead, derived_reads, read_fingerprint, regular_file_identity
from shinobi.loaders import yaml_cab
from shinobi.products import FamilySpec
from shinobi.ownership import WorkspaceOwnershipError
from shinobi.steps.schema import Cab, path_accesses


class Inputs(BaseModel):
    prefix: str = "image"
    enabled: bool = True


class Empty(BaseModel):
    pass


def reader(required=True, member="file"):
    return Cab(
        name="derived-reader",
        command="true",
        inputs_model=Inputs,
        outputs_model=Empty,
        derived_reads={
            "model": DerivedRead(
                member=member, family=FamilySpec(root=".", coordinates={}, rules=[{"path": "{prefix}-model.fits", "required": required, "when": {"enabled": [True]}}])
            )
        },
    )


def test_dependency_path_is_claimed_even_before_it_exists(tmp_path):
    cab = reader()
    assert (tmp_path / "image-model.fits", False) in path_accesses(cab, {"prefix": "image", "enabled": True}, workspace=tmp_path)


def test_wired_prefix_does_not_hide_physical_edits_or_deletion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cab = reader(False)
    params = cab.inputs_model().model_dump()
    model = tmp_path / "image-model.fits"
    absent = compute_cache_key(cab, None, params, {"prefix": "upstream"})
    model.write_text("first")
    present = compute_cache_key(cab, None, params, {"prefix": "upstream"})
    model.write_text("a changed model")
    changed = compute_cache_key(cab, None, params, {"prefix": "upstream"})
    model.unlink()
    assert compute_cache_key(cab, None, params, {"prefix": "upstream"}) == absent
    assert len({absent, present, changed}) == 3


def test_required_missing_dependency_refuses_before_cache_lookup(tmp_path, monkeypatch):
    from shinobi.cache import CacheManifest
    from shinobi.steps.dispatch import _dispatch

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "ownership.json"))
    monkeypatch.setattr(CacheManifest, "check", lambda *args, **kwargs: pytest.fail("missing required derived read reached cache lookup"))
    with pytest.raises(ValueError, match="required derived read is absent"):
        _dispatch(reader(), None, cache=True, cache_dir=str(tmp_path / "cache"))


@pytest.mark.parametrize("member", ["file", "directory"])
def test_wrong_member_kind_refuses(tmp_path, member):
    path = tmp_path / "image-model.fits"
    if member == "file":
        path.mkdir()
    else:
        path.write_text("file")
    read = derived_reads(reader(member=member), Inputs().model_dump(), tmp_path)[0]
    with pytest.raises(ValueError, match="wrong member kind"):
        read_fingerprint(read)


def test_symlink_and_duplicate_alias_reads_refuse(tmp_path):
    (tmp_path / "actual").write_text("model")
    (tmp_path / "image-model.fits").symlink_to(tmp_path / "actual")
    read = derived_reads(reader(), Inputs().model_dump(), tmp_path)[0]
    with pytest.raises(ValueError, match="symlink"):
        read_fingerprint(read)
    duplicate = reader().model_copy(update={"derived_reads": {"first": reader().derived_reads["model"], "second": reader().derived_reads["model"]}})
    with pytest.raises(ValueError, match="overlapping"):
        derived_reads(duplicate, Inputs().model_dump(), tmp_path)


@pytest.mark.parametrize("path", ["{unknown}", "{prefix.name}", "{prefix[0]}", "{prefix!r}"])
def test_closed_templates_refuse_unknown_or_executable_shape(path):
    declaration = DerivedRead(member="file", family=FamilySpec(coordinates={}, rules=[{"path": path}]))
    with pytest.raises(ValueError, match="placeholder"):
        reader().model_copy(update={"derived_reads": {"model": declaration}}).__class__(
            name="bad", command="true", inputs_model=Inputs, outputs_model=Empty, derived_reads={"model": declaration}
        )


def test_discovery_is_not_a_dependency_language():
    with pytest.raises(ValidationError, match="finite"):
        DerivedRead(member="file", family=FamilySpec(coordinates={"channel": "int"}, rules=[{"path": "{channel}.fits", "captures": {"channel": "int"}}]))


def test_presence_predicate_has_finite_empty_semantics(tmp_path):
    from shinobi.products import FamilyPlan

    spec = FamilySpec(coordinates={}, rules=[{"path": "model.fits", "when_set": {"solution": True}}])
    for value in [None, [], ""]:
        assert not FamilyPlan(spec, Path, {"solution": value}, tmp_path).candidates
    assert FamilyPlan(spec, Path, {"solution": ["solution.h5"]}, tmp_path).candidates


def test_regular_file_identity_is_independent_of_snapshot_basename(tmp_path):
    first = tmp_path / "image-model.fits"
    second = tmp_path / "snapshot"
    first.write_bytes(b"model")
    second.write_bytes(b"model")
    assert regular_file_identity(first) == regular_file_identity(second)


def test_typed_named_addresses_do_not_collide():
    assert DerivedAddress(name="model", coordinates={"channel": 1}).field != DerivedAddress(name="model", coordinates={"channel": "1"}).field
    assert DerivedAddress(name="model", coordinates={}).field != DerivedAddress(name="gain", coordinates={}).field


def test_yaml_and_worker_schema_preserve_declaration():
    from shinobi.offload.bundle import ScopeSpec

    cab = yaml_cab.loads("""cabs:
  derived:
    command: "true"
    inputs:
      prefix: {dtype: str, default: image}
    derived_reads:
      model:
        member: file
        family:
          root: '.'
          coordinates: {}
          rules:
            - {path: '{prefix}-model.fits', required: true}
""")["derived"]
    assert ScopeSpec.capture(cab).restore().derived_reads == cab.derived_reads


def test_derived_mount_remains_readonly_inside_writable_workdir(tmp_path):
    from shinobi.backends.container import bind_dir_modes

    (tmp_path / "image-model.fits").write_text("model")
    assert (str(tmp_path / "image-model.fits"), False) in bind_dir_modes(reader(), Inputs().model_dump(), str(tmp_path))


def test_sandbox_prediction_stages_reads_without_republishing_them(tmp_path, monkeypatch):
    from shinobi.backends.recording import RecordingBackend
    from shinobi.steps import register_step_backend
    from shinobi.steps.dispatch import _dispatch

    class Prediction(RecordingBackend):
        def run(self, cab, argv, inputs, **kwargs):
            assert (Path(kwargs["cwd"]) / "image-model.fits").read_text() == "model"
            return super().run(cab, argv, inputs, **kwargs)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "ownership.json"))
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / "sandboxes"))
    model = tmp_path / "image-model.fits"
    model.write_text("model")
    before = model.stat()
    register_step_backend("derived-predict", Prediction())
    cab = reader().model_copy(update={"backend": "derived-predict", "sandbox": True, "harvest": ["image-*.fits"]})
    result = _dispatch(cab, None)
    assert result.success
    assert model.stat().st_ino == before.st_ino and model.stat().st_mtime_ns == before.st_mtime_ns


def test_strict_regular_file_identity_refuses_hardlinks(tmp_path):
    import os

    source = tmp_path / "model.fits"
    source.write_text("model")
    os.link(source, tmp_path / "alias.fits")
    with pytest.raises(ValueError, match="hardlinked"):
        regular_file_identity(source)


def test_confirmed_unset_optional_scratch_is_omitted_but_unknown_is_refused(tmp_path):
    from shinobi.steps.schema import _resolved_product_patterns

    class ScratchInputs(BaseModel):
        scratch_dir: Path | None = None

    cab = Cab(name="scratch", command="true", inputs_model=ScratchInputs, outputs_model=Empty, scratch=["{scratch_dir}/*"])
    assert list(_resolved_product_patterns(cab, {"scratch_dir": None}, workspace=tmp_path)) == []
    assert list(_resolved_product_patterns(cab, {}, workspace=tmp_path))[0][1] is None


def test_runtime_dependency_redirect_refuses_before_backend(tmp_path, monkeypatch):
    from shinobi.exceptions import DatasetLifecycleUnavailableError
    from shinobi.steps.dispatch import _dispatch

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "ownership.json"))
    (tmp_path / "image-model.fits").write_text("model")
    (tmp_path / "other-model.fits").write_text("other")
    with pytest.raises(DatasetLifecycleUnavailableError, match="changed after planning"):
        _dispatch(reader(), lambda ctx: ctx.run(prefix="other"))


def test_readonly_worker_executes_and_cache_rechecks_derived_input(tmp_path, monkeypatch):
    from shinobi.steps.schema import Recipe, StepRef
    from tests.test_offload_worker import _prepared_cached, execute_step
    from shinobi.offload.records import AttemptRecord

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "ownership.json"))
    model = tmp_path / "image-model.fits"
    model.write_text("model")
    recipe = Recipe(name="derived-worker", inputs_model=Empty, outputs_model=Empty, steps=[StepRef(name="read", step=reader())])
    cached = []
    for text in [None, None, "edited model"]:
        if text is not None:
            model.write_text(text)
        workflow, bundle, plan = _prepared_cached(tmp_path, recipe)
        assert any(access.path == str(model) and not access.writes for access in plan.accesses)
        attempt = plan.attempts[0]
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
        record = AttemptRecord.read(
            workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json",
            workflow_id=plan.workflow_id,
            attempt_id=attempt.attempt_id,
            step_path=attempt.step_path,
            bundle_digest=bundle.digest,
        )
        cached.append(record.result(reader()).cached)
    assert cached == [False, True, False]


def test_auxiliary_mutation_offload_explicitly_refuses(tmp_path):
    import sys
    from shinobi import DatasetAccess, MSv2
    from shinobi.config import AppConfig
    from shinobi.offload.bundle import freeze_recipe
    from shinobi.offload.slurm import prepare_worker_slurm
    from shinobi.products import ProductFamily
    from shinobi.steps.schema import Mutability, ParamMeta, Recipe, StepRef

    class StrictInputs(Inputs):
        ms: MSv2

    class ModelOutputs(BaseModel):
        model: ProductFamily[Path] | None = None

    strict = Cab(
        name="continue",
        command="true",
        inputs_model=StrictInputs,
        outputs_model=ModelOutputs,
        input_mutability={"ms": Mutability.MUTABLE},
        dataset_accesses=[DatasetAccess(field="ms", mode="write")],
        derived_reads=reader().derived_reads,
        field_meta={"model": ParamMeta(family=FamilySpec(coordinates={}, rules=[{"path": "{prefix}-model.fits", "accept_existing": True}]))},
    )
    recipe = Recipe(name="aux-worker", inputs_model=Empty, outputs_model=Empty, steps=[StepRef(name="continue", step=strict, params={"ms": tmp_path / "obs.ms"})])
    bundle = freeze_recipe(recipe, {}, config=AppConfig(), workspace=tmp_path)
    with pytest.raises(ValueError, match="strict auxiliary mutation workflows cannot be offloaded"):
        prepare_worker_slurm(bundle, submission_root=tmp_path / "runs", worker_python=Path(sys.executable))


@pytest.mark.parametrize("sandbox", [False, True])
@pytest.mark.parametrize("kind", ["ordinary", "family", "directory-parent", "directory-child", "scratch"])
def test_contradictory_output_refuses_before_mutation(tmp_path, monkeypatch, sandbox, kind):
    from shinobi.backends.recording import RecordingBackend
    from shinobi.products import ProductFamily
    from shinobi.steps import register_step_backend
    from shinobi.steps.dispatch import _dispatch
    from shinobi.steps.schema import ParamMeta

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "ownership.json"))
    model = tmp_path / "models" / "model.fits"
    model.parent.mkdir()
    model.write_text("saved model")
    before = model.stat()
    read_path = "models" if kind == "directory-child" else "models/model.fits"
    cab = reader(member="directory" if kind == "directory-child" else "file")
    declaration = DerivedRead(member="directory" if kind == "directory-child" else "file", family=FamilySpec(coordinates={}, rules=[{"path": read_path, "required": True}]))
    meta = {}
    if kind == "scratch":
        outputs = Empty
        cab = cab.model_copy(update={"scratch": ["models/*"]})
    elif kind == "family":

        class Outputs(BaseModel):
            product: ProductFamily[Path] | None = None

        outputs = Outputs
        meta = {"product": ParamMeta(family=FamilySpec(coordinates={}, rules=[{"path": "models/model.fits"}]))}
    else:

        class Outputs(BaseModel):
            product: Path | None = None

        outputs = Outputs
        path = "models" if kind == "directory-parent" else "models/model.fits"
        meta = {"product": ParamMeta(implicit=path)}
    backend = RecordingBackend()
    register_step_backend("derived-conflict", backend)
    cab = cab.model_copy(update={"outputs_model": outputs, "field_meta": meta, "derived_reads": {"model": declaration}, "backend": "derived-conflict"})
    with pytest.raises(WorkspaceOwnershipError, match="overlap"):
        _dispatch(cab, None, sandbox=sandbox)
    assert backend.calls == []
    assert model.read_text() == "saved model"
    assert model.stat().st_ino == before.st_ino


@pytest.mark.parametrize("harvest", ["models/*", "models/**/*", "models"])
def test_readonly_staged_directory_is_not_republished(tmp_path, monkeypatch, harvest):
    from shinobi.backends.recording import RecordingBackend
    from shinobi.steps import register_step_backend
    from shinobi.steps.dispatch import _dispatch

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "ownership.json"))
    model = tmp_path / "models" / "nested" / "model.fits"
    model.parent.mkdir(parents=True)
    model.write_text("saved model")
    before = model.stat()
    declaration = DerivedRead(member="directory", family=FamilySpec(coordinates={}, rules=[{"path": "models", "required": True}]))
    backend = RecordingBackend()
    register_step_backend("derived-directory", backend)
    cab = reader().model_copy(update={"derived_reads": {"model": declaration}, "backend": "derived-directory", "harvest": [harvest]})
    result = _dispatch(cab, None, sandbox=True)
    assert result.success
    assert model.read_text() == "saved model"
    assert model.stat().st_ino == before.st_ino


def test_harvest_ancestor_of_staged_read_refuses_without_replacing_source(tmp_path, monkeypatch):
    from shinobi.backends.recording import RecordingBackend
    from shinobi.steps import register_step_backend
    from shinobi.steps.dispatch import _dispatch

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "ownership.json"))
    model = tmp_path / "models" / "model.fits"
    model.parent.mkdir()
    model.write_text("saved model")
    before = model.stat()
    declaration = DerivedRead(member="file", family=FamilySpec(coordinates={}, rules=[{"path": "models/model.fits", "required": True}]))
    backend = RecordingBackend()
    register_step_backend("derived-ancestor", backend)
    cab = reader().model_copy(update={"derived_reads": {"model": declaration}, "backend": "derived-ancestor", "harvest": ["models"]})
    with pytest.raises(WorkspaceOwnershipError, match="harvest ancestor"):
        _dispatch(cab, None, sandbox=True)
    assert backend.calls == []
    assert model.read_text() == "saved model" and model.stat().st_ino == before.st_ino


@pytest.mark.parametrize("scratch", ["models", "models/*"])
def test_scratch_parent_of_readonly_file_refuses_before_launch(tmp_path, monkeypatch, scratch):
    from shinobi.steps.dispatch import _dispatch

    monkeypatch.chdir(tmp_path)
    model = tmp_path / "models" / "model.fits"
    model.parent.mkdir()
    model.write_text("saved")
    declaration = DerivedRead(member="file", family=FamilySpec(coordinates={}, rules=[{"path": "models/model.fits", "required": True}]))
    cab = reader().model_copy(update={"derived_reads": {"model": declaration}, "scratch": [scratch]})
    with pytest.raises(WorkspaceOwnershipError, match="scratch writes overlap"):
        _dispatch(cab, None)
    assert model.read_text() == "saved"


def test_write_path_parent_reservation_remains_allowed(tmp_path):
    from shinobi.steps.schema import ParamMeta

    class DestinationInputs(BaseModel):
        destination: Path = Path("models")

    declaration = DerivedRead(member="file", family=FamilySpec(coordinates={}, rules=[{"path": "models/model.fits", "required": True}]))
    cab = reader().model_copy(update={"inputs_model": DestinationInputs, "field_meta": {"destination": ParamMeta(write_path=True)}, "derived_reads": {"model": declaration}})
    assert derived_reads(cab, DestinationInputs().model_dump(), tmp_path)[0].path == tmp_path / "models/model.fits"


def test_unrelated_scope_does_not_evaluate_output_default_factory(tmp_path):
    from pydantic import Field

    calls = []

    class Outputs(BaseModel):
        output: Path = Field(default_factory=lambda: calls.append(True) or Path("out"))

    cab = Cab(name="ordinary", command="true", inputs_model=Empty, outputs_model=Outputs)
    assert derived_reads(cab, {}, tmp_path) == ()
    assert calls == []
