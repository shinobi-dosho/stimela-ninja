"""Engine fixtures exercise contracts, without claiming scientific tool completeness."""

import itertools
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import BaseModel, create_model, model_validator

from shinobi import AxisSpec, DirectoryBundle, FamilySpec, MemberRule, ProductFamily, ProductMember
from shinobi.cache import CacheManifest, ProvenanceKey, compute_cache_key, product_contract, selected_key
from shinobi.exceptions import ParameterError
from shinobi.graph import RecipeNotOffloadableError
from shinobi.offload._codec import BundleError, TypeSpec, pack, unpack
from shinobi.offload.bundle import RecipeBundle, freeze_recipe
from shinobi.offload.records import AttemptRecord
from shinobi.offload.slurm import compile_slurm
from shinobi.products import FamilyPlan, bundle_inventory, validate_bundle_inventory
from shinobi.results import StepResult
from shinobi.sandbox import ProductCapture
from shinobi.steps.dispatch import _dispatch
from shinobi.steps.schema import Cab, OutputRef, ParamMeta, Recipe, StepRef


class Empty(BaseModel):
    pass


class Inputs(BaseModel):
    script: str
    prefix: str = "image"
    bands: int = 1
    times: int = 1
    pol: str = "I"


class Outputs(BaseModel):
    image: ProductFamily[Path]


def imaging_spec():
    return FamilySpec(
        coordinates={"time": "int", "frequency": ("int", "str"), "polarization": "str", "component": "str"},
        rules=(
            MemberRule(
                path="{prefix}{time}{frequency}{polarization}-image.fits",
                coordinates={"component": "real"},
                axes={
                    "time": AxisSpec(count_input="times", suffix="-t{index:04d}", singleton_suffix=""),
                    "frequency": AxisSpec(count_input="bands", suffix="-{index:04d}", singleton_suffix="", aggregate={"value": "mfs", "suffix": "-MFS", "only_multiple": True}),
                    "polarization": AxisSpec(
                        values_input="pol", tokens=("I", "Q", "U", "V", "XX", "XY", "YX", "YY"), split="csv_or_compact", uppercase=True, suffix="-{value}", singleton_suffix=""
                    ),
                },
                required=True,
            ),
        ),
    )


def make_cab(tmp_path, spec=None, member=Path, *, sandbox=False, cache=False):
    return Cab(
        name="fixture",
        command=f"{sys.executable} -c",
        inputs_model=Inputs,
        outputs_model=create_model("Families", image=(ProductFamily[member], ...)),
        field_meta={"script": ParamMeta(positional_head=True), "image": ParamMeta(family=spec or imaging_spec())},
        sandbox=sandbox,
        cache=cache,
        cache_dir=str(tmp_path / "cache"),
    )


@pytest.mark.parametrize("bands,times,pol", list(itertools.product((1, 2), (1, 2), ("I", "IQ"))))
def test_wsclean_eight_shapes(tmp_path, bands, times, pol):
    plan = FamilyPlan(imaging_spec(), Path, {"prefix": "deep", "bands": bands, "times": times, "pol": pol}, tmp_path)
    assert len(plan.candidates) == times * (bands + int(bands > 1)) * len(pol)
    for item in plan.candidates:
        assert item.coordinates["component"] == "real"
        assert item.coordinates["polarization"] in pol
        assert ("-t" in item.path.name) is (times > 1)
        if item.coordinates["frequency"] == "mfs":
            assert "-MFS" in item.path.name


@pytest.mark.parametrize("pol", ["Q", "q", "i,q", "iq"])
def test_polarization_normalization(tmp_path, pol):
    plan = FamilyPlan(imaging_spec(), Path, {"prefix": "deep", "bands": 1, "times": 1, "pol": pol}, tmp_path)
    assert [candidate.coordinates["polarization"] for candidate in plan.candidates] == (["Q"] if pol.upper() == "Q" else ["I", "Q"])
    if pol.upper() == "Q":
        assert plan.candidates[0].path.name == "deep-image.fits"


@pytest.mark.parametrize("sandbox,cache", list(itertools.product((False, True), repeat=2)))
def test_success_cache_disabled_and_sandbox_parity(tmp_path, monkeypatch, sandbox, cache):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / "sandboxes"))
    cab = make_cab(tmp_path, sandbox=sandbox, cache=cache)
    script = "from pathlib import Path;Path('image-image.fits').write_text('image')"
    first = _dispatch(cab, None, script=script, pol="Q")
    assert first.outputs.image.resolved
    assert first.outputs.image.select(time=0, frequency=0, polarization="Q", component="real") == Path("image-image.fits")
    second = _dispatch(cab, None, script=script, pol="Q")
    assert second.cached is cache
    assert first.outputs == second.outputs
    Path("image-image.fits").unlink()
    assert not _dispatch(cab, None, script=script, pol="Q").cached


def discovered_spec():
    return FamilySpec(
        root="cubelets",
        coordinates={"source": "int", "product": "str", "format": "str"},
        rules=(
            MemberRule(path="{prefix}_{source}_spec.txt", captures={"source": "int"}, coordinates={"product": "spectrum", "format": "text"}),
            MemberRule(path="{prefix}_{source}_cube.fits", captures={"source": "int"}, coordinates={"product": "cube", "format": "fits"}),
            MemberRule(path="{prefix}_{source}_cube.h5", captures={"source": "int"}, coordinates={"product": "cube", "format": "hdf5"}),
        ),
    )


@pytest.mark.parametrize("sandbox", [False, True])
def test_sparse_discovery_excludes_stale_and_restores_table(tmp_path, monkeypatch, sandbox):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / "sandboxes"))
    Path("cubelets").mkdir()
    Path("cubelets/image_99_spec.txt").write_text("stale")
    cab = make_cab(tmp_path, discovered_spec(), sandbox=sandbox, cache=True)
    script = (
        "from pathlib import Path;p=Path('cubelets');p.mkdir(exist_ok=True);[(p/name).write_text(name) for name in ['image_2_spec.txt','image_17_spec.txt','image_17_cube.h5']]"
    )
    first = _dispatch(cab, None, script=script)
    assert len(first.outputs.image.members) == 3
    assert first.outputs.image.select(source=17, product="cube").name.endswith(".h5")
    Path("cubelets/image_101_spec.txt").write_text("new but not produced")
    second = _dispatch(cab, None, script=script)
    assert second.cached and second.outputs == first.outputs
    Path("cubelets/image_17_cube.h5").unlink()
    assert not _dispatch(cab, None, script=script).cached
    assert Path("cubelets/image_99_spec.txt").exists()


def test_selection_type_identity_and_ambiguity():
    family = ProductFamily[Path](
        resolved=True, members=(ProductMember[Path](coordinates={"band": 1}, value=Path("a")), ProductMember[Path](coordinates={"band": "1"}, value=Path("b")))
    )
    assert family.select(band=1) == Path("a")
    assert family.select(band="1") == Path("b")
    for selection in ({}, {"band": True}, {"unknown": 1}, {"band": 7}):
        with pytest.raises(ValueError):
            family.select(**selection)
    for bad in ([(1, "a"), (1, "b")], [(1, "a"), (2, "a")]):
        with pytest.raises(ValueError):
            ProductFamily[Path](resolved=True, members=tuple(ProductMember[Path](coordinates={"band": band}, value=Path(path)) for band, path in bad))
    with pytest.raises(ValueError):
        ProductMember[Path](coordinates={"band": True}, value=Path("a"))


@pytest.mark.parametrize("count", [-1, True, 100001, "2"])
def test_invalid_counts_and_overlap(tmp_path, count):
    with pytest.raises(ValueError):
        FamilyPlan(imaging_spec(), Path, {"prefix": "image", "bands": count, "times": 1, "pol": "I"}, tmp_path)


@pytest.mark.parametrize("path", ["../escape", "s3://bucket/gains", Path("s3://bucket/gains"), "gains.qc::G"])
def test_remote_and_escape_rejected(tmp_path, path):
    with pytest.raises(ValueError):
        FamilyPlan(FamilySpec(coordinates={}, rules=(MemberRule(path=str(path)),)), Path, {}, tmp_path)


@pytest.mark.parametrize("sandbox", [False, True])
def test_bundle_children_and_empty_directories_cache(tmp_path, monkeypatch, sandbox):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / "sandboxes"))
    spec = FamilySpec(coordinates={"product": "str"}, rules=(MemberRule(path="gains.qc", coordinates={"product": "gains"}, required=True),))
    cab = make_cab(tmp_path, spec, DirectoryBundle, sandbox=sandbox, cache=True)
    script = "from pathlib import Path;p=Path('gains.qc');(p/'G'/'empty').mkdir(parents=True);(p/'G'/'chunk').write_text('gain')"
    first = _dispatch(cab, None, script=script)
    assert first.outputs.image.select(product="gains").path == Path("gains.qc")
    assert _dispatch(cab, None, script=script).cached
    Path("gains.qc/G/chunk").unlink()
    assert not _dispatch(cab, None, script=script).cached
    Path("gains.qc/G/empty").rmdir()
    Path("gains.qc/G/empty").write_text("wrong kind")
    assert not CacheManifest(tmp_path / "cache/manifest.json").check("fixture", first.cache_key, cab, {"script": script, "prefix": "image", "bands": 1, "times": 1, "pol": "I"})


def test_bundle_rejects_symlink_and_inventory_tampering(tmp_path):
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree/empty").mkdir()
    inventory = bundle_inventory(tmp_path / "tree")
    assert validate_bundle_inventory(inventory)
    (tmp_path / "tree/link").symlink_to(tmp_path / "tree/empty")
    with pytest.raises(ValueError):
        bundle_inventory(tmp_path / "tree")
    assert not validate_bundle_inventory({"root": str(tmp_path / "tree"), "entries": []})


@pytest.mark.parametrize("accept", [False, True])
def test_exact_existing_requires_explicit_policy(tmp_path, monkeypatch, accept):
    monkeypatch.chdir(tmp_path)
    Path("existing").write_text("old")
    spec = FamilySpec(coordinates={}, rules=(MemberRule(path="existing", accept_existing=accept),))
    cab = make_cab(tmp_path, spec)
    capture = ProductCapture(cab, {}, tmp_path)
    assert len(capture.resolve_families(True)["image"].members) == int(accept)


def test_relative_sandbox_existing_rejected_before_launch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("existing").write_text("old")
    spec = FamilySpec(coordinates={}, rules=(MemberRule(path="existing", accept_existing=True),))
    cab = make_cab(tmp_path, spec, sandbox=True)
    with pytest.raises(ValueError, match="relative sandbox"):
        _dispatch(cab, None, script="raise AssertionError('must not launch')")


def test_required_missing_and_nonzero_do_not_publish(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cab = make_cab(tmp_path, cache=True)
    with pytest.raises(ParameterError, match="required family member"):
        _dispatch(cab, None, script="pass")
    result = _dispatch(cab, None, script="raise SystemExit(8)")
    assert not result.success and not result.outputs.image.resolved
    assert not (tmp_path / "cache/manifest.json").exists()


def test_selection_changes_key_and_record_preserves_address(tmp_path):
    scope = make_cab(tmp_path)
    key = ProvenanceKey("producer", "image")
    first, second = selected_key(key, {"source": 2}), selected_key(key, {"source": 17})
    assert compute_cache_key(scope, None, {"prefix": "same"}, {"prefix": first}) != compute_cache_key(scope, None, {"prefix": "same"}, {"prefix": second})
    result = StepResult(name="renamed", returncode=0, inputs=Empty(), outputs=create_model("Selected", renamed=(Path, ...))(renamed=Path("a")), output_keys={"renamed": first})
    record = AttemptRecord.from_result(result, workflow_id=uuid4(), attempt_id=uuid4(), step_path="renamed", bundle_digest="digest")
    assert record.schema_version == 4
    restored = AttemptRecord.model_validate_json(record.model_dump_json()).result(type("Scope", (), {"inputs_model": Empty, "outputs_model": type(result.outputs)}))
    assert restored.provenance_key("renamed").producer_field == "image"
    assert restored.provenance_key("renamed").coordinates == {"source": 2}


def test_closed_codec_types_and_subclass_hooks():
    for annotation in (ProductFamily[Path], ProductFamily[DirectoryBundle], DirectoryBundle):
        assert TypeSpec.capture(annotation).restore() is annotation
    value = ProductFamily[Path](resolved=True, members=(ProductMember[Path](coordinates={"band": 1}, value=Path("a")),))
    assert unpack(pack(value)) == value

    class Foreign(ProductFamily[Path]):
        @model_validator(mode="after")
        def external(self):
            return self

    with pytest.raises(BundleError):
        TypeSpec.capture(Foreign)


def test_worker_freeze_restore_and_legacy_gate(tmp_path):
    cab = make_cab(tmp_path)
    recipe = Recipe(
        name="pipeline",
        inputs_model=Empty,
        outputs_model=create_model("Selected", image=(Path, ...)),
        steps=[StepRef(name="make", step=cab, params={"script": "pass"})],
        output_wiring={"image": OutputRef(step="make", field="image").select(time=0, frequency=0, polarization="I", component="real")},
    )
    from shinobi.config import AppConfig

    bundle = freeze_recipe(recipe, {}, workspace=tmp_path, config=AppConfig())
    assert bundle.schema_version == 2
    restored = RecipeBundle.model_validate_json(bundle.model_dump_json()).declaration()
    assert restored.steps[0].step.outputs_model.model_fields["image"].annotation is ProductFamily[Path]
    assert restored.output_wiring["image"].selection == recipe.output_wiring["image"].selection
    assert product_contract(restored.steps[0].step, {}) == product_contract(cab, {})
    with pytest.raises(RecipeNotOffloadableError, match="famil"):
        compile_slurm(recipe, {}, workdir=str(tmp_path))


def test_whole_family_paths_mounts_argv_and_boundary_hash(tmp_path, monkeypatch):
    from shinobi.backends.container import bind_dir_modes
    from shinobi.policies import build_argv
    from shinobi.sandbox import absolutize_path_inputs
    from shinobi.steps.schema import path_fields

    monkeypatch.chdir(tmp_path)
    Path("a").write_text("a")
    family = ProductFamily[Path](resolved=True, members=(ProductMember[Path](coordinates={"band": 0}, value=Path("a")),))
    model = create_model("FamilyConsumer", image=(ProductFamily[Path], ...))
    cab = Cab(name="consume", command="cat", inputs_model=model, outputs_model=Empty, field_meta={"image": ParamMeta(positional=True, repeat_as_tokens=True, writable=False)})
    assert path_fields(model) == {"image"}
    anchored = absolutize_path_inputs(cab, {"image": family}, tmp_path)
    assert anchored["image"].members[0].value == tmp_path / "a"
    assert build_argv(cab, anchored) == ["cat", str(tmp_path / "a")]
    assert dict(bind_dir_modes(cab, anchored, str(tmp_path)))[str(tmp_path / "a")] is False
    key = compute_cache_key(cab, None, {"image": family})
    Path("a").write_text("different")
    from shinobi.cache import invalidate_path_hashes

    invalidate_path_hashes()
    assert compute_cache_key(cab, None, {"image": family}) != key


def test_yaml_python_parity_and_closed_metadata():
    from shinobi.loaders.yaml_cab import loads

    document = """fixture:
  command: touch
  inputs:
    prefix: {dtype: str, default: image, write_path: true}
  outputs:
    image:
      dtype: ProductFamily[File]
      required: true
      family:
        coordinates: {product: str}
        rules:
          - path: "{prefix}.fits"
            coordinates: {product: image}
            required: true
"""
    cab = loads(document)["fixture"]
    assert cab.field_meta["image"].family == FamilySpec(coordinates={"product": "str"}, rules=(MemberRule(path="{prefix}.fits", coordinates={"product": "image"}, required=True),))
    for bad in (
        document.replace("coordinates: {product: str}", "callback: external.compute\n        coordinates: {product: str}"),
        document.replace("ProductFamily[File]", "ProductFamily[MSv2]"),
    ):
        with pytest.raises(ValueError):
            loads(bad)


def test_renamed_recipe_selection_provenance(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cab = make_cab(tmp_path, cache=True)
    ref = OutputRef(step="make", field="image").select(time=0, frequency=0, polarization="I", component="real")
    recipe = Recipe(
        name="renamed",
        inputs_model=Empty,
        outputs_model=create_model("Selected", renamed=(Path, ...)),
        steps=[StepRef(name="make", step=cab, params={"script": "from pathlib import Path;Path('image-image.fits').write_text('x')"})],
        output_wiring={"renamed": ref},
    )
    result = _dispatch(recipe, None)
    key = result.provenance_key("renamed")
    assert key.producer_field == "image" and key.coordinates == ref.selection
    assert result.outputs.renamed == Path("image-image.fits")


def test_detached_native_worker_family_result(tmp_path, monkeypatch):
    from shinobi.config import AppConfig
    from shinobi.offload.slurm import prepare_worker_slurm
    from shinobi.offload.worker import ExecutionPlan
    from tests.test_offload_worker import execute_step

    monkeypatch.chdir(tmp_path)
    cab = make_cab(tmp_path, cache=True)
    recipe = Recipe(
        name="workerfamily",
        inputs_model=Empty,
        outputs_model=Outputs,
        steps=[StepRef(name="make", step=cab, params={"script": "from pathlib import Path;Path('image-image.fits').write_text('x')"})],
        output_wiring={"image": OutputRef(step="make", field="image")},
    )
    bundle = freeze_recipe(recipe, {}, config=AppConfig(), workspace=tmp_path)
    workflow = prepare_worker_slurm(bundle, submission_root=tmp_path / "runs", worker_python=Path(sys.executable))
    plan = ExecutionPlan.model_validate_json((workflow.submission_dir / "execution.json").read_text())
    attempt = plan.attempts[0]
    assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
    record = AttemptRecord.read(
        workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json",
        workflow_id=plan.workflow_id,
        attempt_id=attempt.attempt_id,
        step_path=attempt.step_path,
        bundle_digest=bundle.digest,
    )
    result = record.result(cab)
    assert result.outputs.image.select(time=0, frequency=0, polarization="I", component="real") == Path("image-image.fits")


def test_psf_sharing_and_imaginary_components(tmp_path):
    spec = FamilySpec(
        coordinates={"product": "str", "polarization": "str", "component": "str"},
        rules=(
            MemberRule(path="psf.fits", coordinates={"product": "psf"}),
            MemberRule(
                path="{polarization}{component}.fits",
                coordinates={"product": "image"},
                axes={"polarization": AxisSpec(values=("XY", "YX")), "component": AxisSpec(values=("real", "imaginary"))},
            ),
        ),
    )
    plan = FamilyPlan(spec, Path, {}, tmp_path)
    assert len(plan.candidates) == 5
    assert "polarization" not in plan.candidates[0].coordinates
    assert {candidate.coordinates["component"] for candidate in plan.candidates[1:]} == {"real", "imaginary"}
    bad = spec.model_copy(update={"rules": (MemberRule(path="same.fits", axes={"polarization": AxisSpec(values=("I", "Q"))}),)})
    with pytest.raises(ValueError, match="duplicate"):
        FamilyPlan(bad, Path, {}, tmp_path)


def test_sparse_cycles_and_axis_shrink_do_not_adopt_stale(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = FamilySpec(coordinates={"cycle": "int"}, rules=(MemberRule(path="{prefix}.residual{cycle}.fits", captures={"cycle": "int"}),))
    cab = make_cab(tmp_path, spec)
    Path("image.residual09.fits").write_text("stale")
    result = _dispatch(cab, None, script="from pathlib import Path;[(Path('image.residual'+cycle+'.fits').write_text(cycle)) for cycle in ['00','02']]")
    assert {member.coordinates["cycle"] for member in result.outputs.image.members} == {0, 2}
    image_cab = make_cab(tmp_path, cache=True)
    two = "from pathlib import Path;[(Path('image-'+band+'-image.fits').write_text(band)) for band in ['0000','0001','MFS']]"
    assert len(_dispatch(image_cab, None, script=two, bands=2).outputs.image.members) == 3
    one = _dispatch(image_cab, None, script="from pathlib import Path;Path('image-image.fits').write_text('one')", bands=1)
    assert len(one.outputs.image.members) == 1 and one.outputs.image.members[0].coordinates["frequency"] == 0
    assert Path("image-0001-image.fits").exists()


@pytest.mark.parametrize("corruption", ["wrong_type", "unknown_coordinate", "different_path", "missing_required", "wrong_kind"])
def test_cache_rejects_malformed_saved_table(tmp_path, monkeypatch, corruption):
    from shinobi.cache import get_cache_manifest

    monkeypatch.chdir(tmp_path)
    cab = make_cab(tmp_path, cache=True)
    script = "from pathlib import Path;Path('image-image.fits').write_text('x')"
    result = _dispatch(cab, None, script=script)
    manifest = get_cache_manifest(str(tmp_path / "cache"))

    def mutate(data):
        table = data["fixture"]["outputs"]["image"]
        if corruption == "wrong_type":
            table["members"][0]["coordinates"]["time"] = "0"
        elif corruption == "unknown_coordinate":
            table["members"][0]["coordinates"]["invented"] = "value"
        elif corruption == "different_path":
            Path("other").write_text("other")
            table["members"][0]["value"] = "other"
        elif corruption == "missing_required":
            table["members"] = []

    manifest.update(mutate)
    if corruption == "wrong_kind":
        Path("image-image.fits").unlink()
        Path("image-image.fits").mkdir()
    prepared = {"script": script, "prefix": "image", "bands": 1, "times": 1, "pol": "I"}
    assert manifest.check("fixture", result.cache_key, cab, prepared) is None


def test_readonly_discovery_collision_and_frozen_refinement(tmp_path):
    from shinobi.ownership import validate_contained_execution
    from shinobi.steps.dispatch import _snapshot_product_inputs
    from shinobi.steps.schema import validate_declared_writes

    model = create_model("Protected", data=(Path, ...), prefix=(str, "image"))
    spec = discovered_spec().model_copy(update={"root": "{data}/cubelets"})
    cab = Cab(name="discovery", command="true", inputs_model=model, outputs_model=Outputs, field_meta={"image": ParamMeta(family=spec), "data": ParamMeta(writable=False)})
    values = {"data": tmp_path / "input", "prefix": "image"}
    with pytest.raises(ParameterError, match="writable"):
        validate_declared_writes(cab, values, tmp_path)
    safe = cab.model_copy(update={"field_meta": {"image": ParamMeta(family=discovered_spec())}})
    frozen = _snapshot_product_inputs(safe, {"data": tmp_path / "input", "prefix": "old"})
    # Refining only the basename preserves the claimed directory reservation.
    validate_contained_execution(safe, {"data": tmp_path / "input", "prefix": "new"}, workspace=tmp_path, dataset_resources=set(), planned_inputs=frozen)
    unsafe_spec = discovered_spec().model_copy(update={"root": "other"})
    unsafe = safe.model_copy(update={"field_meta": {"image": ParamMeta(family=unsafe_spec)}})
    from shinobi.exceptions import DatasetLifecycleUnavailableError

    with pytest.raises(DatasetLifecycleUnavailableError, match="planned family"):
        validate_contained_execution(unsafe, values, workspace=tmp_path, dataset_resources=set(), planned_inputs=frozen)


def test_accepted_absolute_bundle_sandbox_and_inventory_record(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "gains/empty").mkdir(parents=True)
    (tmp_path / "gains/chunk").write_text("gain")
    spec = FamilySpec(root=str(tmp_path), coordinates={}, rules=(MemberRule(path=str(tmp_path / "gains"), accept_existing=True, required=True),))
    cab = make_cab(tmp_path, spec, DirectoryBundle, sandbox=True, cache=True)
    result = _dispatch(cab, None, script="pass")
    assert result.bundle_inventories and len(result.bundle_inventories[0]["entries"]) == 3
    assert _dispatch(cab, None, script="pass").cached
    record = AttemptRecord.from_result(result, workflow_id=uuid4(), attempt_id=uuid4(), step_path="fixture", bundle_digest="digest")
    assert record.schema_version == 4
    assert record.result(cab).bundle_inventories == result.bundle_inventories
    Path("gains/chunk").unlink()
    assert not _dispatch(cab, None, script="pass").cached


def test_loop_rebinding_and_pass_through_preserve_selection(tmp_path, monkeypatch):
    from shinobi.steps.loops import passthrough_result

    monkeypatch.chdir(tmp_path)
    cab = make_cab(tmp_path, cache=True)
    selector = OutputRef(step="image", field="image").select(time=0, frequency=0, polarization="I", component="real")
    body_out = create_model("BodyOut", chosen=(Path, ...), converged=(Path | None, None))
    body = Recipe(
        name="body",
        inputs_model=Empty,
        outputs_model=body_out,
        steps=[StepRef(name="image", step=cab, params={"script": "pass"})],
        output_wiring={"chosen": selector, "converged": selector},
    )
    outer = Recipe(name="outer", inputs_model=Empty, outputs_model=Empty)
    loop = outer.add_loop("cycle", body, max_iter=2, until="converged", carry={})
    assert loop.outputs.chosen.selection == selector.selection
    family = ProductFamily[Path](resolved=True, members=(ProductMember[Path](coordinates={"source": 17}, value=Path("member")),))
    output = Outputs(image=family)
    key = selected_key(ProvenanceKey("parent", "original"), {"source": 17})
    prev = StepResult(name="previous", returncode=0, inputs=Empty(), outputs=output, output_keys={"image": key})
    skipped = passthrough_result(outer.steps[-1], prev, Empty())
    assert skipped.outputs is output and skipped.provenance_key("image").coordinates == {"source": 17}
    assert skipped.provenance_key("image").producer_field == "original"


def test_framework_models_do_not_bypass_any_or_default_restrictions():
    from shinobi.offload._codec import _has_model_instance, ModelSpec, pack_model

    value = ProductFamily[Path]()
    assert _has_model_instance(value) is True
    from typing import Any

    any_model = create_model("AnyInput", value=(Any, ...))
    with pytest.raises(BundleError, match="explicit model annotation"):
        pack_model(any_model(value=value))
    defaults = create_model("Defaults", value=(ProductFamily[Path], value))
    with pytest.raises(BundleError, match="model-instance defaults"):
        ModelSpec.capture(defaults)


def test_scalar_bundle_child_overlap_is_rejected_before_harvest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = FamilySpec(coordinates={}, rules=(MemberRule(path="gains.qc", required=True),))
    outputs = create_model("Overlap", image=(ProductFamily[DirectoryBundle], ...), child=(Path, Path("gains.qc/child")))
    cab = make_cab(tmp_path, spec, DirectoryBundle).model_copy(update={"outputs_model": outputs})
    with pytest.raises(ParameterError, match="overlaps family"):
        _dispatch(cab, None, script="from pathlib import Path;p=Path('gains.qc');p.mkdir(exist_ok=True);(p/'child').write_text('x')")


def test_family_metadata_requires_declared_output(tmp_path):
    with pytest.raises(ValueError, match="declared family output"):
        Cab(name="invalid", command="true", inputs_model=Inputs, outputs_model=Empty, field_meta={"prefix": ParamMeta(family=imaging_spec())})


@pytest.mark.parametrize("sandbox", [False, True])
@pytest.mark.parametrize("redirect", ["ancestor", "root", "protected_input"])
def test_run_created_ancestor_redirect_cannot_publish(tmp_path, monkeypatch, sandbox, redirect):
    from shinobi.cache import get_cache_manifest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / "sandboxes"))
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    protected = tmp_path / "protected"
    protected.mkdir()
    target = protected if redirect == "protected_input" else outside
    root, member, link = ("products", "item.fits", "products") if redirect == "root" else (".", "nested/item.fits", "nested")
    spec = FamilySpec(root=root, coordinates={}, rules=(MemberRule(path=member, required=True),))
    cab = make_cab(tmp_path, spec, sandbox=sandbox, cache=True)
    model = create_model("ProtectedRunInputs", __base__=Inputs, data=(Path, ...))
    cab = cab.model_copy(update={"inputs_model": model})
    actual_member = f"{root}/{member}" if root != "." else member
    script = f"from pathlib import Path;p=Path({link!r});p.rmdir();p.symlink_to({str(target)!r},target_is_directory=True);Path({actual_member!r}).write_text('backend-write')"
    with pytest.raises(ParameterError, match="family (root changed|member escapes|output overlaps)"):
        _dispatch(cab, None, script=script, data=protected)
    assert get_cache_manifest(str(tmp_path / "cache")).entry("fixture") is None
    # The backend itself wrote the redirected source; the engine must not
    # publish it as successful evidence or harvest it to a new destination.
    assert (target / "item.fits").read_text() == "backend-write"
    if sandbox:
        assert not (tmp_path / member).exists()


def test_run_created_workspace_destination_redirect_cannot_harvest(tmp_path, monkeypatch):
    from shinobi.cache import get_cache_manifest

    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (outside / "item.fits").write_text("protected destination")
    spec = FamilySpec(coordinates={}, rules=(MemberRule(path="nested/item.fits", required=True),))
    cab = make_cab(tmp_path, spec, sandbox=True, cache=True)
    script = (
        f"from pathlib import Path;Path('nested/item.fits').write_text('sandbox product');Path({str(tmp_path / 'nested')!r}).symlink_to({str(outside)!r},target_is_directory=True)"
    )
    with pytest.raises(ParameterError, match="family member escapes"):
        _dispatch(cab, None, script=script)
    assert (outside / "item.fits").read_text() == "protected destination"
    assert get_cache_manifest(str(tmp_path / "cache")).entry("fixture") is None


@pytest.mark.parametrize("input_kind", ["readonly_alias", "writable_directory", "bundle", "file_family", "bundle_family"])
@pytest.mark.parametrize("sandbox", [False, True])
def test_discovery_cannot_claim_input_namespace_before_backend(tmp_path, monkeypatch, input_kind, sandbox):
    from shinobi.steps.dispatch import register_step_backend

    monkeypatch.chdir(tmp_path)
    counter = []

    class ForbiddenBackend:
        def run(self, *args, **kwargs):
            counter.append(1)
            raise AssertionError("protected input discovery must fail before tool launch")

    backend_name = f"family-input-namespace-{input_kind}-{sandbox}"
    register_step_backend(backend_name, ForbiddenBackend())
    protected = tmp_path / "data"
    if input_kind in ("readonly_alias", "file_family"):
        protected.write_text("input bytes")
        (tmp_path / "input_data.fits").symlink_to(protected)
        spec = FamilySpec(coordinates={"source": "str"}, rules=(MemberRule(path="input_{source}.fits", captures={"source": "str"}),))
        value = protected
        annotation = Path
        if input_kind == "file_family":
            value = ProductFamily[Path](resolved=True, members=(ProductMember[Path](coordinates={}, value=protected),))
            annotation = ProductFamily[Path]
    else:
        protected.mkdir()
        (protected / "input").write_text("input bytes")
        # This member does not exist: the namespace proof must protect future
        # files, rather than scanning today's directory and declaring it safe.
        spec = FamilySpec(root="data", coordinates={"source": "str"}, rules=(MemberRule(path="item_{source}.fits", captures={"source": "str"}),))
        value, annotation = protected, Path
        if input_kind == "bundle":
            value, annotation = DirectoryBundle(path=protected), DirectoryBundle
        elif input_kind == "bundle_family":
            value = ProductFamily[DirectoryBundle](resolved=True, members=(ProductMember[DirectoryBundle](coordinates={}, value=DirectoryBundle(path=protected)),))
            annotation = ProductFamily[DirectoryBundle]
    model = create_model("ProtectedNamespaceInput", data=(annotation, ...))
    cab = Cab(
        name="namespace",
        command="true",
        inputs_model=model,
        outputs_model=Outputs,
        backend=backend_name,
        field_meta={"data": ParamMeta(writable=input_kind != "readonly_alias"), "image": ParamMeta(family=spec)},
        sandbox=sandbox,
    )
    with pytest.raises(ParameterError, match="family discovery overlaps"):
        _dispatch(cab, None, data=value)
    assert counter == []
    assert (protected if protected.is_file() else protected / "input").read_text() == "input bytes"
    if protected.is_dir():
        assert not (protected / "item_new.fits").exists()


def test_flat_discovery_prefix_stays_disjoint_from_ms_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "observation.ms").mkdir()
    (tmp_path / "observation.ms/table.dat").write_text("measurement set")
    model = create_model("MSRead", __base__=Inputs, data=(Path, ...))
    spec = FamilySpec(coordinates={"source": "int"}, rules=(MemberRule(path="image_{source}.fits", captures={"source": "int"}),))
    cab = make_cab(tmp_path, spec).model_copy(
        update={"inputs_model": model, "field_meta": {"data": ParamMeta(writable=False), "script": ParamMeta(positional_head=True), "image": ParamMeta(family=spec)}}
    )
    result = _dispatch(cab, None, script="from pathlib import Path;Path('image_17.fits').write_text('image')", data=tmp_path / "observation.ms")
    assert result.outputs.image.select(source=17) == Path("image_17.fits")
    assert (tmp_path / "observation.ms/table.dat").read_text() == "measurement set"


@pytest.mark.parametrize("text", ["A" * 127 + "B", "A" * 128])
def test_compact_tokenization_has_bounded_suffix_work(text):
    class CountedText(str):
        calls = 0

        def startswith(self, *args):
            self.calls += 1
            return super().startswith(*args)

    value = CountedText(text)
    axis = AxisSpec(values_input="pol", tokens=("A", "AA"), split="csv_or_compact")
    with pytest.raises(ValueError, match="ambiguous or unknown"):
        axis.expand({"pol": value}, 10000)
    # Each (suffix, token) pair is considered at most once; this bound does
    # not depend on machine speed or how many combinatorial parses exist.
    assert value.calls <= len(value) * len(axis.tokens)


def test_long_unique_compact_tokenization():
    token = "A" * 127
    axis = AxisSpec(values_input="pol", tokens=(token, "B", "AA"), split="csv_or_compact")
    assert axis.expand({"pol": token + "B"}, 10000) == [(token, token), ("B", "B")]


def test_exact_rule_union_limit_rejected_before_backend(tmp_path, monkeypatch):
    from shinobi.steps.dispatch import register_step_backend

    monkeypatch.chdir(tmp_path)
    protected = tmp_path / "input"
    protected.write_text("input bytes")
    calls = []

    class ForbiddenBackend:
        def run(self, *args, **kwargs):
            calls.append(1)
            raise AssertionError("excessive family must fail before launch")

    register_step_backend("exact-union-limit", ForbiddenBackend())
    spec = FamilySpec(
        max_members=1,
        coordinates={"product": "str"},
        rules=(MemberRule(path="a", coordinates={"product": "a"}), MemberRule(path="b", coordinates={"product": "b"})),
    )
    cab = make_cab(tmp_path, spec).model_copy(update={"backend": "exact-union-limit", "inputs_model": create_model("LimitInput", __base__=Inputs, data=(Path, ...))})
    with pytest.raises(ValueError, match="member limit exceeded"):
        _dispatch(cab, None, script="pass", data=protected)
    assert calls == []
    assert protected.read_text() == "input bytes"
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_zero_axis_product_does_not_exceed_limit(tmp_path):
    spec = FamilySpec(
        max_members=1,
        coordinates={"band": ("int", "str"), "time": "int"},
        rules=(
            MemberRule(
                path="{band}-{time}",
                axes={"band": AxisSpec(values=(1,), aggregate={"value": "mfs", "suffix": "mfs"}), "time": AxisSpec(count_input="times")},
            ),
        ),
    )
    assert FamilyPlan(spec, Path, {"times": 0}, tmp_path).candidates == []


def test_selected_bundle_list_runs_with_path_argv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = FamilySpec(coordinates={"term": "str"}, rules=(MemberRule(path="gains-{term}", axes={"term": AxisSpec(values=("K", "G"))}, required=True),))
    producer = make_cab(tmp_path, spec, DirectoryBundle)
    model = create_model("BundleListConsumer", script=(str, ...), dirs=(list[DirectoryBundle], ...))
    consumer = Cab(
        name="consume",
        command=f"{sys.executable} -c",
        inputs_model=model,
        outputs_model=Empty,
        field_meta={"script": ParamMeta(positional_head=True), "dirs": ParamMeta(positional=True, repeat_as_tokens=True, writable=False)},
    )
    recipe = Recipe(name="bundle-list", inputs_model=Empty, outputs_model=Empty)
    recipe.add_step("produce", producer, script="from pathlib import Path;Path('gains-K').mkdir();Path('gains-G').mkdir()")
    recipe.add_step(
        "consume",
        consumer,
        script="import sys;from pathlib import Path;assert [Path(v).name for v in sys.argv[1:]]==['gains-K','gains-G'];assert all(Path(v).is_dir() for v in sys.argv[1:]);Path('consumed').write_text('ok')",
        dirs=[OutputRef(step="produce", field="image").select(term="K"), OutputRef(step="produce", field="image").select(term="G")],
    )
    assert _dispatch(recipe, None).success
    assert Path("consumed").read_text() == "ok"


@pytest.mark.parametrize("member", [DirectoryBundle, ProductFamily[Path], ProductFamily[DirectoryBundle]])
@pytest.mark.parametrize("policy", ["joined", "brackets", "repeat", "tokens", "positional"])
def test_framework_list_argv_preserves_policies(tmp_path, member, policy):
    from shinobi.policies import build_argv
    from shinobi.steps.schema import Policies

    paths = [tmp_path / "a", tmp_path / "b"]
    values = []
    for index, path in enumerate(paths):
        if member is DirectoryBundle:
            values.append(DirectoryBundle(path=path))
        else:
            item_type = Path if member is ProductFamily[Path] else DirectoryBundle
            item = path if item_type is Path else DirectoryBundle(path=path)
            values.append(member(resolved=True, members=(ProductMember[item_type](coordinates={"band": index}, value=item),)))
    model = create_model("FrameworkList", dirs=(list[member], ...))
    meta = ParamMeta(repeat_as_tokens=policy in ("tokens", "positional"), positional=policy == "positional")
    policies = Policies(repeat="[]") if policy == "brackets" else Policies(repeat_list=policy == "repeat")
    cab = Cab(name="consume", command="tool", inputs_model=model, outputs_model=Empty, field_meta={"dirs": meta}, policies=policies)
    strings = list(map(str, paths))
    tail = {
        "joined": ["--dirs", ",".join(strings)],
        "brackets": ["--dirs", "[" + ",".join(strings) + "]"],
        "repeat": ["--dirs", strings[0], "--dirs", strings[1]],
        "tokens": ["--dirs", *strings],
        "positional": strings,
    }[policy]
    assert build_argv(cab, {"dirs": values}) == ["tool", *tail]
    if member is not DirectoryBundle:
        with pytest.raises(ValueError, match="unresolved"):
            build_argv(cab, {"dirs": [member()]})


@pytest.mark.parametrize("member", [DirectoryBundle, ProductFamily[Path], ProductFamily[DirectoryBundle]])
def test_framework_list_cache_boundary_and_producer_identity(tmp_path, member):
    from shinobi.cache import invalidate_path_hashes

    files, values = [], []
    for index in range(2):
        path = tmp_path / str(index)
        if member is ProductFamily[Path]:
            file = path
            item_type, item = Path, path
        else:
            path.mkdir()
            file = path / "table"
            item_type, item = DirectoryBundle, DirectoryBundle(path=path)
        file.write_text("before")
        files.append(file)
        values.append(item if member is DirectoryBundle else member(resolved=True, members=(ProductMember[item_type](coordinates={"band": index}, value=item),)))
    model = create_model("CacheFrameworkList", dirs=(list[member], ...))
    cab = Cab(name="consume", command="tool", inputs_model=model, outputs_model=Empty)
    producer = selected_key(ProvenanceKey("upstream", "image"), {"band": 0})
    keys = {"dirs": [producer, None]}
    boundary = compute_cache_key(cab, None, {"dirs": values})
    partial = compute_cache_key(cab, None, {"dirs": values}, keys)
    fully_wired = compute_cache_key(cab, None, {"dirs": values}, {"dirs": [producer, producer]})
    files[0].write_text("changed producer bytes")
    invalidate_path_hashes()
    assert compute_cache_key(cab, None, {"dirs": values}) != boundary
    assert compute_cache_key(cab, None, {"dirs": values}, keys) == partial
    assert compute_cache_key(cab, None, {"dirs": values}, {"dirs": [producer, producer]}) == fully_wired
    files[1].write_text("changed boundary bytes")
    invalidate_path_hashes()
    assert compute_cache_key(cab, None, {"dirs": values}, keys) != partial
    assert compute_cache_key(cab, None, {"dirs": values}, {"dirs": [producer, producer]}) == fully_wired
    other_selection = selected_key(ProvenanceKey("upstream", "image"), {"band": 1})
    assert compute_cache_key(cab, None, {"dirs": values}, {"dirs": [other_selection, producer]}) != fully_wired


@pytest.mark.parametrize("many", [False, True])
@pytest.mark.parametrize("wiring", ["boundary", "wired", "partial"])
def test_ordinary_path_cache_key_retains_historical_vector(tmp_path, many, wiring):
    import hashlib
    import json
    from shinobi.cache import _hash_path, _identity

    paths = [tmp_path / "a", tmp_path / "b"]
    for path in paths:
        path.write_text(path.name)
    value = paths if many else paths[0]
    model = create_model("OrdinaryPaths", data=(list[Path] if many else Path, ...))
    cab = Cab(name="ordinary", command="tool", inputs_model=model, outputs_model=Empty)
    key = ProvenanceKey("upstream", "data")
    keys = {} if wiring == "boundary" else {"data": [key, None if wiring == "partial" else key] if many else key}
    parts = [cab.image, _identity(cab, None)]
    if not keys:
        parts.append(["data", repr(value), [_hash_path(path) for path in (paths if many else [paths[0]])]])
    elif many and wiring == "partial":
        parts.append(["data", repr(value), [None, _hash_path(paths[1])]])
    else:
        parts.append(["data", repr(value)])
    if keys:
        parts.append(["__upstream__", [["data", keys["data"]]]])
    historical = hashlib.sha256(json.dumps(parts, default=str, sort_keys=True).encode()).hexdigest()
    assert compute_cache_key(cab, None, {"data": value}, keys) == historical


@pytest.mark.parametrize("initial,actual", [(1, 2), (0, 1)])
def test_contained_family_growth_refuses_before_backend_or_clear(tmp_path, monkeypatch, initial, actual):
    from shinobi.exceptions import DatasetLifecycleUnavailableError
    from shinobi.steps.dispatch import _run_cab, register_step_backend

    monkeypatch.chdir(tmp_path)
    protected = tmp_path / "obs.ms"
    protected.mkdir()
    (protected / "table.dat").write_text("input bytes")
    (tmp_path / "dir0").mkdir()
    (tmp_path / "dir0/out").write_text("existing product")
    calls = []

    class ForbiddenBackend:
        def run(self, *args, **kwargs):
            calls.append(1)
            raise AssertionError("unclaimed family parent must fail before launch")

    register_step_backend("family-growth-guard", ForbiddenBackend())
    spec = FamilySpec(coordinates={"band": "int"}, rules=(MemberRule(path="{band}/out", axes={"band": AxisSpec(count_input="bands", suffix="dir{index}")}),))
    cab = make_cab(tmp_path, spec)
    values = {"script": "pass", "prefix": "image", "bands": actual, "times": 1, "pol": "I"}
    with pytest.raises(DatasetLifecycleUnavailableError, match="family 'image' pattern changed its planned product directory"):
        _run_cab(cab, values, "family-growth-guard", dataset_resources={protected}, planned_inputs={**values, "bands": initial})
    assert calls == []
    assert (tmp_path / "dir0/out").read_text() == "existing product"
    assert (protected / "table.dat").read_text() == "input bytes"
    assert not (tmp_path / "dir1").exists()


def test_contained_family_activation_cannot_borrow_other_source_parent(tmp_path):
    from shinobi.exceptions import DatasetLifecycleUnavailableError
    from shinobi.ownership import validate_contained_execution
    from shinobi.steps.dispatch import _snapshot_product_inputs

    spec = FamilySpec(coordinates={}, rules=(MemberRule(path="new/out", when={"pol": ("Q",)}),))
    other = FamilySpec(coordinates={}, rules=(MemberRule(path="new/other"),))
    cab = make_cab(tmp_path, spec).model_copy(
        update={
            "outputs_model": create_model("OtherFamily", image=(ProductFamily[Path], ...), other=(ProductFamily[Path], ...)),
            "field_meta": {"image": ParamMeta(family=spec), "other": ParamMeta(family=other)},
            "harvest": ["new/harvest-*"],
        }
    )
    frozen = _snapshot_product_inputs(cab, {"pol": "I"})
    with pytest.raises(DatasetLifecycleUnavailableError, match="family 'image' pattern changed its planned product directory"):
        validate_contained_execution(cab, {"pol": "Q"}, workspace=tmp_path, dataset_resources=set(), planned_inputs=frozen)


def test_contained_family_when_switch_cannot_change_parent(tmp_path):
    from shinobi.exceptions import DatasetLifecycleUnavailableError
    from shinobi.ownership import validate_contained_execution
    from shinobi.steps.dispatch import _snapshot_product_inputs

    spec = FamilySpec(coordinates={}, rules=(MemberRule(path="old/out", when={"pol": ("I",)}), MemberRule(path="new/out", when={"pol": ("Q",)})))
    cab = make_cab(tmp_path, spec)
    with pytest.raises(DatasetLifecycleUnavailableError, match="directory reservation"):
        validate_contained_execution(cab, {"pol": "Q"}, workspace=tmp_path, dataset_resources=set(), planned_inputs=_snapshot_product_inputs(cab, {"pol": "I"}))


@pytest.mark.parametrize("initial,actual,path", [(1, 3, "out-{band}"), (3, 1, "{band}/out"), (3, 0, "{band}/out")])
def test_contained_family_safe_count_refinement_preserves_other_sources(tmp_path, initial, actual, path):
    from shinobi.ownership import validate_contained_execution
    from shinobi.steps.dispatch import _snapshot_product_inputs

    spec = FamilySpec(coordinates={"band": "int"}, rules=(MemberRule(path=path, axes={"band": AxisSpec(count_input="bands", suffix="dir{index}")}),))
    other = FamilySpec(coordinates={}, rules=(MemberRule(path="unrelated/result"),))
    cab = make_cab(tmp_path, spec).model_copy(
        update={
            "outputs_model": create_model("OtherFamily", image=(ProductFamily[Path], ...), other=(ProductFamily[Path], ...)),
            "field_meta": {"image": ParamMeta(family=spec), "other": ParamMeta(family=other)},
            "harvest": ["harvest/result-*"],
            "scratch": ["scratch/result-*"],
        }
    )
    validate_contained_execution(cab, {"bands": actual}, workspace=tmp_path, dataset_resources=set(), planned_inputs=_snapshot_product_inputs(cab, {"bands": initial}))


def test_contained_family_reordered_parents_are_still_claimed(tmp_path):
    from shinobi.ownership import validate_contained_execution
    from shinobi.steps.dispatch import _snapshot_product_inputs

    spec = FamilySpec(coordinates={"pol": "str"}, rules=(MemberRule(path="{pol}/out", axes={"pol": AxisSpec(values_input="pol", tokens=("I", "Q"), split="csv_or_compact")}),))
    cab = make_cab(tmp_path, spec)
    validate_contained_execution(cab, {"pol": "QI"}, workspace=tmp_path, dataset_resources=set(), planned_inputs=_snapshot_product_inputs(cab, {"pol": "IQ"}))


def test_unresolved_frozen_family_cannot_gain_runtime_parent(tmp_path):
    from shinobi.exceptions import DatasetLifecycleUnavailableError
    from shinobi.ownership import validate_contained_execution
    from shinobi.steps.dispatch import _snapshot_product_inputs

    spec = FamilySpec(coordinates={"band": "int"}, rules=(MemberRule(path="{band}/out", axes={"band": AxisSpec(count_input="bands", suffix="dir{index}")}),))
    cab = make_cab(tmp_path, spec)
    with pytest.raises(DatasetLifecycleUnavailableError, match="no frozen parents"):
        validate_contained_execution(cab, {"bands": 1}, workspace=tmp_path, dataset_resources=set(), planned_inputs=_snapshot_product_inputs(cab, {}))
