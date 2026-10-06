"""Successful products are concrete files, independently of public filenames."""

import sys
from pathlib import Path
from typing import Union

import pytest
from pydantic import BaseModel, Field, create_model

from shinobi import pystep
from shinobi.cache import CacheManifest, compute_cache_key, get_cache_manifest, product_contract
from shinobi.results import StepResult
from shinobi.sandbox import ProductCapture
from shinobi.steps.dispatch import _dispatch
from shinobi.steps.schema import Cab, ParamMeta, Scope


class Empty(BaseModel):
    pass


class FamilyInputs(BaseModel):
    script: str
    prefix: str = "image"


class FamilyOutputs(BaseModel):
    mfs: Path | None = None


def family_cab(tmp_path, *, sandbox=False, inputs_model=FamilyInputs):
    return Cab(
        name="family",
        command=f"{sys.executable} -c",
        inputs_model=inputs_model,
        outputs_model=FamilyOutputs,
        cache=True,
        cache_dir=str(tmp_path / "cache"),
        sandbox=sandbox,
        field_meta={"script": ParamMeta(positional_head=True), "prefix": ParamMeta(positional=True), "mfs": ParamMeta(implicit="{prefix}-MFS-image.fits")},
        harvest=["{prefix}-*.fits"],
    )


FAMILY_SCRIPT = "from pathlib import Path;import sys;[(Path(sys.argv[1]+'-'+member+'.fits').write_text(member)) for member in ['0000-I-image','0001-Q-image']]"


@pytest.mark.parametrize("sandbox", [False, True])
@pytest.mark.parametrize("prefix_kind", ["relative", "absolute-string", "path"])
def test_absent_optional_mfs_family_hit_and_member_deletion(tmp_path, monkeypatch, sandbox, prefix_kind):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / "sandboxes"))
    model = create_model("PathPrefix", script=(str, ...), prefix=(Path, Path("image"))) if prefix_kind == "path" else FamilyInputs
    cab = family_cab(tmp_path, sandbox=sandbox, inputs_model=model)
    prefix = str(tmp_path / "image") if prefix_kind == "absolute-string" else "image"
    first = _dispatch(cab, None, prefix=prefix, script=FAMILY_SCRIPT)
    second = _dispatch(cab, None, prefix=prefix, script=FAMILY_SCRIPT)
    assert second.cached
    assert first.outputs.mfs == second.outputs.mfs == (Path("image-MFS-image.fits") if sandbox else Path(prefix + "-MFS-image.fits"))
    assert not first.outputs.mfs.exists()
    entry = get_cache_manifest(str(tmp_path / "cache")).entry("family")
    assert entry["products"] == [str(tmp_path / "image-0000-I-image.fits"), str(tmp_path / "image-0001-Q-image.fits")]
    (tmp_path / "image-0001-Q-image.fits").unlink()
    assert not _dispatch(cab, None, prefix=prefix, script=FAMILY_SCRIPT).cached


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("multiple", [False, True])
def test_optional_missing_scalar_and_list_members(tmp_path, monkeypatch, required, multiple):
    monkeypatch.chdir(tmp_path)
    annotation = list[Path] if multiple else Path
    model = create_model("Products", product=(annotation, ... if required else None))
    scope = Scope(inputs_model=Empty, name="manual", outputs_model=model, cache=True, cache_dir=str(tmp_path / "cache"))
    calls = []

    def run(ctx):
        calls.append(1)
        Path("made").write_text("ok")
        outputs = model(product=[Path("made"), Path("absent")] if multiple else Path("absent"))
        return StepResult(name="manual", returncode=0, outputs=outputs, inputs=ctx.inputs)

    _dispatch(scope, run)
    assert _dispatch(scope, run).cached is (not required)
    if not required and multiple:
        Path("made").unlink()
        assert not _dispatch(scope, run).cached
    assert len(calls) >= (2 if required else 1)


@pytest.mark.parametrize("annotation,value", [(Path | None, None), (list[Path], [])])
def test_required_nullable_and_empty_paths_contribute_none(tmp_path, monkeypatch, annotation, value):
    monkeypatch.chdir(tmp_path)
    model = create_model("Products", product=(annotation, ...))
    scope = Scope(inputs_model=Empty, name="empty", outputs_model=model, cache=True, cache_dir=str(tmp_path / "cache"))

    def run(ctx):
        return StepResult(name="empty", returncode=0, outputs=model(product=value), inputs=ctx.inputs)

    _dispatch(scope, run)
    assert _dispatch(scope, run).cached


def test_nested_harvest_child_survives_parent_move_and_is_checked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / "sandboxes"))
    model = create_model("Directory", directory=(Path, Path("tree")))
    cab = family_cab(tmp_path, sandbox=True).model_copy(update={"outputs_model": model, "harvest": ["tree/*.fits"]})
    script = "from pathlib import Path;p=Path('tree');p.mkdir(exist_ok=True);(p/'child.fits').write_text('x')"
    _dispatch(cab, None, script=script)
    assert _dispatch(cab, None, script=script).cached
    (tmp_path / "tree/child.fits").unlink()
    assert not _dispatch(cab, None, script=script).cached


def test_direct_harvest_excludes_unchanged_and_observes_directory_child_overwrite(tmp_path):
    scope = Scope(inputs_model=Empty, outputs_model=Empty, name="capture", harvest=["*.fits", "tree"], scratch=["scratch/*"])
    (tmp_path / "leftover.fits").write_text("old")
    (tmp_path / "rewritten.fits").write_text("old")
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree/child").write_text("old")
    capture = ProductCapture(scope, {}, tmp_path)
    (tmp_path / "rewritten.fits").write_text("rewritten")
    (tmp_path / "tree/child").write_text("rewritten")
    assert capture.finish(scope.outputs_model()) == [str(tmp_path / "rewritten.fits"), str(tmp_path / "tree")]


def test_sandbox_destination_leftover_is_not_a_produced_optional_output(tmp_path):
    model = create_model("Products", product=(Path, Path("leftover")))
    scope = Scope(inputs_model=Empty, name="capture", outputs_model=model)
    (tmp_path / "leftover").write_text("previous")
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    capture = ProductCapture(scope, {}, tmp_path, sandbox)
    assert capture.finish(model()) == []


@pytest.mark.parametrize("change", ["harvest", "implicit", "required", "shape", "default", "workspace"])
def test_declaration_changes_miss_without_changing_scientific_key(tmp_path, monkeypatch, change):
    monkeypatch.chdir(tmp_path)
    scope = family_cab(tmp_path)
    prepared = {"prefix": "image", "script": FAMILY_SCRIPT}
    original_key = compute_cache_key(scope, None, prepared)
    manifest = CacheManifest(tmp_path / "manifest.json")
    result = StepResult(name="family", returncode=0, inputs=FamilyInputs(**prepared), outputs=FamilyOutputs())
    manifest.record("family", original_key, result, product_contract=product_contract(scope, prepared), products=[])
    assert manifest.check("family", original_key, scope, prepared)
    if change == "harvest":
        scope.harvest = ["*.txt"]
    elif change == "implicit":
        scope.field_meta["mfs"] = ParamMeta(implicit="{prefix}-other.fits")
    elif change == "default":
        scope.outputs_model = create_model("ChangedDefault", mfs=(Path | None, Path("different.fits")))
    elif change in {"required", "shape"}:
        scope.outputs_model = create_model("Changed", mfs=(Path | None if change == "required" else list[Path], ... if change == "required" else None))
    else:
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(other)
    assert compute_cache_key(scope, None, prepared) == original_key
    assert manifest.check("family", original_key, scope, prepared) is None


def test_union_spelling_and_set_order_have_stable_contract(tmp_path):
    one = Scope(name="union", inputs_model=Empty, outputs_model=create_model("One", product=(Path | None, None)))
    two = Scope(name="union", inputs_model=Empty, outputs_model=create_model("Two", product=(Union[Path, None], None)))
    assert product_contract(one, {}, tmp_path) == product_contract(two, {}, tmp_path)


@pytest.mark.parametrize(
    "metadata",
    [{}, {"products": []}, {"product_contract": {}}, {"product_contract": None, "products": []}, {"products": [None]}, {"products": ["relative"]}, {"products": ["/\x00"]}],
)
def test_legacy_and_corrupt_product_metadata_safely_miss(tmp_path, monkeypatch, metadata):
    monkeypatch.chdir(tmp_path)
    scope = family_cab(tmp_path)
    manifest = CacheManifest(tmp_path / "manifest.json")
    manifest.record("family", "key", StepResult(name="family", returncode=0, outputs=FamilyOutputs(), inputs=Empty()))
    entry = manifest.entry("family")
    entry.update(metadata)
    if "products" in metadata and "product_contract" not in metadata and metadata != {"products": []}:
        entry["product_contract"] = product_contract(scope, {})
    manifest.update(lambda data: data.update(family=entry))
    assert manifest.check("family", "key", scope, {}) is None


def test_callback_deterministic_run_override_hits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cab = family_cab(tmp_path)

    def run(ctx):
        return ctx.run(prefix="safe")

    first = _dispatch(cab, run, prefix="original", script=FAMILY_SCRIPT)
    second = _dispatch(cab, run, prefix="original", script=FAMILY_SCRIPT)
    assert second.cached
    assert first.outputs.mfs == second.outputs.mfs == Path("safe-MFS-image.fits")


def test_inprocess_pystep_harvest_inventory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @pystep(cache=True, cache_dir=str(tmp_path / "cache"), harvest=["{prefix}-*.fits"])
    def family(prefix: str) -> FamilyOutputs:
        Path(prefix + "-channel.fits").write_text("x")
        return FamilyOutputs(mfs=Path(prefix + "-MFS.fits"))

    assert not family(prefix="image").cached
    assert family(prefix="image").cached
    Path("image-channel.fits").unlink()
    assert not family(prefix="image").cached


@pytest.mark.parametrize("sandbox", [False, True])
def test_unused_output_parents_and_scratch_are_not_products(tmp_path, monkeypatch, sandbox):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / "sandboxes"))
    model = create_model("UnusedDirectory", directory=(Path | None, Path("unused")))
    cab = family_cab(tmp_path, sandbox=sandbox).model_copy(update={"outputs_model": model, "harvest": ["unused/*.fits"], "scratch": ["scratch/*"]})
    script = "from pathlib import Path;Path('scratch/cache').write_text('junk')"
    result = _dispatch(cab, None, script=script)
    assert result.success
    assert get_cache_manifest(str(tmp_path / "cache")).entry("family")["products"] == []
    assert _dispatch(cab, None, script=script).cached


@pytest.mark.parametrize("absolute", [False, True])
def test_direct_parent_relative_patterns_preserve_lexical_product_paths(tmp_path, absolute):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pattern = str(workspace / "../*.fits") if absolute else "../*.fits"
    scope = Scope(name="parent", inputs_model=Empty, outputs_model=Empty, harvest=[pattern])
    capture = ProductCapture(scope, {}, workspace)
    (tmp_path / "made.fits").write_text("new")
    assert capture.finish(Empty()) == [str(workspace / "../made.fits")]


def test_cache_hit_does_not_expand_harvest_again(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cab = family_cab(tmp_path)
    _dispatch(cab, None, script=FAMILY_SCRIPT)
    Path("image-unrelated.fits").write_text("external")
    assert _dispatch(cab, None, script=FAMILY_SCRIPT).cached
    Path("image-unrelated.fits").unlink()
    assert _dispatch(cab, None, script=FAMILY_SCRIPT).cached


def test_absolute_optional_output_is_recorded_after_sandbox_relativization(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / "sandboxes"))
    cab = family_cab(tmp_path, sandbox=True).model_copy(update={"harvest": []})
    prefix = str(tmp_path / "image")
    script = "from pathlib import Path;import sys;Path(sys.argv[1]+'-MFS-image.fits').write_text('mfs')"
    first = _dispatch(cab, None, prefix=prefix, script=script)
    assert first.outputs.mfs == Path("image-MFS-image.fits")
    assert get_cache_manifest(str(tmp_path / "cache")).entry("family")["products"] == [str(tmp_path / first.outputs.mfs)]
    assert _dispatch(cab, None, prefix=prefix, script=script).cached
    first.outputs.mfs.unlink()
    assert not _dispatch(cab, None, prefix=prefix, script=script).cached


def test_discarded_execution_cannot_supply_manual_result_inventory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cab = family_cab(tmp_path)

    def run(ctx):
        ctx.run(prefix="intermediate")
        Path("final.fits").write_text("final")
        return StepResult(name="family", returncode=0, outputs=FamilyOutputs(mfs=Path("final.fits")), inputs=ctx.inputs)

    _dispatch(cab, run, script=FAMILY_SCRIPT)
    assert get_cache_manifest(str(tmp_path / "cache")).entry("family")["products"] == [str(tmp_path / "final.fits")]
    assert _dispatch(cab, run, script=FAMILY_SCRIPT).cached


@pytest.mark.parametrize("version", [True, 1.0, 2, None])
def test_unsupported_product_contract_version_is_a_safe_miss(tmp_path, monkeypatch, version):
    monkeypatch.chdir(tmp_path)
    scope = family_cab(tmp_path)
    manifest = CacheManifest(tmp_path / "manifest.json")
    contract = product_contract(scope, {})
    contract["version"] = version
    manifest.record("family", "key", StepResult(name="family", returncode=0, outputs=FamilyOutputs(), inputs=Empty()), product_contract=contract, products=[])
    assert manifest.check("family", "key", scope, {}) is None


def test_opaque_path_factory_change_cannot_restore_earlier_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("a").write_text("a")
    Path("b").write_text("b")
    first_model = create_model("FirstFactory", product=(Path, Field(default_factory=lambda: Path("a"))))
    second_model = create_model("SecondFactory", product=(Path, Field(default_factory=lambda: Path("b"))))
    scope = Scope(name="factory", inputs_model=Empty, outputs_model=first_model, cache=True, cache_dir=str(tmp_path / "cache"))

    def run(ctx):
        return StepResult(name="factory", returncode=0, inputs=ctx.inputs, outputs=ctx.scope.outputs_model())

    first = _dispatch(scope, run)
    scope.outputs_model = second_model
    second = _dispatch(scope, run)
    assert first.cache_key == second.cache_key
    assert not second.cached
    assert first.outputs.product == Path("a")
    assert second.outputs.product == Path("b")
    assert not _dispatch(scope, run).cached


def test_builtin_list_path_factory_preserves_reuse(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = create_model("BuiltinFactory", products=(list[Path], Field(default_factory=list)))
    scope = Scope(name="builtin", inputs_model=Empty, outputs_model=model, cache=True, cache_dir=str(tmp_path / "cache"))

    def run(ctx):
        return StepResult(name="builtin", returncode=0, inputs=ctx.inputs, outputs=model())

    _dispatch(scope, run)
    assert _dispatch(scope, run).cached
    assert product_contract(scope, {})["fields"][0][6] == "list"


@pytest.mark.parametrize("sandbox", [False, True])
@pytest.mark.parametrize("cache_enabled", [False, True])
def test_recreated_empty_declared_directory_is_a_product(tmp_path, monkeypatch, sandbox, cache_enabled):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / "sandboxes"))
    model = create_model("EmptyDirectory", directory=(Path | None, Path("unused")))
    cab = family_cab(tmp_path, sandbox=sandbox).model_copy(update={"outputs_model": model, "harvest": ["unused/*.fits"], "cache": cache_enabled})
    script = "from pathlib import Path;Path('unused').rename('old');Path('unused').mkdir()"
    first = _dispatch(cab, None, script=script)
    assert first.success
    assert Path("unused").is_dir()
    if cache_enabled:
        assert get_cache_manifest(str(tmp_path / "cache")).entry("family")["products"] == [str(tmp_path / "unused")]
    assert _dispatch(cab, None, script=script).cached is cache_enabled
    Path("unused").rmdir()
    if Path("old").exists():
        Path("old").rmdir()
    assert not _dispatch(cab, None, script=script).cached


@pytest.mark.parametrize("phase", ["before", "after"])
def test_incomplete_cache_observation_preserves_success_and_exact_run_oracle(tmp_path, monkeypatch, phase):
    import shinobi.sandbox as sandbox_module

    monkeypatch.chdir(tmp_path)
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "child").write_text("initial")
    scope = Scope(name="opaque", inputs_model=Empty, outputs_model=Empty, harvest=["tree"], cache=True, cache_dir=str(tmp_path / "cache"))
    original = sandbox_module.observe_product_path
    state = {"executed": False, "calls": 0}

    def observe(path, *, recursive=False):
        if path == tree and recursive and (phase == "before" or state["executed"]):
            raise PermissionError("cache-only observation denied")
        return original(path, recursive=recursive)

    def run(ctx):
        state["calls"] += 1
        state["executed"] = True
        (tree / "child").write_text(f"success-{state['calls']}")
        return StepResult(name="opaque", returncode=0, outputs=Empty(), inputs=ctx.inputs)

    monkeypatch.setattr(sandbox_module, "observe_product_path", observe)
    for run_id in ["successful-opaque-1", "successful-opaque-2"]:
        state["executed"] = False
        result = _dispatch(scope, run, _run_id=run_id)
        assert result.success and not result.cached
        entry = get_cache_manifest(str(tmp_path / "cache")).entry("opaque")
        assert entry["run_id"] == run_id
        assert entry["products"] is None
        assert entry["outputs"] == {}
    assert state["calls"] == 2


@pytest.mark.parametrize("harvest", [["tree"], ["tree/*.fits"]])
def test_opaque_harvest_directory_preserves_cache_enabled_execution(tmp_path, monkeypatch, harvest):
    monkeypatch.chdir(tmp_path)
    tree = tmp_path / "tree"
    tree.mkdir()
    scope = Scope(name="opaque", inputs_model=Empty, outputs_model=Empty, harvest=harvest, cache=True, cache_dir=str(tmp_path / "cache"))
    calls = []

    def run(ctx):
        calls.append(1)
        (tree / "known-child.fits").write_text("valid scientific result")
        return StepResult(name="opaque", returncode=0, outputs=Empty(), inputs=ctx.inputs)

    tree.chmod(0o300)
    try:
        try:
            list(tree.iterdir())
        except PermissionError:
            pass
        else:
            pytest.skip("Effective privileges bypass directory read permissions")
        assert _dispatch(scope, run, cache=False).success
        first = _dispatch(scope, run, _run_id="opaque-permissions-success")
        assert first.success and not first.cached
        entry = get_cache_manifest(str(tmp_path / "cache")).entry("opaque")
        assert entry["run_id"] == "opaque-permissions-success"
        assert entry["products"] is None
        (tree / "known-child.fits").unlink()
        assert not _dispatch(scope, run).cached
        assert len(calls) == 3
    finally:
        tree.chmod(0o700)


def test_authoritative_unknown_sandbox_capture_cannot_use_trusted_fallback(tmp_path, monkeypatch):
    import shinobi.sandbox as sandbox_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / "sandboxes"))
    cab = family_cab(tmp_path, sandbox=True)
    Path("effective-stale.fits").write_text("leftover")
    original = sandbox_module.observe_product_path
    calls = []

    def observe(path, *, recursive=False):
        if path.name.startswith("effective-") and recursive:
            raise PermissionError("effective execution inventory unknown")
        return original(path, recursive=recursive)

    def run(ctx):
        calls.append(1)
        return ctx.run(prefix=str(tmp_path / "effective"))

    monkeypatch.setattr(sandbox_module, "observe_product_path", observe)
    for run_id in ["override-success-1", "override-success-2"]:
        result = _dispatch(cab, run, prefix="original", script=FAMILY_SCRIPT, _run_id=run_id)
        assert result.success and not result.cached
        entry = get_cache_manifest(str(tmp_path / "cache")).entry("family")
        assert entry["run_id"] == run_id
        assert entry["products"] is None
    assert len(calls) == 2
    assert Path("effective-0000-I-image.fits").read_text() == "0000-I-image"


@pytest.mark.parametrize("sandbox", [False, True])
@pytest.mark.parametrize("cache_enabled", [False, True])
def test_unavailable_setup_observation_does_not_abort_cab(tmp_path, monkeypatch, sandbox, cache_enabled):
    import shinobi.sandbox as sandbox_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / "sandboxes"))
    model = create_model("EmptyDirectory", directory=(Path | None, Path("unused")))
    cab = family_cab(tmp_path, sandbox=sandbox).model_copy(update={"outputs_model": model, "harvest": ["unused/*.fits"], "cache": cache_enabled})
    original = sandbox_module.observe_product_path

    def observe(path, *, recursive=False):
        if path.name == "unused" and not recursive:
            raise PermissionError("setup evidence unavailable")
        return original(path, recursive=recursive)

    monkeypatch.setattr(sandbox_module, "observe_product_path", observe)
    script = "from pathlib import Path;Path('unused').rename('old');Path('unused').mkdir();Path('unused/science.dat').write_text('science')"
    result = _dispatch(cab, None, script=script, _run_id="successful-cab-without-setup-evidence")
    assert result.success and Path("unused/science.dat").read_text() == "science"
    if cache_enabled:
        entry = get_cache_manifest(str(tmp_path / "cache")).entry("family")
        assert entry["run_id"] == "successful-cab-without-setup-evidence"
        assert entry["products"] is None


@pytest.mark.parametrize("pattern", ["tree/*/*.fits", "tree/**/*.fits", "tree/**"])
def test_directory_wildcard_cache_evidence_is_conservatively_unknown(tmp_path, pattern):
    (tmp_path / "tree/branch").mkdir(parents=True)
    (tmp_path / "tree/branch/child.fits").write_text("existing")
    scope = Scope(name="directory-glob", inputs_model=Empty, outputs_model=Empty, harvest=[pattern])
    capture = ProductCapture(scope, {}, tmp_path)
    assert capture.finish(Empty()) is None


@pytest.mark.parametrize("pattern", ["missing/*.fits", "missing/**/*.fits", "missing/*/*.fits"])
def test_missing_literal_harvest_parent_is_known_empty(tmp_path, pattern):
    scope = Scope(name="missing-parent", inputs_model=Empty, outputs_model=Empty, harvest=[pattern])
    assert ProductCapture(scope, {}, tmp_path).finish(Empty()) == []


@pytest.mark.parametrize("sandbox", [False, True])
@pytest.mark.parametrize("pattern", ["tree/*/*.fits", "tree/**/*.fits"])
def test_nested_glob_unknown_direct_evidence_and_fresh_sandbox_inventory(tmp_path, monkeypatch, sandbox, pattern):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / "sandboxes"))
    cab = family_cab(tmp_path, sandbox=sandbox).model_copy(update={"harvest": [pattern]})
    script = "from pathlib import Path;p=Path('tree/branch');p.mkdir(parents=True,exist_ok=True);(p/'child.fits').write_text('nested image')"
    first = _dispatch(cab, None, script=script)
    assert first.success
    entry = get_cache_manifest(str(tmp_path / "cache")).entry("family")
    expected = [str(tmp_path / "tree/branch/child.fits")] if sandbox else None
    assert entry["products"] == expected
    assert _dispatch(cab, None, script=script).cached is sandbox
    Path("tree/branch/child.fits").unlink()
    assert not _dispatch(cab, None, script=script).cached
    assert Path("tree/branch/child.fits").read_text() == "nested image"


def test_literal_harvest_file_does_not_require_parent_read_permission(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tree = tmp_path / "tree"
    tree.mkdir()
    scope = Scope(name="literal", inputs_model=Empty, outputs_model=Empty, harvest=["tree/known-child.fits"], cache=True, cache_dir=str(tmp_path / "cache"))
    calls = []

    def run(ctx):
        calls.append(1)
        (tree / "known-child.fits").write_text("valid scientific result")
        return StepResult(name="literal", returncode=0, outputs=Empty(), inputs=ctx.inputs)

    tree.chmod(0o300)
    try:
        try:
            list(tree.iterdir())
        except PermissionError:
            pass
        else:
            pytest.skip("Effective privileges bypass directory read permissions")
        assert _dispatch(scope, run).success
        assert get_cache_manifest(str(tmp_path / "cache")).entry("literal")["products"] == [str(tree / "known-child.fits")]
        assert _dispatch(scope, run).cached
        assert len(calls) == 1
        (tree / "known-child.fits").unlink()
        assert not _dispatch(scope, run).cached
        assert len(calls) == 2
    finally:
        tree.chmod(0o700)


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("phase", ["preparation", "pruning"])
def test_unknown_empty_setup_identity_cannot_replace_existing_workspace_data(tmp_path, monkeypatch, cache_enabled, phase):
    import shinobi.sandbox as sandbox_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / "sandboxes"))
    Path("unused").mkdir()
    Path("unused/valuable").write_text("caller data")
    model = create_model("UnusedDirectory", directory=(Path | None, Path("unused")))
    cab = family_cab(tmp_path, sandbox=True).model_copy(update={"outputs_model": model, "harvest": ["unused/*.fits"], "cache": cache_enabled})
    original = sandbox_module.observe_product_path
    observations = 0

    def observe(path, *, recursive=False):
        nonlocal observations
        if path.name == "unused" and path.is_relative_to(tmp_path / "sandboxes") and not recursive:
            observations += 1
            if phase == "preparation" or observations > 1:
                raise PermissionError("transient private setup observation failure")
        return original(path, recursive=recursive)

    monkeypatch.setattr(sandbox_module, "observe_product_path", observe)
    result = _dispatch(cab, None, script="pass", _run_id="success-without-phantom-output")
    assert result.success
    assert Path("unused/valuable").read_text() == "caller data"
    if cache_enabled:
        entry = get_cache_manifest(str(tmp_path / "cache")).entry("family")
        assert entry["run_id"] == "success-without-phantom-output"
        assert entry["products"] is None


@pytest.mark.parametrize("initially_inaccessible", [False, True])
def test_literal_harvest_without_search_permission_is_unknown_not_empty(tmp_path, monkeypatch, initially_inaccessible):
    monkeypatch.chdir(tmp_path)
    tree = tmp_path / "tree"
    tree.mkdir()
    known = tree / "known.fits"
    known.write_text("permission probe")
    scope = Scope(name="literal", inputs_model=Empty, outputs_model=Empty, harvest=["tree/known.fits"], cache=True, cache_dir=str(tmp_path / "cache"))
    calls = []

    def run(ctx):
        calls.append(1)
        tree.chmod(0o700)
        known.write_text(f"scientific success {len(calls)}")
        tree.chmod(0o600)
        return StepResult(name="literal", returncode=0, outputs=Empty(), inputs=ctx.inputs)

    tree.chmod(0o600)
    try:
        try:
            known.stat()
        except PermissionError:
            pass
        else:
            pytest.skip("Effective privileges bypass directory search permissions")
        tree.chmod(0o700)
        if not initially_inaccessible:
            known.unlink()
        else:
            tree.chmod(0o600)
        first = _dispatch(scope, run, _run_id="literal-no-search-success-1")
        assert first.success and not first.cached
        entry = get_cache_manifest(str(tmp_path / "cache")).entry("literal")
        assert entry["run_id"] == "literal-no-search-success-1"
        assert entry["products"] is None
        tree.chmod(0o700)
        known.unlink()
        tree.chmod(0o600)
        second = _dispatch(scope, run, _run_id="literal-no-search-success-2")
        assert second.success and not second.cached
        assert len(calls) == 2
        entry = get_cache_manifest(str(tmp_path / "cache")).entry("literal")
        assert entry["run_id"] == "literal-no-search-success-2"
        assert entry["products"] is None
        tree.chmod(0o700)
        assert known.read_text() == "scientific success 2"
    finally:
        tree.chmod(0o700)
