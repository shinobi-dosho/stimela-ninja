"""Bounded MS reader scatter preserves the contained leaf lifecycle."""

from pathlib import Path
import threading
import sys

import pytest
from pydantic import BaseModel, Field

from shinobi import Cab, DatasetAccess, DatasetColumns, FITSAccess, FITSImage, FamilySpec, MeasurementSetV2, MemberRule, ProductFamily, Recipe, pystep
from shinobi.dataset_access import DatasetAccessError, plan_recipe_accesses
from shinobi.dataset_lifecycle import DatasetLifecyclePhase
from shinobi.exceptions import DatasetLifecycleUnavailableError, DatasetLifecycleViolationError
from shinobi.graph import RecipeNotOffloadableError, check_offloadable
from shinobi.steps.schema import InputRef, OutputRef, ParamMeta, Scope
from tests._dataset_fixtures import ROWS, attempts, make_ms, scans, set_scans


class MSInput(BaseModel):
    ms: MeasurementSetV2


class Empty(BaseModel):
    pass


class Values(BaseModel):
    value: int


class Gathered(BaseModel):
    values: list[int]


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))


def test_real_concurrent_readers_reverse_completion_and_provenance(tmp_path):
    tables = pytest.importorskip("casacore.tables")
    ms = make_ms(tmp_path / "obs.ms")
    barrier = threading.Barrier(2)
    second_finished = threading.Event()
    finished = []

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2, index: int) -> Values:
        with tables.table(str(ms), readonly=True, ack=False) as table:
            assert table.nrows() == ROWS
            barrier.wait(timeout=10)
            if index == 0:
                assert second_finished.wait(timeout=10)
            finished.append(index)
            if index == 1:
                second_finished.set()
        return Values(value=index)

    recipe = Recipe(name="parallel", inputs_model=MSInput, outputs_model=Gathered, max_workers=2)
    recipe.add_step("read", read, scatter=["index"], ms=InputRef(field="ms"), index=[0, 1])
    recipe.set_output("values", recipe.outputs.read.value)
    result = recipe(ms=ms)
    assert result.outputs.values == [0, 1]
    assert finished == [1, 0]
    [attempt] = attempts(tmp_path)
    assert attempt.phase is DatasetLifecyclePhase.COMMITTED
    assert {leaf.step_path for leaf in attempt.leaves} == {"parallel.read[0]", "parallel.read[1]"}
    assert all(leaf.outcome == "committed" and not leaf.mutations for leaf in attempt.leaves)


def test_concrete_ms_paths_are_planned_per_slice(tmp_path):
    left, right = make_ms(tmp_path / "left.ms"), make_ms(tmp_path / "right.ms")

    class Many(BaseModel):
        paths: list[MeasurementSetV2]

    @pystep()
    def read(ms: MeasurementSetV2) -> Values:
        return Values(value=scans(ms)[0])

    recipe = Recipe(name="many", inputs_model=Many, outputs_model=Gathered, max_workers=2)
    recipe.add_step("read", read, scatter=["ms"], ms=InputRef(field="paths"))
    recipe.set_output("values", recipe.outputs.read.value)
    plan = plan_recipe_accesses(recipe, {"paths": [left, right]}, workspace=tmp_path)
    assert [access.root for access in plan.accesses["read"]] == [left, right]
    assert recipe(paths=[left, right]).outputs.values == [1, 1]


def test_create_write_then_scatter_read_orders_and_recovers(tmp_path):
    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="create")])
    def create(ms: MeasurementSetV2) -> None:
        make_ms(ms)

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="write", columns=DatasetColumns(write=("SCAN_NUMBER",)))])
    def write(ms: MeasurementSetV2) -> None:
        set_scans(ms, 9)

    @pystep()
    def read(ms: MeasurementSetV2, index: int) -> Values:
        return Values(value=scans(ms)[0] + index)

    recipe = Recipe(name="pipeline", inputs_model=MSInput, outputs_model=Gathered, max_workers=2)
    recipe.add_step("create", create, ms=InputRef(field="ms"))
    recipe.add_step("write", write, ms=InputRef(field="ms"))
    recipe.add_step("read", read, scatter=["index"], ms=InputRef(field="ms"), index=[0, 1])
    recipe.set_output("values", recipe.outputs.read.value)
    ms = tmp_path / "new.ms"
    plan = plan_recipe_accesses(recipe, {"ms": ms}, workspace=tmp_path)
    assert plan.graph.deps == [set(), {0}, {1}]
    assert recipe(ms=ms, cache=True, cache_dir=str(tmp_path / "cache")).outputs.values == [9, 10]
    [attempt] = attempts(tmp_path)
    assert len(attempt.leaves) == 4
    assert all(leaf.outcome == "committed" for leaf in attempt.leaves)


@pytest.mark.parametrize("shape", ["duplicate", "directory", "alias"])
def test_cross_slice_output_collisions_refuse_before_launch(tmp_path, shape):
    ms = make_ms(tmp_path / "obs.ms")
    called = []

    @pystep(write_paths=["destination"])
    def read(ms: MeasurementSetV2, destination: Path) -> None:
        called.append(True)

    directory = tmp_path / "products"
    directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    paths = {"duplicate": [directory / "a", directory / "a"], "directory": [directory, directory / "a"], "alias": [directory / "a", alias / "a"]}[shape]
    recipe = Recipe(name="collision", inputs_model=MSInput, outputs_model=Empty, max_workers=2)
    recipe.add_step("read", read, scatter=["destination"], ms=InputRef(field="ms"), destination=paths)
    with pytest.raises(DatasetAccessError, match="scatter slices.*overlapping"):
        recipe(ms=ms)
    assert called == []


def test_external_closure_cross_slice_write_refuses_before_launch(tmp_path):
    tables = pytest.importorskip("casacore.tables")
    left, right = make_ms(tmp_path / "left.ms"), make_ms(tmp_path / "right.ms")
    external = tmp_path / "external-antenna"
    (left / "ANTENNA").rename(external)
    with tables.table(str(left), readonly=False, ack=False) as main:
        main.putkeyword("ANTENNA", f"Table: {external}")
    called = []

    @pystep(write_paths=["destination"])
    def read(ms: MeasurementSetV2, destination: Path) -> None:
        called.append(True)

    recipe = Recipe(name="external", inputs_model=Empty, outputs_model=Empty)
    recipe.add_step("read", read, scatter=["ms", "destination"], ms=[left, right], destination=[tmp_path / "report", external / "report"])
    with pytest.raises(DatasetAccessError, match="scatter slices.*overlapping"):
        recipe()
    assert not called


@pytest.mark.parametrize("mode", ["write", "create", "inferred"])
def test_mutating_scatter_refused_before_launch(tmp_path, mode):
    called = []

    @pystep(dataset_accesses=[] if mode == "inferred" else [DatasetAccess(field="ms", mode=mode)], write_paths=["ms"] if mode == "inferred" else [])
    def mutate(ms: MeasurementSetV2, index: int) -> None:
        called.append(True)

    recipe = Recipe(name="mutate", inputs_model=MSInput, outputs_model=Empty)
    recipe.add_step("mutate", mutate, scatter=["index"], ms=InputRef(field="ms"), index=[0, 1])
    with pytest.raises(DatasetLifecycleUnavailableError, match="only read-only MSv2 scatter"):
        recipe(ms=tmp_path / "absent.ms")
    assert not called


def test_unresolved_inputs_and_offload_remain_refused(tmp_path):
    class Runtime(BaseModel):
        indices: list[int]

    producer = Scope(name="producer", inputs_model=Empty, outputs_model=Runtime)

    @pystep()
    def read(ms: MeasurementSetV2, index: int) -> None:
        pytest.fail("unresolved scatter launched")

    recipe = Recipe(name="unknown", inputs_model=MSInput, outputs_model=Empty)
    recipe.add_step("producer", producer)
    recipe.add_step("read", read, scatter=["index"], ms=InputRef(field="ms"), index=OutputRef(step="producer", field="indices"))
    with pytest.raises(DatasetAccessError, match="concrete slice inputs"):
        plan_recipe_accesses(recipe, {"ms": tmp_path / "obs.ms"}, workspace=tmp_path)
    with pytest.raises(RecipeNotOffloadableError, match="scatter"):
        check_offloadable(recipe)


def test_frozen_default_factories_and_read_mutation_checks(tmp_path):
    ms = make_ms(tmp_path / "obs.ms")
    calls = []

    def report_default():
        calls.append(True)
        return tmp_path / f"report-{len(calls)}"

    class Inputs(MSInput):
        index: int
        report: Path = Field(default_factory=report_default)

    @pystep(write_paths=["report"])
    def read(ms: MeasurementSetV2, index: int, report: Path) -> None:
        report.write_text(str(index))

    read.step.inputs_model = Inputs
    recipe = Recipe(name="defaults", inputs_model=MSInput, outputs_model=Empty, max_workers=2)
    recipe.add_step("read", read, scatter=["index"], ms=InputRef(field="ms"), index=[0, 1])
    assert recipe(ms=ms).success
    assert len(calls) == 2
    assert [(tmp_path / f"report-{index}").read_text() for index in (1, 2)] == ["0", "1"]

    @pystep()
    def corrupt(ms: MeasurementSetV2, index: int) -> None:
        set_scans(ms, 99)

    bad = Recipe(name="bad", inputs_model=MSInput, outputs_model=Empty)
    bad.add_step("corrupt", corrupt, scatter=["index"], ms=InputRef(field="ms"), index=[0])
    with pytest.raises(DatasetLifecycleViolationError, match="reader changed its dataset"):
        bad(ms=ms)


def test_empty_only_strict_scatter_refused(tmp_path):
    @pystep()
    def read(ms: MeasurementSetV2, index: int) -> None:
        pytest.fail("empty scatter launched")

    recipe = Recipe(name="empty", inputs_model=MSInput, outputs_model=Empty)
    recipe.add_step("read", read, scatter=["index"], ms=InputRef(field="ms"), index=[])
    with pytest.raises(DatasetLifecycleUnavailableError, match="no concrete dataset access"):
        recipe(ms=tmp_path / "absent.ms")


def test_exact_fits_families_gather_without_lifecycle_authority(tmp_path):
    pytest.importorskip("astropy.io.fits")
    pytest.importorskip("numpy")
    ms = make_ms(tmp_path / "obs.ms")

    class Images(BaseModel):
        images: ProductFamily[FITSImage]

    class GatheredImages(BaseModel):
        images: list[ProductFamily[FITSImage]]

    class ImagingInputs(MSInput):
        prefix: str
        script: str

    image = Cab(
        name="image",
        command=f"{sys.executable} -c",
        inputs_model=ImagingInputs,
        outputs_model=Images,
        field_meta={
            "script": ParamMeta(positional_head=True),
            "prefix": ParamMeta(positional=True),
            "images": ParamMeta(family=FamilySpec(coordinates={}, rules=(MemberRule(path="{prefix}.fits", required=True),))),
        },
        fits_accesses=[FITSAccess(field="images", mode="read")],
    )
    script = "import sys; from astropy.io import fits; import numpy as np; hdu=fits.PrimaryHDU(np.ones((2,2))); hdu.header['CTYPE1']='LINEAR'; hdu.header['CTYPE2']='LINEAR'; hdu.writeto(sys.argv[-1]+'.fits')"
    recipe = Recipe(name="imaging", inputs_model=MSInput, outputs_model=GatheredImages, max_workers=2)
    recipe.add_step("image", image, scatter=["prefix"], ms=InputRef(field="ms"), prefix=["natural", "uniform"], script=script)
    recipe.set_output("images", recipe.outputs.image.images)
    result = recipe(ms=ms)
    assert [family.select().name for family in result.outputs.images] == ["natural.fits", "uniform.fits"]
    assert all(family.resolved for family in result.outputs.images)
    assert len(attempts(tmp_path)[0].leaves) == 2


def test_authoritative_fits_scatter_remains_refused(tmp_path):
    called = []

    @pystep()
    def read(image: FITSImage) -> None:
        called.append(True)

    recipe = Recipe(name="fits", inputs_model=Empty, outputs_model=Empty)
    recipe.add_step("read", read, scatter=["image"], image=[tmp_path / "absent.fits"])
    with pytest.raises(DatasetLifecycleUnavailableError, match="authoritative FITS dataset contract"):
        recipe()
    assert not called


def test_shared_parent_output_write_cannot_overlap_another_slice_read(tmp_path):
    ms = make_ms(tmp_path / "obs.ms")
    source = tmp_path / "products" / "input"
    source.parent.mkdir()
    source.write_text("read me")

    @pystep(write_paths=["destination"])
    def read(ms: MeasurementSetV2, source: Path, destination: Path) -> None:
        pytest.fail("collision launched")

    recipe = Recipe(name="parent", inputs_model=MSInput, outputs_model=Empty)
    recipe.add_step(
        "read", read, scatter=["source", "destination"], ms=InputRef(field="ms"), source=[tmp_path / "elsewhere", source], destination=[source.parent, tmp_path / "report"]
    )
    with pytest.raises(DatasetAccessError, match="scatter slices.*overlapping"):
        recipe(ms=ms)


@pytest.mark.parametrize("unresolved", [False, True])
def test_finite_family_scatter_reservations_fail_before_launch(tmp_path, unresolved):
    ms = make_ms(tmp_path / "obs.ms")

    class Images(BaseModel):
        images: ProductFamily[FITSImage]

    @pystep()
    def emit(ms: MeasurementSetV2, prefix: str) -> Images:
        pytest.fail("invalid reservation launched")

    emit.step.field_meta["images"] = ParamMeta(family=FamilySpec(coordinates={}, rules=(MemberRule(path="{prefix}.fits", required=True),)))
    recipe = Recipe(name="finite", inputs_model=MSInput, outputs_model=Empty)
    if unresolved:

        class Prefixes(BaseModel):
            prefixes: list[str]

        recipe.add_step("producer", Scope(name="producer", inputs_model=Empty, outputs_model=Prefixes))
        prefixes = OutputRef(step="producer", field="prefixes")
    else:
        prefixes = ["duplicate", "duplicate"]
    recipe.add_step("emit", emit, scatter=["prefix"], ms=InputRef(field="ms"), prefix=prefixes)
    with pytest.raises((DatasetAccessError, DatasetLifecycleUnavailableError), match="statically resolved slice inputs|scatter slices.*overlapping"):
        recipe(ms=ms)


def test_runtime_output_drift_cannot_redirect_frozen_slice_family(tmp_path):
    ms = make_ms(tmp_path / "obs.ms")

    class Prefixes(BaseModel):
        prefixes: list[str] = ["claimed-a", "claimed-b"]

    class Images(BaseModel):
        images: ProductFamily[FITSImage]

    class EmitInput(MSInput):
        prefix: str
        script: str

    @pystep()
    def change() -> Prefixes:
        return Prefixes(prefixes=["unclaimed-a", "unclaimed-b"])

    emit = Cab(
        name="emit",
        command=f"{sys.executable} -c",
        inputs_model=EmitInput,
        outputs_model=Images,
        field_meta={
            "script": ParamMeta(positional_head=True),
            "images": ParamMeta(family=FamilySpec(coordinates={}, rules=(MemberRule(path="{prefix}.fits", required=True),))),
        },
    )
    recipe = Recipe(name="drift", inputs_model=MSInput, outputs_model=Empty)
    recipe.add_step("change", change)
    recipe.add_step("emit", emit, scatter=["prefix"], ms=InputRef(field="ms"), prefix=recipe.outputs.change.prefixes, script="raise RuntimeError('backend launched')")
    with pytest.raises(DatasetLifecycleUnavailableError, match="changed its frozen scatter filesystem accesses"):
        recipe(ms=ms)
    assert {leaf.step_path for leaf in attempts(tmp_path)[0].leaves} == set()


@pytest.mark.parametrize("actual", [["a", "a"], ["b", "a"], ["a", "b", "c"], ["a"], []])
def test_runtime_generic_path_drift_refuses_whole_batch(tmp_path, actual):
    ms = make_ms(tmp_path / "obs.ms")
    called = []

    class Paths(BaseModel):
        paths: list[str] = ["a", "b"]

    @pystep()
    def producer() -> Paths:
        return Paths(paths=actual)

    @pystep(write_paths=["destination"])
    def read(ms: MeasurementSetV2, destination: Path) -> None:
        called.append(destination)

    recipe = Recipe(name="generic-drift", inputs_model=Empty, outputs_model=Empty, max_workers=2)
    recipe.add_step("producer", producer)
    recipe.add_step("read", read, scatter=["destination"], ms=ms, destination=recipe.outputs.producer.paths)
    with pytest.raises(DatasetLifecycleUnavailableError, match="frozen scatter filesystem accesses|frozen scatter cardinality"):
        recipe()
    assert not called


def test_unchanged_outputref_paths_reuse_each_default_once(tmp_path):
    ms = make_ms(tmp_path / "obs.ms")
    called = []
    defaults = []

    class Paths(BaseModel):
        paths: list[str] = ["a", "b"]

    @pystep()
    def producer() -> Paths:
        return Paths()

    def default():
        defaults.append(True)
        return len(defaults)

    class Inputs(MSInput):
        destination: Path
        value: int = Field(default_factory=default)

    @pystep(write_paths=["destination"])
    def read(ms: MeasurementSetV2, destination: Path, value: int) -> None:
        called.append((destination.name, value))

    read.step.inputs_model = Inputs
    recipe = Recipe(name="same", inputs_model=Empty, outputs_model=Empty)
    recipe.add_step("producer", producer)
    recipe.add_step("read", read, scatter=["destination"], ms=ms, destination=recipe.outputs.producer.paths)
    assert recipe().success
    assert called == [("a", 1), ("b", 2)]
    assert len(defaults) == 2


def test_unscattered_dataset_path_drift_refuses_batch(tmp_path):
    left, right = make_ms(tmp_path / "left.ms"), make_ms(tmp_path / "right.ms")
    called = []

    class Selected(BaseModel):
        ms: str = str(left)

    @pystep()
    def choose() -> Selected:
        return Selected(ms=str(right))

    @pystep()
    def read(ms: MeasurementSetV2, index: int) -> None:
        called.append(True)

    recipe = Recipe(name="selection", inputs_model=Empty, outputs_model=Empty)
    recipe.add_step("choose", choose)
    recipe.add_step("read", read, scatter=["index"], ms=recipe.outputs.choose.ms, index=[0, 1])
    with pytest.raises(DatasetLifecycleUnavailableError, match="frozen scatter filesystem accesses"):
        recipe()
    assert not called


def test_scalar_refinement_preserving_accesses_is_allowed(tmp_path):
    ms = make_ms(tmp_path / "obs.ms")
    called = []

    class Indices(BaseModel):
        indices: list[int] = [0, 1]

    @pystep()
    def choose() -> Indices:
        return Indices(indices=[2, 3])

    @pystep()
    def read(ms: MeasurementSetV2, index: int) -> None:
        called.append(index)

    recipe = Recipe(name="scalar", inputs_model=Empty, outputs_model=Empty)
    recipe.add_step("choose", choose)
    recipe.add_step("read", read, scatter=["index"], ms=ms, index=recipe.outputs.choose.indices)
    assert recipe().success
    assert called == [2, 3]


def test_symlink_retarget_after_planning_cannot_change_slice_claim(tmp_path):
    ms = make_ms(tmp_path / "obs.ms")
    for name in ("claimed", "redirected"):
        (tmp_path / name).mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "claimed", target_is_directory=True)
    called = []

    class Paths(BaseModel):
        paths: list[str] = ["alias/a", "alias/b"]

    @pystep()
    def redirect() -> Paths:
        alias.unlink()
        alias.symlink_to(tmp_path / "redirected", target_is_directory=True)
        return Paths()

    @pystep(write_paths=["destination"])
    def read(ms: MeasurementSetV2, destination: Path) -> None:
        called.append(destination)

    recipe = Recipe(name="retarget", inputs_model=Empty, outputs_model=Empty)
    recipe.add_step("redirect", redirect)
    recipe.add_step("read", read, scatter=["destination"], ms=ms, destination=recipe.outputs.redirect.paths)
    with pytest.raises(DatasetLifecycleUnavailableError, match="frozen scatter filesystem accesses"):
        recipe()
    assert not called
