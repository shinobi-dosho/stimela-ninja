"""FITS byte identity, lifecycle mutations, groups, and durable recovery."""

from pathlib import Path
import os
import subprocess
import sys

import pytest

from pydantic import BaseModel

from shinobi import FITSImage, FITSTable, FITSAccess, FITSHDU, pystep, Recipe, InputRef
from shinobi.steps.dispatch import _dispatch
from shinobi.exceptions import DatasetLifecycleUnavailableError, DatasetLifecycleViolationError
from shinobi.cache import get_cache_manifest
from shinobi.fits_lifecycle import FITSLifecycleStore
from shinobi.snapshots import faults, reconcile, get_journal, chain_id


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield
    faults.hooks.clear()


fits = pytest.importorskip("astropy.io.fits")
np = pytest.importorskip("numpy")

# Gate the helper module before importing it: it also skips without Astropy.
from tests.test_fits import image  # noqa: E402


def run(step, tmp_path, **kwargs):
    return _dispatch(step.step, step.func, cache_dir=str(tmp_path / "cache"), **kwargs)


@pystep(fits_accesses=[FITSAccess(field="file", mode="write")])
def write(file: FITSImage, value: int = 4) -> None:
    with fits.open(file, mode="update") as hdus:
        hdus[0].data[:] = value


@pystep(fits_accesses=[FITSAccess(field="files", mode="write")])
def joint(files: list[FITSImage], value: int = 4) -> None:
    for path in files:
        with fits.open(path, mode="update") as hdus:
            hdus[0].data[:] = value


@pystep()
def read(file: FITSImage) -> None:
    assert file.exists()


def test_native_write_reuse_and_changed_bytes(tmp_path):
    path = image(tmp_path / "x.fits")
    assert run(write, tmp_path, file=path).success
    assert run(write, tmp_path, file=path).cached
    with fits.open(path, mode="update") as hdus:
        hdus[0].data[:] = 20
    with pytest.raises(DatasetLifecycleUnavailableError):
        run(write, tmp_path, file=path)
    assert fits.getdata(path)[0, 0] == 20


def test_writer_invalid_output_restores_exact_bytes(tmp_path):
    path = image(tmp_path / "x.fits")
    before = path.read_bytes()

    @pystep(fits_accesses=[FITSAccess(field="file", mode="write")])
    def invalid(file: FITSImage) -> None:
        file.write_bytes(b"not FITS")

    with pytest.raises(DatasetLifecycleViolationError):
        run(invalid, tmp_path, file=path)
    assert path.read_bytes() == before


def test_writer_tool_failure_restores_exact_bytes(tmp_path):
    path = image(tmp_path / "x.fits")
    before = path.read_bytes()

    @pystep(fits_accesses=[FITSAccess(field="file", mode="write")])
    def invalid(file: FITSImage) -> None:
        file.write_bytes(b"changed")
        raise RuntimeError("tool failed")

    with pytest.raises(RuntimeError, match="tool failed"):
        run(invalid, tmp_path, file=path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("fails", [False, True])
def test_read_detects_change_in_unselected_hdu(tmp_path, fails):
    path = image(tmp_path / "x.fits")
    with fits.open(path, mode="append") as hdus:
        hdus.append(fits.ImageHDU(np.zeros((2, 2))))

    @pystep()
    def bad(file: FITSImage) -> None:
        with fits.open(file, mode="update") as hdus:
            hdus[1].data[:] = 2
        if fails:
            raise RuntimeError("failure")

    with pytest.raises(DatasetLifecycleViolationError):
        run(bad, tmp_path, file=path)


@pytest.mark.parametrize("stage", ["P_MARKED", "P_RESTORED", "S1", "S2", "S3", "S4", "S5"])
def test_joint_exact_oracle_boundary(tmp_path, stage):
    paths = [image(tmp_path / f"{i}.fits") for i in range(2)]
    before = [path.read_bytes() for path in paths]

    def crash():
        raise RuntimeError(stage)

    faults.hooks[stage] = crash
    with pytest.raises((RuntimeError, DatasetLifecycleUnavailableError)):
        run(joint, tmp_path, files=paths)
    faults.hooks.clear()
    committed = stage in ("S3", "S4", "S5")
    assert [path.read_bytes() == one for path, one in zip(paths, before)] == [not committed] * 2
    reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
    assert all(get_journal(str(tmp_path / "cache")).get(chain_id(path)).marker is None for path in paths)
    assert [path.read_bytes() == one for path, one in zip(paths, before)] == [not committed] * 2


def test_empty_and_optional_refuse(tmp_path):
    with pytest.raises(DatasetLifecycleUnavailableError):
        run(joint, tmp_path, files=[])

    @pystep()
    def optional(file: FITSImage | None = None) -> None:
        pass

    with pytest.raises(DatasetLifecycleUnavailableError):
        run(optional, tmp_path)


@pytest.mark.parametrize("alias", ["duplicate", "hardlink", "symlink"])
def test_alias_refusal(tmp_path, alias):
    path = image(tmp_path / "x.fits")
    other = tmp_path / "y.fits"
    if alias == "hardlink":
        os.link(path, other)
    elif alias == "symlink":
        other.symlink_to(path)
    else:
        other = path
    with pytest.raises(DatasetLifecycleUnavailableError):
        run(joint, tmp_path, files=[path, other])


def test_remote_container_refuses_before_write(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "ssh://remote.example")
    path = image(tmp_path / "x.fits")
    before = path.read_bytes()
    with pytest.raises(DatasetLifecycleUnavailableError):
        run(write, tmp_path, file=path, backend="docker")
    assert path.read_bytes() == before


def test_native_table_writer_and_reader(tmp_path):
    path = tmp_path / "t.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns([fits.Column(name="x", format="J", array=[1])])]).writeto(path)

    @pystep(fits_accesses=[FITSAccess(field="table", mode="write", hdu=FITSHDU(index=1))])
    def table_write(table: FITSTable) -> None:
        with fits.open(table, mode="update") as hdus:
            hdus[1].data["x"][:] = 2

    @pystep(fits_accesses=[FITSAccess(field="table", mode="read", hdu=FITSHDU(index=1))])
    def table_read(table: FITSTable) -> None:
        assert table.exists()

    assert run(table_write, tmp_path, table=path).success
    assert run(table_read, tmp_path, table=path).success


def test_scalar_create_and_reuse(tmp_path):
    class Outputs(BaseModel):
        file: FITSImage

    @pystep(fits_accesses=[FITSAccess(field="file", mode="create")])
    def create(file: Path) -> Outputs:
        image(file)
        return Outputs(file=file)

    path = tmp_path / "new.fits"
    assert run(create, tmp_path, file=path).success
    assert run(create, tmp_path, file=path).cached


def test_reader_pending_refusal(tmp_path):
    path = image(tmp_path / "x.fits")

    def stop():
        raise RuntimeError("stage")

    faults.hooks["S3"] = stop
    with pytest.raises(RuntimeError):
        run(write, tmp_path, file=path)
    faults.hooks.clear()
    with pytest.raises(DatasetLifecycleUnavailableError, match="pending"):
        run(read, tmp_path, file=path)


@pytest.mark.parametrize("stage", ["P_MARKED", "P_RESTORED", "S1", "S2", "S3", "S4", "S5", "CORRUPT"])
def test_fresh_process_crash_exact_recovery(tmp_path, stage):
    paths = [image(tmp_path / f"{i}.fits") for i in range(2)]
    before = [path.read_bytes() for path in paths]
    script = tmp_path / "crash.py"
    script.write_text("""from pathlib import Path
import os,sys
from shinobi import FITSImage,FITSAccess,pystep
from shinobi.steps.dispatch import _dispatch
from shinobi.snapshots import faults
from astropy.io import fits
@pystep(fits_accesses=[FITSAccess(field='files',mode='write')])
def joint(files:list[FITSImage])->None:
 for path in files:
  with fits.open(path,mode='update') as hdus: hdus[0].data[:]=7
 if sys.argv[1]=='CORRUPT':
  files[0].write_bytes(b'partial corrupt write')
  os._exit(73)
if sys.argv[1]!='CORRUPT': faults.hooks[sys.argv[1]]=lambda:os._exit(73)
_dispatch(joint.step,joint.func,files=[Path('0.fits'),Path('1.fits')],cache_dir='cache')
""")
    process = subprocess.run([sys.executable, str(script), stage], cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 73, process.stderr
    # Fresh recovery process deliberately imports no Astropy. Its durable
    # byte profile and exact oracle decide every group participant.
    recovery = """from pathlib import Path
import sys
from shinobi.snapshots import reconcile
from shinobi.cache import get_cache_manifest
reconcile('cache',get_cache_manifest('cache'),paths={Path('0.fits').resolve(),Path('1.fits').resolve()},exact=True)
assert not any(name.startswith('astropy') for name in sys.modules)
"""
    process = subprocess.run([sys.executable, "-c", recovery], cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    committed = stage in ("S3", "S4", "S5")
    assert [path.read_bytes() == one for path, one in zip(paths, before)] == [not committed] * 2


def test_list_passthrough_requires_order_and_rolls_back(tmp_path):
    class Outputs(BaseModel):
        files: list[FITSImage]

    @pystep(fits_accesses=[FITSAccess(field="files", mode="write")])
    def bad(files: list[FITSImage]) -> Outputs:
        for path in files:
            with fits.open(path, mode="update") as hdus:
                hdus[0].data[:] = 8
        return Outputs(files=files[::-1])

    paths = [image(tmp_path / f"{i}.fits") for i in range(2)]
    before = [path.read_bytes() for path in paths]
    with pytest.raises(DatasetLifecycleViolationError):
        run(bad, tmp_path, files=paths)
    assert [path.read_bytes() for path in paths] == before


def test_create_failure_restores_absence(tmp_path):
    class Outputs(BaseModel):
        file: FITSImage

    @pystep(fits_accesses=[FITSAccess(field="file", mode="create")])
    def create(file: Path) -> Outputs:
        file.write_text("invalid")
        return Outputs(file=file)

    path = tmp_path / "new.fits"
    with pytest.raises(DatasetLifecycleViolationError):
        run(create, tmp_path, file=path)
    assert not path.exists()


def test_create_existing_unknown_refuses(tmp_path):
    class Outputs(BaseModel):
        file: FITSImage

    @pystep(fits_accesses=[FITSAccess(field="file", mode="create")])
    def create(file: Path) -> Outputs:
        image(file)
        return Outputs(file=file)

    path = image(tmp_path / "new.fits")
    before = path.read_bytes()
    with pytest.raises(DatasetLifecycleUnavailableError):
        run(create, tmp_path, file=path)
    assert path.read_bytes() == before


def test_flat_recipe_writer_then_reader(tmp_path):
    class Inputs(BaseModel):
        file: FITSImage

    class Outputs(BaseModel):
        pass

    recipe = Recipe(name="fits-chain", inputs_model=Inputs, outputs_model=Outputs)
    recipe.add_step("write", write, file=InputRef(field="file"))
    recipe.add_step("read", read, file=InputRef(field="file"))
    path = image(tmp_path / "x.fits")
    result = _dispatch(recipe, None, file=path, cache_dir=str(tmp_path / "cache"))
    assert result.success


def test_flat_create_then_outputref_reader(tmp_path):
    from shinobi import OutputRef

    class Inputs(BaseModel):
        destination: Path

    class Created(BaseModel):
        file: FITSImage

    class Empty(BaseModel):
        pass

    @pystep(fits_accesses=[FITSAccess(field="file", mode="create")])
    def create(file: Path) -> Created:
        image(file)
        return Created(file=file)

    recipe = Recipe(name="create-chain", inputs_model=Inputs, outputs_model=Empty)
    recipe.add_step("create", create, file=InputRef(field="destination"))
    recipe.add_step("read", read, file=OutputRef(step="create", field="file"))
    path = tmp_path / "new.fits"
    assert _dispatch(recipe, None, destination=path, cache_dir=str(tmp_path / "cache")).success
    assert _dispatch(recipe, None, destination=path, cache_dir=str(tmp_path / "cache")).success


def test_native_venv_writer(tmp_path):
    # Importable module is transported; using the project venv demonstrates
    # actual subprocess execution in the same controller namespace.
    import importlib.util

    source = tmp_path / "venv_tool.py"
    source.write_text(
        "from pathlib import Path\nfrom shinobi import FITSImage\ndef write(file:FITSImage,value:int=4)->None:\n from astropy.io import fits\n with fits.open(file,mode='update') as hdus: hdus[0].data[:]=value\n"
    )
    spec = importlib.util.spec_from_file_location("venv_tool", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["venv_tool"] = module
    spec.loader.exec_module(module)
    ref = pystep(venv=str(Path(sys.executable).parent.parent), backend="venv", fits_accesses=[FITSAccess(field="file", mode="write")])(module.write)
    path = image(tmp_path / "x.fits")
    assert run(ref, tmp_path, file=path).success
    assert fits.getdata(path)[0, 0] == 4


def test_group_partial_authority_retains_fences(tmp_path):
    paths = [image(tmp_path / f"{i}.fits") for i in range(2)]
    faults.hooks["S3"] = lambda: (_ for _ in ()).throw(RuntimeError("cleanup"))
    with pytest.raises(RuntimeError):
        run(joint, tmp_path, files=paths)
    faults.hooks.clear()
    with pytest.raises(DatasetLifecycleUnavailableError):
        reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths={paths[0]}, exact=True)
    assert all(get_journal(str(tmp_path / "cache")).get(chain_id(path)).marker for path in paths)


def test_unknown_oracle_keeps_sources_and_fences(tmp_path):
    paths = [image(tmp_path / f"{i}.fits") for i in range(2)]
    faults.hooks["S3"] = lambda: (_ for _ in ()).throw(RuntimeError("cleanup"))
    with pytest.raises(RuntimeError):
        run(joint, tmp_path, files=paths)
    faults.hooks.clear()
    journal = get_journal(str(tmp_path / "cache"))
    marker = journal.get(chain_id(paths[0])).marker
    store = FITSLifecycleStore(Path(marker.success_record))
    record = store.attempt()
    store.replace(record.model_copy(update={"leaves": (record.leaves[0].model_copy(update={"mutations": record.leaves[0].mutations[:1]}),)}))
    before = [path.read_bytes() for path in paths]
    with pytest.raises(DatasetLifecycleUnavailableError):
        reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
    assert [path.read_bytes() for path in paths] == before
    assert all(journal.get(chain_id(path)).marker for path in paths)


def test_scalar_oracle_requires_exact_mutation_evidence(tmp_path):
    path = image(tmp_path / "x.fits")
    faults.hooks["S3"] = lambda: (_ for _ in ()).throw(RuntimeError("cleanup"))
    with pytest.raises(RuntimeError):
        run(write, tmp_path, file=path)
    faults.hooks.clear()
    journal = get_journal(str(tmp_path / "cache"))
    marker = journal.get(chain_id(path)).marker
    store = FITSLifecycleStore(Path(marker.success_record))
    record = store.attempt()
    store.replace(record.model_copy(update={"leaves": (record.leaves[0].model_copy(update={"mutations": ()}),)}))
    with pytest.raises(DatasetLifecycleUnavailableError):
        reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths={path}, exact=True)
    assert journal.get(chain_id(path)).marker


def test_existing_write_history_does_not_authorize_create(tmp_path):
    class Outputs(BaseModel):
        file: FITSImage

    @pystep(fits_accesses=[FITSAccess(field="file", mode="create")])
    def unrelated_create(file: Path) -> Outputs:
        image(file)
        return Outputs(file=file)

    path = image(tmp_path / "x.fits")
    run(write, tmp_path, file=path)
    before = path.read_bytes()
    with pytest.raises(DatasetLifecycleUnavailableError, match="CREATE"):
        run(unrelated_create, tmp_path, file=path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("stage", ["S1", "S2"])
def test_same_key_rerun_keeps_physical_pre_run_group(tmp_path, stage):
    paths = [image(tmp_path / f"{i}.fits") for i in range(2)]
    run(joint, tmp_path, files=paths, value=4)
    run(joint, tmp_path, files=paths, value=7)
    before = [path.read_bytes() for path in paths]
    # Requesting the earlier key restores its consumed predecessors. A
    # pre-oracle failure must restore the physical later chain tip.
    faults.hooks[stage] = lambda: (_ for _ in ()).throw(RuntimeError(stage))
    with pytest.raises(RuntimeError):
        run(joint, tmp_path, files=paths, value=4)
    faults.hooks.clear()
    assert [path.read_bytes() for path in paths] == before


def test_later_restore_failure_keeps_entire_group_fenced(tmp_path, monkeypatch):
    import shinobi.snapshots as snapshots

    paths = [image(tmp_path / f"{i}.fits") for i in range(2)]
    before = [path.read_bytes() for path in paths]
    original = snapshots.clone_tree
    armed = False

    def fail_later(source, target, **kwargs):
        if armed and Path(target) == paths[1]:
            raise OSError("later restore failed")
        return original(source, target, **kwargs)

    monkeypatch.setattr(snapshots, "clone_tree", fail_later)

    def before_oracle():
        nonlocal armed
        armed = True
        raise RuntimeError("pre-oracle")

    faults.hooks["S2"] = before_oracle
    with pytest.raises((RuntimeError, OSError, DatasetLifecycleUnavailableError)):
        run(joint, tmp_path, files=paths)
    faults.hooks.clear()
    journal = get_journal(str(tmp_path / "cache"))
    assert all(journal.get(chain_id(path)).marker for path in paths)
    armed = False
    reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
    assert [path.read_bytes() for path in paths] == before


@pytest.mark.parametrize("stage", ["S2", "S3", "CORRUPT", "MISSING"])
def test_scalar_fresh_process_recovery(tmp_path, stage):
    path = image(tmp_path / "x.fits")
    before = path.read_bytes()
    source = tmp_path / "scalar.py"
    source.write_text("""from pathlib import Path
import os,sys
from shinobi import FITSImage,FITSAccess,pystep
from shinobi.steps.dispatch import _dispatch
from shinobi.snapshots import faults
from astropy.io import fits
@pystep(fits_accesses=[FITSAccess(field='file',mode='write')])
def write(file:FITSImage)->None:
 with fits.open(file,mode='update') as hdus: hdus[0].data[:]=7
 if sys.argv[1]=='CORRUPT':
  file.write_bytes(b'partial')
  os._exit(73)
if sys.argv[1]!='CORRUPT': faults.hooks[sys.argv[1]]=lambda:os._exit(73)
_dispatch(write.step,write.func,file=Path('x.fits'),cache_dir='cache')
""")
    process = subprocess.run([sys.executable, str(source), "S2" if stage == "MISSING" else stage], cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 73, process.stderr
    if stage == "MISSING":
        journal = get_journal(str(tmp_path / "cache"))
        marker = journal.get(chain_id(path)).marker
        store = FITSLifecycleStore(Path(marker.success_record))
        store.path.unlink()
        store.lock_path.unlink()
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; from shinobi.snapshots import reconcile; from shinobi.cache import get_cache_manifest; reconcile('cache',get_cache_manifest('cache'),paths={Path('x.fits').resolve()},exact=True)",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    assert (path.read_bytes() == before) == (stage != "S3")


def test_wired_reader_keeps_producer_state_after_descendant(tmp_path):
    from shinobi import OutputRef

    class Outputs(BaseModel):
        file: FITSImage

    class Inputs(BaseModel):
        file: FITSImage
        tag: int = 0

    class Empty(BaseModel):
        pass

    seen = []
    writes = []

    @pystep(fits_accesses=[FITSAccess(field="file", mode="write")])
    def writer(file: FITSImage, value: int) -> Outputs:
        with fits.open(file, mode="update") as hdus:
            hdus[0].data[:] = value
        writes.append(value)
        return Outputs(file=file)

    @pystep()
    def reader(file: FITSImage, tag: int) -> None:
        seen.append((tag, int(fits.getdata(file)[0, 0])))

    recipe = Recipe(name="chain", inputs_model=Inputs, outputs_model=Empty)
    recipe.add_step("first", writer, file=InputRef(field="file"), value=1)
    recipe.add_step("read", reader, file=OutputRef(step="first", field="file"), tag=InputRef(field="tag"))
    recipe.add_step("last", writer, file=OutputRef(step="first", field="file"), value=2)
    recipe.steps[-1].after = ["read"]
    path = image(tmp_path / "x.fits")
    for tag in (0, 0, 1):
        assert _dispatch(recipe, None, file=path, tag=tag, cache=True, cache_dir=str(tmp_path / "cache")).success
    assert seen == [(0, 1), (1, 1)]
    assert writes == [1, 2]
    assert fits.getdata(path)[0, 0] == 2


@pytest.mark.parametrize("materialized", [True, False])
def test_existing_empty_success_oracle_retains_fence(tmp_path, materialized):
    path = image(tmp_path / "x.fits")
    faults.hooks["S3"] = lambda: (_ for _ in ()).throw(RuntimeError("crash"))
    with pytest.raises(RuntimeError):
        run(write, tmp_path, file=path)
    faults.hooks.clear()
    journal = get_journal(str(tmp_path / "cache"))
    marker = journal.get(chain_id(path)).marker
    committed = path.read_bytes()
    FITSLifecycleStore(Path(marker.success_record)).update(lambda data: data.clear())
    if not materialized:
        Path(marker.success_record).unlink()
    with pytest.raises(DatasetLifecycleUnavailableError):
        reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths={path}, exact=True)
    assert path.read_bytes() == committed
    assert journal.get(chain_id(path)).marker == marker


def test_ancestor_hdu_selector_not_applied_to_descendant(tmp_path):
    class Inputs(BaseModel):
        file: FITSImage
        value: int

    class Outputs(BaseModel):
        file: FITSImage

    class Empty(BaseModel):
        pass

    seen = []

    @pystep(fits_accesses=[FITSAccess(field="file", mode="write", hdu=FITSHDU(index=1))])
    def first(file: FITSImage, value: int) -> Outputs:
        with fits.open(file, mode="update") as hdus:
            hdus[1].data[:] = value
        seen.append(value)
        return Outputs(file=file)

    @pystep(fits_accesses=[FITSAccess(field="file", mode="write")])
    def last(file: FITSImage) -> None:
        with fits.open(file, mode="update") as hdus:
            del hdus[1]

    path = image(tmp_path / "x.fits")
    extension = fits.ImageHDU(np.zeros((2, 2)))
    extension.header["CTYPE1"] = "LINEAR"
    extension.header["CTYPE2"] = "LINEAR"
    with fits.open(path, mode="append") as hdus:
        hdus.append(extension)
    recipe = Recipe(name="selectors", inputs_model=Inputs, outputs_model=Empty)
    recipe.add_step("first", first, file=InputRef(field="file"), value=InputRef(field="value"))
    recipe.add_step("last", last, file=InputRef(field="file"))
    recipe.steps[-1].after = ["first"]
    for value in (1, 1, 2):
        assert _dispatch(recipe, None, file=path, value=value, cache_dir=str(tmp_path / "cache")).success
        with fits.open(path) as hdus:
            assert len(hdus) == 1
    assert seen == [1, 2]
    # The current generation's valid HDU0 belongs to a writer omitted from
    # this new workflow. Revalidate its historical contract for planning,
    # then restore HDU1 before the changed ancestor actually executes.
    ancestor_only = Recipe(name="selectors", inputs_model=Inputs, outputs_model=Empty)
    ancestor_only.add_step("first", first, file=InputRef(field="file"), value=InputRef(field="value"))
    assert _dispatch(ancestor_only, None, file=path, value=3, cache_dir=str(tmp_path / "cache")).success
    assert seen == [1, 2, 3]
    with fits.open(path) as hdus:
        assert len(hdus) == 2 and hdus[1].data[0, 0] == 3
    entry = get_cache_manifest(str(tmp_path / "cache")).entry("selectors.first")
    FITSLifecycleStore(tmp_path / ".shinobi" / "fits-attempts" / f"{entry['run_id']}.json").update(lambda data: data.clear())
    before = path.read_bytes()
    with pytest.raises(DatasetLifecycleUnavailableError):
        _dispatch(ancestor_only, None, file=path, value=4, cache_dir=str(tmp_path / "cache"))
    assert path.read_bytes() == before and seen == [1, 2, 3]


def test_truly_missing_oracle_is_not_committed(tmp_path):
    from shinobi.fits_lifecycle import mutation_committed

    assert not mutation_committed(tmp_path / "absent.json", attempt_id="absent", step_path="write", cache_key="key", strict=True)
    assert not (tmp_path / "absent.json.lock").exists()


def test_read_selector_after_writer_matches_read_contract(tmp_path):
    path = image(tmp_path / "x.fits")
    extension = fits.ImageHDU(np.zeros((2, 2)))
    extension.header["CTYPE1"] = extension.header["CTYPE2"] = "LINEAR"
    with fits.open(path, mode="append") as hdus:
        hdus.append(extension)
    run(write, tmp_path, file=path)
    before = path.read_bytes()

    @pystep(fits_accesses=[FITSAccess(field="file", mode="read", hdu=FITSHDU(index=1))])
    def read_extension(file: FITSImage) -> None:
        pass

    for _ in range(2):
        assert run(read_extension, tmp_path, file=path, cache=True).success
    assert path.read_bytes() == before


def _reader_visit_recipe(seen, *, reader_action=None):
    from shinobi import OutputRef

    class Inputs(BaseModel):
        file: FITSImage
        tag: int

    class Outputs(BaseModel):
        file: FITSImage

    class Empty(BaseModel):
        pass

    @pystep(fits_accesses=[FITSAccess(field="file", mode="write")])
    def writer(file: FITSImage, value: int) -> Outputs:
        with fits.open(file, mode="update") as hdus:
            hdus[0].data[:] = value
        return Outputs(file=file)

    @pystep()
    def reader(file: FITSImage, tag: int) -> None:
        seen.append((tag, int(fits.getdata(file)[0, 0])))
        if reader_action:
            reader_action(tag)

    recipe = Recipe(name="visit", inputs_model=Inputs, outputs_model=Empty)
    recipe.add_step("first", writer, file=InputRef(field="file"), value=1)
    recipe.add_step("read", reader, file=OutputRef(step="first", field="file"), tag=InputRef(field="tag"))
    recipe.add_step("last", writer, file=OutputRef(step="first", field="file"), value=2)
    recipe.steps[-1].after = ["read"]
    return recipe


@pytest.mark.parametrize("stage", ["P_MARKED", "P_RESTORED"])
def test_read_visit_fresh_crash_restores_descendant(tmp_path, stage):
    path = image(tmp_path / "x.fits")
    source = tmp_path / "visit.py"
    source.write_text("""from pathlib import Path
import os,sys
sys.path.insert(0, REPOSITORY_ROOT)
from tests.test_fits_lifecycle import _reader_visit_recipe
from shinobi.steps.dispatch import _dispatch
from shinobi.snapshots import faults
recipe=_reader_visit_recipe([])
_dispatch(recipe,None,file=Path('x.fits'),tag=1,cache=True,cache_dir='cache')
faults.hooks[sys.argv[1]]=lambda:os._exit(73)
_dispatch(recipe,None,file=Path('x.fits'),tag=2,cache=True,cache_dir='cache')
""")
    source.write_text(source.read_text().replace("REPOSITORY_ROOT", repr(str(Path(__file__).resolve().parents[1]))))
    process = subprocess.run([sys.executable, str(source), stage], cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 73, process.stderr
    journal = get_journal(str(tmp_path / "cache"))
    marker = journal.get(chain_id(path)).marker
    assert marker and marker.step_path == "visit.read"
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; from shinobi.snapshots import reconcile; from shinobi.cache import get_cache_manifest; reconcile('cache',get_cache_manifest('cache'),paths={Path('x.fits').resolve()},exact=True)",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    assert fits.getdata(path)[0, 0] == 2
    assert not journal.get(chain_id(path)).marker


@pytest.mark.parametrize("omit_writer", [False, True])
def test_read_visit_restore_failure_retains_fence(tmp_path, monkeypatch, omit_writer):
    import shinobi.snapshots as snapshots

    path = image(tmp_path / "x.fits")
    armed = False

    def action(tag):
        nonlocal armed
        if tag == 2:
            armed = True

    seen = []
    recipe = _reader_visit_recipe(seen, reader_action=action)
    _dispatch(recipe, None, file=path, tag=1, cache=True, cache_dir=str(tmp_path / "cache"))
    before = path.read_bytes()
    previous_entry = get_cache_manifest(str(tmp_path / "cache")).entry("visit.read")
    if omit_writer:
        recipe.steps.pop()
    clone = snapshots.clone_tree

    def fail_restore(source, target, **kwargs):
        if armed and Path(target) == path:
            raise OSError("visit restore failed")
        return clone(source, target, **kwargs)

    monkeypatch.setattr(snapshots, "clone_tree", fail_restore)
    with pytest.raises(DatasetLifecycleUnavailableError):
        _dispatch(recipe, None, file=path, tag=2, cache=True, cache_dir=str(tmp_path / "cache"))
    journal = get_journal(str(tmp_path / "cache"))
    marker = journal.get(chain_id(path)).marker
    assert marker
    assert all(Path(one["rollback_source"]).exists() for one in marker.strict_group)
    assert get_cache_manifest(str(tmp_path / "cache")).entry("visit.read") == previous_entry
    attempt = FITSLifecycleStore(tmp_path / ".shinobi" / "fits-attempts" / f"{marker.run_id}.json").attempt()
    assert all(leaf.outcome != "committed" for leaf in attempt.leaves if leaf.step_path == "visit.read")
    armed = False
    reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths={path}, exact=True)
    assert path.read_bytes() == before
    assert not journal.get(chain_id(path)).marker
    assert seen == [(1, 1), (2, 1)]


def test_mixed_reader_writer_visits_producer_without_boundary_visit(tmp_path):
    from shinobi import OutputRef

    class Inputs(BaseModel):
        source: FITSImage
        target: FITSImage
        boundary: FITSImage
        tag: int

    class Outputs(BaseModel):
        source: FITSImage

    class Empty(BaseModel):
        pass

    seen = []

    @pystep(fits_accesses=[FITSAccess(field="source", mode="write")])
    def writer(source: FITSImage, value: int) -> Outputs:
        with fits.open(source, mode="update") as hdus:
            hdus[0].data[:] = value
        return Outputs(source=source)

    @pystep(fits_accesses=[FITSAccess(field="source", mode="read"), FITSAccess(field="boundary", mode="read"), FITSAccess(field="target", mode="write")])
    def copy(source: FITSImage, target: FITSImage, boundary: FITSImage, tag: int) -> None:
        value = int(fits.getdata(source)[0, 0])
        seen.append((tag, value, int(fits.getdata(boundary)[0, 0])))
        with fits.open(target, mode="update") as hdus:
            hdus[0].data[:] = value + tag

    recipe = Recipe(name="mixed", inputs_model=Inputs, outputs_model=Empty)
    recipe.add_step("first", writer, source=InputRef(field="source"), value=1)
    recipe.add_step("copy", copy, source=OutputRef(step="first", field="source"), target=InputRef(field="target"), boundary=InputRef(field="boundary"), tag=InputRef(field="tag"))
    recipe.add_step("last", writer, source=OutputRef(step="first", field="source"), value=2)
    recipe.steps[-1].after = ["copy"]
    paths = {name: image(tmp_path / f"{name}.fits") for name in ("source", "target", "boundary")}
    for tag in (1, 2):
        assert _dispatch(recipe, None, **paths, tag=tag, cache=True, cache_dir=str(tmp_path / "cache")).success
    assert seen == [(1, 1, 0), (2, 1, 0)]
    assert fits.getdata(paths["source"])[0, 0] == 2
    assert fits.getdata(paths["target"])[0, 0] == 3
    with fits.open(paths["boundary"], mode="update") as hdus:
        hdus[0].data[:] = 9
    assert _dispatch(recipe, None, **paths, tag=2, cache=True, cache_dir=str(tmp_path / "cache")).success
    assert seen[-1] == (2, 1, 9)


def test_wired_read_list_visits_all_producer_states_together(tmp_path, monkeypatch):
    from shinobi import OutputRef

    class Inputs(BaseModel):
        files: list[FITSImage]
        tag: int

    class Outputs(BaseModel):
        files: list[FITSImage]

    class Empty(BaseModel):
        pass

    seen = []
    armed = False

    @pystep(fits_accesses=[FITSAccess(field="files", mode="write")])
    def writer(files: list[FITSImage], value: int) -> Outputs:
        for path in files:
            with fits.open(path, mode="update") as hdus:
                hdus[0].data[:] = value
        return Outputs(files=files)

    @pystep(fits_accesses=[FITSAccess(field="files", mode="read")])
    def reader(files: list[FITSImage], tag: int) -> None:
        nonlocal armed
        armed = tag == 3
        seen.append((tag, [int(fits.getdata(path)[0, 0]) for path in files]))

    recipe = Recipe(name="list_visit", inputs_model=Inputs, outputs_model=Empty)
    recipe.add_step("first", writer, files=InputRef(field="files"), value=1)
    recipe.add_step("read", reader, files=OutputRef(step="first", field="files"), tag=InputRef(field="tag"))
    recipe.add_step("last", writer, files=OutputRef(step="first", field="files"), value=2)
    recipe.steps[-1].after = ["read"]
    paths = [image(tmp_path / f"{index}.fits") for index in range(2)]
    for tag in (1, 2):
        assert _dispatch(recipe, None, files=paths, tag=tag, cache=True, cache_dir=str(tmp_path / "cache")).success
    assert seen == [(1, [1, 1]), (2, [1, 1])]
    assert all(fits.getdata(path)[0, 0] == 2 for path in paths)

    previous_entry = get_cache_manifest(str(tmp_path / "cache")).entry("list_visit.read")
    recipe.steps.pop()
    import shinobi.snapshots as snapshots

    clone = snapshots.clone_tree

    def fail_second(source, target, **kwargs):
        if armed and Path(target) == paths[1]:
            raise OSError("second visit restoration failed")
        return clone(source, target, **kwargs)

    monkeypatch.setattr(snapshots, "clone_tree", fail_second)
    with pytest.raises(DatasetLifecycleUnavailableError):
        _dispatch(recipe, None, files=paths, tag=3, cache=True, cache_dir=str(tmp_path / "cache"))
    journal = get_journal(str(tmp_path / "cache"))
    markers = [journal.get(chain_id(path)).marker for path in paths]
    assert all(markers)
    assert all(Path(one["rollback_source"]).exists() for one in markers[0].strict_group)
    assert get_cache_manifest(str(tmp_path / "cache")).entry("list_visit.read") == previous_entry
    attempt = FITSLifecycleStore(tmp_path / ".shinobi" / "fits-attempts" / f"{markers[0].run_id}.json").attempt()
    assert all(leaf.outcome != "committed" for leaf in attempt.leaves if leaf.step_path == "list_visit.read")
    with pytest.raises(DatasetLifecycleUnavailableError):
        reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths={paths[0]}, exact=True)
    armed = False
    reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
    assert all(fits.getdata(path)[0, 0] == 2 and not journal.get(chain_id(path)).marker for path in paths)


def test_read_visit_before_observation_failure_restores_descendant(tmp_path, monkeypatch):
    from shinobi.fits_lifecycle import StrictFITSLeaf

    path = image(tmp_path / "x.fits")
    seen = []
    recipe = _reader_visit_recipe(seen)
    _dispatch(recipe, None, file=path, tag=1, cache=True, cache_dir=str(tmp_path / "cache"))
    before = path.read_bytes()
    observe = StrictFITSLeaf.observe_before

    def fail_observation(self, guard):
        if self.step_path == "visit.read" and self.prepared["tag"] == 2:
            assert fits.getdata(path)[0, 0] == 1
            raise DatasetLifecycleUnavailableError("pre-run observation failed")
        return observe(self, guard)

    monkeypatch.setattr(StrictFITSLeaf, "observe_before", fail_observation)
    with pytest.raises(DatasetLifecycleUnavailableError):
        _dispatch(recipe, None, file=path, tag=2, cache=True, cache_dir=str(tmp_path / "cache"))
    assert path.read_bytes() == before
    assert not get_journal(str(tmp_path / "cache")).get(chain_id(path)).marker
    assert seen == [(1, 1)]


def test_read_visit_cleanup_exception_does_not_publish(tmp_path, monkeypatch):
    from shinobi.snapshots import SnapshotGuard
    from shinobi.exceptions import StepError

    path = image(tmp_path / "x.fits")
    armed = False

    def action(tag):
        nonlocal armed
        armed = tag == 2

    recipe = _reader_visit_recipe([], reader_action=action)
    _dispatch(recipe, None, file=path, tag=1, cache=True, cache_dir=str(tmp_path / "cache"))
    recipe.steps.pop()
    before = path.read_bytes()
    previous_entry = get_cache_manifest(str(tmp_path / "cache")).entry("visit.read")
    discard = SnapshotGuard._discard_trash

    def fail_cleanup(self, plan):
        if armed and self.step_path == "visit.read":
            raise OSError("visit cleanup failed")
        return discard(self, plan)

    monkeypatch.setattr(SnapshotGuard, "_discard_trash", fail_cleanup)
    before_attempts = set((tmp_path / ".shinobi" / "fits-attempts").glob("*.json"))
    with pytest.raises(StepError, match="visit cleanup failed"):
        _dispatch(recipe, None, file=path, tag=2, cache=True, cache_dir=str(tmp_path / "cache"))
    assert path.read_bytes() == before
    assert get_cache_manifest(str(tmp_path / "cache")).entry("visit.read") == previous_entry
    for attempt_path in set((tmp_path / ".shinobi" / "fits-attempts").glob("*.json")) - before_attempts:
        attempt = FITSLifecycleStore(attempt_path).attempt()
        assert all(leaf.outcome != "committed" for leaf in attempt.leaves if leaf.step_path == "visit.read")


def test_atomic_passthrough_validates_live_selected_hdu(tmp_path):
    class Outputs(BaseModel):
        file: FITSImage

    calls = []

    @pystep(fits_accesses=[FITSAccess(field="file", mode="write", hdu=FITSHDU(index=1))])
    def first(file: FITSImage) -> Outputs:
        calls.append("first")
        return Outputs(file=file)

    @pystep(fits_accesses=[FITSAccess(field="file", mode="write")])
    def last(file: FITSImage) -> None:
        with fits.open(file, mode="update") as hdus:
            del hdus[1]

    path = image(tmp_path / "x.fits")
    extension = fits.ImageHDU(np.zeros((2, 2)))
    extension.header["CTYPE1"] = extension.header["CTYPE2"] = "LINEAR"
    with fits.open(path, mode="append") as hdus:
        hdus.append(extension)
    assert run(first, tmp_path, file=path).success
    assert run(last, tmp_path, file=path).success
    assert run(first, tmp_path, file=path).success
    assert calls == ["first", "first"]
    with fits.open(path) as hdus:
        assert len(hdus) == 2


@pytest.mark.parametrize("create", [False, True])
def test_committed_index_gap_reuses_complete_payload(tmp_path, monkeypatch, create):
    from shinobi.cache import CacheManifest

    class Outputs(BaseModel):
        file: FITSImage
        count: int

    calls = []

    @pystep(fits_accesses=[FITSAccess(field="file", mode="create" if create else "write")])
    def emit(file: Path if create else FITSImage) -> Outputs:
        calls.append(1)
        if create:
            image(file)
        else:
            with fits.open(file, mode="update") as hdus:
                hdus[0].data[:] = 4
        return Outputs(file=file, count=37)

    # CREATE accepts the ordinary output-path parameter without inspecting absence.
    path = tmp_path / "x.fits"
    if not create:
        image(path)
    record = CacheManifest.record

    def fail_index(*args, **kwargs):
        raise OSError("cache index failed")

    monkeypatch.setattr(CacheManifest, "record", fail_index)
    with pytest.raises(OSError):
        run(emit, tmp_path, file=path)
    journal = get_journal(str(tmp_path / "cache"))
    generation = journal.get(chain_id(path)).generation(journal.get(chain_id(path)).head)
    assert generation.fits_commit
    committed = path.read_bytes()
    monkeypatch.setattr(CacheManifest, "record", record)
    assert get_cache_manifest(str(tmp_path / "cache")).entry("emit") is None
    result = run(emit, tmp_path, file=path)
    assert result.cached and result.outputs.count == 37
    assert calls == [1] and path.read_bytes() == committed


def test_same_key_index_gap_repairs_new_scalar_outputs(tmp_path, monkeypatch):
    from shinobi.cache import CacheManifest

    class Outputs(BaseModel):
        file: FITSImage
        count: int

    calls = []

    @pystep(fits_accesses=[FITSAccess(field="file", mode="write")])
    def emit(file: FITSImage) -> Outputs:
        calls.append(1)
        with fits.open(file, mode="update") as hdus:
            hdus[0].data[:] = len(calls)
        return Outputs(file=file, count=len(calls))

    path = image(tmp_path / "x.fits")
    assert run(emit, tmp_path, file=path).outputs.count == 1
    check, record = CacheManifest.check, CacheManifest.record
    force_miss = True

    def miss_once(self, *args, **kwargs):
        nonlocal force_miss
        if force_miss and kwargs.get("candidate") is None:
            force_miss = False
            return None
        return check(self, *args, **kwargs)

    monkeypatch.setattr(CacheManifest, "check", miss_once)
    monkeypatch.setattr(CacheManifest, "record", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("index failed")))
    with pytest.raises(OSError):
        run(emit, tmp_path, file=path)
    monkeypatch.setattr(CacheManifest, "record", record)

    @pystep(fits_accesses=[FITSAccess(field="file", mode="write")])
    def tail(file: FITSImage) -> None:
        with fits.open(file, mode="update") as hdus:
            hdus[0].data[:] = 9

    assert run(tail, tmp_path, file=path).success
    result = run(emit, tmp_path, file=path)
    assert result.cached and result.outputs.count == 2 and len(calls) == 2
    assert fits.getdata(path)[0, 0] == 9


@pytest.mark.parametrize("group", [False, True])
def test_same_key_s2_rollback_restores_frozen_commit_reference(tmp_path, monkeypatch, group):
    from shinobi.cache import CacheManifest

    paths = [image(tmp_path / f"{index}.fits") for index in range(2 if group else 1)]
    step = joint if group else write
    kwargs = {"files": paths} if group else {"file": paths[0]}
    run(step, tmp_path, **kwargs)
    journal = get_journal(str(tmp_path / "cache"))
    before = [journal.get(chain_id(path)).generation(journal.get(chain_id(path)).head).fits_commit for path in paths]
    check = CacheManifest.check

    def miss(self, *args, **kwargs):
        return None if kwargs.get("candidate") is None else check(self, *args, **kwargs)

    monkeypatch.setattr(CacheManifest, "check", miss)
    faults.hooks["S2"] = lambda: (_ for _ in ()).throw(RuntimeError("failed after refs"))
    with pytest.raises(RuntimeError):
        run(step, tmp_path, **kwargs)
    faults.hooks.clear()
    after = [journal.get(chain_id(path)).generation(journal.get(chain_id(path)).head).fits_commit for path in paths]
    assert after == before
    assert all(not journal.get(chain_id(path)).marker for path in paths)


@pytest.mark.parametrize("create", [False, True])
def test_fresh_process_committed_index_gap_exact_repair(tmp_path, create):
    path = tmp_path / "x.fits"
    if not create:
        image(path)
    script = tmp_path / "index_gap.py"
    script.write_text("""from pathlib import Path
import os,sys
from pydantic import BaseModel
from shinobi import FITSImage,FITSAccess,pystep
from shinobi.steps.dispatch import _dispatch
from shinobi.cache import CacheManifest
from astropy.io import fits
import numpy as np
create=sys.argv[1]=='create'
class Outputs(BaseModel):
 file:FITSImage
 count:int
@pystep(fits_accesses=[FITSAccess(field='file',mode='create' if create else 'write')])
def emit(file:Path if create else FITSImage)->Outputs:
 if create:
  h=fits.PrimaryHDU(np.zeros((2,3),dtype=np.int16));h.header['CTYPE1']='LINEAR';h.header['CTYPE2']='LINEAR';h.writeto(file)
 else:
  with fits.open(file,mode='update') as h:h[0].data[:]=4
 return Outputs(file=file,count=81)
if sys.argv[2]=='crash':CacheManifest.record=lambda *a,**k:os._exit(73)
else:
 from shinobi.ownership import reconcile_ownership
 reconcile_ownership(Path.cwd())
result=_dispatch(emit.step,emit.func,file=Path('x.fits'),cache_dir='cache')
assert result.cached and result.outputs.count==81
""")
    mode = "create" if create else "write"
    process = subprocess.run([sys.executable, str(script), mode, "crash"], cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 73, process.stderr
    committed = path.read_bytes()
    process = subprocess.run([sys.executable, str(script), mode, "resume"], cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert path.read_bytes() == committed


@pytest.mark.parametrize("group", [False, True])
def test_committed_reference_participant_mismatch_retains_fences(tmp_path, group):
    paths = [image(tmp_path / f"{index}.fits") for index in range(2 if group else 1)]
    faults.hooks["S3"] = lambda: (_ for _ in ()).throw(RuntimeError("leave fence"))
    with pytest.raises(RuntimeError):
        run(joint if group else write, tmp_path, **({"files": paths} if group else {"file": paths[0]}))
    faults.hooks.clear()
    journal = get_journal(str(tmp_path / "cache"))
    chain = journal.get(chain_id(paths[0]))
    old_reference = chain.generation(chain.head).fits_commit
    bad = dict(old_reference)
    bad["participants"] = list(old_reference["participants"])
    from shinobi.snapshots import state_name

    extra = dict(bad["participants"][0], root=str(tmp_path / "other.fits"), field="other", label="other", element_index=None, successor=state_name(bad["cache_key"], "other"))
    bad["participants"].append(extra)
    journal.update_chain(chain_id(paths[0]), lambda one: setattr(one.generation(one.head), "fits_commit", bad) or one)
    before = [path.read_bytes() for path in paths]
    with pytest.raises(DatasetLifecycleUnavailableError):
        reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
    assert [path.read_bytes() for path in paths] == before
    assert all(journal.get(chain_id(path)).marker for path in paths)


@pytest.mark.parametrize("compatible", [False, True])
def test_recipe_reexport_validates_current_declared_hdu(tmp_path, compatible):
    from shinobi import OutputRef

    class Paths(BaseModel):
        file: FITSImage

    @pystep(fits_accesses=[FITSAccess(field="file", mode="write", hdu=FITSHDU(index=1))])
    def first(file: FITSImage) -> Paths:
        return Paths(file=file)

    @pystep(fits_accesses=[FITSAccess(field="file", mode="write", hdu=FITSHDU(index=1))])
    def last(file: FITSImage) -> None:
        if not compatible:
            with fits.open(file, mode="update") as hdus:
                del hdus[0].header["CTYPE1"]
                del hdus[0].header["CTYPE2"]

    path = image(tmp_path / "x.fits")
    extension = fits.ImageHDU(np.zeros((2, 2)))
    extension.header["CTYPE1"] = extension.header["CTYPE2"] = "LINEAR"
    with fits.open(path, mode="append") as hdus:
        hdus.append(extension)
    recipe = Recipe(name="exports", inputs_model=Paths, outputs_model=Paths)
    recipe.add_step("first", first, file=InputRef(field="file"))
    recipe.add_step("last", last, file=OutputRef(step="first", field="file"))
    recipe.set_output("file", OutputRef(step="first", field="file"))
    if compatible:
        result = _dispatch(recipe, None, file=path, cache=True, cache_dir=str(tmp_path / "cache"))
        assert result.success and result.outputs.file == path
    else:
        with pytest.raises(DatasetLifecycleViolationError, match="FITS output"):
            _dispatch(recipe, None, file=path, cache=True, cache_dir=str(tmp_path / "cache"))


@pytest.mark.parametrize("group", [False, True])
def test_fresh_same_key_s2_recovers_previous_references(tmp_path, group):
    paths = [image(tmp_path / f"{index}.fits") for index in range(2 if group else 1)]
    step = joint if group else write
    run(step, tmp_path, **({"files": paths} if group else {"file": paths[0]}))
    journal = get_journal(str(tmp_path / "cache"))
    before = [journal.get(chain_id(path)).generation(journal.get(chain_id(path)).head).fits_commit for path in paths]
    source = tmp_path / "same_key.py"
    source.write_text(
        """import os,sys
from pathlib import Path
sys.path.insert(0,REPOSITORY_ROOT)
from tests.test_fits_lifecycle import write,joint
from shinobi.cache import CacheManifest
from shinobi.snapshots import faults
CacheManifest.check=lambda *a,**k:None
faults.hooks['S2']=lambda:os._exit(73)
if sys.argv[1]=='group':joint(files=[Path('0.fits').resolve(),Path('1.fits').resolve()],cache_dir=str(Path('cache').resolve()))
else:write(file=Path('0.fits').resolve(),cache_dir=str(Path('cache').resolve()))
""".replace("REPOSITORY_ROOT", repr(str(Path(__file__).resolve().parents[1])))
    )
    process = subprocess.run([sys.executable, str(source), "group" if group else "scalar"], cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 73, process.stderr
    assert [journal.get(chain_id(path)).head for path in paths] == [reference["participant"]["successor"] for reference in before]
    reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
    assert [journal.get(chain_id(path)).generation(journal.get(chain_id(path)).head).fits_commit for path in paths] == before


def test_first_failed_writer_boundary_can_retry(tmp_path):
    path = image(tmp_path / "x.fits")
    faults.hooks["S2"] = lambda: (_ for _ in ()).throw(RuntimeError("before oracle"))
    with pytest.raises(RuntimeError):
        run(write, tmp_path, file=path)
    faults.hooks.clear()
    assert run(write, tmp_path, file=path).success


@pytest.mark.parametrize("damage", ["extra-field", "wrong-oracle", "wrong-key"])
def test_damaged_exact_reference_retains_fences(tmp_path, damage):
    path = image(tmp_path / "x.fits")
    faults.hooks["S3"] = lambda: (_ for _ in ()).throw(RuntimeError("leave committed fence"))
    with pytest.raises(RuntimeError):
        run(write, tmp_path, file=path)
    faults.hooks.clear()
    journal = get_journal(str(tmp_path / "cache"))
    chain = journal.get(chain_id(path))
    bad = dict(chain.generation(chain.head).fits_commit)
    if damage == "extra-field":
        bad["unexpected"] = True
    elif damage == "wrong-oracle":
        bad["oracle"] = str(tmp_path / "other.json")
    else:
        bad["cache_key"] = "0" * 64
    journal.update_chain(chain_id(path), lambda one: setattr(one.generation(one.head), "fits_commit", bad) or one)
    before = path.read_bytes()
    with pytest.raises(DatasetLifecycleUnavailableError):
        reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths={path}, exact=True)
    raw = journal.read()[chain_id(path)]
    assert raw["marker"] and path.read_bytes() == before
    assert all(Path(one["rollback_source"]).exists() for one in raw["marker"]["strict_group"])


def test_head_observation_never_discovers_oracle_through_cache(tmp_path, monkeypatch):
    from shinobi.cache import CacheManifest
    from shinobi.fits_lifecycle import observe_journal_head

    path = image(tmp_path / "x.fits")
    run(write, tmp_path, file=path)
    journal = get_journal(str(tmp_path / "cache"))
    chain = journal.get(chain_id(path))
    monkeypatch.setattr(CacheManifest, "read", lambda *args: (_ for _ in ()).throw(AssertionError("cache discovery forbidden")))
    assert observe_journal_head(path, chain, workspace=tmp_path, cache_dir=str(tmp_path / "cache")).root == path


@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize("remove_marker_discriminator", [False, True])
def test_new_committed_missing_reference_retains_all_fences(tmp_path, group, remove_marker_discriminator):
    paths = [image(tmp_path / f"{index}.fits") for index in range(2 if group else 1)]
    faults.hooks["S3"] = lambda: (_ for _ in ()).throw(RuntimeError("committed before cleanup"))
    with pytest.raises(RuntimeError):
        run(joint if group else write, tmp_path, **({"files": paths} if group else {"file": paths[0]}))
    faults.hooks.clear()
    journal = get_journal(str(tmp_path / "cache"))

    def remove(one):
        if one.path == str(paths[0]):
            one.generation(one.head).fits_commit = None
        if remove_marker_discriminator:
            one.marker.strict_group = [{key: value for key, value in item.items() if key != "rollback_fits_commit"} for item in one.marker.strict_group]
        return one

    journal.update_chains([(chain_id(path), remove) for path in paths])
    before = [path.read_bytes() for path in paths]
    with pytest.raises(DatasetLifecycleUnavailableError, match="missing.*reference"):
        reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
    assert [path.read_bytes() for path in paths] == before
    markers = [journal.get(chain_id(path)).marker for path in paths]
    assert all(markers)
    assert all(Path(item["rollback_source"]).exists() for item in markers[0].strict_group)


@pytest.mark.parametrize("group", [False, True])
def test_genuine_legacy_completed_marker_recovery_remains_supported(tmp_path, group):
    paths = [image(tmp_path / f"{index}.fits") for index in range(2 if group else 1)]
    faults.hooks["S3"] = lambda: (_ for _ in ()).throw(RuntimeError("legacy completed before cleanup"))
    with pytest.raises(RuntimeError):
        run(joint if group else write, tmp_path, **({"files": paths} if group else {"file": paths[0]}))
    faults.hooks.clear()
    journal = get_journal(str(tmp_path / "cache"))
    marker = journal.get(chain_id(paths[0])).marker
    # Model the actual pre-reference schema: no leaf publication, no frozen\n    # reference discriminator and no generation references.
    store = FITSLifecycleStore(Path(marker.success_record))

    def legacy_oracle(data):
        for leaf in data["record"]["leaves"]:
            leaf.pop("publication", None)

    store.update(legacy_oracle)

    def legacy_chain(one):
        one.generation(one.head).fits_commit = None
        one.marker.strict_group = [{key: value for key, value in item.items() if key != "rollback_fits_commit"} for item in one.marker.strict_group]
        return one

    journal.update_chains([(chain_id(path), legacy_chain) for path in paths])
    before = [path.read_bytes() for path in paths]
    notes = reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths=set(paths), exact=True)
    assert notes and [path.read_bytes() for path in paths] == before
    assert all(not journal.get(chain_id(path)).marker for path in paths)
