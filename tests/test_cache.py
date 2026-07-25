import os
from pathlib import Path

import pytest
import shinobi
from pydantic import BaseModel

from shinobi import cache
from shinobi.backends.recording import RecordingBackend
from shinobi.cache import CacheManifest, compute_cache_key, get_cache_manifest, invalidate_path_hashes
from shinobi.results import StepResult
from shinobi.steps import Cab, register_step_backend
from shinobi.steps.schema import Mutability
from shinobi.steps.dispatch import _dispatch


class Inputs(BaseModel):
    x: int = 1


class Outputs(BaseModel):
    y: str | None = None


def _cab(cache_dir: Path, **kwargs) -> tuple[Cab, RecordingBackend]:
    recorder = RecordingBackend()
    register_step_backend("record", recorder)
    cab = Cab(
        name="tool",
        command="tool",
        inputs_model=Inputs,
        outputs_model=Outputs,
        backend="record",
        cache=True,
        cache_dir=str(cache_dir),
        **kwargs,
    )
    return cab, recorder


def test_cab_run_twice_with_unchanged_inputs_executes_once(tmp_path):
    cab, recorder = _cab(tmp_path)
    _dispatch(cab, None, x=1)
    _dispatch(cab, None, x=1)
    assert len(recorder.calls) == 1


def test_cab_run_with_different_params_executes_twice(tmp_path):
    cab, recorder = _cab(tmp_path)
    _dispatch(cab, None, x=1)
    _dispatch(cab, None, x=2)
    assert len(recorder.calls) == 2


def test_second_run_result_is_marked_cached(tmp_path):
    cab, _recorder = _cab(tmp_path)
    first = _dispatch(cab, None, x=1)
    second = _dispatch(cab, None, x=1)
    assert first.cached is False
    assert second.cached is True


def test_cache_disabled_by_default_executes_every_time(tmp_path):
    recorder = RecordingBackend()
    register_step_backend("record", recorder)
    cab = Cab(name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs, backend="record")
    _dispatch(cab, None, x=1)
    _dispatch(cab, None, x=1)
    assert len(recorder.calls) == 2


# -- pystep coverage (the dominant step shape in real usage -- a bare Scope
# whose adapter never calls ctx.run(), so caching must gate _dispatch itself,
# not just _run_cab) --


class CounterOutputs(BaseModel):
    count: int = 0


def _make_counter_step(image=None):
    calls = {"n": 0}

    @shinobi.pystep(image=image)
    def counter(ctx, x: int = 1) -> CounterOutputs:
        calls["n"] += 1
        return CounterOutputs(count=calls["n"])

    return counter, calls


def test_pystep_run_twice_with_unchanged_inputs_executes_once(tmp_path):
    counter, calls = _make_counter_step()
    _dispatch(counter.step, counter.func, cache=True, cache_dir=str(tmp_path), x=1)
    _dispatch(counter.step, counter.func, cache=True, cache_dir=str(tmp_path), x=1)
    assert calls["n"] == 1


def test_editing_pystep_source_forces_rerun(tmp_path):
    @shinobi.pystep()
    def step_v1(ctx, x: int = 1) -> CounterOutputs:
        return CounterOutputs(count=1)

    @shinobi.pystep()
    def step_v2(ctx, x: int = 1) -> CounterOutputs:
        return CounterOutputs(count=2)

    r1 = _dispatch(step_v1.step, step_v1.func, cache=True, cache_dir=str(tmp_path), x=1)
    r2 = _dispatch(step_v2.step, step_v2.func, cache=True, cache_dir=str(tmp_path), x=1)
    assert r1.cached is False
    assert r2.cached is False
    assert r2.count == 2


class FileInputs(BaseModel):
    src: Path


class FileOutputs(BaseModel):
    marker: int = 0


def test_touching_input_file_mtime_forces_rerun(tmp_path):
    src = tmp_path / "input.dat"
    src.write_text("hello")
    calls = {"n": 0}

    @shinobi.pystep()
    def read_step(ctx, src: Path) -> FileOutputs:
        calls["n"] += 1
        return FileOutputs(marker=calls["n"])

    cache_dir = tmp_path / "cache"
    _dispatch(read_step.step, read_step.func, cache=True, cache_dir=str(cache_dir), src=src)
    _dispatch(read_step.step, read_step.func, cache=True, cache_dir=str(cache_dir), src=src)
    assert calls["n"] == 1

    # touch (mtime changes, size doesn't) -> cache key changes, forces a rerun
    os_utime = src.stat().st_mtime + 5
    import os

    os.utime(src, (os_utime, os_utime))
    _dispatch(read_step.step, read_step.func, cache=True, cache_dir=str(cache_dir), src=src)
    assert calls["n"] == 2


class InPlaceInputs(BaseModel):
    vis: Path


class InPlaceOutputs(BaseModel):
    vis: Path


def test_inplace_mutated_path_not_invalidated_by_its_own_mtime(tmp_path):
    """`vis` is declared on both inputs_model and outputs_model (the
    dominant caracal2 pattern -- flagging/calibration steps read and
    write the same MS) -- its own mtime moving between runs must not,
    by itself, count as "the input changed".
    """
    vis = tmp_path / "data.ms"
    vis.write_text("original")
    calls = {"n": 0}

    @shinobi.pystep()
    def mutate_in_place(ctx, vis: Path) -> InPlaceOutputs:
        calls["n"] += 1
        vis.write_text(f"mutated {calls['n']}")  # simulates flagdata-style in-place rewrite
        return InPlaceOutputs(vis=vis)

    cache_dir = tmp_path / "cache"
    _dispatch(mutate_in_place.step, mutate_in_place.func, cache=True, cache_dir=str(cache_dir), vis=vis)
    assert calls["n"] == 1

    # a second run, params unchanged -- despite `vis`'s mtime/content having
    # just been rewritten by the first run's own side effect
    _dispatch(mutate_in_place.step, mutate_in_place.func, cache=True, cache_dir=str(cache_dir), vis=vis)
    assert calls["n"] == 1


class MutableOnlyOutputs(BaseModel):
    """No path fields -- the flag/gaincal/applycal shape, which rewrites its
    input in place and declares that with `Mutability.MUTABLE` rather than by
    re-declaring the MS as an output. Mirrors `_mutator`'s `OkOut` in
    `tests/test_offload_slurm.py`.
    """

    ok: bool = True


def test_mutable_declared_input_is_not_keyed_on_content_it_overwrites(tmp_path):
    """Regression: `compute_cache_key` derived "mutated in place" only from
    `input_paths & output_paths`, ignoring `Mutability.MUTABLE`. A cab
    declaring its MS mutable but not re-declaring it as an output was keyed
    on the very bytes it was about to overwrite, so its own side effect moved
    the key and it re-ran on every resumed run, forever -- the bug
    `test_inplace_mutated_path_not_invalidated_by_its_own_mtime` pins for the
    other spelling.
    """
    vis = tmp_path / "data.ms"
    vis.write_text("original")
    cab = Cab(
        name="applycal",
        command="applycal",
        inputs_model=InPlaceInputs,
        outputs_model=MutableOnlyOutputs,
        input_mutability={"vis": Mutability.MUTABLE},
    )

    before = compute_cache_key(cab, None, {"vis": vis})
    vis.write_text("rewritten in place by the step itself")
    # Without this the memoized fingerprint would serve both calls the same
    # answer and the test would pass no matter what the key logic does.
    invalidate_path_hashes()
    after = compute_cache_key(cab, None, {"vis": vis})

    assert before == after, "a MUTABLE input's own rewrite must not move the key"


def test_two_boundary_paths_with_identical_content_do_not_share_a_key(tmp_path):
    """Regression: an unwired boundary path contributed only its content
    hash, never its path string. `cp -a`/`rsync -a`/`tar -x` preserve mtimes,
    so two identically-laid-out copies of one MS hashed identically and a step
    repointed from one to the other took a false cache hit.
    """
    cab = Cab(name="t", command="t", inputs_model=InPlaceInputs, outputs_model=Outputs)

    a, b = tmp_path / "A.ms", tmp_path / "B.ms"
    for root in (a, b):
        (root / "ANTENNA").mkdir(parents=True)
        (root / "table.dat").write_text("x")
        (root / "ANTENNA" / "table.f0").write_text("y")
    for root in (a, b):  # what `cp -a` leaves behind
        for path in root.rglob("*"):
            os.utime(path, (1700000000, 1700000000))

    assert compute_cache_key(cab, None, {"vis": a}) != compute_cache_key(cab, None, {"vis": b})


def test_two_missing_boundary_paths_do_not_share_a_key(tmp_path):
    """Same root cause, starker: a non-existent path hashes to `None`, so
    every absent boundary input keyed identically to every other one.
    """
    cab = Cab(name="t", command="t", inputs_model=InPlaceInputs, outputs_model=Outputs)
    assert compute_cache_key(cab, None, {"vis": tmp_path / "gone1.ms"}) != compute_cache_key(cab, None, {"vis": tmp_path / "gone2.ms"})


def test_reader_after_an_inplace_mutation_is_not_served_a_stale_memoized_hash(tmp_path):
    """`_hash_path` is memoized (an MS is thousands of files and the walk
    runs per unwired boundary input, per step, per run), and this is the
    false hit that memo must never produce.

    A reader hashes the MS, a mutating step rewrites it in place -- adding no
    memo entry of its own, since a path that is both input and output is
    dropped from the key -- and a later reader with identical params must
    still see the *post*-mutation content. Serving it the first reader's
    cached hash would key it identically and silently skip it.
    """
    vis = tmp_path / "data.ms"
    vis.mkdir()
    (vis / "table.dat").write_text("original")
    reads = {"n": 0}

    @shinobi.pystep()
    def read_vis(ctx, src: Path) -> FileOutputs:
        reads["n"] += 1
        return FileOutputs(marker=reads["n"])

    @shinobi.pystep()
    def mutate_in_place(ctx, vis: Path) -> InPlaceOutputs:
        (vis / "table.dat").write_text("mutated -- longer, so size moves too")
        return InPlaceOutputs(vis=vis)

    cache_dir = tmp_path / "cache"
    kw = {"cache": True, "cache_dir": str(cache_dir)}

    _dispatch(read_vis.step, read_vis.func, src=vis, **kw)
    assert reads["n"] == 1

    _dispatch(mutate_in_place.step, mutate_in_place.func, vis=vis, **kw)

    # Same params as the first read, but the bytes underneath changed.
    _dispatch(read_vis.step, read_vis.func, src=vis, **kw)
    assert reads["n"] == 2, "second reader was served a pre-mutation memoized hash"


def test_repeated_hash_of_one_path_walks_it_once_between_executions(tmp_path):
    """The win the memo exists for: several unwired boundary fields naming
    one MS are walked once, not once each.
    """
    import shinobi.cache as cache_mod

    vis = tmp_path / "data.ms"
    vis.mkdir()
    (vis / "table.dat").write_text("x")

    cache_mod.invalidate_path_hashes()
    walks = []
    original_scandir = cache_mod.os.scandir

    def counting_scandir(path, *args, **kwargs):
        walks.append(path)
        return original_scandir(path, *args, **kwargs)

    cache_mod.os.scandir = counting_scandir
    try:
        first = cache_mod._hash_path(vis)
        second = cache_mod._hash_path(vis)
    finally:
        cache_mod.os.scandir = original_scandir

    assert first == second
    assert len(walks) == 1

    # ...and the memo is dropped the moment anything could have written.
    cache_mod.invalidate_path_hashes()
    cache_mod.os.scandir = counting_scandir
    try:
        cache_mod._hash_path(vis)
    finally:
        cache_mod.os.scandir = original_scandir
    assert len(walks) == 2


# -- the directory branch of `_hash_path`, which had no coverage at all --


def _make_ms(root: Path) -> Path:
    """A casacore-shaped tree: a table at the top plus subtable directories,
    each with its own descriptor, data-manager and lock files. Modelled on
    the real MSs this project runs against (~60-140 files); the shape is what
    matters here, not the bytes.
    """
    root.mkdir(parents=True)
    for name in ("table.dat", "table.info", "table.f0", "table.lock"):
        (root / name).write_text(name)
    for sub in ("ANTENNA", "FIELD", "SPECTRAL_WINDOW"):
        (root / sub).mkdir()
        for name in ("table.dat", "table.f0", "table.lock"):
            (root / sub / name).write_text(f"{sub}/{name}")
    return root


def _reference_walk(path: Path):
    """The pre-scandir implementation, kept as the equivalence oracle: any
    divergence in the rewrite shows up as a digest change and a silently
    invalidated cache.
    """
    entries = []
    for root, _dirs, files in os.walk(path):
        for fname in files:
            fpath = Path(root) / fname
            st = fpath.stat()
            entries.append([str(fpath.relative_to(path)), st.st_mtime_ns, st.st_size])
    return sorted(entries)


def test_walk_fingerprint_matches_the_reference_implementation(tmp_path):
    """The rewrite must be byte-identical on every tree the old code could
    actually return for -- otherwise it silently invalidates every cached
    step with a directory input.
    """
    ms = _make_ms(tmp_path / "data.ms")
    (ms / "FIELD" / "link.dat").symlink_to(ms / "table.dat")  # symlinked file: followed
    (ms / "linkdir").symlink_to(ms / "ANTENNA")  # symlinked dir: not descended, not listed

    assert cache._walk_fingerprint(ms) == _reference_walk(ms)


def test_walk_fingerprint_is_order_independent(tmp_path):
    """Two trees with identical contents built in different creation orders
    must key identically -- `scandir` yields in directory order, which is not
    creation order and not sorted.
    """
    a, b = tmp_path / "a.ms", tmp_path / "b.ms"
    a.mkdir()
    b.mkdir()
    for name in ("z.dat", "a.dat", "m.dat"):
        (a / name).write_text(name)
    for name in ("m.dat", "z.dat", "a.dat"):
        (b / name).write_text(name)
    for path in list(a.rglob("*")) + list(b.rglob("*")):
        os.utime(path, (1700000000, 1700000000))

    assert cache._walk_fingerprint(a) == cache._walk_fingerprint(b)


def test_walk_fingerprint_uses_nested_relative_paths(tmp_path):
    """Entries are keyed by path *relative to the input*, with subdirectories
    spelled `SUB/name` -- so an absolute move of the MS doesn't change the
    fingerprint but a rename inside it does.
    """
    ms = _make_ms(tmp_path / "data.ms")
    names = {entry[0] for entry in cache._walk_fingerprint(ms)}

    assert "table.dat" in names
    assert "ANTENNA/table.f0" in names
    assert not any(name.startswith("/") for name in names)


def test_deep_change_inside_a_subtable_changes_the_fingerprint(tmp_path):
    """The case the whole directory branch exists for: an MS is rewritten
    several levels down, with the top-level directory untouched.
    """
    ms = _make_ms(tmp_path / "data.ms")
    before = cache._walk_fingerprint(ms)

    target = ms / "ANTENNA" / "table.f0"
    os.utime(target, (target.stat().st_atime + 5, target.stat().st_mtime + 5))

    assert cache._walk_fingerprint(ms) != before


def test_adding_deleting_or_renaming_a_file_changes_the_fingerprint(tmp_path):
    ms = _make_ms(tmp_path / "data.ms")
    before = cache._walk_fingerprint(ms)

    (ms / "FIELD" / "extra.f1").write_text("new")
    added = cache._walk_fingerprint(ms)
    assert added != before

    (ms / "FIELD" / "extra.f1").rename(ms / "FIELD" / "renamed.f1")
    assert cache._walk_fingerprint(ms) != added

    (ms / "FIELD" / "renamed.f1").unlink()
    assert cache._walk_fingerprint(ms) == before


def test_a_dangling_symlink_does_not_raise(tmp_path):
    """Regression: `os.walk` lists a broken symlink under `files`, and the
    old per-file `stat` then raised `FileNotFoundError` straight out of
    `compute_cache_key`, killing the run. A boundary input is data shinobi
    neither produced nor controls; a fingerprint is the wrong place to fail.
    """
    ms = _make_ms(tmp_path / "data.ms")
    (ms / "broken").symlink_to(ms / "does_not_exist")

    with pytest.raises(FileNotFoundError):  # what the old walk did
        _reference_walk(ms)

    names = {entry[0] for entry in cache._walk_fingerprint(ms)}
    assert "broken" not in names
    assert "table.dat" in names


def test_a_symlink_loop_does_not_raise(tmp_path):
    """Regression: a cycle raised `OSError: [Errno 40] ELOOP`."""
    ms = _make_ms(tmp_path / "data.ms")
    (ms / "loop_a").symlink_to(ms / "loop_b")
    (ms / "loop_b").symlink_to(ms / "loop_a")

    with pytest.raises(OSError, match="Too many levels of symbolic links"):
        _reference_walk(ms)

    names = {entry[0] for entry in cache._walk_fingerprint(ms)}
    assert "loop_a" not in names and "loop_b" not in names
    assert "table.dat" in names


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_an_unreadable_subdirectory_is_skipped_not_fatal(tmp_path):
    """Regression: a subdirectory with `r` but no `x` raised `PermissionError`
    (the name is listable, so `os.walk` descended, and the stat then failed).
    """
    ms = _make_ms(tmp_path / "data.ms")
    locked = ms / "FIELD"
    locked.chmod(0o444)
    try:
        with pytest.raises(PermissionError):  # what the old walk did
            _reference_walk(ms)

        names = {entry[0] for entry in cache._walk_fingerprint(ms)}
        assert "table.dat" in names
        assert not any(name.startswith("FIELD/") for name in names)
    finally:
        locked.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_an_unreadable_input_does_not_key_as_a_missing_one(tmp_path):
    """An unreadable path used to raise. It now returns a distinct marker
    rather than `None`: keying it as absent would let a step take a hit
    against a run where the file genuinely wasn't there, the moment
    permissions were restored.
    """
    ms = _make_ms(tmp_path / "data.ms")
    ms.chmod(0o000)
    try:
        unreadable = cache._walk_fingerprint(ms)
        assert unreadable is not None
        assert unreadable != cache._walk_fingerprint(tmp_path / "never_existed.ms")
    finally:
        ms.chmod(0o755)


def test_a_missing_path_still_keys_as_absent(tmp_path):
    assert cache._walk_fingerprint(tmp_path / "nope.ms") is None
    # a path whose *parent* is a regular file -- ENOTDIR, which `Path.exists()`
    # also treated as absent
    (tmp_path / "afile").write_text("x")
    assert cache._walk_fingerprint(tmp_path / "afile" / "under.ms") is None


def test_deleting_declared_output_forces_rerun(tmp_path):
    out_path = tmp_path / "out.dat"
    calls = {"n": 0}

    @shinobi.pystep()
    def write_step(ctx) -> InPlaceOutputs:
        calls["n"] += 1
        out_path.write_text("data")
        return InPlaceOutputs(vis=out_path)

    cache_dir = tmp_path / "cache"
    _dispatch(write_step.step, write_step.func, cache=True, cache_dir=str(cache_dir))
    assert calls["n"] == 1

    out_path.unlink()
    _dispatch(write_step.step, write_step.func, cache=True, cache_dir=str(cache_dir))
    assert calls["n"] == 2


# -- CacheManifest / compute_cache_key unit coverage --


def test_wrangled_non_path_output_is_restored_verbatim_on_a_hit(tmp_path):
    class WrangledOutputs(BaseModel):
        note: str = ""
        marker: Path | None = None

    class NoInputs(BaseModel):
        pass

    scope = Cab(name="w", command="w", inputs_model=NoInputs, outputs_model=WrangledOutputs)
    manifest = CacheManifest(tmp_path / "manifest.json")
    outputs = WrangledOutputs(note="hello from stdout wrangling", marker=None)

    manifest.record("w", "key1", StepResult(name="w", returncode=0, outputs=outputs, inputs=NoInputs()))
    hit = manifest.check("w", "key1", scope, {})
    assert hit is not None
    assert hit.outputs.note == "hello from stdout wrangling"


def test_manifest_reused_instance_shares_lock(tmp_path):
    m1 = get_cache_manifest(str(tmp_path))
    m2 = get_cache_manifest(str(tmp_path))
    assert m1 is m2


def test_concurrent_record_does_not_corrupt_manifest(tmp_path):
    import threading

    class NoInputs(BaseModel):
        pass

    class SimpleOutputs(BaseModel):
        value: int = 0

    scope = Cab(name="c", command="c", inputs_model=NoInputs, outputs_model=SimpleOutputs)
    manifest = CacheManifest(tmp_path / "manifest.json")

    def worker(i):
        manifest.record(
            f"step{i}",
            f"key{i}",
            StepResult(name=f"step{i}", returncode=0, outputs=SimpleOutputs(value=i), inputs=NoInputs()),
        )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i in range(20):
        hit = manifest.check(f"step{i}", f"key{i}", scope, {})
        assert hit is not None
        assert hit.outputs.value == i


def test_compute_cache_key_differs_for_different_params():
    key1 = compute_cache_key(Cab(name="c", command="c", inputs_model=Inputs, outputs_model=Outputs), None, {"x": 1})
    key2 = compute_cache_key(Cab(name="c", command="c", inputs_model=Inputs, outputs_model=Outputs), None, {"x": 2})
    assert key1 != key2


def test_venvless_cache_key_is_unchanged_by_the_venv_field():
    # Regression: a scope with no venv must key exactly as before the venv
    # field existed, so existing cache entries survive the upgrade. The venv
    # part is *conditionally appended* only when scope.venv is set, so a
    # venv-less scope's `parts` list is byte-identical to the pre-venv code.
    # This pins the resulting hash; if the append ever becomes unconditional,
    # this fails.
    key = compute_cache_key(Cab(name="c", command="c", inputs_model=Inputs, outputs_model=Outputs), None, {"x": 1})
    assert key == "f340d7de89951576429745e8ce95234d282934e42a577a54fce71084c9a87f08"


def test_venv_changes_cache_key():
    plain = compute_cache_key(Cab(name="c", command="c", inputs_model=Inputs, outputs_model=Outputs), None, {"x": 1})
    with_venv = compute_cache_key(Cab(name="c", command="c", venv="/opt/env", inputs_model=Inputs, outputs_model=Outputs), None, {"x": 1})
    other_venv = compute_cache_key(Cab(name="c", command="c", venv="/opt/other", inputs_model=Inputs, outputs_model=Outputs), None, {"x": 1})
    assert plain != with_venv
    assert with_venv != other_venv


# -- nested Recipe (the real-world shape: a Recipe-of-Recipes pipeline
# assembling several workers, each itself a Recipe of pysteps/cabs) --


def test_caching_through_a_nested_recipe_only_skips_unchanged_leaf_steps():
    from shinobi.steps import InputRef, OutputRef, Recipe

    calls = {"a": 0, "b": 0}

    @shinobi.pystep()
    def step_a(ctx, x: int = 1) -> CounterOutputs:
        calls["a"] += 1
        return CounterOutputs(count=calls["a"])

    @shinobi.pystep()
    def step_b(ctx, x: int = 1) -> CounterOutputs:
        calls["b"] += 1
        return CounterOutputs(count=calls["b"])

    class RecipeInputs(BaseModel):
        x: int = 1

    inner = Recipe(
        name="inner",
        inputs_model=RecipeInputs,
        outputs_model=CounterOutputs,
        steps=[
            step_a.model_copy(update={"wiring": {"x": InputRef(field="x")}}),
            step_b.model_copy(update={"wiring": {"x": InputRef(field="x")}}),
        ],
        output_wiring={"count": OutputRef(step=step_b.name, field="count")},
    )

    import tempfile

    with tempfile.TemporaryDirectory() as cache_dir:
        inner(x=1, cache=True, cache_dir=cache_dir)
        inner(x=1, cache=True, cache_dir=cache_dir)

    assert calls["a"] == 1
    assert calls["b"] == 1


# -- upstream provenance: an in-place mutator must notice that the step which
# *produced* its path re-ran, which mtime alone cannot tell it (see
# shinobi.cache's module docstring) --


class MsOut(BaseModel):
    ms: Path


class SpwInputs(BaseModel):
    spw: str = "*"


def _split_and_flag(tmp_path, calls):
    """A two-step in-place chain: `split` writes the MS from scratch, `flag`
    reads and rewrites that same MS. `flag`'s `ms` is on both its inputs and
    its outputs model, so the in-place exclusion drops it from the input hash
    entirely -- the exact shape the fix is about.
    """
    from shinobi.steps import InputRef, OutputRef, Recipe

    ms = tmp_path / "data.ms"

    @shinobi.pystep()
    def split(ctx, spw: str = "*") -> MsOut:
        calls["split"] += 1
        ms.write_text(f"visibilities for {spw}")
        return MsOut(ms=ms)

    @shinobi.pystep()
    def flag(ctx, ms: Path) -> MsOut:
        calls["flag"] += 1
        ms.write_text(ms.read_text() + " | flagged")
        return MsOut(ms=ms)

    return (
        Recipe(
            name="pipe",
            inputs_model=SpwInputs,
            outputs_model=MsOut,
            steps=[
                split.model_copy(update={"wiring": {"spw": InputRef(field="spw")}}),
                flag.model_copy(update={"wiring": {"ms": OutputRef(step="split", field="ms")}}),
            ],
            output_wiring={"ms": OutputRef(step="flag", field="ms")},
        ),
        ms,
    )


def test_unchanged_rerun_still_skips_the_whole_in_place_chain(tmp_path):
    """The property the in-place exclusion exists to protect, and which
    provenance must not break: re-running an untouched chain skips all of it,
    even though every step's own last run moved the shared MS's mtime.
    """
    calls = {"split": 0, "flag": 0}
    pipeline, _ms = _split_and_flag(tmp_path, calls)
    cache_dir = str(tmp_path / "cache")

    pipeline(spw="*", cache=True, cache_dir=cache_dir)
    assert calls == {"split": 1, "flag": 1}

    pipeline(spw="*", cache=True, cache_dir=cache_dir)
    assert calls == {"split": 1, "flag": 1}


def test_changing_an_upstream_param_reruns_the_downstream_in_place_step(tmp_path):
    """Changing `split`'s params rebuilds the MS, so `flag` must re-run --
    even though `flag`'s own params are byte-identical and its only path
    input is excluded from hashing. Without provenance `flag` cache-hits
    here, leaving the MS split with new parameters but never flagged.
    """
    calls = {"split": 0, "flag": 0}
    pipeline, ms = _split_and_flag(tmp_path, calls)
    cache_dir = str(tmp_path / "cache")

    pipeline(spw="*", cache=True, cache_dir=cache_dir)
    pipeline(spw="*:880~1658MHz", cache=True, cache_dir=cache_dir)

    assert calls == {"split": 2, "flag": 2}
    assert ms.read_text() == "visibilities for *:880~1658MHz | flagged"


def test_provenance_crosses_nested_recipe_boundaries(tmp_path):
    """The real pipeline shape: each worker is its own Recipe, so the
    producer and the consumer are in *different* recipes and the only link
    between them is the outer recipe's wiring. Provenance has to survive both
    boundary crossings -- out of the producing recipe via `output_wiring`, and
    into the consuming recipe via its sub-step's `InputRef`.
    """
    from shinobi.steps import InputRef, OutputRef, Recipe

    calls = {"split": 0, "flag": 0}
    ms = tmp_path / "data.ms"

    @shinobi.pystep()
    def split(ctx, spw: str = "*") -> MsOut:
        calls["split"] += 1
        ms.write_text(f"visibilities for {spw}")
        return MsOut(ms=ms)

    @shinobi.pystep()
    def flag(ctx, ms: Path) -> MsOut:
        calls["flag"] += 1
        ms.write_text(ms.read_text() + " | flagged")
        return MsOut(ms=ms)

    transform = Recipe(
        name="transform",
        inputs_model=SpwInputs,
        outputs_model=MsOut,
        steps=[split.model_copy(update={"wiring": {"spw": InputRef(field="spw")}})],
        output_wiring={"ms": OutputRef(step="split", field="ms")},
    )
    prep = Recipe(
        name="prep",
        inputs_model=MsOut,
        outputs_model=MsOut,
        steps=[flag.model_copy(update={"wiring": {"ms": InputRef(field="ms")}})],
        output_wiring={"ms": OutputRef(step="flag", field="ms")},
    )
    pipeline = (
        Recipe(name="pipe", inputs_model=SpwInputs, outputs_model=MsOut)
        .add_step("transform", transform, spw=InputRef(field="spw"))
        .add_step("prep", prep, ms=OutputRef(step="transform", field="ms"))
        .set_output("ms", OutputRef(step="prep", field="ms"))
    )

    cache_dir = str(tmp_path / "cache")
    pipeline(spw="*", cache=True, cache_dir=cache_dir)
    assert calls == {"split": 1, "flag": 1}

    pipeline(spw="*", cache=True, cache_dir=cache_dir)
    assert calls == {"split": 1, "flag": 1}

    pipeline(spw="*:880~1658MHz", cache=True, cache_dir=cache_dir)
    assert calls == {"split": 2, "flag": 2}


def test_rerunning_an_unconsumed_sibling_does_not_invalidate_the_consumer(tmp_path):
    """Provenance is per *output field*, not per recipe. A recipe's outputs
    each come from a different sub-step, so re-running one of them must not
    invalidate a consumer wired to a different one -- keying a whole recipe
    off "something in here changed" would throw away most of the cache on any
    edit.
    """
    from shinobi.steps import InputRef, OutputRef, Recipe

    calls = {"split": 0, "listobs": 0, "flag": 0}
    ms = tmp_path / "data.ms"

    class Products(BaseModel):
        ms: Path
        summary: Path

    class ListobsInputs(BaseModel):
        verbose: bool = False

    class ListobsOut(BaseModel):
        summary: Path

    @shinobi.pystep()
    def split(ctx, spw: str = "*") -> MsOut:
        calls["split"] += 1
        ms.write_text(f"visibilities for {spw}")
        return MsOut(ms=ms)

    @shinobi.pystep()
    def listobs(ctx, verbose: bool = False) -> ListobsOut:
        calls["listobs"] += 1
        summary = tmp_path / "summary.txt"
        summary.write_text(f"verbose={verbose}")
        return ListobsOut(summary=summary)

    @shinobi.pystep()
    def flag(ctx, ms: Path) -> MsOut:
        calls["flag"] += 1
        ms.write_text(ms.read_text() + " | flagged")
        return MsOut(ms=ms)

    class InnerInputs(BaseModel):
        spw: str = "*"
        verbose: bool = False

    inner = Recipe(
        name="transform",
        inputs_model=InnerInputs,
        outputs_model=Products,
        steps=[
            split.model_copy(update={"wiring": {"spw": InputRef(field="spw")}}),
            listobs.model_copy(update={"wiring": {"verbose": InputRef(field="verbose")}}),
        ],
        output_wiring={
            "ms": OutputRef(step="split", field="ms"),
            "summary": OutputRef(step="listobs", field="summary"),
        },
    )
    pipeline = (
        Recipe(name="pipe", inputs_model=InnerInputs, outputs_model=MsOut)
        .add_step("transform", inner, spw=InputRef(field="spw"), verbose=InputRef(field="verbose"))
        .add_step("flag", flag, ms=OutputRef(step="transform", field="ms"))
        .set_output("ms", OutputRef(step="flag", field="ms"))
    )

    cache_dir = str(tmp_path / "cache")
    pipeline(spw="*", verbose=False, cache=True, cache_dir=cache_dir)
    assert calls == {"split": 1, "listobs": 1, "flag": 1}

    # `listobs` re-runs; `flag` consumes only `transform.ms`, so it must not.
    pipeline(spw="*", verbose=True, cache=True, cache_dir=cache_dir)
    assert calls == {"split": 1, "listobs": 2, "flag": 1}


def test_a_later_in_place_step_does_not_rerun_a_pure_input_consumer(tmp_path):
    """`listobs` reads the MS; `flag`, declared after it, rewrites that same
    MS. Identifying a wired path by content makes `listobs` look changed on
    every subsequent run -- and each run moves the mtime again for the next
    one, so it re-runs forever. Its input is really "the MS as `split` left
    it", which is what `split`'s cache key names and what an mtime cannot.
    """
    from shinobi.steps import InputRef, OutputRef, Recipe

    calls = {"split": 0, "listobs": 0, "flag": 0}
    ms = tmp_path / "data.ms"

    class SummaryOut(BaseModel):
        summary: Path

    @shinobi.pystep()
    def split(ctx, spw: str = "*") -> MsOut:
        calls["split"] += 1
        ms.write_text(f"visibilities for {spw}")
        return MsOut(ms=ms)

    @shinobi.pystep()
    def listobs(ctx, ms: Path) -> SummaryOut:
        calls["listobs"] += 1
        summary = tmp_path / "summary.txt"
        summary.write_text(ms.read_text())
        return SummaryOut(summary=summary)

    @shinobi.pystep()
    def flag(ctx, ms: Path) -> MsOut:
        calls["flag"] += 1
        ms.write_text(ms.read_text() + " | flagged")
        return MsOut(ms=ms)

    pipeline = Recipe(
        name="pipe",
        inputs_model=SpwInputs,
        outputs_model=MsOut,
        steps=[
            split.model_copy(update={"wiring": {"spw": InputRef(field="spw")}}),
            listobs.model_copy(update={"wiring": {"ms": OutputRef(step="split", field="ms")}}),
            flag.model_copy(update={"wiring": {"ms": OutputRef(step="split", field="ms")}}),
        ],
        output_wiring={"ms": OutputRef(step="flag", field="ms")},
    )

    cache_dir = str(tmp_path / "cache")
    for _ in range(3):
        pipeline(spw="*", cache=True, cache_dir=cache_dir)

    assert calls == {"split": 1, "listobs": 1, "flag": 1}


def test_unwired_boundary_path_is_still_content_hashed(tmp_path):
    """The other half of the same rule: a path the DAG did *not* produce is
    the boundary, and there is no provenance to identify it by -- so its
    content still decides, exactly as before.
    """
    from shinobi.steps import InputRef, Recipe

    calls = {"n": 0}
    external = tmp_path / "external.cfg"
    external.write_text("v1")

    class CfgInputs(BaseModel):
        cfg: Path

    @shinobi.pystep()
    def read_cfg(ctx, cfg: Path) -> CounterOutputs:
        calls["n"] += 1
        return CounterOutputs(count=calls["n"])

    pipeline = Recipe(
        name="pipe",
        inputs_model=CfgInputs,
        outputs_model=CounterOutputs,
        steps=[read_cfg.model_copy(update={"wiring": {"cfg": InputRef(field="cfg")}})],
    )

    cache_dir = str(tmp_path / "cache")
    pipeline(cfg=external, cache=True, cache_dir=cache_dir)
    pipeline(cfg=external, cache=True, cache_dir=cache_dir)
    assert calls["n"] == 1

    external.write_text("v2 -- edited out of band")
    pipeline(cfg=external, cache=True, cache_dir=cache_dir)
    assert calls["n"] == 2


def test_provenance_is_absent_when_a_step_has_no_wired_inputs(tmp_path):
    """A step with nothing wired in has no provenance to contribute, so its
    key must be exactly what it was before provenance existed -- otherwise
    every such cache entry would be invalidated by the upgrade alone.
    """
    cab = Cab(name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs)
    assert compute_cache_key(cab, None, {"x": 1}, {}) == compute_cache_key(cab, None, {"x": 1})
    assert compute_cache_key(cab, None, {"x": 1}, None) == compute_cache_key(cab, None, {"x": 1})
    assert compute_cache_key(cab, None, {"x": 1}, {"x": "upstreamkey"}) != compute_cache_key(cab, None, {"x": 1})


# -- sandbox path normalization (issue #28: sandbox state must not affect
# cache entry portability -- outputs are normalized to workspace-relative
# paths regardless of whether the step ran sandboxed) --


def test_sandboxed_field_is_recorded_and_restored(tmp_path):
    """The `sandboxed` field travels through the cache round-trip, so a
    later hit carries the same provenance as the original run."""
    from shinobi.cache import CacheManifest
    from shinobi.results import StepResult

    class NoInputs(BaseModel):
        pass

    scope = Cab(name="s", command="s", inputs_model=NoInputs, outputs_model=CounterOutputs)
    manifest = CacheManifest(tmp_path / "manifest.json")
    outputs = CounterOutputs(count=1)

    manifest.record(
        "s",
        "key1",
        StepResult(name="s", returncode=0, outputs=outputs, inputs=NoInputs(), sandboxed=True),
    )
    hit = manifest.check("s", "key1", scope, {})
    assert hit is not None
    assert hit.sandboxed is True

    manifest.record(
        "s",
        "key2",
        StepResult(name="s", returncode=0, outputs=outputs, inputs=NoInputs(), sandboxed=False),
    )
    hit2 = manifest.check("s", "key2", scope, {})
    assert hit2 is not None
    assert hit2.sandboxed is False


def test_sandboxed_cab_result_is_marked_sandboxed(tmp_path, monkeypatch):
    """A cab that ran with sandbox=True reports sandboxed=True."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / ".shinobi/work"))

    recorder = RecordingBackend()
    register_step_backend("sandbox-rec", recorder)

    class FileOut(BaseModel):
        result: Path | None = None

    cab = Cab(
        name="tool",
        command="/bin/true",
        inputs_model=Inputs,
        outputs_model=FileOut,
        backend="sandbox-rec",
        sandbox=True,
    )
    result = _dispatch(cab, None, x=1)
    assert result.sandboxed is True


def test_unsandboxed_cab_result_is_not_marked_sandboxed(tmp_path):
    """A cab that ran without sandboxing reports sandboxed=False."""
    recorder = RecordingBackend()
    register_step_backend("no-sandbox-rec", recorder)

    cab = Cab(
        name="tool",
        command="tool",
        inputs_model=Inputs,
        outputs_model=Outputs,
        backend="no-sandbox-rec",
    )
    result = _dispatch(cab, None, x=1)
    assert result.sandboxed is False
