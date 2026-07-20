from pathlib import Path

import shinobi
from pydantic import BaseModel

from shinobi.backends.recording import RecordingBackend
from shinobi.cache import CacheManifest, compute_cache_key, get_cache_manifest
from shinobi.results import StepResult
from shinobi.steps import Cab, register_step_backend
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
