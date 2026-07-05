import threading
import time

import pytest
from pydantic import BaseModel, ValidationError

from shinobi.backends.recording import RecordingBackend
from shinobi.config import AppConfig, ExecutionConfig
from shinobi.results import BackendRun
from shinobi.steps import Cab, InputRef, OutputRef, Recipe, StepRef, register_step_backend
from shinobi.steps.dispatch import _dispatch
from shinobi.steps.schema import Mutability
from .fixtures.sample_steps import (
    chained_recipe,
    immutable_list_cab,
    make_value_cab,
    mutable_list_cab,
    use_value_cab,
)


class Inputs(BaseModel):
    x: int


class Outputs(BaseModel):
    y: str | None = None


def make_recording_cab(**kwargs) -> tuple[Cab, RecordingBackend]:
    recorder = RecordingBackend()
    register_step_backend("record", recorder)
    cab = Cab(
        name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs, backend="record", **kwargs
    )
    return cab, recorder


class MutatingBackend:
    """Appends a marker to every list-valued input it receives, so tests can
    observe whether that input was the caller's own object (MUTABLE) or a
    deep copy (IMMUTABLE)."""

    def run(self, cab, argv, inputs):
        for value in inputs.values():
            if isinstance(value, list):
                value.append(99)
        return BackendRun(0, "", "")


# -- validation --


def test_missing_required_field_raises_before_backend_runs():
    cab, recorder = make_recording_cab()
    with pytest.raises(ValidationError):
        _dispatch(cab, None)
    assert recorder.calls == []


def test_bad_override_raises_at_the_step_boundary():
    cab, recorder = make_recording_cab()

    def orchestrate(ctx):
        return ctx.run(x="not an int")

    with pytest.raises(ValidationError):
        _dispatch(cab, orchestrate, x=1)
    assert recorder.calls == []


def test_dynamic_pattern_param_warns_it_is_unvalidated():
    from shinobi.loaders._modelgen import build_model
    from shinobi.steps.schema import ParamMeta, ParamPattern, ParamSegment

    recorder = RecordingBackend()
    register_step_backend("record", recorder)
    cab = Cab(
        name="tool",
        command="tool",
        inputs_model=build_model("In", {"x": ("int", True, None)}, allow_extra=True),
        outputs_model=Outputs,
        backend="record",
        input_patterns=[
            ParamPattern(segments=[ParamSegment(regex=r".+?"), ParamSegment(attrs={"solvable": ParamMeta()})])
        ],
    )

    with pytest.warns(UserWarning, match=r"g1\.solvable"):
        _dispatch(cab, None, x=1, **{"g1.solvable": True})

    _, _, inputs = recorder.calls[0]
    assert inputs["g1.solvable"] is True


# -- auto-run + snapshot --


def test_none_return_auto_runs_and_snapshot_is_untouched():
    cab, recorder = make_recording_cab()

    captured = {}

    def orchestrate(ctx):
        captured["snapshot"] = ctx.inputs.x
        return None

    _dispatch(cab, orchestrate, x=4)
    assert captured["snapshot"] == 4
    _, _, inputs = recorder.calls[0]
    assert inputs["x"] == 4


def test_ctx_inputs_snapshot_survives_overrides():
    cab, recorder = make_recording_cab()

    seen = {}

    def orchestrate(ctx):
        result = ctx.run(x=100)
        seen["snapshot"] = ctx.inputs.x
        seen["effective"] = result.inputs.x
        return result

    _dispatch(cab, orchestrate, x=1)
    assert seen["snapshot"] == 1  # original call, never mutated
    assert seen["effective"] == 100  # override applied at run()


# -- mutability preservation through run(**overrides) --


def test_immutable_input_is_deepcopied_backend_cannot_mutate_caller():
    register_step_backend("mutating", MutatingBackend())
    cab = immutable_list_cab.model_copy(update={"backend": "mutating"})
    original = [1, 2, 3]
    _dispatch(cab, None, items=original)
    assert original == [1, 2, 3]


def test_mutable_input_reaches_backend_by_reference():
    register_step_backend("mutating", MutatingBackend())
    cab = mutable_list_cab.model_copy(update={"backend": "mutating"})
    original = [1, 2, 3]
    _dispatch(cab, None, items=original)
    assert original == [1, 2, 3, 99]


def test_mutable_preservation_through_run_override():
    register_step_backend("mutating", MutatingBackend())
    cab = mutable_list_cab.model_copy(update={"backend": "mutating"})
    override_list = [7, 8]

    def orchestrate(ctx):
        return ctx.run(items=override_list)

    _dispatch(cab, orchestrate, items=[1])
    assert override_list == [7, 8, 99]


# -- cab dispatch + output fill --


def test_cab_dispatch_calls_resolved_backend_with_final_inputs():
    cab, recorder = make_recording_cab()
    _dispatch(cab, None, x=5)
    defn, _, inputs = recorder.calls[0]
    assert defn is cab
    assert inputs["x"] == 5


def test_wrangler_output_populates_outputs_model():
    class Out(BaseModel):
        n: int | None = None

    class NoIn(BaseModel):
        pass

    class FixedBackend:
        def run(self, cab, argv, inputs):
            return BackendRun(0, "answer=42\n", "")

    register_step_backend("fixed", FixedBackend())
    cab = Cab(
        name="tool",
        command="tool",
        inputs_model=NoIn,
        outputs_model=Out,
        backend="fixed",
        wranglers={r"answer=(?P<n>\d+)": ["PARSE_OUTPUT:n:int"]},
    )
    result = _dispatch(cab, None)
    assert result.outputs.n == 42
    assert result.success


# -- standalone StepRef call --


def test_standalone_stepref_merges_params_under_kwargs():
    cab, recorder = make_recording_cab()
    ref = StepRef(name="a", step=cab, params={"x": 3})

    ref()
    _, _, inputs = recorder.calls[0]
    assert inputs["x"] == 3

    ref(x=8)
    _, _, inputs = recorder.calls[1]
    assert inputs["x"] == 8


def test_standalone_stepref_runs_its_func():
    cab, recorder = make_recording_cab()

    def orchestrate(ctx):
        return ctx.run(x=ctx.inputs.x + 1)

    ref = StepRef(name="a", step=cab, func=orchestrate, params={"x": 10})
    ref()
    _, _, inputs = recorder.calls[0]
    assert inputs["x"] == 11


# -- recipe --


def test_recipe_runs_substeps_end_to_end():
    result = _dispatch(chained_recipe, None, name="whatever.txt")
    assert result.outputs.ok is True


def test_recipe_wires_a_real_output_value_into_next_steps_input():
    class MakeInputs(BaseModel):
        name: str = "x"

    class PathOut(BaseModel):
        path: str | None = None

    class UseInputs(BaseModel):
        path: str | None = None

    class OkOut(BaseModel):
        ok: bool = True

    class FixedOutputBackend:
        def run(self, cab, argv, inputs):
            return BackendRun(0, "result=/tmp/real-value.txt", "")

    use_recorder = RecordingBackend()
    register_step_backend("fixed-output", FixedOutputBackend())
    register_step_backend("use-recorder", use_recorder)

    make_cab = Cab(
        name="make",
        command="x",
        inputs_model=MakeInputs,
        outputs_model=PathOut,
        backend="fixed-output",
        wranglers={r"result=(?P<path>\S+)": ["PARSE_OUTPUT:path:str"]},
    )
    use_cab = Cab(
        name="use", command="x", inputs_model=UseInputs, outputs_model=OkOut, backend="use-recorder"
    )
    recipe = Recipe(
        name="r",
        inputs_model=MakeInputs,
        outputs_model=OkOut,
        steps=[
            StepRef(name="make", step=make_cab, wiring={"name": InputRef(field="name")}),
            StepRef(name="use", step=use_cab, wiring={"path": OutputRef(step="make", field="path")}),
        ],
        output_wiring={"ok": OutputRef(step="use", field="ok")},
    )

    _dispatch(recipe, None, name="whatever")
    _, _, use_inputs = use_recorder.calls[0]
    assert use_inputs["path"] == "/tmp/real-value.txt"


# -- backend resolution priority --


def test_cab_backend_wins_over_recipe_backend():
    cab_recorder = RecordingBackend()
    recipe_recorder = RecordingBackend()
    register_step_backend("cab-choice", cab_recorder)
    register_step_backend("recipe-choice", recipe_recorder)

    cab = Cab(name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs, backend="cab-choice")
    recipe = Recipe(
        name="r",
        inputs_model=Inputs,
        outputs_model=Outputs,
        backend="recipe-choice",
        steps=[StepRef(name="a", step=cab, wiring={"x": InputRef(field="x")})],
        output_wiring={"y": OutputRef(step="a", field="y")},
    )
    _dispatch(recipe, None, x=1)
    assert len(cab_recorder.calls) == 1
    assert recipe_recorder.calls == []


def test_recipe_backend_wins_over_appconfig_default_when_cab_has_none():
    recipe_recorder = RecordingBackend()
    register_step_backend("recipe-only-choice", recipe_recorder)

    cab = Cab(name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs)  # no backend
    recipe = Recipe(
        name="r",
        inputs_model=Inputs,
        outputs_model=Outputs,
        backend="recipe-only-choice",
        steps=[StepRef(name="a", step=cab, wiring={"x": InputRef(field="x")})],
        output_wiring={"y": OutputRef(step="a", field="y")},
    )
    _dispatch(recipe, None, x=1)
    assert len(recipe_recorder.calls) == 1


def test_explicit_backend_arg_wins_over_everything():
    explicit = RecordingBackend()
    register_step_backend("explicit", explicit)
    cab = Cab(name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs, backend="record")
    _dispatch(cab, None, backend="explicit", x=1)
    assert len(explicit.calls) == 1


def test_falls_through_to_appconfig_default(tmp_path, monkeypatch):
    monkeypatch.delenv("SHINOBI_BACKEND__DEFAULT", raising=False)
    fallback = RecordingBackend()
    register_step_backend("native", fallback)  # temporarily shadow "native"
    try:
        cab = Cab(name="tool", command="tool", inputs_model=Inputs, outputs_model=Outputs)
        assert AppConfig.load(config_file=tmp_path / "missing.yml").backend.default == "native"
        _dispatch(cab, None, x=1)
        assert len(fallback.calls) == 1
    finally:
        from shinobi.steps.dispatch import _STEP_BACKENDS

        _STEP_BACKENDS.pop("native", None)


def test_two_decorated_functions_share_one_scope_run_independently():
    cab, recorder = make_recording_cab()

    def a(ctx):
        return ctx.run(x=1)

    def b(ctx):
        return ctx.run(x=2)

    _dispatch(cab, a, x=0)
    _dispatch(cab, b, x=0)
    assert [inp["x"] for _, _, inp in recorder.calls] == [1, 2]


def test_fixture_cabs_are_reused_by_chained_recipe():
    assert chained_recipe.steps[0].step is make_value_cab
    assert chained_recipe.steps[1].step is use_value_cab
    assert mutable_list_cab.mutability_of("items") is Mutability.MUTABLE


# -- parallel DAG execution --


def _two_independent_recipe(cab_a, cab_b, *, max_workers=None):
    """A recipe of two steps that each wire only from the recipe's own input
    -- no OutputRef between them, so they're independent and eligible to run
    concurrently.
    """
    return Recipe(
        name="r",
        inputs_model=Inputs,
        outputs_model=Outputs,
        max_workers=max_workers,
        steps=[
            StepRef(name="a", step=cab_a, wiring={"x": InputRef(field="x")}),
            StepRef(name="b", step=cab_b, wiring={"x": InputRef(field="x")}),
        ],
    )


def _cab(name, backend):
    return Cab(name=name, command="t", inputs_model=Inputs, outputs_model=Outputs, backend=backend)


def test_independent_steps_run_concurrently_when_max_workers_allows():
    barrier = threading.Barrier(2, timeout=5)

    class BarrierBackend:
        def run(self, cab, argv, inputs):
            barrier.wait()  # only both return if two threads arrive together
            return BackendRun(0, "", "")

    register_step_backend("barrier", BarrierBackend())
    cab = _cab("t", "barrier")
    recipe = _two_independent_recipe(cab, cab, max_workers=2)
    # completes only if both steps were in flight at once; otherwise the
    # barrier times out and raises BrokenBarrierError out of the recipe.
    _dispatch(recipe, None, x=1)


def test_default_max_workers_is_sequential():
    barrier = threading.Barrier(2, timeout=1)

    class BarrierBackend:
        def run(self, cab, argv, inputs):
            barrier.wait()
            return BackendRun(0, "", "")

    register_step_backend("barrier", BarrierBackend())
    cab = _cab("t", "barrier")
    recipe = _two_independent_recipe(cab, cab)  # max_workers None -> config default 1
    with pytest.raises(threading.BrokenBarrierError):
        _dispatch(recipe, None, x=1)


def test_aggregation_is_declaration_order_not_completion_order():
    class SlowBackend:
        def run(self, cab, argv, inputs):
            time.sleep(0.15)
            return BackendRun(0, "AAA", "errA")

    class FastBackend:
        def run(self, cab, argv, inputs):
            return BackendRun(0, "BBB", "errB")

    register_step_backend("slow", SlowBackend())
    register_step_backend("fast", FastBackend())
    # step 'a' (declared first) finishes LAST; 'b' finishes first.
    recipe = _two_independent_recipe(_cab("a", "slow"), _cab("b", "fast"), max_workers=2)
    result = _dispatch(recipe, None, x=1)
    assert result.stdout == "AAA\nBBB"  # declaration order, not completion order
    assert result.stderr == "errA\nerrB"


def test_failure_stops_dependents_but_drains_independent_inflight():
    class FailBackend:
        def run(self, cab, argv, inputs):
            return BackendRun(2, "", "boom")

    ok_recorder = RecordingBackend()
    down_recorder = RecordingBackend()
    register_step_backend("fail", FailBackend())
    register_step_backend("ok", ok_recorder)
    register_step_backend("down", down_recorder)

    fail_cab = _cab("fail", "fail")
    ok_cab = _cab("ok", "ok")
    down_cab = Cab(
        name="down", command="t", inputs_model=Outputs, outputs_model=Outputs, backend="down"
    )
    recipe = Recipe(
        name="r",
        inputs_model=Inputs,
        outputs_model=Outputs,
        max_workers=2,
        steps=[
            StepRef(name="fail", step=fail_cab, wiring={"x": InputRef(field="x")}),
            StepRef(name="ok", step=ok_cab, wiring={"x": InputRef(field="x")}),
            # depends on 'fail' -> must never run once 'fail' fails
            StepRef(name="down", step=down_cab, wiring={"y": OutputRef(step="fail", field="y")}),
        ],
    )
    result = _dispatch(recipe, None, x=1)
    assert result.returncode == 2
    assert down_recorder.calls == []  # dependent of the failed step never ran


def test_first_failure_by_declaration_order_wins():
    class RC2Backend:
        def run(self, cab, argv, inputs):
            return BackendRun(2, "", "")

    class RC3Backend:
        def run(self, cab, argv, inputs):
            return BackendRun(3, "", "")

    register_step_backend("rc2", RC2Backend())
    register_step_backend("rc3", RC3Backend())
    recipe = _two_independent_recipe(_cab("a", "rc2"), _cab("b", "rc3"), max_workers=2)
    result = _dispatch(recipe, None, x=1)
    assert result.returncode == 2  # 'a' is declared first


def test_worker_exception_propagates_out_of_recipe():
    cab, _ = make_recording_cab()

    def bad_override(ctx):
        return ctx.run(x="not an int")

    recipe = Recipe(
        name="r",
        inputs_model=Inputs,
        outputs_model=Outputs,
        max_workers=2,
        steps=[StepRef(name="a", step=cab, func=bad_override, wiring={"x": InputRef(field="x")})],
    )
    with pytest.raises(ValidationError):
        _dispatch(recipe, None, x=1)


def test_recipe_max_workers_overrides_config_default():
    barrier = threading.Barrier(2, timeout=5)

    class BarrierBackend:
        def run(self, cab, argv, inputs):
            barrier.wait()
            return BackendRun(0, "", "")

    register_step_backend("barrier", BarrierBackend())
    cab = _cab("t", "barrier")
    recipe = _two_independent_recipe(cab, cab, max_workers=2)
    # config says 1, but the recipe's own max_workers=2 must win
    config = AppConfig(execution=ExecutionConfig(max_workers=1))
    _dispatch(recipe, None, _config=config, x=1)


def test_config_execution_max_workers_is_honoured(monkeypatch):
    monkeypatch.delenv("SHINOBI_EXECUTION__MAX_WORKERS", raising=False)
    barrier = threading.Barrier(2, timeout=5)

    class BarrierBackend:
        def run(self, cab, argv, inputs):
            barrier.wait()
            return BackendRun(0, "", "")

    register_step_backend("barrier", BarrierBackend())
    cab = _cab("t", "barrier")
    recipe = _two_independent_recipe(cab, cab)  # no per-recipe override
    config = AppConfig(execution=ExecutionConfig(max_workers=2))
    _dispatch(recipe, None, _config=config, x=1)
