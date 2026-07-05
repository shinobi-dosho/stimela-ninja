from shinobi.loaders import build_model
from shinobi.policies import build_argv
from shinobi.steps.schema import Cab


def make_cab(**kwargs) -> Cab:
    kwargs.setdefault("command", "/bin/echo")
    kwargs.setdefault("inputs_model", build_model("In", {}))
    kwargs.setdefault("outputs_model", build_model("Out", {}))
    return Cab(name="tool", **kwargs)


def test_native_backend_runs_and_captures_stdout(native):
    cab = make_cab(inputs_model=build_model("In", {"text": ("str", False, "hello there")}))
    run = native.run(cab, build_argv(cab, {"text": "hello there"}), {"text": "hello there"})
    assert run.success
    assert "hello there" in run.stdout


def test_native_backend_returns_raw_backendrun_without_wrangling(native):
    # wrangling is the dispatch layer's job now; the backend just runs.
    cab = make_cab(inputs_model=build_model("In", {"text": ("str", False, "x")}))
    run = native.run(cab, build_argv(cab, {"text": "x"}), {"text": "x"})
    assert hasattr(run, "returncode")
    assert not hasattr(run, "outputs")


def test_failing_command_reports_nonzero(native):
    cab = make_cab(command="/bin/false")
    run = native.run(cab, build_argv(cab, {}), {})
    assert not run.success
    assert run.returncode != 0
