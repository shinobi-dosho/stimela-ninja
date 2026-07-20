"""Tests for container-executable pysteps (src/shinobi/steps/pyfunc.py).

Verifies that `@shinobi.pystep(image=...)` correctly dispatches to a
container backend when one is resolved, and falls back to in-process
execution when the backend is `native`.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from shinobi import pystep
from shinobi.backends.recording import RecordingBackend
from shinobi.exceptions import CabRunError
from shinobi.steps import InputRef, Recipe, register_step_backend
from shinobi.steps.dispatch import _dispatch
from shinobi.steps.schema import Cab, Scope

# Captured before any test patches `shinobi.steps.pyfunc.run_streaming` (the
# module-level name pyfunc.py actually calls) -- the end-to-end helper needs
# the genuine subprocess.run to avoid recursing into that patch.
_REAL_SUBPROCESS_RUN = subprocess.run


class ContainerOutputs(BaseModel):
    result: str


def container_func(ms: str, niter: int = 100) -> ContainerOutputs:
    return ContainerOutputs(result=f"{ms}:{niter}")


def no_output_func(ms: str) -> None:
    pass


class CtxOutputs(BaseModel):
    joined: str


def ctx_func(ctx, a: str, b: str) -> CtxOutputs:
    """A pystep whose body uses the injected ctx (the documented pattern)."""
    join = ctx.import_func("join", "os.path")
    return CtxOutputs(joined=join(a, b))


class PathOutputs(BaseModel):
    basename: str


def path_func(vis: Path) -> PathOutputs:
    """A pystep that relies on its input actually being a Path object."""
    return PathOutputs(basename=vis.name)


def _run_runner_on_host(argv, *args, **kwargs):
    """subprocess.run stand-in that simulates the container by executing the
    generated runner with the host interpreter. Because the runner and its
    inputs use identity-bind-mounted (host) paths, running it directly on the
    host exercises the real generated script end-to-end -- catching any
    path/import/ctx-shim breakage without needing a container runtime.
    """
    runner = next(a for a in argv if a.endswith("runner.py"))
    return _REAL_SUBPROCESS_RUN([sys.executable, runner], capture_output=True, text=True)


def _fake_container_run(outputs_data, capture_inputs=None, stdout=""):
    """Build a subprocess.run stand-in for mocked-argv tests: it mimics a
    successful container run by writing `outputs_data` to the runner's
    outputs.json (as the real runner would) before returning a zero-exit
    stub. If `capture_inputs` is a dict, the pickled inputs are loaded into
    it.
    """

    def fake_run(argv, *args, **kwargs):
        runner = next(a for a in argv if a.endswith("runner.py"))
        io_dir = Path(runner).parent
        if capture_inputs is not None:
            capture_inputs.update(pickle.loads((io_dir / "inputs.pkl").read_bytes()))
        (io_dir / "outputs.json").write_text(json.dumps(outputs_data))
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = stdout
        proc.stderr = ""
        return proc

    return fake_run


def test_scope_has_image_field():
    scope = Scope(
        name="test",
        inputs_model=ContainerOutputs,
        outputs_model=ContainerOutputs,
        image="myimage:latest",
    )
    assert scope.image == "myimage:latest"


def test_scope_image_defaults_to_none():
    scope = Scope(
        name="test",
        inputs_model=ContainerOutputs,
        outputs_model=ContainerOutputs,
    )
    assert scope.image is None


def test_cab_inherits_image_from_scope():
    cab = Cab(
        name="test",
        command="test",
        image="cab-image:latest",
        inputs_model=ContainerOutputs,
        outputs_model=ContainerOutputs,
    )
    assert cab.image == "cab-image:latest"


def test_pystep_accepts_image_kwarg():
    ref = pystep(image="myimage:latest")(container_func)
    assert ref.step.image == "myimage:latest"


def test_pystep_accepts_backend_kwarg():
    ref = pystep(backend="docker")(container_func)
    assert ref.step.backend == "docker"


def test_pystep_image_defaults_to_none():
    ref = pystep()(container_func)
    assert ref.step.image is None


def test_pystep_with_image_runs_in_process_when_backend_is_native():
    ref = pystep(image="myimage:latest")(container_func)
    result = ref(ms="test.ms", niter=500, backend="native")
    assert result.success
    assert result.outputs.result == "test.ms:500"


def test_pystep_without_image_always_runs_in_process():
    ref = pystep()(container_func)
    result = ref(ms="test.ms")
    assert result.success
    assert result.outputs.result == "test.ms:100"


def test_pystep_container_generates_correct_argv():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    fake = _fake_container_run({"result": "test.ms:100"})
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=fake) as mock_run:
        ref(ms="test.ms")

    mock_run.assert_called_once()
    argv = mock_run.call_args[0][0]

    assert argv[0] == "docker"
    assert argv[1] == "run"
    assert "--rm" in argv
    assert "casa:latest" in argv
    assert "python3" in argv

    image_idx = argv.index("casa:latest")
    inner_argv = argv[image_idx + 1 :]
    assert inner_argv[0] == "python3"

    # The runner path must be a real path that is identity-bind-mounted into
    # the container (regression guard: a hardcoded /shinobi_io that is never
    # mounted would not be reachable inside the container).
    runner_path = inner_argv[1]
    assert runner_path.endswith("runner.py")
    assert "shinobi_pystep_" in runner_path
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert any(runner_path.startswith(m.split(":")[0]) for m in mounts)


def test_pystep_container_mounts_source_and_io_dirs():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    fake = _fake_container_run({"result": "test.ms:100"})
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=fake) as mock_run:
        ref(ms="test.ms")

    argv = mock_run.call_args[0][0]
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]

    io_mounts = [m for m in mounts if "shinobi_pystep_" in m]
    assert len(io_mounts) >= 1

    # The function's source file must live under some mounted directory so
    # the runner can import it (the mounted dir is the package root put on
    # sys.path, not necessarily the file's immediate parent).
    source_file = str(Path(__file__).resolve())
    assert any(source_file.startswith(m.split(":")[0]) for m in mounts)


def test_pystep_container_apptainer_argv():
    ref = pystep(image="casa:latest", backend="apptainer")(container_func)

    fake = _fake_container_run({"result": "test.ms:100"})
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=fake) as mock_run:
        ref(ms="test.ms")

    argv = mock_run.call_args[0][0]
    assert argv[0] == "apptainer"
    assert argv[1] == "exec"
    assert "docker://casa:latest" in argv


def test_pystep_container_parses_outputs_file():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    fake = _fake_container_run({"result": "parsed.ms:999"})
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=fake):
        result = ref(ms="parsed.ms", niter=999)

    assert result.success
    assert result.outputs.result == "parsed.ms:999"


def test_pystep_container_handles_empty_outputs():
    ref = pystep(image="casa:latest", backend="docker")(no_output_func)

    # The runner writes `null` for a -> None function that returned None.
    fake = _fake_container_run(None)
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=fake):
        result = ref(ms="test.ms")

    assert result.success
    assert type(result.outputs).model_fields == {}


def test_pystep_container_nonzero_exit_raises():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stdout = ""
    mock_proc.stderr = "Traceback: something went wrong"

    with patch("shinobi.steps.pyfunc.run_streaming", return_value=mock_proc):
        with pytest.raises(CabRunError, match="returncode 1"):
            ref(ms="test.ms")


def test_failed_pystep_in_recipe_raises_not_attribute_error():
    """Regression for issue #27: a failed container pystep must raise
    CabRunError, not return a hollow outputs model that masks the real failure
    behind an AttributeError when the recipe collects downstream wiring.
    """

    class RecipeInputs(BaseModel):
        ms: str = "in.ms"

    class RecipeOutputs(BaseModel):
        result: str | None = None

    class DownInputs(BaseModel):
        path: str | None = None

    class DownOutputs(BaseModel):
        result: str | None = None

    down_cab = Cab(
        name="down",
        command="down",
        inputs_model=DownInputs,
        outputs_model=DownOutputs,
        backend="recording",
    )

    ref = pystep(image="casa:latest", backend="docker")(container_func)
    recorder = RecordingBackend()
    register_step_backend("recording", recorder)

    recipe = Recipe(name="r", inputs_model=RecipeInputs, outputs_model=RecipeOutputs)
    recipe.add_step("fail", ref, ms=InputRef(field="ms"))
    recipe.add_step("down", down_cab, path=recipe.outputs("fail", "result"))
    recipe.set_output("result", recipe.outputs("down", "result"))

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stdout = ""
    mock_proc.stderr = "boom"

    with patch("shinobi.steps.pyfunc.run_streaming", return_value=mock_proc):
        with pytest.raises(CabRunError, match="returncode 1"):
            _dispatch(recipe, None)

    assert recorder.calls == []  # downstream step never ran


def test_pystep_container_pickles_inputs():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    captured_inputs = {}
    fake = _fake_container_run({"result": "ok"}, capture_inputs=captured_inputs)
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=fake):
        ref(ms="serialised.ms", niter=42)

    assert captured_inputs["ms"] == "serialised.ms"
    assert captured_inputs["niter"] == 42


def test_pystep_container_path_inputs_stay_paths():
    # Regression guard: the old JSON round-trip degraded Path-typed inputs
    # to str inside the container, diverging from the in-process path.
    ref = pystep(image="casa:latest", backend="docker")(path_func)

    captured_inputs = {}
    fake = _fake_container_run({"basename": "test.ms"}, capture_inputs=captured_inputs)
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=fake):
        ref(vis="/data/test.ms")

    assert isinstance(captured_inputs["vis"], Path)


def test_pystep_container_runner_script_content():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    captured_runner = {}

    original_write_text = Path.write_text

    def capture_write_text(self, content, *args, **kwargs):
        if self.name == "runner.py":
            captured_runner["content"] = content
        return original_write_text(self, content, *args, **kwargs)

    fake = _fake_container_run({"result": "ok"})
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=fake):
        with patch.object(Path, "write_text", capture_write_text):
            ref(ms="test.ms")

    runner = captured_runner["content"]
    assert "import json" in runner
    # The target is loaded from its file as an isolated module, with the
    # framework/target packages stubbed via a front-of-meta_path finder --
    # never imported by dotted path onto the host site-packages sys.path.
    assert "sys.meta_path.insert(0, _StubFinder())" in runner
    assert "spec_from_file_location" in runner
    assert "container_func" in runner
    assert "inputs.pkl" in runner
    assert "outputs.json" in runner
    assert "model_dump" in runner


def test_pystep_container_backend_override_at_call_time():
    ref = pystep(image="casa:latest")(container_func)

    fake = _fake_container_run({"result": "ok"})
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=fake) as mock_run:
        ref(ms="test.ms", backend="docker")

    argv = mock_run.call_args[0][0]
    assert argv[0] == "docker"


def test_pystep_container_stdout_noise_does_not_corrupt_outputs():
    # Outputs travel via the outputs file, so anything the function prints
    # to stdout is irrelevant to output parsing.
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    fake = _fake_container_run({"result": "from-file"}, stdout="some log line\nanother log\n")
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=fake):
        result = ref(ms="test.ms")

    assert result.success
    assert result.outputs.result == "from-file"


def test_pystep_container_missing_outputs_file_raises():
    # Exit 0 without an outputs file is a broken contract: fail loudly
    # instead of fabricating empty outputs and reporting success.
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = ""
    mock_proc.stderr = ""

    with patch("shinobi.steps.pyfunc.run_streaming", return_value=mock_proc):
        with pytest.raises(TypeError, match="no readable outputs file"):
            ref(ms="test.ms")


def test_pystep_container_non_dict_output_raises():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    fake = _fake_container_run(42)
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=fake):
        with pytest.raises(TypeError, match="must return"):
            ref(ms="test.ms")


def test_pystep_container_nested_function_raises():
    def nested(ms: str) -> ContainerOutputs:
        return ContainerOutputs(result="x")

    ref = pystep(image="casa:latest", backend="docker")(nested)
    with pytest.raises(TypeError, match="importable module path"):
        ref(ms="test.ms")


def test_exec_context_import_func_builtin():
    from shinobi.steps.dispatch import ExecContext
    from shinobi.steps.schema import Scope
    from pydantic import BaseModel

    class DummyModel(BaseModel):
        pass

    scope = Scope(name="test", inputs_model=DummyModel, outputs_model=DummyModel)
    ctx = ExecContext(scope, {})

    len_fn = ctx.import_func("len")
    assert len_fn([1, 2, 3]) == 3

    print_fn = ctx.import_func("print")
    assert print_fn is print


def test_exec_context_import_func_module():
    from shinobi.steps.dispatch import ExecContext
    from shinobi.steps.schema import Scope
    from pydantic import BaseModel

    class DummyModel(BaseModel):
        pass

    scope = Scope(name="test", inputs_model=DummyModel, outputs_model=DummyModel)
    ctx = ExecContext(scope, {})

    join_fn = ctx.import_func("join", "os.path")
    assert join_fn("/a", "b") == "/a/b"

    path_class = ctx.import_func("Path", "pathlib")
    assert path_class("/tmp") == Path("/tmp")


# --- ctx injection (leading `ctx` parameter) -------------------------------


def test_pystep_ctx_param_is_not_an_input_field():
    ref = pystep()(ctx_func)
    fields = set(ref.step.inputs_model.model_fields)
    assert fields == {"a", "b"}
    assert "ctx" not in fields


def test_pystep_ctx_injected_in_process():
    ref = pystep()(ctx_func)
    result = ref(a="/x", b="y", backend="native")
    assert result.success
    assert result.outputs.joined == "/x/y"


def test_pystep_ctx_shim_in_container_runner():
    ref = pystep(image="casa:latest", backend="docker")(ctx_func)

    captured_runner = {}
    original_write_text = Path.write_text

    def capture_write_text(self, content, *args, **kwargs):
        if self.name == "runner.py":
            captured_runner["content"] = content
        return original_write_text(self, content, *args, **kwargs)

    fake = _fake_container_run({"joined": "/x/y"})
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=fake):
        with patch.object(Path, "write_text", capture_write_text):
            ref(a="/x", b="y")

    runner = captured_runner["content"]
    assert "class _Ctx" in runner
    # The shim body is lifted from the real ExecContext.import_func, so the
    # two cannot drift.
    assert "def import_func" in runner
    assert "ctx_func(ctx, **inputs)" in runner


def test_pystep_no_ctx_runner_has_no_shim():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    captured_runner = {}
    original_write_text = Path.write_text

    def capture_write_text(self, content, *args, **kwargs):
        if self.name == "runner.py":
            captured_runner["content"] = content
        return original_write_text(self, content, *args, **kwargs)

    fake = _fake_container_run({"result": "ok"})
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=fake):
        with patch.object(Path, "write_text", capture_write_text):
            ref(ms="test.ms")

    runner = captured_runner["content"]
    assert "class _Ctx" not in runner
    assert "container_func(**inputs)" in runner


def test_pystep_runner_references_real_io_paths_not_shinobi_io():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    captured_runner = {}
    original_write_text = Path.write_text

    def capture_write_text(self, content, *args, **kwargs):
        if self.name == "runner.py":
            captured_runner["content"] = content
        return original_write_text(self, content, *args, **kwargs)

    fake = _fake_container_run({"result": "ok"})
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=fake):
        with patch.object(Path, "write_text", capture_write_text):
            ref(ms="test.ms")

    runner = captured_runner["content"]
    assert "/shinobi_io" not in runner
    assert "shinobi_pystep_" in runner  # the real temp io dir path
    assert "inputs.pkl" in runner
    assert "outputs.json" in runner


# --- end-to-end: actually execute the generated runner ---------------------
#
# These run the generated runner with the host interpreter (see
# _run_runner_on_host). Because the runner uses identity-mounted paths, this
# reproduces exactly what happens inside the container -- a genuine
# regression guard for the mount/path wiring that mocked argv checks miss.


def test_pystep_container_runner_executes_end_to_end():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=_run_runner_on_host):
        result = ref(ms="real.ms", niter=7)

    assert result.success, result.stderr
    assert result.outputs.result == "real.ms:7"


def test_pystep_container_runs_without_framework_in_container():
    # The whole point of the stub runner: the container's Python needs
    # neither shinobi nor pydantic (nor their compiled deps like
    # pydantic_core, which are ABI-locked to the host venv's Python). Prove
    # it by executing the generated runner in a subprocess where importing
    # shinobi or pydantic *raises* -- it must still produce outputs, entirely
    # via the injected stubs.
    bootstrap = (
        "import runpy, sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('shinobi', 'pydantic'):\n"
        "            raise ImportError('blocked in container: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        "runpy.run_path(sys.argv[1], run_name='__main__')\n"
    )

    def hardened_run(argv, *args, **kwargs):
        runner = next(a for a in argv if a.endswith("runner.py"))
        proc = _REAL_SUBPROCESS_RUN(
            [sys.executable, "-c", bootstrap, runner],
            capture_output=True,
            text=True,
        )
        result = MagicMock()
        result.returncode = proc.returncode
        result.stdout = proc.stdout
        result.stderr = proc.stderr
        return result

    ref = pystep(image="casa:latest", backend="docker")(container_func)
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=hardened_run):
        result = ref(ms="hardened.ms", niter=3)

    assert result.success, result.stderr
    assert result.outputs.result == "hardened.ms:3"


def test_pystep_container_runner_executes_ctx_end_to_end():
    ref = pystep(image="casa:latest", backend="docker")(ctx_func)

    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=_run_runner_on_host):
        result = ref(a="/x", b="y")

    assert result.success, result.stderr
    assert result.outputs.joined == "/x/y"


def test_pystep_container_runner_executes_path_inputs_end_to_end():
    # Regression guard for the JSON round-trip that turned Path inputs into
    # str inside the container: path_func calls vis.name, which only works
    # if the runner hands it a real Path.
    ref = pystep(image="casa:latest", backend="docker")(path_func)

    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=_run_runner_on_host):
        result = ref(vis="/data/real.ms")

    assert result.success, result.stderr
    assert result.outputs.basename == "real.ms"


# A module-level *decorated* pystep: `@pystep` rebinds this name to a StepRef,
# so the container runner -- which re-imports the function by name -- gets the
# StepRef, not the function. This is how dosho's cabs are all defined, and it
# regressed as `StepRef.__call__() takes 1 positional argument but 2 were
# given` until the runner learned to unwrap StepRef -> adapter -> function.
@pystep(image="casa:latest", backend="docker")
def decorated_container_func(ms: str, niter: int = 100) -> ContainerOutputs:
    return ContainerOutputs(result=f"{ms}:{niter}")


def test_pystep_decorator_form_runner_executes_end_to_end():
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=_run_runner_on_host):
        result = decorated_container_func(ms="dec.ms", niter=5)

    assert result.success, result.stderr
    assert result.outputs.result == "dec.ms:5"


# -- sandbox path normalization (issue #28: containerized pysteps with
# sandboxing must normalize output paths to workspace-relative, so cache
# entries are portable across sandbox states) --


class PathEchoOutputs(BaseModel):
    target: Path


def path_echo(target: Path) -> PathEchoOutputs:
    """A pystep that echoes its input path back as an output -- the exact
    shape that exposed issue #28: sandboxing absolutizes the input, the
    function echoes it, and the output ends up absolute unless normalized."""
    return PathEchoOutputs(target=target)


def test_sandboxed_container_pystep_relativizes_path_outputs(tmp_path, monkeypatch):
    """A sandboxed container pystep that echoes a relative path input back
    as an output must produce a relative path in the result -- not the
    absolutized workspace-anchored value the container received."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / ".shinobi/work"))

    ref = pystep(image="test:latest", backend="docker", sandbox=True)(path_echo)

    captured_inputs = {}
    fake = _fake_container_run(
        {"target": str(tmp_path / "out/probe.txt")},
        capture_inputs=captured_inputs,
    )
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=fake):
        result = ref(target=Path("out/probe.txt"))

    assert result.success, result.stderr
    assert result.outputs.target == Path("out/probe.txt")
    assert result.sandboxed is True
    # The container received the absolute path (sandbox anchoring worked)
    assert Path(captured_inputs["target"]).is_absolute()


def test_unsandboxed_container_pystep_keeps_relative_path_outputs(tmp_path, monkeypatch):
    """An unsandboxed container pystep that echoes a relative path input
    back as an output must also produce a relative path -- same result as
    the sandboxed case, so cache entries are interchangeable."""
    monkeypatch.chdir(tmp_path)

    ref = pystep(image="test:latest", backend="docker")(path_echo)

    fake = _fake_container_run({"target": "out/probe.txt"})
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=fake):
        result = ref(target=Path("out/probe.txt"))

    assert result.success, result.stderr
    assert result.outputs.target == Path("out/probe.txt")
    assert result.sandboxed is False


def test_sandboxed_container_pystep_keeps_absolute_paths_outside_workspace(tmp_path, monkeypatch):
    """An absolute output path outside the workspace passes through
    unchanged -- only paths within the workspace are relativized."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / ".shinobi/work"))

    ref = pystep(image="test:latest", backend="docker", sandbox=True)(path_echo)

    fake = _fake_container_run({"target": "/elsewhere/data.txt"})
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=fake):
        result = ref(target=Path("/elsewhere/data.txt"))

    assert result.success, result.stderr
    assert result.outputs.target == Path("/elsewhere/data.txt")
