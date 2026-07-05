"""Tests for container-executable pysteps (src/shinobi/steps/pyfunc.py).

Verifies that `@shinobi.pystep(image=...)` correctly dispatches to a
container backend when one is resolved, and falls back to in-process
execution when the backend is `native`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from shinobi import pystep
from shinobi.steps.schema import Cab, Scope

# Captured before any test patches subprocess.run -- patching
# `shinobi.steps.pyfunc.subprocess.run` rebinds the shared subprocess module's
# `run`, so the end-to-end helper needs the genuine implementation to avoid
# recursing into its own patch.
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


def _run_runner_on_host(argv, *args, **kwargs):
    """subprocess.run stand-in that simulates the container by executing the
    generated runner with the host interpreter. Because the runner and its
    inputs use identity-bind-mounted (host) paths, running it directly on the
    host exercises the real generated script end-to-end -- catching any
    path/import/ctx-shim breakage without needing a container runtime.
    """
    runner = next(a for a in argv if a.endswith("runner.py"))
    return _REAL_SUBPROCESS_RUN(
        [sys.executable, runner], capture_output=True, text=True
    )


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

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps({"result": "test.ms:100"})
    mock_proc.stderr = ""

    with patch("shinobi.steps.pyfunc.subprocess.run", return_value=mock_proc) as mock_run:
        ref(ms="test.ms")

    mock_run.assert_called_once()
    argv = mock_run.call_args[0][0]

    assert argv[0] == "docker"
    assert argv[1] == "run"
    assert "--rm" in argv
    assert "casa:latest" in argv
    assert "python3" in argv

    image_idx = argv.index("casa:latest")
    inner_argv = argv[image_idx + 1:]
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

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps({"result": "test.ms:100"})
    mock_proc.stderr = ""

    with patch("shinobi.steps.pyfunc.subprocess.run", return_value=mock_proc) as mock_run:
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

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps({"result": "test.ms:100"})
    mock_proc.stderr = ""

    with patch("shinobi.steps.pyfunc.subprocess.run", return_value=mock_proc) as mock_run:
        ref(ms="test.ms")

    argv = mock_run.call_args[0][0]
    assert argv[0] == "apptainer"
    assert argv[1] == "exec"
    assert "casa:latest" in argv


def test_pystep_container_parses_json_output():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps({"result": "parsed.ms:999"})
    mock_proc.stderr = ""

    with patch("shinobi.steps.pyfunc.subprocess.run", return_value=mock_proc):
        result = ref(ms="parsed.ms", niter=999)

    assert result.success
    assert result.outputs.result == "parsed.ms:999"


def test_pystep_container_handles_empty_outputs():
    ref = pystep(image="casa:latest", backend="docker")(no_output_func)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = ""
    mock_proc.stderr = ""

    with patch("shinobi.steps.pyfunc.subprocess.run", return_value=mock_proc):
        result = ref(ms="test.ms")

    assert result.success
    assert type(result.outputs).model_fields == {}


def test_pystep_container_nonzero_exit_returns_failure():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stdout = ""
    mock_proc.stderr = "Traceback: something went wrong"

    with patch("shinobi.steps.pyfunc.subprocess.run", return_value=mock_proc):
        result = ref(ms="test.ms")

    assert not result.success
    assert result.returncode == 1
    assert "something went wrong" in result.stderr


def test_pystep_container_serializes_inputs_to_json():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps({"result": "ok"})
    mock_proc.stderr = ""

    captured_inputs = {}

    original_write_text = Path.write_text

    def capture_write_text(self, content, *args, **kwargs):
        if self.name == "inputs.json":
            captured_inputs.update(json.loads(content))
        return original_write_text(self, content, *args, **kwargs)

    with patch("shinobi.steps.pyfunc.subprocess.run", return_value=mock_proc):
        with patch.object(Path, "write_text", capture_write_text):
            ref(ms="serialised.ms", niter=42)

    assert captured_inputs["ms"] == "serialised.ms"
    assert captured_inputs["niter"] == 42


def test_pystep_container_runner_script_content():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps({"result": "ok"})
    mock_proc.stderr = ""

    captured_runner = {}

    original_write_text = Path.write_text

    def capture_write_text(self, content, *args, **kwargs):
        if self.name == "runner.py":
            captured_runner["content"] = content
        return original_write_text(self, content, *args, **kwargs)

    with patch("shinobi.steps.pyfunc.subprocess.run", return_value=mock_proc):
        with patch.object(Path, "write_text", capture_write_text):
            ref(ms="test.ms")

    runner = captured_runner["content"]
    assert "import json" in runner
    assert "sys.path.insert" in runner
    assert "container_func" in runner
    assert "inputs.json" in runner
    assert "model_dump" in runner


def test_pystep_container_backend_override_at_call_time():
    ref = pystep(image="casa:latest")(container_func)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps({"result": "ok"})
    mock_proc.stderr = ""

    with patch("shinobi.steps.pyfunc.subprocess.run", return_value=mock_proc) as mock_run:
        ref(ms="test.ms", backend="docker")

    argv = mock_run.call_args[0][0]
    assert argv[0] == "docker"


def test_pystep_container_handles_multiline_stdout():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "some log line\nanother log\n" + json.dumps({"result": "last-line"})
    mock_proc.stderr = ""

    with patch("shinobi.steps.pyfunc.subprocess.run", return_value=mock_proc):
        result = ref(ms="test.ms")

    assert result.success
    assert result.outputs.result == "last-line"


def test_pystep_container_handles_invalid_json_gracefully():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "not valid json"
    mock_proc.stderr = ""

    with patch("shinobi.steps.pyfunc.subprocess.run", return_value=mock_proc):
        result = ref(ms="test.ms")

    assert result.success
    assert getattr(result.outputs, "result", None) is None


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

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps({"joined": "/x/y"})
    mock_proc.stderr = ""

    captured_runner = {}
    original_write_text = Path.write_text

    def capture_write_text(self, content, *args, **kwargs):
        if self.name == "runner.py":
            captured_runner["content"] = content
        return original_write_text(self, content, *args, **kwargs)

    with patch("shinobi.steps.pyfunc.subprocess.run", return_value=mock_proc):
        with patch.object(Path, "write_text", capture_write_text):
            ref(a="/x", b="y")

    runner = captured_runner["content"]
    assert "class _Ctx" in runner
    assert "ctx_func(ctx, **inputs)" in runner


def test_pystep_no_ctx_runner_has_no_shim():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps({"result": "ok"})
    mock_proc.stderr = ""

    captured_runner = {}
    original_write_text = Path.write_text

    def capture_write_text(self, content, *args, **kwargs):
        if self.name == "runner.py":
            captured_runner["content"] = content
        return original_write_text(self, content, *args, **kwargs)

    with patch("shinobi.steps.pyfunc.subprocess.run", return_value=mock_proc):
        with patch.object(Path, "write_text", capture_write_text):
            ref(ms="test.ms")

    runner = captured_runner["content"]
    assert "class _Ctx" not in runner
    assert "container_func(**inputs)" in runner


def test_pystep_runner_references_real_inputs_path_not_shinobi_io():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps({"result": "ok"})
    mock_proc.stderr = ""

    captured_runner = {}
    original_write_text = Path.write_text

    def capture_write_text(self, content, *args, **kwargs):
        if self.name == "runner.py":
            captured_runner["content"] = content
        return original_write_text(self, content, *args, **kwargs)

    with patch("shinobi.steps.pyfunc.subprocess.run", return_value=mock_proc):
        with patch.object(Path, "write_text", capture_write_text):
            ref(ms="test.ms")

    runner = captured_runner["content"]
    assert "/shinobi_io" not in runner
    assert "shinobi_pystep_" in runner  # the real temp io dir path
    assert "inputs.json" in runner


# --- end-to-end: actually execute the generated runner ---------------------
#
# These run the generated runner with the host interpreter (see
# _run_runner_on_host). Because the runner uses identity-mounted paths, this
# reproduces exactly what happens inside the container -- a genuine
# regression guard for the mount/path wiring that mocked argv checks miss.


def test_pystep_container_runner_executes_end_to_end():
    ref = pystep(image="casa:latest", backend="docker")(container_func)

    with patch(
        "shinobi.steps.pyfunc.subprocess.run", side_effect=_run_runner_on_host
    ):
        result = ref(ms="real.ms", niter=7)

    assert result.success, result.stderr
    assert result.outputs.result == "real.ms:7"


def test_pystep_container_runner_executes_ctx_end_to_end():
    ref = pystep(image="casa:latest", backend="docker")(ctx_func)

    with patch(
        "shinobi.steps.pyfunc.subprocess.run", side_effect=_run_runner_on_host
    ):
        result = ref(a="/x", b="y")

    assert result.success, result.stderr
    assert result.outputs.joined == "/x/y"
