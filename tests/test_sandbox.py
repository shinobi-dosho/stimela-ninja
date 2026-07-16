"""Tests for per-step sandbox execution (src/shinobi/sandbox.py) and its
dispatch wiring: absolutized path inputs, allowlist harvest, keep-on-failure,
and the cache-style enable precedence chain.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shinobi.backends.recording import RecordingBackend
from shinobi.exceptions import ParameterError
from shinobi.loaders import build_model
from shinobi.sandbox import (
    absolutize_path_inputs,
    create_sandbox,
    discard_sandbox,
    harvest_outputs,
)
from shinobi.steps.dispatch import register_step_backend
from shinobi.steps.schema import Cab, Recipe, Scope, StepRef

WORK_ROOT = ".shinobi/work"


def make_scope(inputs=None, outputs=None, harvest=None) -> Scope:
    return Scope(
        name="s",
        inputs_model=build_model("In", inputs or {}),
        outputs_model=build_model("Out", outputs or {}),
        harvest=harvest or [],
    )


# ---------------------------------------------------------------- unit: inputs


def test_absolutize_anchors_relative_path_inputs_only(tmp_path):
    scope = make_scope(inputs={"ms": ("MS", True, None), "label": ("str", True, None)})
    prepared = {"ms": Path("data.ms"), "label": "x"}
    anchored = absolutize_path_inputs(scope, prepared, tmp_path)
    assert anchored["ms"] == tmp_path / "data.ms"
    assert anchored["label"] == "x"  # not path-typed: untouched
    assert prepared["ms"] == Path("data.ms")  # original dict untouched


def test_absolutize_leaves_absolute_paths_alone(tmp_path):
    scope = make_scope(inputs={"ms": ("MS", True, None)})
    anchored = absolutize_path_inputs(scope, {"ms": Path("/elsewhere/data.ms")}, tmp_path)
    assert anchored["ms"] == Path("/elsewhere/data.ms")


def test_absolutize_handles_list_valued_path_inputs(tmp_path):
    scope = make_scope(inputs={"vis": ("List[File]", True, None)})
    anchored = absolutize_path_inputs(scope, {"vis": [Path("a.ms"), Path("/abs/b.ms")]}, tmp_path)
    assert anchored["vis"] == [tmp_path / "a.ms", Path("/abs/b.ms")]


# --------------------------------------------------------------- unit: harvest


def _sandbox_with(tmp_path: Path, *files: str) -> Path:
    sandbox = tmp_path / "sandbox"
    for name in files:
        path = sandbox / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    sandbox.mkdir(exist_ok=True)
    return sandbox


def test_harvest_moves_declared_outputs_and_leaves_junk(tmp_path):
    scope = make_scope(outputs={"result": ("File", False, "out.dat")})
    sandbox = _sandbox_with(tmp_path, "out.dat", "junk.log")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outputs = scope.outputs_model()

    moved = harvest_outputs(scope, outputs, {}, sandbox, workspace)

    assert moved == [workspace / "out.dat"]
    assert (workspace / "out.dat").read_text() == "out.dat"
    assert not (workspace / "junk.log").exists()
    assert (sandbox / "junk.log").exists()  # junk stays behind for discard


def test_harvest_globs_rescue_dynamic_output_families(tmp_path):
    scope = make_scope(harvest=["{prefix}-*.fits"])
    sandbox = _sandbox_with(tmp_path, "img-0000.fits", "img-0001.fits", "other.fits")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    moved = harvest_outputs(scope, scope.outputs_model(), {"prefix": "img"}, sandbox, workspace)

    assert sorted(p.name for p in moved) == ["img-0000.fits", "img-0001.fits"]
    assert not (workspace / "other.fits").exists()


def test_harvest_skips_absolute_and_never_written_outputs(tmp_path):
    scope = make_scope(
        outputs={"abs_out": ("File", False, "/elsewhere/x.dat"), "ghost": ("File", False, "ghost.dat")}
    )
    sandbox = _sandbox_with(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert harvest_outputs(scope, scope.outputs_model(), {}, sandbox, workspace) == []


def test_harvest_creates_subdir_parents_and_replaces_existing(tmp_path):
    scope = make_scope(outputs={"result": ("File", False, "sub/out.dat")})
    sandbox = _sandbox_with(tmp_path, "sub/out.dat")
    workspace = tmp_path / "ws"
    (workspace / "sub").mkdir(parents=True)
    (workspace / "sub/out.dat").write_text("stale")

    harvest_outputs(scope, scope.outputs_model(), {}, sandbox, workspace)

    assert (workspace / "sub/out.dat").read_text() == "sub/out.dat"


def test_harvest_moves_directory_outputs(tmp_path):
    scope = make_scope(outputs={"table": ("Directory", False, "gains.tbl")})
    sandbox = _sandbox_with(tmp_path, "gains.tbl/data.bin")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    harvest_outputs(scope, scope.outputs_model(), {}, sandbox, workspace)

    assert (workspace / "gains.tbl/data.bin").read_text() == "gains.tbl/data.bin"


@pytest.mark.parametrize("pattern", ["../escape-*", "/abs/*"])
def test_harvest_rejects_escaping_patterns(tmp_path, pattern):
    scope = make_scope(harvest=[pattern])
    sandbox = _sandbox_with(tmp_path)
    with pytest.raises(ParameterError, match="relative glob"):
        harvest_outputs(scope, scope.outputs_model(), {}, sandbox, tmp_path)


def test_harvest_rejects_pattern_with_unknown_input(tmp_path):
    scope = make_scope(harvest=["{nope}-*"])
    sandbox = _sandbox_with(tmp_path)
    with pytest.raises(ParameterError, match="unknown input"):
        harvest_outputs(scope, scope.outputs_model(), {}, sandbox, tmp_path)


def test_create_and_discard_sandbox(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sandbox = create_sandbox(WORK_ROOT, "recipe/step")
    assert sandbox.is_dir() and sandbox.is_absolute()
    assert sandbox.parent == (tmp_path / WORK_ROOT).resolve()
    assert sandbox.name.startswith("recipe_step-")  # label sanitized, not nested
    (sandbox / "junk").write_text("x")
    discard_sandbox(sandbox)
    assert not sandbox.exists()


# ------------------------------------------------- integration: native backend


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Override conftest's global sandbox-dir isolation: these tests assert on
    # the sandbox root's location relative to the workspace.
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / WORK_ROOT))
    return tmp_path


def _script(workspace: Path, body: str) -> Path:
    path = workspace / "tool.sh"
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


def _messy_cab(script: Path, **kwargs) -> Cab:
    return Cab(
        name="messy",
        command=str(script),
        inputs_model=build_model("In", {}),
        outputs_model=build_model("Out", {"result": ("File", False, "out.dat")}),
        **kwargs,
    )


def test_sandboxed_run_keeps_declared_output_and_mops_junk(workspace):
    script = _script(workspace, "echo data > out.dat\necho junk > junk.log\n")
    result = _messy_cab(script, sandbox=True)(backend="native")

    assert result.success
    assert (workspace / "out.dat").read_text() == "data\n"
    assert not (workspace / "junk.log").exists()
    assert list((workspace / WORK_ROOT).iterdir()) == []  # sandbox discarded


def test_unsandboxed_run_still_drops_junk_in_cwd(workspace):
    # Off by default (AppConfig.sandbox.enabled=False): behavior unchanged.
    script = _script(workspace, "echo data > out.dat\necho junk > junk.log\n")
    result = _messy_cab(script)(backend="native")

    assert result.success
    assert (workspace / "junk.log").exists()


def test_call_time_sandbox_flag_overrides_scope_default(workspace):
    script = _script(workspace, "echo data > out.dat\necho junk > junk.log\n")
    result = _messy_cab(script)(backend="native", sandbox=True)

    assert result.success
    assert not (workspace / "junk.log").exists()


def test_failed_sandboxed_run_keeps_sandbox_for_post_mortem(workspace):
    script = _script(workspace, "echo junk > junk.log\nexit 3\n")
    with pytest.warns(UserWarning, match="post-mortem"):
        result = _messy_cab(script, sandbox=True)(backend="native")

    assert result.returncode == 3
    assert not (workspace / "junk.log").exists()
    kept = list((workspace / WORK_ROOT).iterdir())
    assert len(kept) == 1
    assert (kept[0] / "junk.log").exists()


def test_sandboxed_run_harvest_globs_end_to_end(workspace):
    script = _script(workspace, "echo a > img-0000.fits\necho junk > junk.log\n")
    cab = Cab(
        name="imager",
        command=str(script),
        inputs_model=build_model("In", {}),
        outputs_model=build_model("Out", {}),
        sandbox=True,
        harvest=["img-*.fits"],
    )
    result = cab(backend="native")

    assert result.success
    assert (workspace / "img-0000.fits").exists()
    assert not (workspace / "junk.log").exists()


def test_sandbox_absolutizes_path_inputs_and_passes_cwd_to_backend(workspace):
    recorder = RecordingBackend()
    register_step_backend("sandbox-recorder", recorder)
    (workspace / "data.txt").write_text("x")
    cab = Cab(
        name="c",
        command="/bin/echo",
        inputs_model=build_model("In", {"infile": ("File", True, None)}),
        outputs_model=build_model("Out", {}),
        sandbox=True,
    )
    result = cab(backend="sandbox-recorder", infile="data.txt")

    assert result.success
    _, _, recorded_inputs = recorder.calls[0]
    infile = Path(str(recorded_inputs["infile"]))
    assert infile.is_absolute()
    assert infile.name == "data.txt" and infile.parent.samefile(workspace)
    assert Path(recorder.cwds[0]).parent == (workspace / WORK_ROOT).resolve()


def test_recipe_sandbox_is_inherited_by_steps(workspace):
    script = _script(workspace, "echo data > out.dat\necho junk > junk.log\n")
    recipe = Recipe(
        name="r",
        inputs_model=build_model("RIn", {}),
        outputs_model=build_model("ROut", {}),
        sandbox=True,
        steps=[StepRef(name="messy", step=_messy_cab(script))],
    )
    result = recipe(backend="native")

    assert result.success
    assert (workspace / "out.dat").exists()
    assert not (workspace / "junk.log").exists()
