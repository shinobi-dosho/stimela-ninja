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
    prepare_output_parents,
    prune_unused_parents,
    relativize_path_outputs,
)
from shinobi.steps.dispatch import register_step_backend
from shinobi.steps.schema import Cab, ParamMeta, Recipe, Scope, StepRef

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


# --------------------------------------------------------- unit: output relativization


def test_relativize_converts_absolute_paths_within_workspace(tmp_path):
    scope = make_scope(outputs={"result": ("File", False, None)})
    outputs = scope.outputs_model(result=tmp_path / "out.dat")
    relativized = relativize_path_outputs(scope, outputs, tmp_path)
    assert relativized.result == Path("out.dat")


def test_relativize_leaves_paths_outside_workspace_absolute(tmp_path):
    scope = make_scope(outputs={"result": ("File", False, None)})
    outputs = scope.outputs_model(result=Path("/elsewhere/out.dat"))
    relativized = relativize_path_outputs(scope, outputs, tmp_path)
    assert relativized.result == Path("/elsewhere/out.dat")


def test_relativize_leaves_relative_paths_unchanged(tmp_path):
    scope = make_scope(outputs={"result": ("File", False, None)})
    outputs = scope.outputs_model(result=Path("out.dat"))
    relativized = relativize_path_outputs(scope, outputs, tmp_path)
    assert relativized.result == Path("out.dat")


def test_relativize_handles_list_valued_outputs(tmp_path):
    scope = make_scope(outputs={"files": ("List[File]", False, None)})
    outputs = scope.outputs_model(files=[tmp_path / "a.dat", Path("/elsewhere/b.dat")])
    relativized = relativize_path_outputs(scope, outputs, tmp_path)
    assert relativized.files == [Path("a.dat"), Path("/elsewhere/b.dat")]


def test_relativize_ignores_non_path_outputs(tmp_path):
    scope = make_scope(outputs={"note": ("str", False, None)})
    outputs = scope.outputs_model(note="hello")
    relativized = relativize_path_outputs(scope, outputs, tmp_path)
    assert relativized.note == "hello"


def test_relativize_returns_same_instance_when_nothing_changed(tmp_path):
    scope = make_scope(outputs={"result": ("File", False, None)})
    outputs = scope.outputs_model(result=Path("out.dat"))
    relativized = relativize_path_outputs(scope, outputs, tmp_path)
    assert relativized is outputs


def test_relativize_handles_none_values(tmp_path):
    scope = make_scope(outputs={"result": ("File", False, None)})
    outputs = scope.outputs_model(result=None)
    relativized = relativize_path_outputs(scope, outputs, tmp_path)
    assert relativized.result is None


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


def test_harvest_moves_nested_target_with_its_parent(tmp_path):
    # A declared output living inside a directory-valued declared output:
    # the parent moves wholesale and the child travels inside it. Field-name
    # order used to move the child first, then rmtree it again when the
    # parent dir landed on the same destination (data loss).
    scope = make_scope(
        outputs={
            "htmlname": ("File", False, "plots/gain.html"),
            "outdir": ("Directory", False, "plots"),
        }
    )
    sandbox = _sandbox_with(tmp_path, "plots/gain.html")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    moved = harvest_outputs(scope, scope.outputs_model(), {}, sandbox, workspace)

    assert moved == [workspace / "plots"]
    assert (workspace / "plots/gain.html").read_text() == "plots/gain.html"


def test_harvest_glob_matches_travel_with_declared_dir_parent(tmp_path):
    # Same hazard via the glob route: matches inside a directory-valued
    # declared output ride along with the parent's move.
    scope = make_scope(
        outputs={"outdir": ("Directory", False, "plots")}, harvest=["plots/*.html"]
    )
    sandbox = _sandbox_with(tmp_path, "plots/gain.html", "plots/phase.html")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    moved = harvest_outputs(scope, scope.outputs_model(), {}, sandbox, workspace)

    assert moved == [workspace / "plots"]
    assert (workspace / "plots/gain.html").exists()
    assert (workspace / "plots/phase.html").exists()


def test_harvest_skips_absolutely_resolved_patterns(tmp_path):
    # `"{prefix}-*"` with an absolute prefix: the tool wrote straight to the
    # absolute destination, so there's nothing in the sandbox to rescue --
    # a successful run must not fail on ordinary input.
    scope = make_scope(harvest=["{prefix}-*"])
    sandbox = _sandbox_with(tmp_path)
    moved = harvest_outputs(scope, scope.outputs_model(), {"prefix": "/data/img"}, sandbox, tmp_path)
    assert moved == []


def test_harvest_warns_and_skips_dotdot_escapes(tmp_path):
    scope = make_scope(harvest=["../escape-*"])
    sandbox = _sandbox_with(tmp_path)
    with pytest.warns(UserWarning, match="escapes the sandbox"):
        moved = harvest_outputs(scope, scope.outputs_model(), {}, sandbox, tmp_path)
    assert moved == []


def test_harvest_rejects_pattern_with_unknown_input(tmp_path):
    scope = make_scope(harvest=["{nope}-*"])
    sandbox = _sandbox_with(tmp_path)
    with pytest.raises(ParameterError, match="unknown input"):
        harvest_outputs(scope, scope.outputs_model(), {}, sandbox, tmp_path)


# ------------------------------------------------- unit: output parent pre-creation


def test_prepare_parents_for_relative_output_from_same_named_input(tmp_path):
    # The ragavi shape: a string-typed stem input feeding a same-named
    # path-typed output -- the stem stays relative so the tool writes in the
    # sandbox, which means its parent dir must exist there before the run.
    scope = make_scope(
        inputs={"htmlname": ("str", True, None)},
        outputs={"htmlname": ("File", False, None)},
    )
    prepare_output_parents(scope, {"htmlname": "plots/gain.html"}, tmp_path)
    assert (tmp_path / "plots").is_dir()


def test_prepare_parents_resolves_implicit_output_templates(tmp_path):
    # The wsclean shape: `-name img/run1` with implicit output templates.
    cab = Cab(
        name="imager",
        command="/bin/true",
        inputs_model=build_model("In", {"prefix": ("str", True, None)}),
        outputs_model=build_model("Out", {"image": ("File", False, None)}),
        field_meta={"image": ParamMeta(implicit="{prefix}-MFS-image.fits")},
    )
    prepare_output_parents(cab, {"prefix": "img/run1"}, tmp_path)
    assert (tmp_path / "img").is_dir()


def test_prepare_parents_uses_output_field_defaults(tmp_path):
    scope = make_scope(outputs={"result": ("File", False, "sub/out.dat")})
    prepare_output_parents(scope, {}, tmp_path)
    assert (tmp_path / "sub").is_dir()


def test_prepare_parents_creates_literal_prefix_of_harvest_patterns(tmp_path):
    scope = make_scope(harvest=["{prefix}-*.fits", "logs/*/detail-*.txt"])
    prepare_output_parents(scope, {"prefix": "img/run1"}, tmp_path)
    assert (tmp_path / "img").is_dir()
    assert (tmp_path / "logs").is_dir()
    # The glob part and anything under it is the tool's to create.
    assert list((tmp_path / "logs").iterdir()) == []


def test_prepare_parents_none_input_suppresses_implicit_like_fill_outputs(tmp_path):
    # `_fill_outputs` lets a same-named input that is present-but-None win
    # over `implicit`; pre-creation must agree and create nothing.
    cab = Cab(
        name="c",
        command="/bin/true",
        inputs_model=build_model("In", {"image": ("str", False, None)}),
        outputs_model=build_model("Out", {"image": ("File", False, None)}),
        field_meta={"image": ParamMeta(implicit="img/{image}.fits")},
    )
    prepare_output_parents(cab, {"image": None}, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_prepare_parents_skips_malformed_implicit_templates(tmp_path):
    # A bad format spec raises ValueError, not KeyError -- best-effort means
    # no crash here; `_fill_outputs` reports the real error post-run.
    cab = Cab(
        name="c",
        command="/bin/true",
        inputs_model=build_model("In", {"prefix": ("str", True, None)}),
        outputs_model=build_model("Out", {"image": ("File", False, None)}),
        field_meta={"image": ParamMeta(implicit="{prefix:d}-img.fits")},
    )
    prepare_output_parents(cab, {"prefix": "img/run1"}, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_prune_removes_only_unused_precreated_dirs(tmp_path):
    scope = make_scope(
        inputs={"a": ("str", True, None), "b": ("str", True, None)},
        outputs={"a": ("File", False, None), "b": ("File", False, None)},
    )
    created = prepare_output_parents(
        scope, {"a": "used/x.dat", "b": "unused/deep/y.dat"}, tmp_path
    )
    (tmp_path / "used/x.dat").write_text("x")

    prune_unused_parents(created)

    assert (tmp_path / "used/x.dat").exists()
    assert not (tmp_path / "unused").exists()  # pruned bottom-up, deep first


def test_prepare_parents_skips_absolute_escapes_and_bare_names(tmp_path):
    scope = make_scope(
        inputs={"stem": ("str", True, None)},
        outputs={
            "abs_out": ("File", False, "/elsewhere/x.dat"),
            "flat": ("File", False, "out.dat"),
            "stem": ("File", False, None),
        },
        harvest=["../escape-*", "{missing}-*"],
    )
    prepare_output_parents(scope, {"stem": "../up/x.dat"}, tmp_path)
    assert list(tmp_path.iterdir()) == []  # nothing created, nothing raised


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


def test_sandboxed_tool_can_write_relative_output_in_fresh_subdir(workspace):
    # Regression: a tool that doesn't `mkdir -p` its own output stem (ragavi's
    # htmlname, wsclean's -name) must not crash on a relative output like
    # `plots/gain.html` just because the sandbox starts empty.
    script = _script(workspace, "echo report > plots/gain.html\necho junk > junk.log\n")
    cab = Cab(
        name="plotter",
        command=str(script),
        inputs_model=build_model("In", {"htmlname": ("str", True, None)}),
        outputs_model=build_model("Out", {"htmlname": ("File", False, None)}),
        sandbox=True,
    )
    result = cab(backend="native", htmlname="plots/gain.html")

    assert result.success
    assert (workspace / "plots/gain.html").read_text() == "report\n"
    assert not (workspace / "junk.log").exists()
    assert list((workspace / WORK_ROOT).iterdir()) == []  # sandbox discarded


def test_unused_precreated_dir_never_clobbers_workspace_dir_output(workspace):
    # A directory-typed passthrough output whose value equals a pre-created
    # parent: if the tool writes nothing (no-op run), the empty pre-created
    # dir must not be harvested over the workspace's real `plots/`.
    script = _script(workspace, "true\n")
    (workspace / "plots").mkdir()
    (workspace / "plots/precious.txt").write_text("keep me")
    cab = Cab(
        name="noop",
        command=str(script),
        inputs_model=build_model(
            "In", {"outdir": ("str", True, None), "htmlname": ("str", True, None)}
        ),
        outputs_model=build_model(
            "Out", {"outdir": ("Directory", False, None), "htmlname": ("File", False, None)}
        ),
        sandbox=True,
    )
    result = cab(backend="native", outdir="plots", htmlname="plots/gain.html")

    assert result.success
    assert (workspace / "plots/precious.txt").read_text() == "keep me"


def test_harvest_glob_never_rescues_unused_precreated_dirs(workspace):
    # `logs/x/*.txt` pre-creates `logs/x`; the broader `logs/*` glob must not
    # then rescue that empty dir over the workspace's populated `logs/x`.
    script = _script(workspace, "true\n")
    (workspace / "logs/x").mkdir(parents=True)
    (workspace / "logs/x/old.txt").write_text("keep me")
    cab = Cab(
        name="noop",
        command=str(script),
        inputs_model=build_model("In", {}),
        outputs_model=build_model("Out", {}),
        sandbox=True,
        harvest=["logs/x/*.txt", "logs/*"],
    )
    result = cab(backend="native")

    assert result.success
    assert (workspace / "logs/x/old.txt").read_text() == "keep me"


def test_tool_created_empty_dir_output_still_harvests(workspace):
    # Pruning only touches pre-created dirs: an empty dir the tool itself
    # made is a real output and harvests as it would have unsandboxed.
    script = _script(workspace, "mkdir emptydir\n")
    cab = Cab(
        name="dirmaker",
        command=str(script),
        inputs_model=build_model("In", {}),
        outputs_model=build_model("Out", {"d": ("Directory", False, "emptydir")}),
        sandbox=True,
    )
    result = cab(backend="native")

    assert result.success
    assert (workspace / "emptydir").is_dir()


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
