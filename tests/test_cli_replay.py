"""CLI coverage for `ninja replay`: re-running a recorded run from its manifest."""

import json
from datetime import datetime, timezone

import pytest
from click.testing import CliRunner

from shinobi.backends.recording import RecordingBackend
from shinobi.cli import main
from shinobi.provenance import RunManifest, StepRecord
from shinobi.steps import register_step_backend

_DIGEST = "sha256:" + "a" * 64

# Imported in-process by _resolve_target, so the "record" backend the test
# registers is visible to the cab it defines.
_RECIPE_SRC = """
from pydantic import BaseModel

from shinobi.steps.schema import Cab, Recipe


class I(BaseModel):
    x: int = 0


class O(BaseModel):
    pass


step1 = Cab(name="step1", command="true", image="alpine:3.19", backend="record",
            inputs_model=I, outputs_model=O)
rec = Recipe(name="rec", inputs_model=I, outputs_model=O)
rec.add_step("step1", step1, x=rec.inputs.x)
"""


@pytest.fixture
def recipe_file(tmp_path):
    path = tmp_path / "replay_recipe.py"
    path.write_text(_RECIPE_SRC)
    return path


@pytest.fixture
def recorder():
    recorder = RecordingBackend()
    register_step_backend("record", recorder)
    return recorder


def _manifest(recipe_file, *, target=..., digest=_DIGEST, inputs=None):
    """A frozen run of `rec`: one containerized step, digest-pinned unless
    `digest=None`. `target=...` (default) means "record the real target".
    """
    if target is ...:
        target = f"{recipe_file}:rec"
    return RunManifest(
        shinobi_version="0",
        target=target,
        generated_at=datetime.now(timezone.utc),
        backend="record",
        returncode=0,
        root=StepRecord(
            name="rec", kind="recipe", returncode=0, cached=False,
            inputs=inputs or {"x": 3}, outputs={},
            steps=[StepRecord(
                name="step1", kind="cab", returncode=0, cached=False,
                image="alpine:3.19", image_digest=digest, containerized=True,
                inputs={"x": 3}, outputs={},
            )],
        ),
    )


def _write(manifest, tmp_path):
    return manifest.write(tmp_path / "rec.run.json")


def test_replay_forces_pinned_image_and_recorded_inputs(tmp_path, recipe_file, recorder):
    mpath = _write(_manifest(recipe_file), tmp_path)
    result = CliRunner().invoke(main, ["replay", str(mpath)])
    assert result.exit_code == 0, result.output
    (cab, _argv, inputs), = recorder.calls
    assert cab.image == f"alpine@{_DIGEST}"  # the digest that originally ran
    assert inputs["x"] == 3  # the manifest's recorded inputs


def test_replay_missing_target_errors(tmp_path, recipe_file, recorder):
    mpath = _write(_manifest(recipe_file, target=None), tmp_path)
    result = CliRunner().invoke(main, ["replay", str(mpath)])
    assert result.exit_code != 0
    assert "--target" in result.output
    assert not recorder.calls


def test_replay_target_override(tmp_path, recipe_file, recorder):
    mpath = _write(_manifest(recipe_file, target=None), tmp_path)
    result = CliRunner().invoke(main, ["replay", str(mpath), "--target", f"{recipe_file}:rec"])
    assert result.exit_code == 0, result.output
    assert recorder.calls


def test_replay_refuses_unpinned_manifest(tmp_path, recipe_file, recorder):
    mpath = _write(_manifest(recipe_file, digest=None), tmp_path)
    result = CliRunner().invoke(main, ["replay", str(mpath)])
    assert result.exit_code != 0
    assert "step1" in result.output and "--allow-unpinned" in result.output
    assert not recorder.calls


def test_replay_allow_unpinned_runs_original_ref(tmp_path, recipe_file, recorder):
    mpath = _write(_manifest(recipe_file, digest=None), tmp_path)
    result = CliRunner().invoke(main, ["replay", str(mpath), "--allow-unpinned"])
    assert result.exit_code == 0, result.output
    (cab, _argv, _inputs), = recorder.calls
    assert cab.image == "alpine:3.19"  # unpinned step keeps its original ref


def test_replay_recipe_shape_mismatch_errors(tmp_path, recipe_file, recorder):
    manifest = _manifest(recipe_file)
    manifest.root.steps.append(StepRecord(
        name="gone", kind="cab", returncode=0, cached=False, inputs={}, outputs={},
    ))
    mpath = _write(manifest, tmp_path)
    result = CliRunner().invoke(main, ["replay", str(mpath)])
    assert result.exit_code != 0
    assert "gone" in result.output
    assert not recorder.calls


def test_replay_emits_new_manifest_with_target(tmp_path, recipe_file, recorder, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setenv("SHINOBI_PROVENANCE__DIR", str(runs))
    mpath = _write(_manifest(recipe_file), tmp_path)
    result = CliRunner().invoke(main, ["replay", str(mpath)])
    assert result.exit_code == 0, result.output
    files = list(runs.glob("rec.*.run.json"))
    assert files, "a replay is itself a provenance run and must emit a manifest"
    assert json.loads(files[-1].read_text())["target"] == f"{recipe_file}:rec"


def test_replay_invalid_manifest_inputs_error(tmp_path, recipe_file, recorder):
    mpath = _write(_manifest(recipe_file, inputs={"x": "abc"}), tmp_path)
    result = CliRunner().invoke(main, ["replay", str(mpath)])
    assert result.exit_code != 0
    assert "manifest inputs" in result.output
    assert not recorder.calls


def test_replay_unreadable_manifest_errors(tmp_path, recorder):
    bad = tmp_path / "bad.run.json"
    bad.write_text("{not json")
    result = CliRunner().invoke(main, ["replay", str(bad)])
    assert result.exit_code != 0
    assert "cannot read run manifest" in result.output
