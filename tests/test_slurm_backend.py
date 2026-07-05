import subprocess
from pathlib import Path

import pytest

from shinobi.backends.slurm import SlurmBackend
from shinobi.exceptions import BackendError
from shinobi.loaders import build_model
from shinobi.steps.schema import Cab


def make_cab(**kwargs) -> Cab:
    kwargs.setdefault("name", "tool")
    kwargs.setdefault("command", "tool")
    kwargs.setdefault("inputs_model", build_model("In", {}))
    kwargs.setdefault("outputs_model", build_model("Out", {}))
    return Cab(**kwargs)


def test_script_contains_sbatch_directives_and_command():
    backend = SlurmBackend(container_runtime=None, workdir="/work", sbatch_opts={"time": "01:00:00"})
    script = backend._script(make_cab(), ["tool", "--x", "1"], {}, Path("/tmp/out.log"), Path("/tmp/err.log"))
    assert script.startswith("#!/bin/bash\n")
    assert "#SBATCH --job-name=tool" in script
    assert "#SBATCH --chdir=/work" in script
    assert "#SBATCH --output=/tmp/out.log" in script
    assert "#SBATCH --error=/tmp/err.log" in script
    assert "#SBATCH --time=01:00:00" in script
    assert script.strip().endswith("tool --x 1")


def test_script_wraps_command_in_apptainer_when_cab_has_image():
    backend = SlurmBackend(workdir="/work")
    script = backend._script(make_cab(image="tool:latest"), ["tool"], {}, Path("/tmp/o"), Path("/tmp/e"))
    assert "apptainer exec" in script
    assert "tool:latest" in script


def test_submit_returns_bare_job_id(monkeypatch):
    backend = SlurmBackend()

    def fake_run(argv, **kwargs):
        assert argv[0] == "sbatch"
        return subprocess.CompletedProcess(argv, 0, stdout="12345;cluster\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert backend._submit(Path("/tmp/job.sh")) == "12345"


def test_submit_failure_raises_backend_error(monkeypatch):
    backend = SlurmBackend()
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, stdout="", stderr="bad script"),
    )
    with pytest.raises(BackendError):
        backend._submit(Path("/tmp/job.sh"))


def test_wait_ignores_batch_and_extern_steps_and_returns_exit_code(monkeypatch):
    backend = SlurmBackend(poll_interval=0)
    calls = {"n": 0}

    def fake_run(argv, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            out = "42|RUNNING|0:0\n42.batch|RUNNING|0:0\n42.extern|RUNNING|0:0\n"
        else:
            out = "42|COMPLETED|3:0\n42.batch|COMPLETED|3:0\n42.extern|COMPLETED|0:0\n"
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("shinobi.backends.slurm.time.sleep", lambda s: None)
    assert backend._wait("42") == 3
    assert calls["n"] == 2


def test_run_end_to_end_returns_backendrun(monkeypatch):
    backend = SlurmBackend(container_runtime=None, poll_interval=0)
    cab = make_cab()

    def fake_run(argv, **kwargs):
        if argv[0] == "sbatch":
            script_text = Path(argv[-1]).read_text()
            out_path = next(
                line.split("=", 1)[1] for line in script_text.splitlines()
                if line.startswith("#SBATCH --output=")
            )
            Path(out_path).write_text("answer=99\n")
            return subprocess.CompletedProcess(argv, 0, stdout="7;cluster\n", stderr="")
        if argv[0] == "sacct":
            return subprocess.CompletedProcess(argv, 0, stdout="7|COMPLETED|0:0\n", stderr="")
        raise AssertionError(f"unexpected command {argv}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run = backend.run(cab, ["tool"], {})
    assert run.success
    assert "answer=99" in run.stdout
