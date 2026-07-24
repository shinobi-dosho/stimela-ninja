import os

import pytest

from shinobi.backends.container import ApptainerBackend, DockerBackend
from shinobi.exceptions import BackendError
from shinobi.loaders import build_model
from shinobi.resources import Resources
from shinobi.steps.schema import Cab, ParamMeta, ParamPattern, ParamSegment

OUT = build_model("Out", {})


@pytest.fixture(autouse=True)
def _no_registry_digest(monkeypatch):
    # These are pure argv-construction tests -- never shell out to skopeo, so
    # they stay hermetic and fast and assert the (unpinned) reference form.
    monkeypatch.setattr("shinobi.backends.container._registry_digest", lambda ref: None)


def make_cab(fields=None, image="tool:latest") -> Cab:
    return Cab(
        name="tool",
        command="tool",
        image=image,
        inputs_model=build_model("In", fields or {}),
        outputs_model=OUT,
    )


def test_no_image_raises_backend_error():
    cab = make_cab(image=None)
    with pytest.raises(BackendError):
        DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["tool"], {})


def test_docker_wrap_mounts_workdir_only_when_no_file_params():
    cab = make_cab({"threshold": ("float", False, None)})
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["tool", "--threshold", "1.0"], {"threshold": 1.0})
    assert argv == [
        "docker",
        "run",
        "--rm",
        "-v",
        "/work:/work",
        "-w",
        "/work",
        "tool:latest",
        "tool",
        "--threshold",
        "1.0",
    ]


def test_docker_wrap_user_flags_default_on():
    cab = make_cab({"threshold": ("float", False, None)})
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=True)._wrap(cab, ["tool", "--threshold", "1.0"], {"threshold": 1.0})
    assert argv == [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/work",
        "-v",
        "/work:/work",
        "-w",
        "/work",
        "tool:latest",
        "tool",
        "--threshold",
        "1.0",
    ]


def test_docker_wrap_user_flags_can_be_disabled():
    cab = make_cab({"threshold": ("float", False, None)})
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["tool", "--threshold", "1.0"], {"threshold": 1.0})
    assert "--user" not in argv
    assert "HOME=/work" not in argv


def test_apptainer_ignores_run_as_host_user():
    cab = make_cab({"restored_image": ("File", False, None)})
    argv, _ = ApptainerBackend(workdir="/work", run_as_host_user=True)._wrap(cab, ["tool", "--restored-image", "/data/img.fits"], {"restored_image": "/data/img.fits"})
    assert "--user" not in argv


def test_docker_wrap_mounts_file_param_parent_dir():
    cab = make_cab({"restored_image": ("File", False, None)})
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["tool", "--restored-image", "/data/in/img.fits"], {"restored_image": "/data/in/img.fits"})
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work", "/data/in:/data/in"}


def test_docker_wrap_mounts_relative_file_param_under_workdir():
    cab = make_cab({"mask": ("File", False, None)})
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["tool", "--mask", "out/mask.fits"], {"mask": "out/mask.fits"})
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work", "/work/out:/work/out"}


def test_docker_wrap_mounts_pattern_matched_file_param():
    cab = Cab(
        name="quartical",
        command="quartical",
        image="tool:latest",
        inputs_model=build_model("QC_In", {}, allow_extra=True),
        outputs_model=OUT,
        input_patterns=[ParamPattern(segments=[ParamSegment(regex=r".+?"), ParamSegment(attrs={"model_column": ParamMeta(dtype="File")})])],
    )
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(
        cab,
        ["quartical", "--K.model_column", "/data/model.fits"],
        {"K.model_column": "/data/model.fits"},
    )
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work", "/data:/data"}


def test_docker_wrap_dedupes_and_handles_list_of_files():
    cab = make_cab({"mslist": ("list:MS", False, None)})
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["tool", "--mslist", "a.ms,b.ms"], {"mslist": ["/data/a.ms", "/data/b.ms"]})
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work", "/data:/data"}


def test_non_file_dtype_is_not_mounted():
    cab = make_cab({"name": ("str", False, None)})
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["tool", "--name", "/looks/like/a/path"], {"name": "/looks/like/a/path"})
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work"}


def test_docker_backend_defaults_run_as_host_user_from_config(tmp_path, monkeypatch):
    monkeypatch.delenv("SHINOBI_BACKEND__RUN_AS_HOST_USER", raising=False)
    from shinobi.config import AppConfig

    monkeypatch.setattr(AppConfig, "_config_file", tmp_path / "missing.yml")
    cab = make_cab({"threshold": ("float", False, None)})
    argv, _ = DockerBackend(workdir="/work")._wrap(cab, ["tool", "--threshold", "1.0"], {"threshold": 1.0})
    assert "--user" in argv


def test_apptainer_uses_bind_and_exec():
    cab = make_cab({"restored_image": ("File", False, None)})
    argv, _ = ApptainerBackend(workdir="/work")._wrap(cab, ["tool", "--restored-image", "/data/img.fits"], {"restored_image": "/data/img.fits"})
    assert argv[0:2] == ["apptainer", "exec"]
    binds = {argv[i + 1] for i, a in enumerate(argv) if a == "--bind"}
    assert binds == {"/work:/work", "/data:/data"}
    pwd_index = argv.index("--pwd")
    assert argv[pwd_index + 1] == "/work"
    # apptainer needs an explicit source scheme for a registry ref
    assert argv[pwd_index + 2] == "docker://tool:latest"


# ---- read-only bind mounts (writable: false) --------------------------------

from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

from pydantic import Field, create_model  # noqa: E402

from shinobi.backends.container import bind_dir_modes  # noqa: E402


def make_cab_with_paths(ro_fields=(), rw_fields=(), image="tool:latest") -> Cab:
    """A Cab whose inputs_model has Path fields, some marked writable: false
    (as the YAML loader would via json_schema_extra)."""
    defs: dict = {}
    for f in ro_fields:
        defs[f] = (Optional[Path], Field(None, json_schema_extra={"writable": False}))
    for f in rw_fields:
        defs[f] = (Optional[Path], Field(None, json_schema_extra={"writable": True}))
    return Cab(name="tool", command="tool", image=image, inputs_model=create_model("In", **defs), outputs_model=OUT)


def test_docker_mounts_writable_false_directory_read_only():
    cab = make_cab_with_paths(ro_fields=["raw_ms"])
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["tool", "--raw-ms", "/rawdata/obs.ms"], {"raw_ms": "/rawdata/obs.ms"})
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work", "/rawdata:/rawdata:ro"}  # workdir stays writable


def test_apptainer_mounts_writable_false_directory_read_only():
    cab = make_cab_with_paths(ro_fields=["raw_ms"])
    argv, _ = ApptainerBackend(workdir="/work")._wrap(cab, ["tool", "--raw-ms", "/rawdata/obs.ms"], {"raw_ms": "/rawdata/obs.ms"})
    binds = {argv[i + 1] for i, a in enumerate(argv) if a == "--bind"}
    assert binds == {"/work:/work", "/rawdata:/rawdata:ro"}


def test_shared_parent_stays_writable_when_any_contributor_is_writable():
    # a read-only and a writable input resolving to the same parent -> writable
    # wins (an in-place MS in msdir must stay writable).
    cab = make_cab_with_paths(ro_fields=["raw"], rw_fields=["work_ms"])
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["tool"], {"raw": "/shared/a.ms", "work_ms": "/shared/b.ms"})
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert "/shared:/shared" in mounts
    assert "/shared:/shared:ro" not in mounts


def test_unmarked_path_field_mounts_read_write():
    # no writable marker -> writable (the default; preserves prior behaviour).
    cab = make_cab({"restored_image": ("File", False, None)})
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["tool", "--restored-image", "/data/img.fits"], {"restored_image": "/data/img.fits"})
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work", "/data:/data"}


def test_bind_dir_modes_classifies_read_only_and_workdir():
    cab = make_cab_with_paths(ro_fields=["raw_ms"], rw_fields=["out_ms"])
    modes = dict(bind_dir_modes(cab, {"raw_ms": "/rawdata/obs.ms", "out_ms": "/msdir/obs.ms"}, "/work"))
    assert modes == {"/work": True, "/rawdata": False, "/msdir": True}


def test_apptainer_image_uri_scheme_handling():
    from shinobi.backends.container import _apptainer_image_uri

    # bare registry refs get a docker:// source so apptainer pulls them
    assert _apptainer_image_uri("quay.io/stimela2/casa6:6.7") == "docker://quay.io/stimela2/casa6:6.7"
    assert _apptainer_image_uri("tool:latest") == "docker://tool:latest"
    # already-schemed or local images are left untouched
    assert _apptainer_image_uri("docker://quay.io/x:1") == "docker://quay.io/x:1"
    assert _apptainer_image_uri("library://x/y:1") == "library://x/y:1"
    assert _apptainer_image_uri("/images/casa6.sif") == "/images/casa6.sif"
    assert _apptainer_image_uri("./casa6.sif") == "./casa6.sif"


# -- declared resource limits --


def test_docker_wrap_emits_declared_limits():
    cab = make_cab()
    cab = cab.model_copy(update={"resources": Resources(cpus=4, memory="8GiB")})
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["tool"], {})
    assert "--cpus" in argv and argv[argv.index("--cpus") + 1] == "4"
    assert "--memory" in argv and argv[argv.index("--memory") + 1] == str(8 * 1024**3)
    # the limits must precede the image reference, not land in the command
    assert argv.index("--memory") < argv.index("tool:latest")


def test_apptainer_wrap_emits_declared_limits():
    """Apptainer really enforces these -- `--memory 256M --cpus 2` produces a
    cgroup scope with memory.max=268435456 -- so they are emitted rather than
    dropped as unsupported.
    """
    cab = make_cab().model_copy(update={"resources": Resources(cpus=2, memory="256MiB")})
    argv, _ = ApptainerBackend(workdir="/work")._wrap(cab, ["tool"], {})
    assert argv[:5] == ["apptainer", "exec", "--cpus", "2", "--memory"]
    assert argv[5] == str(256 * 1024**2)


def test_partial_declaration_emits_only_what_was_declared():
    cab = make_cab().model_copy(update={"resources": Resources(memory="1GiB")})
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["tool"], {})
    assert "--memory" in argv
    assert "--cpus" not in argv


def test_undeclared_resources_change_nothing():
    cab = make_cab()
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["tool"], {})
    assert "--cpus" not in argv
    assert "--memory" not in argv
