import pytest

from shinobi.backends.container import ApptainerBackend, DockerBackend
from shinobi.exceptions import BackendError
from shinobi.loaders._modelgen import build_model
from shinobi.steps.schema import Cab, ParamMeta, ParamPattern

OUT = build_model("Out", {})


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
        DockerBackend(workdir="/work")._wrap(cab, ["tool"], {})


def test_docker_wrap_mounts_workdir_only_when_no_file_params():
    cab = make_cab({"threshold": ("float", False, None)})
    argv = DockerBackend(workdir="/work")._wrap(cab, ["tool", "--threshold", "1.0"], {"threshold": 1.0})
    assert argv == [
        "docker", "run", "--rm", "-v", "/work:/work", "-w", "/work", "tool:latest",
        "tool", "--threshold", "1.0",
    ]


def test_docker_wrap_mounts_file_param_parent_dir():
    cab = make_cab({"restored_image": ("File", False, None)})
    argv = DockerBackend(workdir="/work")._wrap(
        cab, ["tool", "--restored-image", "/data/in/img.fits"], {"restored_image": "/data/in/img.fits"}
    )
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work", "/data/in:/data/in"}


def test_docker_wrap_mounts_relative_file_param_under_workdir():
    cab = make_cab({"mask": ("File", False, None)})
    argv = DockerBackend(workdir="/work")._wrap(cab, ["tool", "--mask", "out/mask.fits"], {"mask": "out/mask.fits"})
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work", "/work/out:/work/out"}


def test_docker_wrap_mounts_pattern_matched_file_param():
    cab = Cab(
        name="quartical",
        command="quartical",
        image="tool:latest",
        inputs_model=build_model("QC_In", {}, allow_extra=True),
        outputs_model=OUT,
        input_patterns=[ParamPattern(attrs={"model_column": ParamMeta(dtype="File")})],
    )
    argv = DockerBackend(workdir="/work")._wrap(
        cab,
        ["quartical", "--K.model_column", "/data/model.fits"],
        {"K.model_column": "/data/model.fits"},
    )
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work", "/data:/data"}


def test_docker_wrap_dedupes_and_handles_list_of_files():
    cab = make_cab({"mslist": ("list:MS", False, None)})
    argv = DockerBackend(workdir="/work")._wrap(
        cab, ["tool", "--mslist", "a.ms,b.ms"], {"mslist": ["/data/a.ms", "/data/b.ms"]}
    )
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work", "/data:/data"}


def test_non_file_dtype_is_not_mounted():
    cab = make_cab({"name": ("str", False, None)})
    argv = DockerBackend(workdir="/work")._wrap(
        cab, ["tool", "--name", "/looks/like/a/path"], {"name": "/looks/like/a/path"}
    )
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work"}


def test_apptainer_uses_bind_and_exec():
    cab = make_cab({"restored_image": ("File", False, None)})
    argv = ApptainerBackend(workdir="/work")._wrap(
        cab, ["tool", "--restored-image", "/data/img.fits"], {"restored_image": "/data/img.fits"}
    )
    assert argv[0:2] == ["apptainer", "exec"]
    binds = {argv[i + 1] for i, a in enumerate(argv) if a == "--bind"}
    assert binds == {"/work:/work", "/data:/data"}
    pwd_index = argv.index("--pwd")
    assert argv[pwd_index + 1] == "/work"
    assert argv[pwd_index + 2] == "tool:latest"
