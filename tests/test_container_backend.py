import pytest

from shinobi.backends.container import ApptainerBackend, DockerBackend
from shinobi.exceptions import BackendError
from shinobi.schema import CabDef, ParamSchema


def make_cab(**inputs: ParamSchema) -> CabDef:
    return CabDef(name="tool", command="tool", image="tool:latest", inputs=inputs)


def test_no_image_raises_backend_error():
    cab = CabDef(name="tool", command="tool")  # no image
    backend = DockerBackend(workdir="/work")
    with pytest.raises(BackendError):
        backend._wrap(cab, ["tool"], {})


def test_docker_wrap_mounts_workdir_only_when_no_file_params():
    cab = make_cab(threshold=ParamSchema(dtype="float"))
    backend = DockerBackend(workdir="/work")
    argv = backend._wrap(cab, ["tool", "--threshold", "1.0"], {"threshold": 1.0})
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


def test_docker_wrap_mounts_file_param_parent_dir():
    cab = make_cab(restored_image=ParamSchema(dtype="File"))
    backend = DockerBackend(workdir="/work")
    argv = backend._wrap(
        cab, ["tool", "--restored-image", "/data/in/img.fits"], {"restored_image": "/data/in/img.fits"}
    )
    assert "-v" in argv
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work", "/data/in:/data/in"}


def test_docker_wrap_mounts_relative_file_param_under_workdir():
    cab = make_cab(mask=ParamSchema(dtype="File"))
    backend = DockerBackend(workdir="/work")
    argv = backend._wrap(cab, ["tool", "--mask", "out/mask.fits"], {"mask": "out/mask.fits"})
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work", "/work/out:/work/out"}


def test_docker_wrap_dedupes_and_handles_list_of_files():
    cab = make_cab(mslist=ParamSchema(dtype="list:MS"))
    backend = DockerBackend(workdir="/work")
    argv = backend._wrap(
        cab,
        ["tool", "--mslist", "a.ms,b.ms"],
        {"mslist": ["/data/a.ms", "/data/b.ms"]},
    )
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work", "/data:/data"}


def test_non_file_dtype_is_not_mounted():
    cab = make_cab(name=ParamSchema(dtype="str"))
    backend = DockerBackend(workdir="/work")
    argv = backend._wrap(cab, ["tool", "--name", "/looks/like/a/path"], {"name": "/looks/like/a/path"})
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work"}


def test_apptainer_uses_bind_and_exec():
    cab = make_cab(restored_image=ParamSchema(dtype="File"))
    backend = ApptainerBackend(workdir="/work")
    argv = backend._wrap(
        cab, ["tool", "--restored-image", "/data/img.fits"], {"restored_image": "/data/img.fits"}
    )
    assert argv[0:2] == ["apptainer", "exec"]
    binds = {argv[i + 1] for i, a in enumerate(argv) if a == "--bind"}
    assert binds == {"/work:/work", "/data:/data"}
    pwd_index = argv.index("--pwd")
    assert argv[pwd_index + 1] == "/work"
    assert argv[pwd_index + 2] == "tool:latest"
