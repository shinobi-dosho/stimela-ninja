import os

import pytest

from shinobi.backends import container
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


def test_rootless_podman_gets_home_but_not_user_flags():
    """A rootless podman already runs the container as the invoking user, so
    `--user` adds nothing and costs everything: it names a uid *inside* the
    user namespace, which on a bind-mounted host path is an unmapped subuid
    with no write access, so every output write fails with EACCES. Verified
    against real rootless podman 4.9.3.
    """
    from shinobi.backends.container import PodmanBackend

    cab = make_cab({"threshold": ("float", False, None)})
    argv, _ = PodmanBackend(workdir="/work", run_as_host_user=True)._wrap(cab, ["tool"], {"threshold": 1.0})
    assert "--user" not in argv
    assert "HOME=/work" in argv  # the intent still holds, only the flag doesn't apply


def test_rootful_podman_keeps_user_flags(monkeypatch):
    # Running podman as root is the daemon-shaped case docker is in: nothing
    # maps the container's root back to a human, so `--user` is needed.
    from shinobi.backends.container import PodmanBackend

    monkeypatch.setattr(os, "geteuid", lambda: 0)
    cab = make_cab({"threshold": ("float", False, None)})
    argv, _ = PodmanBackend(workdir="/work", run_as_host_user=True)._wrap(cab, ["tool"], {"threshold": 1.0})
    assert "--user" in argv
    assert argv[argv.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"


def test_docker_keeps_user_flags_regardless_of_invoking_uid():
    # docker's root daemon writes as root whatever this session is, so the
    # rootless reasoning must not leak across to it.
    cab = make_cab({"threshold": ("float", False, None)})
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=True)._wrap(cab, ["tool"], {"threshold": 1.0})
    assert "--user" in argv


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


def test_shared_parent_stays_writable_and_reasserts_the_read_only_input():
    # A read-only and a writable input resolving to the same parent: writable
    # still wins for the *directory* (an in-place MS in msdir must stay
    # writable), but the read-only input beside it is no longer collateral --
    # it is re-asserted `:ro` at its own path, nested inside. Verified live on
    # docker: the writable MS mutates, the read-only one refuses the write.
    cab = make_cab_with_paths(ro_fields=["raw"], rw_fields=["work_ms"])
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["tool"], {"raw": "/shared/a.ms", "work_ms": "/shared/b.ms"})
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert "/shared:/shared" in mounts
    assert "/shared:/shared:ro" not in mounts
    assert "/shared/a.ms:/shared/a.ms:ro" in mounts  # the read-only input itself
    assert mounts.index("/shared:/shared") < mounts.index("/shared/a.ms:/shared/a.ms:ro")


def test_read_only_input_alone_still_mounts_its_whole_parent_read_only():
    # No writable contributor -> nothing to nest inside; the directory itself
    # carries the classification, exactly as before.
    cab = make_cab_with_paths(ro_fields=["raw"])
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["tool"], {"raw": "/shared/a.ms"})
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert mounts == ["/work:/work", "/shared:/shared:ro"]


def test_writable_input_inside_a_read_only_directory_input_is_refused(tmp_path):
    # The reverse of the write-target contradiction, and the same verdict: a
    # cab cannot declare a directory untouchable and something writable inside
    # it. Before, the inner input silently mounted its parent read-write.
    store = tmp_path / "store"
    (store / "sub").mkdir(parents=True)
    cab = make_cab_with_paths(ro_fields=["store"], rw_fields=["inner"])
    with pytest.raises(BackendError) as exc:
        bind_dir_modes(cab, {"store": str(store), "inner": f"{store}/sub/x.ms"}, "/work")
    assert "'store'" in str(exc.value) and "'inner'" in str(exc.value)


def test_workdir_inside_a_read_only_directory_input_is_refused(tmp_path):
    # The working directory is always writable, so it collides the same way.
    store = tmp_path / "store"
    store.mkdir()
    cab = make_cab_with_paths(ro_fields=["store"])
    with pytest.raises(BackendError, match="working directory"):
        bind_dir_modes(cab, {"store": str(store)}, f"{store}/run")


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


# -- declared output directories ----------------------------------------------
#
# A tool's output stem is conventionally a *string*-typed input (wsclean's
# `prefix`), so it contributes no path field and no mount of its own. Point one
# outside the workdir and, without the output side being read too, the tool
# writes into the container: silently discarded on `docker run --rm`.


def make_prefix_cab(implicit="{prefix}-MFS-image.fits", harvest=(), outputs=None) -> Cab:
    """The wsclean shape: a string-typed output stem, and a path-typed output
    field whose `implicit` template is what actually declares where it lands."""
    return Cab(
        name="wsclean",
        command="wsclean",
        image="tool:latest",
        inputs_model=build_model("In", {"prefix": ("str", False, None), "ms": ("MS", False, None)}),
        outputs_model=build_model("Out", outputs if outputs is not None else {"restored_image": ("File", False, None)}),
        field_meta={"restored_image": ParamMeta(implicit=implicit)} if implicit else {},
        harvest=list(harvest),
    )


def test_string_typed_output_prefix_mounts_its_directory_read_write(tmp_path):
    outdir = tmp_path / "imaging"
    outdir.mkdir()
    cab = make_prefix_cab()
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["wsclean"], {"prefix": f"{outdir}/img"})
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work", f"{outdir}:{outdir}"}


def test_output_directory_that_does_not_exist_yet_mounts_nearest_existing_ancestor(tmp_path):
    # The tool creates its own output tree (`mkdir -p`); mounting the deepest
    # existing ancestor gives it exactly what a native run would have.
    cab = make_prefix_cab()
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["wsclean"], {"prefix": f"{tmp_path}/run1/deep/img"})
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work", f"{tmp_path}:{tmp_path}"}


def test_unmountable_output_directory_is_refused_by_name():
    cab = make_prefix_cab()
    with pytest.raises(BackendError) as exc:
        DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["wsclean"], {"prefix": "/no-such-root-a9f3/run1/img"})
    message = str(exc.value)
    assert "restored_image" in message  # names the declaration...
    assert "/no-such-root-a9f3/run1" in message  # ...and the directory


def test_apptainer_binds_declared_output_directory(tmp_path):
    cab = make_prefix_cab()
    argv, _ = ApptainerBackend(workdir="/work")._wrap(cab, ["wsclean"], {"prefix": f"{tmp_path}/img"})
    binds = {argv[i + 1] for i, a in enumerate(argv) if a == "--bind"}
    assert binds == {"/work:/work", f"{tmp_path}:{tmp_path}"}


def test_harvest_pattern_directory_is_mounted(tmp_path):
    # No path-typed output field at all -- `harvest` is the whole declaration.
    cab = make_prefix_cab(implicit=None, outputs={}, harvest=["{prefix}-*.fits"])
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["wsclean"], {"prefix": f"{tmp_path}/img"})
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work", f"{tmp_path}:{tmp_path}"}


def test_relative_output_prefix_adds_no_mount():
    # It lands under the workdir, which is always mounted; and the sandbox
    # relies on exactly this staying relative.
    cab = make_prefix_cab()
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["wsclean"], {"prefix": "img/run1"})
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work"}


def test_output_under_an_input_directory_adds_no_second_mount(tmp_path):
    ms = tmp_path / "obs.ms"
    cab = make_prefix_cab()
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["wsclean"], {"ms": str(ms), "prefix": f"{tmp_path}/img"})
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work", f"{tmp_path}:{tmp_path}"}  # the input's mount already covers it


def make_readonly_input_cab() -> Cab:
    """A cab whose write target and whose `writable: false` input can be aimed
    at the same directory -- the collision the nesting rule exists for."""
    return Cab(
        name="wsclean",
        command="wsclean",
        image="tool:latest",
        inputs_model=create_model(
            "In",
            prefix=(Optional[str], None),
            ms=(Optional[Path], Field(None, json_schema_extra={"writable": False})),
        ),
        outputs_model=build_model("Out", {"restored_image": ("File", False, None)}),
        field_meta={"restored_image": ParamMeta(implicit="{prefix}-MFS-image.fits")},
    )


def test_declared_output_reasserts_a_read_only_input_nested_inside_the_directory_it_upgrades(tmp_path):
    # Both declarations hold: the directory goes read-write so the tool can
    # write its product, and the `writable: false` input is re-asserted `:ro`
    # at its own path inside it. Verified against real docker, podman and apptainer.
    cab = make_readonly_input_cab()
    mounts = bind_dir_modes(cab, {"ms": f"{tmp_path}/obs.ms", "prefix": f"{tmp_path}/img"}, "/work")
    assert mounts == [("/work", True), (str(tmp_path), True), (f"{tmp_path}/obs.ms", False)]
    # the nested `:ro` entry must follow the directory it nests in, so a
    # runtime that honours emission order rather than depth still gets it right
    assert [m[0] for m in mounts].index(str(tmp_path)) < [m[0] for m in mounts].index(f"{tmp_path}/obs.ms")


def test_docker_emits_the_nested_read_only_input_as_a_mount(tmp_path):
    cab = make_readonly_input_cab()
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["wsclean"], {"ms": f"{tmp_path}/obs.ms", "prefix": f"{tmp_path}/img"})
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert mounts == ["/work:/work", f"{tmp_path}:{tmp_path}", f"{tmp_path}/obs.ms:{tmp_path}/obs.ms:ro"]


def test_apptainer_binds_the_nested_read_only_input(tmp_path):
    cab = make_readonly_input_cab()
    argv, _ = ApptainerBackend(workdir="/work")._wrap(cab, ["wsclean"], {"ms": f"{tmp_path}/obs.ms", "prefix": f"{tmp_path}/img"})
    binds = [argv[i + 1] for i, a in enumerate(argv) if a == "--bind"]
    assert binds == ["/work:/work", f"{tmp_path}:{tmp_path}", f"{tmp_path}/obs.ms:{tmp_path}/obs.ms:ro"]


def test_no_nested_mount_when_nothing_was_marked_read_only(tmp_path):
    # The upgrade path only re-asserts what a `writable: false` earned; an
    # ordinary writable input in the same directory adds no nested entry.
    cab = make_cab({"prefix": ("str", False, None), "ms": ("MS", False, None)})
    cab = cab.model_copy(
        update={
            "outputs_model": build_model("Out", {"restored_image": ("File", False, None)}),
            "field_meta": {"restored_image": ParamMeta(implicit="{prefix}-MFS-image.fits")},
        }
    )
    mounts = bind_dir_modes(cab, {"ms": f"{tmp_path}/obs.ms", "prefix": f"{tmp_path}/img"}, "/work")
    assert mounts == [("/work", True), (str(tmp_path), True)]


def test_write_target_under_a_read_only_directory_keeps_the_parent_read_only(tmp_path):
    # The other nesting shape: read-write target inside a read-only parent.
    # The parent keeps `:ro` for the input; the target is mounted inside it.
    (tmp_path / "products").mkdir()
    cab = make_readonly_input_cab()
    mounts = bind_dir_modes(cab, {"ms": f"{tmp_path}/obs.ms", "prefix": f"{tmp_path}/products/img"}, "/work")
    assert mounts == [("/work", True), (str(tmp_path), False), (f"{tmp_path}/products", True)]


def test_write_target_inside_a_read_only_input_is_refused(tmp_path):
    # Nesting resolves a target *beside* a read-only input; it cannot resolve
    # one *inside* it. "Never write this" and "put a product here" name the
    # same tree, so the declarations contradict and the run is refused on
    # every backend -- rather than mounting the read-only input read-write,
    # which is what happened before: `readonly_paths` is keyed by parent, so a
    # fresh mount key that IS a read-only input path recorded no upgrade.
    store = tmp_path / "store"
    store.mkdir()
    cab = Cab(
        name="tool",
        command="tool",
        image="tool:latest",
        inputs_model=create_model(
            "In",
            prefix=(Optional[str], None),
            store=(Optional[Path], Field(None, json_schema_extra={"writable": False})),
        ),
        outputs_model=build_model("Out", {"img": ("File", False, None)}),
        field_meta={"img": ParamMeta(implicit="{prefix}-image.fits")},
    )
    with pytest.raises(BackendError) as exc:
        bind_dir_modes(cab, {"store": str(store), "prefix": f"{store}/out"}, "/work")
    assert "'store'" in str(exc.value)  # names the input, not just the path
    assert "writable: false" in str(exc.value)


def test_write_target_deep_inside_a_read_only_input_is_refused(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    cab = Cab(
        name="tool",
        command="tool",
        image="tool:latest",
        inputs_model=create_model(
            "In",
            prefix=(Optional[str], None),
            store=(Optional[Path], Field(None, json_schema_extra={"writable": False})),
        ),
        outputs_model=build_model("Out", {"img": ("File", False, None)}),
        field_meta={"img": ParamMeta(implicit="{prefix}-image.fits")},
    )
    with pytest.raises(BackendError):
        bind_dir_modes(cab, {"store": str(store), "prefix": f"{store}/a/b/out"}, "/work")


def test_refusal_holds_even_when_the_directory_is_writable_for_other_reasons(tmp_path):
    # The contradiction is in the schema, not in how the mount table came out:
    # a writable sibling input making the parent read-write must not let a
    # target inside the read-only input through.
    store = tmp_path / "store"
    store.mkdir()
    cab = Cab(
        name="tool",
        command="tool",
        image="tool:latest",
        inputs_model=create_model(
            "In",
            prefix=(Optional[str], None),
            store=(Optional[Path], Field(None, json_schema_extra={"writable": False})),
            scratch=(Optional[Path], Field(None, json_schema_extra={"writable": True})),
        ),
        outputs_model=build_model("Out", {"img": ("File", False, None)}),
        field_meta={"img": ParamMeta(implicit="{prefix}-image.fits")},
    )
    inputs = {"store": str(store), "scratch": f"{tmp_path}/scratch.ms", "prefix": f"{store}/out"}
    with pytest.raises(BackendError):
        bind_dir_modes(cab, inputs, "/work")


def test_scratch_directory_is_mounted_like_a_product(tmp_path):
    """A `scratch` declaration exists to get a cache/log directory mounted
    without it being harvested. The mount half is indistinguishable from a
    product's -- the tool has to be able to write there either way.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    cab = Cab(
        name="ddf",
        command="ddf",
        image="tool:latest",
        inputs_model=build_model("In", {"cache_dir": ("str", False, None)}),
        outputs_model=OUT,
        scratch=["{cache_dir}/*"],
    )
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["ddf"], {"cache_dir": str(cache)})
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work", f"{cache}:{cache}"}


def test_unset_scratch_target_adds_no_mount():
    # An optional cache directory left unset declares nothing -- and must not
    # resolve to a literal "None" path.
    cab = Cab(
        name="ddf",
        command="ddf",
        image="tool:latest",
        inputs_model=build_model("In", {"cache_dir": ("str", False, None)}),
        outputs_model=OUT,
        scratch=["{cache_dir}/*"],
    )
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["ddf"], {"cache_dir": None})
    mounts = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    assert mounts == {"/work:/work"}


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


# -- partial cgroup delegation (issue #39) --


@pytest.fixture
def delegate(monkeypatch):
    """Pretend this session was delegated exactly `controllers`."""

    def _set(*controllers):
        monkeypatch.setattr(container, "delegated_controllers", lambda *a, **k: frozenset(controllers))
        container._warned_drops.clear()

    return _set


def test_undelegated_dimension_is_dropped_not_the_whole_declaration(delegate, caplog):
    """The bug: one unenforceable dimension took the other down with it, and
    the run with it. Memory is delegated here, so memory is still enforced.
    """
    delegate("memory", "pids")
    cab = make_cab().model_copy(update={"resources": Resources(cpus=2, memory="256MiB")})
    with caplog.at_level("WARNING"):
        argv, _ = ApptainerBackend(workdir="/work")._wrap(cab, ["tool"], {})
    assert "--cpus" not in argv
    assert "--memory" in argv and argv[argv.index("--memory") + 1] == str(256 * 1024**2)
    # dropping enforcement silently would be the other half of the bug
    assert "--cpus" in caplog.text and "memory, pids" in caplog.text


def test_drop_is_warned_once_per_run(delegate, caplog):
    delegate("memory")
    cab = make_cab().model_copy(update={"resources": Resources(cpus=2, memory="256MiB")})
    with caplog.at_level("WARNING"):
        for _ in range(3):
            ApptainerBackend(workdir="/work")._wrap(cab, ["tool"], {})
    assert caplog.text.count("dropping --cpus") == 1


def test_docker_ignores_session_delegation(delegate):
    """Docker's limits are applied by a root daemon in its own cgroup tree,
    so what systemd delegated to *this* session is not evidence about them.
    """
    delegate("memory")
    cab = make_cab().model_copy(update={"resources": Resources(cpus=2, memory="256MiB")})
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["tool"], {})
    assert "--cpus" in argv


def test_unknown_delegation_emits_everything(monkeypatch):
    """`None` means "couldn't tell" -- which must stay the loud old behaviour,
    never a silent drop.
    """
    monkeypatch.setattr(container, "delegated_controllers", lambda *a, **k: None)
    cab = make_cab().model_copy(update={"resources": Resources(cpus=2, memory="256MiB")})
    argv, _ = ApptainerBackend(workdir="/work")._wrap(cab, ["tool"], {})
    assert "--cpus" in argv and "--memory" in argv


def test_enforce_always_emits_undelegated_limits(delegate, monkeypatch):
    delegate("memory")
    monkeypatch.setenv("SHINOBI_EXECUTION__ENFORCE_RESOURCES", "always")
    cab = make_cab().model_copy(update={"resources": Resources(cpus=2, memory="256MiB")})
    argv, _ = ApptainerBackend(workdir="/work")._wrap(cab, ["tool"], {})
    assert "--cpus" in argv and "--memory" in argv


def test_enforce_never_emits_nothing(monkeypatch):
    monkeypatch.setenv("SHINOBI_EXECUTION__ENFORCE_RESOURCES", "never")
    cab = make_cab().model_copy(update={"resources": Resources(cpus=4, memory="8GiB")})
    argv, _ = DockerBackend(workdir="/work", run_as_host_user=False)._wrap(cab, ["tool"], {})
    assert "--cpus" not in argv and "--memory" not in argv


def test_remote_build_never_probes_this_host(delegate):
    """A Slurm job script is compiled here and run on a compute node, whose
    delegation this host knows nothing about.
    """
    delegate("memory")
    cab = make_cab().model_copy(update={"resources": Resources(cpus=2, memory="256MiB")})
    argv, _ = container.build_container_argv("apptainer", cab, ["tool"], {}, "/work", runs_here=False)
    assert "--cpus" in argv


@pytest.mark.parametrize(
    "stderr",
    [
        # apptainer 1.3.0, verbatim from issue #39
        (
            "FATAL:   container creation failed: while applying cgroups config: while setting cgroup limits: openat2 "
            "/sys/fs/cgroup/user.slice/user-20001.slice/user@20001.service/user.slice/apptainer-3628201.scope/cpu.max: "
            "no such file or directory"
        ),
        # crun/podman's spelling of the same wall
        "Error: OCI runtime error: crun: the requested cgroup controller `cpu` is not available",
    ],
)
def test_cgroup_failure_hint_names_cause_and_remedy(stderr, delegate):
    delegate("memory", "pids")
    hint = container.cgroup_failure_hint(stderr, "apptainer")
    assert hint is not None
    assert "`cpu` cgroup controller is not delegated" in hint
    assert "memory, pids" in hint  # what you *do* have
    assert "Delegate=cpu" in hint  # and how to get the rest


def test_cgroup_failure_hint_stays_out_of_unrelated_failures():
    assert container.cgroup_failure_hint("Segmentation fault (core dumped)", "apptainer") is None
    assert container.cgroup_failure_hint("wsclean: error: no such file or directory", "apptainer") is None
