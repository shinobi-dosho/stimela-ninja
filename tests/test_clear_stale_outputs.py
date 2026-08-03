"""Re-running a step replaces the previous run's products
(`sandbox.clear_stale_outputs`).

A sandboxed *relative* output already got that from harvest -- fresh scratch
dir, tool writes, `_move` replaces the destination. An output the tool writes
straight to its destination (an absolute path, a path-typed input anchored
into one, anything at all when unsandboxed) got nothing: the tool started with
the last run's product still sitting there, which CASA-family tools refuse to
overwrite and appending tools silently corrupt.

The carve-out is the interesting half. `flagdata(vis=...) -> vis` and
`mstransform(outputvis=...) -> outputvis` are the *same* declaration shape --
one name on both models -- and only `ParamMeta.write_path` says which is the
caller's data and which is this step's product. Unmarked means data, so a cab
that says nothing keeps its inputs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

import shinobi
from shinobi.loaders import build_model
from shinobi.sandbox import clear_stale_outputs
from shinobi.steps.schema import Cab, ParamMeta, Scope

# ------------------------------------------------------------------ fixtures


def _scope(inputs=None, outputs=None, field_meta=None, harvest=None) -> Scope:
    return Scope(
        name="s",
        inputs_model=build_model("In", inputs or {}),
        outputs_model=build_model("Out", outputs or {}),
        field_meta=field_meta or {},
        harvest=harvest or [],
    )


def _ms(path: Path) -> Path:
    """A directory-shaped product, which is what an MS or an image tree is --
    clearing one is an `rmtree`, not a file overwrite."""
    path.mkdir(parents=True)
    (path / "table.dat").write_text("rows")
    return path


# --------------------------------------------------------- what gets cleared


def test_absolute_declared_output_is_cleared(tmp_path):
    stale = _ms(tmp_path / "out.ms")
    scope = _scope(outputs={"vis": ("MS", False, None)})

    removed = clear_stale_outputs(scope, {"vis": stale}, tmp_path, sandboxed=True)

    assert removed == [stale]
    assert not stale.exists()


def test_a_declared_output_that_does_not_exist_yet_is_a_no_op(tmp_path):
    scope = _scope(outputs={"vis": ("MS", False, None)})
    assert clear_stale_outputs(scope, {"vis": tmp_path / "out.ms"}, tmp_path, sandboxed=True) == []


def test_implicit_template_output_is_cleared(tmp_path):
    """The wsclean/tclean shape: a string-typed stem, products named by an
    `implicit` template off it. No input names the product, so nothing has to
    be declared for this to be safe."""
    (tmp_path / "img-MFS-image.fits").write_text("old")
    scope = _scope(
        inputs={"prefix": ("str", True, None)},
        outputs={"image": ("File", False, None)},
        field_meta={"image": ParamMeta(implicit="{prefix}-MFS-image.fits")},
    )

    removed = clear_stale_outputs(scope, {"prefix": str(tmp_path / "img")}, tmp_path, sandboxed=True)

    assert removed == [tmp_path / "img-MFS-image.fits"]


def test_every_element_of_a_list_valued_output_is_cleared(tmp_path):
    first, second = _ms(tmp_path / "a.ms"), _ms(tmp_path / "b.ms")
    scope = _scope(outputs={"vis": ("List[MS]", False, None)})

    clear_stale_outputs(scope, {"vis": [first, second]}, tmp_path, sandboxed=True)

    assert not first.exists() and not second.exists()


def test_relative_output_is_cleared_unsandboxed_and_left_for_harvest_when_sandboxed(tmp_path):
    """Unsandboxed the tool writes into the workspace, so the stale product is
    in its way. Sandboxed it writes into the fresh scratch dir and harvest
    replaces the destination afterwards -- clearing early would only widen the
    window where a failed run leaves the caller with neither product."""
    (tmp_path / "out.dat").write_text("old")
    scope = _scope(outputs={"result": ("File", False, "out.dat")})

    assert clear_stale_outputs(scope, {}, tmp_path, sandboxed=True) == []
    assert (tmp_path / "out.dat").exists()

    assert clear_stale_outputs(scope, {}, tmp_path, sandboxed=False) == [tmp_path / "out.dat"]
    assert not (tmp_path / "out.dat").exists()


def test_harvest_glob_matches_are_never_cleared(tmp_path):
    """A glob's matches are named by the tool at run time, so they can collide
    with workspace data this step knows nothing about -- the same reason
    `_move` refuses to rmtree an undeclared directory."""
    (tmp_path / "img-0000.fits").write_text("old")
    scope = _scope(harvest=["img-*.fits"])

    assert clear_stale_outputs(scope, {}, tmp_path, sandboxed=False) == []
    assert (tmp_path / "img-0000.fits").exists()


# ------------------------------------------------- what is protected from it


def test_an_output_echoing_an_unmarked_input_is_the_callers_data(tmp_path):
    """`flagdata(vis=...) -> vis`: the MS is rewritten in place, and that
    "output" is the caller's data. Deleting it destroys the pipeline's input."""
    vis = _ms(tmp_path / "obs.ms")
    scope = _scope(inputs={"vis": ("MS", True, None)}, outputs={"vis": ("MS", False, None)})

    assert clear_stale_outputs(scope, {"vis": vis}, tmp_path, sandboxed=False) == []
    assert (vis / "table.dat").exists()


def test_the_same_shape_is_cleared_once_declared_a_write_target(tmp_path):
    """`mstransform(outputvis=...) -> outputvis` declares exactly what
    `flagdata` above declares. `write_path` is the whole difference."""
    outputvis = _ms(tmp_path / "obs_mst.ms")
    scope = _scope(
        inputs={"vis": ("MS", True, None), "outputvis": ("MS", True, None)},
        outputs={"outputvis": ("MS", False, None)},
        field_meta={"outputvis": ParamMeta(write_path=True)},
    )

    removed = clear_stale_outputs(scope, {"vis": _ms(tmp_path / "obs.ms"), "outputvis": outputvis}, tmp_path, sandboxed=False)

    assert removed == [outputvis]
    assert not outputvis.exists()
    assert (tmp_path / "obs.ms" / "table.dat").exists()  # the real input is untouched


def test_an_output_inside_a_declared_input_is_that_inputs_data(tmp_path):
    """An MS *is* a directory, so a declared output that resolves inside one
    is part of the caller's data even though the two paths compare unequal."""
    vis = _ms(tmp_path / "obs.ms")
    (vis / "CORRECTED").mkdir()
    scope = _scope(inputs={"vis": ("MS", True, None)}, outputs={"col": ("Directory", False, None)})

    assert clear_stale_outputs(scope, {"vis": vis, "col": vis / "CORRECTED"}, tmp_path, sandboxed=False) == []
    assert (vis / "CORRECTED").exists()


def test_an_output_containing_a_declared_input_is_not_cleared(tmp_path):
    """The other direction, and the more expensive one: clearing a declared
    output directory that happens to *hold* an input would take the input
    with it."""
    msdir = tmp_path / "msdir"
    vis = _ms(msdir / "obs.ms")
    scope = _scope(inputs={"vis": ("MS", True, None)}, outputs={"outdir": ("Directory", False, None)})

    assert clear_stale_outputs(scope, {"vis": vis, "outdir": msdir}, tmp_path, sandboxed=False) == []
    assert (vis / "table.dat").exists()


def test_an_output_resolving_onto_the_workspace_is_refused_loudly(tmp_path):
    """A mis-templated output (an empty stem, a stray `..`) that resolves to
    the workspace itself would take the whole run's inputs with it."""
    scope = _scope(outputs={"out": ("Directory", False, None)})

    with pytest.warns(UserWarning, match="contains the workspace"):
        assert clear_stale_outputs(scope, {"out": tmp_path}, tmp_path, sandboxed=False) == []
    assert tmp_path.exists()


def test_a_symlinked_destination_loses_the_link_not_the_target(tmp_path):
    target = _ms(tmp_path / "real.ms")
    link = tmp_path / "out.ms"
    link.symlink_to(target)
    scope = _scope(outputs={"vis": ("MS", False, None)})

    clear_stale_outputs(scope, {"vis": link}, tmp_path, sandboxed=False)

    assert not link.exists() and not link.is_symlink()
    assert (target / "table.dat").exists()


# --------------------------------------------------------- schema plumbing


def test_write_path_accepts_a_same_named_path_output_as_its_declaration():
    """The complete-destination shape declares its write target by *being* an
    output field, with no `implicit` template to name it."""
    Cab(
        name="mstransform",
        command="mstransform",
        inputs_model=build_model("In", {"outputvis": ("MS", True, None)}),
        outputs_model=build_model("Out", {"outputvis": ("MS", False, None)}),
        field_meta={"outputvis": ParamMeta(write_path=True)},
    )


def test_write_path_on_an_input_nothing_declares_is_still_rejected():
    with pytest.raises(ValueError, match="marked write_path but named by no write declaration"):
        Cab(
            name="c",
            command="t",
            inputs_model=build_model("In", {"prefix": ("str", True, None)}),
            outputs_model=build_model("Out", {}),
            field_meta={"prefix": ParamMeta(write_path=True)},
        )


# ------------------------------------------------------------- end to end


class _Out(BaseModel):
    outputvis: Path


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _refuses_to_overwrite(workspace: Path) -> Path:
    """A tool of the CASA shape: it checks its output for existence first and
    fails rather than clobbering."""
    script = workspace / "tool.sh"
    script.write_text('#!/bin/sh\n[ -e "$2" ] && { echo "already exists" >&2; exit 1; }\nmkdir -p "$2"\n')
    script.chmod(0o755)
    return script


def _mstransform_cab(script: Path, **kwargs) -> Cab:
    return Cab(
        name="mstransform",
        command=str(script),
        inputs_model=build_model("In", {"outputvis": ("MS", True, None)}),
        outputs_model=build_model("Out", {"outputvis": ("MS", False, None)}),
        field_meta={"outputvis": ParamMeta(write_path=True)},
        **kwargs,
    )


@pytest.mark.parametrize("sandbox", [False, True])
def test_a_refusing_tool_can_be_rerun_over_its_own_absolute_output(workspace, sandbox):
    cab = _mstransform_cab(_refuses_to_overwrite(workspace), sandbox=sandbox)
    outputvis = workspace / "msdir" / "obs_mst.ms"

    assert cab(backend="native", outputvis=outputvis).success
    assert outputvis.is_dir()
    assert cab(backend="native", outputvis=outputvis).success, "the re-run saw the previous product"


def test_a_relative_destination_input_is_cleared_even_under_a_sandbox(workspace):
    """A path-typed input is anchored at the workspace by
    `absolutize_path_inputs`, so the tool writes to the real destination and
    the sandbox never sees the product -- which is exactly the case harvest
    cannot cover, however the caller spelled the path."""
    cab = _mstransform_cab(_refuses_to_overwrite(workspace), sandbox=True)

    assert cab(backend="native", outputvis=Path("obs_mst.ms")).success
    assert (workspace / "obs_mst.ms").is_dir()
    assert cab(backend="native", outputvis=Path("obs_mst.ms")).success


def test_a_failed_rerun_does_not_get_the_previous_product_back(workspace):
    """The cost of clearing before the tool starts, stated: a re-run that
    then fails leaves neither product. Unavoidable -- a tool that refuses to
    overwrite has already failed by the time anything could harvest -- and the
    same thing a per-cab `overwrite` flag would do. Tier 1 snapshots restore
    it when they are on (`shinobi.snapshots`); this test pins the plain
    behaviour so it is never a surprise."""
    script = workspace / "tool.sh"
    script.write_text("#!/bin/sh\nexit 7\n")
    script.chmod(0o755)
    cab = _mstransform_cab(script)
    outputvis = _ms(workspace / "obs_mst.ms")

    assert not cab(backend="native", outputvis=outputvis).success
    assert not outputvis.exists()


def test_the_config_switch_turns_the_whole_thing_off(workspace, monkeypatch):
    monkeypatch.setenv("SHINOBI_EXECUTION__CLEAR_STALE_OUTPUTS", "false")
    cab = _mstransform_cab(_refuses_to_overwrite(workspace))
    outputvis = workspace / "obs_mst.ms"

    assert cab(backend="native", outputvis=outputvis).success
    assert not cab(backend="native", outputvis=outputvis).success


def test_pystep_write_paths_clears_the_stale_product(workspace):
    @shinobi.pystep(write_paths=["outputvis"])
    def mstransform(outputvis: Path) -> _Out:
        if outputvis.exists():
            raise AssertionError("already exists")
        outputvis.mkdir(parents=True)
        return _Out(outputvis=outputvis)

    outputvis = workspace / "obs_mst.ms"
    assert mstransform(outputvis=outputvis).success
    assert mstransform(outputvis=outputvis).success


def test_pystep_without_write_paths_keeps_the_callers_data(workspace):
    """The in-place-mutation pystep: same signature shape, and the MS it is
    handed must still be there when the function runs."""
    seen = []

    @shinobi.pystep()
    def flagdata(vis: Path) -> _Out:
        seen.append((vis / "table.dat").read_text())
        return _Out(outputvis=vis)

    vis = _ms(workspace / "obs.ms")
    assert flagdata(vis=vis).success
    assert seen == ["rows"]


def _mstransform_body(outputvis: Path) -> _Out:
    """Module-level so the out-of-process runner can import it by path."""
    return _Out(outputvis=outputvis)


def test_out_of_process_pystep_clears_before_the_child_starts(workspace, monkeypatch):
    """The reported shape: a containerized CASA pystep. The clearing has to
    happen on *this* side, before the child is launched -- the child is the
    thing that would trip over the stale product."""
    from unittest.mock import MagicMock, patch

    outputvis = _ms(workspace / "obs_mst.ms")
    seen = {}

    def fake_run(argv, *args, **kwargs):
        seen["existed"] = outputvis.exists()
        runner = next(a for a in argv if a.endswith("runner.py"))
        (Path(runner).parent / "outputs.json").write_text(f'{{"outputvis": "{outputvis}"}}')
        proc = MagicMock(returncode=0, stdout="", stderr="")
        return proc

    ref = shinobi.pystep(image="test:latest", backend="docker", write_paths=["outputvis"])(_mstransform_body)
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=fake_run):
        assert ref(outputvis=outputvis).success

    assert seen["existed"] is False, "the child was launched with the previous run's product in place"


def test_pystep_write_paths_must_name_a_real_parameter():
    with pytest.raises(TypeError, match="write_paths names \\['nope'\\]"):

        @shinobi.pystep(write_paths=["nope"])
        def step(vis: Path) -> _Out:
            return _Out(outputvis=vis)
