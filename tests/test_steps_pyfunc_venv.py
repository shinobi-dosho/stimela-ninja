"""End-to-end proof that a `venv`-backed pystep runs under the venv's own
interpreter and imports the venv's real packages -- something the container
tests can only fake on the host. These run a real subprocess (the venv's
python), so no runtime is mocked.

The decorated functions live in `_venv_pystep_funcs` (framework + stdlib
imports only): the runner execs that whole file under the venv, so this test
module's own `import pytest` must never be in the loaded source. The venv-only
package name (`venvonlypkg`) also deliberately differs from that module's own
package -- the venv launcher does not stub the target's own package, so an
import of the function's own module would resolve for real and prove nothing.
"""

from __future__ import annotations

import pytest

from shinobi import pystep
from shinobi.config import AppConfig, BackendConfig, VenvConfig

from tests import _venv_pystep_funcs as funcs

_VENV_ONLY_SOURCE = "MAGIC = 4242\n"


def _venv_with_pkg(make_venv):
    return make_venv(package=("venvonlypkg", "1.0.0", _VENV_ONLY_SOURCE))


def test_pystep_runs_in_venv_and_imports_venv_only_package(make_venv):
    venv = _venv_with_pkg(make_venv)
    ref = pystep(venv=str(venv), backend="venv")(funcs.use_venv_only_pkg)

    result = ref(n=8)

    assert result.success, result.stderr
    assert result.outputs.value == 4250  # 4242 + 8
    assert result.kind == "pyfunc"
    assert result.backend == "venv"
    assert result.venv == str(venv)
    assert result.containerized is False


def test_pystep_venv_records_digest_under_provenance(make_venv):
    from shinobi.backends.venv import venv_digest
    from shinobi.steps.dispatch import _dispatch

    venv = _venv_with_pkg(make_venv)
    venv_digest.cache_clear()
    ref = pystep(venv=str(venv), backend="venv")(funcs.use_venv_only_pkg)

    # `pin` rides on the provenance flag threaded through dispatch.
    result = _dispatch(ref.step, ref.func, provenance=True, n=1)

    assert result.success
    assert result.venv == str(venv)
    assert result.venv_digest is not None


def test_pystep_venv_no_venv_declared_falls_back_in_process(monkeypatch):
    # backend=venv but nothing declared -> in-process, with a warning.
    # Isolate from any host config default so the fallback is really tested.
    clean = AppConfig(backend=BackendConfig(venv=VenvConfig(default=None, envs={})))
    monkeypatch.setattr(AppConfig, "load", lambda config_file=None, **overrides: clean)
    ref = pystep(venv=None, backend="venv")(funcs.plain_double)
    with pytest.warns(UserWarning, match="running in-process"):
        result = ref(n=21)
    assert result.success
    assert result.outputs.value == 42
    assert result.venv is None


@pytest.mark.parametrize("sandbox", [False, True])
def test_pystep_venv_nested_output_end_to_end(make_venv, tmp_path, monkeypatch, sandbox):
    # A sandboxed venv pystep: the runner must launch with cwd inside the
    # sandbox (the venv path has no container --workdir flag), and its declared
    # Path output must be harvested back to the workspace.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / ".shinobi/work"))
    venv = _venv_with_pkg(make_venv)
    ref = pystep(venv=str(venv), backend="venv", sandbox=sandbox)(funcs.write_report)

    result = ref(n=1)

    assert result.success, result.stderr
    assert result.sandboxed is sandbox
    # harvested from the sandbox cwd back to the workspace
    assert (tmp_path / "reports/deep/report.txt").read_text() == "magic=4243\n"
    # sandbox discarded on success
    if sandbox:
        assert list((tmp_path / ".shinobi/work").iterdir()) == []


@pytest.mark.parametrize("sandbox", [False, True])
def test_venv_optional_filename_and_harvest_inventory(make_venv, tmp_path, monkeypatch, sandbox):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / "sandboxes"))
    venv = _venv_with_pkg(make_venv)
    ref = pystep(venv=str(venv), backend="venv", sandbox=sandbox, cache=True, cache_dir=str(tmp_path / "cache"), harvest=["{prefix}-*.fits"])(funcs.produce_family)
    first = ref(prefix="image")
    second = ref(prefix="image")
    assert second.cached
    assert first.outputs.mfs == second.outputs.mfs == funcs.Path("image-MFS-image.fits")
    (tmp_path / "image-0000-I-image.fits").unlink()
    assert not ref(prefix="image").cached


@pytest.mark.parametrize("sandbox", [False, True])
@pytest.mark.parametrize("cache_enabled", [False, True])
def test_venv_recreated_empty_declared_directory_survives(make_venv, tmp_path, monkeypatch, sandbox, cache_enabled):
    from shinobi.cache import get_cache_manifest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / "sandboxes"))
    venv = _venv_with_pkg(make_venv)
    ref = pystep(venv=str(venv), backend="venv", sandbox=sandbox, cache=cache_enabled, cache_dir=str(tmp_path / "cache"), harvest=["{prefix}/*.fits"])(
        funcs.produce_recreated_empty_directory
    )
    first = ref(prefix="empty")
    assert first.outputs.mfs == funcs.Path("empty")
    assert (tmp_path / "empty").is_dir()
    if cache_enabled:
        assert get_cache_manifest(str(tmp_path / "cache")).entry(ref.step.name)["products"] == [str(tmp_path / "empty")]
    assert ref(prefix="empty").cached is cache_enabled
    (tmp_path / "empty").rmdir()
    assert not ref(prefix="empty").cached
    assert (tmp_path / "empty").is_dir()


@pytest.mark.parametrize("sandbox", [False, True])
def test_venv_unavailable_setup_observation_preserves_success(make_venv, tmp_path, monkeypatch, sandbox):
    import shinobi.sandbox as sandbox_module
    from shinobi.cache import get_cache_manifest
    from shinobi.steps.dispatch import _dispatch

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / "sandboxes"))
    venv = _venv_with_pkg(make_venv)
    ref = pystep(venv=str(venv), backend="venv", sandbox=sandbox, cache=True, cache_dir=str(tmp_path / "cache"), harvest=["{prefix}/*.fits"])(funcs.produce_recreated_directory)
    original = sandbox_module.observe_product_path

    def observe(path, *, recursive=False):
        if path.name == "empty" and not recursive:
            raise PermissionError("setup cache evidence unavailable")
        return original(path, recursive=recursive)

    monkeypatch.setattr(sandbox_module, "observe_product_path", observe)
    for index, run_id in enumerate(["successful-venv-without-evidence-1", "successful-venv-without-evidence-2"]):
        result = _dispatch(ref.step, ref.func, prefix="empty", _run_id=run_id)
        assert result.success and not result.cached
        assert (tmp_path / "empty/science.dat").read_text() == "science"
        entry = get_cache_manifest(str(tmp_path / "cache")).entry(ref.step.name)
        assert entry["run_id"] == run_id
        expected = None if sandbox or index == 0 else [str(tmp_path / "empty")]
        assert entry["products"] == expected
