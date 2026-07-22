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


def test_pystep_venv_no_venv_declared_falls_back_in_process():
    # backend=venv but nothing declared -> in-process, with a warning.
    ref = pystep(venv=None, backend="venv")(funcs.plain_double)
    with pytest.warns(UserWarning, match="running in-process"):
        result = ref(n=21)
    assert result.success
    assert result.outputs.value == 42
    assert result.venv is None


def test_pystep_venv_sandboxed_end_to_end(make_venv, tmp_path, monkeypatch):
    # A sandboxed venv pystep: the runner must launch with cwd inside the
    # sandbox (the venv path has no container --workdir flag), and its declared
    # Path output must be harvested back to the workspace.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / ".shinobi/work"))
    venv = _venv_with_pkg(make_venv)
    ref = pystep(venv=str(venv), backend="venv", sandbox=True)(funcs.write_report)

    result = ref(n=1)

    assert result.success, result.stderr
    assert result.sandboxed is True
    # harvested from the sandbox cwd back to the workspace
    assert (tmp_path / "report.txt").read_text() == "magic=4243\n"
    # sandbox discarded on success
    assert list((tmp_path / ".shinobi/work").iterdir()) == []
