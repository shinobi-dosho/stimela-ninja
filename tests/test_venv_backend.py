import warnings

import pytest

from shinobi.backends.venv import resolve_command, resolve_venv, venv_digest, venv_env
from shinobi.config import AppConfig
from shinobi.exceptions import BackendError
from shinobi.loaders import build_model
from shinobi.policies import build_argv
from shinobi.steps.schema import Cab

# A tool that dumps the environment shinobi set up, so tests can assert what
# the venv backend does to PATH/VIRTUAL_ENV/PYTHONPATH.
_ENV_DUMP = """#!/usr/bin/env python3
import os, sys
print("VIRTUAL_ENV=" + os.environ.get("VIRTUAL_ENV", ""))
print("PATH0=" + os.environ.get("PATH", "").split(os.pathsep)[0])
print("has_pythonhome=" + str("PYTHONHOME" in os.environ))
print("has_pythonpath=" + str("PYTHONPATH" in os.environ))
print("argv0=" + sys.argv[0])
"""


def make_cab(**kwargs) -> Cab:
    kwargs.setdefault("command", "/bin/echo")
    kwargs.setdefault("inputs_model", build_model("In", {}))
    kwargs.setdefault("outputs_model", build_model("Out", {}))
    return Cab(name="tool", **kwargs)


def test_runs_tool_from_venv_with_activated_env(venv_backend, make_venv):
    venv = make_venv(tool=("mytool", _ENV_DUMP))
    cab = make_cab(command="mytool", venv=str(venv))
    run = venv_backend.run(cab, build_argv(cab, {}), {}, stream=False)
    assert run.success
    out = dict(line.split("=", 1) for line in run.stdout.splitlines())
    assert out["VIRTUAL_ENV"] == str(venv)
    assert out["PATH0"] == str(venv / "bin")
    assert out["has_pythonhome"] == "False"
    assert out["has_pythonpath"] == "False"
    # argv0 was rewritten to the venv's own copy of the tool.
    assert out["argv0"] == str(venv / "bin" / "mytool")


def test_pythonpath_is_stripped_even_if_set_on_host(venv_backend, make_venv, monkeypatch):
    # Build the venv first -- a bogus PYTHONHOME on the host would break
    # `python -m venv` itself. Only the backend's own launch should see them.
    venv = make_venv(tool=("mytool", _ENV_DUMP))
    monkeypatch.setenv("PYTHONPATH", "/some/host/path")
    monkeypatch.setenv("PYTHONHOME", "/some/host/home")
    cab = make_cab(command="mytool", venv=str(venv))
    run = venv_backend.run(cab, build_argv(cab, {}), {}, stream=False)
    out = dict(line.split("=", 1) for line in run.stdout.splitlines())
    assert out["has_pythonpath"] == "False"
    assert out["has_pythonhome"] == "False"


def test_missing_tool_fails_loudly_not_silently(venv_backend, make_venv):
    # A tool that exists in neither the venv nor on PATH must fail loudly
    # (FileNotFoundError, same as the native backend), never silently no-op.
    venv = make_venv()
    cab = make_cab(command="definitely_not_a_real_tool_xyz", venv=str(venv))
    with pytest.raises(FileNotFoundError):
        venv_backend.run(cab, build_argv(cab, {}), {}, stream=False)


def test_no_venv_declared_falls_back_to_native_with_warning(venv_backend):
    cab = make_cab(command="/bin/echo")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = venv_backend.run(cab, ["/bin/echo", "hi"], {}, stream=False)
    assert run.success
    assert run.venv is None
    assert run.venv_digest is None
    assert any("running natively" in str(w.message) for w in caught)


def test_pin_yields_stable_digest_and_no_pin_yields_none(venv_backend, make_venv):
    venv = make_venv(tool=("mytool", _ENV_DUMP))
    cab = make_cab(command="mytool", venv=str(venv))
    unpinned = venv_backend.run(cab, build_argv(cab, {}), {}, stream=False, pin=False)
    assert unpinned.venv == str(venv)
    assert unpinned.venv_digest is None

    venv_digest.cache_clear()
    a = venv_backend.run(cab, build_argv(cab, {}), {}, stream=False, pin=True)
    b = venv_backend.run(cab, build_argv(cab, {}), {}, stream=False, pin=True)
    assert a.venv_digest is not None
    assert a.venv_digest == b.venv_digest


def test_digest_changes_when_a_package_is_added(make_venv):
    venv = make_venv(package=("dummypkg", "1.0.0", "x = 1\n"))
    before = venv_digest(venv)
    venv_digest.cache_clear()
    # Drop a second dist-info in place to simulate an install.
    site = next((venv / "lib").glob("python*")) / "site-packages"
    (site / "another.py").write_text("y = 2\n")
    dist = site / "another-2.0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: another\nVersion: 2.0.0\n")
    after = venv_digest(venv)
    assert before != after


def test_resolve_command_rewrites_only_bare_names_present_in_venv(make_venv):
    venv = make_venv(tool=("mytool", "#!/bin/sh\ntrue\n"))
    assert resolve_command(venv, "mytool") == str(venv / "bin" / "mytool")
    # not in the venv -> left as-is (host PATH still applies, activate-style)
    assert resolve_command(venv, "grep") == "grep"
    # an explicit path is never rewritten
    assert resolve_command(venv, "/usr/bin/grep") == "/usr/bin/grep"


def test_resolve_venv_from_config_default_and_named_env(make_venv):
    venv = make_venv(name="named")
    cfg = AppConfig.load(**{"backend": {"venv": {"default": "myenv", "envs": {"myenv": str(venv)}}}})
    # scope declares nothing -> config default 'myenv' -> mapped path
    assert resolve_venv(None, cfg) == venv
    # scope value that is a bare path also resolves
    assert resolve_venv(str(venv), cfg) == venv
    # nothing declared anywhere -> None
    empty = AppConfig.load()
    assert resolve_venv(None, empty) is None


def test_resolve_venv_missing_python_raises(tmp_path):
    bogus = tmp_path / "not-a-venv"
    bogus.mkdir()
    with pytest.raises(BackendError, match="does not exist"):
        resolve_venv(str(bogus), AppConfig.load())


def test_venv_env_prepends_bin_and_sets_virtual_env(make_venv):
    venv = make_venv()
    env = venv_env(venv)
    assert env["VIRTUAL_ENV"] == str(venv)
    assert env["PATH"].startswith(str(venv / "bin"))
    assert "PYTHONHOME" not in env
    assert "PYTHONPATH" not in env
