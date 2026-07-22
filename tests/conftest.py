import pytest

from shinobi.backends.native import NativeBackend


@pytest.fixture(autouse=True)
def _isolate_run_manifests(tmp_path_factory, monkeypatch):
    # Run-manifest emission is on by default; point it at a throwaway dir so
    # the suite still exercises the path without writing into the repo.
    monkeypatch.setenv("SHINOBI_PROVENANCE__DIR", str(tmp_path_factory.mktemp("runs")))


@pytest.fixture(autouse=True)
def _isolate_sandboxes(tmp_path_factory, monkeypatch):
    # Step sandboxes (and `ninja clean`'s default --sandboxes target) resolve
    # AppConfig.sandbox.dir, which is cwd-relative by default; point it at a
    # throwaway dir so no test can ever create or delete .shinobi/work in the
    # repo. Tests that care about the location override this env var.
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path_factory.mktemp("work")))


@pytest.fixture(autouse=True)
def _offline_digest_resolution(monkeypatch):
    # Image-digest resolution shells out to a registry/daemon. Neutralize all
    # resolvers by default so the suite never touches the network (steps run
    # unpinned); tests that care re-patch a specific resolver explicitly.
    from shinobi.backends import container

    for name in ("_registry_api_digest", "_registry_digest", "_docker_digest"):
        monkeypatch.setattr(container, name, lambda *a, **k: None)


@pytest.fixture
def native():
    return NativeBackend()


@pytest.fixture
def venv_backend():
    from shinobi.backends.venv import VenvBackend, venv_digest

    # venv_digest is lru_cached on the venv path; per-test venvs live in fresh
    # tmp dirs so paths don't collide, but clear it anyway for isolation.
    venv_digest.cache_clear()
    return VenvBackend()


@pytest.fixture
def make_venv(tmp_path):
    """Build a throwaway virtualenv and return its path. Optionally drops an
    executable tool script into its bin/, and/or a stub package into its
    site-packages (offline -- no pip install), so tests can prove venv-only
    imports without touching the network.
    """
    import subprocess
    import sys

    def _make(name: str = "venv", *, tool: tuple[str, str] | None = None, package: tuple[str, str, str] | None = None):
        venv = tmp_path / name
        # --without-pip: the venv backend reads distributions via stdlib
        # importlib.metadata, never pip, so bootstrapping pip is wasted time.
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, capture_output=True)
        if tool is not None:
            tool_name, body = tool
            script = venv / "bin" / tool_name
            script.write_text(body)
            script.chmod(0o755)
        if package is not None:
            mod_name, version, source = package
            site = next((venv / "lib").glob("python*")) / "site-packages"
            (site / f"{mod_name}.py").write_text(source)
            dist = site / f"{mod_name}-{version}.dist-info"
            dist.mkdir(parents=True, exist_ok=True)
            (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {mod_name}\nVersion: {version}\n")
        return venv

    return _make
