import pytest

from shinobi.backends.native import NativeBackend


@pytest.fixture(autouse=True)
def _native_test_backend(monkeypatch):
    # Host-only tools and in-process Python fixtures should not require a
    # container runtime. Tests of the production default clear this override.
    monkeypatch.setenv("SHINOBI_BACKEND__DEFAULT", "native")


@pytest.fixture(autouse=True)
def _isolate_runtime_paths(tmp_path_factory, monkeypatch):
    # Keep manifests, sandboxes and ownership outside the test's workspace:
    # cleanup/harvest tests may remove or inspect everything under tmp_path.
    # One numbered root avoids three scans of the growing pytest temp tree.
    root = tmp_path_factory.mktemp("runtime")
    for name in ("runs", "work", "ownership"):
        (root / name).mkdir()
    monkeypatch.setenv("SHINOBI_PROVENANCE__DIR", str(root / "runs"))
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(root / "work"))
    # Retain real cross-workflow admission within each test, without claims
    # leaking to another test or the production ~/.shinobi registry.
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(root / "ownership" / "owners.json"))


@pytest.fixture(autouse=True)
def _offline_digest_resolution(monkeypatch):
    # Image-digest resolution shells out to a registry/daemon. Neutralize all
    # resolvers by default so the suite never touches the network (steps run
    # unpinned); tests that care re-patch a specific resolver explicitly.
    from shinobi.backends import container

    for name in ("_registry_api_digest", "_registry_digest", "_docker_digest"):
        monkeypatch.setattr(container, name, lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _full_cgroup_delegation(monkeypatch):
    # What a rootless runtime may enforce is a property of the *developer's*
    # systemd session (see resources.delegated_controllers) -- a laptop with
    # only `memory pids` delegated would otherwise make every apptainer
    # limit assertion fail there and pass in CI, for no reason to do with the
    # code. Pin it to "everything delegated"; the tests that care about
    # partial delegation patch this themselves.
    from shinobi.backends import container

    monkeypatch.setattr(container, "delegated_controllers", lambda *a, **k: frozenset({"cpu", "memory", "pids", "io"}))


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
