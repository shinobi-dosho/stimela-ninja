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
