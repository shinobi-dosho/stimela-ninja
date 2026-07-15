"""Unit coverage for run-manifest provenance and image pinning."""

import hashlib

import pytest
from pydantic import BaseModel

from shinobi.backends import container as C
from shinobi.provenance import build_manifest
from shinobi.results import StepResult


class _M(BaseModel):
    v: int = 0


_CONTAINERISH = {"docker", "podman", "apptainer", "slurm"}


def _cab_result(name, *, backend=None, image=None, digest=None, cached=False, containerized=None):
    if containerized is None:
        containerized = backend in _CONTAINERISH
    return StepResult(
        name=name,
        returncode=0,
        outputs=_M(),
        inputs=_M(),
        kind="cab",
        backend=backend,
        image=image,
        image_digest=digest,
        containerized=containerized,
        cached=cached,
    )


# -- pure ref helpers --


def test_strip_tag_leaves_registry_port_intact():
    assert C._strip_tag("quay.io/org/img:1.0") == "quay.io/org/img"
    assert C._strip_tag("localhost:5000/img") == "localhost:5000/img"  # port, no tag
    assert C._strip_tag("localhost:5000/img:1.0") == "localhost:5000/img"


def test_with_digest_produces_canonical_ref():
    d = "sha256:" + "a" * 64
    assert C._with_digest("alpine:3.19", d) == f"alpine@{d}"
    assert C._with_digest("docker://quay.io/x:1", d) == f"docker://quay.io/x@{d}"


# -- _pin_image, hermetically (no real skopeo/docker) --


def test_pin_local_sif_is_content_hash(tmp_path):
    sif = tmp_path / "img.sif"
    sif.write_bytes(b"fake sif bytes")
    ref, digest = C._pin_image("apptainer", str(sif))
    assert ref == str(sif)
    assert digest == "sha256:" + hashlib.sha256(b"fake sif bytes").hexdigest()


def test_pin_prefers_pure_python_registry_api(monkeypatch):
    # The pure-Python registry query is primary and wins over the binaries.
    d = "sha256:" + "b" * 64
    monkeypatch.setattr(C, "_registry_api_digest", lambda ref: d)
    monkeypatch.setattr(C, "_registry_digest", lambda ref: "sha256:" + "9" * 64)
    ref, digest = C._pin_image("docker", "alpine:3.19")
    assert digest == d
    assert ref == f"alpine@{d}"


def test_pin_falls_back_through_skopeo_then_docker(monkeypatch):
    # api miss -> skopeo miss -> docker-native buildx query.
    d = "sha256:" + "c" * 64
    monkeypatch.setattr(C, "_registry_api_digest", lambda ref: None)
    monkeypatch.setattr(C, "_registry_digest", lambda ref: None)
    monkeypatch.setattr(C, "_docker_digest", lambda runtime, image: d)
    ref, digest = C._pin_image("docker", "alpine:3.19")
    assert digest == d and ref == f"alpine@{d}"


def test_pin_unresolvable_is_honestly_unpinned(monkeypatch):
    # (conftest already neutralizes all three; being explicit here.)
    monkeypatch.setattr(C, "_registry_api_digest", lambda ref: None)
    monkeypatch.setattr(C, "_registry_digest", lambda ref: None)
    monkeypatch.setattr(C, "_docker_digest", lambda runtime, image: None)
    ref, digest = C._pin_image("apptainer", "quay.io/x/y:1")
    assert ref == "docker://quay.io/x/y:1"
    assert digest is None


def test_pin_registry_ref_pins_apptainer_via_pure_python(monkeypatch):
    # The gap-closer: an apptainer registry ref pins with no skopeo/buildx,
    # because the pure-Python resolver is runtime-agnostic.
    d = "sha256:" + "e" * 64
    monkeypatch.setattr(C, "_registry_api_digest", lambda ref: d)
    ref, digest = C._pin_image("apptainer", "quay.io/stimela2/casa6:latest")
    assert digest == d
    assert ref == f"docker://quay.io/stimela2/casa6@{d}"


@pytest.mark.parametrize(
    "ref, expected",
    [
        ("alpine:3.19", ("registry-1.docker.io", "library/alpine", "3.19")),
        ("alpine", ("registry-1.docker.io", "library/alpine", "latest")),
        ("quay.io/stimela2/casa6:6.7", ("quay.io", "stimela2/casa6", "6.7")),
        ("docker://quay.io/org/img:1", ("quay.io", "org/img", "1")),
        ("localhost:5000/img:1", ("localhost:5000", "img", "1")),
        ("ghcr.io/o/i", ("ghcr.io", "o/i", "latest")),
    ],
)
def test_split_ref_parses_registry_repo_reference(ref, expected):
    assert C._split_ref(ref) == expected


def test_split_ref_rejects_non_docker_schemes():
    assert C._split_ref("oras://reg/x:1") is None
    assert C._split_ref("library://x/y:1") is None


# -- registry credentials (hermetic; no network) --


def test_docker_config_static_auth(tmp_path, monkeypatch):
    import base64
    import json

    cfg = tmp_path / "config.json"
    token = base64.b64encode(b"alice:s3cret").decode()
    cfg.write_text(json.dumps({"auths": {"quay.io": {"auth": token}}}))
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    C._docker_config_auth.cache_clear()

    assert C._docker_config_auth("quay.io") == ("alice", "s3cret")
    assert C._docker_config_auth("no.such.registry") is None
    C._docker_config_auth.cache_clear()


def test_auth_keys_include_docker_hub_legacy_key():
    keys = C._auth_keys("registry-1.docker.io")
    assert "https://index.docker.io/v1/" in keys
    assert "quay.io" in C._auth_keys("quay.io")


def test_authorize_basic_scheme_uses_creds():
    header = C._authorize('Basic realm="x"', ("alice", "s3cret"))
    assert header == C._basic_auth(("alice", "s3cret"))
    assert C._authorize('Basic realm="x"', None) is None  # no creds -> can't


def test_authorize_bearer_delegates_to_token(monkeypatch):
    monkeypatch.setattr(C, "_bearer_token", lambda challenge, creds=None: "tok123")
    assert C._authorize('Bearer realm="x",service="y"', None) == "Bearer tok123"


# -- pin gate (opt-in behaviour) --


def _image_cab():
    from shinobi.loaders import build_model
    from shinobi.steps.schema import Cab

    return Cab(
        name="t", command="t", image="alpine:3.19",
        inputs_model=build_model("I", {}), outputs_model=build_model("O", {}),
    )


def test_build_container_argv_pin_off_runs_original_ref():
    argv, digest = C.build_container_argv("docker", _image_cab(), ["t"], {}, "/w", pin=False)
    assert digest is None
    assert "alpine:3.19" in argv and not any("@sha256:" in a for a in argv)


def test_build_container_argv_pin_on_pins(monkeypatch):
    d = "sha256:" + "a" * 64
    monkeypatch.setattr(C, "_registry_api_digest", lambda ref: d)
    argv, digest = C.build_container_argv("docker", _image_cab(), ["t"], {}, "/w", pin=True)
    assert digest == d
    assert f"alpine@{d}" in argv


# -- provenance emission is opt-in --


def _true_cab(name):
    from shinobi.loaders import build_model
    from shinobi.steps.schema import Cab

    return Cab(
        name=name, command="true", backend="native",
        inputs_model=build_model("I", {}), outputs_model=build_model("O", {}),
    )


def _runs_dir():
    import os

    from pathlib import Path

    return Path(os.environ["SHINOBI_PROVENANCE__DIR"])


def test_dispatch_provenance_off_emits_no_manifest():
    from shinobi.steps.dispatch import _dispatch

    _dispatch(_true_cab("p_off"), None, provenance=False)
    assert not list(_runs_dir().glob("p_off*.run.json"))


def test_dispatch_provenance_on_emits_manifest():
    import json

    from shinobi.steps.dispatch import _dispatch

    _dispatch(_true_cab("p_on"), None, provenance=True)
    files = list(_runs_dir().glob("p_on*.run.json"))
    assert files, "provenance=True must emit a run manifest"
    assert json.loads(files[-1].read_text())["root"]["name"] == "p_on"


# -- manifest + pinned --


def test_manifest_pinned_true_for_native_and_digested_container():
    root = StepResult(
        name="r",
        returncode=0,
        outputs=_M(),
        inputs=_M(),
        kind="recipe",
        sub_results={
            "native": _cab_result("native", backend="native"),  # image irrelevant
            "docker": _cab_result("docker", backend="docker", image="a:1", digest="sha256:" + "d" * 64),
        },
    )
    m = build_manifest(root, backend="native")
    assert m.pinned is True
    assert [s.name for s in m.root.steps] == ["native", "docker"]  # declaration order


def test_manifest_unpinned_when_container_step_lacks_digest():
    root = StepResult(
        name="r",
        returncode=0,
        outputs=_M(),
        inputs=_M(),
        kind="recipe",
        sub_results={"docker": _cab_result("docker", backend="docker", image="a:1", digest=None)},
    )
    assert build_manifest(root, backend="docker").pinned is False


def test_manifest_pinned_keys_on_containerized_not_backend_name():
    # A native cab with an image set (image is mere metadata) does NOT drag
    # pinned false; a Slurm-under-apptainer step that containerized but
    # couldn't pin DOES -- even though "slurm" isn't a container-runtime name.
    native_with_image = _cab_result("native", backend="native", image="a:1", digest=None)
    assert build_manifest(native_with_image, backend="native").pinned is True

    slurm_unpinned = _cab_result("s", backend="slurm", image="a:1", digest=None)  # containerized via _CONTAINERISH
    assert slurm_unpinned.containerized is True
    assert build_manifest(slurm_unpinned, backend="slurm").pinned is False

    slurm_pinned = _cab_result("s", backend="slurm", image="a:1", digest="sha256:" + "f" * 64)
    assert build_manifest(slurm_pinned, backend="slurm").pinned is True


def test_cache_roundtrip_preserves_provenance(tmp_path):
    from shinobi.cache import CacheManifest
    from shinobi.steps import Cab

    class NoIn(BaseModel):
        pass

    scope = Cab(name="c", command="c", inputs_model=NoIn, outputs_model=_M)
    manifest = CacheManifest(tmp_path / "manifest.json")
    d = "sha256:" + "e" * 64
    manifest.record("c", "k", _cab_result("c", backend="docker", image="a:1", digest=d))

    hit = manifest.check("c", "k", scope, {})
    # A cache hit reconstructs full provenance, so it doesn't spuriously mark
    # a re-run unpinned (caveat 1).
    assert hit.kind == "cab" and hit.backend == "docker"
    assert hit.image == "a:1" and hit.image_digest == d and hit.cached is True


def test_manifest_serializes_non_jsonable_inputs_lossily():
    class Weird(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        obj: object = object()

    result = StepResult(name="w", returncode=0, outputs=_M(), inputs=Weird(), kind="cab", backend="native")
    m = build_manifest(result, backend="native")
    assert isinstance(m.root.inputs["obj"], str)  # degraded to str, did not crash


def test_manifest_step_records_use_step_names_not_cab_names():
    # One cab backing two steps: sub_results keys are the *step* names
    # (what replay matches on); each value's own .name is the cab's.
    root = StepResult(
        name="r",
        returncode=0,
        outputs=_M(),
        inputs=_M(),
        kind="recipe",
        sub_results={
            "first": _cab_result("echo", backend="native"),
            "second": _cab_result("echo", backend="native"),
        },
    )
    m = build_manifest(root, backend="native")
    assert [s.name for s in m.root.steps] == ["first", "second"]


# -- target recording (for `ninja replay`) --


def test_dispatch_records_provenance_target():
    import json

    from shinobi.steps.dispatch import _dispatch

    _dispatch(_true_cab("p_target"), None, provenance=True, _provenance_target="f.py:r")
    files = list(_runs_dir().glob("p_target*.run.json"))
    assert files and json.loads(files[-1].read_text())["target"] == "f.py:r"


def test_dispatch_without_target_records_honest_null():
    import json

    from shinobi.steps.dispatch import _dispatch

    _dispatch(_true_cab("p_notarget"), None, provenance=True)
    files = list(_runs_dir().glob("p_notarget*.run.json"))
    assert files and json.loads(files[-1].read_text())["target"] is None


def test_old_manifest_without_target_still_validates():
    from shinobi.provenance import RunManifest

    m = build_manifest(_cab_result("old", backend="native"), backend="native")
    payload = m.model_dump(mode="json")
    del payload["target"]  # a manifest written before the field existed
    import json

    old = RunManifest.model_validate_json(json.dumps(payload))
    assert old.target is None


# -- replay: unpinned_steps + apply_manifest_pins --


def _rec(name, *, kind="cab", image=None, digest=None, containerized=False, steps=(), inputs=None):
    from shinobi.provenance import StepRecord

    return StepRecord(
        name=name, kind=kind, returncode=0, cached=False,
        image=image, image_digest=digest, containerized=containerized,
        inputs=inputs or {}, outputs={}, steps=list(steps),
    )


def test_unpinned_steps_names_offenders():
    from shinobi.provenance import unpinned_steps

    d = "sha256:" + "a" * 64
    root = _rec("outer", kind="recipe", steps=[
        _rec("ok", image="a:1", digest=d, containerized=True),
        _rec("inner", kind="recipe", steps=[_rec("bad", image="b:1", containerized=True)]),
        _rec("native"),  # not containerized; never an offender
    ])
    assert unpinned_steps(root) == ["bad"]


class _RIn(BaseModel):
    x: int = 0


class _Empty(BaseModel):
    pass


def _image_step_cab(name, image):
    from shinobi.steps.schema import Cab

    return Cab(name=name, command=name, image=image, inputs_model=_RIn, outputs_model=_Empty)


def test_apply_pins_rewrites_cab_image():
    from shinobi.provenance import apply_manifest_pins

    d = "sha256:" + "a" * 64
    cab = _image_step_cab("t", "alpine:3.19")
    pinned = apply_manifest_pins(cab, _rec("t", image="alpine:3.19", digest=d, containerized=True))
    assert pinned.image == f"alpine@{d}"
    assert cab.image == "alpine:3.19"  # original untouched


def test_apply_pins_leaves_native_and_sif_unchanged():
    from shinobi.provenance import apply_manifest_pins

    native = _image_step_cab("n", "alpine:3.19")
    pinned = apply_manifest_pins(native, _rec("n", image="alpine:3.19"))
    assert pinned.image == "alpine:3.19"
    assert pinned is not native  # unchanged still means a fresh instance

    sif = _image_step_cab("s", "/imgs/tool.sif")
    d = "sha256:" + "b" * 64  # a content hash, not a registry ref
    pinned = apply_manifest_pins(sif, _rec("s", image="/imgs/tool.sif", digest=d, containerized=True))
    assert pinned.image == "/imgs/tool.sif"
    assert pinned is not sif


def _nested_recipe():
    from shinobi.steps.schema import Recipe

    inner = Recipe(name="inner", inputs_model=_RIn, outputs_model=_Empty)
    inner.add_step("c", _image_step_cab("c", "quay.io/x/y:1"), x=inner.inputs.x)
    outer = Recipe(name="outer", inputs_model=_RIn, outputs_model=_Empty)
    outer.add_step("a", _image_step_cab("a", "alpine:3.19"), x=outer.inputs.x)
    outer.add_step("b", inner, x=outer.inputs.x)
    return outer


def test_apply_pins_recipe_matches_by_name_and_recurses():
    from shinobi.graph import build_graph
    from shinobi.provenance import apply_manifest_pins

    d1, d2 = "sha256:" + "1" * 64, "sha256:" + "2" * 64
    outer = _nested_recipe()
    root = _rec("outer", kind="recipe", steps=[
        _rec("a", image="alpine:3.19", digest=d1, containerized=True),
        _rec("b", kind="recipe", steps=[_rec("c", image="quay.io/x/y:1", digest=d2, containerized=True)]),
    ])
    pinned = apply_manifest_pins(outer, root)
    assert pinned.steps[0].step.image == f"alpine@{d1}"
    assert pinned.steps[1].step.steps[0].step.image == f"quay.io/x/y@{d2}"
    # ref identity data (name/wiring) survives the copy, so by-name wiring
    # and the dependency graph still hold together.
    assert [ref.name for ref in pinned.steps] == ["a", "b"]
    assert pinned.steps[0].wiring == outer.steps[0].wiring
    build_graph(pinned)  # must not raise
    assert outer.steps[0].step.image == "alpine:3.19"  # original untouched
    # No node of the pinned tree is shared with the original, and Recipe's
    # mutable builder surface (steps list, output_wiring dict) is its own --
    # add_step/set_output on one tree can't leak into the other.
    assert pinned is not outer
    assert pinned.steps is not outer.steps
    assert pinned.output_wiring is not outer.output_wiring
    inner_pinned, inner_orig = pinned.steps[1].step, outer.steps[1].step
    assert inner_pinned is not inner_orig
    assert inner_pinned.steps is not inner_orig.steps
    assert inner_pinned.output_wiring is not inner_orig.output_wiring


def test_apply_pins_shape_mismatches_error():
    from shinobi.exceptions import ReplayError
    from shinobi.provenance import apply_manifest_pins

    outer = _nested_recipe()

    # a manifest step the recipe no longer has
    extra = _rec("outer", kind="recipe", steps=[
        _rec("a", containerized=False), _rec("b", kind="recipe"), _rec("gone"),
    ])
    with pytest.raises(ReplayError, match="gone"):
        apply_manifest_pins(outer, extra)

    # a recipe step the manifest never ran (recipe grew, or run stopped early)
    partial = _rec("outer", kind="recipe", steps=[_rec("a", containerized=False)])
    with pytest.raises(ReplayError, match="'b'"):
        apply_manifest_pins(outer, partial)

    # the target changed shape entirely
    with pytest.raises(ReplayError, match="changed shape"):
        apply_manifest_pins(_image_step_cab("outer", "a:1"), _rec("outer", kind="recipe"))
