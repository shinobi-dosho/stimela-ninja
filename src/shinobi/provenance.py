"""Run manifest: a static, self-contained record of a completed run.

Emitted once per top-level run, the manifest freezes everything reproducible
about that run -- every step's resolved inputs and outputs, the backend, and
the digest of each container image that actually executed (pin-then-run
guarantees the recorded digest is the one that ran; see
`backends.container._pin_image`).

`pinned` is True only when every containerized step ran a digest-resolved
image. A run that can't honestly claim reproducibility -- a native step, an
unpinnable local-built image, an offline resolution -- reports `pinned:
false` rather than implying a reproducibility it doesn't have.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, computed_field

from shinobi import __version__ as _shinobi_version
from shinobi.resources import Resources
from shinobi.results import StepResult

if TYPE_CHECKING:
    from shinobi.config import AppConfig
    from shinobi.steps.schema import Scope


def _jsonable(model: BaseModel) -> dict[str, Any]:
    """Dump a pydantic model to JSON-safe primitives, degrading anything not
    natively serializable (e.g. a MUTABLE input field holding a live Python
    object) to its `str` form instead of failing the whole manifest.
    """
    return json.loads(json.dumps(model.model_dump(mode="python"), default=str))


class StepRecord(BaseModel):
    """One step's frozen contribution to a run: its resolved inputs and
    outputs plus the container provenance (`kind` stamped explicitly at the
    source, never inferred). `steps` holds a recipe's sub-steps in
    declaration order.
    """

    name: str
    kind: str
    returncode: int
    cached: bool
    # True for an unrolled loop iteration that ran after the loop had
    # converged. Defaulted so manifests written before loops existed still
    # load. This is what makes the manifest an exact record of how many
    # cycles a run actually performed, as opposed to how many it declared.
    skipped: bool = False
    backend: str | None = None
    image: str | None = None
    image_digest: str | None = None
    containerized: bool = False
    # The venv this step ran in and its version-parity digest. A step with
    # `venv` set is always reported unpinned (`RunManifest.pinned`) -- the
    # digest is informational, not an OS-level pin. Defaulted so manifests
    # written before the venv backend existed still load.
    venv: str | None = None
    venv_digest: str | None = None
    sandboxed: bool = False
    # What the step declared it needed, if anything. Purely diagnostic: it is
    # what turns a post-mortem `returncode -9` into "SIGKILL, and it had
    # declared 200GiB". Deliberately NOT restored by `apply_manifest_pins` --
    # a footprint describes the box a run happened on, not the run itself, so
    # replaying it elsewhere should use that machine's own declaration.
    resources: Resources | None = None
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    steps: list["StepRecord"] = []


class RunManifest(BaseModel):
    """The whole run, frozen. `stimela_version` and `cab_repo_commit` are
    nullable and only populated once those provenance sources are wired --
    they are honest nulls, never fabricated. `target` records the CLI target
    string ('path/to/file.py:name' or 'pkg.mod:name') that produced the run,
    so `ninja replay` can find the recipe again; it is null for programmatic
    runs and manifests written before the field existed.
    """

    schema_version: int = 1
    shinobi_version: str
    stimela_version: str | None = None
    cab_repo_commit: str | None = None
    target: str | None = None
    generated_at: datetime
    backend: str
    returncode: int
    root: StepRecord

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pinned(self) -> bool:
        """False if any step that ran *inside a container* lacks a resolved
        image digest, or ran in a *venv* at all. Keyed on `containerized`
        (not the backend name) so Slurm-under-apptainer counts, and a native
        cab whose image is mere metadata does not. A venv step is always
        unpinned: its `venv_digest` is version-parity, not an OS-level image
        pin, so it never earns the reproducibility claim a pinned container
        does (see `backends.venv`).
        """

        def ok(record: StepRecord) -> bool:
            self_ok = ((not record.containerized) or record.image_digest is not None) and record.venv is None
            return self_ok and all(ok(child) for child in record.steps)

        return ok(self.root)

    def write(self, path: Path) -> Path:
        """Serialize the manifest to `path` (parent dirs created), returning it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path


def _record(result: StepResult, name: str | None = None) -> StepRecord:
    """Walk a `StepResult` (and its `sub_results`) into a `StepRecord` tree.

    `name` overrides `result.name`: a recipe's `sub_results` are keyed by the
    *step* name (the `StepRef.name` replay matches on), while each value's
    own `.name` is the underlying cab's -- ambiguous when one cab backs
    several steps.
    """
    return StepRecord(
        name=name or result.name,
        kind=result.kind,
        returncode=result.returncode,
        cached=result.cached,
        skipped=result.skipped,
        backend=result.backend,
        image=result.image,
        image_digest=result.image_digest,
        containerized=result.containerized,
        venv=result.venv,
        venv_digest=result.venv_digest,
        sandboxed=result.sandboxed,
        resources=result.resources,
        inputs=_jsonable(result.inputs),
        outputs=_jsonable(result.outputs),
        steps=[_record(sub, name=key) for key, sub in (result.sub_results or {}).items()],
    )


def build_manifest(result: StepResult, *, backend: str, target: str | None = None) -> RunManifest:
    """Build a `RunManifest` freezing a completed top-level run's `result`."""
    return RunManifest(
        shinobi_version=_shinobi_version,
        target=target,
        generated_at=datetime.now(timezone.utc),
        backend=backend,
        returncode=result.returncode,
        root=_record(result),
    )


def load_manifest(path: Path) -> RunManifest:
    """Read and validate a run manifest from `path`."""
    return RunManifest.model_validate_json(path.read_text())


def unpinned_steps(record: StepRecord) -> list[str]:
    """Names of every step in `record`'s tree that makes `RunManifest.pinned`
    false -- ran containerized but has no resolved image digest, or ran in a
    venv (always unpinned) -- named so a refusal to replay can say which ones.
    """
    names: list[str] = []
    if (record.containerized and record.image_digest is None) or record.venv is not None:
        names.append(record.name)
    for child in record.steps:
        names.extend(unpinned_steps(child))
    return names


def apply_manifest_pins(scope: "Scope", record: StepRecord) -> "Scope":
    """Return a copy of `scope` with every containerized step's image forced
    to the `repo@sha256:...` its manifest `record` recorded, so a replay runs
    exactly the images of the original run (an already-digest-pinned ref
    passes through pin-then-run without a registry round-trip). A venv step is
    likewise forced to the venv `record` recorded, so a replay runs the venv
    that originally ran rather than whatever the current declaration carries
    (a venv is always unpinned, so this only takes effect under
    `--allow-unpinned`).

    Recipe sub-steps are matched to `record.steps` by name; any shape
    difference -- the recipe changed since the manifest was written, or the
    recorded run stopped before finishing -- raises `ReplayError` rather than
    replaying something other than what the manifest froze. Never mutates
    `scope`, and every node of the returned tree is a fresh instance (even
    steps left unchanged), so the pinned tree shares no Scope objects with
    the original.
    """
    from shinobi.backends.container import _with_digest
    from shinobi.exceptions import ReplayError
    from shinobi.steps.schema import Recipe

    if isinstance(scope, Recipe) != (record.kind == "recipe"):
        raise ReplayError(
            f"step {record.name!r}: manifest records a {record.kind}, but the target "
            f"now resolves to a {type(scope).__name__} -- the code has changed shape "
            "since this manifest was written"
        )
    if isinstance(scope, Recipe):
        by_name = {rec.name: rec for rec in record.steps}
        current = {ref.name for ref in scope.steps}
        if extra := sorted(set(by_name) - current):
            raise ReplayError(f"manifest step(s) {extra} no longer exist in recipe {scope.name!r} -- the recipe has changed since this manifest was written")
        if missing := sorted(current - set(by_name)):
            raise ReplayError(
                f"recipe {scope.name!r} step(s) {missing} are not in the manifest -- "
                "either the recipe gained steps since the manifest was written, or the "
                "recorded run failed/stopped before reaching them; such a run cannot "
                "be replayed exactly"
            )
        new_steps = [ref.model_copy(update={"step": apply_manifest_pins(ref.step, by_name[ref.name])}) for ref in scope.steps]
        # `output_wiring` is the rest of Recipe's mutable builder surface --
        # give the copy its own dict so `set_output` on one can't leak into
        # the other.
        return scope.model_copy(update={"steps": new_steps, "output_wiring": dict(scope.output_wiring)})
    updates: dict[str, Any] = {}
    if record.venv is not None:
        # Replay a venv step using the venv that originally ran, not whatever
        # the current scope declaration happens to carry.
        updates["venv"] = record.venv
    if record.containerized and record.image_digest and record.image:
        if not record.image.endswith(".sif"):
            updates["image"] = _with_digest(record.image, record.image_digest)
        # A .sif's recorded digest is a content hash of the local file, not a
        # registry ref -- the path already names the exact image, so there is
        # nothing to rewrite.
    return scope.model_copy(update=updates)


def run_manifest_path(config: "AppConfig", name: str) -> Path:
    """Destination for a run's manifest: `{provenance.dir}/{name}.{utc}.{pid}.run.json`.
    The pid keeps two runs of the same recipe in one timestamp granule from
    colliding.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(config.provenance.dir) / f"{name}.{ts}.{os.getpid()}.run.json"
