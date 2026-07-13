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
from shinobi.results import StepResult

if TYPE_CHECKING:
    from shinobi.config import AppConfig


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
    backend: str | None = None
    image: str | None = None
    image_digest: str | None = None
    containerized: bool = False
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    steps: list["StepRecord"] = []


class RunManifest(BaseModel):
    """The whole run, frozen. `stimela_version` and `cab_repo_commit` are
    nullable and only populated once those provenance sources are wired --
    they are honest nulls, never fabricated.
    """

    schema_version: int = 1
    shinobi_version: str
    stimela_version: str | None = None
    cab_repo_commit: str | None = None
    generated_at: datetime
    backend: str
    returncode: int
    root: StepRecord

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pinned(self) -> bool:
        """False if any step that ran *inside a container* lacks a resolved
        image digest. Keyed on `containerized` (not the backend name) so
        Slurm-under-apptainer counts, and a native cab whose image is mere
        metadata does not.
        """

        def ok(record: StepRecord) -> bool:
            self_ok = (not record.containerized) or record.image_digest is not None
            return self_ok and all(ok(child) for child in record.steps)

        return ok(self.root)

    def write(self, path: Path) -> Path:
        """Serialize the manifest to `path` (parent dirs created), returning it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path


def _record(result: StepResult) -> StepRecord:
    """Walk a `StepResult` (and its `sub_results`) into a `StepRecord` tree."""
    return StepRecord(
        name=result.name,
        kind=result.kind,
        returncode=result.returncode,
        cached=result.cached,
        backend=result.backend,
        image=result.image,
        image_digest=result.image_digest,
        containerized=result.containerized,
        inputs=_jsonable(result.inputs),
        outputs=_jsonable(result.outputs),
        steps=[_record(sub) for sub in (result.sub_results or {}).values()],
    )


def build_manifest(result: StepResult, *, backend: str) -> RunManifest:
    """Build a `RunManifest` freezing a completed top-level run's `result`."""
    return RunManifest(
        shinobi_version=_shinobi_version,
        generated_at=datetime.now(timezone.utc),
        backend=backend,
        returncode=result.returncode,
        root=_record(result),
    )


def run_manifest_path(config: "AppConfig", name: str) -> Path:
    """Destination for a run's manifest: `{provenance.dir}/{name}.{utc}.{pid}.run.json`.
    The pid keeps two runs of the same recipe in one timestamp granule from
    colliding.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(config.provenance.dir) / f"{name}.{ts}.{os.getpid()}.run.json"
