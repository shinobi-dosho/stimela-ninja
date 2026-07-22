"""Execution results.

`BackendRun` is the raw, schema-agnostic outcome of a backend actually
running a command: exit status and captured console output, nothing more.
Backends return this; they never see a cab's output schema.

`StepResult` is the schema-aware result the dispatch layer builds on top:
it carries the validated `outputs` and effective `inputs` models plus the
same console output, and is what a step call (cab or recipe) returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class BackendRun:
    """What a backend returns after running a command -- just the raw run
    outcome. Wrangling stdout/stderr into structured outputs and filling
    an `outputs_model` is the dispatch layer's job, not the backend's.
    """

    returncode: int
    stdout: str = ""
    stderr: str = ""
    # Registry digest (``sha256:...``) of the image that actually ran, when
    # the backend could pin it (see ``backends.container._pin_image``);
    # ``None`` for native runs or images that couldn't be resolved to a
    # digest (local-built/untagged, offline, no skopeo).
    image_digest: str | None = None
    # Whether this run actually executed inside a container. True even when
    # ``image_digest`` is None (containerized but unpinnable) -- that pair is
    # exactly what makes a run report ``pinned: false``. Set by every backend
    # that wraps in a container runtime (incl. Slurm-under-apptainer, whose
    # backend *name* isn't a container-runtime name).
    containerized: bool = False
    # The virtualenv this ran in (its path), and a ``sha256`` of its sorted
    # ``name==version`` list -- set only by the ``venv`` backend, and the
    # digest only under ``pin=True``. The digest is a version-parity record,
    # not an OS-level pin, so a venv run is always reported unpinned; these
    # are informational provenance (see ``backends.venv``).
    venv: str | None = None
    venv_digest: str | None = None

    @property
    def success(self) -> bool:
        """Whether the run exited with return code 0."""
        return self.returncode == 0


@dataclass
class StepResult:
    """The outcome of running a step (a Cab or a Recipe).

    `outputs` is a validated instance of the step's `outputs_model`;
    `inputs` is a validated instance of the *effective* (post-override)
    inputs the step actually ran with. For a Recipe these aggregate from
    its sub-steps.
    """

    name: str
    returncode: int
    outputs: BaseModel
    inputs: BaseModel
    stdout: str = ""
    stderr: str = ""
    # True when this result was synthesized from `shinobi.cache` (the step
    # itself never actually ran) rather than produced by a real backend run.
    cached: bool = False
    # True when this step belongs to an unrolled loop iteration that ran
    # after the loop had already converged (see `shinobi.steps.loops`): its
    # `outputs` are the same body step's outputs from the last iteration
    # that really ran. Distinct from `cached` -- nothing was looked up, and
    # distinct from `kind`, which still reports what the step *is*.
    skipped: bool = False
    # Provenance, for the run manifest (see `shinobi.provenance`). `kind` is
    # stamped explicitly at each construction site rather than inferred, so a
    # containerized pystep can never be mistaken for a plain one. `backend`,
    # `image`, and `image_digest` are set for steps that ran a container;
    # `sub_results` holds a recipe's per-step results in declaration order.
    kind: str = "cab"
    backend: str | None = None
    image: str | None = None
    image_digest: str | None = None
    # Whether the step ran inside a container (see BackendRun.containerized).
    # `pinned` requires a digest only for containerized steps, so this -- not
    # the backend name -- is what distinguishes a native cab (image is just
    # metadata) from a Slurm-under-apptainer run that must be pinned.
    containerized: bool = False
    # The venv this step ran in and its version-parity digest (see
    # BackendRun.venv/venv_digest). A step with `venv` set is always reported
    # unpinned in the manifest regardless of `venv_digest`.
    venv: str | None = None
    venv_digest: str | None = None
    # Whether the step ran with sandboxing enabled (see `shinobi.sandbox`).
    # Recorded for diagnostics -- sandbox state affects path anchoring but
    # not the cache key, so output paths are normalized before recording
    # (see `sandbox.relativize_path_outputs`).
    sandboxed: bool = False
    sub_results: "dict[str, StepResult] | None" = None
    # Provenance keys, for the cache's upstream-invalidation term (see
    # `shinobi.cache`). `cache_key` is this step's own key, set by `_dispatch`
    # whenever the step was cacheable; `output_keys` is the per-output-field
    # override a *Recipe* carries, since a recipe is never itself cached and
    # each of its declared outputs is really produced by a different sub-step.
    # Read them through `provenance_key`, never directly.
    cache_key: str | None = None
    output_keys: "dict[str, Any] | None" = None

    @property
    def success(self) -> bool:
        """Whether the step exited with return code 0."""
        return self.returncode == 0

    def provenance_key(self, field: str) -> Any:
        """The cache key identifying whatever produced output `field`.

        A leaf step produces all its outputs in one run, so every field
        resolves to that step's own `cache_key`. A `Recipe` fans out to
        `output_keys` instead -- each declared output comes from a distinct
        sub-step, and invalidating a downstream consumer because some
        *unrelated* sub-step re-ran would throw away most of the cache's
        value.

        `None` means "no provenance available" (caching disabled, or a step
        that isn't cacheable) -- callers must treat that as "contribute
        nothing", not as a key in its own right.
        """
        if self.output_keys is not None:
            return self.output_keys.get(field)
        return self.cache_key

    def __getattr__(self, name: str) -> Any:
        """Read through to `outputs` for convenience (`result.<output_field>`).

        Args:
            name: Attribute name, looked up on `self.outputs` if not a
                dataclass field.

        Returns:
            The corresponding attribute of `self.outputs`.

        Raises:
            AttributeError: If `name` isn't found on `self.outputs` either.
        """
        # convenience: result.<output_field> reads through to outputs
        try:
            return getattr(self.__dict__["outputs"], name)
        except (KeyError, AttributeError):
            raise AttributeError(f"StepResult for '{self.__dict__.get('name')}' has no attribute '{name}'") from None
