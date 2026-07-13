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
    sub_results: "dict[str, StepResult] | None" = None

    @property
    def success(self) -> bool:
        """Whether the step exited with return code 0."""
        return self.returncode == 0

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
            raise AttributeError(
                f"StepResult for '{self.__dict__.get('name')}' has no attribute '{name}'"
            ) from None
