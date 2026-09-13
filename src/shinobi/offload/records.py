"""Per-attempt observations, separate from the frozen plan and scheduler state.

Only a final, atomically published record commits a result. A scheduler's
COMPLETED state, an absent file or a stale running record cannot do so.
This is a transport contract, not the shared cache journal or a lease.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import ConfigDict, Field, JsonValue, model_validator

from shinobi.cache import ProvenanceKey
from shinobi.offload._codec import BundleError, WireModel, pack_model, unpack
from shinobi.offload.bundle import write_new
from shinobi.provenance import StepRecord, _record
from shinobi.results import StepResult
from shinobi.steps.schema import Scope


class ProducedState(WireModel):
    cache_key: str
    producer_field: str


class Observation(StepRecord):
    """Provenance metadata, with lossless tagged I/O instead of display JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    inputs: dict[str, JsonValue]
    outputs: dict[str, JsonValue]

    @model_validator(mode="after")
    def _values(self) -> Observation:
        for values in (self.inputs, self.outputs):
            for value in values.values():
                unpack(value)
        return self


class AttemptRecord(WireModel):
    schema_version: Literal[1] = 1
    workflow_id: UUID
    attempt_id: UUID
    step_path: str
    bundle_digest: str
    state: Literal["running", "succeeded", "failed", "cached", "skipped", "unknown"]
    observation: Observation | None = None
    stdout: str = ""
    stderr: str = ""
    cache_key: str | None = None
    output_keys: dict[str, ProducedState] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_outcome(self) -> AttemptRecord:
        result = self.observation
        if self.state in ("running", "unknown"):
            if result is not None or self.cache_key is not None or self.output_keys:
                raise BundleError("an unfinished/unknown attempt cannot publish a result or produced state")
        else:
            if result is None or result.steps:
                raise BundleError("a final worker record needs one leaf observation")
            if result.name != self.step_path:
                raise BundleError("observation name does not match the logical step_path")
            expected = "failed" if result.returncode else "skipped" if result.skipped else "cached" if result.cached else "succeeded"
            if self.state != expected:
                raise BundleError(f"attempt state {self.state!r} disagrees with observation ({expected})")
            if self.state == "failed" and (self.cache_key is not None or self.output_keys):
                raise BundleError("a failed attempt cannot publish reusable produced state")
        return self

    @property
    def committed(self) -> bool:
        return self.state in ("succeeded", "cached", "skipped")

    @classmethod
    def from_result(cls, result: StepResult, *, workflow_id: UUID, attempt_id: UUID,
                    step_path: str, bundle_digest: str) -> AttemptRecord:
        inputs, outputs = pack_model(result.inputs), pack_model(result.outputs)
        # Reuse provenance's metadata mapping, but not its lossy I/O payload.
        metadata = _record(result, name=step_path).model_dump(exclude={"inputs", "outputs"})
        keys = {}
        if result.success:
            for field in type(result.outputs).model_fields:
                key = result.provenance_key(field)
                if key is not None:
                    keys[field] = ProducedState(cache_key=str(key), producer_field=getattr(key, "producer_field", None) or field)
        state = "failed" if not result.success else "skipped" if result.skipped else "cached" if result.cached else "succeeded"
        return cls(workflow_id=workflow_id, attempt_id=attempt_id, step_path=step_path, bundle_digest=bundle_digest,
                   state=state, observation=Observation(inputs=inputs, outputs=outputs, **metadata),
                   stdout=result.stdout, stderr=result.stderr, cache_key=result.cache_key if result.success else None, output_keys=keys)

    def result(self, scope: Scope) -> StepResult:
        """Restore validated outputs and their *original* producing-field keys."""
        if self.observation is None:
            raise BundleError("attempt has no final worker result")
        record = self.observation
        metadata = {name: getattr(record, name) for name in Observation.model_fields if name not in {"steps", "inputs", "outputs", "name", "returncode"}}
        return StepResult(name=record.name, returncode=record.returncode, inputs=scope.inputs_model.model_validate({k: unpack(v) for k, v in record.inputs.items()}),
                          outputs=scope.outputs_model.model_validate({k: unpack(v) for k, v in record.outputs.items()}), stdout=self.stdout, stderr=self.stderr,
                          cache_key=self.cache_key, output_keys={f: ProvenanceKey(v.cache_key, v.producer_field) for f, v in self.output_keys.items()}, **metadata)

    def write(self, directory: Path) -> Path:
        """Publish once per identity and phase; logical step names aren't paths."""
        snapshot = type(self).model_validate_json(self.model_dump_json())
        phase = "started" if self.state == "running" else "unknown" if self.state == "unknown" else "final"
        return write_new(directory / str(self.workflow_id) / "attempts" / str(self.attempt_id) / f"{phase}.json", snapshot)

    @classmethod
    def read(cls, path: Path, *, workflow_id: UUID, attempt_id: UUID, step_path: str, bundle_digest: str) -> AttemptRecord:
        record = cls.model_validate_json(path.read_text(encoding="utf-8"))
        expected = (workflow_id, attempt_id, step_path, bundle_digest)
        if (record.workflow_id, record.attempt_id, record.step_path, record.bundle_digest) != expected:
            raise BundleError("attempt record does not belong to the requested workflow, attempt, step and bundle")
        return record
