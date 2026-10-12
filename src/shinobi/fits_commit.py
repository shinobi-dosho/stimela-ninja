"""Closed references to exact FITS producer evidence, independent of cache indexes."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FITSParticipant(_Closed):
    root: Path
    field: str
    element_index: int | None = Field(default=None, ge=0, strict=True)
    label: str
    successor: str
    signature: Literal["fits-file-bytes/v1"]
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    mode: Literal["write", "create"]

    @model_validator(mode="after")
    def address(self):
        from shinobi.dataset_access import DatasetAddress

        if not self.root.is_absolute() or DatasetAddress(self.field, self.element_index).label != self.label:
            raise ValueError("invalid FITS participant address")
        return self


class FITSCommitReference(_Closed):
    profile: Literal["fits-commit-reference/v1"] = "fits-commit-reference/v1"
    attempt_id: str
    oracle: Path
    step_path: str
    cache_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    participant: FITSParticipant
    participants: tuple[FITSParticipant, ...]

    @model_validator(mode="after")
    def complete(self):
        from shinobi.snapshots import state_name

        if any(one.successor != state_name(self.cache_key, one.field, one.element_index) for one in self.participants):
            raise ValueError("FITS participant successor contradicts cache key/address")
        if not self.participants or self.participant not in self.participants or len({one.root for one in self.participants}) != len(self.participants):
            raise ValueError("invalid FITS invocation participants")
        if not self.attempt_id or Path(self.attempt_id).name != self.attempt_id or not self.oracle.is_absolute():
            raise ValueError("invalid FITS oracle identity")
        return self


class FITSCachePublication(_Closed):
    bundle_inventories: list[dict[str, Any]]
    cache_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    run_id: str
    outputs: dict[str, Any]
    kind: str | None
    backend: str | None
    image: str | None
    image_digest: str | None
    containerized: bool
    venv: str | None
    venv_digest: str | None
    sandboxed: bool
    product_contract: dict[str, Any]
    products: list[str]


def make_commit_reference(guard, plan):
    """Called in the S2 transaction; S3's exact oracle must still prove success."""
    from shinobi.snapshots import state_name

    participants = tuple(
        FITSParticipant(
            root=one.path,
            field=one.address.field,
            element_index=one.address.element_index,
            label=one.field,
            successor=state_name(guard.cache_key, one.address.field, one.address.element_index),
            signature=guard.successor_identities[one.field].signature,
            fingerprint=guard.successor_identities[one.field].fingerprint,
            mode="create" if one.path_was_absent else "write",
        )
        for one in guard.plans
    )
    own = next(one for one in participants if one.root == plan.path)
    return FITSCommitReference(
        attempt_id=guard.run_id,
        oracle=guard.success_record,
        step_path=guard.success_step_path or guard.step_path,
        cache_key=guard.cache_key,
        participant=own,
        participants=participants,
    ).model_dump(mode="json")
