"""Contained FITS lifecycle using shared ownership and exact mutation snapshots.

FITS observations carry full physical file byte identity independently of the
selected HDU structural contract, skip-cache key, and named journal lineage.
"""

from __future__ import annotations
import hashlib
from pathlib import Path
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shinobi.dataset_access import DatasetMode
from shinobi.dataset_lifecycle import (
    DatasetLifecycle,
    DatasetLifecycleStore,
    DatasetLifecyclePhase,
    DatasetLifecycleEvent,
    DatasetCacheDecision,
    DatasetMutationOutcome,
    _MissingDatasetAttempt,
    _claimed_for,
)
from shinobi.dataset_backends import DatasetBackendStatus, DatasetNamespaceMode
from shinobi.exceptions import DatasetLifecycleUnavailableError, DatasetLifecycleViolationError
from shinobi.fits import FITSDescriptor, inspect_fits
from shinobi.fits_commit import FITSCachePublication, FITSCommitReference, FITSParticipant
from shinobi.fits_access import ResolvedFITSAccess, fits_fields, resolve_scope_fits_accesses
from shinobi.ownership import WorkspaceOwner
from shinobi.snapshots import StateIdentity
from shinobi.storage import SharedStorageError

FITS_BYTE_PROFILE = "fits-file-bytes/v1"


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FITSBackendCapability(_Record):
    profile: Literal["fits-backend-route/v1"] = "fits-backend-route/v1"
    backend: str
    lifecycle: Literal["contained-fits-read/v1", "contained-fits-mutation/v1"]
    status: DatasetBackendStatus
    namespace_mode: DatasetNamespaceMode
    reason: str


def fits_backend_capability(backend, *, mutation=False):
    supported = backend in ("native", "venv")
    return FITSBackendCapability(
        backend=backend,
        lifecycle="contained-fits-mutation/v1" if mutation else "contained-fits-read/v1",
        status=DatasetBackendStatus.TESTED if supported else DatasetBackendStatus.UNSUPPORTED,
        namespace_mode=DatasetNamespaceMode.LOCAL if supported else DatasetNamespaceMode.UNPROVEN,
        reason="controller and native/venv executor share local namespace" if supported else "FITS containers and detached/offloaded routes are not qualified",
    )


class FITSObservation(_Record):
    root: Path
    byte_profile: Literal["fits-file-bytes/v1"] = FITS_BYTE_PROFILE
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size: int = Field(ge=0)
    descriptor: FITSDescriptor


class FITSSnapshot(_Record):
    accesses: tuple[ResolvedFITSAccess, ...]
    observations: tuple[FITSObservation, ...]
    absent_roots: tuple[Path, ...] = ()


class FITSMutationRecord(_Record):
    field: str
    element_index: int | None = Field(default=None, ge=0, strict=True)
    mode: DatasetMode
    root: Path
    profile: Literal["fits-file-bytes/v1"] = FITS_BYTE_PROFILE
    predecessor_state: str | None = None
    predecessor_signature: str | None = None
    predecessor_fingerprint: str | None = None
    predecessor_snapshot: Path | None = None
    successor_state: str
    successor_signature: str | None = None
    successor_fingerprint: str | None = None
    successor_snapshot: Path | None = None
    outcome: DatasetMutationOutcome = DatasetMutationOutcome.PENDING
    restored_state: str | None = None


class FITSLeafAttempt(_Record):
    step_path: str
    accesses: tuple[ResolvedFITSAccess, ...]
    cache: DatasetCacheDecision | None = None
    publication: FITSCachePublication | None = None
    pre_observations: tuple[FITSObservation, ...] = ()
    post_observations: tuple[FITSObservation, ...] = ()
    absent_before: tuple[Path, ...] = ()
    mutations: tuple[FITSMutationRecord, ...] = ()
    outcome: Literal["pending", "reused", "validated", "committed", "failed", "refused"]
    reason: str


class FITSLifecycleAttempt(_Record):
    schema_version: Literal[2] = 2
    profile: Literal["fits-lifecycle/v1"] = "fits-lifecycle/v1"
    attempt_id: str
    scope: str
    workspace: Path
    phase: DatasetLifecyclePhase
    events: tuple[DatasetLifecycleEvent, ...]
    backends: tuple[str, ...]
    backend_capabilities: tuple[FITSBackendCapability, ...] = ()
    capability: Literal["contained-fits-read/v1", "contained-fits-mutation/v1"]
    capability_supported: bool
    planned_accesses: tuple[ResolvedFITSAccess, ...] = ()
    claim: WorkspaceOwner | None = None
    pre_observations: tuple[FITSObservation, ...] = ()
    post_observations: tuple[FITSObservation, ...] = ()
    absent_roots: tuple[Path, ...] = ()
    leaves: tuple[FITSLeafAttempt, ...] = ()
    recovery: tuple[str, ...] = ()
    outcome: Literal["pending", "committed", "refused", "failed"] = "pending"
    reason: str

    @model_validator(mode="after")
    def _consistent(self):
        # Shared phase-machine rules, with independently closed FITS evidence.
        if not self.events or self.events[-1].phase != self.phase or self.events[0].phase != DatasetLifecyclePhase.PLANNED or self.reason != self.events[-1].reason:
            raise ValueError("invalid FITS phase evidence")
        allowed = {
            "planned": {"planned", "claimed", "refused"},
            "claimed": {"revalidated", "refused"},
            "revalidated": {"executing", "refused"},
            "executing": {"validated", "failed"},
            "validated": {"committed", "failed"},
            "committed": set(),
            "failed": set(),
            "refused": set(),
        }
        if any(current.phase.value not in allowed[prior.phase.value] for prior, current in zip(self.events, self.events[1:])):
            raise ValueError("invalid FITS phase transition")
        expected = self.phase.value if self.phase.value in ("committed", "failed", "refused") else "pending"
        if self.outcome != expected:
            raise ValueError("FITS outcome disagrees with phase")
        return self


class FITSLifecycleStore(DatasetLifecycleStore):
    record_model = FITSLifecycleAttempt


class FITSLifecycle(DatasetLifecycle):
    record_model = FITSLifecycleAttempt
    kind = "fits"
    success_kind = "fits-lifecycle"

    @property
    def mutation(self):
        return self.record.capability == "contained-fits-mutation/v1"

    @classmethod
    def start(cls, *, workspace, attempt_id, scope, backends, mutation=False):
        reason = "strict FITS lifecycle planning started"
        record = FITSLifecycleAttempt(
            attempt_id=attempt_id,
            scope=scope,
            workspace=workspace,
            phase=DatasetLifecyclePhase.PLANNED,
            events=(DatasetLifecycleEvent(phase=DatasetLifecyclePhase.PLANNED, observed_at=time.time(), reason=reason),),
            backends=backends,
            capability="contained-fits-mutation/v1" if mutation else "contained-fits-read/v1",
            capability_supported=False,
            reason=reason,
        )
        store = FITSLifecycleStore(workspace / ".shinobi" / "fits-attempts" / f"{attempt_id}.json")
        store.create(record)
        return cls(store, record, workspace=workspace)

    def resolve_claimed_leaf_accesses(self, scope, prepared, **kwargs):
        return resolve_claimed_leaf_accesses(self, scope, prepared, **kwargs)


def fits_file_identity(path, workspace=None):
    """Full streaming SHA256; durable recovery requires no Astropy import."""
    from shinobi.derived import _check_path

    path = Path(path)
    _check_path(path)
    before = path.stat()
    if not path.is_file() or before.st_nlink != 1:
        raise DatasetLifecycleUnavailableError(f"FITS byte identity refuses non-file or hardlinks: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise DatasetLifecycleUnavailableError(f"FITS file changed while byte hashing: {path}")
    return StateIdentity(FITS_BYTE_PROFILE, digest.hexdigest())


def observe_access(access):
    before = fits_file_identity(access.root)
    descriptor = inspect_fits(access.root, access.expected, hdu=access.declaration.hdu)
    if not descriptor.valid:
        raise DatasetLifecycleUnavailableError(f"{access.field}: {descriptor.status.value}: {descriptor.message}")
    after = fits_file_identity(access.root)
    if after != before:
        raise DatasetLifecycleUnavailableError(f"FITS file changed during structural inspection: {access.root}")
    return FITSObservation(root=access.root, sha256=after.fingerprint, size=access.root.stat().st_size, descriptor=descriptor)


def dataset_observation_matches(before, after):
    return before == after


def dataset_snapshot_matches(before, after):
    return before == after


def observe_roots(roots, workspace):
    # The controller supplies the qualified accesses to preserve selectors.
    raise RuntimeError("FITS observe_roots requires explicit access declarations")


def current_accesses(accesses):
    """Current whole-file contracts follow the latest writer, not ancestors.

    Individual leaves still validate every selected HDU immediately before
    execution and before committing. Readers cannot change the file lineage.
    """
    selected = {}
    for access in accesses:
        if access.root not in selected or access.writes:
            selected[access.root] = access
    return tuple(selected.values())


def observe_accesses(accesses):
    observations, absent, seen = [], [], set()
    for access in accesses:
        if access.root in seen:
            continue
        seen.add(access.root)
        if not access.root.exists():
            absent.append(access.root)
        else:
            observations.append(observe_access(access))
    return tuple(observations), tuple(absent)


def committed_generation(generation, root, *, workspace):
    """Reopen the exact S2 reference; only matching durable S3 proves commit."""
    if generation is None or generation.fits_commit is None:
        raise DatasetLifecycleUnavailableError("FITS producer generation lacks an exact durable commit reference")
    reference = FITSCommitReference.model_validate(generation.fits_commit)
    own = reference.participant
    if (
        own.root != root
        or own.successor != generation.name
        or own.fingerprint != generation.content_fingerprint
        or own.signature != generation.structural_signature
        or generation.identity_profile != FITS_BYTE_PROFILE
        or reference.oracle != workspace / ".shinobi" / "fits-attempts" / f"{reference.attempt_id}.json"
    ):
        raise DatasetLifecycleUnavailableError("FITS commit reference contradicts generation or intended oracle path")
    try:
        record = FITSLifecycleStore(reference.oracle).attempt()
    except (OSError, ValueError, SharedStorageError, DatasetLifecycleUnavailableError) as exc:
        raise DatasetLifecycleUnavailableError("FITS exact commit oracle is unreadable or malformed") from exc
    if record.attempt_id != reference.attempt_id or record.workspace != workspace:
        raise DatasetLifecycleUnavailableError("FITS commit reference contradicts durable attempt identity")
    leaves = [
        leaf for leaf in record.leaves if leaf.step_path == reference.step_path and leaf.outcome == "committed" and leaf.cache and leaf.cache.cache_key == reference.cache_key
    ]
    for leaf in leaves:
        participants = tuple(
            FITSParticipant(
                root=one.root,
                field=one.field.removesuffix(f"[{one.element_index}]"),
                element_index=one.element_index,
                label=one.field,
                successor=one.successor_state,
                signature=one.successor_signature,
                fingerprint=one.successor_fingerprint,
                mode=one.mode.value,
            )
            for one in leaf.mutations
            if one.outcome is DatasetMutationOutcome.COMMITTED
        )
        if set(participants) != set(reference.participants) or len(participants) != len(leaf.mutations):
            continue
        if any(not any(observation.root == one.root and observation.sha256 == one.fingerprint for observation in leaf.post_observations) for one in participants):
            continue
        if leaf.publication is not None and (leaf.publication.cache_key != reference.cache_key or leaf.publication.run_id != reference.attempt_id):
            raise DatasetLifecycleUnavailableError("FITS durable publication payload contradicts exact commit")
        return reference, leaf
    raise DatasetLifecycleUnavailableError("FITS commit reference lacks exact complete committed participant evidence")


def observe_journal_head(root, chain, *, workspace, cache_dir, record=None):
    """Revalidate the exact referenced committed generation's selected contract."""
    head = chain.generation(chain.head)
    identity = fits_file_identity(root)
    if not head or head.identity_profile != FITS_BYTE_PROFILE or identity.fingerprint != head.content_fingerprint:
        raise DatasetLifecycleUnavailableError("FITS live bytes differ from trusted journal head")
    _, leaf = committed_generation(head, root, workspace=workspace)
    for access in leaf.accesses:
        if access.root == root and access.writes:
            observed = observe_access(access)
            if observed in leaf.post_observations:
                return observed
    raise DatasetLifecycleUnavailableError("FITS journal head lacks exact durable structural evidence")


def observe_current_accesses(accesses, lifecycle, cache_dir):
    from shinobi.snapshots import get_journal, chain_id, HeadStatus

    observations, absent = [], []
    chains = get_journal(cache_dir).all_chains()
    for access in current_accesses(accesses):
        if not access.root.exists():
            absent.append(access.root)
            continue
        chain = chains.get(chain_id(access.root))
        if access.writes and chain and chain.status is HeadStatus.TRUSTED and chain.head and not chain.marker and not chain.head.startswith("gen0__"):
            observations.append(observe_journal_head(access.root, chain, workspace=lifecycle.workspace, cache_dir=cache_dir, record=lifecycle.record))
        else:
            observations.append(observe_access(access))
    return tuple(observations), tuple(absent)


def resolve_lifecycle_snapshot(scope, values, *, workspace, validated_steps=None, mutation=False, cache_dir=None):
    from shinobi.ownership import _resolved_leaf_inputs

    accesses = []
    creators = {}
    for leaf, known, _ in _resolved_leaf_inputs(scope, values, step_inputs={}, validated_steps=validated_steps):
        resolved = resolve_scope_fits_accesses(leaf, known, workspace=workspace)
        step_names = {scope.name + "." + ref.name for ref in getattr(scope, "steps", ()) if ref.step is leaf} or {scope.name}
        for one in resolved:
            if one.mode is DatasetMode.CREATE:
                creators[one.root] = (one, step_names)
        accesses.extend(resolved)
    if not accesses:
        raise DatasetLifecycleUnavailableError("strict FITS requires at least one concrete access; all-None invocation is unsupported")
    creates = {one.root for one in accesses if one.mode is DatasetMode.CREATE}
    from shinobi.snapshots import get_journal, chain_id, HeadStatus, strict_reuse_issue

    chains = get_journal(cache_dir).all_chains() if cache_dir else {}
    pending_writes = {
        one.root
        for one in accesses
        if one.writes and (chain := chains.get(chain_id(one.root))) is not None and chain.marker is not None and chain.marker.identity_profile == FITS_BYTE_PROFILE
    }
    for root in creates:
        if root.exists() or root.is_symlink():
            chain = chains.get(chain_id(root))
            access, names = creators[root]
            marker = chain.marker if chain else None
            pending_create = (
                marker is not None and marker.identity_profile == FITS_BYTE_PROFILE and marker.path_was_absent and marker.step_path in names and marker.field == access.field
            )
            owned_create = False
            if not pending_create and chain and chain.head:
                for generation in chain.generations:
                    if generation.fits_commit is None:
                        continue
                    reference, leaf = committed_generation(generation, root, workspace=workspace)
                    if (
                        reference.step_path in names
                        and reference.participant.field == access.declaration.field
                        and reference.participant.mode == "create"
                        and strict_reuse_issue(get_journal(cache_dir), root, generation.name, fits_file_identity(root)) is None
                    ):
                        owned_create = True
                        break
            head = chain.generation(chain.head) if chain and chain.head else None
            if not pending_create and (
                not owned_create
                or not chain
                or chain.status is not HeadStatus.TRUSTED
                or not head
                or head.identity_profile != FITS_BYTE_PROFILE
                or fits_file_identity(root).fingerprint != head.content_fingerprint
            ):
                raise DatasetLifecycleUnavailableError(f"FITS CREATE target already exists without exact prior CREATE authority: {root}")
    for one in accesses:
        if not mutation and one.writes:
            raise DatasetLifecycleUnavailableError("FITS writer requires mutation recovery")
    roots = sorted({one.root for one in accesses}, key=str)
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise DatasetLifecycleUnavailableError("overlapping FITS lifecycle roots")
    observations = []
    seen = set()
    for one in accesses:
        chain = chains.get(chain_id(one.root))
        head = chain.generation(chain.head) if chain and chain.head else None
        if one.root not in creates and one.root not in pending_writes and one.root not in seen:
            if head and head.name.startswith("gen0__") and head.identity_profile == FITS_BYTE_PROFILE and chain.status is HeadStatus.TRUSTED and not chain.marker:
                if fits_file_identity(one.root).fingerprint != head.content_fingerprint:
                    raise DatasetLifecycleUnavailableError("FITS boundary bytes differ from trusted generation zero")
            if (
                mutation
                and any(access.root == one.root and access.writes for access in accesses)
                and head
                and chain.status is HeadStatus.TRUSTED
                and head.identity_profile == FITS_BYTE_PROFILE
                and not head.name.startswith("gen0__")
                and not chain.marker
            ):
                observations.append(observe_journal_head(one.root, chain, workspace=workspace, cache_dir=cache_dir))
            else:
                observations.append(observe_access(one))
            seen.add(one.root)
    return FITSSnapshot(accesses=tuple(accesses), observations=tuple(observations), absent_roots=tuple(sorted(creates, key=str)))


def resolve_claimed_leaf_accesses(lifecycle, scope, prepared, **kwargs):
    accesses = resolve_scope_fits_accesses(scope, prepared, workspace=lifecycle.workspace)
    planned = {(one.address.field, one.element_index, one.root, one.mode, one.expected, one.declaration.hdu) for one in lifecycle.record.planned_accesses}
    for one in accesses:
        if (one.address.field, one.element_index, one.root, one.mode, one.expected, one.declaration.hdu) not in planned or not _claimed_for(
            lifecycle.record.claim, one.root, one.writes
        ):
            raise DatasetLifecycleUnavailableError("FITS leaf access differs from claimed workflow plan")
    return accesses


def mutation_committed(path, *, attempt_id, step_path, cache_key, participants=None, strict=False):
    store = FITSLifecycleStore(path)
    try:
        record = store.attempt()
    except _MissingDatasetAttempt as exc:
        if strict and (store.path.exists() or store.lock_path.exists()):
            raise DatasetLifecycleUnavailableError("strict FITS success oracle exists without an attempt record") from exc
        return False
    except (OSError, ValueError, SharedStorageError, DatasetLifecycleUnavailableError) as exc:
        if strict:
            raise DatasetLifecycleUnavailableError("strict FITS success oracle is unreadable or malformed") from exc
        return False
    from shinobi.snapshots import _validate_group_evidence

    leaves = [one for one in record.leaves if one.step_path == step_path and one.outcome == "committed"]
    matching = [one for one in leaves if record.attempt_id == attempt_id and cache_key is not None and one.cache is not None and one.cache.cache_key == cache_key]
    complete = any(participants is None or _validate_group_evidence(one, participants) for one in matching)
    if strict and participants is not None and leaves and not complete:
        raise DatasetLifecycleUnavailableError("FITS success oracle lacks exact matching participant evidence")
    return complete


def committed_oracle_requires_reference(marker):
    """Distinguish new committed payloads from genuine legacy FITS oracles."""
    record = FITSLifecycleStore(Path(marker.success_record)).attempt()
    matches = [
        leaf
        for leaf in record.leaves
        if leaf.step_path == (marker.success_step_path or marker.step_path) and leaf.outcome == "committed" and leaf.cache and leaf.cache.cache_key == marker.cache_key
    ]
    if record.attempt_id != marker.run_id or not matches:
        raise DatasetLifecycleUnavailableError("FITS completed oracle evidence changed during recovery")
    return any(leaf.publication is not None for leaf in matches)


class StrictFITSLeaf:
    def __init__(self, lifecycle, step_path, scope, prepared):
        self.lifecycle, self.step_path, self.scope, self.prepared = lifecycle, step_path, scope, prepared
        self.accesses = resolve_claimed_leaf_accesses(lifecycle, scope, prepared)
        self.fields = {one.field: one.root for one in self.accesses if one.writes}
        self.addresses = {one.field: one.address for one in self.accesses if one.writes}
        self.creates = frozenset(one.field for one in self.accesses if one.mode is DatasetMode.CREATE)
        self.pre, self.absent, self.mutations, self.cache, self.cache_key = {}, (), {}, None, None

    @property
    def writes(self):
        return bool(self.fields)

    def policy(self):
        from shinobi.snapshots import StrictMutation

        return StrictMutation(identity=fits_file_identity, create_fields=self.creates, profiles=dict.fromkeys(self.fields, FITS_BYTE_PROFILE))

    def select_derived_states(self, journal):
        return None

    def decide_reuse(self, cache_key, input_keys, journal):
        from shinobi.snapshots import state_name, strict_reuse_issue

        self.cache_key = cache_key
        issues = []
        for field, root in self.fields.items():
            try:
                issue = strict_reuse_issue(journal, root, state_name(cache_key, self.addresses[field].field, self.addresses[field].element_index), fits_file_identity(root))
                if issue:
                    issues.append(f"{field}: {issue}")
            except (OSError, ValueError, DatasetLifecycleUnavailableError) as exc:
                issues.append(f"{field}: {exc}")
        self.cache = DatasetCacheDecision(
            decision="rejected-hit" if issues else "hit",
            cache_key=cache_key,
            identity="skip-cache key",
            dataset_coverage=("fits-file-bytes/v1 whole-file SHA256 and journal lineage",),
            reason="; ".join(issues) or "journal vouches for live descendant generation",
        )
        self._record("pending" if issues else "reused", self.cache.reason)
        return not issues

    def decide_run(self, cache_key, input_keys, *, cacheable, automatic=False):
        self.cache_key = cache_key
        if self.cache is None:
            self.cache = DatasetCacheDecision(
                decision="miss" if cacheable else "disabled",
                cache_key=cache_key,
                identity="skip-cache key",
                dataset_coverage=("full file bytes tracked independently in snapshot journal",),
                reason="no reusable entry",
            )

    def freeze_publication(self, result, run_id, contract, products):
        from shinobi.cache import cache_entry

        self.publication = FITSCachePublication.model_validate(cache_entry(self.cache_key, result, run_id, product_contract=contract, products=products))

    def repair_cache(self, manifest, cache_key, scope, prepared, *, defer_fits_fields=()):
        """Restore only this exact committed payload, then use ordinary checks."""
        from shinobi.snapshots import get_journal, state_name, chain_id, strict_reuse_issue

        journal = get_journal(str(manifest.path.parent))
        payloads = []
        for label, root in self.fields.items():
            chain = journal.get(chain_id(root))
            name = state_name(cache_key, self.addresses[label].field, self.addresses[label].element_index)
            generation = chain.generation(name) if chain else None
            if generation is None or generation.fits_commit is None:
                return
            if strict_reuse_issue(journal, root, name, fits_file_identity(root)):
                return
            reference, leaf = committed_generation(generation, root, workspace=self.lifecycle.workspace)
            if reference.cache_key != cache_key or reference.step_path != self.step_path or leaf.publication is None:
                return
            payloads.append(leaf.publication.model_dump(mode="json"))
        if not payloads or any(payload != payloads[0] for payload in payloads):
            return
        payload = payloads[0]
        if manifest.check(self.step_path, cache_key, scope, prepared, candidate=payload, defer_fits_fields=defer_fits_fields) is None:
            return
        existing = manifest.entry(self.step_path)
        if existing is not None and (existing.get("cache_key") != cache_key or existing.get("run_id") == payload["run_id"]):
            return

        def repair(data):
            current = data.get(self.step_path)
            if current != existing:
                raise DatasetLifecycleUnavailableError("FITS cache entry changed during exact index repair")
            data[self.step_path] = payload

        # Publication is recovered as a cache candidate, never as a fabricated
        # result. Normal contract/product checks below still decide reuse.
        manifest.update(repair)

    def deferred_output_fields(self):
        inputs, outputs = fits_fields(self.scope.inputs_model), fits_fields(self.scope.outputs_model)
        return {access.declaration.field for access in self.accesses if access.writes and access.declaration.field in inputs and access.declaration.field in outputs}

    def materialize_reads(self, cache_dir, input_keys, cache_key, run_id, force_copy=False):
        """Temporarily visit exact producer states under existing write authority."""
        from shinobi.snapshots import SnapshotGuard, StrictMutation, get_journal, chain_id, state_name
        from shinobi.cache import ProvenanceKey

        journal = get_journal(cache_dir)
        fields, addresses, predecessors = {}, {}, {}
        for access in self.accesses:
            if access.writes:
                continue
            raw = (input_keys or {}).get(access.declaration.field)
            key = raw[access.element_index] if isinstance(raw, list) and access.element_index is not None else raw
            if key is None:
                continue
            if not isinstance(key, ProvenanceKey) or key.producer_field is None:
                raise DatasetLifecycleUnavailableError("FITS wired READ producer key names no exact state")
            required = state_name(str(key), key.producer_field, key.element_index)
            chain = journal.get(chain_id(access.root))
            if chain is None or chain.generation(required) is None:
                raise DatasetLifecycleUnavailableError("FITS wired READ producer state lacks journal evidence")
            if chain.head == required:
                # No physical visit is necessary; normal READ validation still applies.
                continue
            if not self.lifecycle.mutation or not _claimed_for(self.lifecycle.record.claim, access.root, True):
                raise DatasetLifecycleUnavailableError("FITS producer-state visit requires existing workflow exclusive authority")
            fields[access.field], addresses[access.field], predecessors[access.field] = access.root, access.address, required
        if not fields:
            return
        guard = SnapshotGuard(
            journal=journal,
            step_path=self.step_path,
            cache_key=cache_key or hashlib.sha256((run_id + self.step_path).encode()).hexdigest(),
            run_id=run_id,
            fields=fields,
            input_keys=input_keys,
            wired_fields=set(input_keys or ()),
            addresses=addresses,
            force_copy=force_copy,
            strict=StrictMutation(identity=fits_file_identity, profiles=dict.fromkeys(fields, FITS_BYTE_PROFILE), predecessors=predecessors),
            success_record=self.lifecycle.workspace / ".shinobi" / "fits-read-visits" / f"{run_id}-{hashlib.sha256(self.step_path.encode()).hexdigest()}.json",
            success_kind="fits-lifecycle",
        )
        self.read_visit = guard
        guard.before_run()

    def finish_read_visit(self):
        guard = getattr(self, "read_visit", None)
        if guard is not None:
            from shinobi.snapshots import HeadStatus

            guard.after_failure()
            # after_failure deliberately retains fences and returns when a
            # strict group restore fails. A reader may publish only after
            # positive evidence that every participant is back at its exact
            # frozen pre-visit generation; guard presence alone is not proof.
            for plan in guard.plans:
                chain = guard.journal.get(plan.cid)
                generation = chain.generation(plan.rollback_state) if chain else None
                if (
                    plan.outcome not in ("pre-run-restored", "rolled-back")
                    or plan.marked
                    or chain is None
                    or chain.marker is not None
                    or chain.status is not HeadStatus.TRUSTED
                    or chain.head != plan.rollback_state
                    or generation is None
                    or generation.identity_profile != FITS_BYTE_PROFILE
                    or plan.rollback_identity is None
                    or generation.content_fingerprint != plan.rollback_identity.fingerprint
                    or generation.structural_signature != plan.rollback_identity.signature
                    or fits_file_identity(plan.path) != plan.rollback_identity
                ):
                    raise DatasetLifecycleUnavailableError("FITS reader visit did not restore every exact pre-visit generation; publication refused")
            self.restored_visit_roots = frozenset(plan.path for plan in guard.plans)
            self.read_visit = None

    def observe_before(self, guard):
        from shinobi.snapshots import state_name

        observations, absent = observe_accesses(self.accesses)
        self.pre = {one.root: one for one in observations}
        self.absent = absent
        if set(absent) - {self.fields[field] for field in self.creates}:
            raise DatasetLifecycleUnavailableError("FITS read/write root absent before launch")
        plans = {one.field: one for one in guard.plans} if guard else {}
        for field, root in self.fields.items():
            plan = plans[field]
            predecessor = fits_file_identity(root) if root.exists() else None
            self.mutations[field] = FITSMutationRecord(
                field=field,
                element_index=self.addresses[field].element_index,
                root=root,
                mode=DatasetMode.CREATE if field in self.creates else DatasetMode.WRITE,
                predecessor_state=plan.required,
                predecessor_signature=predecessor.signature if predecessor else None,
                predecessor_fingerprint=predecessor.fingerprint if predecessor else None,
                predecessor_snapshot=guard.journal.snapshot_dir(plan.required) if plan.required else None,
                successor_state=state_name(self.cache_key, self.addresses[field].field, self.addresses[field].element_index),
            )
        self._record("pending", "predecessors observed after restore")

    def validate(self, guard, outputs=None):
        observed = []
        issues = []
        for access in self.accesses:
            try:
                post = observe_access(access)
                observed.append(post)
                if not access.writes and post != self.pre.get(access.root):
                    issues.append(f"{access.field}: read-only whole file changed")
            except (OSError, ValueError, DatasetLifecycleUnavailableError) as exc:
                issues.append(str(exc))
        if outputs is not None:
            for name, shape in fits_fields(self.scope.outputs_model).items():
                if name in fits_fields(self.scope.inputs_model):
                    actual = getattr(outputs, name)
                    expected = self.prepared.get(name)

                    def canonical(value):
                        if value is None:
                            return None
                        members = value if shape.is_list else [value]
                        return tuple((one if one.is_absolute() else self.lifecycle.workspace / one).resolve() for one in members)

                    if canonical(actual) != canonical(expected):
                        issues.append(f"{name}: output must preserve exact ordered input roots")
        if issues:
            raise DatasetLifecycleViolationError("strict FITS postcondition failed: " + "; ".join(issues))
        self._post = tuple(observed)
        for field, root in self.fields.items():
            identity = fits_file_identity(root)
            guard.successor_identities[field] = identity
            record = self.mutations[field]
            self.mutations[field] = record.model_copy(
                update={
                    "successor_signature": identity.signature,
                    "successor_fingerprint": identity.fingerprint,
                    "successor_snapshot": guard.journal.snapshot_dir(record.successor_state),
                }
            )
        visited = getattr(self, "read_visit", None) is not None
        self.finish_read_visit()
        if visited:
            from shinobi.fits_products import validate_fits_outputs

            validate_fits_outputs(self.scope, outputs)
        self._record("validated", "FITS postconditions hold", post=self._post)

    def commit(self):
        self.mutations = {field: one.model_copy(update={"outcome": DatasetMutationOutcome.COMMITTED}) for field, one in self.mutations.items()}
        self._record("committed", "successors snapshotted and journalled; exact success oracle", post=getattr(self, "_post", ()))

    def fail(self, guard, reason, *, refused=False):
        plans = {one.field: one for one in guard.plans} if guard else {}
        for field, one in self.mutations.items():
            plan = plans.get(field)
            outcome = DatasetMutationOutcome.REFUSED if refused else DatasetMutationOutcome(plan.outcome) if plan and plan.outcome != "pending" else one.outcome
            self.mutations[field] = one.model_copy(
                update={
                    "outcome": outcome,
                    "restored_state": one.predecessor_state
                    if outcome is DatasetMutationOutcome.ROLLED_BACK
                    else plan.pre_run_head
                    if plan and outcome is DatasetMutationOutcome.PRE_RUN_RESTORED
                    else None,
                }
            )
        self._record("refused" if refused else "failed", reason, post=getattr(self, "_post", ()))
        try:
            if not refused and self.pre:
                for access in self.accesses:
                    if not access.writes and access.root not in getattr(self, "restored_visit_roots", ()):
                        try:
                            unchanged = observe_access(access) == self.pre.get(access.root)
                        except (OSError, ValueError, DatasetLifecycleUnavailableError):
                            unchanged = False
                        if not unchanged:
                            raise DatasetLifecycleViolationError(f"strict FITS reader changed whole file before failure: {access.root}")
        finally:
            self.finish_read_visit()

    def _record(self, outcome, reason, *, post=()):
        self.lifecycle.record_leaf(
            FITSLeafAttempt(
                step_path=self.step_path,
                accesses=self.accesses,
                cache=self.cache,
                publication=getattr(self, "publication", None),
                pre_observations=tuple(self.pre.values()),
                absent_before=self.absent,
                post_observations=post,
                mutations=tuple(self.mutations.values()),
                outcome=outcome,
                reason=reason,
            )
        )
