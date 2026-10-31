"""Trusted local-dispatch authority for exact same-workspace CREATE replay.

This is not an overwrite grant: every replay still requires an ordinary valid
cache hit and strict descendant reuse, and a cache miss may never launch CREATE.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
from pathlib import Path

from shinobi.exceptions import DatasetLifecycleUnavailableError

_active = ContextVar("shinobi_required_cache_authority", default=None)


@dataclass(frozen=True)
class CreateReplay:
    """A trusted live replay, or an explicitly provisional frozen source."""

    rollback_source: Path | None = None
    marker: str | None = None


class RequiredCacheAuthority:
    def __init__(self, scope, required, cache_dir, workspace, *, resume_creates=False, overwrite_steps=()):
        from shinobi.steps.schema import Recipe

        self.required = dict(required)
        self.cache_dir = cache_dir
        self.workspace = Path(workspace).resolve()
        self.names = {}
        self.pending = {}
        self.resume_creates = resume_creates
        self.replayed = {}
        self.overwritten = {scope.name + "." + name if isinstance(scope, Recipe) else name for name in overwrite_steps}
        scattered = set()

        def visit(node, path, in_scatter=False):
            if isinstance(node, Recipe):
                for ref in node.steps:
                    visit(ref.step, path + "." + ref.name, in_scatter or ref.scatter is not None)
            else:
                self.names.setdefault(id(node), set()).add(path)
                if in_scatter:
                    scattered.add(path)

        visit(scope, scope.name)
        unknown = self.required.keys() - set().union(set(), *self.names.values())
        if unknown:
            raise DatasetLifecycleUnavailableError("required cache names unknown leaf step(s): " + ", ".join(sorted(unknown)))
        if self.required.keys() & scattered:
            raise DatasetLifecycleUnavailableError("required cache replay does not support scattered leaf instances")

    def preflight(self):
        """Reject absent keys and recorded products before recovery is entered.

        Runtime still computes the key and applies the complete ordinary cache
        contract and strict lineage checks using resolved producer outputs.
        """
        from shinobi.cache import get_cache_manifest
        from shinobi.products import validate_bundle_inventory

        manifest = get_cache_manifest(self.cache_dir)
        for name, key in self.required.items():
            entry = manifest.entry(name)
            if not isinstance(entry, dict) or entry.get("cache_key") != key:
                raise DatasetLifecycleUnavailableError(f"required cache {name!r} has no valid reusable hit for the exact prior key")
            products = entry.get("products", [])
            inventories = entry.get("bundle_inventories", [])
            if (
                not isinstance(products, list)
                or any(not isinstance(path, str) or not Path(path).is_absolute() or not Path(path).exists() for path in products)
                or not isinstance(inventories, list)
                or any(not validate_bundle_inventory(inventory) for inventory in inventories)
            ):
                raise DatasetLifecycleUnavailableError(f"required cache {name!r} has no valid reusable hit: recorded products are absent or invalid")

    @contextmanager
    def activate(self):
        token = _active.set(self)
        try:
            yield
        finally:
            _active.reset(token)

    def replay(self, scope, root, field, element_index):
        """Admit only a verified prior creation; require its key again at dispatch.

        Automatic resume discovers the candidate from the ordinary cache index.
        Explicit required-cache pins remain authoritative. Neither grants a
        cache-miss execution or replacement of the existing dataset.
        """
        from shinobi.cache import get_cache_manifest
        from shinobi.dataset_lifecycle import DatasetMode, DatasetMutationOutcome, dataset_attempt_path, read_dataset_attempt, dataset_identity
        from shinobi.snapshots import get_journal, chain_id, state_name, strict_reuse_issue, _frozen_rollback_source

        root = Path(root)
        self.pending.pop(root, None)
        journal = get_journal(self.cache_dir)
        chain = journal.get(chain_id(root))
        if root.is_symlink() or chain is None:
            return None
        for name in self.names.get(id(scope), ()):
            entry = get_cache_manifest(self.cache_dir).entry(name)
            key = self.required.get(name)
            if key is None and self.resume_creates and name not in self.overwritten and isinstance(entry, dict):
                key = entry.get("cache_key")
            if not key or not entry or entry.get("cache_key") != key or not entry.get("run_id"):
                continue
            record = read_dataset_attempt(dataset_attempt_path(self.workspace, entry["run_id"]))
            if record.workspace != self.workspace or record.attempt_id != entry["run_id"]:
                continue
            state = state_name(key, field, element_index)
            committed = any(
                leaf.step_path == name
                and leaf.outcome == "committed"
                and leaf.cache is not None
                and leaf.cache.cache_key == key
                and any(
                    mutation.mode is DatasetMode.CREATE
                    and mutation.root == root
                    and mutation.field == field
                    and mutation.element_index == element_index
                    and mutation.successor_state == state
                    and mutation.outcome is DatasetMutationOutcome.COMMITTED
                    for mutation in leaf.mutations
                )
                for leaf in record.leaves
            )
            if not committed:
                continue
            if chain.marker is None:
                if strict_reuse_issue(journal, root, state, dataset_identity(root, self.workspace)) is None:
                    self.replayed[name] = key
                    return CreateReplay()
                continue
            # A partial downstream writer is not live evidence. Provisional
            # admission uses only its exact frozen, verified rollback ancestry;
            # normal reconciliation runs later under the workflow's claim.
            marker = chain.marker
            if marker.path_was_absent:
                continue
            if marker.strict_group:
                item = next((item for item in marker.strict_group if item["path"] == str(root)), None)
            else:
                # Historical scalar MS markers froze the same exact source
                # directly on the marker, before joint groups were introduced.
                item = {
                    "path": str(root),
                    "rollback": marker.strict_rollback_state,
                    "rollback_source": marker.strict_rollback_source,
                    "rollback_signature": marker.strict_rollback_signature,
                    "rollback_fingerprint": marker.strict_rollback_fingerprint,
                    "identity_profile": marker.identity_profile,
                }
            if item is None or not item.get("rollback_source") or item.get("rollback") != marker.strict_rollback_state:
                continue
            source = _frozen_rollback_source(journal, marker, item)
            current = chain.generation(item["rollback"])
            seen = set()
            while current is not None and current.name not in seen:
                if current.name == state:
                    proof = CreateReplay(source, json.dumps(marker.as_json(), sort_keys=True))
                    self.pending[root] = proof
                    self.replayed[name] = key
                    return proof
                seen.add(current.name)
                current = chain.generation(current.parent) if current.parent else None
        return None


def existing_create_replay(scope, root, field, element_index=None):
    authority = _active.get()
    return authority.replay(scope, root, field, element_index) if authority is not None else None


def pending_create_recoveries(roots):
    """Exact frozen sources/markers proved by this plan's CREATE resolutions."""
    authority = _active.get()
    if authority is None:
        return ()
    return tuple((root, proof.rollback_source, proof.marker) for root in sorted(roots, key=str) if (proof := authority.pending.get(root)) is not None)


def replay_closure(root, workspace, proof=None):
    """Resolve a live replay or a root-confined provisional rollback closure.

    The provisional closure describes only the resources claimed for recovery;
    it must never be published as an observation of the live partial dataset.
    """
    from shinobi.dataset_closure import resolve_dataset_closure

    root = Path(root)
    authority = _active.get()
    if proof is None:
        proof = authority.pending.get(root) if authority is not None else None
    if proof is None:
        return None
    source = proof.rollback_source or root
    closure = resolve_dataset_closure(source, storage_namespace=source.parent if proof.rollback_source else workspace)
    if not closure.valid:
        raise DatasetLifecycleUnavailableError(f"required CREATE replay has no validated rollback or live closure: {source}")
    if proof.rollback_source is None:
        return closure
    if (
        source.is_symlink()
        or any(member.is_symlink() for member in source.rglob("*"))
        or any(resource.external_to_root or not resource.path.is_relative_to(source) for resource in closure.resources)
    ):
        raise DatasetLifecycleUnavailableError("provisional CREATE rollback closure has external or symlink resources")
    return closure.model_copy(
        update={
            "requested_root": root,
            "root": root,
            "storage_namespace": Path(workspace),
            "message": "provisional exact frozen rollback closure; live root awaits recovery",
            "resources": tuple(
                resource.model_copy(
                    update={
                        "path": root / resource.path.relative_to(source),
                        "namespace_path": (root / resource.path.relative_to(source)).relative_to(workspace),
                    }
                )
                for resource in closure.resources
            ),
        }
    )


def clear_pending_create_recoveries():
    """Begin a fresh whole-workflow plan, requiring CREATE proof anew."""
    authority = _active.get()
    if authority is not None:
        authority.pending.clear()
