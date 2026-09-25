"""The strict (exact-recovery) policy of the Tier 1 snapshot guard.

Casacore-free: a strict guard only needs *a* structural signature, so these
tests supply one computed from plain files. The real-table route is covered
by ``test_dataset_mutation.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from shinobi import DatasetAccess, MeasurementSetV2, pystep
from shinobi.cache import get_cache_manifest
from shinobi.dataset_lifecycle import (
    DATASET_MUTATION_CAPABILITY,
    DATASET_READ_CAPABILITY,
    DatasetLeafAttempt,
    DatasetLifecycleAttempt,
    DatasetLifecycleEvent,
    DatasetLifecyclePhase,
    mutation_committed,
)
from shinobi.exceptions import DatasetLifecycleUnavailableError
from shinobi.snapshots import (
    Chain,
    Generation,
    HeadStatus,
    Marker,
    SnapshotGuard,
    StateIdentity,
    StrictMutation,
    chain_id,
    faults,
    get_journal,
    reconcile,
    state_name,
    strict_reuse_issue,
)
from shinobi.steps.schema import mutated_path_fields


def _signature(path: Path) -> str:
    """Structure = the sorted names of the tree's entries."""
    return hashlib.sha256(json.dumps(sorted(p.name for p in path.iterdir())).encode()).hexdigest()


def _identity(path: Path) -> StateIdentity:
    """Structure plus each file's (name, size, mtime), as the real one does."""
    files = sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in path.iterdir() if p.is_file())
    return StateIdentity(signature=_signature(path), fingerprint=hashlib.sha256(json.dumps(files).encode()).hexdigest())


STRICT = StrictMutation(identity=_identity)


@pytest.fixture(autouse=True)
def _clean_faults():
    yield
    faults.hooks.clear()


def _dataset(tmp_path: Path, text: str = "v0") -> Path:
    ms = tmp_path / "obs.ms"
    ms.mkdir()
    (ms / "table.dat").write_text(text)
    return ms


def _guard(tmp_path: Path, ms: Path, key: str, *, step: str = "step", run: str = "run1", **kwargs) -> SnapshotGuard:
    return SnapshotGuard(get_journal(str(tmp_path / "cache")), step, key, run, {"ms": ms}, {}, set(), force_copy=True, strict=STRICT, **kwargs)


def _commit(guard: SnapshotGuard, ms: Path) -> None:
    guard.successor_identities["ms"] = _identity(ms)
    guard.after_success(lambda: None)


def test_strict_guard_requires_a_name_for_every_write(tmp_path):
    ms = _dataset(tmp_path)
    journal = get_journal(str(tmp_path / "cache"))
    with pytest.raises(DatasetLifecycleUnavailableError):
        SnapshotGuard(journal, "step", None, "run", {"ms": ms}, {}, set(), strict=STRICT)
    with pytest.raises(DatasetLifecycleUnavailableError):
        SnapshotGuard(journal, "step", "k" * 64, "run", {"ms": ms}, {}, set(), strict=STRICT, tainting={"other": (ms,)})


def test_strict_commit_records_signature_and_parent(tmp_path):
    ms = _dataset(tmp_path)
    guard = _guard(tmp_path, ms, "a" * 64)
    guard.before_run()
    (ms / "table.f0").write_text("new column")
    _commit(guard, ms)

    chain = get_journal(str(tmp_path / "cache")).get(chain_id(ms))
    produced = chain.generation(state_name("a" * 64, "ms"))
    consumed = chain.generation(produced.parent)
    assert chain.head == produced.name and chain.marker is None
    assert produced.structural_signature == _signature(ms)
    assert consumed.name.startswith("gen0__") and consumed.structural_signature != produced.structural_signature
    assert [plan.outcome for plan in guard.plans] == ["committed"]


def test_strict_failure_rolls_back_immediately_and_clears_the_marker(tmp_path):
    ms = _dataset(tmp_path)
    guard = _guard(tmp_path, ms, "a" * 64)
    guard.before_run()
    (ms / "table.dat").write_text("partial")
    (ms / "junk").write_text("x")

    guard.after_failure()

    assert (ms / "table.dat").read_text() == "v0" and not (ms / "junk").exists()
    chain = get_journal(str(tmp_path / "cache")).get(chain_id(ms))
    assert chain.marker is None and chain.status is HeadStatus.TRUSTED and chain.head.startswith("gen0__")
    assert guard.plans[0].outcome == "rolled-back"
    assert not list(tmp_path.glob("obs.ms.shinobi-trash.*"))


def test_strict_reconcile_restores_exact_predecessor_without_a_retry(tmp_path):
    ms = _dataset(tmp_path)
    guard = _guard(tmp_path, ms, "a" * 64)
    guard.before_run()
    (ms / "table.dat").write_text("unrecorded successor")
    guard.successor_identities["ms"] = _identity(ms)
    faults.hooks["S2"] = lambda: (_ for _ in ()).throw(KeyboardInterrupt("hard stop before oracle"))
    with pytest.raises(KeyboardInterrupt):
        guard.after_success(lambda: None)
    faults.hooks.clear()

    notes = reconcile(
        str(tmp_path / "cache"),
        get_cache_manifest(str(tmp_path / "cache")),
        paths={ms},
        exact=True,
    )

    assert (ms / "table.dat").read_text() == "v0"
    chain = get_journal(str(tmp_path / "cache")).get(chain_id(ms))
    assert chain.marker is None and chain.status is HeadStatus.TRUSTED
    assert chain.head.startswith("gen0__")
    assert any("restored exact state" in note for note in notes)


def test_strict_reconcile_returns_a_mid_chain_rerun_to_the_head_it_found(tmp_path):
    from shinobi.cache import ProvenanceKey

    ms = _dataset(tmp_path)
    first = _guard(tmp_path, ms, "a" * 64, step="first")
    first.before_run()
    (ms / "table.dat").write_text("A")
    _commit(first, ms)
    journal = get_journal(str(tmp_path / "cache"))
    second = SnapshotGuard(journal, "second", "b" * 64, "run2", {"ms": ms}, {"ms": ProvenanceKey("a" * 64, producer_field="ms")}, {"ms"}, force_copy=True, strict=STRICT)
    second.before_run()
    (ms / "table.dat").write_text("B")
    _commit(second, ms)

    # Re-running the first step restores its predecessor over B; dying
    # before the oracle must put back B, the head it found, not gen0.
    rerun = _guard(tmp_path, ms, "a" * 64, step="first", run="run3")
    rerun.before_run()
    assert (ms / "table.dat").read_text() == "v0"
    (ms / "table.dat").write_text("partial")
    rerun.successor_identities["ms"] = _identity(ms)
    faults.hooks["S2"] = lambda: (_ for _ in ()).throw(KeyboardInterrupt("hard stop before oracle"))
    with pytest.raises(KeyboardInterrupt):
        rerun.after_success(lambda: None)
    faults.hooks.clear()

    notes = reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths={ms}, exact=True)

    assert (ms / "table.dat").read_text() == "B"
    chain = journal.get(chain_id(ms))
    assert chain.marker is None and chain.status is HeadStatus.TRUSTED
    assert chain.head == state_name("b" * 64, "ms")
    assert any("restored exact state" in note for note in notes)


def test_exact_reconcile_treats_a_marker_without_rollback_state_as_ordinary(tmp_path):
    # A non-strict Tier 1 crash (or a strict one from before markers froze
    # their rollback state) must not wedge every later strict run.
    ms = _dataset(tmp_path)
    guard = SnapshotGuard(get_journal(str(tmp_path / "cache")), "step", "c" * 64, "run1", {"ms": ms}, {}, set(), force_copy=True)
    guard.before_run()
    (ms / "table.dat").write_text("partial")
    faults.hooks["S2"] = lambda: (_ for _ in ()).throw(KeyboardInterrupt("hard stop before oracle"))
    with pytest.raises(KeyboardInterrupt):
        guard.after_success(lambda: None)
    faults.hooks.clear()
    assert get_journal(str(tmp_path / "cache")).get(chain_id(ms)).marker.strict_rollback_state is None

    notes = reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths={ms}, exact=True)

    chain = get_journal(str(tmp_path / "cache")).get(chain_id(ms))
    assert chain.marker is None and chain.status is HeadStatus.UNTRUSTED
    assert any("partial write" in note for note in notes)


def test_strict_refusals_undo_before_launch(tmp_path):
    ms = _dataset(tmp_path)
    first = _guard(tmp_path, ms, "a" * 64)
    first.before_run()
    (ms / "table.dat").write_text("v1")
    _commit(first, ms)
    journal = get_journal(str(tmp_path / "cache"))
    gen0 = journal.get(chain_id(ms)).generation(state_name("a" * 64, "ms")).parent

    # The predecessor's snapshot vanished: the default guard would warn and
    # run against live disk; the strict one refuses and leaves no marker.
    import shutil

    shutil.rmtree(journal.snapshot_dir(gen0))
    rerun = _guard(tmp_path, ms, "b" * 64, run="run2")
    with pytest.raises(DatasetLifecycleUnavailableError, match="has no snapshot to restore"):
        rerun.before_run()
    chain = journal.get(chain_id(ms))
    assert chain.marker is None and (ms / "table.dat").read_text() == "v1"


def test_strict_restore_verifies_the_restored_structure(tmp_path):
    ms = _dataset(tmp_path)
    first = _guard(tmp_path, ms, "a" * 64)
    first.before_run()
    (ms / "table.dat").write_text("v1")
    _commit(first, ms)
    journal = get_journal(str(tmp_path / "cache"))
    gen0 = journal.get(chain_id(ms)).generation(state_name("a" * 64, "ms")).parent
    # Corrupt the predecessor snapshot's structure.
    (journal.snapshot_dir(gen0) / "stray").write_text("x")

    rerun = _guard(tmp_path, ms, "b" * 64, run="run2")
    with pytest.raises(DatasetLifecycleUnavailableError, match="differs from the structure recorded"):
        rerun.before_run()
    # The live successor was swapped back, not left replaced by a bad restore.
    assert (ms / "table.dat").read_text() == "v1" and not (ms / "stray").exists()
    assert journal.get(chain_id(ms)).marker is None


def test_strict_refuses_a_taint_blocked_predecessor(tmp_path):
    ms = _dataset(tmp_path)
    first = _guard(tmp_path, ms, "a" * 64)
    first.before_run()
    (ms / "table.dat").write_text("v1")
    _commit(first, ms)
    journal = get_journal(str(tmp_path / "cache"))
    head = journal.get(chain_id(ms)).head

    def taint(chain: Chain | None) -> Chain | None:
        chain.tainted_through = head
        return chain

    journal.update_chain(chain_id(ms), taint)
    rerun = _guard(tmp_path, ms, "b" * 64, run="run2")
    with pytest.raises(DatasetLifecycleUnavailableError, match="predates a write this journal could not name"):
        rerun.before_run()


def test_strict_create_starts_a_fresh_chain_and_failure_restores_absence(tmp_path):
    target = tmp_path / "new.ms"
    policy = StrictMutation(identity=_identity, create_fields=frozenset({"ms"}))
    journal = get_journal(str(tmp_path / "cache"))
    # A chain for an earlier, since-deleted dataset at the same path.
    journal.update_chain(chain_id(target), lambda _chain: Chain(dev=1, ino=2, ctime_ns=3, path=str(target), head="old", generations=[Generation(name="old")]))

    guard = SnapshotGuard(journal, "create", "c" * 64, "run", {"ms": target}, {}, set(), force_copy=True, strict=policy)
    guard.before_run()
    assert journal.get(chain_id(target)).generations == []
    target.mkdir()
    (target / "table.dat").write_text("partial")
    guard.after_failure()

    assert not target.exists()
    assert journal.get(chain_id(target)) is None
    assert guard.plans[0].outcome == "absent-restored"


def test_strict_create_refuses_an_existing_target(tmp_path):
    ms = _dataset(tmp_path)
    policy = StrictMutation(identity=_identity, create_fields=frozenset({"ms"}))
    guard = SnapshotGuard(get_journal(str(tmp_path / "cache")), "create", "c" * 64, "run", {"ms": ms}, {}, set(), strict=policy)
    with pytest.raises(DatasetLifecycleUnavailableError, match="already exists"):
        guard.before_run()


def test_strict_rule_b_replaces_a_stale_same_name_snapshot(tmp_path):
    ms = _dataset(tmp_path)
    first = _guard(tmp_path, ms, "a" * 64)
    first.before_run()
    (ms / "one").write_text("x")
    _commit(first, ms)
    journal = get_journal(str(tmp_path / "cache"))
    name = state_name("a" * 64, "ms")
    assert sorted(p.name for p in journal.snapshot_dir(name).iterdir()) == ["one", "table.dat"]

    # Same key again (e.g. after the manifest entry was lost), but a
    # nondeterministic tool produced a different structure this time.
    again = _guard(tmp_path, ms, "a" * 64, run="run2")
    again.before_run()
    # The re-run consumes its predecessor again, not its own last output.
    assert sorted(p.name for p in ms.iterdir()) == ["table.dat"]
    (ms / "two").write_text("y")
    _commit(again, ms)
    assert sorted(p.name for p in journal.snapshot_dir(name).iterdir()) == ["table.dat", "two"]
    assert journal.get(chain_id(ms)).generation(name).structural_signature == _signature(ms)


def test_strict_reuse_requires_the_head_to_descend_from_the_state(tmp_path):
    ms = _dataset(tmp_path)
    journal = get_journal(str(tmp_path / "cache"))
    a = _guard(tmp_path, ms, "a" * 64, step="a")
    a.before_run()
    (ms / "table.dat").write_text("after a")
    _commit(a, ms)
    b = _guard(tmp_path, ms, "b" * 64, step="b", run="run2")
    b.before_run()
    (ms / "b-col").write_text("after b")
    _commit(b, ms)
    state_a, state_b = state_name("a" * 64, "ms"), state_name("b" * 64, "ms")
    live = _identity(ms)

    assert strict_reuse_issue(journal, ms, state_b, live) is None
    assert strict_reuse_issue(journal, ms, state_a, live) is None
    assert "does not descend" in strict_reuse_issue(journal, ms, state_name("z" * 64, "ms"), live)
    assert "structure differs" in strict_reuse_issue(journal, ms, state_b, StateIdentity(signature="0" * 64, fingerprint=live.fingerprint))
    assert "table files differ" in strict_reuse_issue(journal, ms, state_b, StateIdentity(signature=live.signature, fingerprint="0" * 64))

    def marked(chain: Chain | None) -> Chain | None:
        chain.marker = Marker(step_path="b", field="ms", cache_key="b" * 64, run_id="x", started_at=0.0)
        return chain

    journal.update_chain(chain_id(ms), marked)
    assert "in-flight" in strict_reuse_issue(journal, ms, state_b, live)


def test_non_strict_journal_records_are_unchanged(tmp_path):
    # New optional fields are written only when set, so an existing journal
    # (and anything diffing it) sees byte-identical records.
    assert Generation(name="g", size=1).as_json() == {"name": "g", "size": 1, "snapshot_present": True}
    marker = Marker(step_path="s", field="f", cache_key=None, run_id="r", started_at=0.0)
    assert "success_kind" not in marker.as_json()
    assert Generation(**Generation(name="g", structural_signature="s", parent="p").as_json()).parent == "p"


def test_declared_dataset_write_joins_the_shared_mutated_set():
    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="write")])
    def flag(ms: MeasurementSetV2, other: Path) -> None:
        pass

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        pass

    # The cache key drops exactly this set and the snapshotter protects it.
    assert mutated_path_fields(flag.step) == {"ms"}
    assert mutated_path_fields(read.step) == set()


def _attempt(**changes) -> dict:
    event = DatasetLifecycleEvent(phase=DatasetLifecyclePhase.PLANNED, observed_at=0.0, reason="r")
    base = {
        "attempt_id": "run",
        "scope": "s",
        "workspace": Path("/w"),
        "phase": DatasetLifecyclePhase.PLANNED,
        "events": (event,),
        "backends": ("native",),
        "capability_supported": True,
        "reason": "r",
    }
    return {**base, **changes}


def test_attempt_schema_versions_are_closed():
    assert DatasetLifecycleAttempt(**_attempt()).capability == DATASET_READ_CAPABILITY
    leaf = DatasetLeafAttempt(step_path="s", accesses=())
    with pytest.raises(ValueError, match="schema 1"):
        DatasetLifecycleAttempt(**_attempt(leaves=(leaf,)))
    with pytest.raises(ValueError, match="schema 1"):
        DatasetLifecycleAttempt(**_attempt(capability=DATASET_MUTATION_CAPABILITY))
    assert DatasetLifecycleAttempt(**_attempt(schema_version=2, capability=DATASET_MUTATION_CAPABILITY, leaves=(leaf,))).leaves == (leaf,)


def test_unreadable_or_foreign_attempt_is_never_a_success_oracle(tmp_path):
    assert not mutation_committed(tmp_path / "missing.json", attempt_id="run", step_path="s", cache_key="k")


def test_strict_guard_refuses_an_in_place_write_it_never_saw(tmp_path):
    # Rewriting a member file moves neither the structure nor the root's
    # ctime -- exactly what an in-place casacore cell update does -- so only
    # the member fingerprint can tell the live head from the recorded one.
    import os

    ms = _dataset(tmp_path)
    first = _guard(tmp_path, ms, "a" * 64)
    first.before_run()
    (ms / "table.dat").write_text("v1")
    _commit(first, ms)
    root_ctime = ms.stat().st_ctime_ns
    (ms / "table.dat").write_text("vX")  # same size, new content and mtime
    os.utime(ms / "table.dat", ns=(1, 1))
    assert ms.stat().st_ctime_ns == root_ctime

    journal = get_journal(str(tmp_path / "cache"))
    assert "table files differ" in strict_reuse_issue(journal, ms, state_name("a" * 64, "ms"), _identity(ms))
    again = _guard(tmp_path, ms, "b" * 64, run="run2")
    with pytest.raises(DatasetLifecycleUnavailableError, match="table files changed"):
        again.before_run()
    assert (ms / "table.dat").read_text() == "vX"
    assert journal.get(chain_id(ms)).marker is None
