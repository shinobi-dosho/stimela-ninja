"""Regular files share the strict group's existing recovery and success oracle."""

import pytest

from shinobi.cache import get_cache_manifest
from shinobi.dataset_lifecycle import AuxiliaryMutationRecord, DatasetLeafAttempt, DatasetMutationOutcome, DatasetMutationRecord
from shinobi.derived import DerivedAddress, ResolvedDerivedRead, regular_file_identity
from shinobi.exceptions import DatasetLifecycleUnavailableError
from shinobi.snapshots import SnapshotGuard, StrictMutation, chain_id, faults, get_journal, reconcile, required_predecessor, state_name
from tests.test_snapshots_strict import _identity


@pytest.fixture(autouse=True)
def clean_faults(monkeypatch):
    monkeypatch.setattr("shinobi.dataset_lifecycle.dataset_identity", lambda path, workspace: _identity(path))
    yield
    faults.hooks.clear()


def setup(tmp_path):
    ms = tmp_path / "obs.ms"
    ms.mkdir()
    (ms / "table.dat").write_text("ms0")
    model = tmp_path / "image-model.fits"
    model.write_text("model0")
    read = ResolvedDerivedRead(DerivedAddress(name="model", coordinates={}), model, model, "file", True, True)
    return ms, model, read


def guard(tmp_path, ms, read, key="a" * 64, step="continue", run="run1"):
    journal = get_journal(str(tmp_path / "cache"))
    selected, identity = required_predecessor(journal, step, read)
    result = SnapshotGuard(
        journal,
        step,
        key,
        run,
        {"ms": ms, read.address.field: read.path},
        {},
        set(),
        force_copy=True,
        strict=StrictMutation(identity=_identity, profiles={read.address.field: "regular-file/v1"}, predecessors={read.address.field: selected}),
        addresses={read.address.field: read.address},
    )
    return result, identity


def validate(g, ms, model, read):
    g.successor_identities["ms"] = _identity(ms)
    g.successor_identities[read.address.field] = regular_file_identity(model)


def test_joint_failure_rolls_back_regular_file_and_ms(tmp_path):
    ms, model, read = setup(tmp_path)
    g, _ = guard(tmp_path, ms, read)
    g.before_run()
    (ms / "table.dat").write_text("partial ms")
    model.write_text("partial model")
    g.after_failure()
    assert (ms / "table.dat").read_text() == "ms0" and model.read_text() == "model0"
    assert all(g.journal.get(chain_id(path)).marker is None for path in (ms, model))


def test_same_step_keys_original_predecessor_and_new_step_keys_current_head(tmp_path):
    ms, model, read = setup(tmp_path)
    first, initial = guard(tmp_path, ms, read)
    first.before_run()
    (ms / "table.dat").write_text("ms1")
    model.write_text("model1")
    validate(first, ms, model, read)
    first.after_success(lambda: None)
    original, original_identity = required_predecessor(first.journal, "continue", read)
    current, current_identity = required_predecessor(first.journal, "next-cycle", read)
    assert original_identity == initial
    assert current_identity == regular_file_identity(model)
    assert original != current
    second, _ = guard(tmp_path, ms, read, key="b" * 64, run="run2")
    second.before_run()
    assert model.read_text() == "model0"
    model.write_text("model2")
    (ms / "table.dat").write_text("ms2")
    validate(second, ms, model, read)
    second.after_success(lambda: None)
    assert required_predecessor(second.journal, "continue", read)[1] == initial


def test_external_model_edits_refuse_without_restoring_over_them(tmp_path):
    ms, model, read = setup(tmp_path)
    first, _ = guard(tmp_path, ms, read)
    first.before_run()
    model.write_text("model1")
    validate(first, ms, model, read)
    first.after_success(lambda: None)
    model.write_text("external edit")
    with pytest.raises(DatasetLifecycleUnavailableError, match="outside this pipeline"):
        required_predecessor(first.journal, "continue", read)
    assert model.read_text() == "external edit"


@pytest.mark.parametrize("stage", ["S1", "S2", "S3", "S4", "S5"])
def test_existing_joint_recovery_protocol_handles_regular_files(tmp_path, monkeypatch, stage):
    ms, model, read = setup(tmp_path)
    g, _ = guard(tmp_path, ms, read)
    g.success_record = tmp_path / "oracle.json"
    g.success_kind = "dataset-lifecycle"
    g.before_run()
    (ms / "table.dat").write_text("ms1")
    model.write_text("model1")
    validate(g, ms, model, read)
    published = []
    monkeypatch.setattr("shinobi.dataset_lifecycle.mutation_committed", lambda *args, **kwargs: bool(published))

    def crash():
        raise KeyboardInterrupt(stage)

    faults.hooks[stage] = crash
    with pytest.raises(KeyboardInterrupt, match=stage):
        g.after_success(lambda: published.append(True))
    faults.hooks.clear()
    reconcile(str(tmp_path / "cache"), get_cache_manifest(str(tmp_path / "cache")), paths={ms, model}, exact=True)
    expected = "1" if stage in {"S3", "S4", "S5"} else "0"
    assert model.read_text() == "model" + expected
    assert (ms / "table.dat").read_text() == "ms" + expected
    assert all(g.journal.get(chain_id(path)).marker is None for path in (ms, model))


def test_same_key_failed_attempt_restores_frozen_successor(tmp_path):
    ms, model, read = setup(tmp_path)
    first, _ = guard(tmp_path, ms, read)
    first.before_run()
    model.write_text("model1")
    (ms / "table.dat").write_text("ms1")
    validate(first, ms, model, read)
    first.after_success(lambda: None)
    rerun, _ = guard(tmp_path, ms, read, run="run2")
    rerun.before_run()
    model.write_text("failed model2")
    (ms / "table.dat").write_text("failed ms2")
    rerun.after_failure()
    assert model.read_text() == "model1" and (ms / "table.dat").read_text() == "ms1"


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_auxiliary_snapshot_damage_refuses_preserving_all_live_paths(tmp_path, damage):
    ms, model, read = setup(tmp_path)
    g, _ = guard(tmp_path, ms, read)
    g.before_run()
    model.write_text("model1")
    validate(g, ms, model, read)
    g.after_success(lambda: None)
    selected, _ = required_predecessor(g.journal, "continue", read)
    snapshot = g.journal.snapshot_dir(selected)
    if damage == "missing":
        snapshot.unlink()
    else:
        snapshot.write_text("corrupted")
    with pytest.raises(DatasetLifecycleUnavailableError):
        guard(tmp_path, ms, read, run="run2")
    assert model.read_text() == "model1" and (ms / "table.dat").read_text() == "ms0"


def test_auxiliary_oracle_participants_are_separate_from_dataset_mutations(tmp_path):
    from shinobi.snapshots import _validate_group_evidence

    ms, model, read = setup(tmp_path)
    g, _ = guard(tmp_path, ms, read)
    g.before_run()
    plans = {plan.field: plan for plan in g.plans}
    msplan, auxplan = plans["ms"], plans[read.address.field]
    mutation = DatasetMutationRecord(
        field="ms",
        mode="write",
        root=ms,
        predecessor_state=msplan.required,
        predecessor_signature=msplan.predecessor.signature,
        predecessor_fingerprint=msplan.predecessor.fingerprint,
        predecessor_snapshot=g.journal.snapshot_dir(msplan.required),
        successor_state=state_name(g.cache_key, "ms"),
        outcome=DatasetMutationOutcome.COMMITTED,
    )
    auxiliary = AuxiliaryMutationRecord(
        address=read.address,
        path=model,
        predecessor_state=auxplan.required,
        predecessor_signature=auxplan.predecessor.signature,
        predecessor_fingerprint=auxplan.predecessor.fingerprint,
        predecessor_snapshot=g.journal.snapshot_dir(auxplan.required),
        successor_state=state_name(g.cache_key, read.address.field),
        outcome=DatasetMutationOutcome.COMMITTED,
    )
    leaf = DatasetLeafAttempt(step_path="continue", accesses=(), mutations=(mutation,), auxiliary_mutations=(auxiliary,))
    assert _validate_group_evidence(leaf, g.group)
    assert not _validate_group_evidence(leaf.model_copy(update={"auxiliary_mutations": ()}), g.group)
    assert "auxiliary_mutations" not in DatasetLeafAttempt(step_path="old", accesses=()).model_dump()
    g.after_failure()
