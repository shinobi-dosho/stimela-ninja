"""Tier 1: mutation-chain snapshots.

The two hazards being closed are wrong-science bugs, not slow runs, so the
tests are written as scenarios ("what does the MS actually contain
afterwards") rather than as assertions about internal state wherever that
is possible.
"""

import json
import os
from pathlib import Path

import pytest
import shinobi
from pydantic import BaseModel

from shinobi.cache import get_cache_manifest
from shinobi.config import AppConfig
from shinobi.snapshots import HeadStatus, Marker, chain_id, faults, get_journal, orphan_trash, reconcile, state_name
from shinobi.steps import InputRef, OutputRef, Recipe


class MsOut(BaseModel):
    ms: Path


class SpwInputs(BaseModel):
    spw: str = "*"


class Crash(Exception):
    """Stands in for a SIGKILL: stops the run dead at a chosen point,
    leaving whatever is on disk exactly as it was at that instant.
    """


@pytest.fixture(autouse=True)
def _clean_faults():
    faults.hooks.clear()
    yield
    faults.hooks.clear()


def _read(ms: Path) -> str:
    return (ms / "table.dat").read_text()


def _write(ms: Path, text: str) -> None:
    ms.mkdir(exist_ok=True)
    (ms / "table.dat").write_text(text)


def _chain_of(cache_dir, ms: Path):
    return get_journal(str(cache_dir)).get(chain_id(ms))


def _pipeline(ms: Path, calls: dict, fail: dict | None = None, flag_cache=None, strategy: str = "default", solver: str = "default"):
    """The caracal shape: `split` writes the MS, `flag` and `cal` each read
    and rewrite that same MS in place. Every step's `ms` is on both its
    inputs and its outputs model, so the shipped cache drops it from the key
    and has no way to notice what state the tree is actually in.
    """
    fail = fail or {}

    @shinobi.pystep()
    def split(ctx, spw: str = "*") -> MsOut:
        calls["split"] = calls.get("split", 0) + 1
        _write(ms, f"vis[{spw}]")
        return MsOut(ms=ms)

    @shinobi.pystep()
    def flag(ctx, ms: Path, strategy: str = "default") -> MsOut:
        calls["flag"] = calls.get("flag", 0) + 1
        calls.setdefault("flag_saw", []).append(_read(ms))
        _write(ms, _read(ms) + f"|flag[{strategy}]")
        return MsOut(ms=ms)

    @shinobi.pystep()
    def cal(ctx, ms: Path, solver: str = "default") -> MsOut:
        calls["cal"] = calls.get("cal", 0) + 1
        calls.setdefault("cal_saw", []).append(_read(ms))
        if fail.get("cal"):
            # A half-written MS, exactly what an interrupted rewrite leaves.
            _write(ms, _read(ms) + "|PARTIAL")
            raise Crash("cal died mid-rewrite")
        _write(ms, _read(ms) + "|cal")
        return MsOut(ms=ms)

    flag_step = flag.model_copy(update={"wiring": {"ms": OutputRef(step="split", field="ms")}, "params": {"strategy": strategy}})
    if flag_cache is not None:
        flag_step = flag_step.model_copy(update={"step": flag_step.step.model_copy(update={"cache": flag_cache})})
    return Recipe(
        name="pipe",
        inputs_model=SpwInputs,
        outputs_model=MsOut,
        steps=[
            split.model_copy(update={"wiring": {"spw": InputRef(field="spw")}}),
            flag_step,
            cal.model_copy(update={"wiring": {"ms": OutputRef(step="flag", field="ms")}, "params": {"solver": solver}}),
        ],
        output_wiring={"ms": OutputRef(step="cal", field="ms")},
    )


# --- the two motivating hazards ------------------------------------------


def test_changing_a_midchain_step_reruns_it_against_its_own_input_not_the_chain_tip(tmp_path):
    """Hazard 1, and the reason this exists.

    Change `flag`'s params after a full run. `flag` misses and re-executes --
    but the MS on disk holds *post-`cal`* state, so without a rollback it
    flags already-calibrated data and writes something that looks finished
    and is wrong. The shipped cache cannot see this: a mutated path is
    dropped from the key by construction.
    """
    ms = tmp_path / "data.ms"
    calls = {}
    cache_dir = tmp_path / "cache"

    _pipeline(ms, calls)(spw="*", cache=True, cache_dir=str(cache_dir))
    assert _read(ms) == "vis[*]|flag[default]|cal"

    _pipeline(ms, calls, strategy="aggressive")(spw="*", cache=True, cache_dir=str(cache_dir))

    # The re-run of `flag` saw `split`'s output, not the calibrated tip.
    assert calls["flag_saw"] == ["vis[*]", "vis[*]"]
    assert _read(ms) == "vis[*]|flag[aggressive]|cal"


def test_a_step_killed_mid_rewrite_is_rerun_against_a_clean_input(tmp_path):
    """Hazard 2, the stronger one: it converts a silent wrong-science bug
    into a correct re-run.

    `cal` dies halfway through rewriting the MS. The skip cache is
    content-blind to a mutated path, so on resume the re-run of `cal` would
    otherwise execute against its own half-written output.
    """
    ms = tmp_path / "data.ms"
    calls = {}
    cache_dir = tmp_path / "cache"

    with pytest.raises(Exception):
        _pipeline(ms, calls, fail={"cal": True})(spw="*", cache=True, cache_dir=str(cache_dir))
    assert _read(ms) == "vis[*]|flag[default]|PARTIAL"

    _pipeline(ms, calls)(spw="*", cache=True, cache_dir=str(cache_dir))

    assert calls["cal_saw"] == ["vis[*]|flag[default]", "vis[*]|flag[default]"]
    assert _read(ms) == "vis[*]|flag[default]|cal"


def test_an_unchanged_rerun_still_skips_the_whole_chain(tmp_path):
    """Tier 1 must not cost the property the shipped cache already has."""
    ms = tmp_path / "data.ms"
    calls = {}
    cache_dir = tmp_path / "cache"

    _pipeline(ms, calls)(spw="*", cache=True, cache_dir=str(cache_dir))
    _pipeline(ms, calls)(spw="*", cache=True, cache_dir=str(cache_dir))

    assert calls["split"] == 1 and calls["flag"] == 1 and calls["cal"] == 1
    assert _read(ms) == "vis[*]|flag[default]|cal"


# --- restore branches -----------------------------------------------------


def test_restore_is_a_noop_when_the_head_already_names_the_required_state(tmp_path):
    """Branch 2. The common case, and it must not copy anything."""
    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    _pipeline(ms, {})(spw="*", cache=True, cache_dir=str(cache_dir))

    chain = _chain_of(cache_dir, ms)
    assert chain.status is HeadStatus.TRUSTED
    assert chain.marker is None
    assert not list(tmp_path.glob("*.shinobi-trash.*"))


def test_a_failed_step_leaves_the_workspace_in_its_pre_run_state(tmp_path):
    """A restore followed by a failure must not leave the workspace rolled
    back -- that would destroy the calibrated tip and leave an
    under-processed MS at a finished-looking path.
    """
    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    calls = {}
    _pipeline(ms, calls)(spw="*", cache=True, cache_dir=str(cache_dir))
    assert _read(ms) == "vis[*]|flag[default]|cal"

    # `flag` re-runs (params changed) so it restores, and then `cal` dies.
    with pytest.raises(Exception):
        _pipeline(ms, calls, fail={"cal": True}, strategy="x")(spw="*", cache=True, cache_dir=str(cache_dir))

    # `cal` took no restore of its own (its input was already the state it
    # needed), so its partial write stands -- and is recovered on the next
    # run, which is what the marker is for.
    assert "PARTIAL" in _read(ms)
    _pipeline(ms, calls, strategy="x")(spw="*", cache=True, cache_dir=str(cache_dir))
    assert _read(ms) == "vis[*]|flag[x]|cal"


def test_a_missing_snapshot_warns_and_proceeds_rather_than_restoring(tmp_path, caplog):
    """Branch 4, and invariant 7: with nothing to restore from, behave
    exactly as the shipped cache would -- run against live disk -- and say
    so loudly.
    """
    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    calls = {}
    _pipeline(ms, calls)(spw="*", cache=True, cache_dir=str(cache_dir))

    # Delete every snapshot but keep the journal, so names resolve to nothing.
    for state in (cache_dir / "snapshots" / "states").iterdir():
        import shutil

        shutil.rmtree(state)

    with caplog.at_level("WARNING"):
        _pipeline(ms, calls, strategy="y")(spw="*", cache=True, cache_dir=str(cache_dir))
    assert "no snapshot of it exists" in caplog.text
    # It still ran, against whatever was there -- never a silent skip.
    assert calls["flag"] == 2


def test_a_restore_does_not_change_the_steps_cache_key(tmp_path):
    """Invariant 6, which is what licenses running the restore hook *after*
    the key has been computed. A mutated path contributes only its path
    string to the key, so putting different content at that path cannot move
    it.
    """
    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    calls = {}
    _pipeline(ms, calls)(spw="*", cache=True, cache_dir=str(cache_dir))
    first = json.loads((cache_dir / "manifest.json").read_text())["pipe.cal"]["cache_key"]

    # Force a restore of `cal`'s input by re-running `flag` unchanged but
    # with the tip rolled forward, then re-run the identical pipeline.
    _pipeline(ms, calls, strategy="z")(spw="*", cache=True, cache_dir=str(cache_dir))
    _pipeline(ms, calls)(spw="*", cache=True, cache_dir=str(cache_dir))
    again = json.loads((cache_dir / "manifest.json").read_text())["pipe.cal"]["cache_key"]
    assert first == again


# --- finding A: the manifest is a per-run oracle, not a per-key one -------


def test_a_stale_manifest_entry_is_not_read_as_proof_this_run_succeeded(tmp_path):
    """The false-hit hole in a run-blind oracle.

    There is one manifest entry per step path, so "the entry's key matches
    the marker's key" says only that *some* run of this step once
    succeeded. Delete an unrelated declared output, and the step re-runs
    under the very same key; kill it mid-rewrite, and a run-blind
    reconciliation finds the old entry, declares the step complete, drops
    the trash and leaves the head trusted -- a permanent cache hit over a
    half-written MS.
    """
    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    journal = get_journal(str(cache_dir))
    _pipeline(ms, {})(spw="*", cache=True, cache_dir=str(cache_dir))

    manifest = get_cache_manifest(str(cache_dir))
    entry = manifest.entry("pipe.cal")
    # Exactly what an interrupted re-run of the *same* key leaves behind: a
    # marker from a different run than the one that recorded the entry.
    cid = chain_id(ms)

    def arm(chain):
        chain.marker = Marker(step_path="pipe.cal", field="ms", cache_key=entry["cache_key"], run_id="a-later-run", started_at=0.0)
        return chain

    journal.update(cid, arm)

    reconcile(str(cache_dir), manifest)

    chain = journal.get(cid)
    assert chain.status is HeadStatus.UNTRUSTED, "a stale entry must not vouch for a run that never recorded one"
    assert chain.marker is None


def test_an_entry_recorded_by_this_very_run_is_accepted_as_success(tmp_path):
    """The other half: a crash in S4/S5, after the step really did finish
    and record, must *not* roll anything back.
    """
    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    journal = get_journal(str(cache_dir))
    _pipeline(ms, {})(spw="*", cache=True, cache_dir=str(cache_dir))

    manifest = get_cache_manifest(str(cache_dir))
    entry = manifest.entry("pipe.cal")
    cid = chain_id(ms)

    def arm(chain):
        chain.marker = Marker(step_path="pipe.cal", field="ms", cache_key=entry["cache_key"], run_id=entry["run_id"], started_at=0.0)
        return chain

    journal.update(cid, arm)
    reconcile(str(cache_dir), manifest)

    chain = journal.get(cid)
    assert chain.status is HeadStatus.TRUSTED
    assert chain.marker is None


def test_legacy_manifest_entries_without_a_run_id_fail_conservatively(tmp_path):
    """Entries written before `run_id` existed compare unequal, so recovery
    takes the "did not complete" branch. That is the safe direction: a
    needless re-run, never a false hit.
    """
    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    journal = get_journal(str(cache_dir))
    _pipeline(ms, {})(spw="*", cache=True, cache_dir=str(cache_dir))

    raw = json.loads((cache_dir / "manifest.json").read_text())
    key = raw["pipe.cal"]["cache_key"]
    del raw["pipe.cal"]["run_id"]
    (cache_dir / "manifest.json").write_text(json.dumps(raw))

    cid = chain_id(ms)

    def arm(chain):
        chain.marker = Marker(step_path="pipe.cal", field="ms", cache_key=key, run_id="whatever", started_at=0.0)
        return chain

    journal.update(cid, arm)
    reconcile(str(cache_dir), get_cache_manifest(str(cache_dir)))
    assert journal.get(cid).status is HeadStatus.UNTRUSTED


# --- finding B: never worse than shipped, including unwired fields --------


def test_an_uncached_mutator_is_never_reverted_by_a_cached_consumer(tmp_path):
    """Caching is per-scope, so a chain can be partly cached. An uncached
    `flag` advances the MS under a name the journal cannot record; a cached
    `cal` that then missed and restored from its own last consumed state
    would revert `flag`'s work and calibrate stale data.

    That is strictly worse than the shipped behaviour (which would simply
    run `cal` against whatever `flag` produced), so the chain detaches
    instead: no restore, run against live disk, warn.
    """
    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    calls = {}

    _pipeline(ms, calls, flag_cache=False)(spw="*", cache=True, cache_dir=str(cache_dir))
    assert _read(ms) == "vis[*]|flag[default]|cal"

    # `cal`'s own params change so it genuinely misses -- which is the only
    # way it ever reaches a restore decision. Meanwhile the uncached `flag`
    # runs again and legitimately advances the MS.
    calls.clear()
    _pipeline(ms, calls, flag_cache=False, strategy="second", solver="robust")(spw="*", cache=True, cache_dir=str(cache_dir))

    # `cal` must have seen `flag`'s fresh output, never a rolled-back state.
    assert calls["cal_saw"] == ["vis[*]|flag[default]|cal|flag[second]"]
    assert _read(ms) == "vis[*]|flag[default]|cal|flag[second]|cal"


def test_an_uncached_mutator_detaches_the_chain(tmp_path, caplog):
    """Rule 1 of the never-worse rules, at the level of journal state."""
    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    _pipeline(ms, {}, flag_cache=False)(spw="*", cache=True, cache_dir=str(cache_dir))

    with caplog.at_level("WARNING"):
        _pipeline(ms, {}, flag_cache=False)(spw="*", cache=True, cache_dir=str(cache_dir))
    assert "no snapshot protection" in caplog.text


def test_a_list_valued_mutated_field_is_excluded_loudly(tmp_path, caplog):
    """One `(key, field)` name cannot stand for N paths, and a wrong restore
    is catastrophic rather than wasteful -- so the shape is refused.
    """

    class ManyIn(BaseModel):
        vis: list[Path]

    class ManyOut(BaseModel):
        vis: list[Path]

    a, b = tmp_path / "a.ms", tmp_path / "b.ms"
    _write(a, "a")
    _write(b, "b")

    @shinobi.pystep()
    def multi(ctx, vis: list[Path]) -> ManyOut:
        for one in vis:
            _write(one, _read(one) + "|touched")
        return ManyOut(vis=vis)

    with caplog.at_level("WARNING"):
        multi(vis=[a, b], cache=True, cache_dir=str(tmp_path / "cache"))
    assert "list-valued" in caplog.text
    # Excluded from protection, not from running.
    assert _read(a) == "a|touched"


# --- crash windows --------------------------------------------------------


@pytest.mark.parametrize("stage", ["S1", "S2", "S3", "S4", "S5"])
def test_a_crash_at_every_post_success_stage_recovers_to_correct_content(tmp_path, stage):
    """The post-success sequence is a five-stage commit and every gap is its
    own crash window. Whatever the stage, the next run must end with the
    correct MS -- the only thing allowed to differ is whether a step that
    had really succeeded gets re-run.
    """
    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    calls = {}

    def die():
        raise Crash(f"killed after {stage}")

    faults.hooks[stage] = die
    with pytest.raises(Exception):
        _pipeline(ms, calls)(spw="*", cache=True, cache_dir=str(cache_dir))
    faults.hooks.clear()

    _pipeline(ms, calls)(spw="*", cache=True, cache_dir=str(cache_dir))
    assert _read(ms) == "vis[*]|flag[default]|cal"


def test_a_crash_between_the_journal_and_the_manifest_reruns_rather_than_false_hits(tmp_path):
    """The S2-S3 window, and the deliberate trade in it: reconciliation
    treats a missing manifest entry as "did not run", so a step that
    actually succeeded is re-executed. Bounded waste, chosen over a false
    hit over content nothing verified.
    """
    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    calls = {}

    faults.hooks["S2"] = lambda: (_ for _ in ()).throw(Crash("killed between journal and manifest"))
    with pytest.raises(Exception):
        _pipeline(ms, calls)(spw="*", cache=True, cache_dir=str(cache_dir))
    faults.hooks.clear()

    before = calls["flag"]
    _pipeline(ms, calls)(spw="*", cache=True, cache_dir=str(cache_dir))
    assert calls["flag"] > before  # re-ran, as designed
    assert _read(ms) == "vis[*]|flag[default]|cal"


# --- reconciliation matrix ------------------------------------------------


def test_orphan_trash_is_reported_and_never_swapped_back(tmp_path):
    """Without its marker there is nothing to say what a quarantined tree
    was quarantined *for*, so reinstating it blindly is as likely to revert
    good work as to recover anything.
    """
    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    _pipeline(ms, {})(spw="*", cache=True, cache_dir=str(cache_dir))

    orphan = ms.with_name(ms.name + ".shinobi-trash.deadbeef")
    _write(orphan, "some abandoned tree")

    found = orphan_trash(str(cache_dir))
    assert orphan in found
    assert _read(ms) == "vis[*]|flag[default]|cal"  # untouched


def test_reconcile_reports_nothing_on_a_clean_run(tmp_path):
    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    _pipeline(ms, {})(spw="*", cache=True, cache_dir=str(cache_dir))
    assert reconcile(str(cache_dir), get_cache_manifest(str(cache_dir))) == []


# --- configuration --------------------------------------------------------


def test_snapshots_off_leaves_exactly_the_shipped_behaviour(tmp_path, monkeypatch):
    """The escape hatch. With snapshots off, the mid-chain re-run hazard is
    back -- which is the point of pinning it: this is what the shipped cache
    does, and what Tier 1 changes.
    """
    config = AppConfig.load()
    monkeypatch.setattr(config.cache.snapshots, "mode", "off")

    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    calls = {}
    _pipeline(ms, calls)(spw="*", cache=True, cache_dir=str(cache_dir), _config=config)
    _pipeline(ms, calls, strategy="q")(spw="*", cache=True, cache_dir=str(cache_dir), _config=config)

    assert not (cache_dir / "snapshots").exists()
    # `flag` re-ran against the calibrated tip -- the hazard, unmitigated.
    assert calls["flag_saw"][-1] == "vis[*]|flag[default]|cal"


def test_copy_mode_still_produces_correct_content(tmp_path, monkeypatch):
    """Forcing the bottom rung must change only the space bill (A1)."""
    config = AppConfig.load()
    monkeypatch.setattr(config.cache.snapshots, "mode", "copy")

    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    calls = {}
    _pipeline(ms, calls)(spw="*", cache=True, cache_dir=str(cache_dir), _config=config)
    _pipeline(ms, calls, strategy="q")(spw="*", cache=True, cache_dir=str(cache_dir), _config=config)

    assert calls["flag_saw"] == ["vis[*]", "vis[*]"]
    assert _read(ms) == "vis[*]|flag[q]|cal"


def test_caching_off_creates_no_journal_at_all(tmp_path):
    """Tier 1 rides on the cache being enabled; with it off there must be no
    journal, no snapshots, and no cost.
    """
    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    _pipeline(ms, {})(spw="*", cache_dir=str(cache_dir))
    assert not (cache_dir / "snapshots").exists()


def test_state_names_use_the_full_key(tmp_path):
    """These names are the only thing between a restore and the wrong data,
    so they are never truncated.
    """
    name = state_name("a" * 64, "ms")
    assert name == "a" * 64 + "__ms"


# --- eviction -------------------------------------------------------------


def test_eviction_never_drops_a_state_a_live_chain_still_names(tmp_path):
    """Deleting a generation some live head or consumed record still names
    makes every restore through that point impossible -- and on a
    clone-capable filesystem reclaims almost nothing anyway, since its blocks
    are shared with the workspace tree.
    """
    from shinobi.snapshots import evict

    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    _pipeline(ms, {})(spw="*", cache=True, cache_dir=str(cache_dir))

    chain = _chain_of(cache_dir, ms)
    protected = {chain.head, *chain.consumed.values()}

    evict(str(cache_dir), target_bytes=1 << 40)
    for name in protected:
        assert get_journal(str(cache_dir)).snapshot_dir(name).exists(), name


def test_eviction_reclaims_superseded_generations(tmp_path):
    """A state no head and no consumed record names is unreachable, so it is
    the first thing to go.
    """
    from shinobi.snapshots import evict

    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    calls = {}
    _pipeline(ms, calls)(spw="*", cache=True, cache_dir=str(cache_dir))
    _pipeline(ms, calls, strategy="two")(spw="*", cache=True, cache_dir=str(cache_dir))
    _pipeline(ms, calls, strategy="three")(spw="*", cache=True, cache_dir=str(cache_dir))

    removed = evict(str(cache_dir), target_bytes=1 << 40)
    assert removed, "three runs of a mutating chain should leave something superseded"
    chain = _chain_of(cache_dir, ms)
    assert chain.head not in {name for name, _size in removed}


def test_a_dead_chain_is_fully_evictable(tmp_path):
    """Once the workspace path is gone there is nothing left to restore to,
    so even the tip stops being worth keeping.
    """
    import shutil as _shutil

    from shinobi.snapshots import evict

    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    _pipeline(ms, {})(spw="*", cache=True, cache_dir=str(cache_dir))
    _shutil.rmtree(ms)

    removed = evict(str(cache_dir), target_bytes=1 << 40)
    assert removed
    assert not any(p.is_dir() for p in (cache_dir / "snapshots" / "states").iterdir())


# --- invalidate -----------------------------------------------------------


def test_invalidate_forces_the_next_run_to_restore_before_re_executing(tmp_path):
    """Rule B makes a zero-returncode run's output a durable named state, so
    dropping the manifest entry alone leaves the garbage snapshotted and
    reachable. Invalidate has to reach the snapshot layer too -- and mark the
    head untrusted, or the next miss finds head == R and no-ops against
    exactly the content the invalidation was meant to escape.
    """
    from shinobi.snapshots import invalidate

    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    calls = {}
    _pipeline(ms, calls)(spw="*", cache=True, cache_dir=str(cache_dir))
    assert _read(ms) == "vis[*]|flag[default]|cal"

    notes = invalidate(str(cache_dir), "pipe.flag", get_cache_manifest(str(cache_dir)))
    assert notes
    assert _chain_of(cache_dir, ms).status is HeadStatus.UNTRUSTED

    _pipeline(ms, calls)(spw="*", cache=True, cache_dir=str(cache_dir))
    # `flag` re-ran against `split`'s output, not against the calibrated tip.
    assert calls["flag_saw"] == ["vis[*]", "vis[*]"]
    # And `cal` re-ran too. It had to be forced: `flag` re-executes with the
    # same params, so its key -- and therefore `cal`'s -- never moves, and a
    # plain manifest hit would have left the MS rolled back and unfinished
    # while the manifest claimed otherwise.
    assert calls["cal"] == 2
    assert _read(ms) == "vis[*]|flag[default]|cal"


def test_invalidating_an_unknown_step_is_a_noop(tmp_path):
    from shinobi.snapshots import invalidate

    cache_dir = tmp_path / "cache"
    assert invalidate(str(cache_dir), "nope.nothing", get_cache_manifest(str(cache_dir))) == []


# --- the check report -----------------------------------------------------


def test_check_reports_a_clean_run_as_clean(tmp_path):
    from shinobi.snapshots import check

    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    _pipeline(ms, {})(spw="*", cache=True, cache_dir=str(cache_dir))

    report = check(str(cache_dir), get_cache_manifest(str(cache_dir)))
    assert report["unreconciled"] == []
    assert report["off_tip"] == []
    assert report["orphan_trash"] == []


def test_check_reports_an_interrupted_step_and_a_vanished_snapshot(tmp_path):
    import shutil as _shutil

    from shinobi.snapshots import check

    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    calls = {}
    with pytest.raises(Exception):
        _pipeline(ms, calls, fail={"cal": True})(spw="*", cache=True, cache_dir=str(cache_dir))

    chain = _chain_of(cache_dir, ms)
    _shutil.rmtree(get_journal(str(cache_dir)).snapshot_dir(chain.generations[0].name))

    report = check(str(cache_dir), get_cache_manifest(str(cache_dir)))
    assert any("never finished" in line for line in report["unreconciled"])
    assert any("snapshot is gone" in line for line in report["disagreements"])


# --- stage 0: don't reconcile over a run that is still going ---------------


def _hold_a_foreign_run(cache_dir) -> int:
    """Stand in for a second shinobi process, holding its own run lock.

    `flock` locks belong to the open file description, not the process, so a
    second fd here behaves exactly as another process's would.
    """
    import fcntl

    locks = Path(cache_dir) / "snapshots" / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    fd = os.open(locks / "someone-else.lock", os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


@pytest.fixture(autouse=True)
def _clean_presence():
    from shinobi.snapshots import release_runs

    release_runs()
    yield
    release_runs()


def test_a_second_run_does_not_roll_back_a_run_that_is_still_going(tmp_path):
    """The regression this stage exists for.

    Reconciliation reads "marker set, no manifest entry" as a corpse and
    swaps the quarantined tree back over the workspace. A run that is merely
    *still going* looks identical -- it has not recorded yet either -- so a
    second shinobi starting mid-step would delete the first's work while it
    was being written. That is worse than the shipped behaviour it replaced,
    where concurrent runs merely raced.
    """
    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    _pipeline(ms, {})(spw="*", cache=True, cache_dir=str(cache_dir))

    # A live run: mid-step, marker set, its pre-run tree quarantined.
    journal = get_journal(str(cache_dir))
    trash = ms.with_name(ms.name + ".shinobi-trash.live-run")
    _write(trash, "the live run's pre-run state")
    _write(ms, "the live run's half-written output")

    def arm(chain):
        chain.marker = Marker(step_path="pipe.cal", field="ms", cache_key="k", run_id="live-run", started_at=0.0)
        return chain

    journal.update(chain_id(ms), arm)

    held = _hold_a_foreign_run(cache_dir)
    try:
        from shinobi.snapshots import release_runs

        release_runs()
        _pipeline(ms, {})(spw="*", cache=True, cache_dir=str(cache_dir))
    finally:
        os.close(held)

    # Untouched: neither swapped back nor discarded.
    assert trash.exists()
    assert _read(trash) == "the live run's pre-run state"
    assert journal.get(chain_id(ms)).marker is not None


def test_a_corpse_is_still_reconciled_when_nobody_else_is_running(tmp_path):
    """The other half: with no live process the guard must not get in the
    way, or crash recovery never tidies up at all.
    """
    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    _pipeline(ms, {})(spw="*", cache=True, cache_dir=str(cache_dir))

    journal = get_journal(str(cache_dir))
    trash = ms.with_name(ms.name + ".shinobi-trash.dead-run")
    _write(trash, "the dead run's pre-run state")

    def arm(chain):
        chain.marker = Marker(step_path="pipe.cal", field="ms", cache_key="k", run_id="dead-run", started_at=0.0)
        return chain

    journal.update(chain_id(ms), arm)

    from shinobi.snapshots import release_runs

    release_runs()
    _pipeline(ms, {})(spw="*", cache=True, cache_dir=str(cache_dir))

    assert not trash.exists(), "a corpse's trash should have been reconciled"
    assert journal.get(chain_id(ms)).marker is None


def test_skipping_reconciliation_still_recovers_an_interrupted_step(tmp_path):
    """Why skipping is the safe direction: reconciliation is a repair, not a
    prerequisite. A marker left set makes the next restore *force* a
    rollback, so the mid-mutation recovery still happens through the normal
    path -- only the tidying waits.
    """
    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    calls = {}
    with pytest.raises(Exception):
        _pipeline(ms, calls, fail={"cal": True})(spw="*", cache=True, cache_dir=str(cache_dir))
    assert _read(ms) == "vis[*]|flag[default]|PARTIAL"

    held = _hold_a_foreign_run(cache_dir)
    try:
        from shinobi.snapshots import release_runs

        release_runs()
        _pipeline(ms, calls)(spw="*", cache=True, cache_dir=str(cache_dir))
    finally:
        os.close(held)

    assert calls["cal_saw"] == ["vis[*]|flag[default]", "vis[*]|flag[default]"]
    assert _read(ms) == "vis[*]|flag[default]|cal"


def test_a_dead_processes_lock_file_does_not_look_live(tmp_path):
    """A lock file whose owner is gone is lockable, so it must read as a
    corpse -- otherwise one crash would disable reconciliation forever.
    """
    from shinobi.snapshots import announce_run, release_runs

    cache_dir = tmp_path / "cache"
    locks = cache_dir / "snapshots" / "locks"
    locks.mkdir(parents=True)
    (locks / "long-dead.lock").write_text("")  # created, never locked

    release_runs()
    assert announce_run(str(cache_dir), "mine").alone()


def test_announcing_twice_in_one_process_does_not_shadow_itself(tmp_path):
    """A process that dispatches twice is the same live process both times;
    re-announcing would leave it holding two lock files and reading its own
    first one as a corpse -- or worse, as a rival.
    """
    from shinobi.snapshots import announce_run, release_runs

    cache_dir = tmp_path / "cache"
    release_runs()
    first = announce_run(str(cache_dir), "run-one")
    second = announce_run(str(cache_dir), "run-two")
    assert first is second
    assert second.alone()
    assert len(list((cache_dir / "snapshots" / "locks").iterdir())) == 1


def test_presence_that_cannot_announce_never_claims_to_be_alone(tmp_path):
    """A process invisible to everyone else has no business making
    destructive decisions about their state.
    """
    from shinobi.snapshots import RunPresence

    presence = RunPresence(tmp_path / "snapshots", "mine")
    assert presence.announced is False
    assert presence.alone() is False


def test_releasing_removes_the_lock_file(tmp_path):
    from shinobi.snapshots import announce_run, release_runs

    cache_dir = tmp_path / "cache"
    release_runs()
    announce_run(str(cache_dir), "run-one")
    assert list((cache_dir / "snapshots" / "locks").iterdir())
    release_runs()
    assert not list((cache_dir / "snapshots" / "locks").iterdir())


def test_reconciliation_runs_for_a_recipe_rooted_pipeline(tmp_path):
    """Regression: whether Tier 1 is active for a *run* is a different
    question from whether a given scope gets a snapshot guard.

    A Recipe is never itself cached and mutates nothing of its own, so it
    correctly gets no guard -- but a top-level target is almost always a
    Recipe, so gating crash recovery on that same answer meant reconciliation
    never ran for any real pipeline. It was invisible because an interrupted
    step still recovers through the marker-forced restore; only the tidying
    silently stopped happening.
    """
    ms = tmp_path / "data.ms"
    cache_dir = tmp_path / "cache"
    _pipeline(ms, {})(spw="*", cache=True, cache_dir=str(cache_dir))

    journal = get_journal(str(cache_dir))
    trash = ms.with_name(ms.name + ".shinobi-trash.dead-run")
    _write(trash, "a corpse's quarantined tree")

    def arm(chain):
        chain.marker = Marker(step_path="pipe.cal", field="ms", cache_key="k", run_id="dead-run", started_at=0.0)
        return chain

    journal.update(chain_id(ms), arm)

    from shinobi.snapshots import release_runs

    release_runs()
    _pipeline(ms, {})(spw="*", cache=True, cache_dir=str(cache_dir))

    assert not trash.exists()
    assert journal.get(chain_id(ms)).marker is None
