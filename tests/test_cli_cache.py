"""`ninja cache check/invalidate/evict`, and `clean`'s refusal to orphan a
quarantined tree.
"""

import json
from pathlib import Path

import shinobi
from click.testing import CliRunner
from pydantic import BaseModel

from shinobi.cli import main
from shinobi.snapshots import Marker, chain_id, get_journal
from shinobi.steps import InputRef, OutputRef, Recipe


class MsOut(BaseModel):
    ms: Path


class SpwInputs(BaseModel):
    spw: str = "*"


def _write(ms: Path, text: str) -> None:
    ms.mkdir(exist_ok=True)
    (ms / "table.dat").write_text(text)


def _run_chain(tmp_path, monkeypatch, strategy="default"):
    """A two-step in-place chain, run to completion under a cache dir the
    CLI will find through config.
    """
    ms = tmp_path / "data.ms"
    cache = tmp_path / "cache"
    monkeypatch.setenv("SHINOBI_CACHE__DIR", str(cache))

    @shinobi.pystep()
    def split(ctx, spw: str = "*") -> MsOut:
        _write(ms, f"vis[{spw}]")
        return MsOut(ms=ms)

    @shinobi.pystep()
    def flag(ctx, ms: Path, strategy: str = "default") -> MsOut:
        _write(ms, (ms / "table.dat").read_text() + f"|flag[{strategy}]")
        return MsOut(ms=ms)

    Recipe(
        name="pipe",
        inputs_model=SpwInputs,
        outputs_model=MsOut,
        steps=[
            split.model_copy(update={"wiring": {"spw": InputRef(field="spw")}}),
            flag.model_copy(update={"wiring": {"ms": OutputRef(step="split", field="ms")}, "params": {"strategy": strategy}}),
        ],
        output_wiring={"ms": OutputRef(step="flag", field="ms")},
    )(spw="*", cache=True, cache_dir=str(cache))
    return ms, cache


def test_check_on_a_clean_cache_reports_nothing(tmp_path, monkeypatch):
    _run_chain(tmp_path, monkeypatch)
    result = CliRunner().invoke(main, ["cache", "check"])
    assert result.exit_code == 0, result.output
    assert "nothing to report" in result.output


def test_check_reports_an_interrupted_step(tmp_path, monkeypatch):
    ms, cache = _run_chain(tmp_path, monkeypatch)
    journal = get_journal(str(cache))

    def arm(chain):
        chain.marker = Marker(step_path="pipe.flag", field="ms", cache_key="k", run_id="dead-run", started_at=0.0)
        return chain

    journal.update(chain_id(ms), arm)

    result = CliRunner().invoke(main, ["cache", "check"])
    assert result.exit_code == 0, result.output
    assert "never finished" in result.output
    assert "pipe.flag" in result.output


def test_check_reports_the_clone_capability_it_chose(tmp_path, monkeypatch):
    """When the space arithmetic looks wrong on some future deployment, the
    answer should be in this report rather than in a debug re-run.
    """
    _run_chain(tmp_path, monkeypatch)
    result = CliRunner().invoke(main, ["cache", "check"])
    assert "clone capability by filesystem" in result.output


def test_invalidate_removes_the_entry_and_rolls_the_chain_back(tmp_path, monkeypatch):
    ms, cache = _run_chain(tmp_path, monkeypatch)
    assert "pipe.flag" in json.loads((cache / "manifest.json").read_text())

    result = CliRunner().invoke(main, ["cache", "invalidate", "pipe.flag"])
    assert result.exit_code == 0, result.output
    assert "removed the manifest entry" in result.output
    assert "untrusted" in result.output
    assert "pipe.flag" not in json.loads((cache / "manifest.json").read_text())


def test_invalidate_of_an_unknown_step_says_so(tmp_path, monkeypatch):
    _run_chain(tmp_path, monkeypatch)
    result = CliRunner().invoke(main, ["cache", "invalidate", "pipe.nonexistent"])
    assert result.exit_code == 0, result.output
    assert "nothing recorded" in result.output


def test_evict_keeps_everything_a_live_chain_still_names(tmp_path, monkeypatch):
    _run_chain(tmp_path, monkeypatch)
    result = CliRunner().invoke(main, ["cache", "evict", "--bytes", "1000000000"])
    assert result.exit_code == 0, result.output
    assert "nothing evictable" in result.output


def test_clean_refuses_to_orphan_a_quarantined_tree(tmp_path, monkeypatch):
    """The journal is the only thing that says what a quarantined tree was
    quarantined for, and these trees are typically the biggest things in the
    workspace -- so removing the journal while one is outstanding is refused.
    """
    ms, cache = _run_chain(tmp_path, monkeypatch)
    journal = get_journal(str(cache))
    trash = ms.with_name(ms.name + ".shinobi-trash.deadrun")
    _write(trash, "quarantined")

    def arm(chain):
        chain.marker = Marker(step_path="pipe.flag", field="ms", cache_key="k", run_id="deadrun", started_at=0.0)
        return chain

    journal.update(chain_id(ms), arm)

    result = CliRunner().invoke(main, ["clean", "--no-runs", "--no-sandboxes"])
    assert result.exit_code != 0
    assert "refusing" in result.output
    assert cache.exists() and trash.exists()


def test_clean_force_removes_the_cache_and_the_quarantined_tree(tmp_path, monkeypatch):
    ms, cache = _run_chain(tmp_path, monkeypatch)
    journal = get_journal(str(cache))
    trash = ms.with_name(ms.name + ".shinobi-trash.deadrun")
    _write(trash, "quarantined")

    def arm(chain):
        chain.marker = Marker(step_path="pipe.flag", field="ms", cache_key="k", run_id="deadrun", started_at=0.0)
        return chain

    journal.update(chain_id(ms), arm)

    result = CliRunner().invoke(main, ["clean", "--no-runs", "--no-sandboxes", "--force"])
    assert result.exit_code == 0, result.output
    assert not cache.exists()
    assert not trash.exists()


def test_clean_still_works_when_nothing_is_quarantined(tmp_path, monkeypatch):
    _ms, cache = _run_chain(tmp_path, monkeypatch)
    result = CliRunner().invoke(main, ["clean", "--no-runs", "--no-sandboxes"])
    assert result.exit_code == 0, result.output
    assert not cache.exists()
