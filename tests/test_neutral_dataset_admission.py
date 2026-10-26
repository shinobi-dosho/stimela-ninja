"""Neutral Path annotations retain their explicit strict dataset authority."""

from pathlib import Path

import dosho
import pytest
from pydantic import BaseModel

from shinobi.cabs import adapt_definition
from shinobi.cache import get_cache_manifest
from shinobi.exceptions import DatasetLifecycleUnavailableError, DatasetLifecycleViolationError
from shinobi.dataset_access import DatasetAccessError
from tests._dataset_fixtures import ROWS, attempts, make_ms, scans, set_scans


class Output(BaseModel):
    ms: Path


class Report(BaseModel):
    report: Path | None = None


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    return tmp_path


def access(mode, **kwargs):
    return {"field": "ms", "mode": mode, **kwargs}


@pytest.mark.parametrize("mode", ["write", "create"])
def test_neutral_path_contract_executes_and_reuses_claim(work, mode):
    calls = []

    @dosho.pystep(dataset_accesses=[access(mode, columns=None if mode != "write" else {"write": ["SCAN_NUMBER"]})])
    def operation(ms: Path) -> Output:
        calls.append(ms)
        if mode == "create":
            make_ms(ms)
        elif mode == "write":
            set_scans(ms, 7)
        return Output(ms=ms)

    step = adapt_definition(operation)
    ms = work / "source.ms" if mode == "create" else make_ms(work / "source.ms")
    for _ in range(1 if mode == "create" else 2):
        assert step(ms=ms, cache=True, cache_dir=str(work / "cache")).outputs.ms == ms
    assert len(calls) == 1
    assert scans(ms) == [7 if mode == "write" else 1] * ROWS
    assert all(attempt.outcome == "committed" for attempt in attempts(work))
    if mode == "create":
        with pytest.raises(DatasetAccessError, match="already exists"):
            step(ms=ms, cache=True, cache_dir=str(work / "cache"))


@pytest.mark.parametrize("optional", [False, True])
def test_neutral_reader_fixed_report_and_absent_optional_output(work, optional):
    @dosho.pystep(write_paths=["report"], dataset_accesses=[access("read")])
    def read(ms: Path, report: Path | None = None) -> Report:
        if report is not None:
            report.write_text("report")
        return Report(report=report)

    ms = make_ms(work / "source.ms")
    report = None if optional else work / "report.txt"
    assert adapt_definition(read)(ms=ms, report=report).success


@pytest.mark.parametrize("mode", ["write", "create"])
def test_neutral_returned_dataset_redirect_is_refused(work, mode):
    @dosho.pystep(dataset_accesses=[access(mode)])
    def operation(ms: Path) -> Output:
        if mode == "create":
            make_ms(ms)
        return Output(ms=work / "unclaimed.ms")

    ms = work / "source.ms" if mode == "create" else make_ms(work / "source.ms")
    with pytest.raises(DatasetLifecycleViolationError, match="claimed path"):
        adapt_definition(operation)(ms=ms, cache=True, cache_dir=str(work / "cache"))
    if mode == "create":
        assert not ms.exists()


def test_neutral_cached_and_precommit_redirects_are_refused(work, monkeypatch):
    from shinobi.steps.dispatch import _dispatch

    @dosho.pystep(dataset_accesses=[access("write")])
    def read(ms: Path) -> Output:
        return Output(ms=ms)

    step = adapt_definition(read)
    ms = make_ms(work / "source.ms")
    cache_dir = str(work / "cache")
    assert step(ms=ms, cache=True, cache_dir=cache_dir).success
    manifest = get_cache_manifest(cache_dir)
    original = manifest.check

    def redirect(*args, **kwargs):
        result = original(*args, **kwargs)
        result.outputs.ms = work / "unclaimed.ms"
        return result

    monkeypatch.setattr(manifest, "check", redirect)
    with pytest.raises(DatasetLifecycleViolationError, match="claimed path"):
        step(ms=ms, cache=True, cache_dir=cache_dir)

    monkeypatch.setattr(manifest, "check", original)

    def precommit(result):
        result.outputs.ms = work / "unclaimed.ms"

    with pytest.raises(DatasetLifecycleViolationError, match="claimed path"):
        _dispatch(step.step, step.func, ms=ms, cache=True, cache_dir=str(work / "precommit-cache"), _result_precommit=precommit)


@pytest.mark.parametrize("alias", ["root", "descendant", "ancestor"])
def test_neutral_access_does_not_authorize_undeclared_alias(work, alias):
    @dosho.pystep(dataset_accesses=[access("read")])
    def read(ms: Path, other: Path) -> None:
        pytest.fail("undeclared alias launched")

    ms = make_ms(work / "source.ms")
    other = {"root": ms, "descendant": ms / "table.dat", "ancestor": work}[alias]
    with pytest.raises((DatasetLifecycleUnavailableError, DatasetAccessError), match="overlap"):
        adapt_definition(read)(ms=ms, other=other)


def test_neutral_undeclared_schema_write_rolls_back(work):
    from tests._dataset_fixtures import ms_fixture_tables

    @dosho.pystep(dataset_accesses=[access("write", columns={"write": ["SCAN_NUMBER"]})])
    def mutate(ms: Path) -> Output:
        set_scans(ms, 99)
        tables = ms_fixture_tables()
        with tables.table(str(ms), readonly=False, ack=False) as table:
            table.addcols(tables.maketabdesc([tables.makearrcoldesc("MODEL_DATA", 0j, ndim=2)]))
        return Output(ms=ms)

    ms = make_ms(work / "source.ms")
    with pytest.raises(DatasetLifecycleViolationError, match="columns changed"):
        adapt_definition(mutate)(ms=ms, cache=True, cache_dir=str(work / "cache"))
    assert scans(ms) == [1] * ROWS


@pytest.mark.parametrize("backend", ["native", "venv", "docker", "podman", "apptainer", "singularity"])
def test_neutral_external_read_preserves_backend_authority(work, monkeypatch, backend):
    from shinobi.dataset_access import resolve_scope_dataset_accesses
    from shinobi.dataset_backends import plan_dataset_backend
    from tests.test_dataset_lifecycle import _install_local_container_configuration

    @dosho.pystep(dataset_accesses=[access("read", external_read=True)])
    def read(ms: Path) -> None:
        assert ms.is_dir()

    ms = make_ms(work / "external.ms")
    launch = work / "launch"
    launch.mkdir()
    monkeypatch.chdir(launch)
    _install_local_container_configuration(monkeypatch, work)
    step = adapt_definition(read)
    assert step(ms=ms).success
    accesses = resolve_scope_dataset_accesses(step.step, {"ms": ms}, workspace=launch)
    assert all(not item.writes for item in accesses)
    plan = plan_dataset_backend(backend, accesses, workspace=launch, mutation=False)
    assert all(not mount.writable for mount in plan.mounts)
    if backend not in {"native", "venv"}:
        assert {mount.source for mount in plan.mounts} == {ms}


def test_neutral_unknown_unreserved_and_overlapping_declarations_refused(work):
    from shinobi.dataset_access import resolve_scope_dataset_accesses

    @dosho.pystep(dataset_accesses=[access("read")])
    def read(ms: Path) -> None:
        pytest.fail("unresolved access launched")

    step = adapt_definition(read)
    with pytest.raises(DatasetAccessError, match="unknown path and no reservation"):
        resolve_scope_dataset_accesses(step.step, {}, workspace=work, unresolved_inputs={"ms"})

    @dosho.pystep(dataset_accesses=[access("read")], write_paths=["other"])
    def conflicting(ms: Path, other: Path) -> None:
        pytest.fail("overlapping write launched")

    ms = make_ms(work / "source.ms")
    with pytest.raises((DatasetAccessError, DatasetLifecycleUnavailableError), match="write|overlap"):
        adapt_definition(conflicting)(ms=ms, other=ms / "table.dat")


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("target", ["dataset", "unclaimed"])
def test_absent_ordinary_output_cannot_gain_runtime_path(work, cache, target):
    @dosho.pystep(write_paths=["report"], dataset_accesses=[access("read")])
    def read(ms: Path, report: Path | None = None) -> Report:
        return Report(report=ms if target == "dataset" else work / "unclaimed.txt")

    ms = make_ms(work / "source.ms")
    with pytest.raises(DatasetLifecycleViolationError, match="frozen destination"):
        adapt_definition(read)(ms=ms, cache=cache, cache_dir=str(work / "cache"))


@pytest.mark.parametrize("route", ["cached", "precommit"])
@pytest.mark.parametrize("target", ["dataset", "unclaimed"])
def test_absent_ordinary_output_stays_absent_through_publication(work, monkeypatch, route, target):
    from shinobi.steps.dispatch import _dispatch

    @dosho.pystep(write_paths=["report"], dataset_accesses=[access("read")])
    def read(ms: Path, report: Path | None = None) -> Report:
        return Report(report=report)

    ms = make_ms(work / "source.ms")
    step = adapt_definition(read)
    cache_dir = str(work / "cache")
    assert step(ms=ms, cache=True, cache_dir=cache_dir).outputs.report is None
    assert step(ms=ms, cache=True, cache_dir=cache_dir).outputs.report is None
    other = ms if target == "dataset" else work / "unclaimed.txt"

    def precommit(result):
        result.outputs.report = other

    if route == "cached":
        manifest = get_cache_manifest(cache_dir)
        original = manifest.check

        def redirect(*args, **kwargs):
            result = original(*args, **kwargs)
            result.outputs.report = other
            return result

        monkeypatch.setattr(manifest, "check", redirect)
        with pytest.raises(DatasetLifecycleViolationError, match="frozen destination"):
            step(ms=ms, cache=True, cache_dir=cache_dir)
    else:
        with pytest.raises(DatasetLifecycleViolationError, match="frozen destination"):
            _dispatch(step.step, step.func, ms=ms, cache=False, _result_precommit=precommit)


@pytest.mark.parametrize("route", ["cached", "precommit"])
def test_missing_dataset_output_model_refused_before_success(work, monkeypatch, route):
    from shinobi.dataset_lifecycle import DatasetLifecyclePhase, DatasetMutationOutcome
    from shinobi.steps.dispatch import _dispatch

    @dosho.pystep(dataset_accesses=[access("write", columns={"write": ["SCAN_NUMBER"]})])
    def mutate(ms: Path) -> Output:
        set_scans(ms, 7)
        return Output(ms=ms)

    step = adapt_definition(mutate)
    ms = make_ms(work / "source.ms")
    cache_dir = str(work / "cache")
    if route == "cached":
        assert step(ms=ms, cache=True, cache_dir=cache_dir).success
        manifest = get_cache_manifest(cache_dir)
        original = manifest.check

        def missing(*args, **kwargs):
            result = original(*args, **kwargs)
            result.outputs = None
            return result

        monkeypatch.setattr(manifest, "check", missing)
        with pytest.raises(DatasetLifecycleViolationError, match="missing.*dataset output"):
            step(ms=ms, cache=True, cache_dir=cache_dir)
        assert scans(ms) == [7] * ROWS
    else:

        def precommit(result):
            result.outputs = None

        with pytest.raises(DatasetLifecycleViolationError, match="missing.*dataset output"):
            _dispatch(step.step, step.func, ms=ms, cache=True, cache_dir=cache_dir, _result_precommit=precommit)
        assert scans(ms) == [1] * ROWS
        assert get_cache_manifest(cache_dir).entry("mutate") is None
        assert attempts(work)[-1].leaves[0].mutations[0].outcome is DatasetMutationOutcome.ROLLED_BACK
    attempt = attempts(work)[-1]
    assert attempt.phase is DatasetLifecyclePhase.FAILED
    assert attempt.outcome != "committed"
    assert all(leaf.outcome != "committed" for leaf in attempt.leaves)


@pytest.mark.parametrize("route", ["cached", "precommit"])
def test_missing_output_model_allowed_when_no_dataset_outputs(work, monkeypatch, route):
    from shinobi.dataset_lifecycle import DatasetLifecyclePhase
    from shinobi.steps.dispatch import _dispatch

    @dosho.pystep(dataset_accesses=[access("read")])
    def read(ms: Path) -> None:
        assert ms.is_dir()

    step = adapt_definition(read)
    ms = make_ms(work / "source.ms")
    cache_dir = str(work / "cache")
    if route == "cached":
        assert step(ms=ms, cache=True, cache_dir=cache_dir).success
        manifest = get_cache_manifest(cache_dir)
        original = manifest.check

        def missing(*args, **kwargs):
            result = original(*args, **kwargs)
            result.outputs = None
            return result

        monkeypatch.setattr(manifest, "check", missing)
        result = step(ms=ms, cache=True, cache_dir=cache_dir)
    else:

        def precommit(result):
            result.outputs = None

        result = _dispatch(step.step, step.func, ms=ms, cache=False, _result_precommit=precommit)
    assert result.success and result.outputs is None
    assert attempts(work)[-1].phase is DatasetLifecyclePhase.COMMITTED
    assert scans(ms) == [1] * ROWS
