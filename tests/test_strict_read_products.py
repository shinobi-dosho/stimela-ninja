"""Real-MS regressions for fixed ordinary products and external read authority."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from shinobi import DatasetAccess, MeasurementSetV2, pystep
from shinobi.dataset_access import DatasetAccessError, resolve_scope_dataset_accesses
from shinobi.dataset_lifecycle import DatasetLifecyclePhase
from shinobi.exceptions import DatasetLifecycleUnavailableError, DatasetLifecycleViolationError
from shinobi.steps.schema import Mutability, Scope
from tests._dataset_fixtures import attempts, make_ms, ms_fixture_tables, set_scans


class Report(BaseModel):
    report: Path


class ReadInputs(BaseModel):
    ms: MeasurementSetV2
    report: Path


class MSInputs(BaseModel):
    ms: MeasurementSetV2


class Empty(BaseModel):
    pass


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    return tmp_path


@pytest.mark.parametrize("cache", [False, True])
def test_fixed_report_echo_is_claimed_without_snapshot_taint(work, cache):
    ms = make_ms(work / "source.ms")
    report = work / "report.txt"
    called = []

    @pystep(write_paths=["report"], dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2, report: Path) -> Report:
        called.append(ms)
        report.write_text("report")
        return Report(report=report)

    for _ in range(2):
        assert read(ms=ms, report=report, cache=cache).success
    assert len(called) == (1 if cache else 2)
    for attempt in attempts(work):
        assert attempt.phase is DatasetLifecyclePhase.COMMITTED
        assert any(Path(access.path) == ms and not access.writes for access in attempt.claim.accesses)
        assert any(Path(access.path) == report and access.writes for access in attempt.claim.accesses)
        assert all(not leaf.mutations for leaf in attempt.leaves)
    assert not list((work / ".shinobi").glob("**/inflight*"))


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("redirect", ["other", "symlink", "dataset"])
def test_actual_report_redirect_refused_with_or_without_cache(work, cache, redirect):
    ms = make_ms(work / "source.ms")
    report = work / "report.txt"

    @pystep(write_paths=["report"], dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2, report: Path) -> Report:
        if redirect == "symlink":
            report.symlink_to(ms, target_is_directory=True)
            return Report(report=report)
        report.write_text("report")
        return Report(report=ms if redirect == "dataset" else report.with_name("other.txt"))

    with pytest.raises(DatasetLifecycleViolationError, match="frozen destination"):
        read(ms=ms, report=report, cache=cache)
    assert all(attempt.phase is not DatasetLifecyclePhase.COMMITTED for attempt in attempts(work))


@pytest.mark.parametrize("destination", ["below", "ancestor", "alias"])
def test_report_cannot_overlap_any_dataset_resource(work, destination):
    ms = make_ms(work / "source.ms")
    if destination == "below":
        report = ms / "report.txt"
    elif destination == "ancestor":
        report = work
    else:
        report = work / "alias"
        report.symlink_to(ms, target_is_directory=True)

    @pystep(write_paths=["report"], dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2, report: Path) -> Report:
        pytest.fail("unsafe destination launched")

    with pytest.raises((DatasetLifecycleUnavailableError, DatasetAccessError), match="mutation|overlaps|filesystem write"):
        read(ms=ms, report=report)


@pytest.mark.parametrize("explicit", [False, True])
def test_same_name_report_requires_creation_and_nonmutable_input(work, explicit):
    ms = make_ms(work / "source.ms")

    @pystep(write_paths=["report"] if explicit else [], dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2, report: Path) -> Report:
        pytest.fail("unsafe mutation launched")

    if explicit:
        read.step.input_mutability["report"] = Mutability.MUTABLE
    with pytest.raises(DatasetLifecycleUnavailableError, match="in-place generic mutation"):
        read(ms=ms, report=work / "report.txt")


def test_manual_scope_and_precommit_outputs_obey_same_frozen_destination(work):
    from shinobi.steps.dispatch import _dispatch
    from shinobi.results import StepResult

    ms = make_ms(work / "source.ms")
    from shinobi import Recipe
    from shinobi.steps.schema import StepRef, InputRef

    class Destination(BaseModel):
        report: Path

    scope = Scope(name="manual", inputs_model=Destination, outputs_model=Report, field_meta={"report": {"write_path": True}})

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        pass

    def run(ctx):
        return StepResult(kind="python", name="manual", returncode=0, inputs=ctx.inputs, outputs=Report(report=work / "other.txt"))

    recipe = Recipe(name="manual-report", inputs_model=ReadInputs, outputs_model=Empty)
    recipe.add_step("read", read, ms=InputRef(field="ms"))
    recipe.steps.append(StepRef(name="manual", step=scope, func=run, wiring={"report": InputRef(field="report")}))
    with pytest.raises(DatasetLifecycleViolationError, match="frozen destination"):
        _dispatch(recipe, None, ms=ms, report=work / "report.txt", cache=False)


@pytest.mark.parametrize("external", [False, True])
def test_sibling_external_read_requires_opt_in(work, external):
    launch = work / "launch"
    launch.mkdir()
    ms = make_ms(work / "source.ms")

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read", external_read=external)])
    def read(ms: MeasurementSetV2) -> None:
        assert ms.name == "source.ms"

    import os

    os.chdir(launch)
    if external:
        assert read(ms=ms, cache=False).success
        assert attempts(launch)[0].pre_observations[0].closure.storage_namespace == ms
    else:
        with pytest.raises(DatasetAccessError, match="namespace"):
            read(ms=ms, cache=False)


def test_external_read_still_refuses_read_mutation(work):
    launch = work / "launch"
    launch.mkdir()
    ms = make_ms(work / "source.ms")

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read", external_read=True)])
    def read(ms: MeasurementSetV2) -> None:
        set_scans(ms, 9)

    import os

    os.chdir(launch)
    with pytest.raises(DatasetLifecycleViolationError):
        read(ms=ms, cache=False)
    assert attempts(launch)[0].phase is not DatasetLifecyclePhase.COMMITTED


@pytest.mark.parametrize("external_first", [False, True])
def test_closure_cache_keeps_namespace_authority_separate(work, external_first):
    launch = work / "launch"
    launch.mkdir()
    ms = make_ms(work / "source.ms")
    cache = {}
    for external in (external_first, not external_first):
        scope = Scope(name="read", inputs_model=MSInputs, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="ms", mode="read", external_read=external)])
        if external:
            assert resolve_scope_dataset_accesses(scope, {"ms": ms}, workspace=launch, closure_cache=cache)[0].root == ms
        else:
            with pytest.raises(DatasetAccessError, match="namespace"):
                resolve_scope_dataset_accesses(scope, {"ms": ms}, workspace=launch, closure_cache=cache)
    assert len(cache) == 2


def test_external_read_flag_is_strict_read_only_and_compatible():
    assert "external_read" not in DatasetAccess(field="ms", mode="read").model_dump()
    declaration = DatasetAccess(field="ms", mode="read", external_read=True)
    assert DatasetAccess.model_validate_json(declaration.model_dump_json()) == declaration
    for mode in ("write", "create"):
        with pytest.raises(ValidationError, match="requires READ"):
            DatasetAccess(field="ms", mode=mode, external_read=True)
    with pytest.raises(ValidationError):
        DatasetAccess(field="ms", mode="read", external_read="true")


@pytest.mark.parametrize("override", [False, True])
def test_nullable_cab_report_is_fixed_before_launch(work, monkeypatch, override):
    import sys
    from shinobi import Cab
    from shinobi.steps.dispatch import _dispatch
    from shinobi.steps.schema import ParamMeta
    from shinobi.backends.recording import RecordingBackend

    class Inputs(MSInputs):
        report: Path | None = None

    class Outputs(BaseModel):
        report: Path | None = None

    ms = make_ms(work / "source.ms")
    report = work / "report.txt"
    cab = Cab(
        name="report",
        command=sys.executable,
        inputs_model=Inputs,
        outputs_model=Outputs,
        field_meta={"report": ParamMeta(write_path=True)},
        dataset_accesses=[DatasetAccess(field="ms", mode="read")],
    )
    backend = RecordingBackend()
    monkeypatch.setattr("shinobi.steps.dispatch.get_step_backend", lambda *_: backend)
    if override:
        # Cab orchestration is excluded from dataset admission, so exercise the
        # launcher boundary directly with an independent claim-time snapshot.
        from shinobi.steps.dispatch import _run_cab, _snapshot_product_inputs

        with pytest.raises(DatasetLifecycleUnavailableError, match="frozen reservation"):
            _run_cab(
                cab,
                {"ms": ms, "report": work / "other.txt"},
                backend,
                dataset_resources={ms},
                planned_inputs=_snapshot_product_inputs(cab, {"ms": ms, "report": report}, workspace=work),
            )
        assert not backend.calls
    else:
        result = _dispatch(cab, None, ms=ms, report=report, cache=False)
        assert result.success and result.outputs.report == report
        assert len(backend.calls) == 1


def test_cached_redirect_and_precommit_redirect_are_refused(work, monkeypatch):
    from shinobi.cache import get_cache_manifest
    from shinobi.steps.dispatch import _dispatch

    ms = make_ms(work / "source.ms")
    report = work / "report.txt"
    other = work / "other.txt"
    other.write_text("other")
    calls = []

    @pystep(write_paths=["report"], dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2, report: Path) -> Report:
        calls.append(ms)
        report.write_text("report")
        return Report(report=report)

    cache_dir = str(work / "cache")
    assert read(ms=ms, report=report, cache=True, cache_dir=cache_dir).success
    manifest = get_cache_manifest(cache_dir)
    original = manifest.check

    def tampered(*args, **kwargs):
        hit = original(*args, **kwargs)
        assert hit is not None
        hit.outputs.report = other
        return hit

    monkeypatch.setattr(manifest, "check", tampered)
    with pytest.raises(DatasetLifecycleViolationError, match="frozen destination"):
        read(ms=ms, report=report, cache=True, cache_dir=cache_dir)
    assert len(calls) == 1

    def precommit(result):
        result.outputs.report = other

    with pytest.raises(DatasetLifecycleViolationError, match="frozen destination"):
        _dispatch(read.step, read.func, ms=ms, report=report, cache=False, _result_precommit=precommit)
    assert attempts(work)[-1].phase is not DatasetLifecyclePhase.COMMITTED


def test_external_root_must_close_inside_itself_even_with_shared_workspace(work):
    import shutil

    tables = ms_fixture_tables()

    ms = make_ms(work / "source.ms")
    external = work / "antenna"
    shutil.move(ms / "ANTENNA", external)
    with tables.table(str(ms), readonly=False, ack=False) as main:
        main.putkeyword("ANTENNA", f"Table: {external}")
    scope = Scope(name="read", inputs_model=MSInputs, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="ms", mode="read", external_read=True)])
    with pytest.raises(DatasetAccessError, match="namespace"):
        resolve_scope_dataset_accesses(scope, {"ms": ms}, workspace=work)


def test_external_reader_claim_excludes_writer_and_container_is_readonly(work):
    from shinobi.dataset_backends import plan_dataset_backend
    from shinobi.ownership import WorkspaceOwnershipError, acquire_workspace

    ms = make_ms(work / "source.ms")
    launch = work / "launch"
    launch.mkdir()
    scope = Scope(name="read", inputs_model=MSInputs, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="ms", mode="read", external_read=True)])
    accesses = resolve_scope_dataset_accesses(scope, {"ms": ms}, workspace=launch)
    lease = acquire_workspace(launch, "reader", kind="local", accesses=[(path, False) for path in accesses[0].resources])
    try:
        with pytest.raises(WorkspaceOwnershipError):
            acquire_workspace(work, "writer", kind="local", accesses=[(ms, True)])
    finally:
        lease.release()
    plan = plan_dataset_backend("docker", accesses, workspace=launch, mutation=False)
    assert [(mount.source, mount.writable) for mount in plan.mounts] == [(ms, False)]


@pytest.mark.parametrize("redirect", [False, True])
def test_qualified_worker_external_source_and_nullable_report(work, redirect):
    import sys
    from shinobi import Cab, Recipe
    from shinobi.steps.schema import InputRef, ParamMeta, StepRef
    from shinobi.offload.worker import execute_step
    from tests.test_offload_dataset_lifecycle import _prepared, _record

    launch = work / "launch"
    launch.mkdir()
    ms = make_ms(work / "source.ms")
    report = launch / "report.txt"
    returned = launch / "other.txt" if redirect else report

    fixed_report = report

    class Inputs(MSInputs):
        report: Path | None = fixed_report
        script: str

    class Outputs(BaseModel):
        report: Path | None = None

    cab = Cab(
        name="external-reader",
        command=f"{sys.executable} -c",
        inputs_model=Inputs,
        outputs_model=Outputs,
        field_meta={"script": ParamMeta(positional_head=True), "ms": ParamMeta(positional=True), "report": ParamMeta(positional=True, write_path=True)},
        dataset_accesses=[DatasetAccess(field="ms", mode="read", external_read=True)],
        wranglers={r"^REPORT=(?P<report>.+)$": ["PARSE_OUTPUT:report:str"]},
    )
    script = "from pathlib import Path;import sys;Path(sys.argv[2]).write_text('report');print('REPORT=' + " + repr(str(returned)) + ")"
    recipe = Recipe(
        name="external-worker",
        inputs_model=MSInputs,
        outputs_model=Empty,
        cache_dir=str(launch / "cache"),
        steps=[StepRef(name="read", step=cab, params={"script": script}, wiring={"ms": InputRef(field="ms")})],
    )
    workflow, plan, lease = _prepared(launch, recipe, ms, qualification_root=work)
    try:
        assert plan.dataset_lifecycle.initial_snapshot.observations[0].closure.storage_namespace == ms
        attempt = plan.attempts[0]
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == (1 if redirect else 0)
        record = _record(workflow, plan)
        assert record.committed is not redirect
    finally:
        lease.release()
    with pytest.raises(ValidationError, match="outside qualified shared storage"):
        _prepared(launch, recipe, ms, qualification_root=launch)


def test_strict_writer_report_is_not_a_scientific_snapshot_participant(work):
    from shinobi import DatasetColumns

    ms = make_ms(work / "source.ms")
    report = work / "report.txt"

    @pystep(write_paths=["report"], dataset_accesses=[DatasetAccess(field="ms", mode="write", columns=DatasetColumns(write=("SCAN_NUMBER",)))])
    def write(ms: MeasurementSetV2, report: Path) -> Report:
        set_scans(ms, 8)
        report.write_text("updated")
        return Report(report=report)

    assert write(ms=ms, report=report, cache=True).success
    [attempt] = attempts(work)
    [leaf] = attempt.leaves
    assert {mutation.field for mutation in leaf.mutations} == {"ms"}
    assert not leaf.auxiliary_mutations


def test_external_flag_roundtrips_through_static_loader(tmp_path):
    from shinobi.loaders.yaml_cab import load_file

    path = tmp_path / "read.yml"
    path.write_text("""read:
  command: read-ms
  inputs:
    ms:
      dtype: MSv2
  dataset_accesses:
    - field: ms
      mode: read
      external_read: true
""")
    cab = load_file(path)["read"]
    assert cab.dataset_accesses[0].external_read is True


def test_ctx_run_cannot_borrow_another_leafs_report_reservation(work, monkeypatch):
    from shinobi import Cab, Recipe
    from shinobi.steps.schema import InputRef, ParamMeta, StepRef
    from shinobi.backends.recording import RecordingBackend

    class Destination(BaseModel):
        report: Path

    ms = make_ms(work / "source.ms")
    backend = RecordingBackend()
    monkeypatch.setattr("shinobi.steps.dispatch.get_step_backend", lambda *_: backend)

    @pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    def read(ms: MeasurementSetV2) -> None:
        pass

    cab = Cab(name="report", command="report-tool", inputs_model=Destination, outputs_model=Report, field_meta={"report": ParamMeta(write_path=True)})
    first = work / "first.txt"
    second = work / "second.txt"

    def redirect(ctx):
        return ctx.run(report=second)

    recipe = Recipe(name="cross-leaf", inputs_model=MSInputs, outputs_model=Empty)
    recipe.add_step("read", read, ms=InputRef(field="ms"))
    recipe.steps.extend(
        [
            StepRef(name="first", step=cab, func=redirect, params={"report": first}),
            StepRef(name="second", step=cab, params={"report": second}),
        ]
    )
    with pytest.raises(DatasetLifecycleUnavailableError, match="frozen reservation"):
        recipe(ms=ms, cache=False)
    assert not backend.calls


@pytest.mark.parametrize("route", ["cab", "venv"])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("target", ["report.txt", "other.txt", "source.ms"])
def test_sandbox_returned_paths_are_checked_before_harvest(work, make_venv, route, cache, target):
    import sys
    from shinobi import Cab
    from shinobi.cache import get_cache_manifest
    from shinobi.datasets import DatasetStatus, inspect_measurement_set_v2
    from shinobi.steps.schema import ParamMeta

    ms = make_ms(work / "source.ms")
    original = (ms / "table.dat").read_bytes()
    other = work / "other.txt"
    other.write_text("ORIGINAL")
    report = work / "report.txt"
    cache_dir = str(work / "cache")
    accesses = [DatasetAccess(field="ms", mode="read")]
    if route == "cab":

        class Inputs(ReadInputs):
            script: str

        reader = Cab(
            name="sandbox-report",
            command=f"{sys.executable} -c",
            inputs_model=Inputs,
            outputs_model=Report,
            field_meta={"script": ParamMeta(positional_head=True), "report": ParamMeta(write_path=True)},
            dataset_accesses=accesses,
            wranglers={r"^REPORT=(?P<report>.+)$": ["PARSE_OUTPUT:report:str"]},
        )
        script = (
            "from pathlib import Path;p=Path("
            + repr(target)
            + ");"
            + ("p.mkdir();(p/'table.dat').write_text('REPLACED');" if target == "source.ms" else "p.write_text('REPLACED');")
            + "print('REPORT=' + str(p))"
        )
        kwargs = {"script": script}
    else:
        from tests._venv_pystep_funcs import strict_sandbox_report

        reader = pystep(name="sandbox-report", backend="venv", venv=str(make_venv()), write_paths=["report"], dataset_accesses=accesses)(strict_sandbox_report)
        kwargs = {"target": target}

    def run():
        return reader(ms=ms, report=report, sandbox=True, cache=cache, cache_dir=cache_dir, **kwargs)

    if target == "report.txt":
        assert run().success
        assert report.read_text() == "REPLACED"
    else:
        with pytest.raises(DatasetLifecycleViolationError, match="frozen destination"):
            run()
        assert get_cache_manifest(cache_dir).entry("sandbox-report") is None
        assert attempts(work)[-1].phase is not DatasetLifecyclePhase.COMMITTED
    assert other.read_text() == "ORIGINAL"
    assert (ms / "table.dat").read_bytes() == original
    assert inspect_measurement_set_v2(ms).status is DatasetStatus.VALID


@pytest.mark.parametrize("route", ["cab", "venv"])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("alias", ["parent", "direct", "rebind"])
def test_sandbox_source_namespace_alias_is_refused_before_move(work, make_venv, monkeypatch, route, cache, alias):
    import sys
    from shinobi import Cab
    from shinobi.cache import get_cache_manifest
    from shinobi.datasets import DatasetStatus, inspect_measurement_set_v2
    from shinobi.steps.schema import ParamMeta

    # Retain the previous declared report so the test detects any publication
    # before refusal, independently of the authorized pre-run clearing policy.
    monkeypatch.setenv("SHINOBI_EXECUTION__CLEAR_STALE_OUTPUTS", "false")
    ms = make_ms(work / "source.ms")
    original = (ms / "table.dat").read_bytes()
    report = work / ("table.dat" if alias == "parent" else "report.txt")
    report.write_text("ORIGINAL REPORT")
    accesses = [DatasetAccess(field="ms", mode="read")]
    cache_dir = str(work / "cache")
    if route == "cab":

        class Inputs(ReadInputs):
            script: str

        reader = Cab(
            name="alias-report",
            command=f"{sys.executable} -c",
            inputs_model=Inputs,
            outputs_model=Report,
            field_meta={"script": ParamMeta(positional_head=True), "report": ParamMeta(write_path=True)},
            dataset_accesses=accesses,
            wranglers={r"^REPORT=(?P<report>.+)$": ["PARSE_OUTPUT:report:str"]},
        )
        if alias == "parent":
            script = "from pathlib import Path;Path('link').symlink_to(" + repr(str(ms / "ANTENNA")) + ",target_is_directory=True);print('REPORT=link/../table.dat')"
        else:
            target = str(ms) if alias == "direct" else "source.ms"
            script = "from pathlib import Path;Path('report.txt').symlink_to(" + repr(target) + ",target_is_directory=True);print('REPORT=report.txt')"
        kwargs = {"script": script}
    else:
        from tests._venv_pystep_funcs import strict_sandbox_alias

        reader = pystep(name="alias-report", backend="venv", venv=str(make_venv()), write_paths=["report"], dataset_accesses=accesses)(strict_sandbox_alias)
        kwargs = {"alias": alias}
    with pytest.raises(DatasetLifecycleViolationError, match="strict harvest"):
        reader(ms=ms, report=report, sandbox=True, cache=cache, cache_dir=cache_dir, **kwargs)
    assert (ms / "table.dat").read_bytes() == original
    assert inspect_measurement_set_v2(ms).status is DatasetStatus.VALID
    assert report.read_text() == "ORIGINAL REPORT"
    assert not report.is_symlink()
    assert get_cache_manifest(cache_dir).entry("alias-report") is None
    assert attempts(work)[-1].phase is not DatasetLifecyclePhase.COMMITTED


def test_strict_harvest_preflights_all_targets_before_first_move(work):
    from shinobi.sandbox import harvest_outputs

    class Outputs(BaseModel):
        first: Path
        second: Path

    ms = make_ms(work / "source.ms")
    original = (ms / "table.dat").read_bytes()
    sandbox = work / "scratch"
    sandbox.mkdir()
    (work / "first.txt").write_text("ORIGINAL")
    (sandbox / "first.txt").write_text("REPLACED")
    (sandbox / "link").symlink_to(ms / "ANTENNA", target_is_directory=True)
    scope = Scope(name="joint-products", inputs_model=Empty, outputs_model=Outputs)
    outputs = Outputs(first=Path("first.txt"), second=Path("link/../table.dat"))
    with pytest.raises(DatasetLifecycleViolationError, match="escapes its sandbox"):
        harvest_outputs(scope, outputs, {}, sandbox, work, dataset_resources={ms})
    assert (work / "first.txt").read_text() == "ORIGINAL"
    assert (sandbox / "first.txt").read_text() == "REPLACED"
    assert (ms / "table.dat").read_bytes() == original


def test_sandboxed_strict_writer_echo_is_a_host_direct_passthrough(work):
    import sys
    from shinobi import Cab, DatasetColumns
    from shinobi.steps.schema import ParamMeta

    class Inputs(MSInputs):
        script: str

    ms = make_ms(work / "source.ms")
    cab = Cab(
        name="writer-echo",
        command=f"{sys.executable} -c",
        inputs_model=Inputs,
        outputs_model=MSInputs,
        field_meta={"script": ParamMeta(positional_head=True)},
        dataset_accesses=[DatasetAccess(field="ms", mode="write", columns=DatasetColumns(write=("SCAN_NUMBER",)))],
    )
    assert cab(ms=ms, script="pass", sandbox=True, cache=True, cache_dir=str(work / "cache")).success
    [attempt] = attempts(work)
    [leaf] = attempt.leaves
    assert {mutation.field for mutation in leaf.mutations} == {"ms"}


def test_sandboxed_strict_create_output_is_a_host_direct_passthrough(work):
    import sys
    from shinobi import Cab
    from shinobi.steps.schema import ParamMeta
    from shinobi.datasets import DatasetStatus, inspect_measurement_set_v2

    ms_fixture_tables()

    class Inputs(BaseModel):
        script: str
        target: Path

    target = work / "created.ms"
    cab = Cab(
        name="create-echo",
        command=f"{sys.executable} -c",
        inputs_model=Inputs,
        outputs_model=MSInputs,
        field_meta={"script": ParamMeta(positional_head=True), "target": ParamMeta(positional=True), "ms": ParamMeta(implicit="{target}")},
        dataset_accesses=[DatasetAccess(field="ms", mode="create")],
    )
    script = "from casacore import tables;import sys;t=tables.default_ms(sys.argv[1]);t.addcols(tables.maketabdesc([tables.makearrcoldesc('DATA',0j,ndim=2)]));t.close()"
    assert cab(target=target, script=script, sandbox=True, cache=True, cache_dir=str(work / "cache")).success
    assert inspect_measurement_set_v2(target).status is DatasetStatus.VALID
    [attempt] = attempts(work)
    [leaf] = attempt.leaves
    assert {mutation.field for mutation in leaf.mutations} == {"ms"}
