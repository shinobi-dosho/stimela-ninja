"""Declaration-aware strict dataset product families and execution paths."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

import shinobi.steps.dispatch as dispatch_module
from shinobi import Cab, DatasetAccess, MeasurementSetV2
from shinobi.dataset_access import DatasetColumns
from shinobi.dataset_lifecycle import DatasetLifecyclePhase
from shinobi.loaders import build_model
from shinobi.offload.worker import execute_step
from shinobi.ownership import WorkspaceOwnershipError, acquire_workspace, contained_access_issues, scope_path_accesses
from shinobi.sandbox import create_sandbox
from shinobi.steps.schema import ParamMeta
from tests._dataset_fixtures import ROWS, attempts, make_ms, scans
from tests.test_offload_dataset_lifecycle import _install_dataset, _prepared, _recipe, _record


class Empty(BaseModel):
    pass


WRITE_SCANS = DatasetAccess(field="ms", mode="write", columns=DatasetColumns(write=("SCAN_NUMBER",)))


def run_kwargs(tmp_path: Path, **extra) -> dict:
    return {"cache": True, "cache_dir": str(tmp_path / "cache"), **extra}


@pytest.mark.parametrize("pattern", ["out-*", "out-?.fits", "out-[ab].fits", "out.fits", "products/out-*", "{prefix}-*"])
def test_contained_bounded_products_keep_parent_reservations(tmp_path, pattern):
    class Inputs(BaseModel):
        prefix: str

    leaf = Cab(name="image", command="/bin/true", inputs_model=Inputs, outputs_model=Empty, harvest=[pattern])
    values = {"prefix": str(tmp_path / "out")}
    assert contained_access_issues(leaf, values, workspace=tmp_path, dataset_resources={tmp_path / "obs.ms"}) == ()
    accesses, _ = scope_path_accesses(leaf, values, workspace=tmp_path)
    expected = tmp_path / "products" if pattern.startswith("products/") else tmp_path
    assert (expected, True) in accesses
    lease = acquire_workspace(tmp_path, "products", kind="slurm", accesses=accesses)
    with pytest.raises(WorkspaceOwnershipError):
        acquire_workspace(tmp_path, "other", kind="slurm", accesses=[(expected / "unrelated", True)])
    lease.release()


@pytest.mark.parametrize("pattern", ["*", "obs*", "obs.ms/out-*", "**/out-*", "products/*/out-*", "../out-*", "{missing}-*", "{prefix}-*", ""])
def test_contained_products_refuse_unbounded_or_overlapping_declarations(tmp_path, pattern):
    class Inputs(BaseModel):
        prefix: str | None = None

    leaf = Cab(name="image", command="/bin/true", inputs_model=Inputs, outputs_model=Empty, scratch=[pattern])
    issues = contained_access_issues(leaf, {"prefix": None}, workspace=tmp_path, dataset_resources={tmp_path / "obs.ms"})
    assert issues and f"scratch pattern {pattern!r}" in issues[0]


@pytest.mark.parametrize("existing", [True, False])
@pytest.mark.parametrize("alias_kind", ["directory", "basename"])
def test_contained_product_symlinks_protect_existing_and_planned_roots(tmp_path, existing, alias_kind):
    protected = tmp_path / "obs.ms"
    if existing:
        protected.mkdir()
    if alias_kind == "directory":
        (tmp_path / "alias").symlink_to(protected, target_is_directory=True)
        pattern = "alias/out-*"
    else:
        (tmp_path / "out-alias").symlink_to(protected, target_is_directory=True)
        pattern = "out-*"
    leaf = Cab(name="image", command="/bin/true", inputs_model=Empty, outputs_model=Empty, harvest=[pattern])
    assert contained_access_issues(leaf, {}, workspace=tmp_path, dataset_resources={protected})


@pytest.mark.parametrize("writer", ["input", "default", "implicit", "list", "same-name"])
def test_safe_pattern_never_exempts_ordinary_writer_with_same_parent(tmp_path, writer):
    class Inputs(BaseModel):
        target: Path

    class Outputs(BaseModel):
        target: Path = tmp_path

    class ListOutputs(BaseModel):
        target: list[Path] = [tmp_path]

    leaf = Cab(
        name="image",
        command="/bin/true",
        inputs_model=Inputs if writer in ("input", "same-name") else Empty,
        outputs_model=Empty if writer == "input" else ListOutputs if writer == "list" else Outputs,
        harvest=["{target}/out-*"] if writer == "input" else ["out-*"],
        field_meta={"target": ParamMeta(write_path=True)} if writer == "input" else {"target": ParamMeta(implicit=str(tmp_path))} if writer == "implicit" else {},
    )
    assert any(
        "generic write overlaps" in issue
        for issue in contained_access_issues(leaf, {"target": tmp_path} if writer in ("input", "same-name") else {}, workspace=tmp_path, dataset_resources={tmp_path / "obs.ms"})
    )


@pytest.mark.parametrize("sandbox,prefix", [(False, "out"), (True, "out"), (False, "products/out"), (True, "products/out"), (False, "absolute"), (True, "absolute")])
def test_wsclean_shaped_list_mutation_with_bounded_products(tmp_path, monkeypatch, sandbox, prefix):
    import sys
    from shinobi.ownership import WorkspaceOwnershipError, acquire_workspace, inspect_workspace, ownership_workspace
    from shinobi.steps.schema import ParamMeta

    monkeypatch.chdir(tmp_path)
    roots = [make_ms(tmp_path / "first.ms"), make_ms(tmp_path / "second.ms")]
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    prefix = str(tmp_path / "absolute") if prefix == "absolute" else prefix

    class Inputs(BaseModel):
        script: str
        prefix: str
        ms: list[MeasurementSetV2]

    class Outputs(BaseModel):
        ms: list[MeasurementSetV2]

    script = (
        "from pathlib import Path;import sys;from casacore.tables import table;prefix=sys.argv[1];[Path(prefix+'-'+suffix).write_text('product') for suffix in ('image.fits','model.fits','junk')];"
        "\nfor name in sys.argv[2:]:\n with table(name,readonly=False,ack=False) as ms: ms.putcol('SCAN_NUMBER',ms.getcol('SCAN_NUMBER')+1)"
    )
    cab = Cab(
        name="wsclean-shaped",
        command=f"{sys.executable} -c",
        inputs_model=Inputs,
        outputs_model=Outputs,
        field_meta={"script": ParamMeta(positional_head=True), "prefix": ParamMeta(positional=True), "ms": ParamMeta(positional=True, repeat_as_tokens=True)},
        dataset_accesses=[WRITE_SCANS],
        harvest=["{prefix}-*.fits"],
        scratch=["{prefix}-junk"],
        sandbox=sandbox,
    )
    original_run = dispatch_module._run_cab

    def run_with_claim(*args, **kwargs):
        parent = Path(prefix).parent
        parent = parent if parent.is_absolute() else tmp_path / parent
        owner = inspect_workspace(ownership_workspace(tmp_path, [parent, *roots]))
        assert any(access.writes and Path(access.path) == parent for access in owner.accesses)
        with pytest.raises(WorkspaceOwnershipError):
            acquire_workspace(tmp_path, "other-writer", kind="slurm", accesses=[(parent / "sibling", True)])
        return original_run(*args, **kwargs)

    monkeypatch.setattr(dispatch_module, "_run_cab", run_with_claim)
    result = cab(ms=roots, prefix=prefix, script=script, backend="native", **run_kwargs(tmp_path))
    assert result.success
    assert [scans(root) for root in roots] == [[2] * ROWS, [2] * ROWS]
    for suffix in ("image.fits", "model.fits"):
        assert Path(prefix + "-" + suffix).read_text() == "product"
    assert Path(prefix + "-junk").exists() is (not sandbox or Path(prefix).is_absolute())
    assert attempts(tmp_path)[-1].phase is DatasetLifecyclePhase.COMMITTED


@pytest.mark.parametrize("location", ["inside", "equal", "alias"])
def test_contained_sandbox_refuses_protected_root_before_creation(tmp_path, location):
    from shinobi.exceptions import DatasetLifecycleUnavailableError

    protected = tmp_path / "planned.ms"
    root = protected / "scratch" if location == "inside" else protected
    if location == "alias":
        root = tmp_path / "alias"
        root.symlink_to(protected, target_is_directory=True)
    with pytest.raises(DatasetLifecycleUnavailableError, match="sandbox root.*MSv2 closure"):
        create_sandbox(str(root), "tool", dataset_resources={protected})
    assert not protected.exists()


def test_contained_sandbox_accepts_existing_ancestor_with_fresh_sibling(tmp_path):
    protected = tmp_path / "obs.ms"
    protected.mkdir()
    sandbox = create_sandbox(str(tmp_path), "tool", dataset_resources={protected})
    assert sandbox.parent == tmp_path and sandbox != protected


def test_actual_anchored_product_pattern_is_rechecked_before_preparing_parent(tmp_path, monkeypatch):
    from shinobi.exceptions import DatasetLifecycleUnavailableError
    from shinobi.steps.dispatch import _run_cab

    class Inputs(BaseModel):
        prefix: Path

    protected = tmp_path / "obs.ms"
    protected.mkdir()
    # Model a changed execution value at the path-anchoring boundary: both
    # the workspace declaration and the actual prepared source need proof.
    scope = Cab(name="tool", command="/bin/true", inputs_model=Inputs, outputs_model=build_model("Out", {}), harvest=["{prefix}/out-*"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dispatch_module, "absolutize_path_inputs", lambda *_args: {"prefix": protected})
    with pytest.raises(DatasetLifecycleUnavailableError, match="harvest pattern"):
        _run_cab(scope, {"prefix": Path("products")}, "native", sandbox_root=str(tmp_path / "sandbox"), dataset_resources={protected})
    assert not (protected / "out-").exists()


def test_worker_bounded_products_keep_broad_claim_and_publish_success(tmp_path, monkeypatch):
    root = tmp_path / "obs.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    recipe = _recipe(tmp_path)
    leaf = recipe.steps[0]
    leaf.step.harvest = ["out-*.fits"]
    leaf.step.scratch = ["out-junk"]
    leaf.params["script"] = (
        leaf.params["script"].rstrip(";") + ";Path('out-image.fits').write_text('image');Path('out-model.fits').write_text('model');Path('out-junk').write_text('scratch')"
    )
    workflow, plan, lease = _prepared(tmp_path, recipe, root)
    assert any(Path(access.path) == tmp_path and access.writes for access in plan.accesses)
    try:
        attempt = plan.attempt("tool")
        assert execute_step(workflow.submission_dir, attempt.step_path, attempt.attempt_id) == 0
        record = _record(workflow, plan)
        assert record.committed
        assert (tmp_path / "out-image.fits").read_text() == "image"
        assert (tmp_path / "out-model.fits").read_text() == "model"
        assert not (tmp_path / "out-junk").exists()
    finally:
        lease.release()


def test_flat_recipe_product_leaf_checks_all_other_leaf_resources(tmp_path):
    from shinobi.steps.schema import Recipe, StepRef

    class MSInputs(BaseModel):
        ms: list[MeasurementSetV2]

    reader = Cab(name="reader", command="/bin/true", inputs_model=MSInputs, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    products = Cab(name="products", command="/bin/true", inputs_model=Empty, outputs_model=Empty, harvest=["out-*"])
    roots = [tmp_path / "obs.ms", tmp_path / "out-future.ms"]
    recipe = Recipe(name="whole", inputs_model=Empty, outputs_model=Empty, steps=[StepRef(name="read", step=reader, params={"ms": roots}), StepRef(name="products", step=products)])
    issues = contained_access_issues(recipe, {}, workspace=tmp_path, dataset_resources=set(roots))
    assert len(issues) == 1 and "scope 'products' harvest pattern 'out-*'" in issues[0]
    assert str(roots[1]) in issues[0]


def test_exact_issue_pattern_harvests_dynamic_family(tmp_path, monkeypatch):
    import sys
    from shinobi.steps.dispatch import _run_cab

    class Inputs(BaseModel):
        script: str
        prefix: str

    monkeypatch.chdir(tmp_path)
    cab = Cab(
        name="products", command=f"{sys.executable} -c", inputs_model=Inputs, outputs_model=Empty, field_meta={"script": ParamMeta(positional_head=True)}, harvest=["{prefix}-*"]
    )
    result = _run_cab(
        cab,
        {"prefix": "out", "script": "from pathlib import Path;Path('out-unpredictable').write_text('product')"},
        "native",
        sandbox_root=str(tmp_path / "sandbox"),
        dataset_resources={tmp_path / "obs.ms"},
    )
    assert result.success and (tmp_path / "out-unpredictable").read_text() == "product"


def test_product_inspection_failure_names_declaration_and_refuses(tmp_path, monkeypatch):
    def denied(_path):
        raise PermissionError("cannot inspect directory")

    monkeypatch.setattr(Path, "iterdir", denied)
    leaf = Cab(name="products", command="/bin/true", inputs_model=Empty, outputs_model=Empty, harvest=["out-*"])
    issues = contained_access_issues(leaf, {}, workspace=tmp_path, dataset_resources={tmp_path / "obs.ms"})
    assert len(issues) == 1 and "harvest pattern 'out-*' cannot inspect" in issues[0]


def test_read_collision_diagnostic_names_unsafe_pattern(tmp_path, monkeypatch):
    from shinobi.dataset_access import DatasetAccessError, ResolvedAccessPlanner

    class Inputs(BaseModel):
        ms: MeasurementSetV2

    root = tmp_path / "obs.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    leaf = Cab(name="reader", command="/bin/true", inputs_model=Inputs, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="ms", mode="read")], harvest=["obs*"])
    with pytest.raises(DatasetAccessError, match="READ.*harvest pattern 'obs\\*'"):
        ResolvedAccessPlanner(tmp_path).order_after("reader", leaf, {"ms": root})


@pytest.mark.parametrize("launcher", ["native", "manual", "cab"])
@pytest.mark.parametrize("actual", ["obs.ms", "other"])
def test_runtime_product_parent_change_refuses_every_local_launcher(tmp_path, monkeypatch, launcher, actual):
    import sys
    from shinobi import Recipe, pystep
    from shinobi.exceptions import DatasetLifecycleUnavailableError
    from shinobi.steps.schema import OutputRef, Scope, StepRef

    class MSInputs(BaseModel):
        ms: MeasurementSetV2

    class Prefix(BaseModel):
        prefix: str = "out"

    class ProductInputs(BaseModel):
        script: str = ""
        prefix: str

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    root = make_ms(tmp_path / "obs.ms")
    launched = []

    def select() -> Prefix:
        return Prefix(prefix=actual)

    select.__annotations__["return"] = Prefix
    select = pystep()(select)

    if launcher == "native":

        @pystep(harvest=["{prefix}/out-*"])
        def produce(prefix: str) -> None:
            launched.append(True)
            Path(prefix).mkdir(exist_ok=True)
            Path(prefix, "out-test").write_text("unsafe")

        produce.wiring = {"prefix": OutputRef(step="select", field="prefix")}
    elif launcher == "manual":
        scope = Scope(name="produce", inputs_model=ProductInputs, outputs_model=Empty, harvest=["{prefix}/out-*"])

        def manual(ctx):
            launched.append(True)
            Path(ctx.inputs.prefix).mkdir(exist_ok=True)
            Path(ctx.inputs.prefix, "out-test").write_text("unsafe")

        produce = StepRef(name="produce", step=scope, func=manual, wiring={"prefix": OutputRef(step="select", field="prefix")})
    else:
        scope = Cab(
            name="produce",
            command=f"{sys.executable} -c",
            inputs_model=ProductInputs,
            outputs_model=Empty,
            field_meta={"script": ParamMeta(positional_head=True), "prefix": ParamMeta(positional=True)},
            harvest=["{prefix}/out-*"],
        )
        produce = StepRef(
            name="produce",
            step=scope,
            params={"script": "import sys;from pathlib import Path;Path(sys.argv[1],'out-test').write_text('unsafe')"},
            wiring={"prefix": OutputRef(step="select", field="prefix")},
        )
    reader = StepRef(
        name="read",
        step=Cab(name="read", command="/bin/true", inputs_model=MSInputs, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="ms", mode="read")]),
        params={"ms": root},
    )
    steps = [reader, select, produce]
    if actual == "other":
        # Another leaf owns the actual directory, but has no ordering edge
        # with the product's original "out" reservation. Workflow coverage
        # must not grant the product permission to change its own directory.
        from shinobi.results import StepResult

        class ClaimInputs(BaseModel):
            target: Path = tmp_path / "other"

        claimer = Scope(name="claimer", inputs_model=ClaimInputs, outputs_model=Empty, field_meta={"target": ParamMeta(write_path=True)})
        steps.append(StepRef(name="claimer", step=claimer, func=lambda ctx: StepResult(name="claimer", returncode=0, inputs=ctx.inputs.model_dump(), outputs=Empty())))
    recipe = Recipe(name="runtime-parent", inputs_model=Empty, outputs_model=Empty, steps=steps)
    with pytest.raises(DatasetLifecycleUnavailableError, match="harvest pattern.*(MSv2 closure|directory reservation)"):
        recipe(backend="native", cache=False, cache_dir=str(tmp_path / "cache"))
    assert launched == []
    assert not (tmp_path / actual / "out-test").exists()
    assert not (tmp_path / "other").exists()


def _runtime_product_recipe(tmp_path, *, actual, pattern="{prefix}/out-*"):
    import sys
    from shinobi import Recipe
    from shinobi.steps.schema import InputRef, OutputRef, StepRef

    class MSInputs(BaseModel):
        ms: MeasurementSetV2

    class Prefix(BaseModel):
        prefix: str = "out"

    class ProductInputs(BaseModel):
        script: str
        prefix: str

    class ScriptInputs(BaseModel):
        script: str

    select = Cab(
        name="select",
        command=f"{sys.executable} -c",
        inputs_model=ScriptInputs,
        outputs_model=Prefix,
        field_meta={"script": ParamMeta(positional_head=True)},
        wranglers={r"^PREFIX=(?P<prefix>\S+)$": ["PARSE_OUTPUT:prefix:str"]},
    )
    produce = Cab(
        name="produce",
        command=f"{sys.executable} -c",
        inputs_model=ProductInputs,
        outputs_model=Empty,
        field_meta={"script": ParamMeta(positional_head=True), "prefix": ParamMeta(positional=True)},
        harvest=[pattern],
    )
    reader = Cab(name="read", command="/bin/true", inputs_model=MSInputs, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    return Recipe(
        name="runtime-worker",
        inputs_model=MSInputs,
        outputs_model=Empty,
        cache_dir=str(tmp_path / "cache"),
        steps=[
            StepRef(name="read", step=reader, wiring={"ms": InputRef(field="ms")}),
            StepRef(name="select", step=select, params={"script": f"print('PREFIX={actual}')"}),
            StepRef(
                name="produce",
                step=produce,
                params={"script": "import sys;from pathlib import Path;Path(sys.argv[1],'out-test').write_text('unsafe')"},
                wiring={"prefix": OutputRef(step="select", field="prefix")},
            ),
        ],
    )


@pytest.mark.parametrize("actual", ["obs.ms", "other"])
def test_runtime_worker_product_parent_change_refuses_before_launch(tmp_path, monkeypatch, actual):
    root = tmp_path / "obs.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    recipe = _runtime_product_recipe(tmp_path, actual=actual)
    workflow, plan, lease = _prepared(tmp_path, recipe, root)
    try:
        for name in ("read", "select"):
            attempt = plan.attempt(name)
            assert execute_step(workflow.submission_dir, name, attempt.attempt_id) == 0
        attempt = plan.attempt("produce")
        assert execute_step(workflow.submission_dir, "produce", attempt.attempt_id) == 1
        assert not (tmp_path / actual / "out-test").exists()
        assert not (tmp_path / "other").exists()
        final = workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json"
        assert "harvest pattern" in final.read_text()
        assert "directory reservation" in final.read_text()
    finally:
        lease.release()


def test_runtime_basename_change_with_same_directory_reservation_is_allowed(tmp_path, monkeypatch):
    root = tmp_path / "obs.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    monkeypatch.chdir(tmp_path)
    recipe = _runtime_product_recipe(tmp_path, actual="out-new", pattern="{prefix}-*")
    recipe.steps[-1].params["script"] = "import sys;from pathlib import Path;Path(sys.argv[1]+'-test').write_text('safe')"
    assert recipe(ms=root, backend="native", cache=False).success
    assert (tmp_path / "out-new-test").read_text() == "safe"


def test_ctx_run_override_cannot_change_frozen_product_parent(tmp_path, monkeypatch):
    import sys
    from shinobi import Recipe
    from shinobi.exceptions import DatasetLifecycleUnavailableError
    from shinobi.steps.schema import StepRef

    class MSInputs(BaseModel):
        ms: MeasurementSetV2

    class Inputs(BaseModel):
        prefix: str = "out"
        script: str = "from pathlib import Path;Path('launched').touch()"

    root = make_ms(tmp_path / "obs.ms")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    reader = Cab(name="read", command="/bin/true", inputs_model=MSInputs, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    product = Cab(
        name="product", command=f"{sys.executable} -c", inputs_model=Inputs, outputs_model=Empty, harvest=["{prefix}/out-*"], field_meta={"script": ParamMeta(positional_head=True)}
    )
    recipe = Recipe(
        name="overrides",
        inputs_model=Empty,
        outputs_model=Empty,
        steps=[StepRef(name="read", step=reader, params={"ms": root}), StepRef(name="product", step=product, func=lambda ctx: ctx.run(prefix="other"))],
    )
    with pytest.raises(DatasetLifecycleUnavailableError, match="changed its planned product directory reservation"):
        recipe(backend="native", cache=False)
    assert not (tmp_path / "other").exists() and not (tmp_path / "launched").exists()


@pytest.mark.parametrize("nested", [False, True])
def test_atomic_context_deeply_freezes_reservations_before_input_mutation(tmp_path, monkeypatch, nested):
    import sys
    from types import SimpleNamespace
    from shinobi.exceptions import DatasetLifecycleUnavailableError
    from shinobi.steps.dispatch import ExecContext

    class Inputs(BaseModel):
        prefix: str = "out"
        settings: dict[str, list[str]] = {"targets": ["out"]}
        script: str = "from pathlib import Path;Path('launched').touch()"

    pattern = "{settings[targets][0]}/out-*" if nested else "{prefix}/out-*"
    cab = Cab(name="product", command=f"{sys.executable} -c", inputs_model=Inputs, outputs_model=Empty, harvest=[pattern], field_meta={"script": ParamMeta(positional_head=True)})
    monkeypatch.chdir(tmp_path)
    lifecycle = SimpleNamespace(workspace=tmp_path, record=SimpleNamespace(planned_accesses=[SimpleNamespace(resources=(tmp_path / "obs.ms",))]))
    # This models the original atomic lifecycle path: execution and planned
    # inputs initially reuse the exact same validated model.
    validated = Inputs()
    ctx = ExecContext(cab, validated.model_dump(), validated_inputs=validated, planned_inputs=validated, dataset_lifecycle=lifecycle)
    monkeypatch.setattr(ExecContext, "dataset_backend_plan", lambda *_args: None)
    assert ctx.inputs is validated
    if nested:
        ctx.inputs.settings["targets"][0] = "other"
    else:
        ctx.inputs.prefix = "other"
    with pytest.raises(DatasetLifecycleUnavailableError, match="changed its planned product directory reservation"):
        ctx.run(backend="native")
    assert not (tmp_path / "other").exists() and not (tmp_path / "launched").exists()


def test_recipe_callback_input_mutation_cannot_rewrite_its_reservation(tmp_path, monkeypatch):
    import sys
    from shinobi import Recipe
    from shinobi.exceptions import DatasetLifecycleUnavailableError
    from shinobi.steps.schema import StepRef

    class MSInputs(BaseModel):
        ms: MeasurementSetV2

    class Inputs(BaseModel):
        prefix: str = "out"
        script: str = "from pathlib import Path;Path('launched').touch()"

    root = make_ms(tmp_path / "obs.ms")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    reader = Cab(name="read", command="/bin/true", inputs_model=MSInputs, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    cab = Cab(
        name="product", command=f"{sys.executable} -c", inputs_model=Inputs, outputs_model=Empty, harvest=["{prefix}/out-*"], field_meta={"script": ParamMeta(positional_head=True)}
    )

    def mutate(ctx):
        ctx.inputs.prefix = "other"
        return ctx.run()

    recipe = Recipe(
        name="mutated", inputs_model=Empty, outputs_model=Empty, steps=[StepRef(name="read", step=reader, params={"ms": root}), StepRef(name="product", step=cab, func=mutate)]
    )
    with pytest.raises(DatasetLifecycleUnavailableError, match="changed its planned product directory reservation"):
        recipe(backend="native", cache=False)
    assert not (tmp_path / "other").exists() and not (tmp_path / "launched").exists()


def test_nested_recipe_ancestor_cannot_mutate_future_leaf_reservation(tmp_path, monkeypatch):
    import sys
    from shinobi import Recipe
    from shinobi.exceptions import DatasetLifecycleUnavailableError
    from shinobi.steps.schema import InputRef, Mutability, StepRef

    class Settings(BaseModel):
        targets: dict[str, list[str]] = {"names": ["out"]}

    class RecipeInputs(BaseModel):
        settings: Settings

    class ProductInputs(BaseModel):
        settings: Settings
        script: str = "from pathlib import Path;Path('launched').touch()"

    class MSInputs(BaseModel):
        ms: MeasurementSetV2

    root = make_ms(tmp_path / "obs.ms")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))
    settings = Settings()
    reader = Cab(name="read", command="/bin/true", inputs_model=MSInputs, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    cab = Cab(
        name="product",
        command=f"{sys.executable} -c",
        inputs_model=ProductInputs,
        outputs_model=Empty,
        harvest=["{settings.targets[names][0]}/out-*"],
        field_meta={"script": ParamMeta(positional_head=True)},
    )
    nested = Recipe(
        name="nested",
        inputs_model=RecipeInputs,
        outputs_model=Empty,
        input_mutability={"settings": Mutability.MUTABLE},
        steps=[StepRef(name="product", step=cab, wiring={"settings": InputRef(field="settings")})],
    )

    def mutate_ancestor(ctx):
        assert ctx.inputs.settings is settings
        ctx.inputs.settings.targets["names"][0] = "other"
        return ctx.run()

    recipe = Recipe(
        name="ancestor",
        inputs_model=RecipeInputs,
        outputs_model=Empty,
        input_mutability={"settings": Mutability.MUTABLE},
        steps=[StepRef(name="read", step=reader, params={"ms": root}), StepRef(name="nested", step=nested, func=mutate_ancestor, wiring={"settings": InputRef(field="settings")})],
    )
    with pytest.raises(DatasetLifecycleUnavailableError, match="changed its planned product directory reservation"):
        recipe(settings=settings, backend="native", cache=False)
    assert settings.targets["names"] == ["other"]  # execution mutability is unchanged
    assert not (tmp_path / "other").exists() and not (tmp_path / "launched").exists()


@pytest.mark.parametrize("shape", ["dict", "list", "model"])
@pytest.mark.parametrize("prefix", [None, "None", "products"])
def test_nested_template_values_reject_unset_without_string_heuristics(tmp_path, monkeypatch, shape, prefix):
    import sys
    from pydantic import create_model
    from shinobi import Recipe
    from shinobi.exceptions import DatasetLifecycleUnavailableError
    from shinobi.steps.schema import InputRef, StepRef

    class Settings(BaseModel):
        prefix: str | None

    class MSInputs(BaseModel):
        ms: MeasurementSetV2

    annotation, settings, field = {
        "dict": (dict[str, str | None], {"prefix": prefix}, "settings[prefix]"),
        "list": (list[str | None], [prefix], "settings[0]"),
        "model": (Settings, Settings(prefix=prefix), "settings.prefix"),
    }[shape]
    inputs = create_model("NestedTemplateInputs", settings=(annotation, ...), script=(str, ...))
    pattern = "{" + field + "}/out-*"
    product = Cab(
        name="products",
        command=f"{sys.executable} -c",
        inputs_model=inputs,
        outputs_model=Empty,
        field_meta={"script": ParamMeta(positional_head=True)},
        harvest=[pattern],
    )
    root = tmp_path / "obs.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    monkeypatch.chdir(tmp_path)
    reader = Cab(name="read", command="/bin/true", inputs_model=MSInputs, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    recipe = Recipe(
        name="nested-value",
        inputs_model=MSInputs,
        outputs_model=Empty,
        steps=[
            StepRef(name="read", step=reader, wiring={"ms": InputRef(field="ms")}),
            StepRef(
                name="products",
                step=product,
                params={"settings": settings, "script": f"from pathlib import Path;Path({str(prefix)!r},'out-result').write_text('product');Path('launched').touch()"},
            ),
        ],
    )
    if prefix is None:
        with pytest.raises(DatasetLifecycleUnavailableError, match="harvest pattern.*unresolved"):
            recipe(ms=root, backend="native", cache=False)
        assert not (tmp_path / "None").exists() and not (tmp_path / "launched").exists()
    else:
        assert recipe(ms=root, backend="native", cache=False).success
        assert (tmp_path / prefix / "out-result").read_text() == "product"


def test_product_template_preserves_escaped_braces_and_checks_nested_format_spec():
    from shinobi.steps.schema import _resolved_product_patterns

    scope = Cab(
        name="templates",
        command="/bin/true",
        inputs_model=Empty,
        outputs_model=Empty,
        harvest=["products/{{literal}}-{settings[prefix]}-*", "products/out-{count:0{settings[width]}d}-*"],
    )
    assert list(_resolved_product_patterns(scope, {"settings": {"prefix": "out", "width": 3}, "count": 2})) == [
        ("harvest pattern 'products/{{literal}}-{settings[prefix]}-*'", "products/{literal}-out-*"),
        ("harvest pattern 'products/out-{count:0{settings[width]}d}-*'", "products/out-002-*"),
    ]
    assert list(_resolved_product_patterns(scope, {"settings": {"prefix": "out", "width": None}, "count": 2}))[1][1] is None


@pytest.mark.parametrize("recipe", [False, True])
def test_ordinary_manual_inspection_does_not_prepare_noncopyable_inputs(tmp_path, monkeypatch, recipe):
    from pydantic import ConfigDict
    from shinobi import Recipe
    from shinobi.results import StepResult
    from shinobi.steps.schema import InputRef, Mutability, Scope, StepRef

    class Handle:
        def __deepcopy__(self, _memo):
            raise RuntimeError("inspection handle cannot be copied")

    class Inputs(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        handle: Handle

    monkeypatch.chdir(tmp_path)
    handle = Handle()
    called = []

    def inspect(ctx):
        called.append(ctx.inputs.handle)
        return StepResult(name="inspect", returncode=0, inputs={}, outputs=Empty())

    ref = StepRef(name="inspect", step=Scope(name="inspect", inputs_model=Inputs, outputs_model=Empty), func=inspect)
    target = Recipe(name="manual", inputs_model=Inputs, outputs_model=Empty, steps=[ref]) if recipe else ref
    if recipe:
        ref.wiring = {"handle": InputRef(field="handle")}
        target.input_mutability = {"handle": Mutability.MUTABLE}
    assert target(handle=handle, backend="native", cache=False, provenance=False).success
    assert called == [handle]


@pytest.mark.parametrize("mutation", ["replace", "append", "move-to-scratch", "remove-all", "reorder", "scratch"])
def test_callback_cannot_mutate_frozen_product_declarations(tmp_path, monkeypatch, mutation):
    import sys
    from shinobi import Recipe
    from shinobi.exceptions import DatasetLifecycleUnavailableError
    from shinobi.steps.schema import StepRef

    class MSInputs(BaseModel):
        ms: MeasurementSetV2

    class Inputs(BaseModel):
        script: str = "from pathlib import Path;Path('launched').touch()"

    root = tmp_path / "obs.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    monkeypatch.chdir(tmp_path)
    reader = Cab(name="read", command="/bin/true", inputs_model=MSInputs, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    product = Cab(
        name="product",
        command=f"{sys.executable} -c",
        inputs_model=Inputs,
        outputs_model=Empty,
        harvest=["out/out-*", "out/log-*"],
        scratch=["cache/tmp-*"],
        field_meta={"script": ParamMeta(positional_head=True)},
    )

    def mutate(ctx):
        if mutation == "replace":
            ctx.scope.harvest[:] = ["other/out-*", "other/log-*"]
        elif mutation == "append":
            ctx.scope.harvest.append("other/out-*")
        elif mutation == "move-to-scratch":
            ctx.scope.scratch.append(ctx.scope.harvest.pop(0))
        elif mutation == "remove-all":
            ctx.scope.harvest.clear()
            ctx.scope.scratch.clear()
        elif mutation == "reorder":
            ctx.scope.harvest.reverse()
        else:
            ctx.scope.scratch[:] = ["other/tmp-*"]
        return ctx.run()

    recipe = Recipe(
        name="mutable-declarations",
        inputs_model=Empty,
        outputs_model=Empty,
        steps=[StepRef(name="read", step=reader, params={"ms": root}), StepRef(name="product", step=product, func=mutate)],
    )
    with pytest.raises(DatasetLifecycleUnavailableError, match="changed its planned harvest/scratch declarations"):
        recipe(backend="native", cache=False)
    for name in ("other", "out", "cache", "launched"):
        assert not (tmp_path / name).exists()


def test_atomic_context_freezes_product_declarations_and_refuses_additions(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from shinobi.exceptions import DatasetLifecycleUnavailableError
    from shinobi.steps.dispatch import ExecContext

    class Inputs(BaseModel):
        script: str = "from pathlib import Path;Path('launched').touch()"

    monkeypatch.chdir(tmp_path)
    lifecycle = SimpleNamespace(workspace=tmp_path, record=SimpleNamespace(planned_accesses=[SimpleNamespace(resources=(tmp_path / "obs.ms",))]))
    for original in (["out/out-*"], []):
        cab = Cab(
            name="product", command=f"{sys.executable} -c", inputs_model=Inputs, outputs_model=Empty, harvest=original, field_meta={"script": ParamMeta(positional_head=True)}
        )
        validated = Inputs()
        ctx = ExecContext(cab, validated.model_dump(), validated_inputs=validated, planned_inputs=validated, dataset_lifecycle=lifecycle)
        monkeypatch.setattr(ExecContext, "dataset_backend_plan", lambda *_args: None)
        ctx.scope.harvest[:] = ["other/out-*"]
        with pytest.raises(DatasetLifecycleUnavailableError, match="planned harvest/scratch declarations|no frozen product directory reservations"):
            ctx.run(backend="native")
        assert not (tmp_path / "other").exists() and not (tmp_path / "launched").exists()


def test_nested_ancestor_cannot_replace_future_leaf_product_declarations(tmp_path, monkeypatch):
    import sys
    from shinobi import Recipe
    from shinobi.exceptions import DatasetLifecycleUnavailableError
    from shinobi.steps.schema import StepRef

    class MSInputs(BaseModel):
        ms: MeasurementSetV2

    class Inputs(BaseModel):
        script: str = "from pathlib import Path;Path('launched').touch()"

    root = tmp_path / "obs.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    monkeypatch.chdir(tmp_path)
    reader = Cab(name="read", command="/bin/true", inputs_model=MSInputs, outputs_model=Empty, dataset_accesses=[DatasetAccess(field="ms", mode="read")])
    product = Cab(
        name="product", command=f"{sys.executable} -c", inputs_model=Inputs, outputs_model=Empty, harvest=["out/out-*"], field_meta={"script": ParamMeta(positional_head=True)}
    )
    nested = Recipe(name="nested", inputs_model=Empty, outputs_model=Empty, steps=[StepRef(name="product", step=product)])

    def mutate_ancestor(ctx):
        ctx.scope.steps[0].step.harvest[:] = ["other/out-*"]
        return ctx.run()

    recipe = Recipe(
        name="ancestor-declarations",
        inputs_model=Empty,
        outputs_model=Empty,
        steps=[StepRef(name="read", step=reader, params={"ms": root}), StepRef(name="nested", step=nested, func=mutate_ancestor)],
    )
    with pytest.raises(DatasetLifecycleUnavailableError, match="changed its planned harvest/scratch declarations"):
        recipe(backend="native", cache=False)
    assert not (tmp_path / "other").exists() and not (tmp_path / "launched").exists()


def test_ordinary_patterned_cab_keeps_mutable_noncopyable_input_identity(tmp_path, monkeypatch):
    from pydantic import ConfigDict
    from shinobi.backends.recording import RecordingBackend
    from shinobi.steps.schema import Mutability

    class Handle:
        def __deepcopy__(self, _memo):
            raise RuntimeError("mutable backend handle cannot be copied")

    class Inputs(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        handle: Handle

    monkeypatch.chdir(tmp_path)
    backend = RecordingBackend()
    monkeypatch.setitem(dispatch_module._STEP_BACKENDS, "recording", backend)
    handle = Handle()
    cab = Cab(name="ordinary", command="/bin/true", inputs_model=Inputs, outputs_model=Empty, harvest=["out-*"], input_mutability={"handle": Mutability.MUTABLE})
    assert cab(handle=handle, backend="recording", cache=False, provenance=False).success
    assert backend.calls[0][2]["handle"] is handle
