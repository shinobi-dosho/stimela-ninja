"""A finite selection is a planned path, never fabricated family membership."""

from pathlib import Path
import sys

import pytest
from pydantic import BaseModel, create_model

from shinobi import Cab, FamilySpec, FITSImage, MemberRule, ProductFamily
from shinobi.dataset_access import plan_recipe_accesses
from shinobi.exceptions import StepError
from shinobi.offload.records import AttemptRecord
from shinobi.offload.worker import execute_step
from shinobi.ownership import contained_access_issues, scope_path_accesses
from shinobi.products import FamilyPlan, UnresolvedProductReference, resolve_planned_reference
from shinobi.steps.schema import OutputRef, ParamMeta, planned_output_values, static_output_values, static_wiring_values, StepRef
from tests.test_offload_dataset_lifecycle import _install_dataset, _prepared, _recipe


class Empty(BaseModel):
    pass


def producer():
    inputs = create_model("ImageInputs", script=(str, ...), prefix=(str, "image"))
    outputs = create_model("ImageOutputs", image=(ProductFamily[FITSImage], ...))
    spec = FamilySpec(
        coordinates={"kind": "str", "cycle": ("str", "int")},
        rules=(
            MemberRule(path="{prefix}.restored.fits", coordinates={"kind": "restored", "cycle": "final"}),
            MemberRule(path="{prefix}.residual.fits", coordinates={"kind": "residual", "cycle": "final"}),
            MemberRule(path="{prefix}.residual{cycle}.fits", coordinates={"kind": "residual"}, captures={"cycle": "int"}),
        ),
    )
    return Cab(
        name="image",
        command=f"{sys.executable} -c",
        inputs_model=inputs,
        outputs_model=outputs,
        field_meta={"script": ParamMeta(positional_head=True), "image": ParamMeta(family=spec)},
    )


def recipe_with_selection(tmp_path, *, emit=True):
    recipe = _recipe(tmp_path, read=True)
    script = (
        (
            "from astropy.io import fits;"
            "[(lambda h,p:(h.header.update({'CTYPE1':'LINEAR','CTYPE2':'LINEAR'}),h.writeto(p,overwrite=True)))"
            "(fits.PrimaryHDU([[float(i)]]),'image.'+kind+'.fits') "
            "for i,kind in enumerate(('restored','residual'),1)]"
        )
        if emit
        else "pass"
    )
    recipe.add_step("image", producer(), script=script)
    consumer = Cab(
        name="compare",
        command=f"{sys.executable} -c",
        inputs_model=create_model("CompareInputs", script=(str, ...), images=(list[Path], ...), output=(str, ...)),
        outputs_model=create_model("CompareOutputs", output=(Path, ...)),
        field_meta={"script": ParamMeta(positional_head=True), "images": ParamMeta(positional=True, repeat_as_tokens=True), "output": ParamMeta(positional=True, write_path=True)},
    )
    recipe.add_step(
        "compare",
        consumer,
        images=[OutputRef(step="image", field="image").select(kind=kind, cycle="final") for kind in ("restored", "residual")],
        output=str(tmp_path / "comparison.txt"),
        script="import sys;from pathlib import Path;from astropy.io import fits;Path(sys.argv[-1]).write_text(','.join(str(float(fits.getdata(p)[0,0])) for p in sys.argv[1:-1]))",
    )
    return recipe


def test_finite_selection_plans_paths_without_resolving_membership(tmp_path):
    cab = producer()
    inputs = {"script": "pass", "prefix": "image"}
    assert "image" not in static_output_values(cab, inputs)
    outputs = planned_output_values(cab, inputs, workspace=tmp_path)
    assert isinstance(outputs["image"], FamilyPlan)
    source = OutputRef(step="image", field="image").select(kind="residual", cycle="final")
    ref = StepRef(name="reader", step=cab, wiring={"prefix": source})
    known, unresolved = static_wiring_values(ref, {}, {"image": outputs})
    assert known["prefix"] == tmp_path / "image.residual.fits"
    assert not unresolved and not known["prefix"].exists()


@pytest.mark.parametrize("selection", [None, {}, {"kind": "residual"}, {"kind": "residual", "cycle": 2}, {"kind": "missing"}, {"cycle": True}, {"unknown": "x"}])
def test_unknown_discovered_ambiguous_or_invalid_selection_stays_unresolved(tmp_path, selection):
    plan = planned_output_values(producer(), {"prefix": "image"}, workspace=tmp_path)["image"]
    with pytest.raises(UnresolvedProductReference):
        resolve_planned_reference(plan, selection)


def test_selected_paths_join_ownership_and_access_planning(tmp_path, monkeypatch):
    root = tmp_path / "obs.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    recipe = recipe_with_selection(tmp_path)
    inputs = {"ms": root}
    assert not contained_access_issues(recipe, inputs, workspace=tmp_path, dataset_resources={root})
    accesses, _ = scope_path_accesses(recipe, inputs, workspace=tmp_path)
    assert (tmp_path / "image.restored.fits", True) in accesses
    assert (tmp_path / "image.residual.fits", True) in accesses
    plan = plan_recipe_accesses(recipe, inputs, workspace=tmp_path)
    compare = plan.graph.names.index("compare")
    assert plan.graph.names.index("image") in plan.graph.deps[compare]


def test_single_finite_member_allows_an_empty_selector(tmp_path):
    cab = producer()
    cab.field_meta["image"].family = FamilySpec(coordinates={"kind": "str"}, rules=(MemberRule(path="only.fits", coordinates={"kind": "image"}),))
    plan = planned_output_values(cab, {}, workspace=tmp_path)["image"]
    assert resolve_planned_reference(plan, {}) == tmp_path / "only.fits"


def test_capture_can_make_a_finite_selection_ambiguous(tmp_path):
    cab = producer()
    cab.field_meta["image"].family = FamilySpec(
        coordinates={"kind": "str", "cycle": "str"},
        rules=(
            MemberRule(path="final.fits", coordinates={"kind": "image", "cycle": "final"}),
            MemberRule(path="discovered-{cycle}.fits", coordinates={"kind": "image"}, captures={"cycle": "str"}),
        ),
    )
    plan = planned_output_values(cab, {}, workspace=tmp_path)["image"]
    with pytest.raises(UnresolvedProductReference, match="discovered"):
        resolve_planned_reference(plan, {"kind": "image", "cycle": "final"})


def test_selected_output_inside_the_ms_is_still_refused(tmp_path, monkeypatch):
    root = tmp_path / "obs.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    recipe = recipe_with_selection(tmp_path)
    recipe.steps[1].params["prefix"] = str(root / "image")
    assert contained_access_issues(recipe, {"ms": root}, workspace=tmp_path, dataset_resources={root})


def test_unknown_producer_inputs_do_not_use_family_defaults(tmp_path):
    cab = producer()
    outputs = planned_output_values(cab, {"script": "pass", "prefix": "default"}, workspace=tmp_path, unresolved_inputs={"prefix"})
    assert "image" not in outputs


@pytest.mark.parametrize("worker", [False, True])
def test_selected_fits_handoff_executes_in_contained_ms_workflow(tmp_path, monkeypatch, worker):
    pytest.importorskip("astropy")
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "obs.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    recipe = recipe_with_selection(tmp_path)
    if worker:
        workflow, plan, lease = _prepared(tmp_path, recipe, root, qualification_root=tmp_path.parent)
        try:
            for name in ("tool", "image", "compare"):
                attempt = plan.attempt(name)
                assert execute_step(workflow.submission_dir, name, attempt.attempt_id) == 0
                record = AttemptRecord.model_validate_json((workflow.submission_dir / "attempts" / str(attempt.attempt_id) / "final.json").read_text())
                assert record.committed
        finally:
            lease.release()
    else:
        result = recipe(ms=root, backend="native", cache=True, cache_dir=str(tmp_path / "cache"))
        assert result.returncode == 0
        hit = recipe(ms=root, backend="native", cache=True, cache_dir=str(tmp_path / "cache"))
        assert hit.sub_results["image"].cached and hit.sub_results["compare"].cached
    assert (tmp_path / "comparison.txt").read_text() == "1.0,2.0"
    assert (root / "table.dat").read_text() == "raw"


def test_unemitted_selected_candidate_is_not_fabricated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "obs.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    recipe = recipe_with_selection(tmp_path, emit=False)
    with pytest.raises(StepError, match="selection is unavailable"):
        recipe(ms=root, backend="native", cache=False)
    assert not (tmp_path / "comparison.txt").exists()
