"""Catalogue access promises are enforced by schemas, argv, plans and transport."""

from pathlib import Path
import json

import dosho
import dosho.cabs
import pytest
from pydantic import BaseModel, ValidationError

from shinobi.exceptions import ParameterError
from shinobi.cache import cab_semantic_identity, compute_cache_key
from shinobi.cabs import get, build_document, adapt_definition
from shinobi.dataset_access import DatasetMode, plan_recipe_accesses
from shinobi.offload._codec import BundleError, pack
from shinobi.offload.bundle import ScopeSpec
from shinobi.policies import build_argv
from shinobi.steps.schema import Recipe, validate_declared_writes
from tests.test_offload_dataset_lifecycle import _install_dataset


def arguments(name, root):
    if name == "wsclean-readonly":
        return {"ms": [root / "obs.ms"], "prefix": str(root / "ws/image"), "size": [16, 16], "scale": "1asec", "temp_dir": root / "ws-temp"}
    if name == "ddfacet-readonly":
        return {
            "data_ms": [root / "obs.ms"],
            "output_name": str(root / "dd/image"),
            "data_cache_dir": str(root / "dd-cache"),
            "cache_dir_wisdom_fftw": str(root / "dd-wisdom"),
            "montblanc_log_file": str(root / "dd-logs/log"),
        }
    return {"input_ms_path": root / "obs.ms", "output_gain_directory": str(root / "qc/gains"), "output_log_directory": str(root / "qc-logs")}


@pytest.mark.parametrize("name", ["wsclean-readonly", "ddfacet-readonly", "quartical-readonly"])
def test_readonly_routes_constraints_and_transport(name, tmp_path):
    definition = dosho.get(name)
    assert getattr(dosho.cabs, name.replace("-", "_")) is definition
    cab = adapt_definition(definition)
    assert cab_semantic_identity(get(name)) == cab_semantic_identity(cab)
    dialect, text = dosho.registry.get_document(name)
    direct = build_document(dialect, text, name=name, **dosho.registry.loader_options())
    assert cab_semantic_identity(direct) == cab_semantic_identity(cab)
    values = cab.inputs_model(**arguments(name, tmp_path)).model_dump()
    argv = build_argv(cab, values)
    assert all(a.mode is DatasetMode.READ for a in cab.dataset_accesses)
    frozen = ScopeSpec.capture(cab)
    restored = frozen.restore()
    assert build_argv(restored, values) == argv
    assert cab_semantic_identity(restored) == cab_semantic_identity(cab)
    if name == "wsclean-readonly":
        assert "-reorder" in argv and "-no-update-model-required" in argv
        for key in ["predict", "store_imaging_weights", "no_reorder", "update_model_required"]:
            with pytest.raises(ValidationError):
                cab.inputs_model(**{**values, key: True})
        for key in ["reorder", "no_update_model_required"]:
            with pytest.raises(ValidationError):
                cab.inputs_model(**{**values, key: False})
        with pytest.raises(ValidationError):
            cab.inputs_model(**{**values, "temp_dir": None})
        # Even unresolved supplied values cannot override the formatter constants.
        assert "-predict" not in build_argv(cab, {"predict": True})
    elif name == "ddfacet-readonly":
        for flag in ["--Predict-ColName", "--Weight-OutColName"]:
            assert argv[argv.index(flag) + 1] == ""
        for key, value in [("predict_col_name", "MODEL_DATA"), ("weight_out_col_name", "WEIGHT"), ("output_mode", "Predict"), ("parset", Path("bad.parset"))]:
            with pytest.raises(ValidationError):
                cab.inputs_model(**{**values, key: value})
    else:
        assert {"output.products=[]", "output.columns=[]", "output.flags=false"} <= set(argv)
        assert "ms" not in cab.outputs_model.model_fields
        for key, value in [("output_products", ["corrected"]), ("output_columns", ["CORRECTED_DATA"]), ("output_flags", True), ("parset", Path("bad.yaml"))]:
            with pytest.raises(ValidationError):
                restored.inputs_model(**{**values, key: value})
        assert "output.flags=false" in build_argv(cab, {"output_flags": True})


@pytest.mark.parametrize(
    "key", ["output.flags=true.G.type", "output.flags=true\\.type", "output.type", "input_ms.type", "G X.type", "G\n.type", "G.type\n", "output.flags", "output.products"]
)
def test_quartical_dynamic_keys_cannot_override_native_roots(key):
    cab = get("quartical-readonly")
    assert cab.match_pattern(key) is None
    argv = build_argv(cab, {key: "danger", "G.type": "complex", "K.time_interval": 10})
    assert "G.type=complex" in argv and "K.time_interval=10" in argv
    assert not any("danger" in arg for arg in argv)
    assert "output.flags=false" in argv


@pytest.mark.parametrize("name", ["wsclean-readonly", "ddfacet-readonly", "quartical-readonly"])
@pytest.mark.parametrize("destination", ["scratch", "output"])
def test_readonly_rejects_symlinked_output_or_scratch_into_ms(name, tmp_path, destination):
    root = tmp_path / "obs.ms"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    cab = get(name)
    values = arguments(name, tmp_path)
    key = {"wsclean-readonly": "temp_dir", "ddfacet-readonly": "data_cache_dir", "quartical-readonly": "output_log_directory"}[name]
    if destination == "output":
        key = {"wsclean-readonly": "prefix", "ddfacet-readonly": "output_name", "quartical-readonly": "output_gain_directory"}[name]
        values[key] = str(alias / "product")
    else:
        values[key] = str(alias)
    with pytest.raises(ParameterError, match="read.only|protected|write|overlaps"):
        validate_declared_writes(cab, cab.inputs_model(**values).model_dump(), tmp_path)


def test_imaging_readers_can_overlap_but_flaghistory_mutations_exclude_every_reader(tmp_path, monkeypatch):
    root = tmp_path / "obs.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)

    class Empty(BaseModel):
        pass

    recipe = Recipe(name="readers", inputs_model=Empty, outputs_model=Empty)
    producer_values = arguments("wsclean-readonly", tmp_path)
    producer_values.update(prefix=str(tmp_path / "producer/image"), temp_dir=tmp_path / "producer-temp")
    recipe.add_step("producer", dosho.get("wsclean"), **producer_values)
    for name in ["wsclean-readonly", "ddfacet-readonly", "quartical-readonly"]:
        recipe.add_step(name, dosho.get(name), **arguments(name, tmp_path))
    plan = plan_recipe_accesses(recipe, {}, workspace=tmp_path)
    assert not plan.graph.deps[0]
    assert all(deps == {0} for deps in plan.graph.deps[1:])
    for action in ["backup", "delete"]:
        from shinobi.steps.schema import StepRef

        recipe.steps.append(StepRef(name=action, step=get("msutils-flags-" + action), params={"ms": root, "name": "version"}))
    plan = plan_recipe_accesses(recipe, {}, workspace=tmp_path)
    assert set(range(1, 4)) <= plan.graph.deps[4]
    assert 4 in plan.graph.deps[5]
    for action in ["backup", "delete"]:
        cab = get("msutils-flags-" + action)
        assert cab.cache is False and cab.dataset_accesses[0].mode is DatasetMode.READ
        assert cab.dataset_accesses[0].exclusive
        assert not cab.dataset_accesses[0].columns.write
        assert "ms" not in cab.outputs_model.model_fields


@pytest.mark.parametrize("name", ["", ".", "..", ".hidden", "../bad", "a/b", "a\\b"])
def test_flag_delete_refuses_unsafe_name(name):
    with pytest.raises(ValidationError):
        get("msutils-flags-delete").inputs_model(ms="obs.ms", name=name)


def test_execution_access_identity_and_frozen_tampering(tmp_path):
    from shinobi import DatasetAccess
    from shinobi.loaders.yaml_cab import loads

    cab = loads("tool: {command: tool, inputs: {ms: {dtype: MSv2}}}")["tool"]
    read = cab.model_copy(update={"dataset_accesses": [DatasetAccess(field="ms", mode="read")]})
    write = cab.model_copy(update={"dataset_accesses": [DatasetAccess(field="ms", mode="write")]})
    assert compute_cache_key(read, None, {}) != compute_cache_key(write, None, {})
    assert build_argv(read, {}) == build_argv(write, {})
    frozen = ScopeSpec.capture(read)
    assert cab_semantic_identity(frozen.restore()) == cab_semantic_identity(read)
    settings = dict(frozen.settings)
    settings["dataset_accesses"] = pack(write.dataset_accesses)
    with pytest.raises(BundleError, match="semantics"):
        frozen.model_copy(update={"settings": settings}).restore()
    assert json.dumps(cab_semantic_identity(read)) != json.dumps(cab_semantic_identity(cab))


def test_readonly_input_metadata_and_fits_contract_are_cache_semantics():
    from shinobi.loaders.yaml_cab import loads
    from shinobi.steps.schema import ParamMeta
    from shinobi import FITSAccess

    cab = loads("tool: {command: tool, inputs: {path: {dtype: File}}}")["tool"]
    protected = cab.model_copy(update={"field_meta": {"path": ParamMeta(writable=False)}})
    assert cab_semantic_identity(cab) != cab_semantic_identity(protected)
    frozen = ScopeSpec.capture(protected)
    changed = dict(frozen.settings)
    changed["field_meta"] = pack({"path": ParamMeta().model_dump(mode="python")})
    with pytest.raises(BundleError, match="semantics"):
        frozen.model_copy(update={"settings": changed}).restore()
    # Output-only metadata stays in the product contract.
    output_only = cab.model_copy(update={"field_meta": {"unknown_output": ParamMeta(writable=False)}})
    assert cab_semantic_identity(cab) == cab_semantic_identity(output_only)
    fits = loads("tool: {command: tool, inputs: {image: {dtype: FITSImage}}}")["tool"]
    a = fits.model_copy(update={"fits_accesses": [FITSAccess(field="image", mode="read")]})
    b = fits.model_copy(update={"fits_accesses": [FITSAccess(field="image", mode="write")]})
    assert cab_semantic_identity(a) != cab_semantic_identity(b)


def test_readonly_quartical_log_destination_does_not_invalidate_its_own_key(tmp_path):
    from shinobi.cache import invalidate_path_hashes

    cab = get("quartical-readonly")
    values = cab.inputs_model(**arguments("quartical-readonly", tmp_path)).model_dump()
    before = compute_cache_key(cab, None, values)
    logs = Path(values["output_log_directory"])
    logs.mkdir()
    (logs / "run.log").write_text("own output")
    invalidate_path_hashes()
    assert compute_cache_key(cab, None, values) == before


def test_dynamic_readonly_pattern_changes_execution_identity():
    from shinobi.loaders.yaml_cab import loads

    cab = loads('tool: {command: tool, input_patterns: [{segments: [{regex: "[A-Z]+"}, {attrs: {path: {dtype: Directory}}}]}]}')["tool"]
    changed = loads('tool: {command: tool, input_patterns: [{segments: [{regex: "[A-Z]+"}, {attrs: {path: {dtype: Directory, writable: false}}}]}]}')["tool"]
    assert build_argv(cab, {"G.path": "inputs"}) == build_argv(changed, {"G.path": "inputs"})
    assert cab_semantic_identity(cab) != cab_semantic_identity(changed)
    assert cab_semantic_identity(ScopeSpec.capture(changed).restore()) == cab_semantic_identity(changed)


def test_quartical_readonly_dispatch_captures_gains_and_caches(tmp_path, monkeypatch):
    import sys
    from shinobi.steps.dispatch import _dispatch

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "obs.ms"
    root.mkdir()
    (root / "table.dat").write_text("raw")
    _install_dataset(monkeypatch, tmp_path, root)
    script = tmp_path / "gain_writer.py"
    script.write_text(
        "from pathlib import Path\nimport sys\nv=dict(arg.split('=',1) for arg in sys.argv[1:])\ng=Path(v['output.gain_directory']);g.mkdir(parents=True,exist_ok=True);(g/'gain.bin').write_bytes(b'gain')\nl=Path(v['output.log_directory']);l.mkdir(parents=True,exist_ok=True);(l/'run.log').write_text('logged')\n"
    )
    cab = get("quartical-readonly").model_copy(update={"command": f"{sys.executable} {script}", "backend": "native", "cache": True, "cache_dir": str(tmp_path / "cache")})
    first = _dispatch(cab, None, **arguments("quartical-readonly", tmp_path))
    assert first.outputs.gain_directory.select().path.is_dir()
    assert _dispatch(cab, None, **arguments("quartical-readonly", tmp_path)).cached
    assert (root / "table.dat").read_text() == "raw"


def test_quartical_load_from_is_readonly_and_cannot_alias_gain_output(tmp_path):
    cab = get("quartical-readonly")
    source = tmp_path / "source-gains"
    source.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    assert cab.match_pattern("G.load_from").writable is False
    values = cab.inputs_model(**{**arguments("quartical-readonly", tmp_path), "G.load_from": source}).model_dump()
    validate_declared_writes(cab, values, tmp_path)
    values["output_gain_directory"] = str(alias)
    with pytest.raises(ParameterError, match="overlap|write"):
        validate_declared_writes(cab, values, tmp_path)
