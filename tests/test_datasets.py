from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import BaseModel, TypeAdapter

import shinobi.datasets as datasets
from shinobi import Cab, CasaTable, MeasurementSetV2
from shinobi.datasets import (
    CASA_TABLE_PROFILE,
    MSV2_STRUCTURAL_PROFILE,
    DatasetKind,
    DatasetStatus,
    DatasetType,
    InspectionLimits,
    dataset_declarations,
    dataset_fields,
    inspect_casa_table,
    inspect_dataset,
    inspect_measurement_set_v2,
)
from shinobi.exceptions import DatasetLifecycleUnavailableError
from shinobi.graph import RecipeNotOffloadableError, check_offloadable
from shinobi.loaders._modelgen import dtype_to_type
from shinobi.steps import InputRef, OutputRef, Recipe


class DatasetInputs(BaseModel):
    ms: MeasurementSetV2
    table: CasaTable | None = None


class DatasetOutputs(BaseModel):
    ms: MeasurementSetV2


class EmptyOutputs(BaseModel):
    pass


class NestedDataset(BaseModel):
    ms: MeasurementSetV2


class RecursiveDatasets(BaseModel):
    tables: Mapping[str, MeasurementSetV2]
    nested: list[NestedDataset]
    nested_by_name: dict[str, NestedDataset | None]
    child: RecursiveDatasets | None = None


class _FakeTable:
    def __init__(self, *, columns=(), keywords=(), nrows=0, version=2.0, metadata_error: Exception | None = None):
        self._columns = columns
        self._keywords = keywords
        self._nrows = nrows
        self._version = version
        self._metadata_error = metadata_error
        self.closed = False

    def colnames(self):
        if self._metadata_error:
            raise self._metadata_error
        return self._columns

    def keywordnames(self):
        return self._keywords

    def nrows(self):
        return self._nrows

    def getkeyword(self, name):
        assert name == "MS_VERSION"
        return self._version

    def close(self):
        self.closed = True


def _patch_table(monkeypatch, table):
    monkeypatch.setattr(datasets, "_load_table_factory", lambda: lambda *args, **kwargs: table)


def _valid_subtables(**overrides):
    subtables = {name: _FakeTable(columns=tuple(columns)) for name, columns in datasets._MSV2_REQUIRED_SUBTABLE_COLUMNS.items()}
    subtables.update(overrides)
    return subtables


def _patch_msv2_tables(monkeypatch, main, *, subtables=None):
    subtables = _valid_subtables() if subtables is None else subtables

    def open_table(name, **kwargs):
        table_name = str(name)
        if "::" in table_name:
            return subtables[table_name.rsplit("::", 1)[1]]
        return main

    monkeypatch.setattr(datasets, "_load_table_factory", lambda: open_table)


def test_dataset_annotations_are_paths_and_serialize_without_inspection(monkeypatch):
    monkeypatch.setattr(datasets, "_load_table_factory", lambda: pytest.fail("model construction imported casacore"))

    model = DatasetInputs(ms="future.ms", table="calibration.tbl")

    assert model.ms == Path("future.ms")
    assert model.table == Path("calibration.tbl")
    assert model.model_dump(mode="json") == {"ms": "future.ms", "table": "calibration.tbl"}
    ms_schema = DatasetInputs.model_json_schema()["properties"]["ms"]
    assert ms_schema["format"] == "path"
    assert ms_schema["x-shinobi-dataset"] == {"kind": "measurement-set-v2", "profile": "msv2-structural/v1"}
    assert TypeAdapter(DatasetType).dump_python(DatasetType(kind=DatasetKind.MEASUREMENT_SET_V2, profile=MSV2_STRUCTURAL_PROFILE), mode="json") == {
        "kind": "measurement-set-v2",
        "profile": "msv2-structural/v1",
    }


def test_dataset_fields_finds_direct_and_optional_annotations():
    found = dataset_fields(DatasetInputs)
    assert found["ms"].profile == MSV2_STRUCTURAL_PROFILE
    assert found["table"].profile == CASA_TABLE_PROFILE


def test_dataset_declarations_walks_mappings_nested_models_and_cycles():
    found = dataset_declarations(RecursiveDatasets)

    assert set(found) == {"tables.*", "nested[].ms", "nested_by_name.*.ms"}
    assert {declaration.profile for declaration in found.values()} == {MSV2_STRUCTURAL_PROFILE}
    assert dataset_fields(RecursiveDatasets) == found


def test_legacy_ms_loader_dtype_remains_a_plain_path():
    assert dtype_to_type("MS") is Path


def test_missing_and_non_directory_paths_do_not_import_casacore(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "_load_table_factory", lambda: pytest.fail("casacore import should be deferred"))
    assert inspect_measurement_set_v2(tmp_path / "absent.ms").status is DatasetStatus.MISSING_PATH

    regular_file = tmp_path / "not-a-table"
    regular_file.write_text("ordinary file")
    assert inspect_casa_table(regular_file).status is DatasetStatus.NOT_A_DIRECTORY


def test_arbitrary_directory_is_distinguished_from_missing_path(tmp_path, monkeypatch):
    arbitrary = tmp_path / "directory"
    arbitrary.mkdir()

    def cannot_open(*args, **kwargs):
        raise RuntimeError("no table.dat")

    monkeypatch.setattr(datasets, "_load_table_factory", lambda: cannot_open)
    descriptor = inspect_casa_table(arbitrary)
    assert descriptor.status is DatasetStatus.NOT_A_CASA_TABLE
    assert "not a readable CASA table" in descriptor.message


@pytest.mark.parametrize("error", [ModuleNotFoundError("No module named 'casacore'"), OSError("shared library cannot be loaded")])
def test_inspector_unavailable_is_actionable(tmp_path, monkeypatch, error):
    candidate = tmp_path / "candidate.ms"
    candidate.mkdir()

    def unavailable():
        raise error

    monkeypatch.setattr(datasets, "_load_table_factory", unavailable)
    descriptor = inspect_measurement_set_v2(candidate)
    assert descriptor.status is DatasetStatus.INSPECTOR_UNAVAILABLE
    assert descriptor.unsupported_features == ("python-casacore metadata inspection",)


def test_unexpected_inspector_loader_errors_surface(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate.ms"
    candidate.mkdir()

    def broken_loader():
        raise RuntimeError("bug in inspector loader")

    monkeypatch.setattr(datasets, "_load_table_factory", broken_loader)
    with pytest.raises(RuntimeError, match="bug in inspector loader"):
        inspect_measurement_set_v2(candidate)


def test_non_ms_casa_table_preserves_observed_metadata(tmp_path, monkeypatch):
    candidate = tmp_path / "ordinary.table"
    candidate.mkdir()
    table = _FakeTable(columns=("CUSTOM", "VALUE"), keywords=("CUSTOM_KEYWORD",), nrows=3)
    _patch_table(monkeypatch, table)

    descriptor = inspect_measurement_set_v2(candidate)

    assert descriptor.status is DatasetStatus.NOT_A_MEASUREMENT_SET
    assert descriptor.columns == ("CUSTOM", "VALUE")
    assert descriptor.keywords == ("CUSTOM_KEYWORD",)
    assert descriptor.nrows == 3
    assert table.closed


def test_incomplete_msv2_reports_missing_structure(tmp_path, monkeypatch):
    candidate = tmp_path / "incomplete.ms"
    candidate.mkdir()
    table = _FakeTable(columns=("TIME",), keywords=("MS_VERSION",), nrows=1)
    _patch_table(monkeypatch, table)

    descriptor = inspect_measurement_set_v2(candidate)

    assert descriptor.status is DatasetStatus.INCOMPLETE_MEASUREMENT_SET_V2
    assert "ANTENNA1" in descriptor.missing_columns
    assert "ANTENNA" in descriptor.missing_subtables


def test_incomplete_msv2_reports_unreadable_required_subtable(tmp_path, monkeypatch):
    candidate = tmp_path / "broken.ms"
    candidate.mkdir()
    main = _FakeTable(
        columns=tuple(datasets._MSV2_REQUIRED_COLUMNS),
        keywords=tuple(datasets._MSV2_REQUIRED_SUBTABLES) + ("MS_VERSION",),
    )

    subtables = _valid_subtables()

    def open_table(name, **kwargs):
        if str(name).endswith("::ANTENNA"):
            raise RuntimeError("broken reference")
        if "::" in str(name):
            return subtables[str(name).rsplit("::", 1)[1]]
        return main

    monkeypatch.setattr(datasets, "_load_table_factory", lambda: open_table)
    descriptor = inspect_measurement_set_v2(candidate)

    assert descriptor.status is DatasetStatus.INCOMPLETE_MEASUREMENT_SET_V2
    assert descriptor.unreadable_subtables == ("ANTENNA",)
    assert "unreadable required subtables" in descriptor.message


def test_incomplete_msv2_reports_required_subtable_columns(tmp_path, monkeypatch):
    candidate = tmp_path / "partial.ms"
    candidate.mkdir()
    main = _FakeTable(
        columns=tuple(datasets._MSV2_REQUIRED_COLUMNS),
        keywords=tuple(datasets._MSV2_REQUIRED_SUBTABLES) + ("MS_VERSION", "SOURCE", "CUSTOM_SUBTABLE"),
    )
    subtables = _valid_subtables(ANTENNA=_FakeTable(columns=("NAME",)))
    _patch_msv2_tables(monkeypatch, main, subtables=subtables)

    descriptor = inspect_measurement_set_v2(candidate)

    assert descriptor.status is DatasetStatus.INCOMPLETE_MEASUREMENT_SET_V2
    assert "POSITION" in descriptor.missing_subtable_columns["ANTENNA"]
    assert "SOURCE" not in descriptor.missing_subtable_columns
    assert "required subtables missing columns" in descriptor.message


def test_valid_msv2_retains_custom_columns_and_serializes(tmp_path, monkeypatch):
    candidate = tmp_path / "valid.ms"
    candidate.mkdir()
    columns = tuple(datasets._MSV2_REQUIRED_COLUMNS) + (
        "CORRECTED_DATA",
        "PIPELINE_CUSTOM",
    )
    keywords = tuple(datasets._MSV2_REQUIRED_SUBTABLES) + ("MS_VERSION", "CUSTOM_KEYWORD")
    table = _FakeTable(columns=columns, keywords=keywords, nrows=7)
    _patch_msv2_tables(monkeypatch, table)

    descriptor = inspect_measurement_set_v2(candidate)

    assert descriptor.status is DatasetStatus.VALID
    assert descriptor.valid
    assert "PIPELINE_CUSTOM" in descriptor.columns
    assert '"status":"valid"' in descriptor.model_dump_json()


def test_wrong_ms_version_and_metadata_bounds_are_unsupported(tmp_path, monkeypatch):
    candidate = tmp_path / "unsupported.ms"
    candidate.mkdir()
    table = _FakeTable(columns=("A", "B"), keywords=("MS_VERSION",), version=3.0)
    _patch_table(monkeypatch, table)
    version = inspect_measurement_set_v2(candidate)
    assert version.status is DatasetStatus.UNSUPPORTED
    assert version.unsupported_features == ("MeasurementSet version 3.0",)

    table = _FakeTable(columns=("A", "B"), keywords=())
    _patch_table(monkeypatch, table)
    bounded = inspect_dataset(candidate, limits=InspectionLimits(max_columns=1))
    assert bounded.status is DatasetStatus.UNSUPPORTED
    assert bounded.columns == ()  # refused, never silently truncated


def test_subtable_metadata_bounds_are_enforced(tmp_path, monkeypatch):
    candidate = tmp_path / "too-wide.ms"
    candidate.mkdir()
    main = _FakeTable(
        columns=tuple(datasets._MSV2_REQUIRED_COLUMNS),
        keywords=tuple(datasets._MSV2_REQUIRED_SUBTABLES) + ("MS_VERSION",),
    )
    limit = len(datasets._MSV2_REQUIRED_COLUMNS)
    too_many_columns = tuple(datasets._MSV2_REQUIRED_SUBTABLE_COLUMNS["ANTENNA"]) + tuple(f"CUSTOM_{index}" for index in range(limit))
    subtables = _valid_subtables(ANTENNA=_FakeTable(columns=too_many_columns))
    _patch_msv2_tables(monkeypatch, main, subtables=subtables)

    descriptor = inspect_measurement_set_v2(candidate, limits=InspectionLimits(max_columns=limit))

    assert descriptor.status is DatasetStatus.UNSUPPORTED
    assert any(feature.startswith("ANTENNA column count") for feature in descriptor.unsupported_features)

    too_many_keywords = tuple(f"KEYWORD_{index}" for index in range(limit + 1))
    subtables = _valid_subtables(ANTENNA=_FakeTable(columns=tuple(datasets._MSV2_REQUIRED_SUBTABLE_COLUMNS["ANTENNA"]), keywords=too_many_keywords))
    _patch_msv2_tables(monkeypatch, main, subtables=subtables)
    descriptor = inspect_measurement_set_v2(candidate, limits=InspectionLimits(max_keywords=limit))
    assert descriptor.status is DatasetStatus.UNSUPPORTED
    assert any(feature.startswith("ANTENNA keyword count") for feature in descriptor.unsupported_features)


def test_unknown_profile_is_reported_without_touching_the_path(monkeypatch):
    monkeypatch.setattr(datasets, "_load_table_factory", lambda: pytest.fail("unsupported profile must not inspect"))
    declaration = DatasetType(kind=DatasetKind.CASA_TABLE, profile="casa-table/v99")
    descriptor = inspect_dataset("does-not-matter", declaration)
    assert descriptor.status is DatasetStatus.UNSUPPORTED


def test_strict_annotation_refuses_local_execution_before_backend_runs():
    cab = Cab(name="strict", command="true", inputs_model=DatasetInputs, outputs_model=DatasetOutputs)

    with pytest.raises(DatasetLifecycleUnavailableError, match="cannot execute until validation, staging and recovery"):
        cab(ms="future.ms")


def test_nested_strict_annotation_refuses_local_execution():
    class Inputs(BaseModel):
        payload: dict[str, list[NestedDataset]]

    cab = Cab(name="strict-nested", command="true", inputs_model=Inputs, outputs_model=EmptyOutputs)

    with pytest.raises(DatasetLifecycleUnavailableError, match=r"payload\.\*\[\]\.ms"):
        cab(payload={"target": [{"ms": "future.ms"}]})


def test_strict_annotation_is_not_offloadable():
    cab = Cab(name="strict", command="true", inputs_model=DatasetInputs, outputs_model=DatasetOutputs)
    recipe = Recipe(name="r", inputs_model=DatasetInputs, outputs_model=DatasetOutputs)
    recipe.add_step("strict", cab, ms=InputRef(field="ms"))
    recipe.output_wiring["ms"] = OutputRef(step="strict", field="ms")

    with pytest.raises(RecipeNotOffloadableError, match="dataset lifecycle enforcement"):
        check_offloadable(recipe)


def test_nested_strict_annotation_is_not_offloadable():
    class Inputs(BaseModel):
        payload: dict[str, NestedDataset]

    cab = Cab(name="strict-nested", command="true", inputs_model=Inputs, outputs_model=EmptyOutputs)
    recipe = Recipe(name="r", inputs_model=Inputs, outputs_model=EmptyOutputs)
    recipe.add_step("strict", cab, payload=InputRef(field="payload"))

    with pytest.raises(RecipeNotOffloadableError, match=r"payload\.\*\.ms"):
        check_offloadable(recipe)


def test_generated_tiny_casacore_tables_when_available(tmp_path):
    tables = pytest.importorskip("casacore.tables")
    if not hasattr(tables, "default_ms"):
        pytest.fail("installed python-casacore has no default_ms fixture builder")

    ordinary_path = tmp_path / "ordinary.table"
    description = tables.maketabdesc([tables.makescacoldesc("VALUE", 0)])
    ordinary = tables.table(str(ordinary_path), tabledesc=description, nrow=1, readonly=False, ack=False)
    ordinary.close()
    assert inspect_casa_table(ordinary_path).status is DatasetStatus.VALID
    assert inspect_measurement_set_v2(ordinary_path).status is DatasetStatus.NOT_A_MEASUREMENT_SET

    ms_path = tmp_path / "tiny.ms"
    ms = tables.default_ms(str(ms_path))
    ms.close()
    descriptor = inspect_measurement_set_v2(ms_path)
    assert descriptor.status is DatasetStatus.VALID
