from pathlib import Path

import pytest

import shinobi.dataset_closure as closure_module
from shinobi.dataset_closure import ClosureStatus, resolve_dataset_closure
from shinobi.datasets import DatasetDescriptor, DatasetStatus, MSV2_STRUCTURAL_V1


class FakeTable:
    def __init__(self, path, *, keywords=(), references=None, managers=("StandardStMan",), parts=None):
        self.path = Path(path)
        self.keywords = keywords
        self.references = references or {}
        self.managers = managers
        self.parts = parts
        self.closed = False

    def keywordnames(self):
        return self.keywords

    def getkeyword(self, name):
        return self.references[name]

    def getdminfo(self):
        return {str(index): {"TYPE": name} for index, name in enumerate(self.managers)}

    def name(self):
        return str(self.path)

    def partnames(self):
        return [str(self.path)] if self.parts is None else self.parts

    def close(self):
        self.closed = True


def install_tables(monkeypatch, root, references, *, managers=None, parts=None, inspector=None):
    opened = []

    def factory(name, **kwargs):
        path = Path(name).resolve()
        table = FakeTable(
            path,
            keywords=tuple(references) if path == root.resolve() else (),
            references=references if path == root.resolve() else {},
            managers=(managers or {}).get(path, ("StandardStMan",)),
            parts=(parts or {}).get(path),
        )
        opened.append(table)
        return table

    monkeypatch.setattr(closure_module, "_load_table_factory", lambda: factory)
    monkeypatch.setattr(
        closure_module,
        "inspect_measurement_set_v2",
        inspector or (lambda path: DatasetDescriptor(path=path, expected=MSV2_STRUCTURAL_V1, status=DatasetStatus.VALID, message="valid")),
    )
    return opened


def test_contained_alias_and_deterministic_serialization(tmp_path, monkeypatch):
    root = tmp_path / "data" / "target.ms"
    antenna = root / "ANTENNA"
    antenna.mkdir(parents=True)
    alias = root / "ANTENNA_ALIAS"
    alias.symlink_to(antenna, target_is_directory=True)
    (root / "table.dat").write_bytes(b"metadata")
    opened = install_tables(monkeypatch, root, {"ANTENNA": "Table: ANTENNA", "SOURCE": "Table: ANTENNA_ALIAS"})

    result = resolve_dataset_closure(root, storage_namespace=tmp_path / "data")

    assert result.status is ClosureStatus.VALID
    assert [resource.namespace_path.as_posix() for resource in result.resources] == ["target.ms", "target.ms/ANTENNA"]
    assert result.resources[1].members == ("ANTENNA", "SOURCE")
    assert result.model_dump_json() == resolve_dataset_closure(root, storage_namespace=tmp_path / "data").model_dump_json()
    assert all(table.closed for table in opened)


def test_external_shared_resource_is_allowed(tmp_path, monkeypatch):
    namespace = tmp_path / "store"
    root = namespace / "one.ms"
    shared = namespace / "shared" / "SOURCE"
    root.mkdir(parents=True)
    shared.mkdir(parents=True)
    install_tables(monkeypatch, root, {"SOURCE": shared})

    result = resolve_dataset_closure(root, storage_namespace=namespace)

    assert result.valid
    source = next(resource for resource in result.resources if resource.members == ("SOURCE",))
    assert source.external_to_root


@pytest.mark.parametrize("kind,status", [("missing", ClosureStatus.DANGLING_REFERENCE), ("escape", ClosureStatus.ESCAPED_NAMESPACE), ("cycle", ClosureStatus.CYCLIC_REFERENCE)])
def test_reference_failures_have_distinct_statuses(tmp_path, monkeypatch, kind, status):
    namespace = tmp_path / "store"
    root = namespace / "target.ms"
    root.mkdir(parents=True)
    if kind == "missing":
        target = root / "absent"
    elif kind == "escape":
        target = tmp_path / "outside"
        target.mkdir()
    else:
        target = root
    install_tables(monkeypatch, root, {"ANTENNA": target})
    assert resolve_dataset_closure(root, storage_namespace=namespace).status is status


def test_broken_actual_reference_precedes_structural_preflight(tmp_path, monkeypatch):
    root = tmp_path / "target.ms"
    root.mkdir()

    def should_not_inspect(path):
        pytest.fail(f"structural inspection blurred broken reference {path}")

    install_tables(monkeypatch, root, {"ANTENNA": root / "missing"}, inspector=should_not_inspect)
    assert resolve_dataset_closure(root, storage_namespace=tmp_path).status is ClosureStatus.DANGLING_REFERENCE


def test_genuine_schema_failure_is_invalid_root(tmp_path, monkeypatch):
    root = tmp_path / "target.ms"
    root.mkdir()

    def invalid(path):
        return DatasetDescriptor(path=path, expected=MSV2_STRUCTURAL_V1, status=DatasetStatus.INCOMPLETE_MEASUREMENT_SET_V2, message="missing columns")

    install_tables(monkeypatch, root, {}, inspector=invalid)
    assert resolve_dataset_closure(root, storage_namespace=tmp_path).status is ClosureStatus.INVALID_ROOT


def test_unsupported_storage_manager_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "target.ms"
    root.mkdir()
    opened = install_tables(monkeypatch, root, {}, managers={root.resolve(): ("OpaqueExternalStMan",)})
    result = resolve_dataset_closure(root, storage_namespace=tmp_path)
    assert result.status is ClosureStatus.UNSUPPORTED
    assert result.unsupported_features
    assert all(table.closed for table in opened)


def test_reference_parts_are_authoritative(tmp_path, monkeypatch):
    root = tmp_path / "target.ms"
    source = tmp_path / "source.ms"
    root.mkdir()
    source.mkdir()
    install_tables(monkeypatch, root, {}, parts={root.resolve(): [str(source)]})
    assert resolve_dataset_closure(root, storage_namespace=tmp_path).status is ClosureStatus.UNSUPPORTED


def test_backing_symlink_escape_dangling_and_cycle_are_distinct(tmp_path, monkeypatch):
    root = tmp_path / "store" / "target.ms"
    outside = tmp_path / "outside"
    root.mkdir(parents=True)
    outside.write_text("backing")
    install_tables(monkeypatch, root, {})
    link = root / "table.f0"
    link.symlink_to(outside)
    assert resolve_dataset_closure(root, storage_namespace=root.parent).status is ClosureStatus.ESCAPED_NAMESPACE
    link.unlink()
    link.symlink_to(root / "missing")
    assert resolve_dataset_closure(root, storage_namespace=root.parent).status is ClosureStatus.DANGLING_REFERENCE
    link.unlink()
    link.symlink_to(link)
    assert resolve_dataset_closure(root, storage_namespace=root.parent).status is ClosureStatus.CYCLIC_REFERENCE


def test_alias_deduplication_precedes_resource_limit(tmp_path, monkeypatch):
    root = tmp_path / "target.ms"
    shared = root / "shared"
    shared.mkdir(parents=True)
    alias = root / "alias"
    alias.symlink_to(shared, target_is_directory=True)
    install_tables(monkeypatch, root, {"ANTENNA": shared, "SOURCE": alias})
    result = resolve_dataset_closure(root, storage_namespace=tmp_path, max_resources=2)
    assert result.valid
    assert len(result.resources) == 2


def test_two_roots_have_actual_shared_resource_intersection(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    closures = []
    for name in ("one.ms", "two.ms"):
        root = tmp_path / name
        root.mkdir()
        install_tables(monkeypatch, root, {"SOURCE": shared})
        closures.append(resolve_dataset_closure(root, storage_namespace=tmp_path))
    left = {resource.path for resource in closures[0].resources}
    right = {resource.path for resource in closures[1].resources}
    assert left & right == {shared.resolve()}


def test_limits_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        resolve_dataset_closure(tmp_path, storage_namespace=tmp_path, max_resources=0)


def test_late_root_change_is_detected(tmp_path, monkeypatch):
    root = tmp_path / "target.ms"
    subtable = root / "ANTENNA"
    subtable.mkdir(parents=True)

    def mutating_inspector(path):
        (root / "late-change").write_text("changed")
        return DatasetDescriptor(path=path, expected=MSV2_STRUCTURAL_V1, status=DatasetStatus.VALID, message="valid")

    install_tables(monkeypatch, root, {"ANTENNA": subtable}, inspector=mutating_inspector)
    assert resolve_dataset_closure(root, storage_namespace=tmp_path).status is ClosureStatus.CHANGED_DURING_RESOLUTION


def _real_valid_ms(tables, path):
    ms = tables.default_ms(str(path))
    ms.addcols(tables.maketabdesc([tables.makearrcoldesc("DATA", 0j, ndim=2)]))
    return ms


def test_real_casacore_plain_reference_and_readme(tmp_path):
    tables = pytest.importorskip("casacore.tables")
    ms_path = tmp_path / "plain.ms"
    ms = _real_valid_ms(tables, ms_path)
    ms.putinfo({"type": "Measurement Set", "subType": "", "readme": "reference concat words are arbitrary prose"})
    ms.close()

    plain = resolve_dataset_closure(ms_path, storage_namespace=tmp_path)
    assert plain.valid
    assert plain.capabilities is not None

    source = tables.table(str(ms_path), ack=False)
    reference_path = tmp_path / "reference.ms"
    reference = source.query(query="True", name=str(reference_path))
    reference.close()
    source.close()
    refused = resolve_dataset_closure(reference_path, storage_namespace=tmp_path)
    assert refused.status is ClosureStatus.UNSUPPORTED
    assert "reference or virtual-concatenation" in refused.message


def test_real_casacore_virtual_concatenation_parts_are_refused(tmp_path):
    tables = pytest.importorskip("casacore.tables")
    left_path = tmp_path / "left.ms"
    right_path = tmp_path / "right.ms"
    left = _real_valid_ms(tables, left_path)
    right = _real_valid_ms(tables, right_path)
    left.close()
    right.close()
    concatenated = tables.table([str(left_path), str(right_path)], ack=False)
    try:
        with pytest.raises(ValueError, match="virtual-concatenation"):
            closure_module._plain_table_metadata(concatenated)
    finally:
        concatenated.close()


def test_real_casacore_escaped_backing_file_is_refused(tmp_path):
    tables = pytest.importorskip("casacore.tables")
    namespace = tmp_path / "store"
    namespace.mkdir()
    ms_path = namespace / "plain.ms"
    ms = _real_valid_ms(tables, ms_path)
    ms.close()
    backing = ms_path / "table.f0"
    external = tmp_path / "external-table.f0"
    backing.rename(external)
    backing.symlink_to(external)

    result = resolve_dataset_closure(ms_path, storage_namespace=namespace)
    assert result.status is ClosureStatus.ESCAPED_NAMESPACE
