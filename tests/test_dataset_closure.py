from pathlib import Path

import pytest

import shinobi.dataset_closure as closure_module
from shinobi.dataset_closure import ClosureStatus, resolve_dataset_closure
from shinobi.datasets import DatasetDescriptor, DatasetStatus, MSV2_STRUCTURAL_V1


class FakeTable:
    def __init__(self, path, *, keywords=(), references=None, managers=("StandardStMan",), info=None):
        self.path = Path(path)
        self.keywords = keywords
        self.references = references or {}
        self.managers = managers
        self._info = info or {}
        self.closed = False

    def keywordnames(self):
        return self.keywords

    def getkeyword(self, name):
        return self.references[name]

    def getdminfo(self):
        return {str(index): {"TYPE": name} for index, name in enumerate(self.managers)}

    def info(self):
        return self._info

    def close(self):
        self.closed = True


def install_tables(monkeypatch, root, references, *, managers=None, info=None):
    opened = []

    def factory(name, **kwargs):
        path = Path(name).resolve()
        table = FakeTable(
            path,
            keywords=tuple(references) if path == root.resolve() else (),
            references=references if path == root.resolve() else {},
            managers=(managers or {}).get(path, ("StandardStMan",)),
            info=(info or {}).get(path),
        )
        opened.append(table)
        return table

    monkeypatch.setattr(closure_module, "_load_table_factory", lambda: factory)
    monkeypatch.setattr(
        closure_module,
        "inspect_measurement_set_v2",
        lambda path: DatasetDescriptor(path=path, expected=MSV2_STRUCTURAL_V1, status=DatasetStatus.VALID, message="valid"),
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


@pytest.mark.parametrize(
    "managers,info", [(("OpaqueExternalStMan",), None), (("StandardStMan",), {"type": "Reference Table"}), (("StandardStMan",), {"subType": "virtual concatenation"})]
)
def test_unsupported_table_features_are_refused(tmp_path, monkeypatch, managers, info):
    root = tmp_path / "target.ms"
    root.mkdir()
    install_tables(monkeypatch, root, {}, managers={root.resolve(): managers}, info={root.resolve(): info} if info else {})
    result = resolve_dataset_closure(root, storage_namespace=tmp_path)
    assert result.status is ClosureStatus.UNSUPPORTED
    assert result.unsupported_features


def test_changed_during_resolution_is_detected(tmp_path, monkeypatch):
    root = tmp_path / "target.ms"
    subtable = root / "ANTENNA"
    subtable.mkdir(parents=True)
    install_tables(monkeypatch, root, {"ANTENNA": subtable})
    real_signature = closure_module._signature
    calls = 0

    def changing_signature(path, **kwargs):
        nonlocal calls
        calls += 1
        value = real_signature(path, **kwargs)
        if calls == 3:
            (subtable / "changed").write_text("changed")
        return value

    monkeypatch.setattr(closure_module, "_signature", changing_signature)
    assert resolve_dataset_closure(root, storage_namespace=tmp_path).status is ClosureStatus.CHANGED_DURING_RESOLUTION
