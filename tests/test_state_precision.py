"""Explicit approximate artifacts and lossless MSv4 layout contracts."""

from __future__ import annotations


import pytest

from shinobi.dataset_state import DatasetStateStore, StateError, StateRepresentationV2
from shinobi.state_codecs import LosslessProfile, bounded_chunks, tile_slices, rewrite_lossless
from shinobi.state_precision import DerivedLogical, PrecisionPolicy, _compare, _round

np = pytest.importorskip("numpy", reason="precision operations need the measurement-set group")


def test_bounded_tiles_cover_large_trailing_axes():
    chunks = bounded_chunks((2, 2048, 2048), (2, 2048, 2048), 8)
    assert np.prod(chunks) * 8 <= 16 * 1024 * 1024
    assert sum(np.prod([axis.stop - axis.start for axis in region]) for region in tile_slices((2, 2048, 2048), chunks)) == 2 * 2048 * 2048


def test_rounding_ties_nonfinite_payload_signed_zero_and_complex_bound():
    # At two mantissa bits the step is 1/4; these ties round to even neighbors.
    source = np.array([1.125, 1.375, -1.125, -0.0, np.inf], dtype=np.float32)
    source[4:5].view(np.uint32)[0] = np.uint32(0x7FC12345)
    transformed = _round(source, 2)
    assert list(transformed[:3]) == [1.0, 1.5, -1.0]
    assert transformed[3:4].view(np.uint32)[0] == source[3:4].view(np.uint32)[0]
    assert transformed[4:5].view(np.uint32)[0] == source[4:5].view(np.uint32)[0]
    complex_source = np.array([1.2345 - 2.3456j], dtype=np.complex64)
    complex_result = _round(complex_source, 10)
    assert _compare(complex_source, complex_result, PrecisionPolicy(arrays=("VISIBILITY",), keep_bits=10, atol=0.01, rtol=0))[0] == 1
    with pytest.raises(StateError, match="precision-bound"):
        _compare(complex_source, complex_result, PrecisionPolicy(arrays=("VISIBILITY",), keep_bits=10, atol=0, rtol=0))


def test_derived_identity_ignores_physical_source_representation():
    state = "msutils-logical-hash/v1:" + "1" * 64
    msv4 = "msutils-logical-hash/v1:" + "2" * 64
    policy = PrecisionPolicy(arrays=("partition/VISIBILITY",), keep_bits=10, atol=1, rtol=0)
    logical = DerivedLogical(derived_id="shinobi-derived/v1:" + "0" * 64, parent_state_id=state, parent_msv4_id=msv4, policy=policy, msv4_id=msv4)
    assert logical.expected_id() != logical.model_copy(update={"policy": policy.model_copy(update={"keep_bits": 9})}).expected_id()
    assert "source_representation_id" not in DerivedLogical.model_fields


@pytest.mark.parametrize("codec", ["packbits-zstd", "delta-zstd", "packbits-delta-zstd"])
def test_semantic_rewrite_filters_are_dtype_specific(tmp_path, codec):
    import json

    msutils = pytest.importorskip("msutils")
    zarr = pytest.importorskip("zarr")

    source = tmp_path / "source.zarr"
    group = zarr.open_group(source, mode="w", zarr_format=3)
    values = {
        "flags": np.arange(33).reshape(3, 11) % 3 == 0,
        "indices": np.array([[2**31 - 1, -(2**31), 0], [-(2**31), 2**31 - 1, 1]], dtype="int32"),
        "weights": np.array([[1.5, -0.0], [2.5, 3.5]], dtype="float32"),
        "fortran_indices": np.asfortranarray(np.arange(15, dtype="uint32").reshape(3, 5)),
        "scalar_flag": np.array(True),
        "empty_indices": np.empty((0, 3), dtype="int16"),
    }
    values["weights"].view("uint32")[1, 1] = 0x7FC12345
    for name, value in values.items():
        group.create_array(name, data=value, chunks=tuple(max(1, min(2, size)) for size in value.shape), config={"write_empty_chunks": True})
    original_id = msutils.logical_id(source)
    target = tmp_path / "rewritten.zarr"
    assert rewrite_lossless(source, target, LosslessProfile(codec=codec, chunk_rows=2)) == original_id
    reopened = zarr.open_group(target, mode="r", use_consolidated=False)
    for name, value in values.items():
        assert np.asarray(reopened[name][...]).tobytes(order="C") == value.tobytes(order="C")
        names = [entry["name"] for entry in json.loads((target / name / "zarr.json").read_text())["codecs"]]
        expected = "numcodecs.packbits" if value.dtype.kind == "b" and "packbits" in codec else "numcodecs.delta" if value.dtype.kind in "iu" and "delta" in codec else None
        assert (expected in names) if expected else not any(item.startswith("numcodecs.") for item in names)
        assert names[-1] == "zstd"


@pytest.mark.parametrize(
    "codec,shuffle", [("gzip", "none"), ("zstd", "none"), ("blosc-zstd", "bit"), ("packbits-zstd", "none"), ("delta-zstd", "none"), ("packbits-delta-zstd", "none")]
)
def test_live_lossless_export_and_derived(tmp_path, monkeypatch, codec, shuffle):
    pytest.importorskip("msutils")
    pytest.importorskip("xarray_ms")
    from casacore.tables import table
    from tests._state_ms import make_state_ms

    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    source = make_state_ms(tmp_path / "source.ms")
    with table(str(source), readonly=False, ack=False) as main:
        values = main.getcol("DATA")
        main.putcol("DATA", values + np.complex64(0.1234 + 0.4567j))
    store = DatasetStateStore(tmp_path / "store", cache_dir=tmp_path / "cache")
    exact = store.export(source, block_rows=2, lossless_profile=LosslessProfile(codec=codec, shuffle=shuffle, chunk_rows=2))
    rep_path, _, representation = store._load(exact.state_id, exact.representation_id)
    assert isinstance(representation, StateRepresentationV2)
    assert store.verify(exact.state_id, exact.representation_id).verified
    import zarr

    arrays = zarr.open_group(rep_path / "msv4.zarr", mode="r").members(max_depth=None)
    if codec in ("packbits-zstd", "delta-zstd", "packbits-delta-zstd"):
        import json

        metadata = [json.loads(path.read_text()) for path in (rep_path / "msv4.zarr").rglob("zarr.json")]
        for item in metadata:
            if item.get("node_type") != "array":
                continue
            dtype = item["data_type"]
            names = [entry["name"] for entry in item["codecs"]]
            expected = (
                "numcodecs.packbits"
                if dtype == "bool" and "packbits" in codec
                else "numcodecs.delta"
                if dtype in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64") and "delta" in codec
                else None
            )
            assert (expected in names) if expected else not any(name.startswith("numcodecs.") for name in names)
            assert names[-1] == "zstd"
    if LosslessProfile(codec=codec).semantic_filters is not None:
        native_flags = zarr.open_group(rep_path / "preservation/native.zarr", mode="r")["MAIN/FLAG"]
        assert native_flags.dtype.kind == "b"
        native_metadata = json.loads((rep_path / "preservation/native.zarr/MAIN/FLAG/zarr.json").read_text())
        assert ("numcodecs.packbits" in [entry["name"] for entry in native_metadata["codecs"]]) == ("packbits" in codec)
        companion_flags = next(array for name, array in arrays if name.endswith("/FLAG"))
        assert companion_flags.dtype == np.dtype("uint8")
        baseline = store.export(source, block_rows=2, lossless_profile=LosslessProfile(codec="zstd", chunk_rows=2))
        _, _, baseline_rep = store._load(baseline.state_id, baseline.representation_id)
        # Independent exports carry different xarray-ms creation timestamps.
        # rewrite_lossless checks MSv4 identity against its own source tree.
        assert (exact.state_id, representation.payload_id) == (baseline.state_id, baseline_rep.payload_id)
        source.rename(tmp_path / "hidden-source.ms")
        destination = tmp_path / "fresh.ms"
        restored = store.materialize(exact.state_id, destination, representation_id=exact.representation_id)
        assert restored.state_id == exact.state_id
    visibility_path = next(name for name, _ in arrays if name.endswith("/VISIBILITY"))
    policy = PrecisionPolicy(arrays=(visibility_path,), keep_bits=10, atol=0.1, rtol=0.1)
    derived = store.derive_precision(exact.state_id, policy, representation_id=exact.representation_id)
    assert store.verify_derived(derived.derived_id, derived.representation_id).verified
    assert store.list_derived()[0].derived_id == derived.derived_id
    with pytest.raises(StateError, match="approximate-refused"):
        store.materialize(derived.derived_id, tmp_path / "must-not-exist.ms")
    assert not (tmp_path / "must-not-exist.ms").exists()
    assert store.recover_derived() == []
    assert store.recover() == []


def test_live_derived_identity_independent_of_lossless_representation(tmp_path, monkeypatch):
    pytest.importorskip("msutils")
    pytest.importorskip("xarray_ms")
    import shutil
    import zarr
    from casacore.tables import table
    from tests._state_ms import make_state_ms
    from shinobi import _state_adapter as adapter

    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    source = make_state_ms(tmp_path / "source.ms")
    with table(str(source), readonly=False, ack=False) as main:
        main.putcol("DATA", main.getcol("DATA") + np.complex64(0.1234 + 0.4567j))
    store = DatasetStateStore(tmp_path / "store", cache_dir=tmp_path / "cache")
    first = store.export(source, lossless_profile=LosslessProfile(codec="gzip", chunk_rows=2))
    first_rep, _, first_physical = store._load(first.state_id, first.representation_id)

    # Freeze the public export output so the second export differs only in its
    # explicit lossless physical layout, not xarray-ms' creation timestamp.
    def same_export(_source, destination):
        shutil.copytree(first_rep / "msv4.zarr", destination)
        return []

    monkeypatch.setattr(adapter, "export", same_export)
    second = store.export(source, lossless_profile=LosslessProfile(codec="blosc-zstd", shuffle="bit", chunk_rows=3))
    second_rep, _, second_physical = store._load(second.state_id, second.representation_id)
    assert first_physical.msv4_id == second_physical.msv4_id
    assert first.representation_id != second.representation_id
    group = zarr.open_group(first_rep / "msv4.zarr", mode="r")
    visibility_path = next(name for name, _ in group.members(max_depth=None) if name.endswith("/VISIBILITY"))
    policy = PrecisionPolicy(arrays=(visibility_path,), keep_bits=10, atol=0.1, rtol=0.1)
    one = store.derive_precision(first.state_id, policy, representation_id=first.representation_id)
    two = store.derive_precision(second.state_id, policy, representation_id=second.representation_id)
    assert one.derived_id == two.derived_id
    assert one.representation_id != two.representation_id
    assert store.verify_derived(one.derived_id, one.representation_id).verified
    assert store.verify_derived(two.derived_id, two.representation_id).verified


def test_derived_publication_recovery_and_unrelated_race(tmp_path, monkeypatch):
    pytest.importorskip("msutils")
    pytest.importorskip("xarray_ms")
    import zarr
    from casacore.tables import table
    from tests._state_ms import make_state_ms
    from shinobi import state_precision
    from shinobi.storage import publish_directory as real_publish

    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    source = make_state_ms(tmp_path / "source.ms")
    with table(str(source), readonly=False, ack=False) as main:
        main.putcol("DATA", main.getcol("DATA") + np.complex64(0.1234 + 0.4567j))
    store = DatasetStateStore(tmp_path / "store", cache_dir=tmp_path / "cache")
    exact = store.export(source)
    rep, _, _ = store._load(exact.state_id, exact.representation_id)
    group = zarr.open_group(rep / "msv4.zarr", mode="r")
    selected = next(name for name, _ in group.members(max_depth=None) if name.endswith("/VISIBILITY"))
    policy = PrecisionPolicy(arrays=(selected,), keep_bits=10, atol=0.1, rtol=0.1)

    def interrupt_after_publish(stage, target):
        real_publish(stage, target)
        raise OSError("simulated interruption after no-replace publication")

    monkeypatch.setattr(state_precision, "publish_directory", interrupt_after_publish)
    with pytest.raises(StateError, match="simulated interruption"):
        store.derive_precision(exact.state_id, policy)
    recovered = store.recover_derived()
    assert len(recovered) == 1 and recovered[0].phase == "committed" and recovered[0].finished
    assert store.verify_derived(recovered[0].derived_id, recovered[0].representation_id).verified
    assert not recovered[0].stage.exists()

    # Reusing the same policy creates another physical representation under
    # the same logical artifact. An unrelated winner at its exact target is
    # preserved while this operation cleans only its own private stage.
    winners = []

    def unrelated_winner(stage, target):
        target.mkdir()
        winners.append(target)
        raise FileExistsError(target)

    monkeypatch.setattr(state_precision, "publish_directory", unrelated_winner)
    with pytest.raises(StateError):
        store.derive_precision(exact.state_id, policy)
    assert winners and winners[0].is_dir()
    assert store.recover_derived() == []


def test_precision_rejects_overflow_and_preserves_partial_nonfinite_complex():
    maximum = np.array([np.finfo(np.float32).max], dtype=np.float32)
    with pytest.raises(StateError, match="precision-overflow"):
        _round(maximum, 0)
    mixed = np.array([complex(float("inf"), 1.2345)], dtype=np.complex64)
    rounded = _round(mixed, 10)
    assert np.array_equal(mixed.view(np.uint8), rounded.view(np.uint8))
    policy = PrecisionPolicy(arrays=("VISIBILITY",), keep_bits=10, atol=1, rtol=0)
    changed = mixed.copy()
    changed.view(np.float32)[1] += 0.001
    with pytest.raises(StateError, match="precision-nonfinite"):
        _compare(mixed, changed, policy)


def test_independent_wrong_policy_and_untouched_array_detection(tmp_path):
    import shutil

    zarr = pytest.importorskip("zarr", reason="whole-tree precision verification needs Zarr")
    pytest.importorskip("msutils", reason="whole-tree precision verification needs msutils")
    from shinobi.state_precision import _transform, _verify_whole_tree

    source = tmp_path / "source.zarr"
    group = zarr.open_group(source, mode="w", zarr_format=3)
    values = np.full((2, 2, 3, 2), np.complex64(1.2345 + 2.3456j))
    roles = ("time", "baseline_id", "frequency", "polarization")
    group.create_array("VISIBILITY", data=values, dimension_names=roles)
    group.create_array("FLAG", data=np.zeros(values.shape, dtype=bool), dimension_names=roles)
    target = tmp_path / "derived.zarr"
    shutil.copytree(source, target)
    policy = PrecisionPolicy(arrays=("VISIBILITY",), keep_bits=10, atol=0.1, rtol=0.1)
    _transform(source, target, policy)
    _transform(source, target, policy, verify=True)
    with pytest.raises(StateError, match="precision-algorithm"):
        _transform(source, target, policy.model_copy(update={"keep_bits": 8}), verify=True)
    derived = zarr.open_group(target, mode="a")
    derived["FLAG"][0, 0, 0, 0] = True
    with pytest.raises(StateError, match="derived-untouched"):
        _verify_whole_tree(source, target, policy.arrays)


def test_derived_claim_refusal_before_stage_is_recorded(tmp_path, monkeypatch):
    pytest.importorskip("msutils")
    pytest.importorskip("xarray_ms")
    import zarr
    from casacore.tables import table
    from tests._state_ms import make_state_ms
    from shinobi import state_precision
    from shinobi.storage import JsonFileStore

    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    source = make_state_ms(tmp_path / "source.ms")
    with table(str(source), readonly=False, ack=False) as main:
        main.putcol("DATA", main.getcol("DATA") + np.complex64(0.1234 + 0.4567j))
    store = DatasetStateStore(tmp_path / "store", cache_dir=tmp_path / "cache")
    exact = store.export(source)
    rep, _, _ = store._load(exact.state_id, exact.representation_id)
    selected = next(name for name, _ in zarr.open_group(rep / "msv4.zarr", mode="r").members(max_depth=None) if name.endswith("/VISIBILITY"))
    policy = PrecisionPolicy(arrays=(selected,), keep_bits=10, atol=0.1, rtol=0.1)

    def refuse(*args, **kwargs):
        raise StateError("claim-denied", "simulated conflict")

    monkeypatch.setattr(state_precision, "acquire_workspace", refuse)
    with pytest.raises(StateError, match="claim-denied"):
        store.derive_precision(exact.state_id, policy)
    records = list((store.root / "derived-attempts").glob("*.json"))
    assert len(records) == 1
    attempt = state_precision.DerivedAttempt.model_validate(JsonFileStore(records[0]).read())
    assert attempt.phase == "refused" and attempt.finished and attempt.stage_identity is None
    assert not attempt.stage.exists()
    assert store.recover_derived() == []


def _precision_store(tmp_path, monkeypatch):
    pytest.importorskip("msutils")
    pytest.importorskip("xarray_ms")
    import zarr
    from casacore.tables import table
    from tests._state_ms import make_state_ms

    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "owners.json"))
    source = make_state_ms(tmp_path / "source.ms")
    with table(str(source), readonly=False, ack=False) as main:
        main.putcol("DATA", main.getcol("DATA") + np.complex64(0.1234 + 0.4567j))
    store = DatasetStateStore(tmp_path / "store", cache_dir=tmp_path / "cache")
    exact = store.export(source)
    rep, _, _ = store._load(exact.state_id, exact.representation_id)
    selected = next(name for name, _ in zarr.open_group(rep / "msv4.zarr", mode="r").members(max_depth=None) if name.endswith("/VISIBILITY"))
    policy = PrecisionPolicy(arrays=(selected,), keep_bits=10, atol=0.1, rtol=0.1)
    return store, exact, policy


def test_derived_error_keeps_attempt_unfinished_until_owner_released(tmp_path, monkeypatch):
    from shinobi import state_precision
    from shinobi.ownership import WorkspaceLease, inspect_ownership, release_workspace
    from shinobi.storage import JsonFileStore

    store, exact, policy = _precision_store(tmp_path, monkeypatch)
    bad_policy = policy.model_copy(update={"arrays": ("missing/VISIBILITY",)})
    original_release = WorkspaceLease.release

    def fail_release(_lease):
        raise OSError("simulated ownership release failure")

    monkeypatch.setattr(WorkspaceLease, "release", fail_release)
    with pytest.raises(StateError, match="simulated ownership release failure"):
        store.derive_precision(exact.state_id, bad_policy)
    path = next((store.root / "derived-attempts").glob("*.json"))
    attempt = state_precision.DerivedAttempt.model_validate(JsonFileStore(path).read())
    assert attempt.phase == "refused" and not attempt.finished
    assert inspect_ownership(store.root, str(attempt.attempt_id)).owner is not None
    monkeypatch.setattr(WorkspaceLease, "release", original_release)
    release_workspace(store.root, str(attempt.attempt_id))
    settled = store.recover_derived()
    assert len(settled) == 1 and settled[0].phase == "interrupted" and settled[0].finished


def test_derived_recovery_keeps_attempt_unfinished_when_owner_release_fails(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from shinobi import state_precision
    from shinobi.storage import JsonFileStore, publish_directory as real_publish

    store, exact, policy = _precision_store(tmp_path, monkeypatch)

    def interrupt(stage, target):
        real_publish(stage, target)
        raise OSError("after publication")

    monkeypatch.setattr(state_precision, "publish_directory", interrupt)
    with pytest.raises(StateError, match="after publication"):
        store.derive_precision(exact.state_id, policy)
    path = next((store.root / "derived-attempts").glob("*.json"))
    attempt = state_precision.DerivedAttempt.model_validate(JsonFileStore(path).read())
    assert not attempt.finished
    fake_owner = SimpleNamespace(registry=attempt.registry)
    monkeypatch.setattr(state_precision, "inspect_ownership", lambda *_: SimpleNamespace(owner=fake_owner, liveness="dead", detail="dead owner"))

    def fail_release(*_):
        raise OSError("simulated recovery release failure")

    original_release = state_precision.release_workspace
    monkeypatch.setattr(state_precision, "release_workspace", fail_release)
    with pytest.raises(StateError, match="simulated recovery release failure"):
        store.recover_derived()
    unfinished = state_precision.DerivedAttempt.model_validate(JsonFileStore(path).read())
    assert unfinished.phase == "committed" and not unfinished.finished
    monkeypatch.setattr(state_precision, "release_workspace", original_release)
    settled = store.recover_derived()
    assert len(settled) == 1 and settled[0].finished and settled[0].phase == "committed"


@pytest.mark.parametrize("part", ["root", "base", "representations", "representation"])
def test_derived_symlink_parents_refuse_reads_publish_and_recovery(tmp_path, monkeypatch, part):
    from shinobi import state_precision
    from shinobi.storage import JsonFileStore, publish_directory as real_publish

    store, exact, policy = _precision_store(tmp_path, monkeypatch)

    def interrupt(stage, target):
        real_publish(stage, target)
        raise OSError("after publication")

    monkeypatch.setattr(state_precision, "publish_directory", interrupt)
    with pytest.raises(StateError, match="after publication"):
        store.derive_precision(exact.state_id, policy)
    monkeypatch.setattr(state_precision, "publish_directory", real_publish)
    path = next((store.root / "derived-attempts").glob("*.json"))
    attempt = state_precision.DerivedAttempt.model_validate(JsonFileStore(path).read())
    base = store.root / "derived" / attempt.derived_id.split(":")[1]
    attacked = {
        "root": store.root / "derived",
        "base": base,
        "representations": base / "representations",
        "representation": base / "representations" / attempt.representation_id.split(":")[1],
    }[part]
    backup = attacked.with_name(attacked.name + ".saved")
    attacked.rename(backup)
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep"
    marker.write_text("untouched")
    attacked.symlink_to(outside, target_is_directory=True)
    for operation in (
        lambda: store.list_derived(),
        lambda: store.verify_derived(attempt.derived_id, attempt.representation_id),
        lambda: store.derive_precision(exact.state_id, policy),
        lambda: store.recover_derived(),
    ):
        with pytest.raises(StateError, match="derived-path-kind"):
            operation()
    assert list(outside.iterdir()) == [marker]
    assert marker.read_text() == "untouched"
    attacked.unlink()
    backup.rename(attacked)
    assert store.verify_derived(attempt.derived_id, attempt.representation_id).verified
    assert store.recover_derived()[0].finished


def test_export_read_chunks_are_bounded_before_backend_reads(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from shinobi import _state_adapter as adapter

    xr = pytest.importorskip("xarray", reason="backend chunk planning needs xarray")

    shape = (10, 2048, 2048)
    encoding = {"chunks": shape, "units": "Jy", "scale_factor": 2}
    variable = SimpleNamespace(dtype=np.dtype("complex64"), shape=shape, dims=("time", "frequency", "polarization"), chunks=None, encoding=encoding)
    node = SimpleNamespace(path="/partition", ds=SimpleNamespace(variables={"VISIBILITY": variable}))
    calls = []

    class Tree:
        subtree = [node]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def to_zarr(self, *args, **kwargs):
            assert np.prod([max(axis) for axis in variable.chunks]) * 8 <= adapter.MAX_TILE_BYTES
            assert variable.encoding == {"units": "Jy", "scale_factor": 2}

    def open_tree(source, **options):
        calls.append(options)
        if "preferred_chunks" in options:
            preferred = options["preferred_chunks"]
            variable.chunks = tuple((preferred[dim],) for dim in variable.dims)
        return Tree()

    monkeypatch.setattr(xr, "open_datatree", open_tree)
    adapter.export(tmp_path / "source.ms", tmp_path / "target.zarr")
    assert len(calls) == 2
    assert "preferred_chunks" not in calls[0]
    assert calls[1]["chunks"] == {}
    assert set(calls[1]["preferred_chunks"]) == set(variable.dims)
    eager = SimpleNamespace(dtype=np.dtype("int64"), shape=(3,), chunks=None, encoding={"chunks": (adapter.MAX_TILE_BYTES,), "units": "s"})
    node.ds.variables = {"index": eager}
    adapter._check_export_chunks(Tree())
    assert eager.encoding == {"units": "s"}
    eager.shape = (adapter.MAX_TILE_BYTES,)
    with pytest.raises(ValueError, match="not chunked"):
        adapter._check_export_chunks(Tree())
