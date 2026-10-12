"""Independent qualification comparison refuses large cells before reading."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from shinobi.state_codecs import MAX_TILE_BYTES

pytest.importorskip("numpy", reason="qualification comparison needs NumPy")
_spec = importlib.util.spec_from_file_location("state_codec_qualification", Path(__file__).resolve().parents[1] / "examples/state_codec_qualification.py")
qualification = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qualification)


class NumericTable:
    def __init__(self, shape, value_type="dcomplex"):
        self.shape = shape
        self.value_type = value_type
        self.reads = []

    def getcoldesc(self, column):
        return {"valueType": self.value_type}

    def isscalarcol(self, column):
        return False

    def getcolshapestring(self, column, start, count):
        return [str(self.shape)] * count

    def getcell(self, *args):
        self.reads.append("getcell")
        raise AssertionError("numeric batch planning must not allocate a cell")

    def getcol(self, *args, **kwargs):
        self.reads.append("getcol")
        raise AssertionError("oversized cells must be refused before reading")


def test_qualification_refuses_oversized_numeric_cell_before_allocation():
    tab = NumericTable([MAX_TILE_BYTES // 16 + 1])
    with pytest.raises(ValueError, match="qualification-only"):
        qualification._comparison_batch(tab, "DATA", 128)
    assert tab.reads == []


@pytest.mark.parametrize("shape,expected_rows", [([8, 8], 128), ([0, 8], 128), ([MAX_TILE_BYTES // 16], 1)])
def test_qualification_numeric_batch_planning_reads_only_metadata(shape, expected_rows):
    tab = NumericTable(shape)
    rows, empty = qualification._comparison_batch(tab, "DATA", 128)
    assert rows == expected_rows
    assert empty == (0 in shape)
    assert tab.reads == []


def test_qualification_independent_native_comparison_still_detects_changes(tmp_path):
    pytest.importorskip("casacore")
    import shutil
    from casacore.tables import table
    from tests._state_ms import make_state_ms

    source = make_state_ms(tmp_path / "source.ms")
    restored = tmp_path / "restored.ms"
    shutil.copytree(source, restored)
    result = qualification._compare_values(source, restored, 2, "msutils-native-model/v1")
    assert result["bounded_blocks"] > 0
    with table(str(restored), readonly=False, ack=False) as tab:
        changed = tab.getcell("DATA", 0)
        changed[0, 0] += 1
        tab.putcell("DATA", 0, changed)
    with pytest.raises(ValueError, match="value mismatch"):
        qualification._compare_values(source, restored, 2, "msutils-native-model/v1")


@pytest.mark.parametrize("extent,refuses", [(MAX_TILE_BYTES // 2, False), (MAX_TILE_BYTES // 2 + 1, True)])
def test_qualification_uchar_bound_uses_widened_reader_size(extent, refuses):
    tab = NumericTable([extent], value_type="uchar")
    if refuses:
        with pytest.raises(ValueError, match="qualification-only"):
            qualification._comparison_batch(tab, "BYTES", 128)
    else:
        assert qualification._comparison_batch(tab, "BYTES", 128) == (1, False)
    assert tab.reads == []


def test_qualification_real_uchar_reader_width_sets_batch_size(tmp_path, monkeypatch):
    pytest.importorskip("casacore")
    import numpy as np
    from casacore.tables import makearrcoldesc, maketabdesc, table
    from shinobi import state_codecs

    with table(str(tmp_path / "uchar.table"), maketabdesc([makearrcoldesc("BYTES", 0, shape=[4], valuetype="uchar")]), nrow=2, readonly=False, ack=False) as tab:
        values = np.array([[0, 127, 128, 255], [255, 128, 127, 0]], dtype="uint8")
        tab.putcol("BYTES", values)
        readback = tab.getcol("BYTES")
        assert readback.dtype == np.dtype("uint16")
        assert readback.dtype.itemsize == 2
        assert np.array_equal(readback, values)
        monkeypatch.setattr(state_codecs, "MAX_TILE_BYTES", 8)
        rows, empty = qualification._comparison_batch(tab, "BYTES", 2)
        assert (rows, empty) == (1, False)
        assert tab.getcol("BYTES", startrow=0, nrow=rows).nbytes == 8
