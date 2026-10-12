"""Real Astropy structural scanner and declared WCS contract acceptance."""

import subprocess
import sys

import pytest

from pydantic import ValidationError

from shinobi import FITSHDU
from shinobi.fits import FITSStatus, FITSInspectionLimits, inspect_fits_image, inspect_fits_table


fits = pytest.importorskip("astropy.io.fits")
np = pytest.importorskip("numpy")


def image(path, shape=(2, 3), *, wcs=True):
    hdu = fits.PrimaryHDU(np.zeros(shape, dtype=np.int16)) if shape else fits.PrimaryHDU()
    if wcs:
        for axis in range(1, len(shape) + 1):
            hdu.header[f"CTYPE{axis}"] = "LINEAR"
    hdu.writeto(path, overwrite=True)
    return path


def test_declaration_no_astropy(tmp_path):
    code = "from pydantic import BaseModel; from shinobi import FITSImage,FITSTable; import sys; class_text=''; M=type('M',(BaseModel,),{'__annotations__':{'image':FITSImage,'table':FITSTable}}); print(M(image='missing',table='missing').model_dump_json()); assert not any(x.startswith('astropy') for x in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "kwargs",
    [
        {"index": -1},
        {"index": True},
        {"extname": "SCI"},
        {"extname": "", "extver": 1},
        {"index": 1, "extname": "SCI", "extver": 1},
        {"extname": "SCI", "extver": True},
        {"index": None},
        {"unknown": 0},
    ],
)
def test_closed_selector(kwargs):
    with pytest.raises(ValidationError):
        FITSHDU(**kwargs)


@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 3, 4), (1, 2, 3, 4), (0, 3), (2, 0, 4)])
def test_nd_image_and_wcs(tmp_path, shape):
    observed = inspect_fits_image(image(tmp_path / "x.fits", shape))
    assert observed.valid, observed.message
    assert observed.selected.shape_numpy == shape
    assert observed.selected.shape_fits == shape[::-1]
    assert observed.selected.wcs.pixel_dimensions == len(shape)


def test_no_wcs_and_mismatch_refuse(tmp_path):
    path = image(tmp_path / "x.fits", wcs=False)
    assert not inspect_fits_image(path).valid
    image(path)
    with fits.open(path, mode="update") as hdus:
        hdus[0].header["WCSAXES"] = 3
    observed = inspect_fits_image(path)
    assert not observed.valid
    assert observed.selected.wcs.pixel_dimensions == 3
    assert observed.selected.shape_fits == (3, 2)


def test_header_only_ancillary_table(tmp_path):
    path = tmp_path / "t.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns([], nrows=0, name="DATA")]).writeto(path)
    observed = inspect_fits_table(path, hdu=FITSHDU(extname="DATA", extver=1))
    assert observed.valid, observed.message
    assert observed.selected.row_bytes == observed.selected.row_count == 0
    assert not inspect_fits_image(path).valid
    assert inspect_fits_table(path).status is FITSStatus.WRONG_HDU


@pytest.mark.parametrize("form", ["L", "9X", "B", "I", "J", "K", "3A", "E", "D", "C", "M", "0E", "PE(8)", "QD(5)"])
def test_binary_formats_empty_rows(tmp_path, form):
    path = tmp_path / "t.fits"
    table = fits.BinTableHDU.from_columns([fits.Column(name="same", format=form)], nrows=0)
    if form == "0E":
        path.write_bytes(fits.PrimaryHDU().header.tostring().encode("ascii") + table.header.tostring().encode("ascii"))
    else:
        fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    observed = inspect_fits_table(path, hdu=FITSHDU(index=1))
    assert observed.valid, observed.message
    assert observed.selected.columns[0].format[:2] == form[:2]


@pytest.mark.parametrize("form,width", [("A5", 5), ("I5", 5), ("F8.2", 8), ("E12.4", 12), ("D20.10", 20)])
def test_ascii_empty_rows_direct_header(tmp_path, form, width):
    # Astropy8 from_columns numeric empty ASCII cannot serialize; the legal
    # structure can be authored directly without asking it to invent values.
    primary = fits.PrimaryHDU().header.tostring().encode("ascii")
    header = fits.Header(
        [("XTENSION", "TABLE"), ("BITPIX", 8), ("NAXIS", 2), ("NAXIS1", width), ("NAXIS2", 0), ("PCOUNT", 0), ("GCOUNT", 1), ("TFIELDS", 1), ("TBCOL1", 1), ("TFORM1", form)]
    )
    path = tmp_path / "t.fits"
    path.write_bytes(primary + header.tostring().encode("ascii"))
    observed = inspect_fits_table(path, hdu=FITSHDU(index=1))
    assert observed.valid, observed.message


def test_select_ambiguity_and_wrong_kind(tmp_path):
    path = tmp_path / "t.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns([], name="DATA"), fits.BinTableHDU.from_columns([], name="DATA")]).writeto(path)
    assert inspect_fits_table(path, hdu=FITSHDU(extname="DATA", extver=1)).status is FITSStatus.AMBIGUOUS_HDU
    assert inspect_fits_table(path, hdu=FITSHDU(index=3)).status is FITSStatus.MISSING_HDU


def test_truncation_and_limits(tmp_path):
    path = image(tmp_path / "x.fits")
    assert inspect_fits_image(path, limits=FITSInspectionLimits(max_axes=1)).status is FITSStatus.LIMIT_EXCEEDED
    assert inspect_fits_image(path, limits=FITSInspectionLimits(max_header_bytes=2879)).status is FITSStatus.LIMIT_EXCEEDED
    path.write_bytes(path.read_bytes()[:-1])
    assert inspect_fits_image(path).status is FITSStatus.TRUNCATED


def test_no_payload_api(tmp_path, monkeypatch):
    path = image(tmp_path / "x.fits")

    def forbidden(*args, **kwargs):
        raise AssertionError("payload API called")

    monkeypatch.setattr(fits, "open", forbidden)
    monkeypatch.setattr(fits.PrimaryHDU, "data", property(forbidden))
    monkeypatch.setattr(fits.ImageHDU, "data", property(forbidden))
    monkeypatch.setattr(fits.BinTableHDU, "data", property(forbidden))
    assert inspect_fits_image(path, include_wcs=True).valid


@pytest.mark.parametrize(
    "changes", [{"TFORM1": "2PE(10)", "NAXIS1": 16}, {"TFORM1": "2QE(10)", "NAXIS1": 32}, {"TFORM1": "1Y"}, {"NAXIS1": 99}, {"THEAP": 999}, {"TDIM1": "(2,2)"}, {"ZIMAGE": True}]
)
def test_table_bad_layout(tmp_path, changes):
    path = tmp_path / "t.fits"
    hdu = fits.BinTableHDU.from_columns([fits.Column(name="x", format="J", array=[1])])
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    raw = bytearray(path.read_bytes())
    header = fits.Header.fromstring(bytes(raw[2880:5760]).decode("ascii"))
    for key, value in changes.items():
        header[key] = value
    raw[2880:5760] = header.tostring().encode("ascii")
    path.write_bytes(raw)
    assert not inspect_fits_table(path, hdu=FITSHDU(index=1)).valid


@pytest.mark.parametrize(
    "changes",
    [
        {"CROTA17": 1},
        {"CRDER17": 1},
        {"CSYER17A": 1},
        {"PC017001": 1},
        {"CD001017": 1},
        {"WCSAXES": 100000},
        {"WCSAXESA": 100000},
        {"PV100000_1": 1},
        {"PS100000_1": "x"},
        {"A_ORDER": 100000},
        {"B_100000_0": 1},
        {"CTYPE1": "RA---TAB"},
    ],
)
def test_wcs_preflight_never_constructs_unbounded_wcs(tmp_path, monkeypatch, changes):
    path = image(tmp_path / "x.fits")
    with fits.open(path, mode="update") as hdus:
        for key, value in changes.items():
            hdus[0].header[key] = value
    import astropy.wcs

    def forbidden(*args, **kwargs):
        raise AssertionError("unbounded WCS construction")

    monkeypatch.setattr(astropy.wcs, "WCS", forbidden)
    observed = inspect_fits_image(path)
    assert not observed.valid
    assert observed.selected.wcs.status in ("unsupported", "limit-exceeded")


def test_compressed_table_refuses(tmp_path):
    path = tmp_path / "t.fits"
    table = fits.BinTableHDU.from_columns([])
    table.header["ZTABLE"] = True
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    assert inspect_fits_table(path, hdu=FITSHDU(index=1)).status is FITSStatus.UNSUPPORTED


def test_variable_heap_gap_and_duplicate_physical_columns(tmp_path):
    primary = fits.PrimaryHDU().header.tostring().encode("ascii")
    header = fits.Header(
        [
            ("XTENSION", "BINTABLE"),
            ("BITPIX", 8),
            ("NAXIS", 2),
            ("NAXIS1", 24),
            ("NAXIS2", 0),
            ("PCOUNT", 32),
            ("GCOUNT", 1),
            ("TFIELDS", 2),
            ("THEAP", 8),
            ("TFORM1", "PE(12)"),
            ("TFORM2", "QD(4)"),
            ("TTYPE1", "same"),
            ("TTYPE2", "same"),
        ]
    )
    path = tmp_path / "heap.fits"
    path.write_bytes(primary + header.tostring().encode("ascii") + b"\xff" * 2880)
    descriptor = inspect_fits_table(path, hdu=FITSHDU(index=1))
    assert descriptor.valid, descriptor.message
    assert descriptor.selected.heap_offset == 8
    assert descriptor.selected.heap_bytes == 24
    assert [one.index for one in descriptor.selected.columns] == [1, 2]
    assert [one.name for one in descriptor.selected.columns] == ["same", "same"]


def test_scanner_reads_only_header_offsets(tmp_path, monkeypatch):
    from pathlib import Path

    path = image(tmp_path / "x.fits", (128, 128))
    original = Path.open

    class HeaderOnly:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def tell(self):
            return self.stream.tell()

        def seek(self, *args):
            return self.stream.seek(*args)

        def read(self, count=-1):
            assert self.stream.tell() < 2880, "payload read"
            assert self.stream.tell() + count <= 2880, "read crosses header into pixels"
            return self.stream.read(count)

    def guarded(one, *args, **kwargs):
        stream = original(one, *args, **kwargs)
        return HeaderOnly(stream) if one == path and args == ("rb",) else stream

    monkeypatch.setattr(Path, "open", guarded)
    assert inspect_fits_image(path).valid
