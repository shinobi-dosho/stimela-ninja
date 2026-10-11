"""Bounded structural FITS inspection without opening pixel or table arrays.

The scanner reads header records only and seeks over data. Astropy is imported
at the explicit inspection boundary, never by annotations or schema loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import re
import warnings
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from shinobi.datasets import DatasetType, FITS_IMAGE_V1, FITS_TABLE_V1


class FITSHDU(BaseModel):
    """Exactly one physical HDU index, or an EXTNAME/EXTVER pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    index: StrictInt | None = Field(default=None, ge=0)
    extname: str | None = None
    extver: StrictInt | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _selector(self):
        if not self.model_fields_set:
            object.__setattr__(self, "index", 0)
        if self.index is not None:
            if self.extname is not None or self.extver is not None:
                raise ValueError("FITS HDU uses index OR extname plus extver")
        elif not isinstance(self.extname, str) or not self.extname.strip() or self.extver is None:
            raise ValueError("named FITS HDU requires nonempty extname and positive extver")
        return self


class FITSStatus(str, Enum):
    MISSING_PATH = "missing-path"
    WRONG_KIND = "wrong-kind"
    INSPECTOR_UNAVAILABLE = "inspector-unavailable"
    MALFORMED = "malformed"
    TRUNCATED = "truncated"
    MISSING_HDU = "missing-hdu"
    AMBIGUOUS_HDU = "ambiguous-hdu"
    WRONG_HDU = "wrong-hdu"
    UNSUPPORTED = "unsupported"
    LIMIT_EXCEEDED = "limit-exceeded"
    VALID = "valid"


@dataclass(frozen=True)
class FITSInspectionLimits:
    max_hdus: int = 128
    max_axes: int = 16
    max_cards_per_hdu: int = 4096
    max_header_bytes: int = 16 * 1024 * 1024
    max_columns: int = 4096

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise ValueError("FITS inspection limits must be positive integers")


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FITSColumn(_Record):
    index: int
    name: str | None = None
    unit: str | None = None
    format: str
    offset: int
    width: int
    repeat: int
    element_format: str
    variable: bool = False
    maximum: int | None = None
    dimensions: tuple[int, ...] | None = None
    scale: Any = None
    zero: Any = None
    null: Any = None


class FITSWCS(_Record):
    status: str
    message: str = ""
    pixel_dimensions: int | None = None
    world_dimensions: int | None = None
    pixel_shape: tuple[int, ...] | None = None
    axis_names: tuple[str, ...] = ()
    axis_units: tuple[str, ...] = ()
    physical_types: tuple[str | None, ...] = ()
    warnings: tuple[str, ...] = ()


class FITSHDUDescriptor(_Record):
    index: int
    kind: str
    extname: str | None = None
    extver: int = 1
    bitpix: int
    shape_fits: tuple[int, ...]
    shape_numpy: tuple[int, ...]
    header_offset: int
    header_bytes: int
    data_offset: int
    data_bytes: int
    padded_data_bytes: int
    scale: Any = None
    zero: Any = None
    columns: tuple[FITSColumn, ...] = ()
    row_bytes: int | None = None
    row_count: int | None = None
    heap_offset: int | None = None
    heap_bytes: int | None = None
    wcs: FITSWCS | None = None


class FITSDescriptor(_Record):
    path: Path
    expected: DatasetType
    selector: FITSHDU
    status: FITSStatus
    message: str
    parser_version: str | None = None
    hdus: tuple[FITSHDUDescriptor, ...] = ()
    selected_index: int | None = None

    @property
    def valid(self):
        return self.status is FITSStatus.VALID

    @property
    def selected(self):
        return next((one for one in self.hdus if one.index == self.selected_index), None)


class _Invalid(ValueError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


def _fail(message, status=FITSStatus.MALFORMED):
    raise _Invalid(status, message)


def _integer(header, name, *, default=None, minimum=None):
    value = header.get(name, default)
    if type(value) is not int or (minimum is not None and value < minimum):
        _fail(f"{name} must be an integer" + (f" >= {minimum}" if minimum is not None else ""))
    return value


def _unique(header, name):
    if name in header and header.count(name) > 1:
        _fail(f"duplicate structural keyword {name}")


def _dimensions(header, name, repeat, variable):
    value = header.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"\(\s*\d+(?:\s*,\s*\d+)*\s*\)", value):
        _fail(f"invalid {name}")
    dims = tuple(int(item.strip()) for item in value[1:-1].split(","))
    if not variable and math.prod(dims) != repeat:
        _fail(f"{name} dimensions disagree with TFORM repeat")
    return dims


def _columns(header, kind, row_width, limits):
    count = _integer(header, "TFIELDS", minimum=0)
    if count > limits.max_columns:
        _fail("table column limit exceeded", FITSStatus.LIMIT_EXCEEDED)
    result, offset, occupied = [], 0, []
    sizes = {"L": 1, "B": 1, "I": 2, "J": 4, "K": 8, "A": 1, "E": 4, "D": 8, "C": 8, "M": 16}
    for index in range(1, count + 1):
        for prefix in ("TFORM", "TBCOL", "TDIM", "TTYPE", "TUNIT", "TSCAL", "TZERO", "TNULL"):
            _unique(header, f"{prefix}{index}")
        form = header.get(f"TFORM{index}")
        if not isinstance(form, str):
            _fail(f"missing TFORM{index}")
        variable, maximum = False, None
        if kind == "ascii-table":
            match = re.fullmatch(r"([AIFED])(\d+)(?:\.(\d+))?", form.strip())
            if not match:
                _fail(f"invalid ASCII TFORM{index}: {form!r}")
            element, width, precision = match.groups()
            width = int(width)
            if width < 1 or (element in "FED" and precision is None) or (element in "AI" and precision is not None) or (precision is not None and int(precision) >= width):
                _fail(f"invalid ASCII width/precision in TFORM{index}")
            repeat = width if element == "A" else 1
            offset = _integer(header, f"TBCOL{index}", minimum=1) - 1
            if any(offset < end and offset + width > start for start, end in occupied):
                _fail("overlapping ASCII table columns")
            occupied.append((offset, offset + width))
        else:
            match = re.fullmatch(r"(\d*)([LXBIJKAEDCM])", form.strip())
            vm = re.fullmatch(r"(\d*)([PQ])([LXBIJKAEDCM])(?:\((\d+)\))?", form.strip())
            if vm:
                number, descriptor, element, max_text = vm.groups()
                repeat, variable = int(number or 1), True
                if repeat > 1:
                    _fail("variable array descriptor repeat must be zero or one")
                width = repeat * (8 if descriptor == "P" else 16)
                maximum = int(max_text) if max_text is not None else None
            elif match:
                number, element = match.groups()
                repeat = int(number or 1)
                width = (repeat + 7) // 8 if element == "X" else repeat * sizes[element]
            else:
                _fail(f"invalid binary TFORM{index}: {form!r}")
        if offset + width > row_width:
            _fail(f"column {index} extends beyond NAXIS1")
        dimensions = _dimensions(header, f"TDIM{index}", repeat, variable)
        if dimensions is not None and len(dimensions) > limits.max_axes:
            _fail("TDIM axis limit exceeded", FITSStatus.LIMIT_EXCEEDED)
        result.append(
            FITSColumn(
                index=index,
                name=header.get(f"TTYPE{index}"),
                unit=header.get(f"TUNIT{index}"),
                format=form,
                offset=offset,
                width=width,
                repeat=repeat,
                element_format=element,
                variable=variable,
                maximum=maximum,
                dimensions=dimensions,
                scale=header.get(f"TSCAL{index}"),
                zero=header.get(f"TZERO{index}"),
                null=header.get(f"TNULL{index}"),
            )
        )
        if kind == "binary-table":
            offset += width
    if kind == "binary-table" and offset != row_width:
        _fail("binary table column widths disagree with NAXIS1")
    return tuple(result)


def _wcs(header, shape, limits):
    """Supplemental header-only WCS diagnostics, bounded before construction."""
    try:
        declared = any(re.fullmatch(r"(?:WCSAXES|CTYPE\d+|CRPIX\d+|CRVAL\d+|CDELT\d+|PC\d+_\d+|CD\d+_\d+)", key) for key in header.keys())
        if not declared:
            return FITSWCS(status="missing", message="selected image requires explicitly declared WCS")
        axes = _integer(header, "WCSAXES", default=len(shape), minimum=0)
        for key in header.keys():
            if re.fullmatch(r"WCSAXES[A-Z]?", key):
                axes = max(axes, _integer(header, key, minimum=0))
            sip = re.fullmatch(r"(?:A|B|AP|BP)_(?:ORDER|\d+_\d+)", key)
            if sip:
                # FITS SIP is bounded before Astropy allocates coefficient arrays.
                if key.endswith("_ORDER"):
                    order = _integer(header, key, minimum=0)
                    if order > 9:
                        return FITSWCS(status="limit-exceeded", message="SIP polynomial order exceeds 9")
                elif max(int(item) for item in key.split("_")[1:]) > 9:
                    return FITSWCS(status="limit-exceeded", message="SIP coefficient index exceeds 9")
            param = re.fullmatch(r"(?:PV|PS)(\d+)_(\d+)[A-Z]?", key)
            if param:
                axes = max(axes, int(param[1]))
                if int(param[2]) > 4096:
                    return FITSWCS(status="limit-exceeded", message="WCS parameter index exceeds 4096")
            if key.startswith("CTYPE") and str(header[key]).endswith("-TAB"):
                return FITSWCS(status="unsupported", message="table lookup WCS requires external arrays")
            match = re.fullmatch(r"(?:CTYPE|CRPIX|CRVAL|CDELT|CUNIT|CNAME|CROTA|CRDER|CSYER)(\d+)[A-Z]?", key)
            if match:
                axes = max(axes, int(match[1]))
            matrix = re.fullmatch(r"(?:PC|CD)(\d+)_(\d+)[A-Z]?", key)
            if matrix:
                axes = max(axes, int(matrix[1]), int(matrix[2]))
            legacy = re.fullmatch(r"(?:PC|CD)(\d{3})(\d{3})[A-Z]?", key)
            if legacy:
                axes = max(axes, int(legacy[1]), int(legacy[2]))
            if key.startswith(("CPDIS", "CQDIS", "DP", "DQ", "D2IM")):
                return FITSWCS(status="unsupported", message="external/lookup distortion WCS is unsupported")
        if axes > limits.max_axes:
            return FITSWCS(status="limit-exceeded", message="WCS axis limit exceeded")
        from astropy.wcs import WCS

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            one = WCS(header, fix=False, preserve_units=True)
        diagnostics = tuple(str(item.message)[:2048] for item in caught[:32])
        mismatch = (
            one.pixel_n_dim != len(shape)
            or (one.pixel_shape is not None and tuple(one.pixel_shape) != tuple(shape))
            or (one.array_shape is not None and tuple(one.array_shape) != tuple(shape[::-1]))
        )
        return FITSWCS(
            status="warning" if diagnostics or mismatch else "valid",
            message="WCS pixel dimensions disagree with physical NAXIS" if mismatch else "",
            pixel_dimensions=one.pixel_n_dim,
            world_dimensions=one.world_n_dim,
            pixel_shape=one.pixel_shape,
            axis_names=tuple(str(name)[:1024] for name in one.world_axis_names),
            axis_units=tuple(str(unit)[:1024] for unit in one.world_axis_units),
            physical_types=tuple(one.world_axis_physical_types),
            warnings=diagnostics,
        )
    except Exception as exc:
        return FITSWCS(status="invalid", message=f"{type(exc).__name__}: {exc}"[:2048])


def inspect_fits(
    path: str | Path, expected: DatasetType = FITS_IMAGE_V1, *, hdu: FITSHDU | None = None, limits: FITSInspectionLimits | None = None, include_wcs: bool = False
) -> FITSDescriptor:
    """Inspect every physical HDU and select one image or table contract.

    No payload, rows, heap descriptors, checksums or lookup arrays are read.
    The full physical file must have supported layouts and padded extents.
    """
    candidate, selector, limits = Path(path), hdu or FITSHDU(), limits or FITSInspectionLimits()
    inventory = []
    parser_version = None

    def result(status, message, selected=None):
        return FITSDescriptor(
            path=candidate, expected=expected, selector=selector, status=status, message=message, hdus=tuple(inventory), selected_index=selected, parser_version=parser_version
        )

    if expected not in (FITS_IMAGE_V1, FITS_TABLE_V1):
        return result(FITSStatus.UNSUPPORTED, "unsupported FITS profile")
    if not candidate.exists():
        return result(FITSStatus.MISSING_PATH, f"FITS path does not exist: {candidate}")
    if not candidate.is_file():
        return result(FITSStatus.WRONG_KIND, "FITS requires an ordinary local file")
    try:
        from astropy.io.fits import Header
        import astropy

        parser_version = astropy.__version__
    except (ImportError, OSError) as exc:
        return result(FITSStatus.INSPECTOR_UNAVAILABLE, f"Astropy FITS parser unavailable: {exc}")
    try:
        total_header = 0
        with candidate.open("rb") as stream:
            size = candidate.stat().st_size
            magic = stream.read(6)
            if magic.startswith((b"\x1f\x8b", b"BZh", b"\xfd7zXZ", b"PK")):
                _fail("whole-file compression is unsupported", FITSStatus.UNSUPPORTED)
            stream.seek(0)
            while stream.tell() < size:
                index, start = len(inventory), stream.tell()
                if index >= limits.max_hdus:
                    _fail("HDU limit exceeded", FITSStatus.LIMIT_EXCEEDED)
                blocks, cards, end = [], 0, False
                while not end:
                    if cards + 36 > limits.max_cards_per_hdu or total_header + 2880 > limits.max_header_bytes:
                        _fail("FITS header metadata limit exceeded", FITSStatus.LIMIT_EXCEEDED)
                    block = stream.read(2880)
                    if len(block) != 2880:
                        _fail("truncated FITS header block", FITSStatus.TRUNCATED)
                    total_header += 2880
                    for pos in range(0, 2880, 80):
                        card = block[pos : pos + 80]
                        cards += 1
                        if card[:8] == b"END     ":
                            if card[8:].strip() or block[pos + 80 :].strip():
                                _fail("invalid FITS END card/header padding")
                            end = True
                            break
                    blocks.append(block)
                try:
                    raw = b"".join(blocks).decode("ascii")
                    header = Header.fromstring(raw, sep="")
                except Exception as exc:
                    _fail(f"invalid FITS header: {exc}")
                for key in ("SIMPLE", "XTENSION", "BITPIX", "NAXIS", "PCOUNT", "GCOUNT", "TFIELDS", "THEAP", "EXTNAME", "EXTVER", "GROUPS", "ZIMAGE", "ZTABLE"):
                    _unique(header, key)
                if len(header.cards) < 3 or header.cards[1].keyword != "BITPIX" or header.cards[2].keyword != "NAXIS":
                    _fail("FITS first three cards require SIMPLE/XTENSION, BITPIX, NAXIS order")
                first = header.cards[0].keyword
                if index == 0:
                    if first != "SIMPLE" or header.get("SIMPLE") is not True:
                        _fail("primary FITS HDU must start with SIMPLE = T")
                    kind = "primary-image"
                else:
                    extension = header.get("XTENSION")
                    if first != "XTENSION" or extension not in ("IMAGE", "TABLE", "BINTABLE"):
                        _fail(f"unsupported extension layout {extension!r}", FITSStatus.UNSUPPORTED)
                    kind = {"IMAGE": "image", "TABLE": "ascii-table", "BINTABLE": "binary-table"}[extension]
                if header.get("ZIMAGE", False) or header.get("ZTABLE", False) or header.get("GROUPS", False):
                    _fail("compressed HDUs and random groups/UVFITS are unsupported", FITSStatus.UNSUPPORTED)
                bitpix = _integer(header, "BITPIX")
                if bitpix not in (8, 16, 32, 64, -32, -64):
                    _fail("invalid BITPIX")
                axes = _integer(header, "NAXIS", minimum=0)
                if axes > limits.max_axes:
                    _fail("NAXIS limit exceeded", FITSStatus.LIMIT_EXCEEDED)
                shape = []
                for axis in range(1, axes + 1):
                    _unique(header, f"NAXIS{axis}")
                    shape.append(_integer(header, f"NAXIS{axis}", minimum=0))
                shape = tuple(shape)
                pcount = _integer(header, "PCOUNT", default=0 if index == 0 else None, minimum=0)
                gcount = _integer(header, "GCOUNT", default=1 if index == 0 else None, minimum=1)
                extra = {}
                if kind.endswith("table"):
                    if bitpix != 8 or axes != 2 or gcount != 1:
                        _fail("tables require BITPIX=8, NAXIS=2 and GCOUNT=1")
                    if kind == "ascii-table" and pcount:
                        _fail("ASCII tables require PCOUNT=0")
                    rows = shape[0] * shape[1]
                    heap = _integer(header, "THEAP", default=rows, minimum=rows)
                    if heap > rows + pcount:
                        _fail("THEAP exceeds table data extent")
                    extra = {
                        "columns": _columns(header, kind, shape[0], limits),
                        "row_bytes": shape[0],
                        "row_count": shape[1],
                        "heap_offset": heap,
                        "heap_bytes": rows + pcount - heap,
                    }
                    data_bytes = rows + pcount
                else:
                    if pcount != 0 or gcount != 1:
                        _fail("image requires PCOUNT=0 and GCOUNT=1", FITSStatus.UNSUPPORTED)
                    data_bytes = (math.prod(shape) if shape else 0) * (abs(bitpix) // 8)
                data_offset = stream.tell()
                padded = ((data_bytes + 2879) // 2880) * 2880
                if padded > size - data_offset:
                    _fail("truncated FITS data or padding extent", FITSStatus.TRUNCATED)
                extname = header.get("EXTNAME")
                if extname is not None and not isinstance(extname, str):
                    _fail("EXTNAME must be a string")
                extver = _integer(header, "EXTVER", default=1, minimum=1)
                inventory.append(
                    FITSHDUDescriptor(
                        index=index,
                        kind=kind,
                        extname=extname,
                        extver=extver,
                        bitpix=bitpix,
                        shape_fits=shape,
                        shape_numpy=shape[::-1],
                        header_offset=start,
                        header_bytes=data_offset - start,
                        data_offset=data_offset,
                        data_bytes=data_bytes,
                        padded_data_bytes=padded,
                        scale=header.get("BSCALE"),
                        zero=header.get("BZERO"),
                        wcs=_wcs(header, shape, limits) if (include_wcs or expected == FITS_IMAGE_V1) and not kind.endswith("table") else None,
                        **extra,
                    )
                )
                stream.seek(data_offset + padded)
        matches = (
            [one for one in inventory if one.index == selector.index]
            if selector.index is not None
            else [one for one in inventory if one.extname is not None and one.extname.casefold() == selector.extname.casefold() and one.extver == selector.extver]
        )
        if not matches:
            return result(FITSStatus.MISSING_HDU, "selected HDU does not exist")
        if len(matches) > 1:
            return result(FITSStatus.AMBIGUOUS_HDU, "selected EXTNAME/EXTVER matches multiple HDUs")
        selected = matches[0]
        correct = selected.kind.endswith("table") if expected == FITS_TABLE_V1 else selected.kind in ("primary-image", "image")
        if not correct:
            return result(FITSStatus.WRONG_HDU, "selected HDU has the wrong contract kind", selected.index)
        if expected == FITS_IMAGE_V1 and (selected.wcs is None or selected.wcs.status != "valid"):
            return result(
                FITSStatus.MALFORMED,
                "selected image WCS does not satisfy fits-image/v1: " + (selected.wcs.message or "; ".join(selected.wcs.warnings) or selected.wcs.status),
                selected.index,
            )
        return result(FITSStatus.VALID, f"valid structural {expected.profile}", selected.index)
    except _Invalid as exc:
        return result(exc.status, str(exc))
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        return result(FITSStatus.MALFORMED, f"{type(exc).__name__}: {exc}")


def inspect_fits_image(path, **kwargs):
    return inspect_fits(path, FITS_IMAGE_V1, **kwargs)


def inspect_fits_table(path, **kwargs):
    return inspect_fits(path, FITS_TABLE_V1, **kwargs)
