"""Declarative CASA-table types and bounded structural inspection.

``CasaTable`` and ``MeasurementSetV2`` are path-compatible annotations:
Pydantic validates and serializes their values exactly as ``pathlib.Path``.
The attached :class:`DatasetType` is a declaration only.  It deliberately
performs no filesystem or casacore work while a parameter model is built or
serialized.

Inspection is an explicit operation.  :func:`inspect_dataset` reads table
metadata only (names, row count and the MS version keyword); it never reads a
column cell or scans visibility data.  The metadata retained by Shinobi is
bounded, although casacore may materialize complete name lists before those
bounds can be checked.  ``python-casacore`` is imported lazily, so declaring
these types does not add it as a runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema

from shinobi._annotations import walk_model_annotations

CASA_TABLE_PROFILE = "casa-table/v1"
MSV2_STRUCTURAL_PROFILE = "msv2-structural/v1"


class DatasetKind(str, Enum):
    """Dataset families understood by the structural inspector."""

    CASA_TABLE = "casa-table"
    MEASUREMENT_SET_V2 = "measurement-set-v2"


@dataclass(frozen=True)
class DatasetType:
    """Serializable declaration attached to a path annotation.

    ``profile`` versions the exact structural contract.  New requirements
    must get a new profile rather than silently changing the meaning of an
    existing annotation.
    """

    kind: DatasetKind
    profile: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DatasetKind):
            object.__setattr__(self, "kind", DatasetKind(self.kind))
        if not isinstance(self.profile, str) or not self.profile:
            raise ValueError("dataset profile must be a non-empty string")


CASA_TABLE_V1 = DatasetType(kind=DatasetKind.CASA_TABLE, profile=CASA_TABLE_PROFILE)
MSV2_STRUCTURAL_V1 = DatasetType(kind=DatasetKind.MEASUREMENT_SET_V2, profile=MSV2_STRUCTURAL_PROFILE)


def _json_schema(declaration: DatasetType) -> WithJsonSchema:
    return WithJsonSchema(
        {
            "type": "string",
            "format": "path",
            "x-shinobi-dataset": {"kind": declaration.kind.value, "profile": declaration.profile},
        }
    )


CasaTable = Annotated[Path, CASA_TABLE_V1, _json_schema(CASA_TABLE_V1)]
"""A path declared to contain a readable CASA table (profile v1)."""

MeasurementSetV2 = Annotated[Path, MSV2_STRUCTURAL_V1, _json_schema(MSV2_STRUCTURAL_V1)]
"""A path declared to satisfy the bounded MSv2 structural profile v1."""


class DatasetStatus(str, Enum):
    """Outcome categories returned by :func:`inspect_dataset`."""

    MISSING_PATH = "missing-path"
    NOT_A_DIRECTORY = "not-a-directory"
    INSPECTOR_UNAVAILABLE = "inspector-unavailable"
    NOT_A_CASA_TABLE = "not-a-casa-table"
    NOT_A_MEASUREMENT_SET = "not-a-measurement-set"
    INCOMPLETE_MEASUREMENT_SET_V2 = "incomplete-measurement-set-v2"
    UNSUPPORTED = "unsupported"
    VALID = "valid"


@dataclass(frozen=True)
class InspectionLimits:
    """Bounds on metadata retained by one inspection.

    Casacore supplies column and keyword names as metadata.  The inspector
    refuses an object whose metadata exceeds these bounds instead of silently
    truncating it; column cells are never read at any limit.
    """

    max_columns: int = 4096
    max_keywords: int = 4096

    def __post_init__(self) -> None:
        if self.max_columns < 1 or self.max_keywords < 1:
            raise ValueError("inspection metadata limits must be positive")


class DatasetDescriptor(BaseModel):
    """Serializable observation produced by explicit structural inspection.

    The complete column and keyword name sets are retained (within
    :class:`InspectionLimits`), including optional and custom columns.  The
    descriptor is an observation, not a parameter annotation and not a cache
    identity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    expected: DatasetType
    status: DatasetStatus
    message: str
    nrows: int | None = None
    columns: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    subtables: tuple[str, ...] = ()
    missing_columns: tuple[str, ...] = ()
    missing_primary_data_columns: tuple[str, ...] = ()
    missing_subtables: tuple[str, ...] = ()
    unreadable_subtables: tuple[str, ...] = ()
    missing_subtable_columns: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    unsupported_features: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Whether the observed path satisfies the requested profile."""

        return self.status is DatasetStatus.VALID


# Required predefined columns and subtables from the MeasurementSet v2 main
# table.  DATA/FLOAT_DATA/LAG_DATA are alternatives governed by the collective
# requirement below rather than individually required columns;
# CORRECTED_DATA is derived. SOURCE and calibration/weather tables are optional.
_MSV2_REQUIRED_COLUMNS = frozenset(
    {
        "ANTENNA1",
        "ANTENNA2",
        "ARRAY_ID",
        "DATA_DESC_ID",
        "EXPOSURE",
        "FEED1",
        "FEED2",
        "FIELD_ID",
        "FLAG",
        "FLAG_CATEGORY",
        "FLAG_ROW",
        "INTERVAL",
        "OBSERVATION_ID",
        "PROCESSOR_ID",
        "SCAN_NUMBER",
        "SIGMA",
        "STATE_ID",
        "TIME",
        "TIME_CENTROID",
        "UVW",
        "WEIGHT",
    }
)
_MSV2_PRIMARY_DATA_COLUMNS = frozenset({"DATA", "FLOAT_DATA", "LAG_DATA"})
_MSV2_REQUIRED_SUBTABLES = frozenset(
    {
        "ANTENNA",
        "DATA_DESCRIPTION",
        "FEED",
        "FIELD",
        "FLAG_CMD",
        "HISTORY",
        "OBSERVATION",
        "POINTING",
        "POLARIZATION",
        "PROCESSOR",
        "SPECTRAL_WINDOW",
        "STATE",
    }
)

# The required columns of every mandatory subtable in the MSv2 layout.  Empty
# row sets are legal (and useful for generated fixtures), but an empty or
# partial schema is not.  Optional SOURCE and other standard/custom subtables
# remain legal because the profile only validates mandatory tables.
_MSV2_REQUIRED_SUBTABLE_COLUMNS = {
    "ANTENNA": frozenset({"NAME", "STATION", "TYPE", "MOUNT", "POSITION", "OFFSET", "DISH_DIAMETER", "FLAG_ROW"}),
    "DATA_DESCRIPTION": frozenset({"SPECTRAL_WINDOW_ID", "POLARIZATION_ID", "FLAG_ROW"}),
    "FEED": frozenset(
        {
            "ANTENNA_ID",
            "FEED_ID",
            "SPECTRAL_WINDOW_ID",
            "TIME",
            "INTERVAL",
            "NUM_RECEPTORS",
            "BEAM_ID",
            "BEAM_OFFSET",
            "POLARIZATION_TYPE",
            "POL_RESPONSE",
            "POSITION",
            "RECEPTOR_ANGLE",
        }
    ),
    "FIELD": frozenset({"NAME", "CODE", "TIME", "NUM_POLY", "DELAY_DIR", "PHASE_DIR", "REFERENCE_DIR", "SOURCE_ID", "FLAG_ROW"}),
    "FLAG_CMD": frozenset({"TIME", "INTERVAL", "TYPE", "REASON", "LEVEL", "SEVERITY", "APPLIED", "COMMAND"}),
    "HISTORY": frozenset({"TIME", "OBSERVATION_ID", "MESSAGE", "PRIORITY", "ORIGIN", "OBJECT_ID", "APPLICATION", "CLI_COMMAND", "APP_PARAMS"}),
    "OBSERVATION": frozenset({"TELESCOPE_NAME", "TIME_RANGE", "OBSERVER", "LOG", "SCHEDULE_TYPE", "SCHEDULE", "PROJECT", "RELEASE_DATE", "FLAG_ROW"}),
    "POINTING": frozenset({"ANTENNA_ID", "TIME", "INTERVAL", "NAME", "NUM_POLY", "TIME_ORIGIN", "DIRECTION", "TARGET", "TRACKING"}),
    "POLARIZATION": frozenset({"NUM_CORR", "CORR_TYPE", "CORR_PRODUCT", "FLAG_ROW"}),
    "PROCESSOR": frozenset({"TYPE", "SUB_TYPE", "TYPE_ID", "MODE_ID", "FLAG_ROW"}),
    "SPECTRAL_WINDOW": frozenset(
        {
            "NUM_CHAN",
            "NAME",
            "REF_FREQUENCY",
            "CHAN_FREQ",
            "CHAN_WIDTH",
            "MEAS_FREQ_REF",
            "EFFECTIVE_BW",
            "RESOLUTION",
            "TOTAL_BANDWIDTH",
            "NET_SIDEBAND",
            "IF_CONV_CHAIN",
            "FREQ_GROUP",
            "FREQ_GROUP_NAME",
            "FLAG_ROW",
        }
    ),
    "STATE": frozenset({"SIG", "REF", "CAL", "LOAD", "SUB_SCAN", "OBS_MODE", "FLAG_ROW"}),
}


def dataset_declarations(model: type[BaseModel]) -> dict[str, DatasetType]:
    """Find every dataset declaration reachable from a Pydantic model.

    Returned keys are qualified field paths.  ``[]`` denotes a sequence item
    and ``.*`` a mapping value, for example ``groups.*.members[].ms``.  Nested
    models and unions are traversed recursively.  An ancestor stack prevents
    self-referential models from recursing forever without suppressing the
    same model used independently by two fields.

    Raises:
        TypeError: If one qualified path contains conflicting declarations.
    """

    result: dict[str, DatasetType] = {}

    def record(path: str, declaration: DatasetType) -> None:
        previous = result.get(path)
        if previous is not None and previous != declaration:
            raise TypeError(f"field {model.__name__}.{path} has conflicting dataset declarations: {(previous, declaration)!r}")
        result[path] = declaration

    for node in walk_model_annotations(model):
        for declaration in dict.fromkeys(item for item in node.metadata if isinstance(item, DatasetType)):
            record(node.path, declaration)
    return result


def dataset_fields(model: type[BaseModel]) -> dict[str, DatasetType]:
    """Backward-compatible name for :func:`dataset_declarations`.

    Unlike its original implementation, this returns qualified paths for
    declarations in nested models and containers as well as top-level fields.
    """

    return dataset_declarations(model)


def _load_table_factory() -> Any:
    """Import casacore only at the explicit inspection boundary."""

    from casacore.tables import table

    return table


def _descriptor(path: Path, expected: DatasetType, status: DatasetStatus, message: str, **kwargs: Any) -> DatasetDescriptor:
    return DatasetDescriptor(path=path, expected=expected, status=status, message=message, **kwargs)


def inspect_dataset(
    path: str | Path,
    expected: DatasetType = MSV2_STRUCTURAL_V1,
    *,
    limits: InspectionLimits | None = None,
) -> DatasetDescriptor:
    """Inspect bounded CASA-table metadata against a named profile.

    Args:
        path: Candidate on-disk table directory.
        expected: Dataset declaration/profile to check.
        limits: Maximum metadata-name counts retained in the descriptor.

    Returns:
        A serializable descriptor with a precise status.  Ordinary invalid
        datasets are results rather than exceptions; programmer errors such
        as an unknown profile are reported as ``UNSUPPORTED``.
    """

    candidate = Path(path)
    limits = limits or InspectionLimits()
    supported = {
        (DatasetKind.CASA_TABLE, CASA_TABLE_PROFILE),
        (DatasetKind.MEASUREMENT_SET_V2, MSV2_STRUCTURAL_PROFILE),
    }
    if (expected.kind, expected.profile) not in supported:
        feature = f"dataset profile {expected.profile!r} for {expected.kind.value}"
        return _descriptor(candidate, expected, DatasetStatus.UNSUPPORTED, f"unsupported {feature}", unsupported_features=(feature,))
    if not candidate.exists():
        return _descriptor(candidate, expected, DatasetStatus.MISSING_PATH, f"dataset path does not exist: {candidate}")
    if not candidate.is_dir():
        return _descriptor(candidate, expected, DatasetStatus.NOT_A_DIRECTORY, f"CASA tables must be directories: {candidate}")

    try:
        table_factory = _load_table_factory()
    except (ImportError, OSError):
        feature = "python-casacore metadata inspection"
        return _descriptor(
            candidate,
            expected,
            DatasetStatus.INSPECTOR_UNAVAILABLE,
            "python-casacore is not installed; install it in the inspection environment",
            unsupported_features=(feature,),
        )

    try:
        table = table_factory(str(candidate), readonly=True, ack=False)
    except Exception as exc:
        return _descriptor(candidate, expected, DatasetStatus.NOT_A_CASA_TABLE, f"directory is not a readable CASA table: {type(exc).__name__}: {exc}")

    try:
        try:
            columns = tuple(sorted(str(name) for name in table.colnames()))
            keywords = tuple(sorted(str(name) for name in table.keywordnames()))
            nrows = int(table.nrows())
        except Exception as exc:
            status = DatasetStatus.INCOMPLETE_MEASUREMENT_SET_V2 if expected.kind is DatasetKind.MEASUREMENT_SET_V2 else DatasetStatus.NOT_A_CASA_TABLE
            return _descriptor(candidate, expected, status, f"CASA table metadata is unreadable: {type(exc).__name__}: {exc}")

        over_bounds = []
        if len(columns) > limits.max_columns:
            over_bounds.append(f"column count {len(columns)} exceeds limit {limits.max_columns}")
        if len(keywords) > limits.max_keywords:
            over_bounds.append(f"keyword count {len(keywords)} exceeds limit {limits.max_keywords}")
        if over_bounds:
            return _descriptor(
                candidate,
                expected,
                DatasetStatus.UNSUPPORTED,
                "; ".join(over_bounds),
                nrows=nrows,
                unsupported_features=tuple(over_bounds),
            )

        if expected.kind is DatasetKind.CASA_TABLE:
            return _descriptor(
                candidate,
                expected,
                DatasetStatus.VALID,
                f"readable CASA table satisfying {expected.profile}",
                nrows=nrows,
                columns=columns,
                keywords=keywords,
            )

        if "MS_VERSION" not in keywords:
            return _descriptor(
                candidate,
                expected,
                DatasetStatus.NOT_A_MEASUREMENT_SET,
                "readable CASA table has no MS_VERSION keyword",
                nrows=nrows,
                columns=columns,
                keywords=keywords,
            )
        try:
            version = table.getkeyword("MS_VERSION")
        except Exception as exc:
            return _descriptor(
                candidate,
                expected,
                DatasetStatus.INCOMPLETE_MEASUREMENT_SET_V2,
                f"MS_VERSION keyword cannot be read: {type(exc).__name__}: {exc}",
                nrows=nrows,
                columns=columns,
                keywords=keywords,
            )
        if not isinstance(version, (int, float)) or float(version) != 2.0:
            feature = f"MeasurementSet version {version!r}"
            return _descriptor(
                candidate,
                expected,
                DatasetStatus.UNSUPPORTED,
                f"{feature} is not supported by {expected.profile}",
                nrows=nrows,
                columns=columns,
                keywords=keywords,
                unsupported_features=(feature,),
            )

        missing_columns = tuple(sorted(_MSV2_REQUIRED_COLUMNS.difference(columns)))
        missing_primary_data_columns = () if _MSV2_PRIMARY_DATA_COLUMNS.intersection(columns) else tuple(sorted(_MSV2_PRIMARY_DATA_COLUMNS))
        missing_subtables = tuple(sorted(_MSV2_REQUIRED_SUBTABLES.difference(keywords)))
        subtables = tuple(sorted(_MSV2_REQUIRED_SUBTABLES.intersection(keywords)))
        unreadable_subtables = []
        missing_subtable_columns: dict[str, tuple[str, ...]] = {}
        subtable_bounds = []
        for name in subtables:
            try:
                subtable = table_factory(f"{candidate}::{name}", readonly=True, ack=False)
            except Exception:
                unreadable_subtables.append(name)
                continue
            try:
                # These are metadata-only calls.  Casacore can materialize
                # each complete name list, so enforce the acceptance/retention
                # bounds immediately after each call and never read cells.
                subtable_columns = tuple(str(column) for column in subtable.colnames())
                subtable_keywords = tuple(str(keyword) for keyword in subtable.keywordnames())
                subtable.nrows()
                if len(subtable_columns) > limits.max_columns:
                    subtable_bounds.append(f"{name} column count {len(subtable_columns)} exceeds limit {limits.max_columns}")
                if len(subtable_keywords) > limits.max_keywords:
                    subtable_bounds.append(f"{name} keyword count {len(subtable_keywords)} exceeds limit {limits.max_keywords}")
                missing = tuple(sorted(_MSV2_REQUIRED_SUBTABLE_COLUMNS[name].difference(subtable_columns)))
                if missing:
                    missing_subtable_columns[name] = missing
            except Exception:
                unreadable_subtables.append(name)
            finally:
                subtable.close()
        unreadable_subtables_tuple = tuple(unreadable_subtables)
        if subtable_bounds:
            return _descriptor(
                candidate,
                expected,
                DatasetStatus.UNSUPPORTED,
                "; ".join(subtable_bounds),
                nrows=nrows,
                columns=columns,
                keywords=keywords,
                subtables=subtables,
                unsupported_features=tuple(subtable_bounds),
            )
        if missing_columns or missing_primary_data_columns or missing_subtables or unreadable_subtables_tuple or missing_subtable_columns:
            parts = []
            if missing_columns:
                parts.append(f"missing required columns {list(missing_columns)!r}")
            if missing_primary_data_columns:
                parts.append(f"missing primary data column; at least one of {list(missing_primary_data_columns)!r} is required")
            if missing_subtables:
                parts.append(f"missing required subtables {list(missing_subtables)!r}")
            if unreadable_subtables_tuple:
                parts.append(f"unreadable required subtables {list(unreadable_subtables_tuple)!r}")
            if missing_subtable_columns:
                parts.append(f"required subtables missing columns {missing_subtable_columns!r}")
            return _descriptor(
                candidate,
                expected,
                DatasetStatus.INCOMPLETE_MEASUREMENT_SET_V2,
                "; ".join(parts),
                nrows=nrows,
                columns=columns,
                keywords=keywords,
                subtables=subtables,
                missing_columns=missing_columns,
                missing_primary_data_columns=missing_primary_data_columns,
                missing_subtables=missing_subtables,
                unreadable_subtables=unreadable_subtables_tuple,
                missing_subtable_columns=missing_subtable_columns,
            )
        return _descriptor(
            candidate,
            expected,
            DatasetStatus.VALID,
            f"MeasurementSet satisfies {expected.profile}",
            nrows=nrows,
            columns=columns,
            keywords=keywords,
            subtables=subtables,
        )
    finally:
        table.close()


def inspect_casa_table(path: str | Path, *, limits: InspectionLimits | None = None) -> DatasetDescriptor:
    """Inspect ``path`` against :data:`CasaTable`'s current profile."""

    return inspect_dataset(path, CASA_TABLE_V1, limits=limits)


def inspect_measurement_set_v2(path: str | Path, *, limits: InspectionLimits | None = None) -> DatasetDescriptor:
    """Inspect ``path`` against :data:`MeasurementSetV2`'s current profile."""

    return inspect_dataset(path, MSV2_STRUCTURAL_V1, limits=limits)


__all__ = [
    "CASA_TABLE_PROFILE",
    "CASA_TABLE_V1",
    "MSV2_STRUCTURAL_PROFILE",
    "MSV2_STRUCTURAL_V1",
    "CasaTable",
    "DatasetDescriptor",
    "DatasetKind",
    "DatasetStatus",
    "DatasetType",
    "InspectionLimits",
    "MeasurementSetV2",
    "dataset_declarations",
    "dataset_fields",
    "inspect_casa_table",
    "inspect_dataset",
    "inspect_measurement_set_v2",
]
