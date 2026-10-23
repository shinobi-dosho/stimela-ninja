"""Dosho model generation with explicit trusted Shinobi dtype bindings."""

from __future__ import annotations
from pathlib import Path
from typing import Any
from dosho.validation import modelgen as _shared
from dosho.validation.modelgen import (
    COMMON_LEAF_KEYS,
    ResolutionCycleGuard,
    TypeBinding,
    sanitize,
    sanitize_unique,
    validate_choices,
    constrain_string,
    validate_nullable,
    required_field_spec,
    deep_merge,
    merge_param_meta,
    merge_field_meta,
    get_path,
    resolve_use,
    resolve_directive,
    resolve_package_root,
    contain_include,
)
from shinobi.datasets import FITSImage, FITSTable, CasaTab, DatasetDeclarationError, MSv2
from shinobi.products import DirectoryBundle, ProductFamily

TYPE_BINDINGS = {
    name: TypeBinding(annotation, protect_choices=name in {"msv2", "measurementsetv2", "casatab", "casatable", "fitsimage", "fitstable"})
    for name, annotation in {
        "msv2": MSv2,
        "measurementsetv2": MSv2,
        "casatab": CasaTab,
        "casatable": CasaTab,
        "fitsimage": FITSImage,
        "fitstable": FITSTable,
        "directorybundle": DirectoryBundle,
        "productfamily[file]": ProductFamily[Path],
        "productfamily[fitsimage]": ProductFamily[FITSImage],
        "productfamily[fitstable]": ProductFamily[FITSTable],
        "productfamily[directorybundle]": ProductFamily[DirectoryBundle],
    }.items()
}
TYPE_BINDINGS["msv4"] = TypeBinding(None, error="MSv4 is reserved and unsupported as a dataset dtype")


def _call(function, *args, **kwargs):
    try:
        return function(*args, type_bindings=TYPE_BINDINGS, **kwargs)
    except _shared.DtypeError as exc:
        raise DatasetDeclarationError(str(exc)) from exc


def dtype_to_type(dtype: str) -> Any:
    return _call(_shared.dtype_to_type, dtype)


def narrow_choices(py_type, choices):
    from shinobi.datasets import annotation_has_dataset

    if choices and annotation_has_dataset(py_type):
        raise DatasetDeclarationError("strict dataset choices are unsupported; choices would erase dataset metadata")
    return _call(_shared.narrow_choices, py_type, choices)


def build_model(name, fields, **kwargs):
    return _call(_shared.build_model, name, fields, **kwargs)


def is_file_dtype(dtype: str) -> bool:
    """Classify atomic dynamic path dtypes through the same trusted bindings."""
    from typing import Annotated, get_args, get_origin

    annotation = dtype_to_type(dtype)
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return isinstance(annotation, type) and issubclass(annotation, Path)


__all__ = [
    "COMMON_LEAF_KEYS",
    "ResolutionCycleGuard",
    "sanitize",
    "sanitize_unique",
    "validate_choices",
    "constrain_string",
    "validate_nullable",
    "required_field_spec",
    "deep_merge",
    "merge_param_meta",
    "merge_field_meta",
    "get_path",
    "resolve_use",
    "resolve_directive",
    "resolve_package_root",
    "contain_include",
    "build_model",
    "dtype_to_type",
    "narrow_choices",
    "is_file_dtype",
]
