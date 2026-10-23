"""Load static cab schemas through Dosho and adapt them to exact Shinobi Cabs."""

from __future__ import annotations
from pathlib import Path
from typing import Any
from pydantic import ValidationError
from dosho.validation import compiler
from dosho.validation.compiled import CabSpec, CompiledCab
from dosho.validation.exceptions import CabLoadError as DoshoLoadError
from shinobi.datasets import DatasetDeclarationError, executable_dataset_fields
from shinobi.dataset_access import DatasetAccess
from shinobi.fits_access import FITSAccess
from shinobi.exceptions import CabLoadError
from shinobi.loaders._modelgen import TYPE_BINDINGS
from shinobi.steps.schema import Cab, ParamMeta, ParamPattern


def adapt_cab(compiled: CompiledCab) -> Cab:
    """Validate framework declarations and construct the existing execution adapter."""

    def meta(raw):
        data = raw.model_dump(mode="python")
        data["family"] = data.pop("framework_family")
        return ParamMeta.model_validate(data)

    def pattern(raw):
        data = raw.model_dump(mode="python")
        for segment, source in zip(data["segments"], raw.segments):
            if source.attrs is not None:
                segment["attrs"] = {name: meta(value) for name, value in source.attrs.items()}
        return ParamPattern.model_validate(data)

    for model in (compiled.inputs_model, compiled.outputs_model):
        executable_dataset_fields(model)
    data = {name: getattr(compiled, name) for name in type(compiled).model_fields if name not in {"field_meta", "input_patterns", "output_patterns", "framework_declarations"}}
    declarations = compiled.framework_declarations
    data.update(
        field_meta={name: meta(value) for name, value in compiled.field_meta.items()},
        input_patterns=[pattern(value) for value in compiled.input_patterns],
        output_patterns=[pattern(value) for value in compiled.output_patterns],
        dataset_accesses=_dataset_accesses(declarations.get("dataset_accesses"), cab=compiled.name),
        fits_accesses=_fits_accesses(declarations.get("fits_accesses"), cab=compiled.name),
        derived_reads=declarations.get("derived_reads", {}),
    )
    return Cab(**data)


def _compile(spec: CabSpec) -> Cab:
    try:
        return adapt_cab(compiler.compile_cab(spec, type_bindings=TYPE_BINDINGS))
    except (DoshoLoadError, DatasetDeclarationError) as exc:
        raise CabLoadError(f"cab '{spec.name}': {exc}") from exc
    except ValidationError as exc:
        if any(isinstance(item.get("ctx", {}).get("error"), DatasetDeclarationError) for item in exc.errors()):
            raise CabLoadError(f"cab '{spec.name}': {exc}") from exc
        raise


def load_file(path: str | Path, *, package_roots=None, images=None) -> dict[str, Cab]:
    try:
        return {name: _compile(spec) for name, spec in compiler.load_file(path, package_roots=package_roots, images=images).items()}
    except DoshoLoadError as exc:
        raise CabLoadError(str(exc)) from exc


def loads(text: str, *, package_roots=None, images=None) -> dict[str, Cab]:
    try:
        return {name: _compile(spec) for name, spec in compiler.loads(text, package_roots=package_roots, images=images).items()}
    except DoshoLoadError as exc:
        raise CabLoadError(str(exc)) from exc


def _build_cabdef(name, spec, package_roots, images=None):
    return _compile(compiler.parse_document(name, spec, images=images))


def _fits_accesses(raw: Any, *, cab: str) -> list[FITSAccess]:
    """Validate static FITS access data with cab and index diagnostics."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise CabLoadError(f"cab '{cab}': fits_accesses must be a list of mappings")
    accesses = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise CabLoadError(f"cab '{cab}': fits_accesses[{index}] must be a mapping")
        try:
            accesses.append(FITSAccess.model_validate(entry))
        except ValidationError as exc:
            raise CabLoadError(f"cab '{cab}': fits_accesses[{index}]: {exc}") from exc
    return accesses


def _dataset_accesses(raw: Any, *, cab: str) -> list[DatasetAccess]:
    """Validate static access data with cab and entry-index diagnostics."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise CabLoadError(f"cab '{cab}': dataset_accesses must be a list of mappings")
    accesses = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise CabLoadError(f"cab '{cab}': dataset_accesses[{index}] must be a mapping")
        try:
            accesses.append(DatasetAccess.model_validate(entry))
        except ValidationError as exc:
            raise CabLoadError(f"cab '{cab}': dataset_accesses[{index}]: {exc}") from exc
    return accesses


# Capability negotiated by lazy Dosho runnable-object bridges.
CAB_ADAPTER_CONTRACT_VERSION = 1
