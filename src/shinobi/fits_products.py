"""One validation walker for fresh and cached strict FITS output products."""

from shinobi.datasets import dataset_declarations
from shinobi.fits_access import FITS_KINDS, fits_product_fields
from shinobi.fits import FITSHDU, inspect_fits
from shinobi.products import iter_product_paths
from shinobi.exceptions import DatasetLifecycleViolationError


def fits_product_contract(scope):
    fits_product_fields(scope.outputs_model)
    declarations = dataset_declarations(scope.outputs_model)
    selectors = {one.field: one.hdu for one in scope.fits_accesses}
    return [
        [qualified, {"kind": declaration.kind.value, "profile": declaration.profile}, selectors.get(qualified.split(".")[0].removesuffix("[]"), FITSHDU()).model_dump(mode="json")]
        for qualified, declaration in sorted(declarations.items())
        if declaration.kind in FITS_KINDS
    ]


def validate_fits_outputs(scope, outputs, *, workspace=None, skip_fields=()):
    try:
        fits_product_fields(scope.outputs_model)
    except ValueError as exc:
        raise DatasetLifecycleViolationError(f"FITS output product shape is unsupported: {exc}") from exc
    if outputs is None:
        return
    from pathlib import Path

    workspace = workspace or Path.cwd()
    selectors = {one.field: one.hdu for one in scope.fits_accesses}
    profiles = {}
    for qualified, declaration in dataset_declarations(scope.outputs_model).items():
        if declaration.kind in FITS_KINDS:
            name = qualified.split(".")[0].removesuffix("[]")
            profiles[name] = declaration
    for name, declaration in profiles.items():
        if name in skip_fields:
            continue
        for path in iter_product_paths(getattr(outputs, name, None)):
            absolute = path if path.is_absolute() else workspace / path
            descriptor = inspect_fits(absolute, declaration, hdu=selectors.get(name, FITSHDU()))
            if not descriptor.valid:
                raise DatasetLifecycleViolationError(f"FITS output {name!r} violates {declaration.profile}: {descriptor.status.value}: {descriptor.message}")
