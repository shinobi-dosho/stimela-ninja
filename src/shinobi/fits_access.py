"""Field-local FITS selection and whole-file filesystem authority."""

from __future__ import annotations
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shinobi.dataset_access import DatasetAddress, DatasetMode
from shinobi.datasets import DatasetKind, DatasetType, dataset_declarations, _executable_dataset_shape
from shinobi.exceptions import DatasetLifecycleUnavailableError
from shinobi.fits import FITSHDU
from shinobi.products import framework_type
from shinobi._annotations import walk_annotation

FITS_KINDS = (DatasetKind.FITS_IMAGE, DatasetKind.FITS_TABLE)


class FITSAccess(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    field: str = Field(min_length=1)
    mode: DatasetMode
    hdu: FITSHDU = Field(default_factory=FITSHDU)


class ResolvedFITSAccess(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    field: str
    element_index: int | None = Field(default=None, ge=0, strict=True)
    declaration: FITSAccess
    expected: DatasetType
    root: Path
    resources: tuple[Path, ...]
    mode: DatasetMode

    @model_validator(mode="after")
    def _consistent(self):
        if not self.root.is_absolute() or self.resources != (self.root,) or self.mode != self.declaration.mode or self.field != self.address.label:
            raise ValueError("FITS access must name one absolute whole-file root and matching address/mode")
        if self.expected.kind not in FITS_KINDS or (self.element_index is not None and self.mode is DatasetMode.CREATE):
            raise ValueError("invalid FITS profile/indexed CREATE")
        return self

    @property
    def address(self):
        return DatasetAddress(self.declaration.field, self.element_index)

    @property
    def writes(self):
        return self.mode is not DatasetMode.READ

    @property
    def requires_exclusive(self) -> bool:
        """FITS reservations are exclusive exactly when the access writes."""
        return self.writes


def _fits_product_shape(annotation):
    """Check every annotation branch, never merely filtered product nodes."""
    from typing import Annotated, Union, get_args, get_origin
    from types import UnionType

    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _fits_product_shape(args[0])
    product = framework_type(annotation)
    if product:
        return any(getattr(item, "kind", None) in FITS_KINDS for node in walk_annotation(product[1]) for item in node.metadata)
    if origin in (Union, UnionType):
        branches = [one for one in args if one is not type(None)]
        return len(branches) == 1 and _fits_product_shape(branches[0])
    if origin in (list, tuple, set, frozenset):
        members = [one for one in args if one is not Ellipsis]
        return bool(members) and all(one == members[0] for one in members) and _fits_product_shape(members[0])
    return False


def fits_product_fields(model):
    """Discover canonical FITS products and refuse unsupported full shapes."""
    declarations = dataset_declarations(model)
    declared = {name.split(".")[0].removesuffix("[]") for name, one in declarations.items() if one.kind in FITS_KINDS}
    result = set()
    for name, field in model.model_fields.items():
        if name not in declared:
            continue
        if any(framework_type(node.annotation) for node in walk_annotation(field.annotation)):
            if not _fits_product_shape(field.annotation):
                raise ValueError(f"FITS product field {name!r} requires a canonical FITS family/member, optional None, or homogeneous direct sequence")
            result.add(name)
    return result


def fits_fields(model, *, reject_unsupported=True):
    result = {}
    products = fits_product_fields(model)
    for qualified, declaration in dataset_declarations(model).items():
        if declaration.kind not in FITS_KINDS or qualified.split(".")[0].removesuffix("[]") in products:
            continue
        name = qualified.removesuffix("[]")
        try:
            if name not in model.model_fields:
                raise DatasetLifecycleUnavailableError(f"FITS field {qualified!r} must be a direct scalar or list")
            result[name] = _executable_dataset_shape(model.model_fields[name], qualified, name, declaration, DatasetLifecycleUnavailableError)
        except DatasetLifecycleUnavailableError:
            if reject_unsupported:
                raise
            result[qualified] = declaration
    return result


def scope_has_fits_contract(scope):
    # Canonical family/member outputs are excluded by fits_fields; direct
    # annotations always carry authority, including output-only CREATE.
    inputs = fits_fields(scope.inputs_model, reject_unsupported=False)
    outputs = fits_fields(scope.outputs_model, reject_unsupported=False)
    lifecycle_accesses = [one for one in scope.fits_accesses if one.field in inputs or one.field in outputs]
    return bool(inputs or outputs or lifecycle_accesses)


def scope_tree_has_fits_contract(scope):
    return scope_has_fits_contract(scope) or any(scope_tree_has_fits_contract(ref.step) for ref in getattr(scope, "steps", ()))


def fits_writes(scope):
    from shinobi.steps.schema import mutated_path_fields, write_path_fields

    inputs, outputs = fits_fields(scope.inputs_model, reject_unsupported=False), fits_fields(scope.outputs_model, reject_unsupported=False)
    explicit = {one.field for one in scope.fits_accesses if one.mode is not DatasetMode.READ}
    reads = {one.field for one in scope.fits_accesses if one.mode is DatasetMode.READ}
    return explicit | (((set(outputs) - set(inputs)) | ((mutated_path_fields(scope) | write_path_fields(scope)) & (set(inputs) | set(outputs)))) - reads)


def validate_scope_fits_accesses(scope):
    fields = set(scope.inputs_model.model_fields) | set(scope.outputs_model.model_fields)
    seen = set()
    for access in scope.fits_accesses:
        if access.field not in fields or access.field in seen:
            raise ValueError("FITS accesses require unique direct field names")
        seen.add(access.field)
    if not scope_has_fits_contract(scope) and not scope.fits_accesses:
        return
    inputs, outputs = fits_fields(scope.inputs_model), fits_fields(scope.outputs_model)
    if getattr(scope, "steps", None) is not None:
        if scope.fits_accesses:
            raise ValueError("FITS access declarations belong on atomic leaves")
        # Recipe fields pass values through; concrete authority is declared
        # and validated on each atomic leaf, including explicit list access.
        return
    from shinobi.steps.schema import mutated_path_fields, write_path_fields

    mutates = mutated_path_fields(scope) | write_path_fields(scope)
    declarations = {one.field: one for one in scope.fits_accesses}
    for name, access in declarations.items():
        if name not in inputs and name not in outputs:
            # Explicit selector for validation-only typed product output.
            if name in fits_product_fields(scope.outputs_model):
                if access.mode is not DatasetMode.READ:
                    raise ValueError("FITS family outputs support validation selection only (read), not lifecycle authority")
                continue
            raise ValueError(f"FITS access {name!r} requires a strict FITS field")
        if access.mode is DatasetMode.READ and name in mutates:
            raise ValueError(f"FITS read access contradicts filesystem write field {name!r}")
    for name, shape in inputs.items():
        if shape.is_list:
            access = declarations.get(name)
            if access is None or access.mode not in (DatasetMode.READ, DatasetMode.WRITE):
                raise ValueError("FITS lists require explicit read or write access")
            if name in write_path_fields(scope) or getattr(scope.input_mutability.get(name), "value", None) == "mutable":
                raise ValueError("FITS lists cannot infer writes or use write_path/mutable")
    for name, shape in outputs.items():
        if shape.is_list and (name not in inputs or declarations.get(name) is None or declarations[name].mode is not DatasetMode.WRITE):
            raise ValueError("FITS list outputs require same-name explicit list WRITE input")
        if name in inputs and inputs[name].declaration != shape.declaration:
            raise ValueError("FITS input/output profile mismatch")


def resolve_scope_fits_accesses(scope, values: dict[str, Any], *, workspace: Path, planned_roots=(), **kwargs):
    from shinobi.steps.schema import static_output_values, mutated_path_fields, write_path_fields

    inputs, outputs = fits_fields(scope.inputs_model), fits_fields(scope.outputs_model)
    paths = {**static_output_values(scope, values), **values}
    explicit = {one.field: one for one in scope.fits_accesses}
    mutated = mutated_path_fields(scope) | write_path_fields(scope)
    resolved = []
    for name, shape in (inputs | outputs).items():
        declaration = explicit.get(name) or FITSAccess(field=name, mode=DatasetMode.CREATE if name not in inputs else DatasetMode.WRITE if name in mutated else DatasetMode.READ)
        value = paths.get(name)
        if value is None:
            continue
        members = value if shape.is_list else [value]
        if not members:
            raise DatasetLifecycleUnavailableError(f"FITS list {name!r} must be nonempty")
        for index, member in enumerate(members):
            # An implicit output template resolves to a string before the
            # outputs model parses it. Preserve typed input validation while
            # accepting that declared output path at planning time.
            if name in outputs and name not in inputs and isinstance(member, str):
                member = Path(member)
            if not isinstance(member, Path):
                raise DatasetLifecycleUnavailableError(f"FITS field {name!r} needs a concrete Path")
            lexical = member if member.is_absolute() else workspace / member
            for ancestor in (lexical, *lexical.parents):
                if ancestor.is_symlink():
                    raise DatasetLifecycleUnavailableError(f"FITS symlink paths are unsupported: {ancestor}")
            root = lexical.resolve()
            if not root.is_relative_to(workspace.resolve()):
                raise DatasetLifecycleUnavailableError(f"FITS root must be contained in workspace: {root}")
            if root.exists() and (not root.is_file() or root.stat().st_nlink != 1):
                raise DatasetLifecycleUnavailableError(f"FITS needs an ordinary unaliased file: {root}")
            element = index if shape.is_list else None
            resolved.append(
                ResolvedFITSAccess(
                    field=DatasetAddress(name, element).label,
                    element_index=element,
                    declaration=declaration,
                    expected=shape.declaration,
                    root=root,
                    resources=(root,),
                    mode=declaration.mode,
                )
            )
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if (
                left.root == right.root
                or left.root.is_relative_to(right.root)
                or right.root.is_relative_to(left.root)
                or (left.root.exists() and right.root.exists() and left.root.samefile(right.root))
            ):
                raise DatasetLifecycleUnavailableError("FITS access roots must be disjoint without aliases")
    return tuple(resolved)
