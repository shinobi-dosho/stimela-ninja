"""Serializable MSv2 access declarations and deterministic hazard planning.

Definitions in this module are data only.  Resolving a declaration is an
explicit planning operation: that is the boundary which may inspect an MSv2
closure.  The resulting objects retain paths and metadata, never live
casacore handles.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from enum import Enum
import os
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, model_validator

from shinobi._annotations import walk_annotation
from shinobi.dataset_closure import ClosureStatus, resolve_dataset_closure
from shinobi.datasets import DatasetKind, dataset_declarations


_COLUMN_NAME = r"^[A-Za-z_][A-Za-z0-9_]*$"
_MAX_SELECTION_VALUES = 4096


class DatasetAccessError(ValueError):
    """A dataset access declaration cannot be resolved safely."""


class DatasetMode(str, Enum):
    """How a step uses an MSv2 dataset."""

    READ = "read"
    WRITE = "write"
    CREATE = "create"


class DatasetFallback(str, Enum):
    """Why planning conservatively reserves a whole dataset identity."""

    UNDECLARED = "undeclared"
    UNKNOWN_COLUMNS = "unknown-columns"
    UNKNOWN_PATH = "unknown-path"


class DatasetTable(str, Enum):
    """MSv2 tables whose closure membership is understood by Shinobi."""

    MAIN = "MAIN"
    ANTENNA = "ANTENNA"
    DATA_DESCRIPTION = "DATA_DESCRIPTION"
    DOPPLER = "DOPPLER"
    FEED = "FEED"
    FIELD = "FIELD"
    FLAG_CMD = "FLAG_CMD"
    FREQ_OFFSET = "FREQ_OFFSET"
    HISTORY = "HISTORY"
    OBSERVATION = "OBSERVATION"
    POINTING = "POINTING"
    POLARIZATION = "POLARIZATION"
    PROCESSOR = "PROCESSOR"
    SOURCE = "SOURCE"
    SPECTRAL_WINDOW = "SPECTRAL_WINDOW"
    STATE = "STATE"
    SYSCAL = "SYSCAL"
    WEATHER = "WEATHER"


class DatasetColumns(BaseModel):
    """Column-level intent retained for validation and provenance.

    Column detail deliberately does not relax scheduling: any writer excludes
    every concurrent access to the associated resolved dataset closure.
    ``None`` on :class:`DatasetAccess` means the columns are unknown and
    therefore requests the conservative whole-dataset fallback.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    read: tuple[str, ...] = ()
    write: tuple[str, ...] = ()
    create: tuple[str, ...] = ()
    remove: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _valid_names(self) -> "DatasetColumns":
        import re

        for name in (*self.read, *self.write, *self.create, *self.remove):
            if re.fullmatch(_COLUMN_NAME, name) is None:
                raise ValueError(f"invalid CASA column name {name!r}")
        return self


class DatasetSelection(BaseModel):
    """A bounded, data-only row selection.

    Half-open ``row_ranges`` and finite ID sets cover the common static
    selection shapes without admitting TaQL, Python callables, or another
    expression language.  Scheduling remains conservative at dataset scope.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_ranges: tuple[tuple[int, int], ...] = ()
    data_description_ids: tuple[int, ...] = ()
    field_ids: tuple[int, ...] = ()
    observation_ids: tuple[int, ...] = ()
    scan_numbers: tuple[int, ...] = ()
    array_ids: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _bounded(self) -> "DatasetSelection":
        groups = (
            self.row_ranges,
            self.data_description_ids,
            self.field_ids,
            self.observation_ids,
            self.scan_numbers,
            self.array_ids,
        )
        if sum(len(group) for group in groups) > _MAX_SELECTION_VALUES:
            raise ValueError(f"dataset selection exceeds {_MAX_SELECTION_VALUES} declared values")
        for start, stop in self.row_ranges:
            if start < 0 or stop <= start:
                raise ValueError("row ranges must be non-negative half-open (start, stop) pairs")
        for group in groups[1:]:
            if any(value < 0 for value in group):
                raise ValueError("dataset selection IDs must be non-negative")
        return self


class DatasetAccess(BaseModel):
    """One field-local MSv2 access declaration attached to a ``Scope``.

    ``field`` names an input or output field.  ``root_field`` is needed only
    when ``field`` itself names a referenced subtable whose parent MS cannot
    be inferred from its path (notably an externally stored subtable).
    ``reservation`` is a canonicalizable path envelope used when the dataset
    path is produced dynamically and is therefore unknown during planning.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str
    mode: DatasetMode
    table: DatasetTable = DatasetTable.MAIN
    columns: DatasetColumns | None = None
    selection: DatasetSelection | None = None
    allow_row_count_change: bool = False
    allow_schema_change: bool = False
    allow_keyword_change: bool = False
    root_field: str | None = None
    reservation: Path | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "DatasetAccess":
        columns = self.columns
        if self.mode is DatasetMode.READ:
            if columns is not None and (columns.write or columns.create or columns.remove):
                raise ValueError("read dataset access cannot write, create, or remove columns")
            if self.allow_row_count_change or self.allow_schema_change or self.allow_keyword_change:
                raise ValueError("read dataset access cannot permit row-count, schema, or keyword changes")
        if columns is not None and (columns.create or columns.remove) and not self.allow_schema_change:
            raise ValueError("creating or removing columns requires allow_schema_change=True")
        return self


class ResolvedDatasetAccess(BaseModel):
    """Serializable resolution of one dataset access declaration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str
    declaration: DatasetAccess
    requested_path: Path | None
    root: Path | None
    resources: tuple[Path, ...]
    mode: DatasetMode
    whole_dataset: bool
    path_known: bool
    columns_known: bool
    fallback: DatasetFallback | None = None
    closure_status: ClosureStatus | None = None
    reason: str

    @model_validator(mode="after")
    def _consistent(self) -> "ResolvedDatasetAccess":
        if self.field != self.declaration.field or self.mode is not self.declaration.mode:
            raise ValueError("resolved field/mode must agree with its dataset declaration")
        if not self.resources:
            raise ValueError("resolved dataset access requires at least one resource identity")
        if any(not path.is_absolute() for path in self.resources):
            raise ValueError("resolved dataset resource identities must be absolute")
        if self.columns_known != (self.declaration.columns is not None):
            raise ValueError("columns_known must agree with the dataset declaration")
        if self.path_known:
            if self.requested_path is None or self.root is None:
                raise ValueError("a known dataset path requires requested_path and root")
            if not self.requested_path.is_absolute() or not self.root.is_absolute():
                raise ValueError("known requested/root dataset paths must be absolute")
            if self.root not in self.resources:
                raise ValueError("a known dataset root must be one of its resource identities")
            if self.fallback is DatasetFallback.UNKNOWN_PATH:
                raise ValueError("a known dataset path cannot use the unknown-path fallback")
        elif self.requested_path is not None or self.root is not None or self.fallback is not DatasetFallback.UNKNOWN_PATH:
            raise ValueError("an unknown dataset path requires no requested/root path and the unknown-path fallback")
        if self.fallback in (DatasetFallback.UNDECLARED, DatasetFallback.UNKNOWN_COLUMNS, DatasetFallback.UNKNOWN_PATH) and not self.whole_dataset:
            raise ValueError("a conservative dataset fallback must reserve the whole dataset")
        if self.fallback is DatasetFallback.UNKNOWN_COLUMNS and self.columns_known:
            raise ValueError("the unknown-columns fallback cannot declare known columns")
        if self.fallback is DatasetFallback.UNDECLARED and self.columns_known:
            raise ValueError("an undeclared fallback cannot declare known columns")
        if self.fallback is None and (self.whole_dataset or not self.columns_known):
            raise ValueError("whole-dataset or unknown-column access requires an explicit fallback reason")
        if self.closure_status is not None and (not self.path_known or self.closure_status is not ClosureStatus.VALID):
            raise ValueError("a retained closure status must describe a valid known path")
        if not self.reason:
            raise ValueError("resolved dataset access requires an inspectable reason")
        return self

    @property
    def writes(self) -> bool:
        return self.mode is not DatasetMode.READ


def scope_has_dataset_contract(scope: Any) -> bool:
    """Whether a leaf scope carries an MSv2 annotation or access metadata."""

    declarations = dataset_declarations(scope.inputs_model) | dataset_declarations(scope.outputs_model)
    return bool(declarations or scope.dataset_accesses)


def scope_tree_has_dataset_contract(scope: Any) -> bool:
    """Whether a scope or any nested recipe step carries a dataset contract."""

    if scope_has_dataset_contract(scope):
        return True
    return any(scope_tree_has_dataset_contract(ref.step) for ref in getattr(scope, "steps", ()))


def validate_scope_dataset_accesses(scope: Any) -> None:
    """Validate declaration-to-field links without touching the filesystem."""

    fields = set(scope.inputs_model.model_fields) | set(scope.outputs_model.model_fields)

    def direct_path(model: type[BaseModel], name: str) -> bool:
        field = model.model_fields.get(name)
        if field is None:
            return False
        leaves = [
            node.annotation
            for node in walk_annotation(field.annotation, metadata=tuple(field.metadata), descend_mappings=False, descend_models=False)
            if node.leaf and node.path == "" and node.annotation is not type(None)
        ]
        return bool(leaves) and all(isinstance(annotation, type) and issubclass(annotation, Path) for annotation in leaves)

    def path_compatible(name: str) -> bool:
        return direct_path(scope.inputs_model, name) or direct_path(scope.outputs_model, name)

    seen: set[tuple[str, DatasetTable]] = set()
    for access in scope.dataset_accesses:
        if access.field not in fields:
            raise ValueError(f"scope {scope.name!r} dataset access names unknown field {access.field!r}")
        if not path_compatible(access.field):
            raise ValueError(f"scope {scope.name!r} dataset access field {access.field!r} must be a direct Path or MS-compatible field")
        if access.root_field is not None and access.root_field not in fields:
            raise ValueError(f"scope {scope.name!r} dataset access names unknown root_field {access.root_field!r}")
        if access.root_field is not None and not path_compatible(access.root_field):
            raise ValueError(f"scope {scope.name!r} dataset access root_field {access.root_field!r} must be a direct Path or MS-compatible field")
        if access.root_field == access.field:
            raise ValueError(f"scope {scope.name!r} dataset access root_field must differ from field {access.field!r}")
        key = access.field, access.table
        if key in seen:
            raise ValueError(f"scope {scope.name!r} repeats dataset access for {access.field!r}, table {access.table.value}")
        seen.add(key)


def _canonical(value: Any, workspace: Path) -> Path:
    try:
        path = Path(os.fspath(value))
    except TypeError as exc:
        raise DatasetAccessError(f"dataset path value must be path-compatible, got {type(value).__name__}") from exc
    return (path if path.is_absolute() else workspace / path).resolve()


def _access_label(access: ResolvedDatasetAccess) -> str:
    root = access.root or access.requested_path or access.resources[0]
    return _declaration_label(root.name or str(root), access.declaration)


def _declaration_label(dataset: str, declaration: DatasetAccess) -> str:
    columns = declaration.columns
    if columns is None:
        return dataset
    names = (*columns.write, *columns.create, *columns.remove) if declaration.mode is not DatasetMode.READ else columns.read
    target = declaration.table.value
    return f"{dataset}, {target}.{names[0]}" if names else f"{dataset}, {target}"


def resolve_scope_dataset_accesses(
    scope: Any,
    values: dict[str, Any],
    *,
    workspace: Path,
    planned_roots: dict[Path, tuple[Path, ...]] | None = None,
    unresolved_inputs: set[str] | frozenset[str] = frozenset(),
) -> tuple[ResolvedDatasetAccess, ...]:
    """Resolve one leaf scope's MSv2 declarations against concrete values.

    Unknown columns and omitted declarations degrade to whole-dataset access.
    An unknown path is refused unless its declaration supplies a reservation
    envelope, because otherwise neither ordering nor ownership can cover it.
    """

    workspace = workspace.resolve()
    from shinobi.steps.schema import static_output_values

    values = {**values, **static_output_values(scope, values, unresolved_inputs=unresolved_inputs)}
    planned_roots = planned_roots or {}
    inputs = dataset_declarations(scope.inputs_model)
    outputs = dataset_declarations(scope.outputs_model)
    for name in (*inputs, *outputs):
        if any(token in name for token in (".", "[]", ".*")):
            raise DatasetAccessError(f"scope {scope.name!r} has nested dataset field {name!r}; access planning supports direct fields only")
    declarations = [(item, DatasetFallback.UNKNOWN_COLUMNS if item.columns is None else None) for item in scope.dataset_accesses]
    covered = {item.field for item, _fallback in declarations}
    for field, dataset in {**inputs, **outputs}.items():
        if dataset.kind is not DatasetKind.MEASUREMENT_SET_V2 or field in covered:
            continue
        if field in outputs and field not in inputs:
            mode = DatasetMode.CREATE
        else:
            from shinobi.steps.schema import mutated_path_fields, write_path_fields

            mode = DatasetMode.WRITE if field in mutated_path_fields(scope) | write_path_fields(scope) else DatasetMode.READ
        declarations.append((DatasetAccess(field=field, mode=mode), DatasetFallback.UNDECLARED))

    resolved: list[ResolvedDatasetAccess] = []
    for declaration, fallback in declarations:
        value = values.get(declaration.field)
        if value is None:
            if declaration.reservation is None:
                raise DatasetAccessError(f"scope {scope.name!r} dataset field {declaration.field!r} has an unknown path and no reservation envelope")
            envelope = _canonical(declaration.reservation, workspace)
            resolved.append(
                ResolvedDatasetAccess(
                    field=declaration.field,
                    declaration=declaration,
                    requested_path=None,
                    root=None,
                    resources=(envelope,),
                    mode=declaration.mode,
                    whole_dataset=True,
                    path_known=False,
                    columns_known=declaration.columns is not None,
                    fallback=DatasetFallback.UNKNOWN_PATH,
                    reason=f"unknown path; reserved whole-dataset envelope {envelope}",
                )
            )
            continue

        requested = _canonical(value, workspace)
        if declaration.mode is DatasetMode.CREATE:
            if requested.exists():
                raise DatasetAccessError(
                    f"scope {scope.name!r} CREATE dataset field {declaration.field!r} already exists at {requested}; replacement requires a separate lifecycle policy"
                )
            resolved.append(
                ResolvedDatasetAccess(
                    field=declaration.field,
                    declaration=declaration,
                    requested_path=requested,
                    root=requested,
                    resources=(requested,),
                    mode=declaration.mode,
                    whole_dataset=declaration.columns is None,
                    path_known=True,
                    columns_known=declaration.columns is not None,
                    fallback=fallback,
                    reason=(f"{fallback.value} access; create whole dataset: {requested}" if fallback is not None else f"create {declaration.table.value}: {requested}"),
                )
            )
            continue

        if declaration.root_field is not None and values.get(declaration.root_field) is not None:
            root_candidate = _canonical(values[declaration.root_field], workspace)
        elif declaration.table is not DatasetTable.MAIN and requested.name == declaration.table.value:
            root_candidate = requested.parent
        else:
            root_candidate = requested
        planned = next(((root, resources) for root, resources in planned_roots.items() if root_candidate == root), None)
        if planned is not None:
            planned_root, resources = planned
            resolved.append(
                ResolvedDatasetAccess(
                    field=declaration.field,
                    declaration=declaration,
                    requested_path=requested,
                    root=planned_root,
                    resources=resources,
                    mode=declaration.mode,
                    whole_dataset=declaration.columns is None,
                    path_known=True,
                    columns_known=declaration.columns is not None,
                    fallback=fallback,
                    reason=(
                        f"{fallback.value} access; {declaration.mode.value} planned whole dataset: {planned_root}"
                        if fallback is not None
                        else f"{declaration.mode.value} planned {_declaration_label(planned_root.name or declaration.field, declaration)}"
                    ),
                )
            )
            continue
        closure = resolve_dataset_closure(root_candidate, storage_namespace=workspace)
        if not closure.valid:
            raise DatasetAccessError(f"scope {scope.name!r} cannot resolve dataset field {declaration.field!r}: {closure.status.value}: {closure.message}")
        if declaration.table is not DatasetTable.MAIN and not any(declaration.table.value in resource.members for resource in closure.resources):
            raise DatasetAccessError(f"scope {scope.name!r} dataset field {declaration.field!r} targets absent subtable {declaration.table.value}")
        resources = tuple(resource.path for resource in closure.resources)
        reason = (
            f"{fallback.value} access; {declaration.mode.value} whole dataset: {closure.root}"
            if fallback is not None
            else f"{declaration.mode.value} {_declaration_label(closure.root.name if closure.root is not None else declaration.field, declaration)}"
        )
        resolved.append(
            ResolvedDatasetAccess(
                field=declaration.field,
                declaration=declaration,
                requested_path=requested,
                root=closure.root,
                resources=resources,
                mode=declaration.mode,
                whole_dataset=declaration.columns is None,
                path_known=True,
                columns_known=declaration.columns is not None,
                fallback=fallback,
                closure_status=closure.status,
                reason=reason,
            )
        )
    return tuple(resolved)


@dataclass(frozen=True)
class AccessDecision:
    dependencies: frozenset[str]
    reasons: dict[str, tuple[str, ...]]
    datasets: tuple[ResolvedDatasetAccess, ...]


@dataclass
class _AccessState:
    last_writer: tuple[str, str] | None = None
    readers_since_write: dict[str, str] = dc_field(default_factory=dict)


class ResolvedAccessPlanner:
    """Incremental path+dataset hazard planner in an existing stable order."""

    def __init__(self, workspace: Path | None = None) -> None:
        self._workspace = (workspace or Path.cwd()).resolve()
        self._accesses: dict[Path, _AccessState] = {}
        self._planned_roots: dict[Path, tuple[Path, ...]] = {}

    def order_after(
        self,
        name: str,
        scope: Any,
        values: dict[str, Any],
        *,
        unresolved_inputs: set[str] | frozenset[str] = frozenset(),
        resolved_path_accesses: list[tuple[Path, bool]] | None = None,
    ) -> AccessDecision:
        from shinobi.steps.schema import path_accesses, paths_overlap

        datasets = resolve_scope_dataset_accesses(
            scope,
            values,
            workspace=self._workspace,
            planned_roots=self._planned_roots,
            unresolved_inputs=unresolved_inputs,
        )
        generic = resolved_path_accesses if resolved_path_accesses is not None else path_accesses(scope, values, workspace=self._workspace)
        for access in datasets:
            if access.mode is DatasetMode.READ and any(writes and any(paths_overlap(path, resource) for resource in access.resources) for path, writes in generic):
                raise DatasetAccessError(f"scope {scope.name!r} declares READ access for {access.field!r}, but its schema also declares a filesystem write to the same dataset")
        for access in datasets:
            if access.mode is DatasetMode.CREATE and access.root is not None:
                self._planned_roots[access.root] = access.resources
        declared_resources = {resource for access in datasets for resource in access.resources}
        accesses: list[tuple[Path, bool, str]] = []
        for access in datasets:
            label = _access_label(access)
            accesses.extend((resource, access.writes, label) for resource in access.resources)
        for path, writes in generic:
            if any(paths_overlap(path, resource) for resource in declared_resources):
                continue
            accesses.append((path, writes, str(path)))

        required: set[str] = set()
        reasons: dict[str, set[str]] = {}
        for path, writes, label in accesses:
            overlapping = [state for known, state in self._accesses.items() if paths_overlap(path, known)]
            for state in overlapping:
                if writes and state.readers_since_write:
                    for reader, previous_label in state.readers_since_write.items():
                        if reader != name:
                            required.add(reader)
                            reasons.setdefault(reader, set()).add(f"write-after-read: {label or previous_label}")
                elif state.last_writer is not None:
                    writer, previous_label = state.last_writer
                    if writer != name:
                        required.add(writer)
                        hazard = "write-after-write" if writes else "read-after-write"
                        reasons.setdefault(writer, set()).add(f"{hazard}: {label or previous_label}")

            state = self._accesses.setdefault(path, _AccessState())
            if writes:
                for item in [*overlapping, state]:
                    item.last_writer = (name, label)
                    item.readers_since_write = {}
            else:
                state.readers_since_write[name] = label

        return AccessDecision(
            dependencies=frozenset(required),
            reasons={parent: tuple(sorted(items)) for parent, items in sorted(reasons.items())},
            datasets=datasets,
        )


@dataclass(frozen=True)
class RecipeAccessPlan:
    """A graph augmented with deterministic access-hazard edges."""

    graph: Any
    accesses: dict[str, tuple[ResolvedDatasetAccess, ...]]
    reasons: dict[tuple[str, str], tuple[str, ...]]


def plan_recipe_accesses(
    recipe: Any,
    inputs: dict[str, Any] | BaseModel,
    *,
    workspace: Path | None = None,
    validated_steps: dict[int, tuple[BaseModel, bool]] | None = None,
    validated_inputs: BaseModel | None = None,
) -> RecipeAccessPlan:
    """Resolve dataset contracts and add backward hazard edges before dispatch.

    Explicit wiring/``after`` edges determine the stable topological order.
    Otherwise-unordered access conflicts are then oriented backwards in that
    order, so the result is deterministic and cannot invent data lineage.
    Nested recipes and dataset-bearing scatter are refused explicitly in this
    first bounded planner rather than partially planned.
    """

    from pydantic_core import PydanticUndefined

    from shinobi.graph import RecipeGraph, build_graph
    from shinobi.steps.schema import InputRef, OutputRef, Recipe, static_output_values

    if recipe.dataset_accesses:
        raise DatasetAccessError(f"recipe {recipe.name!r} declares dataset access metadata; attach access to the atomic steps that touch the dataset")
    graph = build_graph(recipe)
    root = (workspace or Path.cwd()).resolve()
    if validated_inputs is not None:
        if not isinstance(validated_inputs, recipe.inputs_model):
            raise TypeError(f"validated_inputs must be an instance of {recipe.inputs_model.__name__}")
        validated = validated_inputs
    elif isinstance(inputs, recipe.inputs_model):
        validated = inputs
    else:
        validated = recipe.inputs_model(**inputs)
    recipe_inputs = {name: getattr(validated, name) for name in recipe.inputs_model.model_fields}
    outputs: dict[str, dict[str, Any]] = {}
    planner = ResolvedAccessPlanner(root)
    dependencies = [set(items) for items in graph.deps]
    accesses: dict[str, tuple[ResolvedDatasetAccess, ...]] = {}
    reasons: dict[tuple[str, str], tuple[str, ...]] = {}
    rank = {index: order for order, index in enumerate(graph.topological_indices())}
    by_name = {name: index for index, name in enumerate(graph.names)}

    for index in graph.topological_indices():
        ref = recipe.steps[index]
        if isinstance(ref.step, Recipe):
            if scope_tree_has_dataset_contract(ref.step):
                raise DatasetAccessError(f"step {ref.name!r} is a nested recipe with dataset contracts; flatten it before access planning")
            continue
        if ref.scatter is not None and scope_has_dataset_contract(ref.step):
            raise DatasetAccessError(f"step {ref.name!r} scatters a dataset contract; declare bounded non-scattered steps instead")

        known = dict(ref.params)
        unresolved_fields: set[str] = set()
        for field_name, source in ref.wiring.items():
            sources = source if isinstance(source, list) else [source]
            found: list[Any] = []
            complete = True
            for item in sources:
                if isinstance(item, InputRef):
                    found.append(recipe_inputs[item.field])
                elif isinstance(item, OutputRef) and item.field in outputs.get(item.step, {}):
                    found.append(outputs[item.step][item.field])
                else:
                    complete = False
            if complete:
                known[field_name] = found if isinstance(source, list) else found[0]
            else:
                known.pop(field_name, None)
                unresolved_fields.add(field_name)
        for field_name, model_field in ref.step.inputs_model.model_fields.items():
            if field_name not in known and field_name not in unresolved_fields and model_field.default is not PydanticUndefined:
                known[field_name] = model_field.default
        snapshot = (validated_steps or {}).get(id(ref))
        if snapshot is not None and snapshot[1]:
            model = snapshot[0]
            known = {name: getattr(model, name) for name in ref.step.inputs_model.model_fields}
        else:
            try:
                model = ref.step.inputs_model(**known)
                known = {name: getattr(model, name) for name in ref.step.inputs_model.model_fields}
                for name in unresolved_fields:
                    known.pop(name, None)
            except Exception:
                pass

        outputs_for_step = static_output_values(ref.step, known, unresolved_inputs=unresolved_fields)
        decision = planner.order_after(ref.name, ref.step, {**known, **outputs_for_step}, unresolved_inputs=unresolved_fields)
        accesses[ref.name] = decision.datasets
        for parent in decision.dependencies:
            parent_index = by_name[parent]
            if rank[parent_index] >= rank[index]:
                raise DatasetAccessError(f"access ordering for {ref.name!r} would require non-backward dependency on {parent!r}")
            dependencies[index].add(parent_index)
            reasons[(ref.name, parent)] = decision.reasons[parent]
        outputs[ref.name] = outputs_for_step

    dependents = [set() for _ in graph.names]
    for child, parents in enumerate(dependencies):
        for parent in parents:
            dependents[parent].add(child)
    planned_graph = RecipeGraph(names=list(graph.names), deps=dependencies, dependents=dependents)
    return RecipeAccessPlan(graph=planned_graph, accesses=accesses, reasons=reasons)


def dataset_workspace_accesses(accesses: Iterable[ResolvedDatasetAccess]) -> list[tuple[Path, bool]]:
    """Flatten resolved closure identities for the ownership registry."""

    merged: dict[Path, bool] = {}
    for access in accesses:
        for resource in access.resources:
            merged[resource] = merged.get(resource, False) or access.writes
    return list(merged.items())


__all__ = [
    "AccessDecision",
    "DatasetAccess",
    "DatasetAccessError",
    "DatasetColumns",
    "DatasetFallback",
    "DatasetMode",
    "DatasetSelection",
    "DatasetTable",
    "RecipeAccessPlan",
    "ResolvedAccessPlanner",
    "ResolvedDatasetAccess",
    "dataset_workspace_accesses",
    "plan_recipe_accesses",
    "resolve_scope_dataset_accesses",
    "scope_tree_has_dataset_contract",
]
