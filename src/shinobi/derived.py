"""Finite file dependencies derived from one step's validated inputs.

These declarations name reads, not products or dataset states. The same finite
resolver feeds planning, ownership, mounts, cache identity and leaf staging.
"""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any, Literal

from pydantic import model_validator

from shinobi.products import Coordinate, DirectoryBundle, FamilyPlan, FamilySpec, ProductModel, address, family_plans


class DerivedRead(ProductModel):
    member: Literal["file", "directory"]
    family: FamilySpec

    @model_validator(mode="after")
    def _finite(self):
        if any(rule.captures for rule in self.family.rules):
            raise ValueError("derived reads require finite candidates; discovery captures are unsupported")
        return self

    def validate_inputs(self, fields):
        """Reject attribute/index traversal and unknown inputs at construction."""
        for rule in self.family.rules:
            if not (set(rule.when) | set(rule.when_set)) <= fields:
                raise ValueError("derived read condition names an unknown input")
            templates = [(self.family.root, fields), (rule.path, fields | set(rule.axes))]
            for axis in rule.axes.values():
                if any(key is not None and key not in fields for key in (axis.count_input, axis.values_input)):
                    raise ValueError("derived read axis names an unknown input")
            for template, allowed in templates:
                for _, key, fmt, conversion in Formatter().parse(template):
                    if key is not None and (key not in allowed or conversion or fmt):
                        raise ValueError(f"unsupported derived read placeholder {key!r}")


class DerivedAddress(ProductModel):
    name: str
    coordinates: dict[str, Coordinate]

    @property
    def field(self):
        payload = json.dumps([self.name, address(self.coordinates)], separators=(",", ":"), ensure_ascii=True)
        return "__derived." + hashlib.sha256(payload.encode()).hexdigest()

    @property
    def element_index(self):
        return None

    @property
    def label(self):
        return self.field


@dataclass(frozen=True)
class ResolvedDerivedRead:
    address: DerivedAddress
    path: Path
    lexical_path: Path
    member: str
    required: bool
    mutable: bool = False


def derived_reads(scope, inputs: dict[str, Any], workspace: Path, *, best_effort=False):
    if not scope.derived_reads:
        return ()
    resolved = []
    products = family_plans(scope, inputs, workspace, best_effort=True)
    for name, declaration in scope.derived_reads.items():
        try:
            plan = FamilyPlan(declaration.family, Path if declaration.member == "file" else DirectoryBundle, inputs, workspace)
        except (KeyError, TypeError, ValueError):
            if best_effort:
                continue
            raise
        for candidate in plan.candidates:
            matching = [one for output in products.values() for one in output.candidates if one.path == candidate.path and one.rule.accept_existing]
            resolved.append(
                ResolvedDerivedRead(
                    DerivedAddress(name=name, coordinates=candidate.coordinates),
                    candidate.path,
                    plan.root_location / candidate.path.relative_to(plan.root),
                    declaration.member,
                    candidate.rule.required,
                    bool(matching),
                )
            )
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if left.path == right.path or left.path.is_relative_to(right.path) or right.path.is_relative_to(left.path):
                raise ValueError("derived reads have duplicate or overlapping physical paths")
    # Resolve contradictions at the common boundary, before ownership,
    # stale clearing, output-parent preparation or sandbox staging.
    from shinobi.steps.schema import _resolved_output_path_values, iter_product_paths, path_fields, write_path_fields, product_pattern_issue, _resolved_product_patterns

    ordinary = [
        (path if path.is_absolute() else workspace / path).resolve()
        for _, value, complete in _resolved_output_path_values(scope, inputs)
        if complete and value is not None
        for path in iter_product_paths(value)
    ]
    for read in resolved:
        for path in ordinary:
            if path == read.path or path.is_relative_to(read.path) or read.path.is_relative_to(path):
                raise ValueError(f"declared output overlaps derived read: {path} / {read.path}")
        for output in products.values():
            for candidate in output.candidates:
                path = candidate.path
                if path == read.path and candidate.rule.accept_existing:
                    continue
                if path == read.path or path.is_relative_to(read.path) or read.path.is_relative_to(path):
                    raise ValueError(f"declared family output overlaps derived read: {path} / {read.path}")
        # Harvest patterns only select products after execution; they do not
        # grant writes to dependencies. Scratch does grant temporary writes.
        import fnmatch

        if not read.mutable:
            for pattern in scope.harvest:
                try:
                    expanded = pattern.format(**inputs)
                except (KeyError, ValueError, TypeError):
                    continue
                if not Path(expanded).is_absolute():
                    expanded = str(workspace / expanded)
                for parent in read.path.parents:
                    if parent == workspace or not parent.is_relative_to(workspace):
                        break
                    if fnmatch.fnmatchcase(str(parent), expanded):
                        raise ValueError(f"harvest ancestor would carry read-only staged dependency: {expanded} / {read.path}")
        if not read.mutable:
            for field in write_path_fields(scope) & path_fields(scope.inputs_model):
                value = inputs.get(field)
                for path in iter_product_paths(value):
                    target = (path if path.is_absolute() else workspace / path).resolve()
                    if target == read.path or target.is_relative_to(read.path):
                        raise ValueError(f"write_path destination overlaps derived read: {target} / {read.path}")
            for source, pattern in _resolved_product_patterns(scope, inputs, declarations=(("scratch", scope.scratch),), workspace=workspace):
                # _resolved_product_patterns also emits family candidates;
                # those were checked above, with explicit reuse preserved.
                if not source.startswith("scratch "):
                    continue
                issue = product_pattern_issue(pattern, workspace=workspace, resources={read.path}, resource_label="derived read")
                if issue:
                    raise ValueError(f"scratch writes overlap derived read: {pattern}: {issue}")
    return tuple(resolved)


def _check_path(path: Path):
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"derived read symlink is unsupported: {part}")


def read_fingerprint(read: ResolvedDerivedRead, *, require=False, content=False):
    """Framework physical metadata and kinds; optional absence stays explicit.

    Strict auxiliary identity requests byte hashing separately. Read-only
    dependencies retain the ordinary cache fingerprint coverage and cost.
    """
    _check_path(read.lexical_path)
    path = read.path
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        if require and read.required:
            raise ValueError(f"required derived read is absent: {path}")
        return None
    if (read.member == "file" and not stat.S_ISREG(mode)) or (read.member == "directory" and not stat.S_ISDIR(mode)):
        raise ValueError(f"derived read has wrong member kind: {path}")

    def file_hash(one):
        digest = hashlib.sha256()
        with one.open("rb") as stream:
            for data in iter(lambda: stream.read(1 << 20), b""):
                digest.update(data)
        return digest.hexdigest()

    from shinobi.cache import _walk_fingerprint

    if read.member == "file":
        return ["file", file_hash(path) if content else _walk_fingerprint(path)]
    result = []
    for child in sorted(path.rglob("*")):
        mode = child.lstat().st_mode
        if stat.S_ISDIR(mode):
            result.append([str(child.relative_to(path)), "directory"])
        elif stat.S_ISREG(mode):
            result.append([str(child.relative_to(path)), "file"])
        else:
            raise ValueError(f"derived directory contains a symlink or unsupported member: {child}")
    return ["directory", result, _walk_fingerprint(path)]


def regular_file_identity(path: Path):
    from shinobi.snapshots import StateIdentity

    _check_path(path)
    if path.stat().st_nlink != 1:
        raise ValueError(f"strict regular-file identity refuses hardlinked paths: {path}")
    read = ResolvedDerivedRead(DerivedAddress(name="identity", coordinates={}), path, path, "file", True)
    fingerprint = read_fingerprint(read, require=True, content=True)
    return StateIdentity("regular-file/v1", fingerprint[1])


def profile_identity(path: Path, profile: str | None = None):
    if profile == "regular-file/v1":
        return regular_file_identity(path)
    if profile not in (None, "msv2"):
        raise ValueError(f"unknown strict identity profile {profile!r}")
    from shinobi.dataset_lifecycle import dataset_identity

    return dataset_identity(path, path.parent)


def stage_derived_reads(scope, inputs, workspace, sandbox):
    """Stage only finite relative dependencies, keeping native absolute paths."""
    from shinobi.clonefs import CloneTier, clone_tree

    sources = {read.address.field: read for read in derived_reads(scope, inputs, workspace)}
    staged = {}
    for target in derived_reads(scope, inputs, sandbox):
        source = sources[target.address.field]
        fingerprint = read_fingerprint(source, require=True)
        if source.path == target.path or fingerprint is None:
            continue
        if not target.path.is_relative_to(sandbox):
            raise ValueError("derived read staging escapes its sandbox")
        target.path.parent.mkdir(parents=True, exist_ok=True)
        clone_tree(source.path, target.path, tier=CloneTier.COPY)
        if read_fingerprint(target, require=True) != fingerprint:
            raise ValueError("derived read staging does not preserve its physical identity")
        staged[target.path] = (source, target)
    return staged
