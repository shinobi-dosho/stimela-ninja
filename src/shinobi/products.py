"""Closed declarations and observed coordinate-labelled filesystem products.

Plans reserve candidate names. Only successful execution evidence constructs a
resolved family; a plan is never a scientific result.
"""

from __future__ import annotations

import itertools
import glob as glob_module
import re
import stat
from pathlib import Path
from string import Formatter
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator, model_validator

Coordinate = StrictInt | StrictStr
T = TypeVar("T")


class ProductModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def address(coordinates: dict[str, Coordinate]) -> tuple:
    return tuple((key, type(value).__name__, value) for key, value in sorted(coordinates.items()))


def local_path(value: Any) -> Path:
    text = str(value)
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:/", text) or "::" in text:
        raise ValueError(f"remote stores and store-relative addresses are unsupported: {text!r}")
    return value if isinstance(value, Path) else Path(value)


class DirectoryBundle(ProductModel):
    """One owned local directory; its recursive inventory is execution evidence."""

    path: Path

    @field_validator("path", mode="before")
    @classmethod
    def _local(cls, value):
        return local_path(value)


class ProductMember(ProductModel, Generic[T]):
    coordinates: dict[str, Coordinate]
    value: T


class ProductFamily(ProductModel, Generic[T]):
    resolved: bool = False
    members: tuple[ProductMember[T], ...] = ()

    @model_validator(mode="after")
    def _members(self):
        if not self.resolved and self.members:
            raise ValueError("unresolved families cannot contain members")
        addresses, physical = set(), set()
        for member in self.members:
            key = address(member.coordinates)
            if key in addresses:
                raise ValueError("duplicate coordinate address")
            addresses.add(key)
            path = next(iter_product_paths(member.value), None)
            if path is None:
                raise ValueError("family members must be File or DirectoryBundle")
            path = local_path(path).resolve()
            if path in physical:
                raise ValueError(f"duplicate physical product: {path}")
            physical.add(path)
        return self

    def select(self, **coordinates) -> T:
        if not self.resolved:
            raise ValueError("family membership is unresolved")
        known: dict[str, set[type]] = {}
        for member in self.members:
            for name, value in member.coordinates.items():
                known.setdefault(name, set()).add(type(value))
        for name, value in coordinates.items():
            if type(value) not in (int, str):
                raise ValueError(f"coordinate {name!r} requires int or str")
            if known and name not in known:
                raise ValueError(f"unknown coordinate {name!r}")
            if name in known and type(value) not in known[name]:
                raise ValueError(f"wrong type for coordinate {name!r}")
        matches = [
            member
            for member in self.members
            if all(name in member.coordinates and type(member.coordinates[name]) is type(value) and member.coordinates[name] == value for name, value in coordinates.items())
        ]
        if len(matches) != 1:
            raise ValueError(f"coordinate selection is {'unavailable' if not matches else 'ambiguous'}: {coordinates!r}")
        return matches[0].value


def framework_type(annotation):
    """Recognize exact framework types, never subclasses with foreign hooks."""
    if annotation is DirectoryBundle:
        return "bundle", None
    metadata = getattr(annotation, "__pydantic_generic_metadata__", {})
    origin, args = metadata.get("origin"), metadata.get("args", ())
    if origin in (ProductFamily, ProductMember) and len(args) == 1 and args[0] in (Path, DirectoryBundle) and annotation is origin[args[0]]:
        return ("family" if origin is ProductFamily else "member"), args[0]
    return None


def iter_product_paths(value):
    """Walk only the supported direct path containers and framework values."""
    if isinstance(value, (Path, str)):
        yield local_path(value)
    elif type(value) is DirectoryBundle:
        yield value.path
    elif framework_type(type(value)):
        if isinstance(value, ProductFamily):
            for member in value.members:
                yield from iter_product_paths(member.value)
        elif isinstance(value, ProductMember):
            yield from iter_product_paths(value.value)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from iter_product_paths(item)


def map_product_paths(value, transform):
    if isinstance(value, (Path, str)):
        return transform(local_path(value))
    if type(value) is DirectoryBundle:
        return value.model_copy(update={"path": transform(value.path)})
    if framework_type(type(value)):
        if isinstance(value, ProductFamily):
            return value.model_copy(update={"members": tuple(map_product_paths(member, transform) for member in value.members)})
        return value.model_copy(update={"value": map_product_paths(value.value, transform)})
    if isinstance(value, (list, tuple, set, frozenset)):
        return type(value)(map_product_paths(item, transform) for item in value)
    return value


def has_product_value(value):
    """Recognize trusted framework values, including supported containers."""
    if framework_type(type(value)):
        return True
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(has_product_value(item) for item in value)
    return False


def product_argv_value(value):
    """Unwrap framework paths while preserving ordinary list token policies."""
    kind = framework_type(type(value))
    if kind:
        if kind[0] == "family":
            if not value.resolved:
                raise ValueError("argv input family is unresolved")
            return list(iter_product_paths(value))
        return next(iter_product_paths(value))
    if isinstance(value, (list, tuple, set, frozenset)):
        items = []
        for item in value:
            converted = product_argv_value(item)
            kind = framework_type(type(item))
            if kind and kind[0] == "family":
                items.extend(converted)
            else:
                items.append(converted)
        return type(value)(items)
    return value


def product_cache_value(value):
    """Canonical data for trusted framework values and their path containers."""
    if framework_type(type(value)):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [product_cache_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((product_cache_value(item) for item in value), key=repr)
    return value


class AggregateSpec(ProductModel):
    value: Coordinate
    suffix: str
    only_multiple: bool = False


class AxisSpec(ProductModel):
    count_input: str | None = None
    values_input: str | None = None
    values: tuple[Coordinate, ...] | None = None
    tokens: tuple[str, ...] | None = None
    uppercase: bool = False
    split: Literal["csv_or_compact"] | None = None
    include: tuple[Coordinate, ...] | None = None
    suffix: str = "{value}"
    singleton_suffix: str | None = None
    aggregate: AggregateSpec | None = None

    @model_validator(mode="after")
    def _source(self):
        if sum(value is not None for value in (self.count_input, self.values_input, self.values)) != 1:
            raise ValueError("axis requires exactly one of count_input, values_input, values")
        if self.tokens is not None and (not self.tokens or any(not token for token in self.tokens) or len(set(self.tokens)) != len(self.tokens)):
            raise ValueError("tokens must be unique nonempty strings")
        return self

    def expand(self, inputs, limit):
        if self.count_input:
            count = inputs[self.count_input]
            if type(count) is not int or count < 0:
                raise ValueError("axis counts must be nonnegative integers")
            if count > limit:
                raise ValueError("family member limit exceeded")
            values = list(range(count))
        else:
            values = inputs[self.values_input] if self.values_input else self.values
            if isinstance(values, str):
                text = values.upper() if self.uppercase else values
                if len(text) > 128:
                    raise ValueError("axis token string is too long")
                if self.split and "," not in text:
                    if not self.tokens:
                        raise ValueError("compact tokenization requires tokens")

                    # Each suffix is solved once. Keep at most two parses:
                    # distinguishing unique from ambiguous is the only question.
                    suffixes = [() for _ in range(len(text) + 1)]
                    suffixes[-1] = ((),)
                    for offset in range(len(text) - 1, -1, -1):
                        solutions = []
                        for token in self.tokens:
                            if text.startswith(token, offset):
                                for tail in suffixes[offset + len(token)]:
                                    solutions.append((token, *tail))
                                    if len(solutions) == 2:
                                        break
                            if len(solutions) == 2:
                                break
                        suffixes[offset] = tuple(solutions)
                    parsed = suffixes[0]
                    if len(parsed) != 1:
                        raise ValueError(f"ambiguous or unknown compact tokens: {text!r}")
                    values = parsed[0]
                else:
                    values = text.split(",") if self.split else [text]
            if not isinstance(values, (list, tuple)):
                raise ValueError("axis values must be a finite sequence")
            values = [value.upper() if self.uppercase and isinstance(value, str) else value for value in values]
            if any(type(value) not in (str, int) for value in values):
                raise ValueError("axis values require int or str")
            if self.tokens is not None and any(value not in self.tokens for value in values):
                raise ValueError("unknown axis token")
        if len(values) > limit or len({(type(value), value) for value in values}) != len(values):
            raise ValueError("excessive or duplicate axis values")
        singleton = len(values) == 1
        expanded = [
            (value, self.singleton_suffix if singleton and self.singleton_suffix is not None else self.suffix.format(index=index, value=value))
            for index, value in enumerate(values)
            if self.include is None or any(type(value) is type(allowed) and value == allowed for allowed in self.include)
        ]
        if self.aggregate and (not self.aggregate.only_multiple or len(values) > 1):
            expanded.append((self.aggregate.value, self.aggregate.suffix))
        return expanded


class MemberRule(ProductModel):
    path: str
    coordinates: dict[str, Coordinate] = Field(default_factory=dict)
    axes: dict[str, AxisSpec] = Field(default_factory=dict)
    captures: dict[str, Literal["int", "str"]] = Field(default_factory=dict)
    when: dict[str, tuple[str | int | bool, ...]] = Field(default_factory=dict)
    required: bool = False
    accept_existing: bool = False
    min_members: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _shape(self):
        if self.axes and self.captures:
            raise ValueError("bounded axes and discovery captures require separate rules")
        if self.captures and self.accept_existing:
            raise ValueError("discovery cannot accept unchanged existing matches")
        if (set(self.axes) | set(self.captures)) & set(self.coordinates):
            raise ValueError("constant and variable coordinates overlap")
        return self


class FamilySpec(ProductModel):
    root: str = "."
    coordinates: dict[str, Literal["int", "str"] | tuple[Literal["int", "str"], ...]]
    rules: tuple[MemberRule, ...]
    min_members: int = Field(default=0, ge=0)
    max_members: int = Field(default=10000, ge=1, le=100000)

    @model_validator(mode="after")
    def _schema(self):
        for rule in self.rules:
            used = set(rule.coordinates) | set(rule.axes) | set(rule.captures)
            if not used <= set(self.coordinates):
                raise ValueError("rule uses undeclared coordinates")
            for key, value in rule.coordinates.items():
                self.validate_coordinate(key, value)
        return self

    def validate_coordinate(self, key, value):
        kinds = self.coordinates[key]
        if isinstance(kinds, str):
            kinds = (kinds,)
        if type(value) not in (int, str) or type(value).__name__ not in kinds:
            raise ValueError(f"wrong type for coordinate {key!r}")


class Candidate(ProductModel):
    path: Path
    coordinates: dict[str, Coordinate]
    rule: MemberRule


class FamilyPlan:
    """Private finite candidates and anchored capture templates for one invocation."""

    def __init__(self, spec: FamilySpec, member_type, inputs, cwd: Path):
        self.spec, self.member_type = spec, member_type
        root = local_path(spec.root.format(**inputs))
        self.root_location = root if root.is_absolute() else cwd / root
        self.root = self.root_location.resolve()
        self.candidates, self.matchers, self.active_rules = [], [], []
        for rule in spec.rules:
            if any(not any(type(inputs[name]) is type(value) and inputs[name] == value for value in allowed) for name, allowed in rule.when.items()):
                continue
            self.active_rules.append(rule)
            if rule.captures:
                regex, glob = "", ""
                for literal, name, fmt, conversion in Formatter().parse(rule.path):
                    regex += re.escape(literal)
                    glob += glob_module.escape(literal)
                    if name is None:
                        continue
                    if name in rule.captures:
                        if fmt or conversion:
                            raise ValueError("capture format/conversion is unsupported")
                        regex += f"(?P<{name}>[0-9]+)" if rule.captures[name] == "int" else f"(?P<{name}>[A-Za-z0-9_-]{{1,128}})"
                        glob += "*"
                    else:
                        text = ("{" + name + (":" + fmt if fmt else "") + "}").format(**inputs)
                        local_path(text)
                        regex += re.escape(text)
                        glob += glob_module.escape(text)
                if "*" in str(Path(glob).parent):
                    raise ValueError("discovery captures in directory components are unsupported")
                self._contain(local_path(glob))
                self.matchers.append((re.compile("^" + regex + "$"), glob, rule))
                continue
            axes = [(name, axis.expand(inputs, spec.max_members)) for name, axis in rule.axes.items()]
            size = 1
            for _, values in axes:
                size *= len(values)
            if size + len(self.candidates) > spec.max_members:
                raise ValueError("family member limit exceeded")
            for combination in itertools.product(*(values for _, values in axes)):
                coordinates = dict(rule.coordinates)
                substitutions = dict(inputs)
                for (name, _), (value, suffix) in zip(axes, combination):
                    coordinates[name], substitutions[name] = value, suffix
                for key, value in coordinates.items():
                    spec.validate_coordinate(key, value)
                path = self._contain(local_path(rule.path.format(**substitutions)))
                self.candidates.append(Candidate(path=path, coordinates=coordinates, rule=rule))
        addresses, paths = set(), set()
        for candidate in self.candidates:
            key = address(candidate.coordinates)
            if key in addresses or candidate.path in paths:
                raise ValueError("overlapping family rules produce duplicate addresses or paths")
            addresses.add(key)
            paths.add(candidate.path)

    def validate_root(self):
        """A run cannot redirect the declaration's originally frozen boundary."""
        if self.root_location.resolve() != self.root:
            raise ValueError(f"family root changed its canonical destination: {self.root_location}")

    def _contain(self, path):
        self.validate_root()
        if ".." in path.parts:
            raise ValueError("family member cannot contain ..")
        actual = path if path.is_absolute() else self.root / path
        if not actual.resolve().is_relative_to(self.root):
            raise ValueError(f"family member escapes root: {path}")
        return actual

    def discovered(self):
        self.validate_root()
        count = 0
        for regex, glob, rule in self.matchers:
            for path in self.root.glob(glob):
                self._contain(path)
                match = regex.fullmatch(path.relative_to(self.root).as_posix())
                if match:
                    count += 1
                    if count + len(self.candidates) > self.spec.max_members:
                        raise ValueError("family member limit exceeded")
                    coordinates = dict(rule.coordinates)
                    coordinates.update({key: int(value) if rule.captures[key] == "int" else value for key, value in match.groupdict().items()})
                    for key, value in coordinates.items():
                        self.spec.validate_coordinate(key, value)
                    yield Candidate(path=path, coordinates=coordinates, rule=rule)

    def validate_table(self, family, workspace):
        """Validate saved membership against declarations, without discovering files."""
        if not family.resolved or not self.spec.min_members <= len(family.members) <= self.spec.max_members:
            raise ValueError("invalid cached family cardinality or resolution")
        accepted, counts = set(), {}
        for member in family.members:
            path = next(iter_product_paths(member.value))
            path = self._contain(path if path.is_absolute() else workspace / path)
            if path.is_symlink() or not (path.is_dir() if self.member_type is DirectoryBundle else path.is_file()):
                raise ValueError("cached member has wrong filesystem kind")
            for name, value in member.coordinates.items():
                if name not in self.spec.coordinates:
                    raise ValueError("cached member has unknown coordinate")
                self.spec.validate_coordinate(name, value)
            rule = None
            for candidate in self.candidates:
                if candidate.path.resolve() == path.resolve() and address(candidate.coordinates) == address(member.coordinates):
                    rule = candidate.rule
                    accepted.add((candidate.path.resolve(), address(candidate.coordinates)))
                    break
            if rule is None:
                relative = path.relative_to(self.root).as_posix()
                for regex, _glob, candidate_rule in self.matchers:
                    match = regex.fullmatch(relative)
                    if match:
                        expected = dict(candidate_rule.coordinates)
                        expected.update({name: int(value) if candidate_rule.captures[name] == "int" else value for name, value in match.groupdict().items()})
                        if address(expected) == address(member.coordinates):
                            rule = candidate_rule
                            break
            if rule is None:
                raise ValueError("cached member disagrees with family declaration")
            counts[id(rule)] = counts.get(id(rule), 0) + 1
        if any(candidate.rule.required and (candidate.path.resolve(), address(candidate.coordinates)) not in accepted for candidate in self.candidates):
            raise ValueError("cached required candidate is absent")
        if any(counts.get(id(rule), 0) < rule.min_members for rule in self.active_rules):
            raise ValueError("cached rule cardinality below minimum")

    def patterns(self):
        return [str(self.root / glob) for _, glob, _ in self.matchers]


def family_plans(scope, inputs, cwd, *, best_effort=False):
    plans = {}
    for name, meta in scope.field_meta.items():
        if meta.family is None:
            continue
        kind = family_annotation(scope.outputs_model.model_fields[name].annotation)
        try:
            plans[name] = FamilyPlan(meta.family, kind, inputs, cwd)
        except KeyError:
            if not best_effort:
                raise ValueError(f"{name}: unresolved family input") from None
    return plans


def family_annotation(annotation):
    from typing import get_args, get_origin, Annotated, Union
    import types

    if get_origin(annotation) is Annotated:
        return family_annotation(get_args(annotation)[0])
    if get_origin(annotation) in (Union, types.UnionType):
        types_ = [family_annotation(item) for item in get_args(annotation) if item is not type(None)]
        return types_[0] if len(types_) == 1 else None
    recognized = framework_type(annotation)
    return recognized[1] if recognized and recognized[0] == "family" else None


def bundle_inventory(root: Path):
    """Complete kind inventory, refusing symlinks, special nodes and failed reads."""
    entries = []

    def visit(path):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f"unsupported bundle node: {path}")
        kind = "directory" if stat.S_ISDIR(mode) else "file"
        entries.append({"path": path.relative_to(root).as_posix(), "kind": kind})
        if kind == "directory":
            for child in sorted(path.iterdir()):
                visit(child)

    if not root.is_dir():
        raise ValueError(f"bundle is not a directory: {root}")
    visit(root)
    return {"root": str(root.resolve()), "entries": entries}


def validate_bundle_inventory(inventory):
    try:
        root = Path(inventory["root"])
        if not root.is_absolute() or not inventory["entries"] or inventory["entries"][0] != {"path": ".", "kind": "directory"}:
            return False
        seen = set()
        for entry in inventory["entries"]:
            if set(entry) != {"path", "kind"} or entry["path"] in seen:
                return False
            seen.add(entry["path"])
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                return False
            path = root / relative
            mode = path.lstat().st_mode
            if entry["kind"] == "directory":
                if not stat.S_ISDIR(mode):
                    return False
            elif entry["kind"] == "file":
                if not stat.S_ISREG(mode):
                    return False
            else:
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def resolve_reference(value, selection):
    if selection is None:
        return value
    if not isinstance(value, ProductFamily):
        raise ValueError("coordinate selection requires a ProductFamily")
    return value.select(**selection)
