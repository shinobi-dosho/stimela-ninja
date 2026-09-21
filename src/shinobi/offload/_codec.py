"""Closed, data-only codec for the parameter models in a frozen bundle.

This is deliberately not a JSON-schema interpreter or a Python serializer.
Only the types and constraints listed here can cross the boundary. Unsupported
validators, factories and model hooks are refused, never silently dropped.
Loading a schema cannot import a caller-named class or execute source code.
"""

from __future__ import annotations

import dataclasses
import math
import types
from pathlib import Path
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import annotated_types
from pydantic import BaseModel, ConfigDict, Field, JsonValue, Strict, WithJsonSchema, create_model, model_validator
from pydantic_core import PydanticUndefined

from shinobi.dataset_access import DatasetAccess
from shinobi.datasets import DatasetType
from shinobi.steps.schema import Mutability, ParamMeta


class BundleError(ValueError):
    """A definition or persisted bundle cannot be represented faithfully."""


class WireModel(BaseModel):
    """Versioned protocol objects reject unknown fields rather than guessing."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def pack(value: Any) -> JsonValue:
    """Encode values without losing Path/tuple types or reserving dict keys."""
    if value is None or type(value) in (str, int, bool):
        return ["scalar", value]
    if type(value) is float and math.isfinite(value):
        return ["scalar", value]
    if isinstance(value, Path):
        return ["path", str(value)]
    if type(value) is ParamMeta:
        return ["param_meta", pack(value.model_dump(mode="python"))]
    if type(value) is Mutability:
        return ["mutability", value.value]
    if type(value) is DatasetAccess:
        return ["dataset_access", value.model_dump(mode="json")]
    if isinstance(value, BaseModel):
        return ["dict", pack_model(value)]
    if type(value) in (list, tuple):
        return ["tuple" if isinstance(value, tuple) else "list", [pack(v) for v in value]]
    if type(value) is dict and all(type(k) is str for k in value):
        return ["dict", {k: pack(v) for k, v in value.items()}]
    raise BundleError(f"cannot transport value of type {type(value).__name__}: use finite JSON values, Paths and tuples")


def unpack(value: JsonValue) -> Any:
    """Decode the closed value vocabulary; ordinary user dictionaries are data."""
    if not isinstance(value, list) or len(value) != 2:
        raise BundleError("invalid tagged bundle value")
    kind, data = value
    if kind == "scalar" and (data is None or type(data) in (str, int, bool) or type(data) is float and math.isfinite(data)):
        return data
    if kind == "path" and isinstance(data, str):
        return Path(data)
    if kind == "param_meta":
        return ParamMeta.model_validate(unpack(data))
    if kind == "mutability" and isinstance(data, str):
        return Mutability(data)
    if kind == "dataset_access" and isinstance(data, dict):
        return DatasetAccess.model_validate(data)
    if kind in ("list", "tuple") and isinstance(data, list):
        items = [unpack(v) for v in data]
        return tuple(items) if kind == "tuple" else items
    if kind == "dict" and isinstance(data, dict):
        return {k: unpack(v) for k, v in data.items()}
    raise BundleError(f"invalid bundle value kind or payload: {kind!r}")


def pack_model(model: BaseModel) -> dict[str, JsonValue]:
    """Transport parameter values, including fields excluded from reporting.

    Model schemas are transported separately. Never use JSON/reporting
    serializers here: strict types and concrete values under Any/unions
    must survive the worker boundary, and unsupported values must fail.
    """
    ModelSpec.capture(type(model))
    values = {name: getattr(model, name) for name in type(model).model_fields}
    values.update(model.model_extra or {})
    for name, value in values.items():
        field = type(model).model_fields.get(name)
        _check_model_values(value, field.annotation if field is not None else Any)
    return {name: pack(value) for name, value in values.items()}


def _check_model_values(value: Any, annotation: Any) -> None:
    """A model under Any has no declared class to reconstruct from its data."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        _check_model_values(value, args[0])
    elif annotation is Any or origin in (Union, types.UnionType) and Any in args:
        if _has_model_instance(value):
            raise BundleError("model values under Any require an explicit model annotation")
    elif origin is dict and isinstance(value, dict):
        for item in value.values():
            _check_model_values(item, args[1])
    elif origin in (list, tuple) and isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            element = args[0] if origin is list or args[-1] is Ellipsis else args[index]
            _check_model_values(item, element)


def _has_model_instance(value: Any) -> bool:
    """Default validation is opt-in, so models nested in defaults also fail."""
    if isinstance(value, BaseModel):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_model_instance(item) for item in value)
    if isinstance(value, dict):
        return any(_has_model_instance(item) for item in value.values())
    return False


_SCALARS = {"str": str, "int": int, "float": float, "bool": bool, "path": Path, "none": type(None), "any": Any}
_CONSTRAINTS = {name: getattr(annotated_types, name) for name in ("Gt", "Ge", "Lt", "Le", "MultipleOf", "MinLen", "MaxLen")}
_CONSTRAINTS["Strict"] = Strict
_FACTORIES = {"list": list, "dict": dict, "tuple": tuple}
# Only explicitly-set attributes are emitted. Unknown future Field options
# fail at freeze time, so upgrading pydantic cannot quietly weaken validation.
_FIELD_ATTRIBUTES = {
    "alias",
    "alias_priority",
    "validation_alias",
    "serialization_alias",
    "title",
    "description",
    "examples",
    "exclude",
    "frozen",
    "validate_default",
    "repr",
    "discriminator",
    "json_schema_extra",
}


class Constraint(WireModel):
    name: str
    attributes: dict[str, JsonValue]

    @classmethod
    def capture(cls, value: Any) -> Constraint:
        if type(value) is DatasetType:
            return cls(name="DatasetType", attributes={"kind": pack(value.kind.value), "profile": pack(value.profile)})
        if type(value) is WithJsonSchema:
            return cls(name="WithJsonSchema", attributes={"json_schema": pack(value.json_schema), "mode": pack(value.mode)})
        if type(value) not in _CONSTRAINTS.values():
            raise BundleError(f"unsupported field constraint {type(value).__name__}")
        return cls(name=type(value).__name__, attributes={k: pack(v) for k, v in dataclasses.asdict(value).items()})

    def restore(self) -> Any:
        if self.name == "DatasetType":
            return DatasetType(kind=unpack(self.attributes["kind"]), profile=unpack(self.attributes["profile"]))
        if self.name == "WithJsonSchema":
            return WithJsonSchema(json_schema=unpack(self.attributes["json_schema"]), mode=unpack(self.attributes["mode"]))
        constructor = _CONSTRAINTS.get(self.name)
        if constructor is None:
            raise BundleError(f"unknown field constraint {self.name!r}")
        return constructor(**{k: unpack(v) for k, v in self.attributes.items()})


class TypeSpec(WireModel):
    kind: Literal["str", "int", "float", "bool", "path", "none", "any", "list", "tuple", "dict", "union", "literal", "model", "annotated"]
    args: tuple[TypeSpec, ...] = ()
    values: tuple[JsonValue, ...] = ()
    variadic: bool = False
    model_spec: ModelSpec | None = None
    constraints: tuple[Constraint, ...] = ()

    @model_validator(mode="after")
    def _shape(self) -> TypeSpec:
        if self.kind != "model" and self.model_spec is not None or self.kind != "literal" and self.values or self.kind != "annotated" and self.constraints:
            raise BundleError("parameter type has attributes belonging to a different kind")
        if self.variadic and (self.kind != "tuple" or len(self.args) != 1):
            raise BundleError("only a homogeneous tuple can be variadic")
        if self.kind in (*_SCALARS, "literal", "model") and self.args:
            raise BundleError("atomic parameter type cannot have type arguments")
        return self

    @classmethod
    def capture(cls, annotation: Any, seen: tuple[type, ...] = ()) -> TypeSpec:
        for name, scalar in _SCALARS.items():
            if annotation is scalar:
                return cls(kind=name)
        origin, args = get_origin(annotation), get_args(annotation)
        if origin is Literal:
            if any(type(v) not in (str, int, bool, type(None)) for v in args):
                raise BundleError("only string, integer, boolean and null Literal choices are supported")
            return cls(kind="literal", values=tuple(pack(v) for v in args))
        if origin is Annotated:
            return cls(kind="annotated", args=(cls.capture(args[0], seen),), constraints=tuple(Constraint.capture(v) for v in args[1:]))
        names = {list: "list", tuple: "tuple", dict: "dict", Union: "union", types.UnionType: "union"}
        if origin in names:
            variadic = origin is tuple and len(args) == 2 and args[1] is Ellipsis
            return cls(kind=names[origin], args=tuple(cls.capture(a, seen) for a in (args[:1] if variadic else args)), variadic=variadic)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return cls(kind="model", model_spec=ModelSpec.capture(annotation, seen))
        raise BundleError(f"unsupported parameter annotation {annotation!r}")

    def restore(self) -> Any:
        if self.kind in _SCALARS:
            return _SCALARS[self.kind]
        args = tuple(a.restore() for a in self.args)
        if self.kind == "model" and self.model_spec is not None:
            return self.model_spec.restore()
        if self.kind == "literal" and self.values:
            return Literal[tuple(unpack(v) for v in self.values)]
        if self.kind == "list" and len(args) == 1:
            return list[args[0]]
        if self.kind == "dict" and len(args) == 2:
            return dict[args]
        if self.kind == "tuple" and args:
            return tuple[args[0], ...] if self.variadic and len(args) == 1 else tuple[args]
        if self.kind == "union" and len(args) >= 2:
            return Union[args]
        if self.kind == "annotated" and len(args) == 1 and self.constraints:
            return Annotated[args[0], *(c.restore() for c in self.constraints)]
        raise BundleError(f"malformed parameter type {self.kind!r}")


class FieldSpec(WireModel):
    name: str
    annotation: TypeSpec
    required: bool
    default: JsonValue = None
    factory: Literal["list", "dict", "tuple"] | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)
    constraints: tuple[Constraint, ...] = ()


class ModelSpec(WireModel):
    """A reconstructible parameter model, including validation and path types."""

    name: str
    fields: tuple[FieldSpec, ...]
    config: JsonValue

    @classmethod
    def capture(cls, model: type[BaseModel], seen: tuple[type, ...] = ()) -> ModelSpec:
        if model is BaseModel:
            raise BundleError("BaseModel itself cannot be instantiated; declare an empty subclass instead")
        if model in seen:
            raise BundleError(f"recursive parameter model {model.__name__!r} is unsupported")
        decorators = model.__pydantic_decorators__
        if any(
            getattr(decorators, n) for n in ("validators", "field_validators", "root_validators", "field_serializers", "model_serializers", "model_validators", "computed_fields")
        ):
            raise BundleError(f"model {model.__name__!r} has custom validators/serializers; cannot freeze it faithfully")
        if model.__pydantic_custom_init__ or model.__pydantic_post_init__ or model.__private_attributes__:
            raise BundleError(f"model {model.__name__!r} has custom initialization or private state")
        if model.__get_pydantic_core_schema__.__func__ is not BaseModel.__get_pydantic_core_schema__.__func__:
            raise BundleError(f"model {model.__name__!r} has a custom core schema")
        fields = []
        for name, field in model.model_fields.items():
            if _has_model_instance(field.default):
                raise BundleError(f"field {model.__name__}.{name}: model-instance defaults are unsupported")
            attrs = {k: v for k, v in field._attributes_set.items() if k not in ("annotation", "default", "default_factory")}
            unknown = attrs.keys() - _FIELD_ATTRIBUTES
            if unknown:
                raise BundleError(f"field {model.__name__}.{name} has unsupported options: {sorted(unknown)}")
            factory = None
            if field.default_factory is not None:
                factory = next((n for n, f in _FACTORIES.items() if f is field.default_factory), None)
                if factory is None:
                    raise BundleError(f"field {model.__name__}.{name} has an executable default factory")
            fields.append(
                FieldSpec(
                    name=name,
                    annotation=TypeSpec.capture(field.annotation, (*seen, model)),
                    required=field.is_required(),
                    default=None if field.default is PydanticUndefined else pack(field.default),
                    factory=factory,
                    attributes={k: pack(v) for k, v in attrs.items()},
                    constraints=tuple(Constraint.capture(m) for m in field.metadata),
                )
            )
        return cls(name=model.__name__, fields=tuple(fields), config=pack(dict(model.model_config)))

    def restore(self) -> type[BaseModel]:
        fields = {}
        for spec in self.fields:
            if spec.name in fields:
                raise BundleError(f"duplicate model field {spec.name!r}")
            unknown = spec.attributes.keys() - _FIELD_ATTRIBUTES
            if unknown:
                raise BundleError(f"unsupported persisted field attributes: {sorted(unknown)}")
            annotation = spec.annotation.restore()
            if spec.constraints:
                annotation = Annotated[annotation, *(c.restore() for c in spec.constraints)]
            attrs = {k: unpack(v) for k, v in spec.attributes.items()}
            if spec.factory is not None:
                attrs["default_factory"] = _FACTORIES[spec.factory]
            elif not spec.required:
                attrs["default"] = unpack(spec.default)
            fields[spec.name] = (annotation, Field(**attrs))
        return create_model(self.name, __config__=ConfigDict(**unpack(self.config)), **fields)


TypeSpec.model_rebuild()
