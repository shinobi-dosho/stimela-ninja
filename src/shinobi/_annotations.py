"""Shared structural traversal for Pydantic field annotations.

This module knows typing shapes, not what any annotation means.  Dataset
declarations and ordinary path fields both consume the same walk so support
for a container origin cannot drift between lifecycle guards and bind
discovery.
"""

from __future__ import annotations

import types
from collections.abc import Callable, Mapping, Sequence, Set
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Annotated, Any, ParamSpec, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel


@dataclass(frozen=True)
class AnnotationNode:
    """One unwrapped annotation encountered during a structural walk."""

    path: str
    annotation: Any
    metadata: tuple[Any, ...]
    leaf: bool


_discovery_cache: ContextVar[dict | None] = ContextVar("shinobi_annotation_discovery", default=None)
_Params = ParamSpec("_Params")
_Discovery = TypeVar("_Discovery", bound=dict | set)


@contextmanager
def annotation_cache():
    """Reuse schema discovery within one pass, never across revalidation."""
    if _discovery_cache.get() is not None:
        yield
        return
    token = _discovery_cache.set({})
    try:
        yield
    finally:
        _discovery_cache.reset(token)


def cached_model_discovery(function: Callable[_Params, _Discovery]) -> Callable[_Params, _Discovery]:
    """Cache a model's declaration mapping/set; give each caller its own copy."""

    @wraps(function)
    def cached(*args: _Params.args, **kwargs: _Params.kwargs) -> _Discovery:
        cache = _discovery_cache.get()
        if cache is None:
            return function(*args, **kwargs)
        key = (function, args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = function(*args, **kwargs)
        return cache[key].copy()

    return cached


def _origin_is(origin: Any, abstract: type) -> bool:
    return isinstance(origin, type) and issubclass(origin, abstract)


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def walk_annotation(
    annotation: Any,
    *,
    path: str = "",
    metadata: tuple[Any, ...] = (),
    descend_mappings: bool = True,
    descend_models: bool = True,
    _ancestors: frozenset[type[BaseModel]] = frozenset(),
):
    """Yield nodes in one annotation, recursively and with qualified paths.

    The finite supported container set is mappings (values only), concrete
    and abstract sequences, concrete and abstract sets, tuples, and unions.
    ``[]`` qualifies sequence/set/tuple items and ``.*`` mapping values.
    Nested Pydantic models are optional because a model-valued field is not
    itself a directly usable path value, while declaration discovery does
    need to inspect its fields.  An ancestor stack bounds recursive models.
    """

    combined_metadata = list(metadata)
    while get_origin(annotation) is Annotated:
        annotation, *annotated_metadata = get_args(annotation)
        combined_metadata.extend(annotated_metadata)

    origin = get_origin(annotation)
    args = get_args(annotation)
    children: list[tuple[Any, str, tuple[Any, ...], frozenset[type[BaseModel]]]] = []
    if origin is Union or origin is types.UnionType:
        children.extend((arg, path, (), _ancestors) for arg in args)
    elif descend_mappings and _origin_is(origin, Mapping):
        if len(args) == 2:
            children.append((args[1], f"{path}.*", (), _ancestors))
    elif origin is tuple:
        children.extend((arg, f"{path}[]", (), _ancestors) for arg in args if arg is not Ellipsis)
    elif _origin_is(origin, Sequence) or _origin_is(origin, Set):
        if args:
            children.append((args[0], f"{path}[]", (), _ancestors))
    elif descend_models and _is_model(annotation) and annotation not in _ancestors:
        nested_ancestors = _ancestors | {annotation}
        children.extend(
            (
                field.annotation,
                f"{path}.{name}" if path else name,
                tuple(field.metadata),
                nested_ancestors,
            )
            for name, field in annotation.model_fields.items()
        )

    yield AnnotationNode(path=path, annotation=annotation, metadata=tuple(combined_metadata), leaf=not children)
    for child, child_path, child_metadata, child_ancestors in children:
        yield from walk_annotation(
            child,
            path=child_path,
            metadata=child_metadata,
            descend_mappings=descend_mappings,
            descend_models=descend_models,
            _ancestors=child_ancestors,
        )


def walk_model_annotations(model: type[BaseModel]):
    """Yield the fully qualified annotation walk for every model field."""

    ancestors = frozenset({model})
    for name, field in model.model_fields.items():
        yield from walk_annotation(field.annotation, path=name, metadata=tuple(field.metadata), _ancestors=ancestors)


__all__ = ["AnnotationNode", "annotation_cache", "cached_model_discovery", "walk_annotation", "walk_model_annotations"]
