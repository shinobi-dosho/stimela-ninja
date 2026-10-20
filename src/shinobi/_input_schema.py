"""Reject Pydantic's import-resolving validator before validating run inputs.

This is a check on a supported schema facility, not a sandbox for arbitrary
Python validators. Inspect the compiled validation schema so aliases, nested
models, mapping keys, dataclasses, and other Pydantic containers share one
rule instead of maintaining another annotation walker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ImportString, TypeAdapter

if TYPE_CHECKING:
    from shinobi.steps.schema import Scope

# Obtain the validator through Pydantic's public schema API rather than
# importing its private implementation or comparing function names.
_IMPORT_STRING_VALIDATOR = TypeAdapter(ImportString).core_schema["function"]["function"]


def validate_input_model(model: type[BaseModel]) -> None:
    """Refuse input schemas that import a value supplied by the caller.

    Run before model instantiation: even an imported *non-callable* can
    execute a module's top-level code. Model rebuilding resolves trusted
    author annotations, but never validates the caller's input values.
    """
    # The bare BaseModel is also used as an empty definition placeholder;
    # unlike subclasses, Pydantic does not support rebuilding it.
    if model is BaseModel:
        return
    model.model_rebuild()
    pending: list[tuple[str, Any]] = [(model.__name__, model.__pydantic_core_schema__)]
    seen: set[int] = set()
    while pending:
        path, node = pending.pop()
        if not isinstance(node, (dict, list, tuple)) or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, dict):
            function = node.get("function")
            if isinstance(function, dict) and function.get("function") is _IMPORT_STRING_VALIDATOR:
                raise ValueError(
                    f"{path}: Pydantic ImportString is not allowed in workflow input schemas; bind implementations in recipe source and expose data or declared choices instead"
                )
            for key, child in node.items():
                if key in {"metadata", "serialization", "default"}:
                    continue
                if key == "fields" and isinstance(child, dict):
                    pending.extend((f"{path}.{name}", field) for name, field in child.items())
                elif key == "choices" and isinstance(child, dict):
                    # Discriminator tags are user data, not schema keys:
                    # tags such as "default" must not hit the filter above.
                    pending.extend((f"{path}[{tag!r}]", branch) for tag, branch in child.items())
                else:
                    pending.append((path, child))
        else:
            pending.extend((path, child) for child in node)


def validate_scope_input_models(scope: Scope) -> None:
    """Check a declared scope tree without instantiating its input models.

    Construction may have replaced a descendant model after its definition
    was checked. Inspect every descendant before planning or reusing any
    validated inputs, preserving the planner's single validation of values.
    """
    from shinobi.steps.schema import Recipe

    pending = [scope]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        validate_input_model(current.inputs_model)
        if isinstance(current, Recipe):
            pending.extend(ref.step for ref in current.steps)
