"""Define cabs directly in Python, as an alternative to loading YAML.

    from shinobi.decorators import cab
    from shinobi.schema import ParamSchema

    @cab(
        "breizorro",
        image="breizorro:latest",
        outputs={"mask": ParamSchema(dtype="File", nom_de_guerre="outfile", required=True)},
    )
    def breizorro(restored_image: str, threshold: float = 6.5, dilate: int = 0):
        '''Mask creation and manipulation for radio astronomy images.'''

``breizorro`` is now a CabDef -- the same object shinobi.loaders.cultcargo
produces from YAML -- ready to pass to shinobi.recipe.call(). The
function's signature is read once, at decoration time, to build the input
schema (name, type hint -> dtype, presence of a default -> required), so
inputs aren't redeclared twice; the function itself is never called for a
binary-flavour cab; its docstring becomes the cab's `info`.

Per-parameter detail a bare signature can't express (info text, a
nom_de_guerre, an implicit value, ...) can be layered on top via the
`inputs=` kwarg: entries there replace the auto-derived ParamSchema for
that name outright, rather than being merged field-by-field.

@recipe is the recipe-side counterpart, for a plain Python function that
*is* a pipeline (like examples/ninja_recipe.py's logic could be, wrapped
in a function). Unlike @cab, it does not replace the function -- a
recipe's body is the orchestration itself and must stay directly callable
exactly as if undecorated. It only attaches a RecipeInfo (derived the same
way @cab derives a CabDef's inputs) as `func.__shinobi_recipe__`, so
tooling such as the `ninja run` CLI can build --options from a recipe's
signature without calling it.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, get_args, get_origin

from shinobi.schema import CabDef, ParamSchema, Policies, RecipeInfo

_DTYPE_NAMES: dict[Any, str] = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
}


def _dtype_from_annotation(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "str"

    origin = get_origin(annotation)
    if origin in (list, tuple):
        args = get_args(annotation)
        item_dtype = _dtype_from_annotation(args[0]) if args else "str"
        return f"list:{item_dtype}"

    if annotation in _DTYPE_NAMES:
        return _DTYPE_NAMES[annotation]

    # anything else (Path, a custom MS/File marker type, ...) -- use its
    # name as the dtype string, same convention cult-cargo YAML uses
    return getattr(annotation, "__name__", str(annotation))


def _inputs_from_signature(func: Callable) -> dict[str, ParamSchema]:
    inputs: dict[str, ParamSchema] = {}
    for name, param in inspect.signature(func).parameters.items():
        has_default = param.default is not inspect.Parameter.empty
        inputs[name] = ParamSchema(
            dtype=_dtype_from_annotation(param.annotation),
            required=not has_default,
            default=param.default if has_default else None,
        )
    return inputs


def cab(
    command: str,
    *,
    name: str | None = None,
    image: str | None = None,
    flavour: str = "binary",
    policies: Policies | None = None,
    inputs: dict[str, ParamSchema] | None = None,
    outputs: dict[str, ParamSchema] | None = None,
    wranglers: dict[str, list[str]] | None = None,
) -> Callable[[Callable], CabDef]:
    """Decorate a function to produce a CabDef. See module docstring."""

    def decorator(func: Callable) -> CabDef:
        derived = _inputs_from_signature(func)
        derived.update(inputs or {})
        return CabDef(
            name=name or func.__name__,
            command=command,
            info=inspect.getdoc(func),
            image=image,
            flavour=flavour,
            policies=policies or Policies(),
            inputs=derived,
            outputs=outputs or {},
            wranglers=wranglers or {},
        )

    return decorator


def recipe(
    *, name: str | None = None, inputs: dict[str, ParamSchema] | None = None
) -> Callable[[Callable], Callable]:
    """Attach schema metadata to a plain function, without replacing it.
    See module docstring.
    """

    def decorator(func: Callable) -> Callable:
        derived = _inputs_from_signature(func)
        derived.update(inputs or {})
        func.__shinobi_recipe__ = RecipeInfo(
            name=name or func.__name__,
            info=inspect.getdoc(func),
            inputs=derived,
        )
        return func

    return decorator
