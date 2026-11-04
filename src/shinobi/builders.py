"""Explicit, trusted Python construction of complete declared recipe graphs."""

from __future__ import annotations

from collections.abc import Callable
from functools import update_wrapper
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from shinobi._input_schema import validate_input_model, validate_scope_input_models
from shinobi.graph import build_graph
from shinobi.steps.schema import Recipe


_Inputs = TypeVar("_Inputs", bound=BaseModel)


class RecipeBuilder(Generic[_Inputs]):
    """A validated construction boundary, separate from runtime recipe inputs.

    Calling the builder validates its complete argument model, calls trusted
    Python once, then checks the returned recipe needs no caller inputs and
    has a valid full DAG. Relative paths retain the invocation cwd semantics.
    """

    def __init__(self, function: Callable[[_Inputs], Recipe], *, inputs_model: type[_Inputs]):
        if not isinstance(inputs_model, type) or not issubclass(inputs_model, BaseModel):
            raise TypeError("recipe builder inputs_model must be a Pydantic BaseModel subclass")
        validate_input_model(inputs_model)
        self.function = function
        self.inputs_model = inputs_model
        self.info = function.__doc__ or ""
        update_wrapper(self, function, updated=())

    def validate_inputs(self, values: dict[str, Any]) -> _Inputs:
        """Validate construction arguments without invoking the builder body."""
        validate_input_model(self.inputs_model)
        return self.inputs_model.model_validate(values)

    def build(self, inputs: _Inputs) -> Recipe:
        """Construct once from validated arguments and validate the complete DAG."""
        validate_input_model(self.inputs_model)
        if not isinstance(inputs, self.inputs_model):
            raise TypeError("builder inputs must be an instance of its inputs_model")
        recipe = self.function(inputs)
        if not isinstance(recipe, Recipe):
            raise TypeError("recipe builder must return a Recipe")
        validate_scope_input_models(recipe)
        required = [name for name, field in recipe.inputs_model.model_fields.items() if field.is_required()]
        if required:
            raise ValueError(f"recipe builder returned a Recipe requiring runtime inputs: {', '.join(required)}; bind these while constructing the graph")
        build_graph(recipe)
        return recipe

    def __call__(self, **kwargs: Any) -> Recipe:
        return self.build(self.validate_inputs(kwargs))


def recipe_builder(*, inputs_model: type[_Inputs]) -> Callable[[Callable[[_Inputs], Recipe]], RecipeBuilder[_Inputs]]:
    """Decorate ``function(validated_inputs) -> Recipe`` as a CLI target."""

    def decorate(function: Callable[[_Inputs], Recipe]) -> RecipeBuilder[_Inputs]:
        return RecipeBuilder(function, inputs_model=inputs_model)

    return decorate
