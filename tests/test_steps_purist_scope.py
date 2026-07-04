"""Regression tests for the manual "purist" bare-Scope step pattern
(build a Scope by hand, write a function that returns its own StepResult,
skip @shinobi.pystep's signature introspection entirely) -- documented on
Scope/StepRef in src/shinobi/steps/schema.py.
"""

import pytest
from pydantic import BaseModel

from shinobi.results import StepResult
from shinobi.steps import Scope, StepRef
from shinobi.steps.dispatch import _dispatch


class Inputs(BaseModel):
    x: int = 0


class Outputs(BaseModel):
    doubled: int = 0


def _manual_func(ctx):
    prepared = ctx.prepare_inputs()
    return StepResult(
        name=ctx.scope.name,
        returncode=0,
        outputs=Outputs(doubled=prepared["x"] * 2),
        inputs=ctx.inputs,
    )


def make_bare_scope() -> Scope:
    return Scope(name="manual", inputs_model=Inputs, outputs_model=Outputs)


def test_stepref_accepts_a_bare_scope_after_the_widening():
    ref = StepRef(name="manual", step=make_bare_scope(), func=_manual_func)
    assert isinstance(ref.step, Scope)


def test_manual_function_runs_and_never_calls_ctx_run():
    ref = StepRef(name="manual", step=make_bare_scope(), func=_manual_func)
    result = ref(x=21)
    assert result.success
    assert result.outputs.doubled == 42
    assert result.doubled == 42


def test_calling_ctx_run_on_a_bare_scope_raises_a_clear_typeerror():
    def calls_run(ctx):
        return ctx.run()

    with pytest.raises(TypeError, match="ctx.run"):
        _dispatch(make_bare_scope(), calls_run, x=1)
