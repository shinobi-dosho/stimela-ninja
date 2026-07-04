"""Execution results.

`BackendRun` is the raw, schema-agnostic outcome of a backend actually
running a command: exit status and captured console output, nothing more.
Backends return this; they never see a cab's output schema.

`StepResult` is the schema-aware result the dispatch layer builds on top:
it carries the validated `outputs` and effective `inputs` models plus the
same console output, and is what a step call (cab or recipe) returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class BackendRun:
    """What a backend returns after running a command -- just the raw run
    outcome. Wrangling stdout/stderr into structured outputs and filling
    an `outputs_model` is the dispatch layer's job, not the backend's.
    """

    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def success(self) -> bool:
        return self.returncode == 0


@dataclass
class StepResult:
    """The outcome of running a step (a Cab or a Recipe).

    `outputs` is a validated instance of the step's `outputs_model`;
    `inputs` is a validated instance of the *effective* (post-override)
    inputs the step actually ran with. For a Recipe these aggregate from
    its sub-steps.
    """

    name: str
    returncode: int
    outputs: BaseModel
    inputs: BaseModel
    stdout: str = ""
    stderr: str = ""

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def __getattr__(self, name: str) -> Any:
        # convenience: result.<output_field> reads through to outputs
        try:
            return getattr(self.__dict__["outputs"], name)
        except (KeyError, AttributeError):
            raise AttributeError(
                f"StepResult for '{self.__dict__.get('name')}' has no attribute '{name}'"
            ) from None
