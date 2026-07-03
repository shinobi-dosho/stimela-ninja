"""Backend protocol for the new CabDef/RecipeDef step model.

Deliberately a new, minimal protocol -- not the existing
`shinobi.backends.Backend` ABC, which is tightly coupled to the old
`CabDef` shape (`.policies`, `.inputs: dict[str, ParamSchema]`,
`resolve_params`/`build_argv` from `shinobi.policies`). The new `CabDef`
has none of that, just an `inputs_model` pydantic class, so reusing the
old ABC would mean either bolting old-schema fields onto the new CabDef
(defeating the point of the simpler shape) or writing an adapter that
fakes an old CabDef from the new one. Neither is worth it for a
prototype; this is real but intentionally thin -- no Policies-equivalent
prefix/replace/list_sep/repeat_list richness yet.
"""

from __future__ import annotations

import subprocess
from typing import Any, Protocol

from pydantic import BaseModel

from shinobi.steps.schema import CabDef


class StepBackend(Protocol):
    def run(self, defn: CabDef, inputs: BaseModel) -> dict[str, Any]: ...


class NativeStepBackend:
    """Runs a CabDef's command directly on the host, via subprocess."""

    def run(self, defn: CabDef, inputs: BaseModel) -> dict[str, Any]:
        argv = [defn.command]
        for field in type(inputs).model_fields:
            value = getattr(inputs, field)
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    argv.append(f"--{field}")
                continue
            argv.append(f"--{field}")
            argv.append(str(value))

        proc = subprocess.run(argv, capture_output=True, text=True)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


class RecordingStepBackend:
    """Records (defn, inputs) instead of executing -- the new-model
    analogue of shinobi.backends.trace.TraceBackend, for tests.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[CabDef, BaseModel]] = []

    def run(self, defn: CabDef, inputs: BaseModel) -> dict[str, Any]:
        self.calls.append((defn, inputs))
        return {}
