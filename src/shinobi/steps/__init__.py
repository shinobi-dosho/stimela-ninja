"""The step model: `Scope` (definition), `ExecContext` (execution state),
`Cab`/`Recipe` (atomic/composite), `StepRef` (binding), and the
`@shinobi.step` decorator. See the module docstrings and the design plan.
"""

from __future__ import annotations

from shinobi.steps.decorator import step
from shinobi.steps.dispatch import (
    ExecContext,
    get_step_backend,
    register_step_backend,
)
from shinobi.steps.pyfunc import pystep
from shinobi.steps.schema import (
    Cab,
    InputRef,
    LoopIteration,
    LoopRef,
    Mutability,
    OutputRef,
    ParamMeta,
    ParamPattern,
    ParamSegment,
    Policies,
    Recipe,
    ScatterSpec,
    Scope,
    StepRef,
    path_fields,
)
from shinobi.datasets import CasaTable, MeasurementSetV2
from shinobi.dataset_access import (
    DatasetAccess,
    DatasetColumns,
    DatasetFallback,
    DatasetMode,
    DatasetSelection,
    DatasetTable,
    ResolvedDatasetAccess,
)

__all__ = [
    "Cab",
    "CasaTable",
    "DatasetAccess",
    "DatasetColumns",
    "DatasetFallback",
    "DatasetMode",
    "DatasetSelection",
    "DatasetTable",
    "ExecContext",
    "InputRef",
    "LoopIteration",
    "LoopRef",
    "Mutability",
    "MeasurementSetV2",
    "OutputRef",
    "ParamMeta",
    "ParamPattern",
    "ParamSegment",
    "Policies",
    "Recipe",
    "ResolvedDatasetAccess",
    "ScatterSpec",
    "Scope",
    "StepRef",
    "get_step_backend",
    "path_fields",
    "pystep",
    "register_step_backend",
    "step",
]
