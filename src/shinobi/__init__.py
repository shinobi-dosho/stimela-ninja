import logging
from importlib.metadata import PackageNotFoundError, version as _dist_version

# Read the version off the installed distribution rather than restating it
# here. A hand-maintained literal is a second place to bump, and it silently
# rotted: it still said "0.1.0b1" three releases later, so `ninja version`
# lied and every run manifest recorded `shinobi_version: "0.1.0b1"` -- the
# one field whose whole job is telling you which shinobi produced a result.
# Derived, it cannot drift from pyproject.toml.
#
# The fallback only applies when `shinobi` is importable but `stimela-ninja`
# is not installed -- a source tree on PYTHONPATH. Nothing legitimate ships
# that way, so it is a marker, not a version anyone should see recorded.
try:
    __version__ = _dist_version("stimela-ninja")
except PackageNotFoundError:  # pragma: no cover -- uninstalled source tree
    __version__ = "0+unknown"

# Library convention: emit through the `shinobi.*` logger hierarchy but
# never print unless a handler is attached (the CLI attaches a file
# handler via shinobi.logsetup when AppConfig.log.file is set). The
# NullHandler also stops logging's last-resort stderr handler from
# echoing unhandled WARNING+ records in unconfigured runs.
logging.getLogger("shinobi").addHandler(logging.NullHandler())

from shinobi.steps import (  # noqa: E402
    Cab,
    CasaTable,
    DatasetAccess,
    DatasetColumns,
    DatasetFallback,
    DatasetMode,
    DatasetSelection,
    DatasetTable,
    ExecContext,
    InputRef,
    LoopIteration,
    LoopRef,
    Mutability,
    MeasurementSetV2,
    OutputRef,
    Recipe,
    ResolvedDatasetAccess,
    ScatterSpec,
    Scope,
    StepRef,
    pystep,
    step,
)
from shinobi.dataset_closure import DatasetClosure, resolve_dataset_closure  # noqa: E402
from shinobi.dataset_access import DatasetAccessError, RecipeAccessPlan, plan_recipe_accesses  # noqa: E402
from shinobi.dataset_lifecycle import (  # noqa: E402
    DATASET_MUTATION_CAPABILITY,
    DATASET_READ_CAPABILITY,
    DatasetCacheDecision,
    DatasetLeafAttempt,
    DatasetLifecycleAttempt,
    DatasetLifecyclePhase,
    DatasetLifecycleSnapshot,
    DatasetMutationOutcome,
    DatasetMutationRecord,
    DatasetOverwrite,
    dataset_attempt_path,
    read_dataset_attempt,
)
from shinobi.exceptions import DatasetLifecycleUnavailableError, DatasetLifecycleViolationError  # noqa: E402

__all__ = [
    "Cab",
    "CasaTable",
    "DatasetClosure",
    "DatasetAccess",
    "DatasetAccessError",
    "DatasetColumns",
    "DatasetCacheDecision",
    "DatasetFallback",
    "DatasetLeafAttempt",
    "DatasetLifecycleAttempt",
    "DatasetLifecyclePhase",
    "DatasetLifecycleSnapshot",
    "DatasetLifecycleUnavailableError",
    "DatasetLifecycleViolationError",
    "DatasetMode",
    "DatasetMutationOutcome",
    "DatasetMutationRecord",
    "DatasetOverwrite",
    "DatasetSelection",
    "DatasetTable",
    "ExecContext",
    "InputRef",
    "LoopIteration",
    "LoopRef",
    "Mutability",
    "MeasurementSetV2",
    "OutputRef",
    "Recipe",
    "RecipeAccessPlan",
    "ResolvedDatasetAccess",
    "ScatterSpec",
    "Scope",
    "StepRef",
    "DATASET_MUTATION_CAPABILITY",
    "DATASET_READ_CAPABILITY",
    "dataset_attempt_path",
    "pystep",
    "plan_recipe_accesses",
    "resolve_dataset_closure",
    "read_dataset_attempt",
    "step",
    "__version__",
]
