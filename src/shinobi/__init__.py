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
    ExecContext,
    InputRef,
    LoopIteration,
    LoopRef,
    Mutability,
    OutputRef,
    Recipe,
    ScatterSpec,
    Scope,
    StepRef,
    pystep,
    step,
)

__all__ = [
    "Cab",
    "ExecContext",
    "InputRef",
    "LoopIteration",
    "LoopRef",
    "Mutability",
    "OutputRef",
    "Recipe",
    "ScatterSpec",
    "Scope",
    "StepRef",
    "pystep",
    "step",
    "__version__",
]
