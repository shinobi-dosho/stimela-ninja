__version__ = "0.1.0b1"

import logging

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
