class ShinobiError(Exception):
    """Base class for all shinobi errors."""


class ParameterError(ShinobiError):
    """A step or cab was called with invalid, missing, or unknown parameters,
    or its outputs could not be validated against the declared schema.
    """


class BackendError(ShinobiError):
    """A backend failed to run a cab."""


class CabRunError(ShinobiError):
    """A step's underlying command or function exited with a non-zero/failure status."""


class StepError(ShinobiError):
    """A step failed during execution for a reason not covered by a more
    specific exception class. The message carries the step/recipe path so the
    caller can tell which step failed and why.
    """


class CabLoadError(ShinobiError):
    """A cab definition file could not be loaded or resolved."""


class ConfigLoadError(ShinobiError):
    """A worker/config schema file could not be loaded or resolved."""


class ReplayError(ShinobiError):
    """A run manifest cannot be replayed against the current target."""


class UnsupportedFlavourError(ShinobiError):
    """A cab's flavour isn't one shinobi knows how to execute.

    Cabs with a non-"binary" flavour (cult-cargo's "python-code"/
    "casa-task"/etc.) have a `command` that is not an executable name --
    it may be inline Python/shell source, or a dotted reference to a
    function to import and call. shinobi never treats that as code to
    eval()/exec(): every backend shells out via subprocess with a list
    argv, so a non-executable `command` would otherwise just fail
    obscurely (subprocess trying to exec a multi-line string as a program
    path). This is raised deliberately, before that happens.
    """
