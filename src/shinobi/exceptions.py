class ShinobiError(Exception):
    """Base class for all shinobi errors."""


class ParameterError(ShinobiError):
    """A cab was called with invalid, missing, or unknown parameters."""


class BackendError(ShinobiError):
    """A backend failed to run a cab."""


class CabRunError(ShinobiError):
    """A cab's underlying command exited with a non-zero/failure status."""


class CabLoadError(ShinobiError):
    """A cab definition file could not be loaded or resolved."""


class ConfigLoadError(ShinobiError):
    """A worker/config schema file could not be loaded or resolved."""


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
