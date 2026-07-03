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
