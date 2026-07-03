"""Backend abstraction: a backend takes a cab and a resolved argv and runs
it somewhere -- natively, in a container, on Slurm, on Kubernetes, ...

A backend knows nothing about recipes or schemas beyond the argv it's
handed and the cab's ``image``/``command`` metadata; it only knows how to
execute and how to capture output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from shinobi.results import Result
from shinobi.schema import CabDef


class Backend(ABC):
    name: str

    @abstractmethod
    def run(self, cab: CabDef, argv: list[str], params: dict[str, Any]) -> Result:
        """Execute argv (as built by shinobi.policies.build_argv) and
        return a Result. Must not raise on a non-zero exit -- that's
        reported via Result.returncode / Result.success.

        ``params`` is the fully resolved (defaults/implicit applied)
        parameter dict argv was built from. Most backends ignore it, but
        container backends need it to know which File/MS-valued params
        have to be bind-mounted.
        """


_REGISTRY: dict[str, type[Backend]] = {}


def register(backend_cls: type[Backend]) -> type[Backend]:
    _REGISTRY[backend_cls.name] = backend_cls
    return backend_cls


def get_backend(name: str, **opts) -> Backend:
    try:
        backend_cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown backend '{name}' (available: {sorted(_REGISTRY)})"
        ) from None
    return backend_cls(**opts)


def registered_backend_classes() -> list[type[Backend]]:
    """Every concrete Backend subclass currently registered. Used by
    `ninja run --dryrun` to patch *all* of them (regardless of which one
    a recipe ends up constructing) so a recipe's own internal
    get_backend()/call() usage is traced instead of actually executed --
    see shinobi.cli for why patching by class, not by function, is the
    only way to do that reliably.
    """
    return list(_REGISTRY.values())


# Import submodules for their @register side effects, so get_backend() finds
# every built-in backend without the caller having to import that specific
# backend module first.
from shinobi.backends import container as _container  # noqa: E402,F401
from shinobi.backends import kubernetes as _kubernetes  # noqa: E402,F401
from shinobi.backends import native as _native  # noqa: E402,F401
from shinobi.backends import slurm as _slurm  # noqa: E402,F401
