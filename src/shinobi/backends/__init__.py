"""Backend abstraction: a backend takes a cab and a resolved argv and runs
it somewhere -- natively, in a container, on Slurm, on Kubernetes, ...

A backend knows nothing about recipes or output schemas beyond the argv
it's handed and the cab's ``image``/``command`` metadata; it only knows
how to execute and how to capture output. Wrangling that output into
structured results is the dispatch layer's job, so a backend returns a
raw ``BackendRun`` (returncode/stdout/stderr), nothing schema-aware.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from shinobi.results import BackendRun
from shinobi.steps.schema import Cab


class Backend(ABC):
    name: str

    @abstractmethod
    def run(
        self,
        cab: Cab,
        argv: list[str],
        inputs: dict[str, Any],
        *,
        label: str = "",
        stream: bool = True,
    ) -> BackendRun:
        """Execute argv (as built by shinobi.policies.build_argv) and
        return a BackendRun. Must not raise on a non-zero exit -- that's
        reported via BackendRun.returncode / BackendRun.success.

        ``inputs`` is the *prepared* inputs dict argv was built from (the
        one `_prepare_inputs` produces, so MUTABLE fields are the caller's
        own objects by reference). Most backends ignore it, but container
        backends need it to know which File/MS-valued params to bind-mount.

        ``label``/``stream`` control live stdout/stderr echo (see
        `shinobi.backends._stream.run_streaming`) -- only `native` and
        `container` act on them today; `slurm`/`kubernetes` accept and
        ignore both (neither has any log-tailing infrastructure yet, so
        they keep reading output once after the job/pod finishes).
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
    return list(_REGISTRY.values())


# Import submodules for their @register side effects.
from shinobi.backends import container as _container  # noqa: E402,F401
from shinobi.backends import kubernetes as _kubernetes  # noqa: E402,F401
from shinobi.backends import native as _native  # noqa: E402,F401
from shinobi.backends import recording as _recording  # noqa: E402,F401
from shinobi.backends import slurm as _slurm  # noqa: E402,F401
