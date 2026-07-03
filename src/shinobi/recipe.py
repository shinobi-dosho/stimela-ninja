"""The recipe layer: recipes are plain Python. There is no separate
recipe/step schema, expression language, or alias-propagation system --
calling a cab is just a function call, and threading one step's output
into the next step's input is just passing a Python value.

    from shinobi.recipe import call
    from shinobi.backends import get_backend

    backend = get_backend("native")

    image = call(wsclean_cab, backend, ms=ms, prefix="out")
    call(casa_rmtables, backend, tablenames=image.model)
"""

from __future__ import annotations

from typing import Any

from shinobi.backends import Backend
from shinobi.exceptions import CabRunError
from shinobi.policies import build_args
from shinobi.results import Result
from shinobi.schema import CabDef


def call(cab: CabDef, backend: Backend, *, check: bool = True, **params: Any) -> Result:
    """Run a cab through the given backend with the given parameters.

    Raises CabRunError if the run fails and check=True (the default) --
    mirroring subprocess.run's check= semantics, since that's exactly what
    this is doing under the hood.
    """
    argv = build_args(cab, params)
    result = backend.run(cab, argv)

    if check and not result.success:
        raise CabRunError(
            f"cab '{cab.name}' exited with status {result.returncode}\n{result.stderr}"
        )

    return result
