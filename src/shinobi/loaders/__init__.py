"""Cab loaders, plus the public helpers for building a cab's schema by hand.

The `shinobi.loaders.cultcargo` and `shinobi.loaders.stimela_classic`
submodules load cab definitions from their respective on-disk formats.
`build_model` (and `sanitize_unique`) are the same helpers those loaders use
to turn a flat ``{name: (dtype, required, default)}`` spec into the pydantic
``inputs_model``/``outputs_model`` a `Cab` needs -- re-exported here as the
supported way to build those models directly, without hand-writing a pydantic
class. The implementation lives in the internal `_modelgen` module.
"""

from __future__ import annotations

from shinobi.loaders._modelgen import build_model, sanitize_unique

__all__ = ["build_model", "sanitize_unique"]
