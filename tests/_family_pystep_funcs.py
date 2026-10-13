"""Stdlib-only function bodies for product-family subprocess fixtures."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from shinobi import ProductFamily


class RootFamilyOutput(BaseModel):
    image: ProductFamily[Path]


def write_root_family(root: Path) -> RootFamilyOutput:
    (root / "out.fits").write_text("python product")
    return RootFamilyOutput(image={"resolved": True, "members": [{"coordinates": {}, "value": root / "out.fits"}]})
