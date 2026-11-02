"""Native, declared catalogue of completed image/PSF products."""

import json
from pathlib import Path
from pydantic import BaseModel
from shinobi import pystep


class Inventory(BaseModel):
    manifest: Path


@pystep(backend="native", write_paths=["manifest"])
def inventory(manifest: str, directories: list[Path]) -> Inventory:
    products = []
    for directory in directories:
        expected = {"wsclean": ("image-dirty.fits", "image-psf.fits"), "ddfacet": ("image.dirty.fits", "image.psf.fits"), "casa": ("dirty.fits", "psf.fits", "metadata.json")}[
            directory.name
        ]
        for filename in expected:
            path = directory / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing or empty required product: {path}")
        fits = sorted(directory.rglob("*.fits"))
        products.extend({"path": str(path), "bytes": path.stat().st_size} for path in fits)
    manifest = Path(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"products": products}, indent=2) + "\n")
    return Inventory(manifest=manifest)
