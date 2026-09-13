"""Dependency-light pysteps used by the physical M1 Slurm acceptance probe."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class Product(BaseModel):
    product: Path


def image_step(source: Path) -> Product:
    product = Path("image-product.txt")
    product.write_text(source.read_text() + " image")
    return Product(product=product)


def venv_step(source: Path) -> Product:
    import venvonlypkg

    product = Path("venv-product.txt")
    product.write_text(source.read_text() + f" venv={venvonlypkg.MAGIC}")
    return Product(product=product)
