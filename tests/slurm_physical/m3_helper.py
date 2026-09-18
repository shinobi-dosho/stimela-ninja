"""Bundled helper used by the physical M3 image pysteps.

Keeping this in a separate module proves that offload captures literal local
imports along with the callable's source instead of consulting the checkout on
the compute node.
"""

from __future__ import annotations


def image_label(version: str) -> str:
    return f"image[{version}]"
