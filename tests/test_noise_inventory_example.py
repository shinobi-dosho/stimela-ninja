"""The campaign inventory is a fixed ordinary product in a strict MS branch."""

import importlib
import json
from pathlib import Path

import pytest

from shinobi.ownership import contained_access_issues, scope_path_accesses


@pytest.fixture
def example(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "examples" / "noise_comparison"
    monkeypatch.syspath_prepend(str(root))
    return importlib.import_module("noise_inventory")


def test_inventory_declares_fixed_product_without_generic_mutation(example, tmp_path):
    destination = tmp_path / "products.json"
    directories = []
    for name, filenames in {
        "wsclean": ("image-dirty.fits", "image-psf.fits"),
        "ddfacet": ("image.dirty.fits", "image.psf.fits"),
        "casa": ("dirty.fits", "psf.fits", "metadata.json"),
    }.items():
        directory = tmp_path / name
        directory.mkdir()
        directories.append(directory)
        for filename in filenames:
            (directory / filename).write_bytes(b"native-validator-tested-product")
    inputs = {"manifest": str(destination), "directories": directories}
    scope = example.inventory.step
    accesses, snapshots = scope_path_accesses(scope, inputs, workspace=tmp_path)
    assert (destination, True) in accesses
    assert contained_access_issues(scope, inputs, workspace=tmp_path, dataset_resources={tmp_path / "simulation.ms"}, validated_steps=snapshots) == ()
    output = example.inventory.func.__wrapped__(**inputs)
    assert output.manifest == destination
    assert len(json.loads(destination.read_text())["products"]) == 6
    (directories[0] / "image-psf.fits").unlink()
    with pytest.raises(ValueError, match="missing or empty required product"):
        example.inventory.func.__wrapped__(**inputs)
