"""Show CLI choices and short options from vendored simms YAML cabs.

Inspect the options with ``ninja run examples/simms_cli.py:skysim --help``
or ``ninja run examples/simms_cli.py:telsim --help``. A dry run shows the
argv without requiring simms::

    ninja run examples/simms_cli.py:telsim --dryrun --ms sim.ms -tel meerkat -nc 16
    ninja run examples/simms_cli.py:skysim --dryrun --ms sim.ms -as examples/input-dir/testsky.txt -fsi linear --mode add
"""

from __future__ import annotations

from pathlib import Path

from shinobi.loaders.yaml_cab import load_file

_CABS = load_file(Path(__file__).parent / "input-dir" / "simms-cabs.yaml")

# The vendored YAML has no image; these cabs use the native backend.
skysim = _CABS["skysim"].model_copy(update={"backend": "native"})
telsim = _CABS["telsim"].model_copy(update={"backend": "native"})
