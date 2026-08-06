"""Choice fields and abbreviated CLI options, exercised end-to-end.

Unlike the recipe examples (which wire cabs together programmatically and
never touch the command line), this one exposes two single cabs as
`ninja run` targets so the *CLI* layer -- `clickutil.build_options` -- is
what gets tested. Both cabs come straight from the vendored, genuine
scabha-dialect YAML (`input-dir/simms-cabs.yaml`), whose `choices:` and
`abbreviation:` keys drive the two features shown here:

* **Choice fields** -- a `choices:` list narrows the field to a
  `typing.Literal`, which `build_options` renders as a `click.Choice`:
  the allowed values show up in `--help` and an out-of-set value is
  rejected by click itself (e.g. `skysim`'s `--mode [sim|add|subtract]`,
  `--fits-sky-interp [nearest|linear|cubic]`).

* **Abbreviated options** -- an `abbreviation:` key adds a single-dash
  short alias for a field's `--long-flag` (e.g. `skysim`'s
  `--ascii-sky/-as`, `--fits-sky-interp/-fsi`, `--polarisation/-pol`;
  `telsim`'s `--telescope/-tel`, `--nchan/-nc`).

Both cabs run under the native backend -- the vendored YAML names no image, and
this example exists to exercise the YAML dialect and the native path, not
dosho's containerised simms cabs -- so the commands below actually execute once
`simms` is installed.

    uv pip install --no-deps simms

See the whole schema, choices and abbreviations included, with:

    ninja run examples/simms_cli.py:skysim --help
    ninja run examples/simms_cli.py:telsim --help

Dry-run (print the argv that would be handed to simms, no execution) --
note the short flags and the choice value:

    ninja run examples/simms_cli.py:telsim --dryrun \
        --ms sim.ms -tel meerkat -nc 16 --dtime 30 --ntime 4

    ninja run examples/simms_cli.py:skysim --dryrun \
        --ms sim.ms -as examples/input-dir/testsky.txt -fsi linear \
        --mode add -pol

A real run -- make an empty MS, then simulate the sky into it:

    ninja run examples/simms_cli.py:telsim \
        --ms sim.ms -tel meerkat -nc 16 --dtime 30 --ntime 4
    ninja run examples/simms_cli.py:skysim \
        --ms sim.ms -as examples/input-dir/testsky.txt --column DATA
"""

from __future__ import annotations

from pathlib import Path

from shinobi.loaders.yaml_cab import load_file

_CABS = load_file(Path(__file__).parent / "input-dir" / "simms-cabs.yaml")

# The vendored YAML declares no image, so pin both cabs to the native backend
# the same way the recipe examples do. (dosho's own simms cabs *do* bind an
# image and run containerised -- this example is about the YAML dialect, and
# reaches simms as a plain binary on purpose.)
skysim = _CABS["skysim"].model_copy(update={"backend": "native"})
telsim = _CABS["telsim"].model_copy(update={"backend": "native"})
