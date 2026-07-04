Vendored copies of real [simms 3.0](https://github.com/wits-cfa/simms) files, used by `examples/meerkat_simulation.py` for the `telsim`/`skysim` cabs (an empty-MS telescope simulator and a sky-model visibility simulator -- together replacing the old stimela-classic `cab/simms` + the MeqTrees-based `cab/simulator`) and an example ASCII sky catalogue.

Vendored (not fetched at runtime, nor installed as a dependency of shinobi itself) so the example's *schema* stays self-contained, reproducible, and testable offline -- matching how the rest of this project avoids network/filesystem dependencies outside the repo. Actually *running* the example for real still needs the `simms` package installed (see the `examples` dependency group in `pyproject.toml`) -- simms has no docker image, so it runs via shinobi's `NativeBackend`.

Source: https://github.com/wits-cfa/simms, `main` branch, commit `7511d43` as of 2026-07-04:
- `simms/apps/simms-cabs.yaml` -- genuine cult-cargo-format YAML; this is the *actual* schema source for both the real `simms` CLI (via `scabha.schema_utils.clickify_parameters`) and the cab metadata, not an approximation, so it can't drift out of sync the way a hand-declared cab could.
- `tests/testsky.txt` -- a tiny example ASCII sky catalogue (plain whitespace-delimited, `#format:` header line), used as the default `skymodel` input for the `simulate` step.

Two things worth knowing if you touch this file:
- Both cabs' `ms` input is `policies: {positional: true}` -- a genuinely positional CLI arg (no `--ms` flag exists on the real `simms telsim`/`simms skysim` commands). shinobi's `ParamMeta.positional` + `build_argv` support this (see `src/shinobi/policies.py`).
- `command: simms telsim` / `command: simms skysim` are two-word subcommand invocations, not a single executable name -- shinobi's `build_argv` splits `cab.command` on whitespace for exactly this reason.
