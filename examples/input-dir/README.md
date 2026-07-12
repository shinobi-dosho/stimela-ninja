Vendored copies of real [simms 3.0](https://github.com/wits-cfa/simms) files.

`testsky.txt` (a tiny example ASCII sky catalogue, plain whitespace-delimited,
`#format:` header line) is used as `examples/meerkat_simulation.py`'s default
`skymodel` input for its `simulate` step.

`simms-cabs.yaml` is no longer used by `meerkat_simulation.py` itself --
that example now gets its `telsim`/`skysim` cabs from
[dosho](https://github.com/SpheMakh/dosho) (`dosho.cabs.simms`), the
native shinobi cab repository, instead of loading this vendored YAML
directly. It's kept as a real-file regression fixture for
`shinobi.loaders.cultcargo`'s own dtype handling
(`tests/test_cultcargo_loader.py::test_bracket_list_dtype_resolves_on_real_simms_example`
locks in bracket-syntax `List[<inner>]` dtype support against this file's
`telsim` cab's `subarray-list`/`subarray-range` fields) -- a genuine
cult-cargo-format YAML file, not a synthetic one, so it's still useful for
that even though it's no longer wired into a runnable recipe here.

Source: https://github.com/wits-cfa/simms, `main` branch, commit `7511d43`
as of 2026-07-04:
- `simms/apps/simms-cabs.yaml` -- genuine cult-cargo-format YAML; this was
  the *actual* schema source for both the real `simms` CLI (via
  `scabha.schema_utils.clickify_parameters`) and the cab metadata, not an
  approximation, so it couldn't drift out of sync the way a hand-declared
  cab could.
- `tests/testsky.txt` -- a tiny example ASCII sky catalogue.
