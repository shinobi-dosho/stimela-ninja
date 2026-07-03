Vendored copies of real [cult-cargo](https://github.com/caracal-pipeline/cult-cargo) cab definitions, used by `examples/ninja_selfcal.py` so the wsclean cab's ~170 parameters come from the actual maintained schema instead of being hand-declared and drifting out of sync (see the bug that motivated this: a hand-declared `cubical` cab was missing several real parameters, silently, until `ninja run --dryrun` started validating for real).

Vendored (not fetched at runtime) so the example stays self-contained, reproducible, and testable offline -- matching how the rest of this project avoids network dependencies at runtime.

Source: https://github.com/caracal-pipeline/cult-cargo, commit `master` as of 2026-07-03:
- `wsclean.yml`
- `genesis/wsclean/wsclean-base.yml`
- `genesis/cult-cargo-base.yml`

Only wsclean is vendored here. `breizorro`/`cubical`/the CASA tasks/`msutils` are still hand-declared in `ninja_selfcal.py` -- see that file's module docstring for why each one currently can't cleanly switch to its real cult-cargo definition.
