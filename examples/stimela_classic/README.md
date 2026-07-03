Vendored copies of real [stimela-classic](https://github.com/ratt-ru/Stimela) `parameters.json` cab definitions, loaded via `shinobi.loaders.stimela_classic` and used by `examples/ninja_selfcal.py` for the CASA tasks and `msutils` -- the cabs cult-cargo either doesn't have a loadable definition for at all, or that this recipe hand-declared as best-effort guesses (see `ninja_selfcal.py`'s module docstring for the prior state and why).

Vendored (not read from a local checkout at runtime) so the example stays self-contained, reproducible, and testable offline -- matching how the rest of this project avoids network/filesystem dependencies outside the repo.

Source: https://github.com/ratt-ru/Stimela, most recently touched 2022-06-01 (commit `4bd0664`):
- `casa_mstransform/parameters.json`
- `casa_listobs/parameters.json`
- `casa_flagdata/parameters.json`
- `casa_flagmanager/parameters.json`
- `msutils/parameters.json`

`breizorro`/`cubical`/`wsclean` are not vendored here -- `wsclean` already loads from a real cult-cargo definition (`examples/cultcargo/`), and `breizorro`/`cubical` stay hand-declared for the reasons `ninja_selfcal.py`'s module docstring already documents (implicit-output mismatch, and a deliberate `ParamPattern` demonstration for cubical's per-Jones-term parameters) -- switching those to classic's schema isn't what this change is about.
