# shinobi (Stimela 3.0)

A spiritual successor to [Stimela classic](https://github.com/ratt-ru/Stimela-classic), built around the same core philosophy: **functional and flexible simplicity for reproducible radio astronomy pipelines**.

Recipes are plain Python. A cab call is a function call; a step's output is a Python value you pass to the next call. There is no YAML expression/substitution language, no alias-propagation system, and no stacked config libraries -- control flow is just Python, and it doesn't need reinventing.

```python
from shinobi.backends import get_backend
from shinobi.loaders.cultcargo import load_file
from shinobi.recipe import call

cabs = load_file("cabs.yml")
backend = get_backend("native")

result = call(cabs["wsclean"], backend, ms="data.ms", prefix="out")
call(cabs["breizorro"], backend, restored_image=result.image)
```

## Architecture

- **Cabs** (`shinobi.schema.CabDef`) -- a typed, backend-agnostic description of an atomic task: inputs/outputs schema, and *policies* for turning parameters into a CLI invocation. Cab definitions can be loaded from existing [cult-cargo](https://github.com/caracal-pipeline/cult-cargo) YAML (`shinobi.loaders.cultcargo`) -- that schema format is good design and is reused as-is, including its `_include` (file composition) and `_use` (dotted-path deep-merge) mechanisms, verified against real upstream cab files. The `=config.x.y` expression language and package-scoped includes are deliberately not implemented -- see the module docstring and `AGENTS.md`.
- **Backends** (`shinobi.backends`) -- pluggable executors: `native` (subprocess) and `docker`/`podman`/`apptainer` (shells out to the runtime binary) ship today; Slurm/Kubernetes are a later milestone.
- **Recipes** (`shinobi.recipe.call`) -- just Python. No separate recipe schema.
- **Config** (`shinobi.config.AppConfig`) -- layered settings via pydantic-settings: built-in defaults < config file < env vars (`SHINOBI_*`) < explicit overrides.

See `AGENTS.md` for design conventions and what's deliberately left out.

## Status

Early scaffolding. Interfaces above are real and tested (`pytest`), but this is not yet ready to run real pipelines.

## Development

```bash
uv venv .venv && uv pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/ruff check src tests
```
