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

- **Cabs** (`shinobi.schema.CabDef`) -- a typed, backend-agnostic description of an atomic task: inputs/outputs schema, and *policies* for turning parameters into a CLI invocation. Two interchangeable ways to define one, both producing the same `CabDef`:
  - loaded from existing [cult-cargo](https://github.com/caracal-pipeline/cult-cargo) YAML (`shinobi.loaders.cultcargo`) -- that schema format is good design and is reused as-is, including its `_include` (file composition) and `_use` (dotted-path deep-merge) mechanisms, verified against real upstream cab files. The `=config.x.y` expression language and package-scoped includes are deliberately not implemented -- see the module docstring and `AGENTS.md`.
  - defined directly in Python (`shinobi.decorators.cab`) -- a decorated function's signature becomes the input schema (type hint -> dtype, default -> optional), no YAML required:
    ```python
    from shinobi.decorators import cab

    @cab("breizorro", image="breizorro:latest")
    def breizorro(restored_image: str, threshold: float = 6.5):
        """Mask creation and manipulation for radio astronomy images."""
    ```
- **Backends** (`shinobi.backends`) -- pluggable executors, all shelling out to the relevant CLI rather than a Python SDK: `native` (subprocess), `docker`/`podman`/`apptainer`, `slurm` (`sbatch`/`sacct`), `kubernetes` (`kubectl`, batch `Job`s). Every backend blocks until the job finishes and returns a `Result` -- no async mode, recipes are plain Python. Container/cluster backends derive bind mounts from the cab's own schema (File/MS-dtype params get their parent dir mounted). `native`/container backends were verified against a real `quay.io/stimela/wsclean` image; `kubernetes` against a real `kind` cluster; `slurm` was not live-verified (no cluster was available in the dev environment) -- see `AGENTS.md` for what that means in practice.
- **Recipes** (`shinobi.recipe.call`, `shinobi.decorators.recipe`) -- just Python. `@recipe` optionally attaches schema metadata (derived the same way `@cab` does) so the CLI can expose a recipe's parameters as options, but it never replaces the function -- the body is the orchestration and stays directly callable.
- **Config** (`shinobi.config.AppConfig`) -- layered settings via pydantic-settings: built-in defaults < config file < env vars (`SHINOBI_*`) < explicit overrides.

## CLI

Every `@cab` or `@recipe`-decorated function can be run directly, without writing a Python entrypoint script -- its signature/schema becomes CLI options automatically:

```bash
ninja run cabs.py:breizorro --restored-image out-image.fits --threshold 7
ninja run myrecipes.py:selfcal --ms data.ms --threshold 6.5
```

`ninja run <target>` resolves `<target>` (`path/to/file.py:name` or a dotted module path) and dispatches to `shinobi.recipe.call()` for a bare `@cab`, or calls a `@recipe`-decorated function directly with the parsed options.

Add `--dryrun` to see the execution graph a target would produce, without running anything:

```
$ ninja run myrecipes.py:selfcal --ms data.ms --dryrun
[ make_image ]
      │
      ▼
[  breizorro  ]
```

This actually runs the recipe's real Python code (so its `if`/`for` do whatever they'd really do for the given options), just with every cab swapped for a no-op that records the call instead of executing it -- so it only ever shows the *one* path taken for these inputs, never an untaken branch. Fan-out/fan-in appear when the recipe genuinely threads one step's output into two later ones (or vice versa); see `AGENTS.md` for how that's detected and why `pipefunc` (a static, declared-pipeline library) wasn't a fit for this.

See `AGENTS.md` for design conventions and what's deliberately left out.

## Status

Early scaffolding. Interfaces above are real and tested (`pytest`), but this is not yet ready to run real pipelines.

## Development

```bash
uv venv .venv && uv pip install -e . --group dev
.venv/bin/pytest
.venv/bin/ruff check src tests
```
