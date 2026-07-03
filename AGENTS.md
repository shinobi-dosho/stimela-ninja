# shinobi -- design conventions

Spiritual successor to Stimela classic, reacting against Stimela 2.0's YAML-recipe complexity. Read this before adding anything to the recipe/orchestration layer -- it's the part most likely to regrow the exact bloat this project exists to avoid.

## Core rule

**Recipes are plain Python.** A cab call is a function call (`shinobi.recipe.call(cab, backend, **params)`); a step's output is a `Result` object passed as a plain Python value to the next call. Loops are `for`, conditionals are `if`, sub-pipelines are functions calling functions.

Do not add:
- A string-based expression/substitution language for referencing other steps' params or outputs (e.g. stimela2's `=recipe.ms`, `{recipe.name}-{info.suffix}`). If you need a value from a previous step, it's a Python variable.
- An alias-propagation system. Stimela2's DEVNOTES.md describes multi-pass up/down propagation logic to keep step and recipe-level params in sync, plus glob re-evaluation hacks to work around it. That entire class of problem only exists because YAML was the orchestration layer. Don't recreate the problem.
- A second, YAML-based way to express control flow. A thin YAML-to-Python-calls *compiler* (for simple linear pipelines) may be worth adding later, but it must compile into the same call graph, not grow its own semantics.

## What's worth keeping (and why)

- **Typed cab schema** (`shinobi.schema.CabDef`, `ParamSchema`, `Policies`): declarative inputs/outputs with `dtype`/`required`/`default`, and policies that auto-generate CLI args. This is genuinely better than Stimela classic, which required a hand-written `run.py` per cab to build argv imperatively.
- **cult-cargo YAML compatibility** (`shinobi.loaders.cultcargo`): the cab schema format itself isn't the problem -- only the recipe layer built on top of it is. Loading existing cult-cargo cab defs unlocks the whole existing radio-astronomy tool library instead of requiring a rewrite.
- **Python-native cab definitions** (`shinobi.decorators.cab`): a decorated function's signature *is* the input schema (type hint -> dtype, presence of a default -> required), producing the exact same `CabDef` the YAML loader does -- the two are fully interchangeable, `shinobi.recipe.call()` can't tell them apart. The function body is never called for a binary-flavour cab; only its signature and docstring (-> `info`) are read, at decoration time. Per-param detail a signature can't express (a `nom_de_guerre`, `info` text, ...) goes through the `inputs=` override kwarg rather than growing annotation syntax to express it -- don't invent an `Annotated[...]`-based mini-language here for the same reason we don't want one in recipes.
- **Output wranglers** (`shinobi.wranglers`): regex-based extraction of structured outputs from a cab's console output. Only `PARSE_OUTPUT` is implemented; add other actions (`HIGHLIGHT`, `SUPPRESS`, ...) only when a real cab needs them, not speculatively.
- **Backend abstraction** (`shinobi.backends`): a cab doesn't know if it's running natively, in a container, or on a cluster. Backends shell out to the runtime CLI (`docker`/`podman`/`apptainer`/`sbatch`+`sacct`/`kubectl`) rather than using each system's Python SDK -- one code path per backend, no heavyweight client dependencies, and it's the only option for runtimes like apptainer that don't have a good Python API anyway. `Backend.run(cab, argv, params)` is handed the *resolved* params dict alongside argv specifically so container/cluster backends can derive bind mounts from the cab's own schema: any resolved param whose `dtype` looks file-like (`File`, `MS`, `list:File`, ...) gets its parent directory mounted at the same path (`shinobi.backends.container.bind_dirs`, shared with the Kubernetes backend). Every backend's `run()` blocks until the job/container/process is done and returns a `Result` -- there's no async/fire-and-forget mode, because recipes are plain Python and the next line usually needs this step's output.
  - `native`/`docker`/`podman`/`apptainer` were verified against a real `quay.io/stimela/wsclean` image and a real bind-mounted host file in `tests/test_docker_live.py` (skipped if docker/the image isn't available) -- not just mocked.
  - `slurm` (submits via `sbatch`, polls `sacct`, wraps the command in apptainer by default since Slurm schedules compute but doesn't run containers itself) and `kubernetes` (submits a batch `Job` via `kubectl apply`, polls `kubectl get job`, mounts File/MS params as `hostPath` volumes) were **not** live-verified -- no cluster was available in the dev environment they were built in. They're reviewed-by-construction and covered by tests that mock the CLI calls (`tests/test_slurm_backend.py`, `tests/test_kubernetes_backend.py`), not proven against a real scheduler/cluster. Verify against a real one before relying on them. `hostPath` volumes on the Kubernetes backend also only work if the node actually running the pod has that path (fine for a single-node dev cluster or nodes sharing storage, not a general multi-node production cluster -- that needs PersistentVolumeClaims, deliberately not built).

## Config: one validation library, not five

Stimela 2.0 stacks `omegaconf` + its own `scabha.configuratt` + `munch` + `python-benedict` for config handling. shinobi uses `pydantic` + `pydantic-settings` only -- the same library already used for cab schemas. Precedence, highest to lowest: explicit overrides (CLI) > env vars (`SHINOBI_*`) > config file > built-in defaults. See `shinobi/config.py`.

## Repo layout

```
src/shinobi/
  schema.py            # ParamSchema, Policies, CabDef
  decorators.py         # @cab -- Python-native cab definitions
  policies.py           # resolve_params / build_argv / build_args
  wranglers.py          # stdout/stderr -> structured outputs
  results.py            # Result (what call() returns)
  recipe.py             # call() -- the entire "recipe" API
  config.py             # AppConfig (pydantic-settings)
  cli.py                # click entrypoint
  backends/
    __init__.py          # Backend ABC + registry (get_backend/register); imports every
                          # backend submodule so @register actually fires without the
                          # caller having to import that specific backend module first
    native.py             # subprocess
    container.py          # docker/podman/apptainer, shells out to the runtime CLI, derives bind mounts from schema
    slurm.py               # sbatch/sacct, not live-verified (no cluster in dev env)
    kubernetes.py          # kubectl, not live-verified (no cluster in dev env)
  loaders/
    cultcargo.py          # cult-cargo YAML -> CabDef
tests/                    # one test module per src module; run via `pytest`
                          # test_docker_live.py is a real (non-mocked) integration test, skipped without docker
                          # test_slurm_backend.py / test_kubernetes_backend.py mock the CLI calls
```

## Before adding a feature

Ask whether Stimela classic or Stimela 2.0 already solved this, and which one solved it *simply*. If neither did, keep the new piece as small and boring as possible -- this project's entire reason to exist is refusing complexity that isn't earning its keep.
