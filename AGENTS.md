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
- **cult-cargo YAML compatibility** (`shinobi.loaders.cultcargo`): the cab schema format itself isn't the problem -- only the recipe layer built on top of it is. Loading existing cult-cargo cab defs unlocks the whole existing radio-astronomy tool library instead of requiring a rewrite. Some real cult-cargo cabs carry non-`"binary"` flavours whose `command` is executable code, not a program name -- see "Never eval()/exec() a cab's `command`" below for how that's handled.
- **Python-native cab definitions** (`shinobi.decorators.cab`): a decorated function's signature *is* the input schema (type hint -> dtype, presence of a default -> required), producing the exact same `CabDef` the YAML loader does -- the two are fully interchangeable, `shinobi.recipe.call()` can't tell them apart. The function body is never called for a binary-flavour cab; only its signature and docstring (-> `info`) are read, at decoration time. Per-param detail a signature can't express (a `nom_de_guerre`, `info` text, ...) goes through the `inputs=` override kwarg rather than growing annotation syntax to express it -- don't invent an `Annotated[...]`-based mini-language here for the same reason we don't want one in recipes.
- **Output wranglers** (`shinobi.wranglers`): regex-based extraction of structured outputs from a cab's console output. Only `PARSE_OUTPUT` is implemented; add other actions (`HIGHLIGHT`, `SUPPRESS`, ...) only when a real cab needs them, not speculatively.
- **Backend abstraction** (`shinobi.backends`): a cab doesn't know if it's running natively, in a container, or on a cluster. Backends shell out to the runtime CLI (`docker`/`podman`/`apptainer`/`sbatch`+`sacct`/`kubectl`) rather than using each system's Python SDK -- one code path per backend, no heavyweight client dependencies, and it's the only option for runtimes like apptainer that don't have a good Python API anyway. `Backend.run(cab, argv, params)` is handed the *resolved* params dict alongside argv specifically so container/cluster backends can derive bind mounts from the cab's own schema: any resolved param whose `dtype` looks file-like (`File`, `MS`, `list:File`, ...) gets its parent directory mounted at the same path (`shinobi.backends.container.bind_dirs`, shared with the Kubernetes backend). Every backend's `run()` blocks until the job/container/process is done and returns a `Result` -- there's no async/fire-and-forget mode, because recipes are plain Python and the next line usually needs this step's output.
  - `native`/`docker`/`podman`/`apptainer` were verified against a real `quay.io/stimela/wsclean` image and a real bind-mounted host file in `tests/test_docker_live.py` (skipped if docker/the image isn't available) -- not just mocked.
  - `slurm` (submits via `sbatch`, polls `sacct`, wraps the command in apptainer by default since Slurm schedules compute but doesn't run containers itself) and `kubernetes` (submits a batch `Job` via `kubectl apply`, polls `kubectl get job`, mounts File/MS params as `hostPath` volumes) were live-verified against a real `kind` cluster (`tests/test_kubernetes_live.py`, skipped without a reachable cluster) -- proven, not just reviewed-by-construction. `slurm` has no equivalent live test yet (no cluster was available to verify against); it's covered by tests that mock the CLI calls (`tests/test_slurm_backend.py`), not proven against a real scheduler. Verify against a real one before relying on it. `hostPath` volumes on the Kubernetes backend also only work if the node actually running the pod has that path (fine for a single-node dev cluster, or nodes sharing storage -- confirmed on `kind` via `extraMounts` -- not a general multi-node production cluster without shared storage, which needs PersistentVolumeClaims instead, deliberately not built).
- **`@recipe` decorator** (`shinobi.decorators.recipe`): the recipe-side counterpart to `@cab` -- derives a `ParamSchema` dict from a function's signature (reusing the same `_inputs_from_signature` helper `@cab` uses), purely so tooling can see a recipe's parameters without executing it. Unlike `@cab`, it does **not** replace the function: the decorated name stays directly callable, because a recipe's body is the orchestration logic itself (see Core rule) and must run exactly as if undecorated. Metadata is attached as `func.__shinobi_recipe__: RecipeInfo` -- a new, deliberately minimal schema type with no `command`/`image`/`policies`/`outputs`/`wranglers`, since a recipe manages its own backend/execution in its body and has no single command of its own. This is unrelated to the earlier-deferred "recipe as cab" idea (making a recipe passable to `call()` with backend-dispatch parity to a `CabDef`) -- `@recipe` targets are always invoked as a plain Python function call, never through `Backend.run()`.
- **`ninja run <target> [OPTIONS]` CLI** (`shinobi.cli`): a thin argv-to-kwargs translator for both `@cab` and `@recipe` targets, nothing more. `<target>` is `path/to/file.py:name` or `dotted.module:name`, resolved via `importlib`; the target's existing schema (`CabDef.inputs`, or a `@recipe`'s derived `ParamSchema` dict) is turned into `click.Option`s *at runtime* (the target isn't known until the CLI parses it -- this needs the underlying `click.Command`/`click.Option` object API directly, since declarative `@click.option` is fixed at import time), then dispatched to exactly the call either would already get from Python: `shinobi.recipe.call(cab, backend, **kwargs)` for a cab, or a direct function call for a recipe. This is not a second orchestration language -- there's no new expression syntax, no step-referencing, no control flow beyond "parse flags, call the one function/cab named"; every recipe's actual pipeline logic still lives entirely in its Python body. The CLI is a caller, not an interpreter. `ninja` is the sole console-script entry point (PyPI/distribution name `ninja-fm`); `shinobi` stays the importable library name only -- there's no `shinobi` command.
- **`ninja run <target> --dryrun`** (`shinobi.dag`, `shinobi.backends.trace`): shows the execution graph a target *would* produce, as a box-drawing diagram, without running anything. Since recipes are plain Python, there's no declared graph anywhere to read -- the only honest way to show one is to actually execute the recipe's real code, with every registered `Backend` subclass's `run()` swapped out (via `patch_all_backends()`, which patches the *classes*, not `get_backend`/`call` -- a recipe usually does `from shinobi.backends import get_backend` at its own module top level, and patching a module-level function wouldn't reach a name already bound that way; a class's `run` method is looked up dynamically via the instance's type at call time, so patching it there reaches every instance regardless of import style) for `TraceBackend`, which records each call instead of executing it and hands back a placeholder value per declared output. A later call's params are scanned for those placeholders (`find_dependencies`) to detect *real* data dependencies -- not just "happened after" -- so a recipe that actually threads one step's output into two later ones (or two steps' outputs into one later one) renders as genuine fan-out/fan-in, matching a CI-pipeline-style diagram; a call with no detected dependency chains after the immediately preceding one instead of floating disconnected. This intentionally only shows the *one* path taken for the given inputs, never an untaken branch -- evaluated and explicitly rejected adopting the `pipefunc` package for this, because its pipeline model requires a fully-declared, static graph at decoration time (the same class of thing the Core rule rejects), so it structurally can't trace arbitrary, unmodified branching Python the way this does. A recipe that does real arithmetic/comparisons on a dry-run placeholder (rather than just threading it through) can raise partway through -- that's reported, and whatever was traced up to that point is still shown, rather than crashing the CLI.

## Never eval()/exec() a cab's `command`

Cab definitions -- especially cult-cargo YAML, which shinobi loads from arbitrary files -- are effectively untrusted content that can contain executable code as data. Real cult-cargo cabs exist where `command:` is inline Python/shell source (e.g. `bdsf.catalog`) or a dotted reference to a function to import and call (e.g. `msutils.copycol`'s `flavour: python`); these are non-`"binary"` flavours.

shinobi never treats a non-`"binary"` cab's `command` as code to run: every backend shells out via `subprocess.run(argv_list, ...)` with a list (never `shell=True`, never `eval()`/`exec()`), and `shinobi.policies.build_argv()` explicitly rejects any cab whose `flavour` isn't `"binary"` with `UnsupportedFlavourError`, *before* argv is ever built -- so a non-executable `command` can never reach subprocess as argv[0] in the first place, let alone be interpreted as code. This check runs even during `ninja run --dryrun` (it's in `build_argv()`, which `shinobi.recipe.call()` always calls before touching the backend), so a recipe hitting an unsupported-flavour cab is reported clearly rather than silently mishandled.

If proper support for a code-carrying flavour is ever added: don't `eval()`/`exec()` the embedded string in-process. The safe shape is to write it to a temp file and invoke a real subprocess on it (`python /tmp/x.py --args`, still a list argv, no shell) -- same sandboxing boundary as every other cab, no in-process code execution. `dynamic_schema: dotted.path` (real cult-cargo's `wsclean.yml` uses this) is a related, separate risk -- resolving it means *importing* an arbitrary module and *calling* a function it names, at cab-load time. Not implemented; `shinobi.loaders.cultcargo` warns when it sees the key rather than silently producing a possibly-incomplete schema (a cab relying solely on `dynamic_schema` with no static `inputs:`/`outputs:` loads empty).

## Config: one validation library, not five

Stimela 2.0 stacks `omegaconf` + its own `scabha.configuratt` + `munch` + `python-benedict` for config handling. shinobi uses `pydantic` + `pydantic-settings` only -- the same library already used for cab schemas. Precedence, highest to lowest: explicit overrides (CLI) > env vars (`SHINOBI_*`) > config file > built-in defaults. See `shinobi/config.py`.

## Repo layout

```
src/shinobi/
  schema.py            # ParamSchema, Policies, CabDef, RecipeInfo, is_file_like_dtype
  decorators.py         # @cab, @recipe -- Python-native cab/recipe definitions
  policies.py           # resolve_params / build_argv / build_args; build_argv() rejects
                         # non-"binary"-flavour cabs (see "Never eval()/exec()..." above)
  wranglers.py          # stdout/stderr -> structured outputs
  results.py            # Result (what call() returns)
  recipe.py             # call() -- the entire "recipe" API
  dag.py                 # TraceStep / find_dependencies / render_dag -- ninja run --dryrun
  config.py             # AppConfig (pydantic-settings)
  cli.py                # click entrypoint (ninja); `run` dynamically builds --options from
                         # a target's CabDef/RecipeInfo schema; --dryrun traces via shinobi.dag
  backends/
    __init__.py          # Backend ABC + registry (get_backend/register/registered_backend_classes);
                          # imports every backend submodule so @register actually fires without
                          # the caller having to import that specific backend module first
    native.py             # subprocess
    container.py          # docker/podman/apptainer, shells out to the runtime CLI, derives bind mounts from schema
    slurm.py               # sbatch/sacct, not live-verified (no cluster in dev env)
    kubernetes.py          # kubectl, live-verified against a real kind cluster
    trace.py               # TraceBackend + patch_all_backends() -- ninja run --dryrun, not
                            # registered in the normal backend registry
  loaders/
    cultcargo.py          # cult-cargo YAML -> CabDef
tests/                    # one test module per src module; run via `pytest`
  fixtures/sample_targets.py  # tiny @cab/@recipe targets used by test_cli.py
                          # test_docker_live.py / test_kubernetes_live.py are real
                          # (non-mocked) integration tests, skipped without docker/a cluster
                          # test_slurm_backend.py / test_kubernetes_backend.py mock the CLI calls
                          # (kubernetes also has the above live test; slurm doesn't yet)
```

## Before adding a feature

Ask whether Stimela classic or Stimela 2.0 already solved this, and which one solved it *simply*. If neither did, keep the new piece as small and boring as possible -- this project's entire reason to exist is refusing complexity that isn't earning its keep.
