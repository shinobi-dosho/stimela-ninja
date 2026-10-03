# shinobi -- repository guardrails

Spiritual successor to Stimela classic, using Python to declare workflow DAGs.
Organisation-wide conventions live in
[`shinobi-dosho/.github`](https://github.com/shinobi-dosho/.github/blob/main/AGENTS.md);
this file contains only `stimela-ninja`-specific rules and wins on conflict.

## Sources of truth

Use current material in this order:

1. Maintained documentation under [`docs/`](docs/), especially the relevant
   concept pages.
2. Public and module docstrings describing the implementation contract.
3. Tests and the working implementation.
4. This file for cross-cutting guardrails.

If these disagree, do not make code conform to stale prose. Reconcile the
documentation or ask the maintainer when the intended contract is ambiguous.

## GitHub and repository state

Use `gh` as the primary GitHub interface. A sandboxed invocation may report an
invalid token or fail to connect despite a valid login; retry the same command
with sandbox network access before diagnosing authentication. If that still
fails, ask the maintainer before switching interfaces.

Treat this working tree, GitHub through `gh`, or a fresh scratch checkout as
the source of truth. Never infer upstream state from another existing local
checkout unless the maintainer explicitly instructs you to use it; it may
contain experimental or untested work.

## Orchestration boundary

**A constructed `Recipe` is a declared DAG.** `build_graph()` must be able to
validate and render every possible node and edge before dispatch. Explicit
`InputRef`/`OutputRef` wiring carries data dependencies; `after` is ordering
only. `add_loop()` may short-circuit already-declared, bounded iterations.

Ordinary Python remains valid at the boundary: a recipe builder may decide
which nodes to declare, and an orchestration function bound by `@shinobi.step`
may perform local behavior inside one opaque node. Such a node is not eligible
for compile-and-offload. Use `add_loop()` when repetition is graph structure.

Do not add:

- string expressions for cross-step references or substitutions;
- alias propagation between recipe and step parameters; or
- run-time control flow, expression evaluation, or alias resolution in YAML.

Static YAML/JSON cab schemas, `AppConfig`, worker configuration, and supported
`_include`/`_use` composition are data. Package-scoped includes resolve only
through caller-supplied filesystem roots; loading markup must never import a
named package or execute `dynamic_schema`. `ParamMeta.implicit`, `harvest`, and
`scratch` may format current validated inputs, but may not address other steps
or evaluate code. See `docs/concepts/loaders.rst`, `docs/concepts/recipes.rst`,
and `SECURITY.md`.

## Implementation guardrails

- Search for an existing behavior before implementing it. Put shared rules in
  one helper and route every caller through it; do not maintain near-copies.
- Pydantic models are authoritative for inputs, outputs, and configuration.
  Use `pydantic-settings` for config precedence; do not introduce another
  validation/configuration stack.
- File-like schema annotations, not string-value heuristics, determine path
  behavior. Keep declaration discovery shared across loaders, sandboxing,
  backends, CLI generation, and dataset planning.
- Outputs and write targets must be declared. Never discard either side of a
  read-only/write collision; preserve nested read-only mounts where possible
  and refuse contradictory access over the same tree.
- `Mutability.IMMUTABLE` protects Python objects, not files. Filesystem access
  is governed by `writable`, declared outputs, `write_path`, and dataset-access
  declarations.
- Backend `run()` calls are blocking and return `BackendRun`. Keep argv
  construction in `shinobi.policies`; backends receive ready-made argv.
- The skip cache, mutation snapshots, reusable dataset states, ownership, and
  worker attempt records are distinct systems with distinct identities. Read
  `docs/concepts/provenance.rst`, `docs/concepts/states.rst`, and
  `docs/offloading.rst` before changing their interaction. Never collapse an
  observed state, cache key, snapshot name, logical state ID, or attempt ID
  merely because two currently contain related hashes.
- Preserve no-replace publication, durable liveness evidence, and exact-path
  recovery. Do not replace them with age tests, broad globs, or PID guesses.

## Security

Never `eval()`/`exec()` a cab's `command`. Cab definitions are untrusted data
and real dialects contain inline code or dotted function references. Read
`SECURITY.md` before changing loaders, `build_argv()`, script generation, path
publication, or include resolution.

## Changes and review

Before adding a feature, check whether Stimela classic or Stimela 2.0 solved
the problem more simply. Add complexity only when a current use case requires
it.

Verify claims against the whole working tree, not only the diff. Before saying
something is missing, dead, or unreferenced, inspect tracked files and search
the repository. Run focused tests for the changed contract and the broader
suite appropriate to its risk; optional live tests must skip clearly when the
required runtime is unavailable.

## Attribution

Commits made with assistant help end with a plain-text trailer:

```text
Assisted-by: <LLM> <MODEL>
```

Do not use an assistant `Co-authored-by:` trailer or email address. Pull
request descriptions carry no assistant trailer; the commit already records
provenance. The trailer never substitutes for a commit message explaining the
decision and its consequences.
