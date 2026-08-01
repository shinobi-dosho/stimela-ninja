# Security Policy

## Supported versions

stimela-ninja is early-beta software. Security fixes are applied to the latest
`0.x` release only; there are no long-term-support branches yet.

| Version | Supported |
| ------- | --------- |
| latest `0.x` | ✅ |
| older       | ❌ |

## Reporting a vulnerability

**Please do not report security issues in public GitHub issues.**

Report vulnerabilities privately by email to **sphemakh@gmail.com** (or via
GitHub's [private vulnerability reporting][ghsa] on this repository, if
enabled). Include enough detail to reproduce — affected version, backend,
inputs, and the impact you observed.

We aim to acknowledge reports within a reasonable time, work with you on a fix,
and credit you in the release notes if you'd like.

[ghsa]: https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability

## Security posture

Cab definitions — especially YAML cabs in the scabha dialect, which shinobi loads from
arbitrary files — are effectively untrusted content that can contain
executable code as data. Real cult-cargo cabs exist where `command:` is
inline Python/shell source (e.g. `bdsf.catalog`) or a dotted reference to a
function to import and call (e.g. `msutils.copycol`'s `flavour: python`);
these are non-`"binary"` flavours.

### Recipes, unlike cab definitions, *always* execute

Everything below is about **cab and config definitions** — the YAML shinobi
*reads*. It does not extend to recipes, and the distinction is easy to
over-generalise, so state it plainly:

**A shinobi recipe is a Python file, and `ninja run file.py:target` imports
and executes it.** That is the project's central design choice, not a gap:
control flow is Python, so running a recipe runs arbitrary Python with your
privileges, before any cab is dispatched or any sandbox exists.

Treat a recipe exactly as you would any script someone sent you — read it
before you run it, and don't run one from a source you wouldn't `curl | sh`
from. The guarantees in this document are about what a *cab definition* can
make shinobi do; none of them constrain what a recipe can do, because a
recipe is code by design.

### Never eval()/exec() a cab's `command`

shinobi never treats a non-`"binary"` cab's `command` as code to run: every
backend shells out via `subprocess.run(argv_list, ...)` with a **list**
(never `shell=True`, never `eval()`/`exec()`), and
`shinobi.policies.build_argv()` explicitly rejects any cab whose `flavour`
isn't `"binary"` with `UnsupportedFlavourError`, *before* argv is ever built
— so a non-executable `command` can never reach subprocess as `argv[0]` in
the first place, let alone be interpreted as code. This check runs even
during `ninja run --dryrun` (it's in `build_argv()`, which dispatch always
calls before touching the backend), so a recipe hitting an
unsupported-flavour cab is reported clearly rather than silently mishandled.

If proper support for a code-carrying flavour is ever added: don't
`eval()`/`exec()` the embedded string in-process. The safe shape is to write
it to a temp file and invoke a real subprocess on it (`python /tmp/x.py
--args`, still a list argv, no shell) — same sandboxing boundary as every
other cab, no in-process code execution.

### `dynamic_schema` and package-scoped includes are not resolved

`dynamic_schema: dotted.path` (real cult-cargo's `wsclean.yml` uses this) is
a related, separate risk — resolving it means *importing* an arbitrary
module and *calling* a function it names, at cab-load time. Not implemented;
`shinobi.loaders.yaml_cab` warns when it sees the key rather than silently
producing a possibly-incomplete schema (a cab relying solely on
`dynamic_schema` with no static `inputs:`/`outputs:` loads empty).

The same boundary extends to "never import a cab package": resolving the
package-scoped `_include` form would normally mean importing the named
package (`importlib`) to find its data directory, but that risks executing
arbitrary code from *any* `__init__.py` on the path. Instead, callers pass
an explicit `package_roots={"cultcargo": Path(...)}` mapping into the
loader, and a dotted name is resolved against the longest registered prefix
as a plain filesystem lookup — never through Python's import machinery.

This applies to **both** loader dialects — `shinobi.loaders.yaml_cab`
(cab schemas) and `shinobi.loaders.worker_schema` (scabha-dialect config
schemas) — which share one `resolve_package_root` helper precisely so
neither can drift away from the rule. A package-scoped `_include` naming a
package with no registered root is a load error, never an import.

#### What a registered root does and does not promise

A `package_roots` entry is a **containment** boundary, not just a lookup
table: an `_include` chain that enters through a root may only read files
*within* that root. `resolve_package_root` constrains the dotted part of the
name, and `_modelgen.contain_include` constrains the file part at the join,
so neither `_include: (cultcargo)../../../etc/anything.yaml` nor
cult-cargo's `{"(cultcargo)": ["../../../etc/anything.yaml"]}` dict form can
read outside the directory you pointed at. Paths are resolved on both sides
before comparison, so a symlink planted inside the package is caught too.

The boundary is **transitive**. A file legitimately included from a root
cannot re-escape with a plain relative `_include: ../../../etc/anything.yaml`
of its own, even though that resolves against its own directory rather than
the root. A nested *package-scoped* include is the one thing that changes the
boundary, replacing it with the new package's root — that hop goes back
through the `package_roots` mapping, which only the caller controls.

What this does **not** cover: a plain relative `_include` below no package
root at all. `../common/base.yaml` from a schema file you passed by path is
ordinary, widely-used layout, and only `package_roots` ever made a
containment promise — so relative includes there are unconstrained, exactly
as the file's own directory placement already implies. Treat the schema
files you point shinobi at as you would the recipe that loads them.

### Backends never shell out through a shell

Backends invoke commands with **list-form** `subprocess.run` — never
`shell=True`, never string interpolation into a shell.

### Snapshots copy in-process, not through `cp`

The mutation-chain snapshots (`shinobi.snapshots`) copy and restore
measurement-set trees through `shinobi.clonefs`, which uses `fcntl.ioctl`,
`os.copy_file_range` and plain reads/writes **in this process**. It never
shells out at all — not to `cp`, not to `tar`.

That is the stronger version of the exec-form discipline above, and it is
deliberate: these operations take *workspace paths* as arguments, which are
user-supplied data, and a path is exactly the kind of string that acquires a
shell meaning it did not have when written. Copying in-process removes the
question rather than answering it, and as a bonus removes a dependency on
which GNU `cp` version is installed. Should a cold tier ever add `tar`/`zstd`
(Tier 2, deferred), it must use exec-form argv, as everything else here does.

### Offload scripts are charset-validated before interpolation

Compiled offload scripts (`shinobi.offload.slurm` and the `slurm` step
backend share one script-writing module, `shinobi.backends.slurm_script`, so
the hardening below can't drift between the two) embed **exec-form argv
only** (via `shlex.join`, never a shell template), and `cab.name`/job-name/
sbatch-option keys are charset-validated before being interpolated into a
`#SBATCH` line — a newline in a cab name pulled from untrusted cult-cargo
YAML would otherwise be able to smuggle in an extra `#SBATCH` directive. The
non-`"binary"` flavour guard is inherited via `build_argv()`, so an offloaded
recipe gets the same guarantee as a locally-run one.

If you find a way around any of these guarantees, it's a security issue — please
report it as above.
