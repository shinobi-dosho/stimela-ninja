# Security Policy

## Supported versions

stimela-ninja is early-beta software. Security fixes are applied to the latest
`0.x` release only; there are no long-term-support branches yet.

| Version | Supported |
| ------- | --------- |
| latest `0.x` | :white_check_mark: |
| older       | :x: |

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

Cab definitions — especially cult-cargo YAML, which shinobi loads from
arbitrary files — are effectively untrusted content that can contain
executable code as data. Real cult-cargo cabs exist where `command:` is
inline Python/shell source (e.g. `bdsf.catalog`) or a dotted reference to a
function to import and call (e.g. `msutils.copycol`'s `flavour: python`);
these are non-`"binary"` flavours.

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
`shinobi.loaders.cultcargo` warns when it sees the key rather than silently
producing a possibly-incomplete schema (a cab relying solely on
`dynamic_schema` with no static `inputs:`/`outputs:` loads empty).

The same boundary extends to "never import a cab package": resolving
cult-cargo's package-scoped `_include` form would normally mean importing
the named package (`importlib`) to find its data directory, but that risks
executing arbitrary code from *any* `__init__.py` on the path. Instead,
callers pass an explicit `package_roots={"cultcargo": Path(...)}` mapping
into `load_file()`/`loads()`, and a dotted name is resolved against the
longest registered prefix as a plain filesystem lookup — never through
Python's import machinery.

### Backends never shell out through a shell

Backends invoke commands with **list-form** `subprocess.run` — never
`shell=True`, never string interpolation into a shell.

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
