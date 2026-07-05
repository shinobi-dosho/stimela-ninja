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

The threat model and defensive design are documented in
[`AGENTS.md`](AGENTS.md) — see the *"Never eval()/exec() a cab's `command`"*
section. In short:

- **Cab definitions are treated as untrusted content.** cult-cargo / stimela-
  classic YAML can come from anywhere, so it is never `eval`/`exec`-ed.
- Only the `binary` cab flavour is executed; other flavours are rejected by
  `build_argv()` with `UnsupportedFlavourError` *before* any argv is built.
- Backends invoke commands with **list-form** `subprocess.run` — never
  `shell=True`, never string interpolation into a shell.
- Compiled offload scripts embed exec-form argv only (via `shlex.join`) with
  charset-validated job names.

If you find a way around any of these guarantees, it's a security issue — please
report it as above.
