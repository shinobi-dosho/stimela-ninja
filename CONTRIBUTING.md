# Contributing to stimela-ninja

Thanks for your interest in contributing! **stimela-ninja** (import name
`shinobi`, CLI `ninja`) is a simple but flexible framework for reproducible
radio-astronomy pipelines — Stimela 3.0. It's early-beta software, so the most
valuable contributions right now are bug reports, focused fixes, tests,
documentation, and feedback on the design.

By participating you agree to abide by our
[Code of Conduct](https://github.com/shinobi-dosho/stimela-ninja/blob/main/CODE_OF_CONDUCT.md).

## Scope and philosophy

Our guiding principle is to **keep things simple and robust, while staying
flexible and effective**. We favour small, focused changes and prefer solving
problems with plain Python over adding new layers of machinery.

For background on how the project is put together and the reasoning behind its
current design, see
**[`AGENTS.md`](https://github.com/shinobi-dosho/stimela-ninja/blob/main/AGENTS.md)**.
It's helpful context when proposing anything that touches the recipe/
orchestration layer. If you're considering a larger change, opening an issue to
discuss it first is a great way to align before writing code.

## Ways to contribute

- **Report bugs** and request features via [issues](https://github.com/shinobi-dosho/stimela-ninja/issues).
- **Improve documentation** under `docs/` or the docstrings that feed the API
  reference.
- **Submit code** — bug fixes, new cabs/loaders/backends, tests.
- **Add examples** under `examples/`.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
.venv/bin/pytest
.venv/bin/ruff check src tests

# enable the repo's pre-commit hook (once per clone)
git config core.hooksPath .githooks
```

(You can equivalently use `uv run pytest` / `uv run ruff check src tests`.)

### The lockfile

`uv.lock` **is** committed, and `uv sync` installs exactly what it pins. CI
runs every job with `--locked`, so the versions the matrix tests are the
versions you ran locally — not whatever PyPI published that morning. When a
build goes red, that keeps "this branch broke it" and "a dependency broke it"
as separate, answerable questions.

The floors in `pyproject.toml` still define what a downstream `pip install`
resolves. The lock binds only this repo's checkouts and CI.

Two consequences worth knowing:

- **Change `pyproject.toml`, re-run `uv lock`, commit both.** CI's `--locked`
  rejects the mismatch, and so does the pre-commit hook.
- **Dependency upgrades are a deliberate commit**, not a side effect of the
  clock: `uv lock --upgrade` (or `--upgrade-package <name>`) produces a
  reviewable diff.

Dependabot (`.github/dependabot.yaml`) opens those upgrade PRs weekly, updating
`pyproject.toml` and `uv.lock` together so they pass the `--locked` jobs.
Dev-tooling and runtime minor/patch bumps arrive batched; a runtime major gets
its own PR on purpose. Security updates are enabled repo-side and are not
batched or scheduled — those PRs arrive when an advisory does.

### The pre-commit hook

Enabling it is the only setup step that is not `uv`'s job — git will not let a
repository turn on an executable hook by itself, which is why the `git config`
above is manual; skip it and you simply get no hook.

`.githooks/pre-commit` is tracked in the repo, and does two things when (and
only when) the commit touches `pyproject.toml` or `uv.lock`:

1. checks the lock is in sync with `pyproject.toml`;
2. runs `pip-audit` over the locked runtime dependencies.

Any other commit skips it in milliseconds — pip-audit is a network round trip
per package, and it cannot tell you anything new about a dependency set you
have not touched. It installs nothing and pins nothing: pip-audit comes from
the `audit` dependency group via `uv run`, so it and CI's `audit` job check the
same versions with the same commands.

Run that audit by hand any time:

```bash
uv export --frozen --no-emit-project --no-default-groups --no-hashes --format requirements-txt |
    uv run --group audit pip-audit --no-deps -r /dev/stdin
```

A finding is usually fixed by `uv lock --upgrade-package <name>`, raising the
floor in `pyproject.toml` too if the vulnerable range is one a downstream
install could still land on. If there is no fixed release and the advisory does
not apply here, pass `--ignore-vuln <ID>` and record why in `pyproject.toml`
alongside the group. `git commit --no-verify` bypasses the hook when you
genuinely need to.

## Testing

Run the suite with:

```bash
pytest -q
```

Most tests are unit/mocked and always run. The **live-backend integration
tests** (`tests/test_docker_live.py`, `tests/test_kubernetes_live.py`,
`tests/test_slurm_live.py`) **auto-skip when the infrastructure isn't present** —
they probe at runtime (`shutil.which(...)` plus a liveness check such as
`docker image inspect`, `kubectl cluster-info`, or `sinfo`), so you don't need
to set any opt-in flag. To run them locally, see the docstrings at the top of
each file for the exact setup (a `kind` cluster for Kubernetes; the throwaway
Slurm cluster in `tests/slurm_live/` for Slurm).

**New features and bug fixes should come with tests.** Follow the existing
layout in `tests/` (flat directory, shared fixtures in `tests/fixtures/`; no
`__init__.py` — the suite runs in importlib mode with namespace packages).

## Code style

- **Lint must be clean**: `ruff check src tests` should report no errors. Ruff
  runs with its default rule set at `line-length = 100` (see `pyproject.toml`).
- `ruff format` is available and uses the same line width if you'd like
  autoformatting.
- Use **type hints** and write **docstrings** on public API — they render into
  the Sphinx API reference via autodoc.
- Match the surrounding code's naming, comment density, and idiom.

## Documentation

Docs are built with Sphinx (Furo theme) and hosted on Read the Docs. Build them
locally with:

```bash
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html
```

Please update the docs when you change public API. If you add a documentation
dependency, keep `docs/requirements.txt` in sync with the `docs` dependency
group in `pyproject.toml` (Read the Docs installs from the former).

## Pull requests

1. Branch off `main` and keep PRs **small and focused** — one logical change per
   PR is much easier to review.
2. Make sure `pytest -q` and `ruff check src tests` pass locally, and that docs
   build if you touched public API.
3. Push and open a PR against `main`. Reference any related issue
   (e.g. "Closes #12").
4. **CI must be green.** The `test` job runs the suite and lint across Python
   3.10, 3.11 and 3.12 — that's the merge gate. An automated `review` job also
   posts an AI code-review comment; treat it as **advisory, not a gate**, and
   weigh its findings with judgement. Per `AGENTS.md`'s *"Reviewing changes:
   check the tree, not just the diff"*, verify any "this doesn't exist / is
   unused" claim against the actual tree before acting on it.

### Commit messages

Write clear, descriptive commit messages explaining *why* a change is made. No
formal convention (Conventional Commits, sign-off/DCO, or CLA) is required.

## Versioning and releases

The project follows [Semantic Versioning](https://semver.org/). **Contributors
don't cut releases** — that's a maintainer task. Releases are tag-driven: the
maintainer bumps `version` in `pyproject.toml` and pushes a `vX.Y.Z` tag, and
the `release.yml` workflow verifies the tag matches the package version, builds,
and publishes to PyPI.

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](https://github.com/shinobi-dosho/stimela-ninja/blob/main/LICENSE).
