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
.venv/bin/ruff check .
.venv/bin/ruff format --check .

# enable the repo's pre-commit hook (once per clone)
git config core.hooksPath .githooks
```

(You can equivalently use `uv run pytest` / `uv run ruff check .`.)

### The lockfile

`uv.lock` **is** committed, and `uv sync` installs exactly what it pins. CI
runs ordinary test and audit jobs with `--locked`, so their versions match
the committed lock. When a build goes red, that keeps "this branch broke it"
and "a dependency broke it" as separate, answerable questions.

The Measurement Set integration job refreshes `msutils[exact-native]` to the
latest release with `uv lock --upgrade-package msutils` before its locked
installs on every run. It also runs daily after the workflow is merged to
`main`; ordinary tests and audits skip scheduled runs. Its installation log
records the resolved runtime, including the installed msutils version, so
failures from a new release can be attributed. To check the latest release
locally:

```bash
uv lock --upgrade-package msutils
uv sync --locked --group dev --group measurement-set
```

Review and commit the resulting lock diff to make that release the new local
baseline. State provenance records the installed msutils version (or the
commit, for a Git install of upstream), and verification and materialization
require the original exact runtime even if a newer release exists.

The floors in `pyproject.toml` still define what a downstream `pip install`
resolves. The lock binds only this repo's checkouts and CI.

Two consequences worth knowing:

- **Change `pyproject.toml`, re-run `uv lock`, commit both.** CI's `--locked`
  rejects the mismatch, and so does the pre-commit hook.
- **Changes to the committed dependency baseline are deliberate**:
  `uv lock --upgrade` (or `--upgrade-package <name>`) produces a reviewable diff.

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
tests** (`tests/test_docker_live.py`, `tests/test_slurm_live.py`) **auto-skip
when the infrastructure isn't present** —
they probe at runtime (`shutil.which(...)` plus a liveness check such as
`docker image inspect` or `sinfo`), so you don't need to set any opt-in flag.
To run them locally, see the docstrings at the top of
each file for the exact setup (including the throwaway Slurm cluster in
`tests/slurm_live/`).

**New features and bug fixes should come with tests.** Follow the existing
layout in `tests/` (flat directory, shared fixtures in `tests/fixtures/`; no
`__init__.py` — the suite runs in importlib mode with namespace packages).

## Code style

- **Lint must be clean**: `ruff check .` should report no errors. Lint the
  whole tree, not just `src tests` — CI does, and scoping it narrower once
  meant `docs/conf.py` and `examples/` went unchecked while the bare command
  failed on commits CI called green.
- Ruff's rule set is **pinned explicitly** in `pyproject.toml`
  (`[tool.ruff.lint] select`), not left at ruff's defaults — those shift
  between releases, which would make "is this tree clean?" a property of
  whichever ruff you happen to have installed. `line-length = 180`, and `E501`
  is ignored because the formatter owns line length. That file also records
  which rule families were deliberately left out, and what each would cost.
- **`ruff format --check .` must pass**: formatting is a CI gate, not an
  optional convenience.
- Use **type hints** and write **docstrings** on public API — they render into
  the Sphinx API reference via autodoc.
- Match the surrounding code's naming, comment density, and idiom.

## Documentation

Docs are built with Sphinx (Furo theme) and hosted on Read the Docs. Build them
locally with:

```bash
uv sync --group docs
uv run sphinx-build -E -n -W --keep-going -b html docs docs/_build/html
```

Please update the docs when you change public API. If you add a documentation
dependency, keep `docs/requirements.txt` in sync with the `docs` dependency
group in `pyproject.toml` (Read the Docs installs from the former).

### Cab ownership and paired changes

From 0.2.0b1, Dosho owns generic parameter schemas/validation, policies,
patterns and argv formatting. Shinobi owns framework types, DAG wiring,
execution, filesystem enforcement, dataset lifecycle and caches. Put shared
schema algorithms in Dosho; Shinobi adapters supply trusted dtype bindings
and validate inert framework declarations.

Shinobi requires `dosho[validation]>=0.2.0b1,<0.3` and Pydantic >=2.14.
Its compatibility CI checks both the locked Dosho release and fresh Dosho main.
For coordinated changes, dispatch `.github/workflows/cab-candidate-pair.yml`
with an exact `dosho-ref`; it builds explicit wheels for both projects and
checks their installed origins before testing. Dosho also runs its full
integration suite and docs against Ninja main automatically, with an optional
`ninja-ref` override for candidate testing. See the maintained
[cab ownership contract](https://stimela-ninja.readthedocs.io/en/latest/design_cab_boundary.html).

## Pull requests

1. Branch off `main` and keep PRs **small and focused** — one logical change per
   PR is much easier to review.
2. Make sure `pytest -q`, `ruff check .` and `ruff format --check .` pass
   locally, and that docs build if you touched public API.
3. Push and open a PR against `main`. Reference any related issue
   (e.g. "Closes #12").
4. **CI must be green.** The `test` job runs the suite, lint and the format
   check across Python 3.11 to 3.14 — that's the merge gate. If an automated
   `review` job posts an AI code-review comment, treat it as advisory. Check
   the tree before acting on a "this doesn't exist / is unused" claim.

### Commit messages

Write clear, descriptive commit messages explaining *why* a change is made. No
formal convention (Conventional Commits, sign-off/DCO, or CLA) is required.

Provenance for an assistant-assisted commit (for example an LLM) goes in a commit trailer as
``Assisted-by: <LLM> <MODEL>`` and never in the PR description — see
[*Attribution: commit trailers yes, PR trailers no*](https://github.com/shinobi-dosho/stimela-ninja/blob/main/AGENTS.md#attribution-commit-trailers-yes-pr-trailers-no).

## Versioning and releases

The project follows [Semantic Versioning](https://semver.org/). **Contributors
don't cut releases** — that's a maintainer task. Releases are tag-driven: the
maintainer bumps `version` in `pyproject.toml` and pushes a `vX.Y.Z` tag, and
the `release.yml` workflow verifies the tag matches the package version, builds,
and publishes to PyPI.

## Repository protection

The active `main` branch ruleset lives in GitHub repository settings.
[`.github/main-ruleset.json`](https://github.com/shinobi-dosho/stimela-ninja/blob/main/.github/main-ruleset.json) records that policy;
committing the file does not apply or update the setting. If the repository is
recreated and has no ruleset, an admin can apply it once from the repository
root:

```bash
gh api -X POST "repos/$(gh repo view --json nameWithOwner --jq .nameWithOwner)/rulesets" \
  --input .github/main-ruleset.json
```

Check for an existing ruleset before running this command, because `POST`
creates another one rather than updating the current rule.

## License

By contributing, you agree that your contributions are licensed under the
project's [Apache License 2.0](https://github.com/shinobi-dosho/stimela-ninja/blob/main/LICENSE).

You are responsible for ensuring that you have the right to submit your
contribution under this licence, including any necessary employer or
third-party permissions. Do not submit confidential, proprietary, or
otherwise restricted material that you are not authorised to release.
The project's open-source terms apply irrespective of contributors'
employment or institutional affiliations. See [NOTICE](https://github.com/shinobi-dosho/stimela-ninja/blob/main/NOTICE).
