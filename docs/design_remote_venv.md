# Provisioning the launcher environment for `ninja run --remote`

**Status:** v1 — **proposed, nothing implemented.** Open questions in §8 need
answers before §9 starts.
**Context:** `ninja run TARGET --remote user@host:/path` (`shinobi.offload.ssh`)

---

## 0. Status

This proposes making `--remote` able to *provision* the Python environment it
launches into, instead of only sourcing one that someone else put there.

The design's load-bearing claim, and the thing most worth attacking:

1. **A provisioned venv can be content-addressed by its inputs, and that is
   enough (§4.1).** The claim is that naming the venv directory after
   `sha256(lockfile ++ interpreter request)` buys idempotence, concurrency
   safety and coexistence *structurally* — with no stamp file, no lock, and no
   "is it up to date?" check anywhere in the launch path. Find a sequence
   where two runs sharing a remote path corrupt each other's environment, or
   where a stale environment is activated under a name that says it is fresh.

Two things this deliberately does **not** claim, because the tree already
declines to claim them and this must not quietly go further:

- It is **not** a pin. `backends/venv.py:14-20` is explicit that a
  `venv_digest` is version-parity only — "identical version lists can sit on
  different compiled C-extensions" — which is why venv steps report *unpinned*
  in the run manifest. Naming a directory by its lockfile hash names the
  *inputs to provisioning*, never the resulting bytes. §4.6 makes the residual
  divergence observable rather than papering over it.
- It does **not** extend to the venv *backend*. `resolve_venv`
  (`backends/venv.py:47`) refuses to create the per-step envs in
  `backend.venv.envs` and says so in its error; that stays true (§4.7).

---

## 1. Problem

`--add-venv` defaults to **on** and expands to a single shell fragment
(`offload/ssh.py:347`):

```
test -f venv/bin/activate && source venv/bin/activate || test -f .venv/bin/activate && source .venv/bin/activate;
```

Three defects, in increasing order of consequence:

1. **It silently no-ops.** If neither directory exists, nothing is sourced and
   `ninja run` resolves against whatever the remote login shell's `PATH`
   happens to hold. The run may work, work differently, or fail with an
   unrelated-looking error; nothing in the launch output distinguishes those.
   `--remote` deliberately skips local input validation (`cli.py`'s `run()`,
   documented in `offload/ssh.py:29-35`), so this is not caught earlier either.

2. **The precedence is wrong.** `A && B || C && D` parses as
   `((A && B) || C) && D`, so the trailing `source .venv/bin/activate` is
   reached whenever the first pair *succeeded*. Verified by running the exact
   fragment under `bash -c`:

   | on disk | actual behaviour |
   |---|---|
   | both `venv/` and `.venv/` | sources **both**, `.venv` last — so `.venv` wins, contradicting the documented `venv/`-first order (`docs/cli.rst:96-97`) |
   | only `venv/` | sources it, then emits `.venv/bin/activate: No such file or directory` into the run log |
   | only `.venv/` | correct |
   | neither | correct (silent) |

3. **There is no way to ask for the environment to exist.** The only remedy is
   to ssh in and build it by hand, which is the step everybody forgets, and
   which has to be repeated whenever the recipe's dependencies move.

## 2. What already exists (the spine)

- **`--remote` transport.** `sync_to_remote` (`offload/ssh.py:282`) rsyncs the
  target plus statically-discovered cab deps under `remote.path`;
  `launch_remote` (`:322`) launches detached via `setsid`, capturing a real pid
  and an exit file. Provisioning slots in between these two, and needs no
  change to either.
- **`venv_digest`** (`backends/venv.py:103`): sha256 of a venv's sorted
  `name==version` list, computed by the venv's *own* interpreter via
  `importlib.metadata` (not `pip freeze` — `uv venv` installs no pip). Already
  returns `None` honestly on any failure. Reusable as-is for §4.6.
- **Content-addressed provisioning, as an established pattern.** `_pin_image`
  (`backends/container.py:436`) resolves `repo:tag` → `repo@sha256:...` so what
  executes is exactly what the manifest records, and every resolution path is
  pull-*free* (`_registry_api_digest` — "no external binary, no image pull";
  skopeo and buildx `imagetools` likewise). Note what this means precisely:
  **ninja never issues a pull.** No `pull` subprocess exists in `src/`. The
  runtime auto-pulls as a side effect of `docker run` / `apptainer exec
  docker://...` (`container.py:85-88`). So the precedent being borrowed is
  *resolve an address, let something else materialize it into a
  content-addressed store* — not *install software on a host*.
- **`.partial`-then-rename staging.** Used by the snapshot writer so a durable
  name never appears over a half-written tree (`design_cache_tiers.md` §9).
  §4.5 reuses the convention verbatim.
- **The project is uv-native**: `uv.lock` is committed (179 KB), CI and
  dependabot both drive uv.

## 3. Assumptions

- The remote has `uv` on `PATH`, or can bootstrap it. Untrue on some clusters;
  §8.1.
- The remote has egress to the index at provisioning time. Frequently untrue on
  compute nodes; §6.2.
- `remote.path` is on a filesystem shared by every node that will run the
  recipe, or the launch node is the run node. `--remote` already assumes this
  for the synced target itself.
- The operator has an account on the remote and is entitled to run code there.
  Provisioning executes build backends; §6.1.

## 4. Design

### 4.1 The venv is named by its provisioning inputs

```
<remote.path>/.shinobi/venvs/<env_id>/
```

where `env_id = sha256(lock bytes ++ b"\0" ++ python_request ++ b"\0" ++ mode)`,
truncated to 16 hex chars for a readable path. `python_request` is the
interpreter constraint passed to `uv venv --python` (or the empty string);
`mode` distinguishes `uv sync --frozen` from `uv pip sync`, since the same
`requirements.txt` means different things to each.

Everything else follows from this one choice:

- **Idempotence is structural.** The directory either exists (nothing to do) or
  does not (provision it). There is no stamp file to write, no "compare hash to
  recorded hash" step, and therefore no way for the recorded hash and the
  directory to disagree.
- **Concurrent runs converge** rather than race. Two launches with the same lock
  target the same path and (via §4.5) at most one publishes it; today, two
  launches syncing into `<remote.path>/.venv` would interleave writes into one
  directory.
- **Environments coexist.** Two recipes under one `remote.path` with different
  locks get different `env_id`s. Rollback is re-running an older revision —
  its lock hashes to a directory that is still there.
- **The name is honest.** It asserts "this was built from *these* inputs", which
  is checkable, rather than "this equals your local environment", which is not.

Hashing the lock *bytes* rather than a parse means a whitespace-only change
re-provisions unnecessarily. Accepted: false re-provisioning costs time and
disk, never correctness, and lockfiles are machine-written.

### 4.2 Provisioning inputs are user-supplied, never introspected

The lockfile is **found, not generated**. Resolution order:

1. `--venv-lock PATH` if given (rsynced alongside the target).
2. Otherwise, beside the target file, in order: `uv.lock` (with its
   `pyproject.toml`, which `uv sync` requires), then `requirements.txt`.
3. Otherwise → **error under `sync`**, naming what it looked for.

Under mode `uv.lock` the remote runs `uv sync --frozen`; under
`requirements.txt`, `uv venv && uv pip sync requirements.txt`. `--frozen` is
not optional: it forbids re-resolution, so the remote cannot silently drift to
a different version set than the lock names.

There is deliberately **no** `pip freeze`-of-the-local-venv path. A freeze of a
working development environment carries editable installs and absolute local
paths (`-e /home/<user>/...`) that do not exist remotely; it would produce
either a hard failure or, worse, a partially-satisfied environment. §7.1.

### 4.3 The flag surface

`--add-venv/--no-add-venv` becomes a three-valued `--venv`:

| value | behaviour |
|---|---|
| `off` | source nothing (today's `--no-add-venv`) |
| `use` *(default)* | activate `<remote.path>/.shinobi/venvs/<env_id>` if it exists; else legacy `venv/`, then `.venv/`; else **warn loudly and continue** |
| `sync` | provision `<env_id>` if absent, then activate it |

`use` stays the default so no existing invocation changes behaviour — except
that the silent no-op of §1.1 becomes a warning naming every path tried, and
the §1.2 precedence bug is gone (see §4.4). `--add-venv` / `--no-add-venv` are
kept as hidden aliases for `use` / `off` for one release, then dropped; the
package is `0.1.0b4`, so this is cheap.

### 4.4 Activation replaces the fragment outright

```sh
if [ -f "$VENV/bin/activate" ]; then . "$VENV/bin/activate"
elif [ -f venv/bin/activate ]; then . venv/bin/activate
elif [ -f .venv/bin/activate ]; then . .venv/bin/activate
else echo "ninja: no venv found (tried ...)" >&2
fi
```

An `if/elif` chain, not a `&&`/`||` expression — exactly one branch runs, which
is what §1.2 shows the operator-precedence form does not guarantee. The
existing no-subshell constraint still applies and its comment
(`offload/ssh.py:341-346`) must survive the edit: the `source` has to land in
the same shell that later runs `ninja run`, so nothing here may be wrapped in
`( )`.

### 4.5 Publication is atomic

Provision into `.shinobi/venvs/.partial-<uuid>/`, then `rename()` to
`<env_id>/`. `rename` onto an existing directory fails, which is the correct
outcome — a concurrent launch won already, and its result is by construction
equivalent. The loser removes its staging directory and proceeds.

A venv is **not** relocatable: `uv venv` writes the absolute target path into
`pyvenv.cfg` and the `bin/` shebangs. So the staging directory must be created
with the final path already known — build at `.partial-<uuid>`, then
`uv venv --relocatable`, or simply accept that the rename invalidates the
shebangs unless `--relocatable` is used. **This needs verifying against the
installed uv before §9 starts** (§8.3): if `--relocatable` proves insufficient,
the fallback is an exclusive `flock` on `.shinobi/venvs/<env_id>.lock` with
in-place construction, which is strictly worse but well-understood.

### 4.6 Divergence is measured, not assumed

After provisioning (and on every `sync` launch, since it is one ssh round-trip),
run `_FREEZE_CODE` (`backends/venv.py:44`) under the remote venv's interpreter
and compare the resulting `venv_digest` to the local one. On mismatch, emit a
warning naming both digests.

They will legitimately differ — different platform, different wheels, and
`--frozen` pins versions rather than artifacts. That is the point: the digest
turns "the environments are probably the same" into an observation the operator
can act on. It is recorded in the launch output and, if a manifest is written,
alongside the existing `venv_digest` field — as informational, never as a pin
(§0).

### 4.7 Out of scope by construction

The venv **backend**'s per-step environments (`backend.venv.envs`,
`backend.venv.default`) are untouched. `resolve_venv` keeps raising "this
backend does not create venvs; provision it first". This design provisions
exactly one thing: the environment the *launcher* runs in, so that `ninja run`
and the recipe's own imports resolve. Extending it to per-step envs would make
ninja a configuration-management tool for arbitrary remote environments, which
is the job containers already do and the reason the venv backend documents
itself as their *complement* (`backends/venv.py:5-8`).

## 5. Invariants

1. A directory under `.shinobi/venvs/<env_id>` was built from inputs hashing to
   `env_id`, or does not exist. There is no partially-published state.
2. `env_id` names provisioning *inputs*. Nothing anywhere may read it as a
   statement about the installed bytes.
3. `use` never writes to the remote. Only `sync` provisions.
4. Exactly one activation branch executes (§4.4).
5. No venv is ever built from an introspection of the local environment (§4.2).
6. A failure to provision fails the launch loudly; it never degrades to
   launching into an unknown environment.

## 6. Known weaknesses

1. **Provisioning executes code.** `uv sync` runs build backends for any sdist
   in the lock; `docker pull` executes nothing at pull time. This is the one
   place the container analogy genuinely does not hold. Mitigated by uv being
   wheel-first and by the operator's own account being the blast radius, but it
   is a real difference and should be stated in the docs, not buried.
2. **Compute nodes often have no egress.** Then `sync` fails at the worst
   moment. Partly why `use` stays the default; a documented pattern of
   provisioning once from the login node is the practical answer.
3. **Unbounded disk growth.** Every distinct lock leaves a venv behind forever.
   No GC in v1; §8.2.
4. **Lock parity is not binary parity** (§0). Ineliminable; §4.6 makes it
   visible.
5. **`env_id` does not cover the remote's own inputs** — a different remote
   interpreter patch version, or a different platform, yields the same
   `env_id` with different contents. This is correct (the id names inputs, not
   outputs) but is exactly the trap invariant 2 exists to guard.

## 7. Alternatives considered and rejected

1. **Replicate the local venv (`pip freeze` → install).** Rejected: not
   reproducible (editable installs, local paths), and it advertises an
   "identical environment" the tree explicitly refuses to claim (§0).
2. **One `.venv` plus a stamp file holding the lock hash.** The obvious
   design, and it was the first proposal here. Rejected: the stamp can
   disagree with the directory (crash between write and install, manual edit),
   two concurrent launches interleave into one directory with no winner, and
   only one environment can exist per path. Content-addressing gets all three
   for free by making the path *be* the stamp.
3. **Require a container for remote runs.** Rejected: the venv backend exists
   precisely because the pip-installable half of a pipeline (quartical,
   tricolour, breizorro) should not need a container runtime on every host.
4. **Ship a wheel of the recipe's environment.** Rejected: no lock, no
   resolution, and it re-invents what uv already does.

## 8. Open questions

1. **uv bootstrap.** If `uv` is absent remotely, do we install it
   (`curl … | sh` — a supply-chain step this project would not otherwise take),
   fall back to `python -m venv` + `pip install -r`, or refuse with
   instructions? *Leaning: refuse with instructions in v1.*
2. **GC.** `ninja remote gc user@host:/path`, an age policy, or nothing?
   *Leaning: nothing in v1, documented as a known cost (§6.3).*
3. **Relocatability** (§4.5) — needs an experiment against the installed uv
   before the staging design is fixed. **Blocks §9 step 2.**
4. **The dosho/simms problem, and it is the big one.** `pyproject.toml:33-43`
   documents that dosho and simms must be installed *manually*, outside the
   lock, because locking them breaks `uv lock` for everyone. So a lock-driven
   provisioner reproduces ninja and its six dependencies — *not* the
   environment a caracal recipe actually needs. Either (a) `--venv-lock` points
   at a **recipe-side** lock the user maintains (which is what §4.2 assumes,
   and which means this feature is only useful to people who keep one), or
   (b) a post-sync hook runs extra `uv pip install --no-deps` lines declared
   somewhere, which reintroduces unlocked inputs and weakens `env_id` to a
   partial description. *No recommendation yet — this decides whether the
   feature is worth building at all.*

## 9. Implementation order

1. **Fix §1.2 standalone.** Replace the `&&`/`||` fragment with the `if/elif`
   chain, add the not-found warning, add a test asserting exactly one branch
   runs for each of the four on-disk shapes. Independently valuable, no new
   surface, ships regardless of what §8.4 decides. *Do this first, separately.*
2. **`env_id` + staging.** Resolve §8.3, then `_env_id()`, the
   `.partial-<uuid>` → `rename` publisher, and unit tests over the hash inputs.
3. **`--venv` flag** with `use`/`off` only, plus the hidden aliases. Still no
   provisioning; pure surface change, keeps behaviour identical.
4. **`sync` mode**: lock discovery (§4.2), rsync of the lock pair, remote
   `uv sync --frozen`, failure surfacing.
5. **§4.6 digest comparison.**
6. **Docs**: `docs/cli.rst:93-104` and `docs/offloading.rst`; state §6.1
   plainly.

Steps 2–5 are all gated on §8.4. Step 1 is not, and should not wait for it.
