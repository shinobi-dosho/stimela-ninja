# Provisioning the launcher environment for `ninja run --remote`

**Status:** v3 — **step 1 shipped (#85); steps 2-5 un-gated and unimplemented.**
§8.1, which gated everything past step 1, is closed: option (a).
**Context:** `ninja run TARGET --remote user@host:/path` (`shinobi.offload.ssh`)

**Change log**

- **v2 → v3 (§8.1 answered against the tree).** v2 left the load-bearing
  question — *what lock does a real recipe use?* — open, with the note
  "*needs checking against the actual repo — nobody has*". Somebody has now.
  1. **§9 step 1 shipped, five weeks before this revision was written.** PR #85
     replaced the `&&`/`||` fragment with the `if/elif` chain, the not-found
     warning and the four-shape test; it is `_venv_activation` at
     `offload/ssh.py:327-354`, comment and no-subshell constraint intact. §1's
     defects 1 and 2 are history. Defect 3 — no way to ask for the environment
     to exist — is the whole of what remains.
  2. **§8.1 closes as option (a).** `caracal-pipeline/caracal2` keeps a
     committed `uv.lock` (1969 lines) that locks both git dependencies at
     pinned commits: `dosho` at `07904a9` and `stimela-ninja` at `85fcf8a`,
     both `source = { git = ... }`. So the recipe-side lock v2 hoped for is
     not hypothetical — it already exists in the one downstream repo this
     feature is for. The circularity that made it doubtful is also gone: dosho
     dropped its `[tool.uv.sources]` redirect once `Scope.scratch` shipped in
     0.1.0b5, which is why this repo's own `examples = ["dosho"]` group
     resolves (`pyproject.toml:30-40`).
  3. **Option (b) is rejected outright, not carried as a caveat.** v2 kept it
     alive for simms, on the strength of a comment in this repo's own
     `pyproject.toml` asserting simms "has no docker image, so it runs via
     NativeBackend, and it is not on PyPI". Every clause of that is false for
     simms 3.x: PyPI has 3.0.2, dosho builds a `SIMMS` image from
     `simms==3.0.2` (`dosho/src/dosho/images.yaml:65-74`), and
     `dosho/cabs/simms.py` binds all three sub-commands (`skysim`, `telsim`,
     `primary_beam`) to `image=images.SIMMS`, running them as pysteps *inside*
     that container. The claims belong to `SIMMS_CLASSIC`, the genuinely
     different pre-3.0 binary. With no unlockable package left, `env_id` stays
     a **complete** description of its inputs rather than a partial one, which
     is worth more than the escape hatch was. Moved to §7.6; the stale comments
     are corrected in the same change.
  4. **The general form of the answer, which outlives the specific one.** A
     lock has to describe the *launcher* environment only. Everything a recipe
     **runs** is containerised — that is what the cab layer is for — so a
     package needing a container is not a hole in the lock, it is the lock
     correctly declining to describe something outside its scope. The question
     to ask of a future unlockable dependency is not "does (b) come back?" but
     "why is this not a cab?".
- **v1 → v2 (review round 1).** Independent review attacked v1 against the tree
  and against a live `uv`. v1's central idea (content-addressing) survived; its
  *publisher*, two of its six invariants, and its hash inputs did not. In order
  of severity:
  1. **The publisher was broken for the mode the doc called primary (§4.5).**
     v1 treated relocatability as an open question. It is not: `uv sync` has no
     `--relocatable` flag at all (only `uv venv` does), and a `uv sync --frozen`
     that creates its own venv produces console scripts whose `/bin/sh` shim
     hardcodes the absolute interpreter path — even under
     `UV_VENV_RELOCATABLE=1`. Verified end to end on uv 0.11.21: after a rename
     the script dies with `exec: /…/.venv/bin/python: not found`. Since `ninja`
     *is* a console script, an implementer following v1's §4.2 command list
     would have shipped a publisher that breaks the launcher it exists to
     provide. **Fixed:** §4.4 mandates the two-step
     `uv venv --relocatable` → `uv sync --frozen`, which was verified to keep
     the marker, emit a relative shim, and work after rename.
  2. **`rename()` onto an existing *empty* directory succeeds silently
     (§4.5).** v1's Invariant 1 rested on "rename onto an existing directory
     fails". That holds only for a *non-empty* destination — verified:
     non-empty raises `Directory not empty`, empty is replaced without error.
     A crashed provision or a stray `mkdir` therefore defeats the exclusion.
     **Fixed:** existence is decided by a completion sentinel *inside* the
     directory (§4.3), never by the directory itself, so partial states are
     unreadable rather than merely unlikely.
  3. **Invariant 2 was aspirational text, not a mechanism.** v1 admitted
     `env_id` did not cover platform, then activated on a bare
     `[ -f "$VENV/bin/activate" ]`. Two hosts sharing a networked `remote.path`
     would activate each other's builds under a name asserting a match.
     **Fixed:** the remote platform triple is *in* `env_id` (§4.2), probed in
     an ssh round-trip already being made, and re-checked from the sentinel
     before activation (§4.4).
  4. **`env_id` omitted load-bearing inputs.** No `pyproject.toml` (which v1's
     own §4.3 noted `uv sync` requires) and no extras/groups selection, so two
     genuinely different environments could collide on one id. **Fixed** in
     §4.2.
  5. **Invariant 6 was false as written.** It forbade degrading to an unknown
     environment; `use` mode's documented behaviour is to warn and continue
     into exactly that. **Fixed:** scoped to `sync`.
  6. **§2 overclaimed reuse and precedent.** `venv_digest`
     (`backends/venv.py:104`) shells out to a *local* path and cannot be
     pointed at a remote interpreter — only `_FREEZE_CODE` and the hashing are
     reusable (§4.6). And the container "never pulls" precedent, though
     factually correct, is decoration: `_pin_image` is a read-only HTTP resolve
     sharing no mechanism with staged local provisioning. §2 now says so.
  7. **§8.4's alarm was overstated.** The dosho/simms exclusion is caused by a
     *circular* self-dependency (dosho redirects `stimela-ninja` back into this
     repo via `[tool.uv.sources]`), not by git-sourced packages being
     unlockable in general — so a downstream recipe repo's own lock has no such
     problem. A third option (hashing post-sync install commands into `env_id`)
     was missed entirely. Both folded into §8.1.
  Smaller: the `pyproject.toml` citation was lines 30-42, not 33-43; §4.5's
  `flock` fallback was called "well-understood" when this project's own
  `design_cache_tiers.md:17` documents it silently no-opping on Lustre —
  moot now that the two-step publisher works, and recorded in §7.5 as
  rejected; §9 step 1 never showed the standalone (pre-`env_id`) form of the
  activation chain, now in §4.4; the Click work to alias a boolean flag pair
  onto a choice option is not free and §4.5 no longer implies it is.

---

## 0. Status

This proposes making `--remote` able to *provision* the Python environment it
launches into, instead of only sourcing one that someone else put there.

The load-bearing claim, restated after review, and the thing most worth
attacking next:

1. **A provisioned venv can be content-addressed by its inputs, and a
   completion sentinel is what makes that safe (§4.2, §4.3).** v1 claimed the
   *path* alone bought idempotence, concurrency safety and coexistence. Review
   showed the path is not self-validating — an empty directory at that path
   defeats it. The claim is now narrower: `env_id` names the inputs, and a
   sentinel written last, inside the directory, is the sole evidence that those
   inputs were fully realised. Find a sequence where a venv is activated whose
   sentinel does not describe it, or where two concurrent launches both publish.

Two things this deliberately does **not** claim:

- It is **not** a pin. `backends/venv.py:14-20` is explicit that a
  `venv_digest` is version-parity only — "identical version lists can sit on
  different compiled C-extensions" — which is why venv steps report *unpinned*
  in the run manifest. `env_id` names *inputs to provisioning*, never the
  resulting bytes. §4.6 makes the residual divergence observable.
- It does **not** extend to the venv *backend*. `resolve_venv`
  (`backends/venv.py:47`) refuses to create the per-step envs in
  `backend.venv.envs`; that stays true (§4.7).

---

## 1. Problem

**Defects 1 and 2 below are fixed** — PR #85 shipped §4.4's `if/elif` chain as
`_venv_activation` (`offload/ssh.py:327-354`). They are kept here as written
because they are what §4.4's branch order and no-subshell constraint are a
response to, and a reader who does not know why that fragment looks the way it
does will eventually "simplify" it back. **Defect 3 is what remains, and is the
whole of what §4 proposes.**

As of the revision this section describes, `--add-venv` defaulted to **on** and
expanded to a single shell fragment:

```
test -f venv/bin/activate && source venv/bin/activate || test -f .venv/bin/activate && source .venv/bin/activate;
```

Three defects, in increasing order of consequence:

1. **It silently no-ops.** If neither directory exists, nothing is sourced and
   `ninja run` resolves against whatever the remote login shell's `PATH`
   happens to hold. The run may work, work differently, or fail with an
   unrelated-looking error; nothing in the launch output distinguishes those.
   `--remote` deliberately skips local input validation (documented at
   `offload/ssh.py:29-35`), so this is not caught earlier either.

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
  and an exit file. Provisioning slots between these two and needs no change to
  either. **Note a property this buys:** `--remote` ssh's to `remote.host` and
  `setsid`s there directly — no scheduler in between — so the host that
  provisions is always the host that runs. Heterogeneity is therefore a
  *cross-launch* problem (two hosts, one shared path, different times), not a
  within-launch one. §4.2 handles it on that basis.
- **`_FREEZE_CODE`** (`backends/venv.py:44`): distribution listing via stdlib
  `importlib.metadata`, deliberately not `pip freeze` (`uv venv` installs no
  pip). Reusable over ssh. `venv_digest` itself (`:104`) is **not** — it
  `subprocess.run`s a *local* `venv/bin/python` (`:113-118`), so §4.6 needs a
  refactor, not a call.
- **`.partial`-then-rename staging.** Used by the snapshot writer so a durable
  name never appears over a half-written tree (`design_cache_tiers.md` §9).
  §4.5 borrows the convention, with the empty-destination caveat review found.
- **The project is uv-native**: `uv.lock` is committed (179 KB), CI and
  dependabot both drive uv.
- **Content-addressing as a house pattern** — but not, on inspection, a
  mechanical precedent. `_pin_image` (`backends/container.py:438`) resolves
  `repo:tag` → `repo@sha256:...`, and every resolution path is pull-*free*
  (`_registry_api_digest` — "no external binary, no image pull"; skopeo and
  buildx `imagetools` likewise). It is literally true that **ninja never issues
  a pull**: no `pull` subprocess exists in `src/`; the runtime auto-pulls as a
  side effect of `docker run` / `apptainer exec docker://...`
  (`container.py:85-88`). v1 leaned on this as justification. Review's
  correction is accepted: `_pin_image` is a read-only HTTP resolve that writes
  nothing and executes nothing, sharing no mechanism with staged local
  provisioning. It establishes that *naming things by content is idiomatic
  here*; it does not transfer any safety property. The safety has to come from
  §4.3 and §4.5 on their own merits.

## 3. Assumptions

- The remote has `uv` on `PATH`, or can bootstrap it. Untrue on some clusters;
  §8.2.
- The remote has egress to the index at provisioning time. Frequently untrue on
  compute nodes; §6.2.
- The operator has an account on the remote and is entitled to run code there.
  Provisioning executes build backends; §6.1.

v1 also assumed `remote.path` is on a filesystem shared by every node that will
run the recipe. Dropped: as §2 notes, `--remote` has no scheduler, so provision
host and run host are the same host by construction.

## 4. Design

### 4.1 Shape

```
<remote.path>/.shinobi/venvs/<env_id>/                  # the venv
<remote.path>/.shinobi/venvs/<env_id>/.shinobi-env.json # the sentinel (§4.3)
<remote.path>/.shinobi/venvs/.partial-<uuid>/           # staging (§4.5)
```

### 4.2 `env_id` names every input to provisioning

```
env_id = sha256(
    lock bytes          ++ b"\0" ++
    pyproject bytes     ++ b"\0" ++   # uv sync requires it; it selects content
    extras/groups spec  ++ b"\0" ++   # --extra/--group, canonically sorted
    python_request      ++ b"\0" ++   # the --python constraint, or ""
    mode                ++ b"\0" ++   # uv-sync | uv-pip-sync: same file, different meaning
    platform_triple                   # from the remote, see below
)[:16]
```

`platform_triple` is `<uname -m>/<libc id+version>/<python X.Y.Z>`, read off the
remote in the same ssh round-trip that tests the sentinel. Folding it in is what
makes Invariant 2 enforceable rather than aspirational: two hosts of different
architecture sharing one `remote.path` compute *different* `env_id`s and cannot
collide. It costs nothing — the round-trip is already being made — and it is the
one input that cannot be known locally.

What this buys, and what it does not:

- **Idempotence**: the sentinel is present (nothing to do) or absent (provision).
  No stamp file to write, no freshness comparison, nothing that can disagree.
- **Coexistence**: different locks, extras, or hosts get different directories.
  Rollback is re-running an older revision, whose lock hashes to a directory
  that is still there.
- **It does not describe the result.** Identical inputs on identical platforms
  can still yield different compiled artifacts. That is §4.6's job, and
  Invariant 2's.

Hashing lock *bytes* rather than a parse means a whitespace-only change
re-provisions unnecessarily. Accepted: false re-provisioning costs time and
disk, never correctness, and lockfiles are machine-written.

### 4.3 The sentinel is the only evidence of completion

Written **last**, inside the directory, after provisioning succeeds:

```json
{"schema": 1,
 "env_id": "...", "lock_sha256": "...", "pyproject_sha256": "...",
 "extras": [...], "python_request": "...", "mode": "uv-sync",
 "platform_triple": "x86_64/glibc-2.39/3.11.9",
 "venv_digest": "sha256:...", "created": "<iso8601>", "uv_version": "..."}
```

Every existence question is asked of this file, never of the directory:

- **Activation** (§4.4) requires the sentinel to parse, its `schema` to be one
  this client knows, *and* its `platform_triple` to match the current host. A
  directory with no sentinel is treated as absent.
- **`sync`** provisions when the sentinel is missing, regardless of what else is
  at that path. A sentinel that parses but carries an unknown `schema` is
  **not** missing — it is a newer client's environment, and `sync` refuses
  rather than provisioning over it (§8.4, Invariant 8).

This is what closes the `rename`-onto-empty hole: an empty or half-populated
`<env_id>/` is unreadable to every consumer, so the exclusion no longer depends
on `rename` refusing it. It also makes §4.6 free — the digest computed at
provisioning time is recorded once rather than recomputed per launch.

### 4.4 Provisioning, and activation

**Provisioning is two steps, in this order, and the order is load-bearing:**

```sh
uv venv --relocatable "$STAGING"                     # 1. relocatable marker
VIRTUAL_ENV="$STAGING" uv sync --frozen --active     # 2. populate it
```

Verified on uv 0.11.21. Doing it the obvious way instead — letting
`uv sync --frozen` create its own venv — produces a venv with **no**
`relocatable` marker in `pyvenv.cfg` and console scripts whose `/bin/sh` shim
hardcodes the absolute interpreter path, *even with `UV_VENV_RELOCATABLE=1`
set*; `uv sync` has no `--relocatable` flag at all. After the §4.5 rename such a
script fails with `exec: /…/.venv/bin/python: not found`. Since `ninja` is
itself a console script, that is the launcher failing to launch. The two-step
form keeps the marker through `uv sync`, emits a relative shim, and was
confirmed to still run after being renamed.

`--frozen` is not optional: it forbids re-resolution, so the remote cannot
silently drift to a version set the lock does not name. For `requirements.txt`
(mode `uv-pip-sync`) the sequence is `uv venv --relocatable "$STAGING"` then
`uv pip sync --python "$STAGING/bin/python" requirements.txt`.

**Activation replaces the §1 fragment outright.** Step 1 of §9 shipped the
standalone form, before `env_id` exists — this is now the tree's
`_venv_activation` (`offload/ssh.py:349-354`), modulo the `; ` separators an
`ssh` one-liner needs:

```sh
if   [ -f venv/bin/activate ];  then . venv/bin/activate
elif [ -f .venv/bin/activate ]; then . .venv/bin/activate
else echo "ninja: no venv found under <path> (tried venv/, .venv/)" >&2
fi
```

and step 3 extends it with the resolved `$VENV` branch ahead of the legacy two.
An `if/elif` chain, not a `&&`/`||` expression — exactly one branch runs, which
§1.2 shows the operator-precedence form does not guarantee. The existing
no-subshell constraint (`offload/ssh.py:341-346`) is preserved and its comment
must survive the edit: the `source` has to land in the same shell that later
runs `ninja run`, so nothing here may be wrapped in `( )`. (`if/then/fi` does
not fork, so this is safe — confirmed.)

### 4.5 Publication

Provision into `.shinobi/venvs/.partial-<uuid>/`, write the sentinel, then
`rename()` to `<env_id>/`.

`rename` onto a **non-empty** directory fails (`Directory not empty`), which is
the desired outcome — a concurrent launch won, and its result is by construction
equivalent. It does **not** fail onto an *empty* directory; that case is
replaced silently. Verified both ways. This is why §4.3 exists: correctness
rests on the sentinel, and `rename`'s behaviour is only an optimisation that
usually avoids the redundant work.

If `rename` fails, re-test the sentinel. Present → adopt it. Absent → a
directory exists at the final path with no sentinel, i.e. a previous provision
died between `mkdir` and completion. **Refuse loudly**, naming the path and
saying to remove it, rather than auto-clobbering: that path may be an
environment someone built by hand, and deleting it unasked is not a decision
this tool should make.

The staging directory is removed on any failure.

### 4.6 Divergence is measured, not assumed

At provisioning time, run `_FREEZE_CODE` (`backends/venv.py:44`) under the new
venv's interpreter over ssh, hash the sorted result, and record it in the
sentinel (§4.3). Compare against the local `venv_digest` when one exists; on
mismatch, warn naming both.

This needs a small refactor, not a call: `venv_digest` (`:104`) hardcodes
`subprocess.run` against a local `venv/bin/python` (`:113-118`) and cannot
address a remote interpreter. Split it into a pure `digest_of_dists(list[str])`
plus the existing local collector, and let the ssh path supply its own dists.
The `None`-on-any-failure contract ("an honest null — never a fabricated
digest") carries over unchanged.

Digests will legitimately differ across platforms — that is the point. The
comparison turns "the environments are probably the same" into an observation.
It is informational, never a pin (§0). Where the launching machine has no
comparable local environment — which is often *why* someone is provisioning
remotely — the comparison is simply skipped and the remote digest recorded
alone.

### 4.7 The flag surface

`--add-venv/--no-add-venv` becomes a three-valued `--venv`:

| value | behaviour |
|---|---|
| `off` | source nothing (today's `--no-add-venv`) |
| `use` *(default)* | activate `<env_id>` if its sentinel matches; else legacy `venv/`, then `.venv/`; else **warn loudly and continue** |
| `sync` | provision `<env_id>` if its sentinel is absent, then activate it |

`use` stays the default, so no existing invocation changes behaviour except that
§1.1's silent no-op becomes a warning naming every path tried and §1.2's
precedence bug is gone. `--add-venv`/`--no-add-venv` are kept as hidden aliases
for `use`/`off` for one release, then dropped; the package is `0.1.0b4`, so this
is cheap. Click has no native way to alias a boolean flag pair onto values of an
unrelated choice option, so this needs a small callback or a custom
`click.Option` — minor, but not free.

### 4.8 Out of scope by construction

The venv **backend**'s per-step environments (`backend.venv.envs`,
`backend.venv.default`) are untouched. `resolve_venv` keeps raising "this
backend does not create venvs; provision it first". This design provisions
exactly one thing: the environment the *launcher* runs in. Extending it to
per-step envs would make ninja a configuration-management tool for arbitrary
remote environments, which is what containers already do and the reason the venv
backend documents itself as their *complement* (`backends/venv.py:5-8`).

## 5. Invariants

1. A directory under `.shinobi/venvs/<env_id>` is *usable* iff it holds a
   parseable sentinel; nothing else may be read as evidence of completion. No
   partial state is observable.
2. `env_id` names provisioning **inputs**, including the remote platform.
   Nothing may read it as a statement about installed bytes; §4.6 is the only
   thing entitled to speak about those.
3. Activation refuses a sentinel whose `platform_triple` does not match the
   current host.
4. `use` never writes to the remote. Only `sync` provisions.
5. Exactly one activation branch executes (§4.4).
6. No venv is ever built from an introspection of the local environment (§7.1).
7. **A `sync` that cannot provision fails the launch loudly.** It never falls
   back to legacy paths and never launches into an unresolved environment. This
   is scoped to `sync` deliberately: `use` is permitted to warn and continue,
   which is the pre-existing behaviour §1.1 is improving, not removing.
8. A pre-existing directory at a final path is never deleted or overwritten by
   this feature (§4.5).

## 6. Known weaknesses

1. **Provisioning executes code.** `uv sync` runs build backends for any sdist
   in the lock; `docker pull` executes nothing at pull time. This is the one
   place the container comparison genuinely does not hold, and it should be
   stated in the user docs, not buried. Mitigated by uv being wheel-first and by
   the operator's own account being the blast radius.
2. **Compute nodes often have no egress.** Then `sync` fails at the worst
   moment. Partly why `use` remains the default; provisioning once from the
   login node is the practical pattern and should be documented as such.
3. **Unbounded disk growth.** Every distinct input set leaves a venv behind
   forever. No GC in v1; §8.3.
4. **Lock parity is not binary parity** (§0). Ineliminable; §4.6 makes it
   visible rather than pretending otherwise.
5. **`platform_triple` is a heuristic identity.** Same arch, same libc version,
   same interpreter, different CPU capabilities (AVX-512) or different BLAS —
   the same `env_id`, legitimately different performance and occasionally
   different results. Invariant 2 is the guard; §4.6 is the detector.
6. **Networked-filesystem rename semantics are weaker than local.** §4.3 is
   designed so correctness does not depend on them, but the `.partial` precedent
   borrowed from `design_cache_tiers.md` was established for in-process,
   same-local-filesystem clones and does not transfer its guarantees to a
   directory of many small files on Lustre or NFS.

## 7. Alternatives considered and rejected

1. **Replicate the local venv (`pip freeze` → install).** Rejected: not
   reproducible (editable installs, absolute local paths that do not exist
   remotely), and it advertises an "identical environment" the tree explicitly
   refuses to claim (§0). This was the original framing of the request.
2. **One `.venv` plus a stamp file holding the lock hash.** Rejected, but the
   margin is narrower than v1 claimed. Content-addressing does *not* get safety
   for free — review showed the bare path is not self-validating, and §4.3 has
   to reintroduce a written record to close the gap. What content-addressing
   still buys over a stamp file, and what decides it: multiple environments
   coexist under one `remote.path`, and a sentinel is scoped to the directory it
   describes, so it cannot be stale *about a different environment* the way a
   single path-level stamp can.
3. **Require a container for remote runs.** Rejected: the venv backend exists
   precisely because the pip-installable half of a pipeline (quartical,
   tricolour, breizorro) should not need a container runtime on every host.
4. **Ship a wheel of the recipe's environment.** Rejected: no lock, no
   resolution, re-invents what uv already does.
5. **`flock` on `<env_id>.lock` with in-place construction.** v1's fallback for
   a relocatability failure that turned out not to exist (§4.4). Rejected on its
   own merits regardless: `design_cache_tiers.md:17` documents `flock` silently
   no-opping on Lustre without the mount option and varying by NFS protocol
   version, and `docs/cli.rst:103`'s own canonical example is
   `user@cluster:/scratch/run1`. A lock that silently does nothing on the target
   filesystem is worse than the sentinel, which needs no locking.
6. **Post-sync unlocked installs, hashed into `env_id`.** v2's option (b) in
   §8.1, kept alive solely because one package (simms) was believed unlockable.
   It is not (§8.1), so this buys nothing and costs the property that makes
   §4.2 worth having: `env_id` would name a *command text* whose effect is
   unpinned, and Invariant 2 would then be guarding a description that is
   complete about inputs but silent about half of what those inputs do. The
   design is better off refusing to provision what it cannot name. If a future
   dependency genuinely cannot be locked, the first question is why it is not a
   cab — everything a recipe *runs* is containerised, and only the launcher
   itself has to come out of a lock.

## 8. Open questions

1. **What lock does a real recipe use? — ANSWERED: option (a), a recipe-side
   lock, which already exists.** This gated §9 steps 2-5; it no longer does.
   `caracal-pipeline/caracal2` — the downstream repo this feature exists to
   serve — keeps a committed `uv.lock` of 1969 lines that locks both git
   dependencies at pinned commits (`dosho` at `07904a9`, `stimela-ninja` at
   `85fcf8a`, each `source = { git = ... }`). Nothing about it is aspirational.
   Three things v2 got wrong, in descending order of consequence:
   - **The circularity is gone.** dosho dropped its `[tool.uv.sources]`
     redirect of `stimela-ninja` once `Scope.scratch` shipped in 0.1.0b5
     (dosho #42). That is precisely why this repo's own `examples = ["dosho"]`
     dependency group resolves today (`pyproject.toml:30-40`), and it is why a
     downstream lock covering dosho was never the problem v2 thought it might
     still be.
   - **simms is not an exception.** v2 preserved option (b) for it, on the
     strength of this repo's own comment claiming simms has no image, runs
     native, and is not on PyPI. PyPI serves simms 3.0.2; dosho builds a
     `SIMMS` image from `simms==3.0.2` (`images.yaml:65-74`); and
     `dosho/cabs/simms.py` binds `skysim`/`telsim`/`primary_beam` to
     `image=images.SIMMS`, executing them as pysteps inside it. The comment
     describes `SIMMS_CLASSIC`, a different tool.
   - **The scope of a lock was drawn too wide.** A launcher lock never had to
     describe what a recipe *runs*, only what runs `ninja`. Container images
     cover the rest, and pin harder than a lock does. So (b) is rejected
     (§7.6) and (c) does not arise.
2. **uv bootstrap — DECIDED: refuse with instructions.** If `uv` is absent
   remotely, `sync` fails naming the host and printing the one-line install
   command, and does not run it. Piping a remote script into a shell on the
   operator's account is a supply-chain step this project takes nowhere else,
   and it would be taken silently, inside a launch the operator is watching for
   *pipeline* failures. Falling back to `python -m venv` + `pip install -r` is
   worse than refusing: it drops `--frozen`, so the environment it builds is
   the one thing §4.4 exists to prevent — a re-resolved version set the lock
   does not name, wearing an `env_id` that asserts otherwise.
3. **GC — DECIDED: nothing, documented.** Every distinct input set leaves a
   venv behind forever (§6.3). No `ninja remote gc`, no age policy: deleting
   directories over ssh on the operator's behalf is the same class of decision
   §4.5 already refuses to make, and the disk cost is bounded in practice by
   how often a lock changes. `rm -rf` under `.shinobi/venvs/` is a documented
   operator action, and the content-addressed layout is what makes it safe to
   perform by hand — every directory names exactly what it is.
4. **Sentinel schema versioning — DECIDED: include it.** `"schema": 1` as the
   first key. One key now; an unreadable-sentinel migration otherwise. A
   sentinel whose `schema` is unrecognised reads as *absent* under `use` (so an
   old client degrades to the legacy paths rather than misreading a newer
   sentinel's fields) and as a refusal under `sync` (Invariant 8 — never
   provision over something a newer client owns).

## 9. Implementation order

1. ~~**Fix §1.2 standalone.**~~ **DONE — PR #85.** The `&&`/`||` fragment is
   gone; `_venv_activation` (`offload/ssh.py:327-354`) is the §4.4 chain, with
   the not-found warning and a test over all four on-disk shapes.
2. **`env_id` + sentinel.** `_env_id()` over the §4.2 inputs, the platform
   probe, sentinel read/write, and unit tests over hash inputs — including that
   changing *each* input moves the id, and that an empty directory reads as
   absent.
3. **`--venv` flag** with `use`/`off` only, plus the hidden aliases and the
   Click callback (§4.7). Still no provisioning; behaviour-preserving.
4. **`sync` mode**: lock discovery, rsync of the lock/pyproject pair, the §4.4
   two-step provisioner, §4.5 publication, failure surfacing. Test the
   rename-collision paths (sentinel present → adopt; absent → refuse).
5. **§4.6 digest**: split `venv_digest`, run `_FREEZE_CODE` over ssh, record and
   compare.
6. **Docs**: `docs/cli.rst:93-104` and `docs/offloading.rst`; state §6.1
   plainly.

Nothing here is gated any more (§8.1). Steps 2 and 3 are behaviour-preserving
and can land before anything provisions: step 2 adds a computation nothing
calls yet, step 3 renames a flag onto a superset of its own semantics. Step 4
is where the feature acquires the ability to fail, and is the one worth
splitting further if it grows.
