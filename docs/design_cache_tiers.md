# Step caching beyond skip-if-unchanged: mutation-chain snapshots

**Status:** v5 — **Tier 1 implemented and merged into the tree.** Supersedes v4 (and v3, v2, and the v1 "Content-addressed caching for MeasurementSets and FITS images")
**Context:** step-level caching for shinobi (radio-interferometry pipeline framework)

**Change log**

- **v4 → v5 (implementation).** Tier 1 ships as `src/shinobi/snapshots.py` + `src/shinobi/clonefs.py`, with `ProvenanceKey`/`_JsonFileStore`/content sample in `cache.py` and `mutated_path_fields` in `steps/schema.py`. Of v4's three load-bearing claims, one held and two had holes; three further defects turned up during implementation. In order of severity:
  1. **The manifest was a *stale* oracle, not a per-run one (§4.2.10, invariant 5 — broken as written).** `CacheManifest` holds one entry per step path, so `entry.cache_key == marker.cache_key` says only that *some* run of that step with that key once succeeded. Reachable sequence: a step succeeds and records; the user deletes some *other* declared output of it; the outputs-exist check misses, the step re-runs under the same key and is killed mid-rewrite; reconciliation finds the old entry, declares the step complete, discards the trash and leaves the head trusted — a durable false hit over a half-written MS, i.e. exactly the failure §0 said would break the design, manufactured by the recovery path. **Fixed:** `record` stamps a `run_id` (conditional field; legacy entries lack it, compare unequal, and so take the conservative branch) and reconciliation compares it.
  2. **The never-worse rules missed the unwired case (§4.2.12, invariant 7 — incomplete).** Rule 2 covers *wired*-but-keyless fields. It does not cover an **unwired** mutated boundary path, where R comes from the consumed record rather than a producer key: an uncached `flag` dirties the head, a cached `cal` then misses and force-restores from its own last consumed state, reverting `flag`'s legitimate work and calibrating stale data — strictly worse than the shipped behaviour. The root cause is that one `untrusted` flag conflates *the disk is a partial write of a nameable state* (restore is right) with *an unnameable writer legitimately advanced the disk* (restore is destructive); §4.2.3 folded the ctime-mismatch case into the same flag and inherited the same defect. **Fixed:** head status is three-valued — `TRUSTED` / `UNTRUSTED` (crash, or `cache invalidate`; force a restore) / `DETACHED` (uncached mutator succeeded, or tree replaced out-of-band; no Tier 1 handling at all, run against live disk, warn). A detached chain keeps its snapshots and re-establishes a trusted head at the next cached mutator's S2.
  3. **Inode-keyed chains cannot survive the operation they exist for (§4.2.3).** Quarantine-and-swap renames the live tree aside and clones a snapshot into its place — a *freshly created directory with a new inode*. Keyed by `(st_dev, st_ino, st_ctime_ns)`, every restore orphans the chain it was restoring and starts an empty one, so the next step finds no generations, no head and no snapshots: protection switching itself off precisely when it had just been used. **Fixed:** chains are keyed by canonical path; the aliasing hazard the inode was for (one MS reached by two paths) is handled by `aliased_chain`, which refuses rather than letting two chains each believe they own the head. `ctime_ns` survives only as an out-of-band-change signal, is refreshed after our own writes, and now maps to `DETACHED` — so a false positive costs protection and a warning, never data.
  4. **The hardlink rung breaks A1 (§4.4).** A1 promises every rung differs "only in space, never in content", but a hardlinked snapshot *shares the inode*: casacore rewrites table files in place, so the write lands in the snapshot too and a later restore reinstates the corruption it was taken to undo, reporting success. **Fixed:** rung dropped. The ladder is FICLONE → `copy_file_range` (ZFS, probable) → copy, and `tests/test_clonefs.py` pins the property per rung by mutating the source in place and asserting the snapshot does not move.
  5. **`cache invalidate` left the chain suffix stranded (§4.2.13).** v4's "downstream chain entries" clause was load-bearing and easy to skip: the invalidated step re-executes with the same params, so its key — and therefore every consumer's key — does not move, and the consumers cache-hit over a tree that was just rolled back, leaving the chain stopped mid-way with a manifest claiming it finished. **Fixed:** invalidate also drops the manifest entries of every step whose consumed record names the invalidated generation or a later one, read off the journal.
  6. **Restore consulted the journal about snapshot existence.** `snapshot_present` records what was *intended*; eviction, a partial clean or an `rm -rf` make it disagree with disk, and trusting it turned a recoverable "warn and proceed" into a `FileNotFoundError` mid-restore. **Fixed:** existence is read from the snapshot directory; the journal flag now only distinguishes "never taken (preflight refused)" from "taken and since removed" in the warning text.
  7. **Claim 2 (`ProvenanceKey` is hash-transparent) verified and kept.** Confirmed empirically in every position the values occupy — list element, dict value, dict *key*, under `sort_keys` and `default=str`, through `combine_keys`, and for equality/ordering/hashing against plain `str` — and end-to-end by running a three-step nested-recipe chain against the pre-change tree and diffing the manifests (byte-identical). One implementation note the design missed: `producer_field` needs a *class-level* default, because `copy.deepcopy`/`pickle` reconstruct a `str` subclass through `str.__new__` before restoring state.
  8. **Reconciliation was gated on the wrong question, and never ran.** "Is Tier 1 active for this *run*" and "does *this scope* get a snapshot guard" are different; a `Recipe` correctly gets no guard (it is never cached and mutates nothing of its own), but a top-level target is almost always a Recipe, so gating crash recovery on the leaf-level answer meant it never fired outside tests that called `reconcile()` directly. Invisible, because an interrupted step still recovers through the marker-forced restore — only the tidying silently stopped. **Fixed:** `snapshots_active` (run-level) is now separate from `snapshots_enabled` (scope-level).
  9. **Concurrent runs: reconciliation could destroy a live run's work (stage 0).** Reconciliation reads "marker set, no matching manifest entry" as a corpse and swaps the quarantined tree back over the workspace. A run that is merely *still going* is indistinguishable — it has not recorded yet either — so a second shinobi starting mid-step would delete the first's work while it was being written. That is *worse* than the shipped multi-process behaviour it replaced, where concurrent runs merely raced. **Fixed** by `RunPresence`: each run holds an exclusive `flock` on its own `snapshots/locks/<run_id>.lock`, and reconciliation only proceeds when every other lock file is lockable (i.e. its owner is gone). Deliberately **best-effort** — `flock` needs the `flock` mount option on Lustre and can otherwise silently no-op, and NFS depends on protocol version — so anything that cannot be established reads as "not alone" and reconciliation is skipped. Skipping is the safe direction: reconciliation is a repair, not a prerequisite, since a marker left set makes the next restore *force* a rollback anyway; only the tidying waits. Making concurrent runs actually work (rather than merely not corrupt each other) needs a path-scoped lock across the mutating window plus a probe that verifies locking functions at all — deferred, and see §6.3.
  10. **`os.copy_file_range` is a build-time option, not a Linux guarantee (found by CI).** CPython compiles the binding in only when the C library had it at *configure* time, so a portable redistributable build -- python-build-standalone, which `uv python install` fetches and which CI runs -- lacks it on a kernel that supports it perfectly well. The system interpreter used for local development had it, so the assumption held everywhere it was tested and failed everywhere it was not. **Fixed:** `_has_copy_file_range()` gates both the copy path (degrade to read/write) and the probe (a ZFS host whose Python lacks the syscall cannot reach the ZFS rung, and its `cache check` line says so rather than reporting a clone it never performed). The content tests now run with the syscall present *and* absent, since that is precisely the kind of environment difference A1 says must cost space and never content.
  Smaller: the clone-capability probe named its scratch files by pid, which collides across hosts sharing one directory (uuid now); `as_provenance_key` must **not** re-wrap an already-named key, or a state is renamed at each recipe boundary and loses uniqueness (§4.2.1 specified this only for the `InputRef` pass-through; it applies equally to the `output_keys` builder); snapshot/restore is implemented **in-process** (`fcntl.ioctl`/`os.copy_file_range`/read-write) rather than shelling out to `cp`, which removes the argv surface §4.4 worried about and the GNU-`cp`-version dependence entirely; `clone_tree` preserves mtimes on every rung, because generation 0 is *named* by its fingerprint; snapshots are written to a `.partial` staging name and renamed, so a durable name never appears over a half-written tree; the §2 citation of `provenance_key` was wrong (`src/shinobi/results.py`, not `src/shinobi/steps/results.py`).

- **v3 → v4 (review round 2).** Review verified every §2 spine claim against the tree and found them all accurate — except one, and it was load-bearing. Four correctness holes in v3's own mechanisms, plus the spine correction:
  1. *Crash reconciliation swapped back after a successful step.* v3's rule "marker present and trash present → swap back" fires on a crash between `manifest.record` and trash deletion: the disk reverts to pre-state while the manifest holds an entry whose outputs-exist check passes, producing a durable false cache hit over reverted content — the exact bug class Tier 1 exists to kill, manufactured by the recovery path. Fixed by making the manifest the success oracle: the in-flight marker now carries the step's `cache_key`, and reconciliation swaps back only when the manifest *lacks* `(step_path, cache_key)` (§4.2.10, invariant 5).
  2. *`ninja cache invalidate` left the journal dishonest.* v3 rolled the journal head back without touching the disk, so the next miss computed R equal to the rolled-back head and no-op'd at branch 2 — re-executing against the garbage the invalidate was meant to escape. Fixed: invalidate also marks the head *untrusted*, forcing restore branch 1 on the next miss (§4.2.13).
  3. *Mixed-caching chains made Tier 1 actively destructive.* With an uncached mutating producer (per-scope `cache=False`) and a cached consumer, `_resolve_input_keys` omits the keyless field (dispatch.py:644–647), the journal head never moves (the uncached producer takes no Rule B), and the consumer's restore would revert the producer's work. Fixed by the never-worse rules: every mutating dispatch — cached or not — dirties the journal (marks the head untrusted), and a wired-but-keyless mutated field gets no Tier 1 handling at all, with a warning (§4.2.12, invariant 7).
  4. *Eviction's tip rule contradicted itself.* "Never evict a generation named by a live head" plus "then tips oldest-access-first" — the tip *is* the head. Fixed: tip eviction only for dead chains (workspace path gone) (§4.2.11).
  5. *Spine correction: the producer field does not cross recipe boundaries.* v3 claimed `_resolve_input_keys` "reaches recipe boundaries via `InputRef`" with the field in hand; false — `output_keys` (dispatch.py:1206–1208) and `provenance_key` (src/shinobi/results.py:172–174) carry bare keys, and the field cannot be added to `input_keys` values because those are hashed into `__upstream__`. Fixed with `ProvenanceKey`, a `str` subclass carrying `.producer_field`: JSON-serializes as the bare string, so all hashed key material is byte-identical and existing entries survive (§4.2.1).
  6. *Scatter × mutation and list-valued mutated fields:* `(key, field)` cannot name N paths; v3 said nothing. v4 excludes both loudly rather than misnaming them (§4.2.2).
  7. *Restore now runs before Rule A* (v3 was unordered), closing the window where Rule A could snapshot corruption under a durable name (marker set + a prior Rule B preflight skip). The property that licenses the hook placement — restoring a mutated path cannot change any computed key, because mutated paths never contribute fingerprints (cache.py:382) — is promoted to invariant 6.
  8. *Tier 2 deferred to Appendix A* with explicit trigger conditions: on a same-device store, clone-only inserts reclaim zero bytes, so only strip/cold-tier reclaim capacity — and those are usable as plain workspace management without a store identity system. The two integration fixes review found for the store (a hit must synthesize a manifest entry, and materialisation must inform the journal) are baked into the appendix so they cannot be re-lost.
  Smaller: generation-0 fingerprint names are hashed before filesystem use; the journal shares a factored `_JsonFileStore` with `CacheManifest` (DRY); `run_id` minted per top-level dispatch (trash was `<pid>` in v3 — pids recycle); journal identity gains `st_ctime_ns` against inode reuse; the content sample uses stdlib `blake2b`, not a new blake3 dependency; the capability probe records its decisions for `ninja cache check`; snapshot/restore shell-outs are exec-form argv per SECURITY.md; the post-success sequence gets fault-injection points for tests; §5's path invariant carries the Rule-B-preflight caveat v3 omitted.
- **v2 → v3 (review round 1).** Six corrections landed, two of which killed pieces of v2 as written:
  1. *Tip-of-chain durability.* v2 snapshotted only pre-states (taken by the mutating consumer), so a chain's final state had no snapshot; combined with trash deletion at *swap* success, a failed re-run after a restore destroyed the calibrated tip and left an under-processed MS at a finished-looking path. Fixed by Rule B (post-success tip snapshots) and trash retention until *step* success with swap-back on failure (§4.2).
  2. *Unwired mutated boundary paths had no restorable name.* The pre-state fingerprint is unknowable at restore time (only the post-chain state is on disk), so the very case that motivates Tier 1 — an ingested MS mutated in place with no in-recipe producer — silently degraded to warn-and-proceed. Fixed by the per-path chain journal (§4.2), which is an admitted new naming authority; v2's "no new naming authority" claim was wrong.
  3. *One producer, two mutated outputs, one name.* `StepResult.provenance_key` resolves every field of a leaf step to that step's single `cache_key` (src/shinobi/results.py:158-174). State names are now `(producer key, producer output field)`. (The mechanism v3 named for threading the field across recipe boundaries was wrong — see v3 → v4 item 5.)
  4. *Sidecar mutability on the read path.* `last_access` rewrote the sidecar on every hit, reintroducing the multi-process race at entry granularity. Split into immutable `<key>.json` + a touched `<key>.atime` file (Appendix A).
  5. *No insert atomicity.* Insert is now to-temp-then-rename with the sidecar written last as the commit marker, plus a GC pass for sidecar-less artefacts and a `key_version` field so key-scheme changes produce recognisable orphans (Appendix A).
  6. *ZFS ladder rungs inverted.* `cp --reflink=always` fails on ZFS, but GNU `cp` attempts `copy_file_range` first, which is exactly where ZFS block cloning hooks — the "full copy" rung may clone for free. ZFS becomes a distinct capability tier, "probable clone, sharing unverifiable", gated on a `zfs_bclone_enabled` check (§4.4).
  Smaller: §4.5 now says explicitly that the content sample does not cover out-of-band intermediate edits.
- **v1 → v2.** v1 was written without knowledge of the shipped step cache (`src/shinobi/cache.py`); v2 adopted it as the spine and re-scoped to rollback of mutation chains, capacity under scarcity, and reuse of deleted artefacts. Offloaded (detached Slurm) execution remains out of scope.

---

## 0. Status

**Tier 1 is implemented** (`shinobi.snapshots`, `shinobi.clonefs`, plus the
`cache.py`/`schema.py` factor-outs). The review round this section used to
solicit has happened; its results, including the three claims below, are in
the v4 → v5 change log. Two of the three had holes and are fixed in the
tree; the third held and is pinned by tests.

The claims, kept because they remain the things most worth re-attacking if
this is ever refactored:

1. **The manifest is the success oracle (§4.2.10).** Reconciliation decides swap-back vs. trash-discard purely on whether the manifest holds `(marker.step_path, marker.cache_key)`. Find a crash window where that oracle still mis-decides — the known residual cost is a bounded re-run in the S2–S3 window (§6.6), accepted deliberately; what would break the design is a window that manufactures a false *hit*.
2. **`ProvenanceKey` is hash-transparent (§4.2.1).** The claim: a `str` subclass passes through `json.dumps(..., sort_keys=True)`, `combine_keys`, sorted ordering, and equality against plain strings, byte-identically, in every position `input_keys`/`output_keys` values occupy. Find a position where it doesn't.
3. **The never-worse rules are complete (§4.2.12, invariant 7).** Every field Tier 1 cannot name or vouch for degrades to shipped behaviour plus a warning. Find a cache-enablement shape (per-scope overrides, a recipe boundary, a keyless producer mid-chain) where Tier 1 still does something the shipped cache wouldn't — something worse.

Where the shipped code diverges from what follows, the code is right and
this document has been annotated; §9 records what landed.

---

## 1. Problem

MS directory trees (GB–TB) and FITS cubes (GB) are produced/consumed by pipeline steps; re-running identical steps is common on resume and during development; **storage is scarce relative to data volume**, so any design whose steady state is "N full copies of a 2 TB MS" is dead on arrival.

**What the shipped cache cannot do** — the scope of this plan:

1. **Rollback/branch of mutation chains, and mid-mutation failure recovery** — *the* problem v4 ships. Chain: `split → flag → cal`, the latter two MUTABLE on one MS (the caracal2 shape). Two sub-cases, both shipped hazards today:
   - *Re-run with changed params.* Change `flag`'s params after a full run: `flag` misses, but the MS on disk holds **post-cal state**, and `flag` re-executes against the wrong input.
   - *Resume after mid-mutation failure.* `cal` dies halfway through rewriting the MS. The skip cache cannot see the corruption (mutated paths are content-blind); the re-run of `cal` executes against its own half-written output. This is the stronger motivation: it converts a silent wrong-science bug into a correct re-run.
2. **Capacity** (a deleted 2 TB intermediate forces hours of recompute) and 3. **cross-workspace reuse** are **deferred to Appendix A** — Tier 2 is not approved for implementation; see §7 for why.

**Explicit non-goal: offloaded runs.** Under `ninja compile --engine slurm`, jobs execute detached with no shinobi process present; "the framework sees writes" is false there. Caching for offloaded DAGs would require `compile_slurm` — currently a pure function — to embed snapshot/store logic into generated shell. Out of scope until a real cluster run demands it.

---

## 2. What already exists (the spine, shipped and opt-in)

`src/shinobi/cache.py` + the gate in `src/shinobi/steps/dispatch.py::_dispatch` (lines 427–443, record at 483–486), enabled via `Scope.cache`/`cache_dir` (schema.py:333–337) or `AppConfig.cache` (config.py:78–85, default **off**). Facts this plan depends on, all re-verified against the tree in review round 2:

- **Key** = `(image tag, identity [command/flavour, or unwrapped function source], canonicalized prepared params, __venv__ if set, __upstream__ provenance)` (`compute_cache_key`, cache.py:320). Decisions adopted unchanged, each with recorded rationale in-tree:
  - image **tag** is keyed, digest only *recorded* (cache.py:92–96) — no `docker inspect` at check time;
  - **no param exclusion list** — all prepared params are hashed; the only deliberate non-key is `resources`, with the rationale "constraints, not identity" (cache.py:373–379);
  - new key parts are appended conditionally (the `__venv__`/`__upstream__` pattern, cache.py:366–372), so existing entries survive upgrades;
  - a `Recipe` is never itself cached; it carries per-output-field provenance (`output_keys`) so editing one branch doesn't invalidate the rest.
- **Wired path inputs are keyed by the producing step's cache key** — Merkle-style, threaded from `_run_recipe` via `_resolve_input_keys` (dispatch.py:636, 962, 1025). Bytes are never examined. This is the only correct treatment of in-place mutation: the key names *which state* of a multiply-rewritten path a step consumed, which no mtime discipline can express (cache.py:24–33).
- **Unwired boundary paths are content-fingerprinted** as sorted `[relpath, mtime_ns, size]` (`_walk_fingerprint`/`_hash_path`, cache.py:146–278), **including the path string in the key** (so `cp -a`/`rsync -a`/`tar -x` copies of one MS don't alias), with symlink/EACCES/DT_UNKNOWN semantics pinned, memoized with documented invalidation points (`invalidate_path_hashes`, dispatch.py:393, 468). A boundary path the step itself mutates (input∩output or `Mutability.MUTABLE`) is **dropped from the key** — "unchanged" then means params unchanged + declared output exists.
- **A hit synthesizes** `StepResult(cached=True)` from the persisted *full outputs model* plus provenance fields after an outputs-exist check (`CacheManifest.check`, cache.py:428). Provenance restoration exists because of a real bug: a hit lacking venv fields laundered an unpinned environment into a "pinned" run manifest (cache.py:468–472).
- **Insert on `result.success` only** (dispatch.py:485).
- **Provenance values are bare keys, without the producer field.** `StepResult.provenance_key(field)` fans out through `output_keys` for a `Recipe` (dispatch.py:1206–1208) and resolves every field of a leaf step to that step's one `cache_key` (results.py:158–174). `_resolve_input_keys.key_of` sees the producer field at the call site for `OutputRef` (dispatch.py:653–657) but discards it; the recipe boundary discards it too. (v3 claimed it survives; it does not. §4.2.1 adds it without touching hashed key material.)
- **Mutated paths never contribute a fingerprint to any key** (cache.py:382 excludes them; they contribute only their path string). Consequence: restoring a mutated path's content under the same path string cannot change any step's computed key. This is what makes a pre-run restore hook safe at all — promoted to invariant 6.
- **Immutability is enforced, not just declared, for containerised steps:** `bind_dir_modes` mounts every `writable: false` input read-only under docker/podman/apptainer/k8s (backends/container.py:455; backends/kubernetes.py:146). Native/venv have no such backstop.
- **Known, documented limitations inherited by this plan:** two processes on one `cache_dir` are unguarded (cache.py:401–404); an intermediate edited out-of-band between runs is not noticed (deletion is, via outputs-exist); an undeclared on-disk dependency gets no protection; and *"if a consumer of a mid-chain path re-runs on its own account, it reads whatever is on disk now"* (cache.py:78–82) — the hazard Tier 1 exists to repair.
- **Housekeeping exists:** `ninja clean` (cli.py:485+). `tests/test_cache.py` is the test surface to mirror.

---

## 3. Assumptions

- **A1 (CoW clones) — accelerator, per-fs, correctness-independent.** No correctness anywhere in this plan may depend on cloning: a reflink, a `copy_file_range` clone, and a full copy differ **only in space**, never in snapshot content. Filesystem reality: XFS (`reflink=1`)/Btrfs support `FICLONE`; ZFS ≥ 2.2 hooks block cloning into `copy_file_range` only (`--reflink=always` *fails* there; plain `cp` may clone, gated on `zfs_bclone_enabled` — VERIFY its default at implementation time, since it was disabled after the 2.2.0-era corruption bug); Lustre/BeeGFS/CephFS/NFS have no clone; GPFS has `mmclone` (privileged). Clones are same-filesystem only (EXDEV), and HPC deployments have several working filesystems. Every operation goes through the capability ladder (§4.4).
- **A2 (mtime trust) — scoped to boundary paths only**, exactly as shipped; nothing here adds mtime exposure to wired paths.
- **A3 (framework sees writes) — true only for local dispatch.** All hooks live in `_dispatch`.
- **A4 (mutability classes) — as shipped:** `Mutability.MUTABLE` per-field (schema.py:30–37), input∩output equivalence, `writable: false` with container ro enforcement. Snapshot eligibility uses exactly `compute_cache_key`'s `mutated_paths` computation (cache.py:362), factored into a shared helper so the two can never drift (§4.2.4).
- **A5 (outputs not byte-reproducible)** — keying is on inputs/identity/params; nothing content-addresses outputs.

---

## 4. Design

### 4.1 Keying and skip semantics — adopted unchanged

No changes to `compute_cache_key`, `CacheManifest.check/record`, or the dispatch gate. Any key change must preserve the in-tree compat discipline: new parts appended conditionally (the `__venv__`/`__upstream__` pattern), so existing entries survive upgrades. The `ProvenanceKey` addition (§4.2.1) changes *types*, not hashed content.

### 4.2 Tier 1 — mutation-chain snapshots

**Prerequisite, now stated:** Tier 1 names states by cache keys, so snapshotting requires the step to be cacheable (`cacheable`, dispatch.py:409). What does *not* require cacheability is journal dirtying (§4.2.12): an uncached mutating step cannot name its output state, but it must still stop the journal from vouching for it.

#### 4.2.1 Threading the producer field: `ProvenanceKey`

State names are `(producer cache key, producer output field)`. The key already crosses recipe boundaries; the field does not (§2). Adding the field to `input_keys` values is rejected: those values are hashed into `__upstream__` (cache.py:392), so changing their shape rewrites every existing key — violating §4.1.

Instead:

```python
class ProvenanceKey(str):
    producer_field: str | None  # the output field of the producer this key names
```

`json.dumps` serializes it as the plain string, so `__upstream__` and `combine_keys` hash byte-identically to today and every existing cache entry survives. It is wrapped at the three sites where the field is in scope: `_resolve_input_keys.key_of`'s `OutputRef` branch (`source.field`; element-wise for list wiring), the `output_keys` builder (`out_ref.field`, dispatch.py:1206–1208), and pass-through of `inbound_keys` for `InputRef` (already-wrapped values survive unchanged). `StepResult.output_keys` becomes `dict[str, ProvenanceKey]`; `provenance_key` returns it. A golden test asserts a nested-recipe pipeline's keys are unchanged by the introduction.

#### 4.2.2 State names

A state of a mutated path is named:

- **Produced state (wired or unwired):** `(producing step's cache key, field)`. The directory name is `<full-64-hex-key>__<field>` (the field is a sanitised identifier; no truncation, no collision handling needed).
- **Base state (generation 0):** a boundary path with no in-recipe producer, named `gen0__<sha256 of the canonical fingerprint JSON>[:16]` at first mutation. The fingerprint structure is thousands of entries for an MS — unusable as a name unhashed. The journal (§4.2.3) is the naming authority for generation 0, because it cannot be recomputed at restore time (only post-chain state is on disk).

**v4 protection exclusions (loud, not silent):**

- *Scattered steps with mutated scatter fields.* Per-slice Rule B names would work (slice keys differ via the path value in params), but Rule A's consumed name would need the slice index threaded into `_dispatch`, and scatter+mutation is not the motivating shape (caracal mutates one MS through a linear chain; its parallel branches are nested recipes). v4 skips Tier 1 protection for these with a warning at expand time.
- *List-valued mutated fields* (one field, N paths, non-scatter). One `(key, field)` name cannot stand for N paths; a wrong restore here is catastrophic, not wasteful. v4 skips with a warning.
- *Wired-but-keyless mutated fields* (producer uncached or uncacheable): no restore, no snapshots — §4.2.12.

A skipped field behaves exactly as shipped: no protection, never worse.

#### 4.2.3 The chain journal

`snapshots/chains.json`. Per tracked path:

- **identity:** `(st_dev, st_ino, st_ctime_ns)` of the tree root, not the canonicalized path string. Two workspaces reaching one physical MS through a symlink, bind mount, or hardlinked tree canonicalize differently; keyed by path, the journal would open two chains over one inode, each believing it owns the head, and a restore in one would silently revert the other. The canonical path is recorded for diagnostics only. Two chains resolving to one identity: refuse. `st_ctime_ns` guards inode reuse: a chain whose recorded `(st_ino, st_ctime_ns)` no longer matches the live tree has had its tree replaced out-of-band — warn, mark the head untrusted, continue (the named snapshots remain valid).
- **generations:** append-only list of `{name, size, snapshot_present}`, plus a **head**, plus an **untrusted** flag on the head ("the head names a state the disk may not hold"). Appending the generation that already *is* the head is a no-op (idempotency — required by recovery from the S2–S3 crash window, §4.2.8).
- **in-flight marker:** `{step_path, field, cache_key, run_id, started_at}` or absent. Written before the step runs, cleared only at the end of the post-success sequence. `cache_key` is what lets reconciliation distinguish "crashed mid-step" from "crashed after recording" (§4.2.10); null for an uncached mutator.
- **consumed records:** per `(step_path, field)`, the state name consumed on the last successful run. This is R for an unwired field's restore (§4.2.6) and an eviction-protection set (§4.2.11).

The journal and the snapshot directory are one unit: created together, destroyed together (`ninja clean --cache` removes both; a partial clean is worse than either alone, because generation 0's name is re-derived from live disk state and would then name the post-chain state under the position the base state used to hold). The journal is the same shape of calculated risk as `CacheManifest` (one JSON file, per-process lock, multi-process unguarded — §6.3).

**Mechanics:** the journal gets `CacheManifest`'s exact discipline — `threading.Lock` plus write-temp-then-rename — via a shared `_JsonFileStore` helper factored out of `CacheManifest` (read/`_write_atomic`/lock, cache.py:398–426), which both classes then use. Two private copies of atomic-JSON-store logic is the `_modelgen` drift waiting to happen. Per-path serialisation against concurrent mutators is inherited from "the declared graph is the truth" (cache.py:70–76): two steps mutating one path without an edge between them are already an unprotected race at `max_workers > 1`; with an edge, they never run concurrently.

#### 4.2.4 Hook placement and gating

All hooks live in `_dispatch`, for any non-Recipe scope with at least one eligible mutated path field. Eligibility is computed by `mutated_path_fields(scope)` — the `mutated_paths` computation factored verbatim out of `compute_cache_key` (cache.py:362) into `steps/schema.py` (it needs only `path_fields` and `mutability_of`, both schema residents; cache.py imports it, so the key and the snapshotter can never drift) — minus the §4.2.2 exclusions.

Pre-run hooks fire **after a cache miss and before `func(ctx)`/`ctx.run()`**, in this order:

1. Compute R per field (§4.2.6).
2. Restore pass (§4.2.6, §4.2.7).
3. Write the in-flight marker per path.
4. Rule A (§4.2.5).
5. The step runs.

Restore before Rule A is load-bearing (§4.2.5). The computed cache key is unaffected by anything the restore pass does (invariant 6). Steps with an orchestration `func` are covered identically: the hook precedes `func(ctx)`, and the mutation happens inside. If the mutated path does not exist at hook time (an acquire-and-mutate first run), there is nothing to stat: skip the pre-run hooks for that field; the post-success sequence starts the chain at Rule B with no generation 0.

#### 4.2.5 Rules A and B

**Rule A — consumed state, pre-run (step 4, after the restore pass and marker write).** Snapshot each eligible mutated input under its consumed-state name, skip-if-name-exists. Wired field → `(producer key, producer field)` from the field's `ProvenanceKey`; unwired field → the journal head for that path, or its live fingerprint if this is the first mutation (becoming generation 0).

Rule A must not run before restore, and must not run with a marker present: in the window where the previous run's Rule B was preflight-skipped *and* the previous run crashed mid-mutation, a marker-present Rule A would snapshot corrupt content under the head's durable name. The restore-first order makes the disk honest (or loudly warned) before anything is snapshotted.

**Rule B — produced state, post-success (stage S1).** After a mutating step succeeds, snapshot each mutated path under `(this step's key, field)`, skip-if-name-exists. Without it a chain's final state exists only at the workspace path and a later restore destroys it. On a clone-capable fs Rule B is ~free and usually dedups against the next consumer's Rule A (same name); on the copy ladder it doubles snapshot volume per mutating step, so it takes the same free-space preflight as restore. If the preflight fails, log loudly, skip the snapshot, and record `snapshot_present: false` on the generation — the chain is then incomplete at that generation in a way restore reports precisely, rather than discovering it as a generic miss.

#### 4.2.6 Restore — pre-run, on cache miss

For each eligible mutated input field, compute **R, the required state name**:

- Wired field → `(producer key, producer field)` from its `ProvenanceKey`. Wired-but-keyless → warn once, exclude (§4.2.12).
- Unwired field → the consumed record if present; else the journal head if the chain exists (this is the mid-first-mutation crash recovery: the live fingerprint would name the corruption); else the live fingerprint (first mutation ever).

Then:

1. **In-flight marker present, or head untrusted** → the disk is untrusted regardless of what the head says. Force restore from R. (The marker/flag downgrades branch 2 from "trust the disk" to "force a restore"; this is what closes the mid-mutation hole and the out-of-band-edit case.)
2. Else **head == R** (or no chain exists and the path does) → no-op.
3. Else **snapshot under R exists** → quarantine-and-swap (§4.2.7).
4. Else → warn loudly and proceed (§6.2).

**Why restore-on-miss is safe (attack this).** The restored snapshot is byte-identical to the state the step's DAG position requires, and provenance keying forces the whole downstream suffix to re-run. Two scopes are explicit. (a) *Suffix re-runs assume whole-DAG invocation*: step-level selection or a second recipe in the same workspace can leave the journal head mid-chain with nothing scheduled to move it forward; it self-heals on the next full run, and `ninja cache check` reports paths sitting off-tip in the meantime — out-of-band readers in that window see reverted data (§6.5). (b) *Standalone invocation of a mutating cab* (its boundary path dropped from the key, hitting on "params unchanged + output exists" while the disk holds a reverted state) is the shipped boundary regime biting exactly as documented today — Tier 1 does not introduce it and does not fix it.

#### 4.2.7 Quarantine-and-swap

Move the live tree to `<path>.shinobi-trash.<run_id>` (a run id minted per top-level `_dispatch` entry, `_cache_path is None`, and threaded down like `_config` — not a pid; pids recycle and collide with trash orphaned by an earlier crash), clone the snapshot in, and retain the trash until the step finishes: delete on success, swap back on failure (§4.2.9). A failed step therefore never leaves the workspace rolled back. On the copy ladder, restore preflights free space against tree size and refuses loudly rather than doubling a 2 TB tree it cannot fit.

#### 4.2.8 Post-success sequence

Stages, in this order, with a fault-injection hook between every pair (tests kill the sequence at every point; see §9):

- **S1.** Rule B snapshot.
- **S2.** Journal: append generation (idempotent if already head), move head, **clear the untrusted flag** (the head now vouches for the disk), record consumed name.
- **S3.** `manifest.record` (and, when Appendix A lands, store insert).
- **S4.** Delete trash.
- **S5.** Clear the in-flight marker.

Two constraints matter: S1 precedes S3, or a crash in between yields a step the skip cache hits forever with no tip snapshot; and the marker clears last, so a crash anywhere in the sequence leaves the path marked in-flight and is reconciled conservatively (§4.2.10). S2 clearing the untrusted flag is what re-vouches the head; it happens only after a real success.

#### 4.2.9 Failure handling (no crash)

On non-zero returncode or worker exception: if trash exists (a restore happened this run), swap it back and delete the failed mutation — the workspace keeps its pre-run state. The marker is left set: the next run's restore branch 1 then forces a restore from R before re-executing. This is the motivating mid-mutation case, and it needs no reconciliation pass because the journal is already honest.

#### 4.2.10 Crash reconciliation

Runs once per cache_dir per process, at top-level `_dispatch` entry (alongside `invalidate_path_hashes`, dispatch.py:393), and on `ninja clean`/`ninja cache check` invocation. A crash — SIGKILL, OOM, node eviction — leaves no live process to swap back. For each journal path with a marker present:

- **Manifest has `(marker.step_path, marker.cache_key)`** → the step completed and recorded (crash in S4/S5): delete trash if present, clear marker. The trash's named snapshot is guaranteed by S1-before-S3. (v3 swapped back here unconditionally — see change log, item 1.)
- **Manifest lacks the entry** → the step did not complete: swap back trash if present, **mark the head untrusted**, clear marker. The next restore takes branch 1. If the crash was between S2 and S3, this re-runs a step that actually succeeded — bounded waste in a narrow window, chosen because the manifest is the only durable success signal and guessing otherwise risks the false hit (§6.6).
- **Orphaned trash (no marker):** never swapped back blindly — there is no marker to say what it was quarantined for. Reported by `ninja cache check`; removed by `ninja clean --force` with a warning.

`ninja clean` refuses to remove unreconciled trash without `--force`.

#### 4.2.11 Eviction

Walk the journal's chains. Candidates, in order: superseded generations — those named by **no live head and no consumed record** — newest-first; then, **for dead chains only** (workspace path no longer exists), the remaining generations oldest-access-first. Never evict a generation named by a live head or a recorded consumed state: deleting mid-chain makes every downstream restore impossible and, on a clone-capable fs, reclaims almost nothing anyway. (v3's "tips oldest-access-first" contradicted the head rule — the tip *is* the head. Dead-chain-only is the corrected form.) The consumed-record rule protects generation 0 automatically for as long as the first mutator's last run consumed it. Sizes recorded at insert; never `du` at eviction time.

#### 4.2.12 Mixed caching: the never-worse rules

Cache enablement is per-scope (the `cache=` precedence chain), so a mutation chain can be partially cached — e.g. `flag` runs `cache=False` inside an otherwise-cached recipe. v3 was silently destructive here: `cal`'s restore, naming its input from a journal head that `flag`'s uncached runs never moved, would revert `flag`'s work. Two rules prevent Tier 1 from ever being worse than the shipped state:

1. **Every mutating dispatch dirties the journal, cached or not.** An uncached mutator cannot snapshot (no key), but its post-run sequence marks the path's head untrusted. No later restore then trusts a stale name.
2. **A wired-but-keyless mutated field gets no Tier 1 handling at all** — no restore, no Rule A/B — with a one-time warning naming the field and the producer. Detecting "wired but keyless" requires the wiring set, which `_dispatch` does not currently see: `_run_recipe` threads `set(ref.wiring)` down as `_wired_fields` alongside `_input_keys` (one more parameter through `_submit_unit`'s payload).

#### 4.2.13 Mechanics and tooling

- Snapshot dir `AppConfig.cache.dir/snapshots`; capability per the ladder (§4.4); EXDEV against the workspace path degrades to the copy rung with a preflight (cross-filesystem "clones" are full copies and must be honest about it). Degradation ladder per `cache.snapshots.mode = auto|copy|off`.
- `ninja clean` gains the snapshots dir under `--cache`; refuses unreconciled trash without `--force`.
- `ninja cache check` reports: off-tip paths, unreconciled trash, orphan trash, journal/manifest disagreements, capability-ladder decisions per filesystem (what the probe chose and why), and generations with `snapshot_present: false`.
- `ninja cache invalidate <step-path>` removes: the manifest entry, downstream chain entries, and the step's Tier 1 generation; rolls the journal head back to the previous generation; **and marks the head untrusted**. Without the rollback the escape hatch doesn't reach Tier 1 (Rule B makes zero-returncode garbage a durable named state, and restore faithfully reinstates it downstream); without the untrusted mark the *disk* still holds the post-invalidation state under a head claiming otherwise, and the next miss no-ops at branch 2. With the mark, the next miss force-restores the rolled-back state before re-executing.

**Cost when it never fires:** one capability probe per run, zero I/O for pure steps; journal dirtying is a locked JSON write per mutating step; Rule A dedups against Rule B by name in the steady state, so the marginal cost of a fully-cached mutating chain is one clone per mutation.

### 4.3 Tier 2 — deferred

Capacity store, column strip, cold tier: see Appendix A. Deferred, not cancelled — the trigger conditions and the two integration fixes review round 2 found live there so they cannot be lost.

### 4.4 Capability detection

One module, probed lazily and memoized per `(fs, operation)`, reporting **tiers**:

1. **FICLONE** via a two-file ioctl probe — not by parsing `cp` stderr.
2. **cfr-clone** for ZFS: GNU `cp` attempts `copy_file_range` before read/write (version-dependent — the probe pins the assumption at runtime, the docstring doesn't), which is where ZFS block cloning hooks. Sharing is not verifiable from userspace on ZFS (no `filefrag` equivalent), so this tier is "probable clone", gated on checking `zfs_bclone_enabled`.
3. **hardlink**, then **copy**.

Two invariants keep this honest: the tier affects **space only, never correctness** (A1), so an unverified-probable rung is acceptable; and `--reflink=auto` is never used, because a silent full copy is the worst degradation under scarcity. Every probe decision (filesystem, tier chosen, why) is recorded and surfaced by `ninja cache check` — when the space math looks wrong on some future Lustre/ZFS deployment, the answer is in the report, not in a re-run with debug logging. This module is where the bugs live; it ships with tests exercising each ladder branch via a faked probe.

All snapshot/restore shell-outs (`cp`, and later `tar`/`zstd`) are exec-form argv, never shell templates — the SECURITY.md discipline applies to framework-generated commands exactly as to cabs.

### 4.5 Boundary fingerprint hardening (key-compat preserved)

Boundary fingerprints gain an *optional* bounded content sample (first+last 4 KiB per file) behind `cache.content_sample = true`, default **false** — appended as a new conditional key part, so existing entries key byte-identically (the in-tree compat pattern; entries also record the flag, Appendix A, so flipping it produces recognisable orphans). Hash: stdlib `hashlib.blake2b` — a new third-party dependency (blake3) is not justified for 8 KiB per file. The sample is memoized inside `_hash_path` and inherits its invalidation points exactly (`invalidate_path_hashes`) — a second memo with its own invalidation rule is the drift bug this repo's DRY section warns about.

It targets **cross-dataset collision** (two same-size MSs untarred in the same granularity window). It does **not** cover the shipped limitation it sits next to — an intermediate edited out-of-band: a rewritten `FLAG` column changes neither size nor the sampled extents, and that case remains undetected by design (the declared graph is the truth). Say so in the docs, or someone will read the sample as covering both. Adoption cost: flipping the flag changes every boundary-input key, so the first run after enabling recomputes the boundary layer wholesale.

---

## 5. Invariants

What an implementer checks against. Each is enforced by the mechanism named.

1. **Path invariant.** A workspace path holds either a state produced by a successful step, a restored state whose consuming step is currently running, or a state being written by an in-flight mutating step (marker set). Enforced by: Rule B, quarantine-and-swap with swap-back on failure, and manifest-consulting reconciliation for crashes. *Caveat:* a Rule B preflight skip degrades this to best-effort for that generation — loudly, and recorded as `snapshot_present: false`, never silently.
2. **Journal honesty.** The journal never asserts a disk state it cannot vouch for. Enforced by: the in-flight marker (written before mutation, cleared last); the untrusted flag (set by reconciliation, by uncached mutators, by `cache invalidate`, by ctime-mismatch detection; cleared only by a successful S2); and restore branch 1, which downgrades `head == R` from "trust the disk" to "force a restore" whenever either is present.
3. **Name uniqueness.** One state, one name; one name, one state. Enforced by: `(producer key, producer field)` via `ProvenanceKey`; hashed generation-0 names; `(st_dev, st_ino, st_ctime_ns)` path identity; and the loud exclusion of shapes that cannot be named uniquely (scatter, list-valued fields) — excluded from protection rather than misnamed.
4. **Clones are free, never load-bearing.** Every ladder rung produces byte-identical content (A1); only space differs.
5. **The manifest is the success oracle.** Recovery decisions never infer "the step finished" from journal state alone; only a manifest entry at `(step_path, cache_key)` says so. Enforced by §4.2.10.
6. **Key restore-invariance.** A restore never changes any step's computed cache key: mutated paths contribute only their path string to keys, never a fingerprint (cache.py:382), and restore preserves the path string. This is what licenses placing the restore hook after key computation. Tested explicitly: a forced restore leaves the step's key byte-identical.
7. **Never worse than shipped.** Any field Tier 1 cannot name or vouch for (keyless producer, excluded shape, preflight refusal, missing snapshot) degrades to exactly the shipped behaviour — proceed against live disk — plus a loud warning.

---

## 6. Known weaknesses (delta over shipped §2 limitations)

1. **Wrongly named snapshots** remain the catastrophic case: state names rest on producer keys, which rest on recipe-path uniqueness (cache.py:98–105). A snapshot dir shared across identically-named recipes collides. No mitigation beyond the existing warning.
2. **Restore-when-absent is loud, not safe** (§4.2.6) — first-generation workspaces, `snapshots.mode=off`, copy-ladder refusals, and `snapshot_present: false` generations still hit today's hazard, now with a warning.
3. **Multi-process: not corrupting, not yet working.** `chains.json` and `CacheManifest` share one lock-per-process discipline, so two concurrent runs can still lose a journal update or race an eviction, and nothing stops two runs rewriting one MS at once. What *is* handled is the hazard Tier 1 itself introduced: reconciliation can no longer roll back a run that is still going (`RunPresence`, v5 item 9), and when it cannot establish it is alone it skips rather than guesses. Remaining work, in order: a path-scoped lock held across the mutating window; a locking-capability probe that refuses rather than degrades, since unlike the clone ladder a bad guess here costs correctness, not space (`flock` silently no-ops on Lustre without the `flock` mount option, and A1 names Lustre); sharding the journal per chain so runs on different paths never contend. SQLite still rejected (NFS/Lustre locking pathology). Per-user, same-filesystem first.
4. **Space spikes.** Trash retention until step success means a multi-hour mutating step holds the old tip alongside the restored state — ~free with clones, a full extra tree on the copy ladder. The preflight refuses up front.
5. **Off-tip window.** Restore without a whole-suffix re-run (step selection, a second recipe in the same workspace) leaves the workspace mid-chain until the next full run; `ninja cache check` reports it, restore logs it loudly, and out-of-band readers in that window see reverted data. The standalone-mutating-cab variant is shipped behaviour Tier 1 neither introduces nor fixes.
6. **Re-run waste in the S2–S3 crash window.** Reconciliation treats a missing manifest entry as "did not run", so a crash between journal append and `manifest.record` re-executes a step that succeeded. Narrow window, bounded cost, chosen over the false-hit alternative (invariant 5).
7. **Orphan trash is report-only.** Without its marker there is no honest automatic disposition; `ninja cache check` reports and `clean --force` removes.
8. **Protection exclusions.** Scatter-mutation and list-valued mutated fields are unprotected in v4 (§4.2.2) — loudly.
9. **Sandboxed steps** inherit `relativize_path_outputs` normalisation and its edge cases; snapshots see workspace paths, so the interaction is expected to be benign, and is marked as test-required territory.
10. **Space accounting on CoW is approximate.** Insert-time sizes overestimate reclaimable space; ceilings under-deliver.

---

## 7. Alternatives considered and rejected

- **v1's recorded-mtime keying in place of shipped provenance keys.** Rejected: strictly weaker for mutated paths, adds mtime exposure to wired paths, re-opens fixed false-hit classes.
- **restic/borg-style chunk store.** Rejected with the precise loss statement: it would buy chunk-level dedup across *different keys* — the only dedup this design forgoes — at the price of a content-addressed read path on every materialisation and a second identity system. Revisit only if capacity pressure survives Appendix A.
- **Filesystem snapshots (ZFS/Btrfs subvolumes).** Rejected: privileged, fs-specific, workspace-shaped rather than state-named. If a deployment is uniformly ZFS, subvolume snapshots make Tier 1 nearly free and §4.4 collapses — revisit per-deployment.
- **Tier 2 now (capacity store / strip / cold tier).** The strongest scoping call of this revision. On a same-device store, clone-only inserts reclaim *zero* bytes (deleting the workspace tree frees nothing the store entry shares) — only strip and the cold tier reclaim capacity, and both are usable as plain workspace management without a store identity system, sidecar protocol, and GC. Tier 1 stands alone; Tier 2 waits for evidence (Appendix A).
- **Do nothing beyond the shipped cache.** The strongest alternative. Tier 1 is justified because the rollback/mid-mutation hazards are *wrong-science* bugs, not wasted compute. A site without that pain should ship neither.

---

## 8. Open questions

1. ~~Restore-on-miss: automatic or behind `--restore`?~~ **Resolved: automatic.** With swap-back, failure handling (§4.2.9), and manifest-consulting reconciliation, failure is cheap; a flag re-opens the corruption window on every resume, which is the plan's stronger motivation. `cache.snapshots.mode = off` is the escape hatch. Residual cost: the off-tip window (§6.5).
2. ~~Journal placement~~ — **Resolved: separate file**, with the marker carrying `cache_key` (§4.2.3). Folding into the manifest would make S2+S3 one atomic write, at the price of the shipped format and a reserved key; the marker achieves the same disambiguation for less diff. Both files share the `_JsonFileStore` helper.
3. Shared multi-user store: deferred (Appendix A).
4. Is stripping ever acceptable for precision-path artefacts (polarisation, faint-source photometry)? Deferred with Tier 2; the principle (producer-side opt-in, off by default, recorded forever) is fixed in Appendix A so the eventual conversation is about science, not plumbing.
5. Does Tier 2 earn its keep where intermediates are rarely deleted? **This is now the gating question for Appendix A**, answered by operating Tier 1, not by argument.

---

## 9. Implementation order

Tier 1 shipped alone, in this order. All nine steps landed; what changed
along the way is in the v4 → v5 change log.

**What is in the tree.** `src/shinobi/snapshots.py` (journal, markers, rules
A/B, restore, quarantine-and-swap, reconciliation, eviction, invalidate,
check) and `src/shinobi/clonefs.py` (the clone ladder), plus `ProvenanceKey`
/ `as_provenance_key` / `_JsonFileStore` / `CacheManifest.entry` /
`record(run_id=)` / the content sample in `cache.py`, `mutated_path_fields`
in `steps/schema.py`, `SnapshotConfig` + `CacheConfig.content_sample` in
`config.py`, the hooks and `run_id`/`_wired_fields`/`_slice_index` threading
in `steps/dispatch.py`, and `ninja cache check|invalidate|evict` plus
`clean --force` in `cli.py`. Tests: `tests/test_snapshots.py` (32),
`tests/test_clonefs.py` (25), `tests/test_cli_cache.py` (9), and additions to
`tests/test_cache.py`.

1. **`ProvenanceKey` + wrap sites** (§4.2.1). ~50 lines. Golden test: existing keys byte-identical across a nested-recipe boundary; new test: producer field survives `OutputRef`, list wiring, and recipe nesting.
2. **`_JsonFileStore` factor-out** (§4.2.3). `CacheManifest` refactored onto it; no behaviour change; existing `test_cache.py` must pass unmodified.
3. **`mutated_path_fields` factor-out** (§4.2.4) into `steps/schema.py`; `compute_cache_key` rerouted. No behaviour change.
4. **Capability module + ladder** (§4.4). ~150 lines plus ~150 of faked-probe tests, including the ZFS cfr-probable rung and decision recording. Prerequisite for everything below.
5. **Journal + marker + reconciliation** (§4.2.3, §4.2.10), `run_id` threading, `_wired_fields` threading (§4.2.12). Reconciliation tests cover the full marker × trash × manifest matrix — including the crash-after-record case v3 got wrong.
6. **Rules A/B, restore, quarantine/swap, post-success stages** (§4.2.5–4.2.9) with fault-injection hooks between every stage pair. Tests mirror `tests/test_cache.py` (`RecordingBackend` covers the dispatch hooks without executing tools): the four restore branches; restore-before-Rule-A; key restore-invariance under forced restore (invariant 6); crash between each S-stage pair; the mixed-caching chain (cached consumer, uncached mutator — assert no revert, warning issued); first-mutation crash recovery (unwired R falls back to journal head, not live fingerprint); preflight refusal on the copy ladder.
7. **Eviction + CLI** (§4.2.11, §4.2.13): `ninja cache check`, `ninja cache invalidate` (assert the untrusted mark forces branch 1 on the next miss), `clean` integration and trash refusal. Eviction tests: superseded-only, dead-chain tips, consumed-record protection of generation 0.
8. **Content sample** (§4.5). Independent; can land anywhere after 3.
9. **Docs:** AGENTS.md (cache bullet, repo layout — required by the repo's own rules), SECURITY.md (exec-form shell-outs, §4.4), user docs for `snapshots.mode` and the content-sample scope wording (§4.5).

Estimated: ~700 lines of implementation plus tests — more than v3's ~450, the difference being reconciliation, the never-worse rules, and fault injection, all of which are where the correctness lives.

---

## Appendix A — Tier 2 (deferred): capacity store

**Status: not approved for implementation.** Build it only when operating Tier 1 shows deletion-under-pressure is routine *and* the cheaper workspace-management options (snapshots plus manual strip/cold-tier scripts, no identity system) demonstrably don't suffice. Until then this appendix exists to record the design and the two fixes that must not be re-lost.

The skip cache fails open on deleted artefacts; Tier 2 lets a deleted intermediate come back without recompute. Not a Bazel CAS; no dedup claim beyond key equality.

- **Opt-in per scope** (`cache_store: bool | None`, same precedence chain as `cache`), config default off. Insert on `result.success` at S3.
- **Entry** = artefact + immutable metadata sidecar `<store>/<key>.json`: the full outputs model; the provenance field set exactly as `CacheManifest.record` persists it — shared code, not a copied field list (DRY), because a store hit must synthesize the same `StepResult` a manifest hit does or the venv-laundering bug re-opens; `bytes_at_insert`; `created`; `codec`; `stripped_columns`; chain links; `key_version` plus the `content_sample` flag, so a key-scheme change produces entries GC can *recognise* as orphaned. Access recency lives in a separate zero-byte `<key>.atime` touched on hit.
- **Commit protocol:** write to `<store>/tmp/<key>.<run_id>/`, rename the artefact into place, sidecar **last** (existence is the commit marker; a hit requires a sidecar that parses and validates). **Decommit is the mirror:** sidecar first, then artefact, then `.atime` — artefact-first leaves a live sidecar advertising a tree being dismantled underneath it for the whole `rm -rf` window. GC removes all four orphan shapes: sidecar-less artefacts, artefact-less sidecars, sidecar-less `.atime`s, stale `tmp/`.
- **Hit path:** skip-cache miss → store lookup by key → materialise (clone/decompress into the workspace, to-temp-then-swap) → validate declared outputs exist → synthesize `StepResult(cached=True)`.
- **Fix baked in (review round 2, finding 7):** a store hit additionally (a) **synthesizes a manifest entry** from the sidecar — otherwise cross-workspace reuse, the store's main justification, re-materialises on every run because no local manifest entry ever exists; and (b) **sets the Tier-1 journal head** for the materialised path — the materialisation is the one writer that knows its content's true name, and without this the next mutating consumer's restore computes head ≠ R over a just-materialised, byte-correct state. If `stripped_columns` is non-empty, the journal records them on the chain entry; consumers proceed under the producer-side strip contract (below), while Tier-1 snapshots remain full-fidelity. One name, two recorded fidelities — documented, not silent.
- **Capacity mechanisms, in shipping order:** (1) *column strip (MS)* — remove regenerable columns (`MODEL_DATA`, `CORRECTED_DATA`, often ~2/3 of volume) at insert, recorded in `stripped_columns`; a schema-contract decision, **producer-side per-scope opt-in, off by default, recorded forever** — the contract travels with the artefact; (2) *cold tier* — `tar | zstd -3` past an age threshold; (3) *Dysco/fpack* — deferred behind per-artefact opt-in; **lossy is never a global policy**.
- **Location:** same-device as the workspace (EXDEV), or an explicitly configured full-copy cross-filesystem store. Eviction: size ceiling, chain-aware order (§4.2.11), `.atime` recency, insert-time byte accounting with the documented CoW imprecision.
- **Tooling:** `ninja clean --store`, `--store-gc`; `cache invalidate` removes the store entry too.
- **Known honest limitation:** on a same-device store without strip or cold tier, deletion reclaims nothing (shared blocks). The store's same-device value is cross-workspace dedup and crash protection, not capacity — which is why this appendix waits for evidence rather than building on spec.
