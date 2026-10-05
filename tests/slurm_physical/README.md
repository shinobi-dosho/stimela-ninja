# Physical M1 acceptance probe

This probe targets the Kudu/Nyala cluster documented in
`shinobi-dosho/shinobi-test-cluster`. It is intentionally separate from the
ordinary pytest suite: it submits real detached jobs and requires shared
`/data`, Slurm, Apptainer, and a pre-provisioned tool venv.

Inside the controller, prepare `/data/images/python-3.12-alpine.sif` and a
shared `/data/m1-tool-venv` containing a `venvonlypkg` module with
`MAGIC = 4242`. Then run `run_m1.py submit`, let that process exit, wait for
the reported finalizer job, and invoke `run_m1.py check` from a fresh process.

The workflow covers a binary cab, image-backed pystep and venv-backed pystep;
runtime output wiring; shared sandboxes and harvesting; immutable attempt/job
records; image, code, tool-environment and worker provenance; declaration-order
finalization; and submitter disconnect.

## Physical M2 shared-metadata probe

``run_m2.py`` proves the transaction assumption M2 makes about the exact shared
filesystem mounted at ``/data``. It submits three independent jobs pinned to
``k1``, ``n1`` and ``n2``. After a shared barrier, all three repeatedly update
the same cache manifest and snapshot journal. First, every node takes a turn
holding the exact OFD lock used by ``JsonFileStore`` while both other nodes must
fail a nonblocking acquisition; this directly detects a filesystem that accepts
a lock but enforces it only within one node. The check, run from a fresh process
after the jobs finish, also verifies actual node placement, synchronizes every
update round to force contention, requires every independent manifest and
journal update to survive, and parses the materialized compatibility JSON files
directly as well as through the transaction-aware readers.

Run from the controller against a unique directory::

    python /data/src/stimela-ninja/tests/slurm_physical/run_m2.py submit \
      --root /data/physical-m2-001
    python /data/src/stimela-ninja/tests/slurm_physical/run_m2.py check \
      --root /data/physical-m2-001 \
      --storage-root /data

The successful checker writes
``/data/physical-m2-001/qualification.json``. This persisted, site-issued
record gates strict dataset worker preparation; its exact bytes are pinned in
the execution plan and rechecked by every compute allocation. Use
``--qualification PATH`` to choose another location under the qualified root.

For another cluster, pass ``--nodes NODE...``, ``--partition PARTITION``,
``--expected-host NODE=HOST`` once per node when runtime hostnames differ from
scheduler names, plus ``--source-root`` and ``--worker-python`` as needed. The
defaults describe Kudu/Nyala only; the protocol and checker are not tied to
those names.

Do not claim a different shared mount as supported until this probe passes on
that mount. It tests metadata transactions only, not concurrent ownership of a
measurement set being mutated; the latter remains a separate milestone.

The strengthened probe passed on 2026-09-13 as Slurm jobs 65--67: every one of
``devk1``, ``devn1`` and ``devn2`` held the lock in turn and excluded both
peers; all 60 synchronized cache-manifest and chain-journal updates survived;
and the transaction-aware readers and materialized JSON views agreed. Earlier
probe runs are useful negative evidence: jobs 58--60 proved ``flock`` was
node-local across Kudu's local ext4 and Nyala's NFS view, while jobs 62--64
proved POSIX exclusion alone did not prevent stale-path lost updates after
atomic replacement. Those failures motivated the OFD lock plus persistent
transaction log now under test.

After the storage-review hardening (bounded acquisition, collaborative modes,
immediate-parent syncing and inode-preserving cleanup), the parameterized probe
passed twice on 2026-09-14 as jobs 116--118 and 119--121 with the same
three-way exclusion, placement and 60-record agreement.

## Physical M2 worker-cache probe

``run_m2_cache.py`` submits four successive detached worker workflows against
one shared cache. Binary, real Apptainer-image pystep and real tool-venv pystep
jobs are pinned across ``k1``, ``n1`` and ``n2``.
The first run executes every branch; the unchanged run cache-hits every branch;
changing the root producer reruns it and both wired descendants while retaining
the unrelated hit; deleting one declared product reruns only its producer. The
fresh-process checker validates immutable attempt/finalization states, all
shared manifest entries, and final products::

    python /data/src/stimela-ninja/tests/slurm_physical/run_m2_cache.py run \
      --root /data/physical-m2-cache-001
    python /data/src/stimela-ninja/tests/slurm_physical/run_m2_cache.py check \
      --root /data/physical-m2-cache-001

This complements ``run_m2.py``: the earlier probe proves the mount's lock and
transaction primitives directly under forced contention, while this one proves
the detached-worker cache/provenance behavior built on them.

The strengthened workflow passed on Kudu/Nyala on 2026-09-13 as scientific
jobs 88--93, 95--100, 102--107 and 109--114. Every job completed on its
requested node; the real Apptainer pystep ran on ``n1`` and the real tool-venv
pystep on ``n2`` in every phase. The unchanged phase retained image/code and
venv/code provenance on its cache hits, and its manifest remained unpinned
because the venv fingerprint establishes version parity rather than an exact
OS/binary pin.

## Physical M3 mutation-recovery probe

``run_m3.py`` is the supported-release gate. It runs a compact reduction-shaped
chain across all three nodes: a native binary split on ``k1``, a CASA-shaped
image pystep on ``n1`` (including ``ctx.import_callable`` and a bundled local
helper), and a shared-tool-venv pystep on ``n2``. Two independent report jobs
meet at a shared barrier on ``n1`` and ``n2`` so their cache/provenance commits
genuinely overlap.

The phases prove a clean run, a fully cached rerun, selective invalidation from
an upstream parameter change, captured-code invalidation, tool-environment
invalidation, and a deleted-product rerun. A delayed image job also proves that
editing its original helper after submission cannot change the captured source;
its detached finalizer is cancelled and reconstructed from durable records.
One-shot worker-process exits at ``S2`` and ``W_RESULT`` cover the pre- and
post-success-oracle mutation boundaries. Real Slurm requeues then interrupt
both pystep mutation modes after the target has written ``PARTIAL-*`` into the
MS. Each restarted invocation must use its new attempt identity, restore the
predecessor snapshot, and publish content containing no partial write. A
missing tool venv is rejected before submission. The fresh-process checker
rereads every immutable finalization/manifest, restart record and the actual
final MS/report contents::

    python /data/src/stimela-ninja-m3-145/tests/slurm_physical/run_m3.py run \
      --root /data/physical-m3-001
    python /data/src/stimela-ninja-m3-145/tests/slurm_physical/run_m3.py check \
      --root /data/physical-m3-001

The defaults target Kudu/Nyala and require the M1 tool venv plus the Python
Alpine SIF described above. The run creates a same-content tool-venv copy at
``/data/m3-tool-venv-alt`` when absent; its distinct resolved environment path
must invalidate only the venv step. For another site, pass ``--nodes``,
``--partition``, ``--source-root``, ``--image``, ``--tool-venv``,
``--alt-tool-venv`` and ``--worker-python`` explicitly. Run the M2 metadata
probe on that site's exact shared mount first.

The complete gate passed on Kudu/Nyala on 2026-09-18 as jobs 344--421. Native
steps ran on ``k1``, image pysteps on ``n1`` and venv pysteps on ``n2``. Job
385 was deliberately cancelled to prove fresh-process finalization; image jobs
389 and 401 exited 86 at ``S2`` and ``W_RESULT`` and their retries (395 and
407) recovered exact content. Jobs 413 and 420 were genuinely requeued once;
their published restart-generation records were the attempts selected by
finalization. The checker passed every state vector and the final MS was
exactly ``vis[two]|image[requeued]|venv-requeued[4242]``.

## Physical M4 strict-MSv2 lifecycle probe

``run_m4_msv2.py`` is the release gate for detached strict dataset support.
Run it only after the M2 shared-metadata probe on the same mount, with a shared
worker/tool interpreter containing ``python-casacore`` and NumPy. It creates a
real MSv2 on one node, writes ``SCAN_NUMBER`` on a second, and reads it on a
third. A separate write is then killed at the pre-oracle ``S2`` boundary; the
detached finalizer must restore the exact predecessor, publish a failed
immutable attempt, and release ownership only after recovery. A final retry on
the third node must commit the intended values::

    python /data/src/stimela-ninja/tests/slurm_physical/run_m4_msv2.py run \
      --root /data/physical-m4-msv2-001 \
      --storage-qualification /data/physical-m2-001/qualification.json
    python /data/src/stimela-ninja/tests/slurm_physical/run_m4_msv2.py check \
      --root /data/physical-m4-msv2-001 \
      --storage-qualification /data/physical-m2-001/qualification.json

Use ``--worker-python`` and ``--casacore-python`` when those environments are
separate, and pass the site's three ``--nodes`` plus ``--partition``. A code
merge does not by itself establish site support: record a successful fresh-
process check here, including job and node identities, after running the gate.

Physical M4 passed on 2026-09-25 against the Kudu/Nyala ``physical-dev``
cluster after M2 jobs 422--424 proved shared metadata and cross-node OFD
locking. Jobs 431 (create, ``k1``), 432 (write, ``n1``) and 433 (read,
``n2``) completed, followed by finalizer 434 on ``k1``. The injected S2
writer was job 435 on ``n1`` and exited ``86:0``; finalizer 436 returned
``1:0`` for the intentionally failed workflow only after restoring
``SCAN_NUMBER`` to ``7,7,7,7`` and releasing ownership. Retry job 437 on
``n2`` and finalizer 438 on ``k1`` completed, and a separate checker process
verified the final MS contained ``9,9,9,9`` plus all execution, attempt,
settlement and finalization records.

The review-hardened qualification path passed again on 2026-09-25. M2 jobs
439--441 produced a persisted ``slurm-shared-storage/v1`` qualification after
the same three-way exclusion and 60-record check. M4 then consumed that exact
file: jobs 442 (create, ``k1``), 443 (write, ``n1``), 444 (read, ``n2``) and
finalizer 445 completed; the injected S2 job 446 on ``n1`` and finalizer 447
failed as intended after exact recovery; retry 448 on ``n2`` and finalizer 449
completed. A fresh checker process verified the final ``9,9,9,9`` contents and
all durable records.

The remaining review findings were verified at commit ``b015df3`` on
2026-09-25 after adding the per-invocation and recovery locks. Jobs 450
(create, ``k1``), 451 (write, ``n1``), 452 (read, ``n2``) and finalizer 453
completed; injected S2 job 454 on ``n1`` and finalizer 455 failed only after
exact rollback; retry 456 on ``n2`` and finalizer 457 completed. A fresh
checker verified final ``9,9,9,9`` contents and every durable record. The
ordinary suite separately races two finalizers through one recovery marker;
the physical M2 qualification proves that the same persistent OFD lock used
by that regression excludes peers across these three nodes.

## Physical M5 joint MSv2 prediction and recovery probe

`run_m5_lists.py` is an optional, bounded real-data gate for explicit
`list[MSv2]` writes. Supply two distinct, already selected seed MSs and an
actual nonzero WSClean model FITS file under a unique test root. The harness
never selects or writes the original observation. It deep-copies the seeds
with Casacore, checks that each copy's closure is contained, and removes only
the copies' `MODEL_DATA` to exercise schema-changing prediction. Each mode
has its own copies, model input, and cache. The supplied seeds remain read-only.

Run a fresh M2 qualification on the exact shared mount first. Provision a
shared WSClean executable (including its libraries on every node) and shared
Python interpreters with Casacore and NumPy. Stage the current source; the
worker captures that source rather than using an older checkout. The FITS
model must match the selected real data's phase centre, frequency and image
geometry. An all-zero model cannot pass the scientific check.

For the Kudu/Nyala setup, run inside the controller, adjusting these paths::

    export BASE=/data/issue177-live-20261005-3f0a6b0e
    export SOURCE="$BASE/source"
    export ROOT="$BASE/final"
    export PYTHONPATH="$SOURCE/src:$SOURCE"
    export PYTHON=/data/venv-issue18/bin/python
    export OPTIONS='["-no-reorder","-j","2","-abs-mem","0.2","-size","64","64","-scale","120asec","-pol","I","-data-column","DATA"]'

    srun --partition dev --nodelist k1 --cpus-per-task 2 --mem 2G \
      "$PYTHON" -m tests.slurm_physical.run_m5_lists run \
      --mode native --root "$ROOT" \
      --seed-ms "$ROOT/seeds/part-0.ms" "$ROOT/seeds/part-1.ms" \
      --model-prefix "$ROOT/model/point" \
      --wsclean "$BASE/tools/wsclean/run-wsclean" \
      --wsclean-options-json "$OPTIONS"
    "$PYTHON" -m tests.slurm_physical.run_m5_lists check \
      --mode native --root "$ROOT"

    "$PYTHON" -m tests.slurm_physical.run_m5_lists run \
      --mode slurm --root "$ROOT" \
      --seed-ms "$ROOT/seeds/part-0.ms" "$ROOT/seeds/part-1.ms" \
      --model-prefix "$ROOT/model/point" \
      --wsclean "$BASE/tools/wsclean/run-wsclean" \
      --wsclean-options-json "$OPTIONS" \
      --storage-qualification "$BASE/m2/qualification.json" \
      --nodes k1 n1 n2 --partition dev
    "$PYTHON" -m tests.slurm_physical.run_m5_lists check \
      --mode slurm --root "$ROOT" \
      --storage-qualification "$BASE/m2/qualification.json"

`--model-file` can explicitly select the single `*-model.fits` input without
`--model-prefix`. The
trusted binary adapter derives WSClean's naming prefix from that declared
file and passes both positional MS roots to one exec-form WSClean command.
`--wsclean-options-json`, `--worker-python`, `--casacore-python`, `--timeout`,
`--nodes`, `--partition`, `--cpus-per-task` (default 2), and `--mem`
(default 2G) expose site/runtime differences; the adapter adds
`-predict` and `-name` itself. No core argv policy or custom collection is
introduced. Model files, targets, reports, caches, submissions and ownership
records are contained under the test root.

Launch native mode inside an allocation with at least two CPUs and 2 GiB,
so scientific work runs on a worker rather than in the controller. Native
mode proves actual prediction and an unchanged cache hit. Slurm mode
pins prediction to the first node and its whole-list `OutputRef` read checker
to the second; the unchanged prediction runs cached on the third. Separate
recovery copies start with `SCAN_NUMBER` constants `[101, 202]`. A genuine
worker `os._exit(86)` at S2 after writing `[301, 402]` must finalize failed and
restore both baseline roots. A retry on another node commits `[301, 402]`.
An S3 exit after publishing `[501, 602]` uses a single-leaf workflow: the
immutable success wins over the scheduler's FAILED status. A final unchanged
run must cache-hit with both roots marker-free.

Phase evidence is written immediately under `evidence/`, before later writes
replace those cells. Each phase checks journal heads against immutable producer
identities; S2 rollback must restore both baseline head names. The fresh check
compares current heads with the terminal phase and checks historical snapshots
for earlier phases. The actual whole-list reader artifact, including its cached
reuse, must report the ordered writer roots and match independently read full
column digests. The fresh checker reads actual cells and snapshots,
requires finite, nonzero `MODEL_DATA` in both roots, verifies full column
digests with one-row memory bounds, and protects the original DATA/FLAG/WEIGHT/
UVW/TIME/SCAN_NUMBER values. Recovery expectations are fixed constants, not
values inferred from captured reports. It also checks schema-3 ordered
producer indexes 0/1, mutation roots and state names, immutable execution/
attempt/finalization identities, dataset settlement, released exact ownership,
pinned M2 bytes, and actual `sacct` node/state/exit records. These bounded
checks qualify the supplied data and runtime; they do not establish
full-observation scaling. Record actual successful job identities separately
only after both fresh checks pass. DDFacet prediction can be exercised as a
separate declared joint-write Cab when its image/configuration differs from
WSClean; this harness does not claim a DDFacet run.

Completed live evidence on 2026-10-05 under
`/data/issue177-live-20261005-3f0a6b0e` includes fresh M2 jobs 525 (`k1`),
526 (`n1`) and 527 (`n2`), with all 60 updates and exclusion checks passing.
Actual WSClean native prediction, cached repetition and the fresh checker
passed on two 2016-row MS selections with 506 channels. A separate declared
DDFacet joint-write probe passed native job 534 (`k1`), detached job 535
(`n1`, finalizer 536 on `k1`), and cached job 537 (`n2`, finalizer 538 on
`k1`). Its fresh checker verified protected columns, snapshots, indexed keys,
cleared markers, settlement and released ownership. DDFacet used the first
32 channels, natural weighting and 65 pixels to fit the containers' 64 MiB
shared memory. Independent source-selection and preservation reports verified
the seeds against the original observation and all 95 original files' sizes
and modification times; the approximately 16 GiB original remained untouched.

Final M5 qualification passed on 2026-10-05 in the physical `dev` partition
on `k1`, `n1` and `n2`, under
`/data/issue177-live-20261005-3f0a6b0e/final`. Actual WSClean 3.0 native
job 555 (`k1`) passed prediction, cached repetition and the fresh check.
The complete Slurm sequence used these jobs (finalizers ran on `k1`):

| Phase | Worker jobs | Finalizer | Required result |
| --- | --- | --- | --- |
| Joint prediction and reader | 556 (`k1`), 557 (`n1`) | 558 | Committed prediction and matching reader artifact |
| Cached prediction and reader | 559 (`n2`), 560 (`n1`) | 561 | Both cached |
| Recovery baseline | 562 (`k1`) | 563 | `[101, 202]` committed |
| S2 interruption | 564 (`n1`), exit 86 | 565, exit 1 | Both baseline roots and heads restored |
| Retry | 566 (`n2`) | 567 | `[301, 402]` committed |
| S3 interruption | 568 (`n2`), exit 86 | 569, exit 0 | Immutable commit preserves `[501, 602]` despite failed worker job |
| Cached recovery | 570 (`n1`) | 571 | Cached, with both markers cleared |

`final/m5-native-check.log` and `final/m5-slurm-check.log` both report
`complete: true` and `fresh_check: true`. The fresh Slurm checker verified
all seven phases, actual reader artifacts, frozen permissions, historical
snapshots, terminal heads, settlement, released ownership, pinned M2 evidence
and scheduler node/state/exit records. The independent
`selection-verification.json` proves both 2016-row seeds match their recorded
selections from the original observation; `original-preservation.json`
confirms all 95 original files retained their sizes and modification times
after the scientific runs. The final WSClean gate used all 506 channels;
the separate DDFacet 1.0.0.0 qualification above used 32 channels. Local
fake-predictor smoke checks validate the adapter contract only; the physical
qualification used the actual scientific executables.
