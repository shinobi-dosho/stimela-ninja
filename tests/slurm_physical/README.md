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
      --root /data/physical-m2-001

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
