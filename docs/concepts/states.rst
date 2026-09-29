Reusable Measurement Set states
===============================

``ninja state`` exports a supported contained MSv2 into a local immutable
store and creates a fresh runnable MSv2 from it later. It is separate from
the skip cache and Tier 1 snapshots. Recipes and workers cannot consume
state IDs in v1; automatic capacity tiers, GC and lossy states remain deferred.
Install from a source checkout with ``uv sync --group measurement-set``.
Ordinary imports, help and metadata listing keep optional libraries unloaded.

.. code-block:: console

    $ ninja state --store /data/states export /data/source.ms --block-rows 128
    $ ninja state --store /data/states list --json
    $ ninja state --store /data/states verify 'msutils-logical-hash/v1:<digest>'
    $ ninja state --store /data/states materialize 'msutils-logical-hash/v1:<digest>' /data/new.ms
    $ ninja state --store /data/states recover --destination /data/new.ms

All commands accept ``--json``. ``verify`` and ``materialize`` accept
``--representation sha256:<digest>``; otherwise the lexically first committed
representation is selected and reported. ``list`` checks manifests only,
labels entries ``metadata only``, and refuses invalid/incomplete metadata.

.. code-block:: python

    from shinobi.dataset_state import DatasetStateStore

    store = DatasetStateStore("/data/states")
    exported = store.export("/data/source.ms", block_rows=128)
    store.verify(exported.state_id, exported.representation_id)
    restored = store.materialize(exported.state_id, "/data/new.ms")
    assert restored.fidelity == "exact-logical"

Identity and fidelity
---------------------

The store key is ``msutils.native_logical_id``: ``msutils-logical-hash/v1``
over ``msutils-native-model/v1``. msutils owns scientific hashing,
preservation and native writing. Each state can have several representations:
an xarray-ms MSv4-compatible Zarr v3 tree, a
``msutils-native-preservation/v2`` manifest and its native Zarr payload.
Reconstruction uses the native payload; MSv4 is a bound companion, so much
of the dataset is duplicated.

Native ID, MSv4 ID, payload ID, representation ID, structural signature and
operation UUID are distinct. Native identity is independent of payload chunk
rows. Independent MSv4 exports can differ because ``creation_date`` differs.
Representation digests bind source observations, warnings, date/lineage,
every file/directory, sizes/checksums and actual Zarr node metadata: chunks,
shards, codecs, dtype, shape, dimensions and schema attributes. Node documents
are JSON text, preserving non-finite fill values without normalization.

Manifests reject unknown fields/contracts. Reopening requires the exact
recorded runtime stack and pinned msutils commit, not only nominal version
3.0.0. There is no implicit migration: recovery of a published-but-uncommitted
attempt also needs its original stack, so settle attempts before upgrading.
The MSv4 adapter contract identifies
xarray-ms 0.5.8's schema; no universal upstream schema version is invented.
Verification reads every physical file and both logical trees. This is
integrity, not authentication, for caller-owned local stores.

Fidelity is **exact-logical**, never physical. The writer is msutils' pinned
dask-ms 0.2.32 writer. casacore recomputes ``StandardStMan IndexLength``,
the native model's explicit exclusion. After msutils succeeds Shinobi checks
its existing structural signature and independently recomputes the native
logical root before publication. Consumers needing identical files refuse.

Transactions and recovery
-------------------------

Export takes a shared READ claim in the workflow ownership registry and
checks a contained single-directory source before and after export/capture.
Source changes and pending mutation markers in the configured cache refuse.
Readers coexist; declared writers conflict. External/multi-root closures refuse.

Materialization takes an exclusive CREATE claim for destination and private
sibling staging. The parent must exist. Existing files/directories/dangling
symlinks, overlap with store/cache, and all recorded source paths refuse.
There is no overwrite. msutils runs inside a recorded outer stage;
independent validation and content sync precede Linux
``renameat2(RENAME_NOREPLACE)`` publication. A racing empty directory survives.

New operations write ``shinobi-state-attempt/v2``.  In addition to IDs,
versions, phase events, staging/parent/candidate identities and outcome, each
committed attempt embeds closed ``shinobi-state-provenance/v1`` evidence.  It
keeps the native logical, MSv4, preservation-payload and representation IDs
separate; names the MSv2/closure/mapping/MSv4 profiles and complete software
stack; records exact-logical requested and actual fidelity; and identifies the
managed native-Zarr preservation sidecar and complete evidence coverage.
Closed cache, materialization, replay and transformation decisions distinguish
``store-requested`` from ``selected-representation``, requested work from
validated reconstruction, and explicit ``not-requested`` behavior.  Producer
and replay-runtime stacks remain separate and each complete stack contains the
exact qualified package-key set plus the pinned msutils commit.  Actual fidelity
is absent until reconstruction validates; a refused or failed request cannot
claim successful exact replay.  Legacy
``shinobi-state-attempt/v1`` recovery records remain readable.

Materialization is the reusable-state replay path in v1.  Before writing, it
strictly reopens the selected representation and rejects different MSv2,
closure, mapping, MSv4 or preservation profiles, any software-stack mismatch,
and any fidelity other than exact-logical.  Recipes and detached workers still
cannot consume state IDs; run-manifest replay therefore cannot silently select
or downgrade a reusable state.

Use ``read_state_attempt(result.attempt)`` to validate the authoritative synced
attempt record.  Preflight refusals after the destination parent and store are
available are durable ``refused`` attempts with the raw requested state,
representation and fidelity; operational failures remain distinct ``failed``
attempts. Recovery uses persistent
operation locks and ownership liveness, never PID age or broad globs. It
removes only recorded private staging. After a crash following publication,
it recognizes the candidate inode, revalidates and records success rather
than deleting the destination. An unrelated file or symlink that won the
publication race is preserved while only recorded private staging is cleaned.
Live/uncertain owners or mismatched directory identities refuse. A crash
between stage creation and recording its inode also refuses
automatic deletion. Same-target materialization recovers prior unpublished
dead attempts; already published destinations require explicit recovery.
An unqualified recovery sweep is all-or-nothing: one live, uncertain or
invalid attempt aborts the sweep. Use ``--destination`` to settle an unrelated
target independently.

v1 is qualified for local Linux and cooperative ownership. Out-of-band writers,
hostile path replacement, network-filesystem/worker qualification are outside
the contract. Cleanup is not GC; persistent lock inodes must never be removed.

Profile and costs
-----------------

``fixed-shape-defined-or-empty/v1`` preserves row order, supported fixed-shape
custom columns, units/measures, native descriptors/keywords/references, empty
tables and wholly undefined columns. Ragged cells, mixed definedness and
unsupported structures/typed metadata fail with msutils code and
table/column/row diagnostics. xarray-ms irregular-grid/imputation warnings
are recorded; native reconstruction does not use the padded MSv4 view.
Codec selection and Zarr rewrite commands are not exposed. msutils enforces
its lossless policy. Its residual memory risk applies here: variable-length
string chunks can expand beyond memory during decoding. Provision for full
data reads during verification and duplicate native/MSv4 storage.
