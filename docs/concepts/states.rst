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
over ``msutils-native-model/v1`` or ``/v2``, as selected by the source's
native content. msutils owns scientific hashing,
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

Fresh source observations include supplemental ``table_columns`` evidence
for supported MSv2 subtables. Historical observations omit this optional field;
an empty default stays omitted even in nested representation serialization,
preserving their original canonical representation digests. The evidence does
not enter the frozen structural-signature v1 inputs or scientific logical IDs.

Manifests reject unknown fields/contracts. Reopening requires the exact
recorded runtime stack and pinned msutils commit, not only nominal version
3.0.0. There is no implicit migration: recovery of a published-but-uncommitted
attempt also needs its original stack, so settle attempts before upgrading.
The MSv4 adapter contract identifies
xarray-ms 0.5.8's schema; no universal upstream schema version is invented.
Verification reads every physical file and both logical trees. This is
integrity, not authentication, for caller-owned local stores.

Fidelity is **exact-logical**, never physical. The writer is msutils' pinned
arcae 0.5.4 value writer; python-casacore constructs schema and metadata and
independently verifies the result. dask-ms is no longer required for state
operations. Its historical runtime-record key remains, with ``not-installed``
when absent, so old records remain parseable without bypassing exact-stack
verification. Settle attempts before upgrading and use their original stack
to reopen older states. Keeping v1 IDs and representation serialization does
not bypass the recorded-stack check: an old state still needs its original
qualified runtime for replay. Native profile v2 accepts typed empty column
keywords, uniformly defined zero-length cells and TiledColumn/TiledShape
managers. Casacore can regenerate physical manager fields, which the v2
native model excludes from its logical identity. After msutils succeeds Shinobi checks
its existing structural signature and independently recomputes the native
logical root before publication. Consumers needing identical files refuse.

Transactions and recovery
-------------------------

Export takes a shared READ claim in the workflow ownership registry and
checks a contained single-directory source before and after export/capture.
Source changes and pending mutation markers in the configured cache refuse.
Readers coexist; declared writers conflict. External/multi-root closures refuse.
The v2 dataset closure includes opaque MAIN ``Table:`` keyword members, so
contained extra tables participate in observation and source-change checks.
New state provenance records the observed closure profile; historical v1
state records remain readable. Export and materialization still require the
state adapter's supported table schemas and exact logical validation.

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
is absent until reconstruction validates. A refusal before validation cannot
claim successful exact replay; a later publication failure may retain the
validated staged-reconstruction evidence while its attempt remains failed. Legacy
``shinobi-state-attempt/v1`` recovery records remain readable.

Materialization is the reusable-state replay path for the paired
``fixed-shape-defined-or-empty/v1`` and ``/v2`` native profiles. Before writing, it
strictly reopens the selected representation and rejects different MSv2,
closure, mapping, MSv4 or preservation profiles, any software-stack mismatch,
and any fidelity other than exact-logical.  Recipes and detached workers still
cannot consume state IDs; run-manifest replay therefore cannot silently select
or downgrade a reusable state. The exact package-key set is part of the v2
attempt and representation reader contract: changing ``adapter.PACKAGES``
requires a compatible reader or migration first, otherwise existing v2 records
and representations are intentionally refused and unfinished attempts cannot
be settled automatically.

Use ``read_state_attempt(result.attempt)`` to validate the authoritative synced
attempt record. Once a materialization attempt has been created, deliberate
contract refusals are durable ``refused`` attempts with the raw requested
state, representation and fidelity; failures before attempt creation cannot be
recorded, and unexpected operational exceptions remain distinct ``failed``
attempts. Recovery uses persistent
operation locks and ownership liveness, never PID age or broad globs. It
removes only recorded private staging. After a crash following publication,
it recognizes the candidate inode, revalidates and records success rather
than deleting the destination. An unrelated file or symlink that won the
publication race is preserved while only recorded private staging is cleaned.
Live/uncertain owners or mismatched directory identities refuse. A crash
between stage creation and recording its inode also refuses
automatic deletion. Same-target materialization recovers prior unpublished
dead attempts before checking its destination. Use explicit recovery to settle
an already published destination after an interrupted materialization.
An unqualified recovery sweep is all-or-nothing: one live, uncertain or
invalid attempt aborts the sweep. Use ``--destination`` to settle an unrelated
target independently.

v1 is qualified for local Linux and cooperative ownership. Out-of-band writers,
hostile path replacement, network-filesystem/worker qualification are outside
the contract. Cleanup is not GC; persistent lock inodes must never be removed.

Profile and costs
-----------------

Both native profiles preserve row order, supported fixed-shape custom columns,
units/measures, native descriptors/keywords/references, empty tables and wholly
undefined columns. V2 additionally accepts typed empty keywords, defined
zero-length cells and supported tiled managers. Ragged cells, mixed definedness
and unsupported structures/typed metadata fail with msutils code and
table/column/row diagnostics. xarray-ms irregular-grid/imputation warnings
are recorded; native reconstruction does not use the padded MSv4 view.

Explicit MSv4 codec and precision artifacts
-------------------------------------------

``export`` accepts an optional named lossless profile: ``--codec zstd|gzip|blosc-zstd``,
``--level``, ``--chunk-rows`` and, for Blosc only, ``--shuffle none|byte|bit``.
For native profile v1, the historical export keeps its v1 representation
manifest; an explicit lossless selection uses ``shinobi-state-representation/v2``.
Native profile v2 uses ``shinobi-state-representation/v3`` with explicit
preservation profile/model fields, whether or not a lossless layout is selected.
No new fields are serialized into historical v1/v2 representation manifests,
so their representation digests remain unchanged. An explicit selection rewrites
the MSv4 tree inside private export staging. It checks the public
``msutils.logical_id`` before and after rewriting, then captures the usual
native preservation sidecar against the final MSv4 tree. Numeric copies use
bounded N-dimensional tiles. Variable-length strings retain the upstream
decode expansion risk. The native payload retains msutils' existing codec;
the profile describes the companion MSv4 tree only.

``derive-precision`` creates a separate approximate MSv4 artifact under
``derived/``. Its closed ``mantissa-round-nearest-even/v1`` policy names exact
MSv4 array paths with repeated ``--array`` options and gives ``--keep-bits``,
``--atol`` and ``--rtol``. Only floating and complex arrays are eligible.
Each finite value must satisfy ``abs(new-old) <= atol + rtol*abs(old)``;
complex magnitudes use the same inequality. Non-finite bit patterns and signed
zero are preserved. Unselected arrays retain their exact values. A no-change
derivation refuses. The derived logical ID binds the exact native state ID,
exact parent MSv4 logical ID, policy and transformed MSv4 logical ID, while
the physical record names the source representation. Recompression of the
same exact content therefore keeps one derived logical identity.

``verify-derived``, ``list-derived`` and ``recover-derived`` operate only on
that separate namespace. Verification rereads the exact parent, checks every
physical byte and node, redoes numerical rounding and bounds, and compares
untouched arrays. Approximate artifacts have no native preservation sidecar
or native replay path. ``materialize`` refuses a derived ID before creating a
destination. Recipes, workers, skip cache and snapshot rollback continue to
consume only exact states.

Provision for full data reads during verification and duplicate native/MSv4
storage. Existing exact states can carry multiple lossless representations;
derived artifacts add separate storage and are opt-in.
