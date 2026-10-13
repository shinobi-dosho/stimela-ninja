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

Ordinary metadata files retain a 64 MiB serialized limit. Only the physical
``representation.json`` ledger has a separate finite 256 MiB limit, allowing
full datasets with many independently hashed chunks. New physical manifests
use compact JSON; historical pretty JSON remains readable within the physical
limit. Formatting does not change the canonical model digest or omit ledger
entries. Reads are bounded to the selected limit plus one byte and reject
nonregular files and duplicate JSON keys before strict model validation.
An oversized physical manifest refuses while export is still private staging.
Decoded JSON and Pydantic ledger objects can consume substantially more memory
than the serialized file; this byte budget is not a peak-memory guarantee.

Fresh source observations include supplemental ``table_columns`` evidence
for supported MSv2 subtables. Historical observations omit this optional field;
an empty default stays omitted even in nested representation serialization,
preserving their original canonical representation digests. The evidence does
not enter the frozen structural-signature v1 inputs or scientific logical IDs.

Manifests reject unknown fields/contracts. Reopening requires the exact
recorded runtime stack and exact installed msutils Git commit, not only its
nominal version. The development group tracks upstream ``main``; the lockfile
resolves it to a commit, and provenance records that installed commit. There
is no implicit migration: recovery of a published-but-uncommitted attempt also
needs its original stack, so settle attempts before upgrading.
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
the frozen structural document and independently recomputes the native
logical root before publication. Only this reusable-state replay comparison
omits per-resource table filenames, which casacore may regenerate when it
renumbers storage managers. Resource paths, members, manager types, profiles,
status, rows, columns, keywords and subtables remain exact. The frozen observed
structural signature, mutation snapshots and their recovery rules are unchanged.
Consumers needing identical files refuse.

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

Exports continue to write ``shinobi-state-attempt/v2``. New materializations
write ``shinobi-state-attempt/v3`` and keep the source structural signature in
both the attempt and its ``shinobi-state-provenance/v1`` record. A separate
``restored_structural_signature`` records the observed candidate's frozen
signature before publication. It may equal the source signature. The field is
omitted while absent; historical v1/v2 attempts reject a non-null restored
signature and retain their serialized shape and exact-signature recovery.
Ready, published and committed v3 attempts require both signatures, validated
exact replay provenance, complete versions and candidate/state identities.
In addition to IDs, versions, phase events, staging/parent/candidate identities
and outcome, each
committed attempt embeds closed ``shinobi-state-provenance/v1`` evidence.  It
keeps the native logical, MSv4, preservation-payload and representation IDs
separate; names the MSv2/closure/mapping/MSv4 profiles and complete software
stack; records exact-logical requested and actual fidelity; and identifies the
managed native-Zarr preservation sidecar and complete evidence coverage.
Closed cache, materialization, replay and transformation decisions distinguish
``store-requested`` from ``selected-representation``, requested work from
validated reconstruction, and explicit ``not-requested`` behavior.  Producer
and replay-runtime stacks remain separate and each complete stack contains the
exact qualified package-key set plus the installed msutils Git commit. Actual
fidelity is absent until reconstruction validates. A refusal before validation cannot
claim successful exact replay; a later publication failure may retain the
validated staged-reconstruction evidence while its attempt remains failed. Legacy
``shinobi-state-attempt/v1`` recovery records remain readable.

Materialization is the reusable-state replay path for the paired
``fixed-shape-defined-or-empty/v1`` and ``/v2`` native profiles. Before writing, it
strictly reopens the selected representation and rejects different MSv2,
closure, mapping, MSv4 or preservation profiles, any software-stack mismatch,
and any fidelity other than exact-logical.  Recipes and detached workers still
cannot consume state IDs; run-manifest replay therefore cannot silently select
or downgrade a reusable state. The exact package-key set is part of the v2/v3
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
it recognizes the candidate inode and revalidates the selected representation.
For v3, recovery also repeats the replay structural comparison and native ID
check and requires the observed destination signature to equal the recorded
restored signature. It records success rather
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

``export`` accepts an optional named lossless profile: ``--codec zstd|gzip|blosc-zstd|packbits-zstd|delta-zstd|packbits-delta-zstd``,
``--level``, ``--chunk-rows`` and, for Blosc only, ``--shuffle none|byte|bit``.
PackBits applies only to boolean arrays; Delta applies only to signed and
unsigned integer arrays. The combined choice applies each where eligible.
Other array types use Zstd alone. These are explicit selections; the default
export still uses its historical layout. Semantic selections write C-order
chunks explicitly and preserve the array's native dtype.
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
decode expansion risk. The original ``zstd``, ``gzip`` and ``blosc-zstd``
profiles retain msutils' existing native payload layout and describe the
companion MSv4 tree only. The semantic profiles select filters for both trees
as described below.

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

The semantic lossless profiles also apply to the exact native preservation
payload: ``packbits-zstd`` packs native boolean columns, ``delta-zstd`` filters
native integers, and ``packbits-delta-zstd`` selects both. Every other dtype
retains Zstd. The mapped MSv4 ``FLAG`` column is uint8 in the qualified exporter;
it retains that type and is eligible for Delta, while the native boolean FLAG
payload is eligible for PackBits. Neither path casts flags or changes logical
identity. Export reads use an explicit Xarray chunk map together with backend
``preferred_chunks`` before any array is computed. The explicit map also bounds
secondary columns, which lack backend preferred-chunk hints. Numeric chunks
are limited to 16 MiB.

``examples/state_codec_qualification.py`` runs full export, independent fresh
verification and replay, and casacore comparisons in separate processes. It
makes its own deep working copy and archives only that generated copy before
replay, proving that its recorded source path is absent. The original source
is retained for independent comparison. Reports retain timings, RSS, actual
codec metadata, runtime origins, source observations, and failure tracebacks.
