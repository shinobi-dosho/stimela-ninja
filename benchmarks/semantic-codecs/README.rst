Semantic codecs and full-state qualification
============================================

This evidence accompanies `Shinobi PR #206
<https://github.com/shinobi-dosho/stimela-ninja/pull/206>`_ and `msutils PR #64
<https://github.com/shinobi-dosho/msutils/pull/64>`_ for `Paper I #16
<https://github.com/shinobi-dosho/paper_shinobi_I/issues/16>`_. The sampled benchmarks ran Shinobi
``ca5780914a3560d502f10d1105070857c8ced2f9`` with msutils
``2856d3ae357c4c2ec70060df641fe48e32ff8288``. Full-volume captures use Shinobi
``d15bab8ad4b5598dd8c52f5c00fabec16dd72d51``; the fresh large-ledger Chi Oph
capture uses ``0b2164db12a1865a90c2a17cbe2c6971de79e82a``. Final replay
qualification uses ``6576e5cc5af5453227e7c711a466a42f22d96791``. These runs all
use public msutils ``25c280a8ab412b5015a80bf4ff6f83b7a6820675`` with identical
recorded runtime versions. Capture and replay revisions are recorded separately. The intermediate chunk-correction
run used ``eb5cec47bdfa7c5361f4afae8f8a3cf6711a0676`` with the earlier pin;
its failures remain diagnostic evidence. Each receipt records its actual
executed implementation.

The opt-in profiles are ``packbits-zstd``, ``delta-zstd`` and
``packbits-delta-zstd``. PackBits applies to booleans and Delta to unchanged
integer dtypes, in both the MSv4 companion and exact native payload. Native
FLAG is boolean; the qualified mapper emits uint8 FLAG, which retains its
type and is eligible for Delta. Floating and complex arrays remain exact.
Existing defaults and the separately derived approximate-state policy are
unchanged. Unsafe dtype narrowing, codec composition and Delta byte order
remain refused before decoding.

Corpus and benchmark method
---------------------------

The five full source MSs span two observations and their processing variants,
not five independent observations:

* Original sandbox calibrators: 247,752 rows, 1,851 channels, four correlations.
* July J0408-6545: 247,752 rows, 450 channels, four correlations.
* July J0410-65off: 98,568 rows, 1,351 channels, four correlations.
* Chi Oph averaged: 1,507,391 rows, 128 channels, two correlations.
* Chi Oph cross-calibrators: 494,479 rows, 1,024 channels, two correlations.

All 90 sampled-column trials passed: five profiles, three fresh-worker
repetitions, and six corpora (the five real MSs plus a deterministic 1,024-row
synthetic corpus). Real samples include all channels from 128-row windows at
the beginning, middle and end. The original also includes all five saved flag
versions; no redundancy between those versions is assumed. Available visibility,
weight, sigma, coordinate, flag, integer-identifier and string metadata columns
are recorded in each report. The Chi Oph sample includes CORRECTED_RESIDUAL
and SIGMA_SPECTRUM.

The matrix holds Zstd at level 3 and compares 32-row versus 128-row chunks,
then PackBits, Delta and both at 32 rows. Each trial checks decoded values
independently and preserves the baseline logical ID. Repeated encoded byte
counts agree. Source and flag-version stat fingerprints remain unchanged;
these fingerprints cover names, modes, sizes and mtimes, not whole-file
physical content checksums. Selected array bytes have separate SHA-256 hashes.

``benchmark-metrics.csv`` records all 30 corpus/profile summaries;
``evidence/benchmark-summary.json`` adds column-class totals. Throughput is
the median of three serial repetitions. Copy/encode includes reading baseline
Zarr and writing the candidate, excluding the logical-ID verification passes.
Decode reads the bounded sample. Stored bytes include array metadata. RSS is
the maximum observed worker high-water across repetitions, including libraries,
input and decoded arrays. Per-column high-water is cumulative within that
worker. Filesystem caches are warm or uncontrolled; these are not isolated
compressor or production-throughput measurements.

Codec selection evidence
------------------------

The following byte totals hold chunking and Zstd level constant (32 rows,
level 3) and sum eligible column directories including their metadata:

.. list-table:: Semantic column-class storage
   :header-rows: 1

   * - Corpus
     - Booleans: Zstd
     - Booleans: PackBits/Zstd
     - Integers: Zstd
     - Integers: Delta/Zstd
   * - synthetic
     - 48,777
     - 30,994
     - 6,829
     - 5,167
   * - original
     - 197,962
     - 129,654
     - 10,162
     - 11,027
   * - j0408
     - 21,440
     - 14,134
     - 10,162
     - 11,027
   * - j0410
     - 39,562
     - 25,905
     - 10,162
     - 11,027
   * - chioph_avg
     - 2,885
     - 2,538
     - 10,222
     - 10,992
   * - xc_calibrators
     - 9,056
     - 6,757
     - 10,222
     - 10,992

PackBits reduces aggregate boolean-column storage by about 12–35% in the real
samples. Delta helps the synthetic ramped-ID case but increases aggregate
integer-column storage in every real sample here; individual identifiers can
still improve. Combined filtering is therefore not a universal space win.
Larger chunks improve measured copy/encode throughput in these samples but
are not always the smallest layout. These results justify explicit selection
and retaining the existing default, rather than choosing one codec globally.

Full-volume qualification
-------------------------

All five datasets completed final-code qualification for both profiles:
source-absent verification, restoration, independent all-cell comparison and
final source audits all passed. This covers 2,595,942 MAIN rows, ten restored
MSs and fourteen tables per restoration. The independent acceptance checker
verified all five complete receipts against the recorded code revisions,
exact runtime, codec metadata, native identities and cancellation evidence.
``evidence/qualification-summary.json`` records representation sizes, inventory
counts, stage times and peak RSS; ``evidence/replay-v5-*.json`` retain the full
receipts. Final remote checks found all ten v3 materializations committed,
empty ownership registries, clean frozen checkouts, and every recorded source
path absent with its archived working copy retained in ``og-data``.

This is separate from the completed sampled-column matrix. Saved flag-version
files are covered by that sample matrix; the full native round trips cover the
MS table trees, not their adjacent flag-version directories.

Each dataset runs seven fresh-process stages: prepare, baseline export, combined
export, archive, verify, materialize and compare. Preparation reads every native
cell and verifies a deep working copy. Both exports use Zstd level 3 and 32-row
chunks; the second adds semantic filters. Only the generated source copy moves
to the workdir's ``og-data`` directory. Verification and replay require its
recorded path to be absent. Both restored MSs are compared against the original
using independently read native metadata, definedness and every defined value.
Numeric values compare as exact bytes, including signed zeros and NaN payloads.
The native profile's declared physical-manager exclusions remain exclusions.

The final continuation runs audit, source-absent verification, separate baseline
and combined restoration, separate full comparisons, and a final source audit.
For Chi Oph averaged and the original sandbox MS, both exports committed and the generated copy was archived
before the obsolete capture-code verify worker was stopped. The operator receipt
records the exact worker/controller identities, successful pidfd SIGINT delivery,
and confirmed exits. Each original report retains the interrupted verify result;
that stage is explicitly superseded, not counted as passed. The continuation
performs the complete verification again under the final replay revision.

The v3 captures began with all five datasets running concurrently (the earlier
diagnostic runs used up to three). The fresh Chi Oph capture and final replay
continuations overlap remaining capture work. Stage times and process peak RSS are
functional qualification measurements, with shared I/O and uncontrolled caches;
they are not controlled codec-throughput comparisons. A 16 MiB bound applies
to individual numeric reads and rewrite tiles, not aggregate process memory.
Variable-length strings can expand beyond that bound.

The initial full-volume run refused all five exports before publication: the
mapped secondary CORRECTED_DATA array retained a whole-partition chunk despite
a preferred-chunk hint. The correction passes explicit Xarray chunks as well
as backend hints, because secondary variables lack the encoding used by primary
variables. A real lazy-backend regression checks bounded reads before allocation.
The corrected metadata preflight passed for every numeric array in all five MSs.
The initial failure reports and execution receipt are retained as
``evidence/qualification-initial-*.json``; they are not successful qualifications.

The next run completed both full exports and source-absent verification for
J0408 and J0410, then failed native restoration: their raw descriptors retain a
``TiledSigmaSpectrum`` hypercolumn naming the removed ``SIGMA_SPECTRUM`` column.
This affects four of the five source MSs. The writer correction reconstructs
that exact observable schema using a temporary column in a private zero-row
table, removes it, and requires unchanged typed descriptor and canonical
manager identity before adding rows. No hash, profile or capture contract is
changed. Other orphan roles or dimensions remain explicitly unqualified.

All five real MAIN schema probes passed under the corrected public pin. These
are zero-row schema checks, not native-content replay. The earlier diagnostic
run's other three workers were interrupted after the defect was established;
``evidence/qualification-v2-interruption.json`` distinguishes those cancellations
from the two actual restore failures. Their ownership registry was empty after
interruption. Complete fresh runs use the fixed public stack because stored
states retain the exact-stack requirement; no compatibility exception was added.

The v3 Chi Oph averaged export then exceeded Shinobi's 64 MiB physical
manifest budget. Its complete per-chunk ledger grows with the row count and
selected chunk sizes. The correction uses compact JSON and a separate finite
256 MiB physical-manifest budget; ordinary node metadata retains 64 MiB.
A 400,000-entry regression loads both 69.6 MB compact and 86.4 MB pretty
manifests without dropping entries or changing their canonical digest. The
serialized limit does not bound decoded JSON/Pydantic memory. The fresh Chi Oph capture and final replay passed with 523,724 inventory
entries per representation; its baseline manifest is 71,611,966 bytes.

Both July datasets passed full baseline/combined exports and source-absent
verification but failed Shinobi's final materialization postcondition. A
preserved full J0408 diagnostic restoration has the exact original native ID.
Only MAIN's physical ``table.f*`` filenames differ: casacore renumbers storage
files. All other frozen structural-signature inputs match. The correction
leaves the frozen signature unchanged and compares reusable-state replay
structure while omitting only those filenames. It still requires every other
structural input and the independently recomputed native ID to match. New
materialization attempts use v3 and preserve the source signature alongside
a separate restored signature. Recovery checks the restored signature;
historical attempts retain their original serialization and checks.

The final continuation driver reads existing committed representations under
their exact recorded runtime, verifies them with the recorded source path
absent, restores to new destinations and compares every native cell against
the archived working source. It preserves old receipts and manifests, records
capture and replay code separately, and checks original/archive fingerprints
and native IDs. Every parent/stage process checks actual loaded Shinobi module
paths and hashes against the recorded checkout. The seven provenance regressions
include an identical but wrongly installed checkout and mixed module paths.
These checks and all five complete continuations passed. The v2 driver also
checks committed capture prerequisites and explicit old-verification disposition
(25 regressions). The exact-process handoff helper has 15 regressions and two
recorded live handoffs; the final acceptance checker has 18 regressions. All
three independent reviewers cleared each final helper and checker.

Validation and boundaries
-------------------------

The final upstream tree passed 567 tests with two existing expected capability
failures. Its 15 final orphan-hypercolumn cases passed on Python 3.11 and 3.12.
Its public CI passed Python 3.11, 3.12 and 3.13, bare-install and docs jobs.
The final downstream replay implementation passed 2,807 tests with 18 skips, including the
qualification comparator and lazy-backend chunk-bound regressions. The secondary-column
correction passed 50 focused tests on Python 3.12 and 33 on Python 3.11. Both ordinary
and Measurement Set CI jobs passed on both versions, along with the dependency
audit. The corrected public-pin installation also passed all 50 focused tests. Focused state tests, optional-dependency skips, strict
Sphinx builds, whole-tree lint/format and fresh public-pin runtime qualification
also passed. The final replay core run passed 147 tests, including all 26
new replay regressions; the manifest-limit suite passed 16 tests. One sandboxed
full-suite attempt stalled in live Zarr I/O and was interrupted; its log is
retained as diagnostic evidence, not a passing result. The final full suite
used the required network-enabled execution environment.
``evidence/validation.json`` distinguishes these runs and links
their JUnit receipts.

Three final independent reviews (Luna high, Astra low, Sol medium) found no
actionable findings after fixing qualification comparison allocation: numeric
cell shape/type is inspected before reading, oversized cells refuse, and
casacore uchar widening to uint16 is included in the allocation bound. This
qualification-only 16 MiB numeric-cell limit does not narrow native preservation.
Variable-length strings retain their allocation-expansion risk. Final reviews
also covered the large-ledger and versioned replay corrections. A confirmed
continuation-driver import-provenance gap was fixed and all three reviewers
reported no actionable findings on repeat review. Big-endian
execution was not exercised; unsafe Delta serializer combinations have direct
refusal coverage.

The full export receipts retain mapper warnings. These include missing rows in
the rectangular time/baseline view, an empty PROCESSOR table, and unsupported
UVW-frame metadata mapped using field-direction metadata. Padded mapped cells
and substituted companion metadata do not redefine the native contract: exact
replay uses the bound native payload and must recover the original contents.
This evidence does not qualify MSv4-only reconstruction or scientific use of
the mapper's imputed metadata.

Independent xarray-ms exports include fresh creation timestamps, so native
and payload IDs are compared across exports. Each lossless rewrite proves
MSv4 ID equality against its own input tree. No hashing fields are normalized
away for this qualification. Approximate states remain separate, with named
error policies and lineage, and cannot satisfy native replay or rollback.

Reproduction
------------

Install the recorded source and locked measurement-set group. Run
``examples/state_codec_benchmark.py --ms SOURCE --workdir SCRATCH --output REPORT``
with three repetitions (the default); add ``--flag-versions FLAGS`` for the
original or omit ``--ms`` for the synthetic corpus. The execution receipt
records exact commands, versions, source hashes and selections.

For full qualification, run ``examples/state_codec_qualification.py --source
SOURCE --workdir NEW_WORKDIR``. The workdir must be disjoint from the source.
The script deep-copies the source, exports both profiles, moves only its own
generated copy into ``og-data``, then verifies, materializes and independently
compares native contents in fresh processes. Original datasets are never moved.

On kudu, all work is under
``/home/sphemakh/sandbox/shinobiMSv24-16/issue16-work/semantic-codecs-20261012``
and TMPDIR is its sibling ``issue16-work/scratch``. No remote ``/tmp`` is used.
The earlier ``../state-codecs`` and ``../native-profile`` records retain their
historical pins and narrower evidence scopes.
