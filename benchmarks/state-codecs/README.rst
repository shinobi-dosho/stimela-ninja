Issue 16 codec and precision evidence
====================================

These measurements accompany the implementation of Paper I #16.
The two initial review defects in derived-state ownership-release ordering and
symlinked publication paths have been fixed and covered by regression tests.
``evidence/review-probes.json`` and its accompanying source retain the historical
failing reproductions; they describe the implementation before remediation.
Three independent reviewers (Luna, Astra and Sol) found no further actionable
code defects. They inspected the implementation without rerunning live tests;
the post-fix executions below provide that validation.

Before remediation, the full local suite passed **2,742 tests, with 18 skipped**.
After remediation, all **86 state tests** and **56 ownership/storage tests**
passed; four targeted symlink tests also passed after the final store-root guard.
Repository-wide Ruff lint and formatting and ``git diff --check`` passed.
Strict Sphinx built with no warnings. ``evidence/validation.json`` distinguishes
these runs; the retained full-suite JUnit/log are from before remediation.
The benchmark source and dependency hashes were unchanged by that remediation.
These measurements use the earlier msutils pin ``76525da``; subsequent native
profile qualification is recorded separately in ``../native-profile/README.rst``.

Corpus and execution
--------------------

Three serial repetitions each ran on kudu for a deterministic 1,024-row
synthetic column corpus and samples from the supplied real Measurement Set.
The real corpus has 247,752 rows, 1,851 channels and four correlations. Each
repetition sampled all channels from 128-row windows starting at rows 0,
123,812 and 247,624. It includes DATA, MODEL_DATA, CORRECTED_DATA, flags,
weights, UVW, time, antenna IDs, representative string metadata, and all five
saved flag versions. These are sampled-column Zarr measurements, not a full
MSv4 mapping or native-state export.

The original MS and flag versions are preserved in
``kudu:/home/sphemakh/sandbox/shinobiMSv24-16/og-data``. Independent working
copies occupy the workdir root. Code, environment, scratch and results are
under ``issue16-work`` in that workdir. ``evidence/relocation.json`` records
the layout and matching names, modes, sizes and modification times; it is
not a whole-file content checksum comparison. Working-copy sample hashes
match the samples read before relocation.

All 30 profile trials preserved the decoded Zarr logical ID and independently
compared decoded values. All six reports preserve source MS and flag-version
stat fingerprints. ``evidence/verification.json`` records these checks and
the benchmark-source hash comparison against the tree used for these runs.

Measured configurations
-----------------------

Real sampled-column results include Zarr metadata. Throughput is the median
of three repetitions in MiB/s. Copy/encode includes reading the baseline
Zarr and writing the candidate, excluding the logical-ID verification passes;
it is not isolated compressor throughput. Decode reads the complete bounded
sample. RSS includes the fresh worker, imported libraries and sample arrays;
per-column high-water observations within a worker are cumulative.

.. list-table:: Real MS and flag-version samples
   :header-rows: 1

   * - Configuration
     - Stored bytes
     - Copy/encode MiB/s
     - Decode MiB/s
     - Maximum worker RSS MiB
   * - Zstd 3, 32 rows
     - 60,084,649
     - 44.35
     - 381.12
     - 416.60
   * - Zstd 6, 128 rows
     - 60,018,851
     - 63.85
     - 445.51
     - 600.68
   * - Gzip 3, 32 rows
     - 60,246,303
     - 17.84
     - 228.77
     - 600.70
   * - Blosc/Zstd 3, byte shuffle, 32 rows
     - 56,712,551
     - 26.70
     - 384.47
     - 600.71
   * - Blosc/Zstd 3, bit shuffle, 128 rows
     - 50,514,866
     - 70.40
     - 442.17
     - 600.72

For this real sample, Blosc bit shuffle with 128-row chunks was the smallest
tested configuration (15.9% below Zstd 3/32) and had the highest median
copy/encode throughput. Its measured worker memory was higher. The synthetic
sample instead had its smallest representation with byte shuffle/32 rows
(1,474,736 bytes versus 1,492,668 for bit shuffle/128). Therefore no automatic
global default is changed; the explicit profile remains a caller choice.
Chunk size and compression change together in this small matrix, so these
results do not isolate their individual effects. Filesystem caches were warm
or uncontrolled, and no production performance claim is made.

Offline reversible candidates
-----------------------------

PackBits and Delta were tested directly with numcodecs, followed by Zstd 3,
and independently decoded. They remain unavailable as published state codecs
because the pinned msutils verifier does not allow their Zarr codec contracts.
These measurements demonstrate candidate utility, not delivered codec support:

* Real FLAG: PackBits/Zstd used 17,252 bytes versus 25,527 for raw/Zstd.
* Real ANTENNA1: Delta/Zstd used 138 bytes versus 167 for raw/Zstd.
* Real ANTENNA2: Delta/Zstd was worse, 146 bytes versus 108.
* Synthetic ramped IDs: Delta/Zstd used 25 bytes versus 1,612.

These offline streams exclude Zarr node/chunk metadata and must not be
compared directly with whole Zarr representation sizes. Enabling the direct
codecs requires an upstream validated codec contract and a qualified msutils
pin. No redundancy between flag versions is inferred from matching samples.

Full state and precision boundary
--------------------------------

At the benchmark pin ``76525da``, the supplied original MS was refused by
the public native logical-ID API: ``empty-typed-keyword at MAIN/FLAG_CATEGORY``. The receipt is
``evidence/profile-probe.json``. No original-corpus exact round trip is claimed
for these codec runs. The later native-profile extension and expanded real-MS
qualification are described in ``../native-profile/README.rst``.

A separate local six-row supported MS probe exported, verified and
materialized the default, Zstd, Gzip and Blosc profiles. All four preserved
one native state ID and independently matched the restored native ID. The
unchanged native-preservation sidecar used 165,810 bytes in each case.
Complete representations, including both trees and manifests, ranged from
445,930 to 516,782 bytes. The raw report and exact executed probe source are
``evidence/full-representation.json`` and
``evidence/full-representation-probe.py.txt``. This single local probe is
functional and storage evidence, not a throughput comparison with kudu.

Precision reduction produces an explicitly approximate MSv4-only artifact;
native replay and rollback remain unavailable for it. Focused tests exercise
numerical bounds, ties, nonfinite values, signed zero, changed logical
identity, independence from lossless parent representation, corruption,
exact-replay refusal, publication recovery, ownership-release failures and retry,
and symlink rejection for reads, publication and recovery.

Reproduction
------------

Use the benchmark-era measurement-set environment (msutils ``76525da``) and
the exact file hashes in
``evidence/source-sha256.txt``. From the source directory on kudu::

    export TMPDIR=/home/sphemakh/sandbox/shinobiMSv24-16/issue16-work/scratch
    python examples/state_codec_benchmark.py \
        --ms /home/sphemakh/sandbox/shinobiMSv24-16/1783252863_2026-07-20T14-53-01_Jdz-cal-calibrators.ms \
        --flag-versions /home/sphemakh/sandbox/shinobiMSv24-16/1783252863_2026-07-20T14-53-01_Jdz-cal-calibrators.ms.flagversions \
        --workdir "$TMPDIR" --output /path/inside/workdir/new-report.json

Omit the MS and flag-version arguments and use ``--synthetic-rows 1024`` for
the synthetic corpus. Each final report contains input selections, sample
hashes, logical IDs, runtime versions, per-column measurements and per-profile
measurements. ``initial-original-sample.json`` is a diagnostic earlier run
without flag-version/metadata coverage and is used only for sample continuity.
