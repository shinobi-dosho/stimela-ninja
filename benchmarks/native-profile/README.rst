Real-MS native profile qualification
====================================

This acceptance record accompanies `msutils PR #63
<https://github.com/shinobi-dosho/msutils/pull/63>`_ at commit
``0e9b136c52e7798b2662228293a60a978d007aa4`` and its Shinobi integration.
It supersedes the native-profile refusal recorded during the earlier codec
benchmarks; those historical measurements and their source pin are unchanged.

The corpus contains five Measurement Sets spanning two observations and their
processing variants, rather than five independent observations:

* Original sandbox calibrator MS: 247,752 rows, 1,851 channels, four correlations.
* July J0408-6545: 247,752 rows, 450 channels, four correlations.
* July J0410-65off: 98,568 rows, 1,351 channels, four correlations.
* Chi Oph averaged: 1,507,391 rows, 128 channels, two correlations; includes
  SIGMA_SPECTRUM and CORRECTED_RESIDUAL.
* Chi Oph cross-calibrators: 494,479 rows, 1,024 channels, two correlations.

The four additional source paths were supplied by the maintainer. Sources are
opened read-only. All generated MSs, bundles, source snapshots and scratch on
kudu live under ``/home/sphemakh/sandbox/shinobiMSv24-16/issue16-work/profile-v2``
(or its sibling ``scratch`` directory), not under ``/tmp``.

What is tested
--------------

Each sampled round trip selects seven MAIN rows from the beginning, middle and
end, deep-copies the referenced subtables, captures the native preservation
bundle, verifies it, and moves the sampled source away before reconstruction.
The restored native ID must match the captured ID. Read-back also checks the
empty CATEGORY keyword, undefined FLAG_CATEGORY cells and defined float64
REST_FREQUENCY cells of shape (0,).

All five corrected-candidate sample round trips passed. These tests exercise
the native preservation layer with a minimal bound Zarr companion; they are
not full MSv4 mapping or full-volume reconstruction tests. Deep copying the
sample may rebuild physical manager bookkeeping, which is explicitly outside
the v2 logical identity. Full-source native-ID scans separately exercise the
original manager records and every native cell, including source-change checks.
All five final full-source scans passed under the committed implementation.

``evidence/qualify.py.txt`` is the executed qualification script. Each
``final-sample-*.json`` or ``final-scan-*.json`` report records its source path,
mode, identity, elapsed time and implementation SHA-256. Sample reports use the
corrected executable candidate; the final commit differs only in docstrings
and comments. ``upstream-validation.json`` records both source hashes.
Elapsed times are functional test timings, not controlled throughput results.

Compatibility and precision
---------------------------

Newly supported content selects ``fixed-shape-defined-or-empty/v2`` and
``msutils-native-model/v2``. Empty numeric keyword dtypes remain native; a bare
empty list is accepted only after native metadata establishes a string array.
Defined-empty cells retain dtype, shape and definedness. Tiled manager identity
retains declared configuration while excluding validated physical bookkeeping
that casacore regenerates. Unsupported manager features still refuse.

Previously supported content retains its v1 identity and interpretation.
``legacy-compatibility.json`` records verification and restoration of a bundle
created with the previous implementation, with an unchanged native ID. The
upstream suite passed 516 tests with two existing arcae-only capability xfails;
exact-native uses casacore for those empty metadata/cell cases. Additional
independent restores preserved defined-empty shapes (0, 2), (2, 0), and (2, 0, 3).

Shinobi integration
-------------------

The full downstream suite passed 2,749 tests with 18 skips. A fresh environment
installed the published GitHub commit through the updated lockfile, passed the
runtime qualifier, and passed the combined v2 live export/verify/restore test
for both plain and lossless representations. Full-tree lint/format and the
Sphinx warnings-as-errors build passed. Three independent integration reviews
reported no actionable findings. ``evidence/downstream-validation.json`` records
the test scopes; upstream CI also passed on Python 3.11, 3.12 and 3.13.
