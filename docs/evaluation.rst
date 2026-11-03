Benchmarking and evaluation
===========================

The following checks cover storage efficiency, reconstruction correctness and
execution reliability. This summary describes evaluation scope and acceptance
criteria. Measurements below describe specific test conditions and do not
certify a deployment.

.. list-table:: Evaluation coverage
   :header-rows: 1
   :widths: 20 45 35

   * - Area
     - What is evaluated
     - Interpretation and limits
   * - Lossless codecs
     - Compare encoded size, encode/decode throughput and peak process memory
       on synthetic columns or bounded Measurement Set samples. Check decoded
       values and logical identity.
     - Sample results depend on data, chunk layout and filesystem caching;
       they do not establish full-workflow throughput.
   * - Exact state reconstruction
     - Export, verify and restore a Measurement Set in separate processes;
       compare restored content with the original while the recorded source
       path is absent.
     - Applies to the supported native profile and tested dataset features.
       Stage timings include storage and verification work.
   * - Approximate precision
     - Check numerical error bounds, signed zeros, non-finite bit patterns
       and separation from exact-state identity.
     - Numerical bounds do not establish scientific suitability. Approximate
       artifacts cannot be used for native replay.
   * - Monitoring scale
     - Exercise a 1,740-leaf regression fixture with increasing log volume;
       bound summary size, metadata size and monitoring memory.
     - Synthetic regression budgets are not deployment capacity guarantees;
       scheduler accounting is mocked.
   * - Shared storage
     - Probe cross-node locking and filesystem behavior on the deployment's
       actual shared mount.
     - A successful probe qualifies that mount and configuration only.
   * - Detached execution and recovery
     - Exercise Slurm execution, injected failures, rollback, committed-result
       preservation and requeues with real Measurement Set or FITS workflows.
     - Requires the relevant cluster, container runtime and data dependencies;
       local tests alone do not qualify a site.

Monitoring measurements
------------------------

The fresh-process monitoring fixture used 1,740 leaves, nested schemas and typed
output tables on Python 3.14/Linux, with a 670 KB frozen bundle. Only scheduler
accounting was mocked; no live cluster was required.

.. list-table:: Recorded monitoring results
   :header-rows: 1

   * - Log per leaf
     - Total log bytes
     - Node/schema JSON bytes
     - Final summary bytes
     - Monitoring peak RSS
     - Status + manifest time
   * - 64 KiB
     - 114,032,640
     - 5,439,352
     - 5,548
     - 63.9 MiB
     - 1.57 s
   * - 256 KiB
     - 456,130,560
     - 5,441,092
     - 5,548
     - 63.6 MiB
     - 5.15 s

These are recorded fixture measurements, not results rerun for this page.
They illustrate monitoring behavior as log volume grows; timings depend on
the host and storage. They do not measure scientific processing throughput
or live scheduler performance.

Reproducing checks
------------------

``examples/state_codec_benchmark.py`` provides synthetic and sampled-column
codec measurements. ``examples/state_codec_qualification.py`` checks complete
export and reconstruction. See :doc:`concepts/states` for the supported state
and precision contracts.

``tests/test_state_precision.py`` checks precision and codec contracts.
``tests/test_offload_record_store.py`` runs the monitoring regression fixture;
its size and memory budgets are described in :doc:`offloading`.

For physical cluster checks, use the drivers documented in
``tests/slurm_physical/README.md`` and follow the deployment requirements in
:doc:`offloading` and :doc:`concepts/config`. Keep generated reports outside
the source tree.
