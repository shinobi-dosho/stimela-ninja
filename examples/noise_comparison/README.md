# Thermal-noise comparison: one declared workflow

`noise_pipeline.py:recipe` declares one outer telescope/declination zip scatter.
Each of its 30 branches creates one strict MS, then locally scatters WSClean,
DDFacet, CASA and CASA FITS export over 14 robustness/band-mode combinations.
The six telescopes, five declinations, seven Briggs values, two spectral modes
and three imagers produce 1,260 image/PSF pairs. Native per-branch inventory
steps require all six image/PSF files and CASA metadata for each combination.
Root outputs are ordered MS paths and `products.json` inventory paths; detached
finalization writes `results/<attempt-uuid>/manifest.json` with the complete
observations and outputs. Use `shinobi.offload.mixed.current_manifest(submission)`
to select the latest terminal result, including after a finalizer requeue.

Both commands consume the same ordinary Recipe. From the repository root,
configure `settings.json` as described below, then inspect or compile it:

```sh
example_root="$PWD/examples/noise_comparison"
export PYTHONPATH="$example_root${PYTHONPATH:+:$PYTHONPATH}"
export NOISE_SETTINGS="$example_root/settings.json"
ninja --config "$example_root/config.json" run noise_pipeline:recipe --dryrun
ninja --config "$example_root/config.json" compile noise_pipeline:recipe \
  --workdir /data/noise-snapshot --code-root "$example_root"
ninja --config "$example_root/config.json" compile noise_pipeline:recipe \
  --submit --workdir /data/noise-snapshot --code-root "$example_root" \
  --worker-python /data/env/bin/python
```

To execute locally instead, use a separate existing workspace:

```sh
(cd /data/noise-local && ninja --config "$example_root/config.json" run noise_pipeline:recipe)
```

For the submitted run, the printed handle works with `ninja status HANDLE`,
`ninja logs HANDLE`, and `ninja runs --workdir /data/noise-snapshot` (subject
to the large-record monitoring limit below).

Use an existing writable shared workspace and a preinstalled matching worker
Python. Set `NOISE_SETTINGS=/absolute/path/settings.json` to select reviewed
settings. Layout paths resolve relative to that JSON file. Set `images` to
absolute shared SIF paths under keys `simms`, `wsclean`, `ddfacet`, `casa-tasks`.
The supplied `config.json` selects Apptainer and enables caching, sandboxing
and provenance. Install the `ms` and `fits` extras in the submitting and worker
Python environments for dataset and product validation. The native inventory
runs on the worker Python. For Slurm, `sbatch_opts` names site routing/resources,
e.g. `{"partition":"dev","cpus-per-task":"8","mem":"32G","time":"04:00:00"}`;
`max_parallel` limits active branches. `max_workers` limits local imaging tasks
inside a branch; `threads` sets per-tool concurrency. No custom launcher or
manual output-directory scaffolding is needed.

The supplied defaults retain four hours (`duration_seconds=14400`,
`integration_seconds=32`, 450 integrations). For the separately reviewed
8-minute snapshot use `duration_seconds=480`, `integration_seconds=8` (60
integrations). A null `start_hour_angle` centres either duration on transit:
-2 hours for four hours, -1/15 hour for the snapshot. Use a fresh workspace to
keep the existing four-hour evidence separate. The example retains the
imagers' original rank-filter settings; filter-disabled controls are separate
experiments.

Dependencies: Dosho must include merged PR #106 (`DirtyPSF`); the older
0.2.0b2 release rejects that mode correctly. Use the reviewed merged source
in the execution environment rather than relaxing validation. Simms must
include local-hour-angle start support (PR #237). WSClean, DDFacet and CASA
are supplied by the configured images. `noise_casa.py` preserves native
metadata and exports the actual dirty image and measured PSF, without
substituting a fitted Gaussian. `layouts/provenance.json` records layout sources.

Compiling without `--submit` is pure validation. Submission snapshots the
constructed branch graph and Python helpers; compute nodes restore that graph
and never invoke `build_pipeline`. Pending jobs do not reserve paths. See the
maintained offloading documentation for claims, recovery, handles and logs.

The complete 30-case snapshot campaign has passed execution and product audits:
1,260 image/PSF pairs, 30 MSs with 60 eight-second integrations, and successful
gather/finalizer records. Separate live checks cover interruption, cache reuse,
failure propagation and local execution of the same placed graph.
That campaign used `threads=4`, `max_workers=1` and `max_parallel=3`, with
eight CPUs and 32 GiB reserved per branch allocation.

**Known monitoring limit:** composite records currently repeat child records,
schemas and captured logs. This campaign produced a 598 MB final record, and
`ninja status` was killed with exit 137 when reading the completed run. The
`current_manifest` accessor also loads full records. The successful scientific
run does not establish bounded-memory monitoring at this scale; see
[issue #251](https://github.com/shinobi-dosho/stimela-ninja/issues/251).
