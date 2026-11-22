# Compare thermal noise across arrays and imagers

One script declares the experiment: simulate noise once for each array and
declination, then image the **same Measurement Set** with WSClean, DDFacet and
CASA at each Briggs robustness value. Each imager produces a dirty image and
PSF for a single channel and for the full band. No deconvolution is performed.

Edit the configuration, then run the recipe. Shinobi handles dependencies,
parallel execution, caching and output checks. `noise_pipeline.py` runs every
case on one machine; `noise_slurm.py` runs the same cases on a Slurm cluster,
one allocation per array/declination case.

## Configuration

| File | What to change |
| --- | --- |
| `settings.yaml` | Arrays, declinations, robustness values, observing duration, integration time and concurrency; fixed arguments for each tool under `simms`, `wsclean`, `ddfacet` and `casa`; Slurm options `sbatch_opts` and `max_parallel`, used only by `noise_slurm.py`. |
| `config.yaml` | Execution backend, caching, sandboxing and provenance. |
| `layouts/*.yaml` | Antenna positions; paths in `settings.yaml` are relative to the invocation working directory. |

The supplied grid has six arrays, five declinations, seven robustness values,
two channel selections and three imagers: **30 simulations and 1,260 image/PSF
pairs**. To try a smaller experiment, keep one entry in `arrays` and set
`declinations` to `[-25]` and `robust` to `[0]`. This produces six pairs from
one simulation. No Python changes are needed.

The default observation is four hours with 32-second integrations. Change
`duration_seconds` and `integration_seconds` together; their ratio must be a
whole number. Observations are centred on local transit unless
`start_hour_angle` is supplied. Array SEFDs, frequencies and image scales are
experiment inputs; adjust them for the comparison you want to make.

## Run

Install Shinobi with the `ms` and `fits` extras and make Apptainer available.
The tool environment needs Simms with local-hour-angle support and a Dosho
catalogue supporting DDFacet's `DirtyPSF` mode. Catalogue container images are
used by default. To use local images, add an `images` mapping to `settings.yaml`
with absolute SIF paths under `simms`, `wsclean`, `ddfacet` and `casa-tasks`.

From the repository root:

```sh
ninja --config examples/noise_comparison/config.yaml run examples/noise_comparison/noise_pipeline.py:recipe examples/noise_comparison/settings.yaml --dryrun
ninja --config examples/noise_comparison/config.yaml run examples/noise_comparison/noise_pipeline.py:recipe examples/noise_comparison/settings.yaml
```

Every scalar setting is also an option, and options override the file:
adding `--declinations -25 --robust 0` keeps every array but only one
declination and one weighting. `--help` lists them all. Arrays and tool
arguments are mappings, so they come only from a settings file; a second file
after the first overrides just the keys it sets.

For a run from another directory, use absolute paths for the target,
execution config and settings, and update layout paths in the settings for
that invocation directory. Paths inside YAML are cwd-relative too.

The default dry-run DAG centers each simulation above its three imagers,
places the imagers side by side, and draws dependency arrows into the inventory.
Each simulation case stays together as one block. Scatter counts appear below
step names, and wired product names appear below each block. Add `--dryrun-details` alongside `--dryrun` to inspect parameter
previews and filesystem access reasons, including paths.

`max_workers` limits concurrent recipe tasks; `threads` controls tool
parallelism. Outputs go under
`noise-products/<array>/dec<declination>/<single|continuum>/robust-<index>/`,
where the index follows the order in `settings.yaml`. Each array/declination
also has a `noise.ms` and a `products.json` inventory of its FITS files.

## Run on Slurm

`noise_slurm.py` declares the same cases as one branch recipe and scatters it
over the array/declination grid with explicit Slurm placement. Compiling it
submits one Slurm array: each task runs one case (its simulation, then the
three imagers and the CASA FITS export as local scatters, then the inventory)
inside its own allocation. At most `max_parallel` cases run at once, and
`max_workers` limits concurrent tasks within each case. Set `sbatch_opts` in
`settings.yaml` to your site's partition and resources; one task on one node
is required, so leave out `nodes` and `ntasks`.

Run from a directory on storage that the compute nodes mount at the same path.
Submission needs absolute local SIF files, so add the `images` mapping to
`settings.yaml`, and a Python environment with Shinobi that the compute nodes
can see. `noise_slurm.py` imports `noise_pipeline.py`, so put this directory on
`PYTHONPATH` and pass it as `--code-root`:

```sh
export PYTHONPATH="$PWD/examples/noise_comparison${PYTHONPATH:+:$PYTHONPATH}"
# Validate and print the plan: one local driver per case. Nothing is submitted.
ninja --config examples/noise_comparison/config.yaml compile examples/noise_comparison/noise_slurm.py:recipe examples/noise_comparison/settings.yaml --code-root examples/noise_comparison
# Submit and detach.
ninja --config examples/noise_comparison/config.yaml compile examples/noise_comparison/noise_slurm.py:recipe examples/noise_comparison/settings.yaml --code-root examples/noise_comparison --submit --worker-python /shared/env/bin/python
```

Submission prints a `handle.json` path; pass it to `ninja status` and
`ninja logs`. The same options and settings overrides as `ninja run` apply,
for example `--declinations -25` to submit six cases. `ninja run` with
`--dryrun` shows the placed scatter and the case DAG, and `ninja run` without
it executes the same declaration locally, up to `max_parallel` cases at a time.
Outputs use the same layout as the local recipe. See `docs/offloading.rst` for
the mixed execution route, its records and its limits.

## Read the script

`noise_pipeline.py` loads the settings, loops over arrays and declinations,
and declares each case with `add_case`: the simulation and imaging steps.
`case_inputs` computes the values that differ between cases, such as the
layout, image scale and output paths. Zipped scatters pair each robustness
value with its channel selection and output directory. Explicit output
references connect the imagers to the simulation.

`noise_slurm.py` reuses `add_case` to declare one branch recipe whose inputs
are those per-case values, then scatters the branch over the `case_inputs` of
every case with `SlurmPlacement`.

The same file holds the two Python steps: CASA dirty imaging with its FITS
export, and an inventory that checks the expected files. The CASA steps run
this file inside the CASA image, so imports needed only to build the recipe
stay inside `recipe`. Layout sources are recorded in `layouts/provenance.json`.
