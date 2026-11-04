# Compare thermal noise across arrays and imagers

One script declares the experiment: simulate noise once for each array and
declination, then image the **same Measurement Set** with WSClean, DDFacet and
CASA at each Briggs robustness value. Each imager produces a dirty image and
PSF for a single channel and for the full band. No deconvolution is performed.

Edit the configuration, then run the recipe. Shinobi handles dependencies,
parallel execution, caching and output checks.

## Configuration

| File | What to change |
| --- | --- |
| `settings.yaml` | Arrays, declinations, robustness values, observing duration, integration time and concurrency; fixed arguments for each tool under `simms`, `wsclean`, `ddfacet` and `casa`. |
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

## Read the script

`noise_pipeline.py` loads the settings, loops over arrays and declinations,
and declares the simulation and imaging steps. Zipped scatters pair each
robustness value with its channel selection and output directory. Explicit
output references connect the imagers to the simulation.

The same file holds the two Python steps: CASA dirty imaging with its FITS
export, and an inventory that checks the expected files. The CASA steps run
this file inside the CASA image, so imports needed only to build the recipe
stay inside `recipe`. Layout sources are recorded in `layouts/provenance.json`.
