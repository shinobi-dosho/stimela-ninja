"""Noise-only array comparison. See README.md before execution.

Build the graph without running it: ``build_pipeline(settings)``. Scientific
inputs (SEFD, frequency, layouts and cell sizes) are required.
Requires Simms with local-hour-angle start support (PR #237). Importing this module launches no simulation.
"""

from __future__ import annotations

from pathlib import Path

import dosho
from pydantic import BaseModel, Field, create_model
from shinobi import Recipe, SlurmPlacement
from shinobi.cabs import adapt_definition
from shinobi.steps.schema import OutputRef

from noise_casa import casa_dirty, export_dirty
from noise_inventory import inventory

INSTRUMENTS = ("meerkat", "vla-b", "vla-c", "gmrt", "askap", "skamid-aastar")
DECLINATIONS = (-25, -15, 5, 20, 30)
ROBUST = (-2, -4 / 3, -2 / 3, 0, 2 / 3, 4 / 3, 2)


class ArraySettings(BaseModel):
    # Supply a reviewed Simms YAML layout, including its array centre.
    layout: Path
    sefd_jy: float = Field(gt=0)
    first_channel_hz: float = Field(gt=0)
    cell_arcsec: float = Field(gt=0)
    image_size: int = Field(default=512, ge=32)


class Settings(BaseModel):
    arrays: dict[str, ArraySettings]
    # Four 16-MHz channels (64 MHz) is a configurable working assumption.
    nchan: int = Field(default=4, ge=2)
    single_channel: int = Field(default=0, ge=0)
    threads: int = Field(default=2, ge=1)
    max_workers: int = Field(default=2, ge=1)
    ra: str = "0h24m20s"
    # Fixed reference epoch; Simms finds the nearest local HA -2 h per array.
    reference_utc: str = "2024-03-14T12:00:00"
    duration_seconds: float = Field(default=14400.0, gt=0)
    integration_seconds: float = Field(default=32.0, gt=0)
    # None centres the requested duration on local transit.
    start_hour_angle: float | None = None
    images: dict[str, str] = Field(default_factory=dict)
    sbatch_opts: dict[str, str] = Field(default_factory=dict)
    max_parallel: int = Field(default=3, ge=1)


class NoInputs(BaseModel):
    pass


def _build_flat(
    settings: Settings,
    root: Path = Path("noise-products"),
    *,
    simulate_only: bool = False,
    instruments: tuple[str, ...] = INSTRUMENTS,
    declinations: tuple[int, ...] = DECLINATIONS,
) -> Recipe:
    """Declare 30 independent simulations and read-only imaging scatters.

    Each imager scatters over the 14 (band mode, robustness) combinations.
    Construction loops declare normal Shinobi steps; they run no subprocesses.
    """
    if set(settings.arrays) != set(INSTRUMENTS):
        raise ValueError(f"Supply settings for exactly these arrays: {INSTRUMENTS}")
    if not set(instruments) <= set(INSTRUMENTS) or not set(declinations) <= set(DECLINATIONS):
        raise ValueError("Select instruments and declinations from the experiment grid")
    if settings.single_channel >= settings.nchan:
        raise ValueError("single_channel must be smaller than nchan")
    if root.is_absolute() or ".." in root.parts:
        raise ValueError("Use a relative product directory inside the run workspace")
    for name, array in settings.arrays.items():
        if not array.layout.is_file():
            raise ValueError(f"{name}: supply an existing reviewed Simms layout: {array.layout}")

    pipeline = Recipe(name="array-noise", inputs_model=NoInputs, outputs_model=NoInputs, max_workers=settings.max_workers, cache=True, sandbox=True)

    # The catalogue calls --telescope a string. A Path annotation makes our
    # supplied layout an explicit, staged read input; the CLI is unchanged.
    telsim = adapt_definition(dosho.get("simms-telsim"))
    layout_inputs = create_model("LayoutTelsimInputs", __base__=telsim.inputs_model, telescope=(Path, ...))
    telsim = telsim.model_copy(update={"inputs_model": layout_inputs})

    for dec in declinations:
        for instrument in instruments:
            array = settings.arrays[instrument]
            case = f"{instrument}_dec{dec:+d}"
            directory = root / instrument / f"dec{dec:+d}"

            # Four hours starting at local HA -2 h, approximately transit-centred.
            simulate = f"{case}_simulate"
            pipeline.add_step(
                simulate,
                telsim,
                ms=directory / "noise.ms",
                telescope=array.layout.resolve(),
                direction=f"J2000,{settings.ra},{dec:+d}d0m0s",
                starttime=settings.reference_utc,
                startha=settings.start_hour_angle if settings.start_hour_angle is not None else -settings.duration_seconds / 7200.0,
                ntime=round(settings.duration_seconds / settings.integration_seconds),
                dtime=settings.integration_seconds,
                startfreq=array.first_channel_hz,
                dfreq="16MHz",
                nchan=settings.nchan,
                correlations="XX,YY",
                column="DATA",
                sefd=array.sefd_jy,
                nworkers=settings.threads,
            )
            shared_ms = OutputRef(step=simulate, field="ms")
            if simulate_only:
                continue

            # Scatter uses zipped lists: each position is one imaging job.
            # Both modes read this SAME MS, with a different channel selection.
            modes = [mode for mode in ("single", "continuum") for _ in ROBUST]
            starts = [settings.single_channel if mode == "single" else 0 for mode in modes]
            ends = [settings.single_channel + 1 if mode == "single" else settings.nchan for mode in modes]
            jobs = [directory / mode / f"robust-{index}" for mode in ("single", "continuum") for index in range(7)]

            pipeline.add_step(
                f"{case}_wsclean",
                dosho.get("wsclean-readonly"),
                scatter=["weight", "channel_range", "prefix", "temp_dir"],
                ms=[shared_ms],
                column="DATA",
                niter=0,
                make_psf=True,
                weight=[("briggs", value) for value in ROBUST] * 2,
                channel_range=list(zip(starts, ends)),
                nchan=1,
                prefix=[str(job / "wsclean/image") for job in jobs],
                temp_dir=[str(job / "wsclean-scratch") for job in jobs],
                size=(array.image_size, array.image_size),
                scale=f"{array.cell_arcsec:g}asec",
                threads=settings.threads,
            )
            pipeline.add_step(
                f"{case}_ddfacet",
                dosho.get("ddfacet-readonly"),
                scatter=["weight_robust", "selection_chan_start", "selection_chan_end", "output_name", "data_cache_dir", "cache_dir_wisdom_fftw", "montblanc_log_file"],
                data_ms=[shared_ms],
                data_col_name="DATA",
                output_mode="DirtyPSF",
                output_images="dP",
                data_chunk_hours=5 / 60,  # Five-minute chunks; DDFacet expects hours.
                weight_mode="Briggs",
                weight_robust=list(ROBUST) * 2,
                selection_chan_start=starts,
                selection_chan_end=ends,
                freq_n_band=1,
                output_name=[str(job / "ddfacet/image") for job in jobs],
                data_cache_dir=[str(job / "ddfacet-scratch/cache") for job in jobs],
                cache_dir_wisdom_fftw=[str(job / "ddfacet-scratch/wisdom") for job in jobs],
                montblanc_log_file=[str(job / "ddfacet-scratch/montblanc.log") for job in jobs],
                cache_psf="off",
                cache_vis_data="off",
                cache_dirty="off",
                cache_last_residual=False,
                cache_cf=False,
                cache_hmp=False,
                image_n_pix=array.image_size,
                image_cell=array.cell_arcsec,
                facets_n_facets=1,
                parallel_ncpu=settings.threads,
            )
            casa = f"{case}_casa"
            pipeline.add_step(
                casa,
                casa_dirty,
                scatter=["robust", "spw", "products"],
                ms=shared_ms,
                robust=list(ROBUST) * 2,
                spw=[f"0:{start}~{end - 1}" for start, end in zip(starts, ends)],
                products=[str(job / "casa-native") for job in jobs],
                image_size=array.image_size,
                cell_arcsec=array.cell_arcsec,
            )
            # Fixed path outputs in a separate step preserve strict MS access.
            pipeline.add_step(
                f"{case}_casa_fits",
                export_dirty,
                scatter=["products", "dirty", "psf", "metadata"],
                products=OutputRef(step=casa, field="products"),
                dirty=[str(job / "casa/dirty.fits") for job in jobs],
                psf=[str(job / "casa/psf.fits") for job in jobs],
                metadata=[str(job / "casa/metadata.json") for job in jobs],
            )
            pipeline.add_step(
                f"{case}_inventory", inventory, manifest=str(directory / "products.json"), directories=[job / imager for job in jobs for imager in ("wsclean", "ddfacet", "casa")]
            )
            pipeline.steps[-1].after = [f"{case}_wsclean", f"{case}_ddfacet", f"{case}_casa_fits"]
    return pipeline


def build_pipeline(
    settings: Settings,
    root: Path = Path("noise-products"),
    *,
    simulate_only: bool = False,
    instruments: tuple[str, ...] = INSTRUMENTS,
    declinations: tuple[int, ...] = DECLINATIONS,
) -> Recipe:
    """One declared outer zip scatter; each allocation owns one local DAG.

    Construction derives the varying constants from the reviewed per-case
    declarations. All graph construction finishes here, on the submit host.
    The worker restores the frozen branch, never calls this builder.
    """
    count = settings.duration_seconds / settings.integration_seconds
    if not count.is_integer():
        raise ValueError("duration_seconds must be an exact number of integrations")
    cases = [_build_flat(settings, root, simulate_only=simulate_only, instruments=(instrument,), declinations=(dec,)) for dec in declinations for instrument in instruments]
    if not cases:
        raise ValueError("select at least one telescope/declination case")
    # Give corresponding leaves identical names across cases. Scientific
    # identity is outer.case[index].leaf in local and compiled execution.
    for case in cases:
        names = {ref.name: ref.name.split("_", 2)[-1] for ref in case.steps}
        for ref in case.steps:
            ref.name = names[ref.name]
            for field, sources in ref.wiring.items():

                def rename(source):
                    return source.model_copy(update={"step": names[source.step]}) if isinstance(source, OutputRef) else source

                ref.wiring[field] = [rename(source) for source in sources] if isinstance(sources, list) else rename(sources)
            ref.after = [names[name] for name in ref.after]
            image_key = "casa-tasks" if ref.name.startswith("casa") else "simms" if ref.name == "simulate" else ref.name
            if image_key in settings.images:
                ref.step = ref.step.model_copy(update={"image": settings.images[image_key], "backend": "apptainer"})
    branch = cases[0]
    fields, columns = {}, {}
    for index, ref in enumerate(branch.steps):
        for field, value in list(ref.params.items()):
            values = [case.steps[index].params[field] for case in cases]
            # Always expose the per-case MS destination, including a one-case
            # selection; this makes the public zip scatter explicit.
            if len(cases) == 1 and ref.name == "simulate" and field == "ms" or any(one != value for one in values[1:]):
                name = ref.name + "__" + field
                annotation = ref.step.inputs_model.model_fields[field].annotation
                if ref.scatter and field in ref.scatter.fields:
                    annotation = list[annotation]
                fields[name] = (annotation, ...)
                columns[name] = values
                del ref.params[field]
                from shinobi.steps.schema import InputRef

                ref.wiring[field] = InputRef(field=name)
    branch.inputs_model = create_model("NoiseCaseInputs", **fields)
    exported = {}
    for ref in branch.steps:
        selected = ("ms",) if ref.name == "simulate" else ("manifest",) if ref.name == "inventory" else ()
        for field in selected:
            output = ref.step.outputs_model.model_fields[field]
            annotation = list[output.annotation] if ref.scatter else output.annotation
            name = ref.name + "__" + field
            exported[name] = (annotation, ...)
            branch.output_wiring[name] = OutputRef(step=ref.name, field=field)
    branch.outputs_model = create_model("NoiseCaseOutputs", **exported)
    branch.name = "noise-case"
    outputs = create_model("NoiseCampaignOutputs", **{name: (list[annotation], ...) for name, (annotation, _) in exported.items()})
    pipeline = Recipe(name="array-noise", inputs_model=NoInputs, outputs_model=outputs, max_workers=settings.max_parallel, cache=True, sandbox=True)
    pipeline.add_step("cases", branch, scatter=list(columns), placement=SlurmPlacement(sbatch_opts=settings.sbatch_opts, max_parallel=settings.max_parallel), **columns)
    for name in exported:
        pipeline.set_output(name, OutputRef(step="cases", field=name))
    return pipeline


def build_branch(settings: dict, instrument: str, declination: int) -> Recipe:
    """Compatibility helper returning a local single-case declaration."""
    return _build_flat(Settings.model_validate(settings), instruments=(instrument,), declinations=(declination,))
