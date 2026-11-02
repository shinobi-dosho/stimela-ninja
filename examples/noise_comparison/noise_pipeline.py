"""Compare dirty images and PSFs from the same simulated noise.

Pass settings.yaml to ninja run with --settings. Python loops declare the
experiment; Shinobi schedules its simulations and imaging jobs.
"""

from __future__ import annotations

from pathlib import Path

import dosho
import yaml
from pydantic import BaseModel, Field, create_model
from shinobi import Recipe, recipe_builder
from shinobi.cabs import adapt_definition
from shinobi.steps.schema import OutputRef

from noise_casa import casa_dirty, export_dirty
from noise_inventory import inventory


class ArraySettings(BaseModel):
    # Simms YAML layout, including the array centre.
    layout: Path
    sefd_jy: float = Field(gt=0)
    first_channel_hz: float = Field(gt=0)
    cell_arcsec: float = Field(gt=0)
    image_size: int = Field(default=512, ge=32)


class Settings(BaseModel):
    arrays: dict[str, ArraySettings] = Field(min_length=1)
    declinations: list[int] = Field(min_length=1)
    robust: list[float] = Field(min_length=1)
    # Four 16-MHz channels (64 MHz) is a configurable working assumption.
    nchan: int = Field(default=4, ge=2)
    single_channel: int = Field(default=0, ge=0)
    threads: int = Field(default=2, ge=1)
    max_workers: int = Field(default=2, ge=1)
    ra: str = "0h24m20s"
    # Simms finds the requested local hour angle near this reference epoch.
    reference_utc: str = "2024-03-14T12:00:00"
    duration_seconds: float = Field(default=14400.0, gt=0)
    integration_seconds: float = Field(default=32.0, gt=0)
    # None centres the requested duration on local transit.
    start_hour_angle: float | None = None
    images: dict[str, str] = Field(default_factory=dict)


class NoInputs(BaseModel):
    pass


class BuilderInputs(BaseModel):
    settings: Path


@recipe_builder(inputs_model=BuilderInputs)
def recipe(inputs: BuilderInputs) -> Recipe:
    """Read the experiment config and declare all simulations and imaging jobs."""
    settings = Settings.model_validate(yaml.safe_load(inputs.settings.read_text()))

    count = settings.duration_seconds / settings.integration_seconds
    if not count.is_integer():
        raise ValueError("duration_seconds must be an exact number of integrations")
    if len(set(settings.declinations)) != len(settings.declinations):
        raise ValueError("declinations must be unique")
    if any(not -90 <= dec <= 90 for dec in settings.declinations):
        raise ValueError("declinations must be between -90 and 90 degrees")
    if any(not -2 <= value <= 2 for value in settings.robust):
        raise ValueError("Briggs robustness must be between -2 and 2")
    root = Path("noise-products")
    if settings.single_channel >= settings.nchan:
        raise ValueError("single_channel must be smaller than nchan")
    for name, array in settings.arrays.items():
        if not name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in name):
            raise ValueError("array names must contain only lowercase letters, digits and hyphens")
        if not array.layout.is_file():
            raise ValueError(f"{name}: supply an existing Simms layout: {array.layout}")

    pipeline = Recipe(name="array-noise", inputs_model=NoInputs, outputs_model=NoInputs, max_workers=settings.max_workers, cache=True, sandbox=True)

    # The catalogue calls --telescope a string. A Path annotation makes our
    # supplied layout an explicit, staged read input; the CLI is unchanged.
    telsim = adapt_definition(dosho.get("simms-telsim"))
    layout_inputs = create_model("LayoutTelsimInputs", __base__=telsim.inputs_model, telescope=(Path, ...))
    telsim = telsim.model_copy(update={"inputs_model": layout_inputs})

    for dec in settings.declinations:
        for instrument, array in settings.arrays.items():
            case = f"{instrument}_dec{dec:+d}"
            directory = root / instrument / f"dec{dec:+d}"

            # Centre the observation on local transit unless an HA is supplied.
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

            # Scatter uses zipped lists: each position is one imaging job.
            # Both modes read this SAME MS, with a different channel selection.
            modes = [mode for mode in ("single", "continuum") for _ in settings.robust]
            starts = [settings.single_channel if mode == "single" else 0 for mode in modes]
            ends = [settings.single_channel + 1 if mode == "single" else settings.nchan for mode in modes]
            jobs = [directory / mode / f"robust-{index}" for mode in ("single", "continuum") for index in range(len(settings.robust))]

            pipeline.add_step(
                f"{case}_wsclean",
                dosho.get("wsclean-readonly"),
                scatter=["weight", "channel_range", "prefix", "temp_dir"],
                ms=[shared_ms],
                column="DATA",
                niter=0,
                make_psf=True,
                weight=[("briggs", value) for value in settings.robust] * 2,
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
                weight_robust=settings.robust * 2,
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
                robust=settings.robust * 2,
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
    for ref in pipeline.steps:
        tool = ref.name.rsplit("_", 1)[-1]
        image_key = {"simulate": "simms", "casa": "casa-tasks", "fits": "casa-tasks"}.get(tool, tool)
        if image_key in settings.images:
            ref.step = ref.step.model_copy(update={"image": settings.images[image_key]})
    return pipeline
