"""Compare dirty images and PSFs from the same simulated noise.

settings.yaml supplies the settings; each scalar one is also a `ninja run`
option that overrides the file. Python loops declare the experiment; Shinobi schedules
its simulations and imaging jobs.

The CASA steps run this whole file inside their image, where Shinobi, Dosho
and Pydantic are inert stubs. Host-only imports stay inside `recipe`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import dosho
from dosho import images
from pydantic import BaseModel, Field, field_validator, model_validator
from shinobi import DatasetAccess, DatasetMode, MSv2, Recipe, pystep, recipe_builder
from shinobi.steps.schema import OutputRef


# tclean keywords that casa_dirty sets for each job.
CASA_JOB_KEYWORDS = ("vis", "imagename", "datacolumn", "spw", "imsize", "cell", "weighting", "robust")


class ArraySettings(BaseModel):
    # Simms YAML layout, including the array centre.
    layout: Path
    sefd: float = Field(gt=0)
    # Passed to simms as written, e.g. 1376MHz.
    first_channel: str
    dpix: float = Field(gt=0)
    npix: int = Field(default=512, ge=32)

    @field_validator("layout")
    @classmethod
    def _layout_exists(cls, layout: Path) -> Path:
        if not layout.is_file():
            raise ValueError(f"supply an existing Simms layout: {layout}")
        return layout


class Settings(BaseModel):
    arrays: dict[Annotated[str, Field(pattern=r"^[a-z0-9-]+$")], ArraySettings] = Field(min_length=1)
    declinations: list[Annotated[int, Field(ge=-90, le=90)]] = Field(min_length=1)
    robust: list[Annotated[float, Field(ge=-2, le=2)]] = Field(min_length=1)
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
    # Simms writes this column and every imager reads it.
    column: str = "DATA"
    output_dir: Path = Path("noise-products")
    images: dict[str, str] = Field(default_factory=dict)
    # Fixed tool arguments; the recipe supplies only per-job values: MS,
    # weighting, channels and paths. They define the experiment, so each is
    # required. The cabs' inputs models check simms, wsclean and ddfacet when
    # the recipe is built; casa keywords reach tclean unchecked.
    simms: dict[str, Any]
    wsclean: dict[str, Any]
    ddfacet: dict[str, Any]
    casa: dict[str, Any]

    @field_validator("declinations")
    @classmethod
    def _unique(cls, declinations: list[int]) -> list[int]:
        if len(set(declinations)) != len(declinations):
            raise ValueError("declinations must be unique")
        return declinations

    @model_validator(mode="after")
    def _consistent(self) -> Settings:
        if not (self.duration_seconds / self.integration_seconds).is_integer():
            raise ValueError("duration_seconds must be an exact number of integrations")
        if self.single_channel >= self.nchan:
            raise ValueError("single_channel must be smaller than nchan")
        if clash := sorted(self.casa.keys() & set(CASA_JOB_KEYWORDS)):
            raise ValueError(f"casa: {', '.join(clash)} are set per job")
        return self


class NoInputs(BaseModel):
    pass


@recipe_builder(inputs_model=Settings)
def recipe(settings: Settings) -> Recipe:
    """Declare all simulations and imaging jobs for the experiment."""
    from pydantic import create_model
    from shinobi.cabs import adapt_definition

    root = settings.output_dir

    pipeline = Recipe(name="array-noise", inputs_model=NoInputs, outputs_model=NoInputs, max_workers=settings.max_workers)

    # The catalogue calls --telescope a string. A Path annotation makes our
    # supplied layout an explicit, staged read input; the CLI is unchanged.
    telsim = adapt_definition(dosho.get("simms-telsim"))
    layout_inputs = create_model("LayoutTelsimInputs", __base__=telsim.inputs_model, telescope=(Path, ...))
    telsim = telsim.model_copy(update={"inputs_model": layout_inputs})
    wsclean = adapt_definition(dosho.get("wsclean-readonly"))
    ddfacet = adapt_definition(dosho.get("ddfacet-readonly"))

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
                startfreq=array.first_channel,
                nchan=settings.nchan,
                column=settings.column,
                sefd=array.sefd,
                nworkers=settings.threads,
                **settings.simms,
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
                wsclean,
                scatter=["weight", "channel_range", "prefix", "temp_dir"],
                ms=[shared_ms],
                column=settings.column,
                weight=[("briggs", value) for value in settings.robust] * 2,
                channel_range=list(zip(starts, ends)),
                prefix=[str(job / "wsclean/image") for job in jobs],
                temp_dir=[str(job / "wsclean-scratch") for job in jobs],
                size=(array.npix, array.npix),
                scale=f"{array.dpix:g}asec",
                threads=settings.threads,
                **settings.wsclean,
            )
            pipeline.add_step(
                f"{case}_ddfacet",
                ddfacet,
                scatter=["weight_robust", "selection_chan_start", "selection_chan_end", "output_name", "data_cache_dir", "cache_dir_wisdom_fftw", "montblanc_log_file"],
                data_ms=[shared_ms],
                data_col_name=settings.column,
                weight_robust=settings.robust * 2,
                selection_chan_start=starts,
                selection_chan_end=ends,
                output_name=[str(job / "ddfacet/image") for job in jobs],
                data_cache_dir=[str(job / "ddfacet-scratch/cache") for job in jobs],
                cache_dir_wisdom_fftw=[str(job / "ddfacet-scratch/wisdom") for job in jobs],
                montblanc_log_file=[str(job / "ddfacet-scratch/montblanc.log") for job in jobs],
                image_n_pix=array.npix,
                image_cell=array.dpix,
                parallel_ncpu=settings.threads,
                **settings.ddfacet,
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
                image_size=array.npix,
                cell_arcsec=array.dpix,
                column=settings.column,
                tclean=settings.casa,
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


# Minimal read-only CASA dirty imaging and a separate FITS export step.


class NativeOutputs(BaseModel):
    products: Path


class FitsOutputs(BaseModel):
    dirty: Path
    psf: Path
    metadata: Path


@pystep(image=getattr(images, "CASA_TASKS"), write_paths=["products"], dataset_accesses=[DatasetAccess(field="ms", mode=DatasetMode.READ)])
def casa_dirty(ctx, ms: MSv2, products: str, robust: float, spw: str, image_size: int, cell_arcsec: float, column: str, tclean: dict[str, Any]) -> NativeOutputs:
    """Run tclean with the fixed `tclean` keywords plus this job's values."""
    root = Path(products)
    root.mkdir(parents=True, exist_ok=True)
    job = {
        "vis": [str(ms)],
        "imagename": str(root / "image"),
        "datacolumn": column.lower(),
        "spw": spw,
        "imsize": [image_size, image_size],
        "cell": f"{cell_arcsec:g}arcsec",
        "weighting": "briggs",
        "robust": robust,
    }
    assert job.keys() == set(CASA_JOB_KEYWORDS)
    options = {**tclean, **job}
    ctx.import_func("tclean", "casatasks")(**options)
    (root / "settings.json").write_text(json.dumps(options, indent=2) + "\n")
    return NativeOutputs(products=root)


@pystep(image=getattr(images, "CASA_TASKS"), write_paths=["dirty", "psf", "metadata"])
def export_dirty(ctx, products: Path, dirty: str, psf: str, metadata: str) -> FitsOutputs:
    """At zero CLEAN iterations CASA's residual is the dirty noise image.

    Retain native header metadata, including beam fits if CASA provides them.
    Do not invent a beam or substitute a Gaussian for the measured PSF.
    """
    export = ctx.import_func("exportfits", "casatasks")
    imhead = ctx.import_func("imhead", "casatasks")
    import shutil
    import tempfile

    headers = {}
    export_telescopes = {}
    native_miscinfo = {}
    for kind, destination in (("residual", dirty), ("psf", psf)):
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        native = str(products / f"image.{kind}")
        headers[kind] = imhead(imagename=native, mode="list")
        telescope = headers[kind].get("telescope", "")
        # Simms names custom arrays by their YAML path. CASA cannot export an
        # overlong TELESCOP card; edit only a disposable copy, retaining metadata.
        if "/" in telescope or len(telescope) > 68:
            label = Path(telescope).stem[:68]
            with tempfile.TemporaryDirectory(prefix="casa-fits-") as scratch:
                copy = Path(scratch) / Path(native).name
                shutil.copytree(native, copy)
                imhead(imagename=str(copy), mode="put", hdkey="telescope", hdvalue=label)
                image = ctx.import_func("image", "casatools")()
                image.open(str(copy))
                try:
                    miscinfo = image.miscinfo()
                    native_miscinfo[kind] = dict(miscinfo)
                    if miscinfo.get("INSTRUME") == telescope:
                        miscinfo["INSTRUME"] = label
                        image.setmiscinfo(miscinfo)
                finally:
                    image.close()
                export(imagename=str(copy), fitsimage=str(path), overwrite=True, dropdeg=False)
            export_telescopes[kind] = label
        else:
            export(imagename=native, fitsimage=str(path), overwrite=True, dropdeg=False)
    info = {
        "settings": json.loads((products / "settings.json").read_text()),
        "native_headers": headers,
        "export_telescope_labels": export_telescopes,
        "native_miscinfo": native_miscinfo,
    }
    metadata_path = Path(metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(info, indent=2, default=str) + "\n")
    return FitsOutputs(dirty=Path(dirty), psf=Path(psf), metadata=metadata_path)


# Native, declared catalogue of completed image/PSF products.


class Inventory(BaseModel):
    manifest: Path


@pystep(backend="native", write_paths=["manifest"])
def inventory(manifest: str, directories: list[Path]) -> Inventory:
    products = []
    for directory in directories:
        expected = {"wsclean": ("image-dirty.fits", "image-psf.fits"), "ddfacet": ("image.dirty.fits", "image.psf.fits"), "casa": ("dirty.fits", "psf.fits", "metadata.json")}[
            directory.name
        ]
        for filename in expected:
            path = directory / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing or empty required product: {path}")
        fits = sorted(directory.rglob("*.fits"))
        products.extend({"path": str(path), "bytes": path.stat().st_size} for path in fits)
    manifest_path = Path(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"products": products}, indent=2) + "\n")
    return Inventory(manifest=manifest_path)
