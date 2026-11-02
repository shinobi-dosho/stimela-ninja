"""Minimal read-only CASA dirty imaging and a separate FITS export step."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from dosho import images

    CASA_IMAGE = images.CASA_TASKS
except ModuleNotFoundError as exc:
    # Shinobi's container loader supplies decorator stubs, without Dosho.
    if exc.name != "dosho":
        raise
    CASA_IMAGE = None

from pydantic import BaseModel
from shinobi import DatasetAccess, MSv2, pystep


class NativeOutputs(BaseModel):
    products: Path


class FitsOutputs(BaseModel):
    dirty: Path
    psf: Path
    metadata: Path


@pystep(image=CASA_IMAGE, write_paths=["products"], dataset_accesses=[DatasetAccess(field="ms", mode="read")])
def casa_dirty(ctx, ms: MSv2, products: str, robust: float, spw: str, image_size: int, cell_arcsec: float) -> NativeOutputs:
    root = Path(products)
    root.mkdir(parents=True, exist_ok=True)
    options = dict(  # noqa: C408 -- mirrors casatasks keyword arguments
        vis=[str(ms)],
        imagename=str(root / "image"),
        datacolumn="data",
        spw=spw,
        imsize=[image_size, image_size],
        cell=f"{cell_arcsec:g}arcsec",
        stokes="I",
        specmode="mfs",
        gridder="standard",
        weighting="briggs",
        robust=robust,
        npixels=0,
        uvtaper=[],
        niter=0,
        interactive=False,
        savemodel="none",
        restart=False,
        calcpsf=True,
        calcres=True,
        restoration=False,
        pbcor=False,
        parallel=False,
    )
    ctx.import_func("tclean", "casatasks")(**options)
    (root / "settings.json").write_text(json.dumps(options, indent=2) + "\n")
    return NativeOutputs(products=root)


@pystep(image=CASA_IMAGE, write_paths=["dirty", "psf", "metadata"])
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
                    info = image.miscinfo()
                    native_miscinfo[kind] = dict(info)
                    if info.get("INSTRUME") == telescope:
                        info["INSTRUME"] = label
                        image.setmiscinfo(info)
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
