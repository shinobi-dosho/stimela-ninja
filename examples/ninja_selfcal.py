"""Ninja selfcal -- a shinobi @recipe for the self-calibration pipeline
originally at https://github.com/SpheMakh/ninja/blob/319cc37/ninja-recipe.py
(stimela classic).

This is a full reimplementation on `@recipe`/`ninja run`, not a
structural port of the mechanical conversion this replaced:

* No Step dataclass / declare() / run_steps() bucketing. That existed to
  let a script register steps now and run a chosen subset later --
  @recipe + `ninja run` remove the reason for that: the pipeline is just
  this one function, its knobs are its parameters (auto-exposed as CLI
  options), and skipping/resuming steps is a plain `if` per step, not a
  data structure to build and slice.
* No hand-rolled DRYRUN/table-printing block. `ninja run
  examples/ninja_selfcal.py:selfcal --dryrun` already shows the step
  graph, for free, for any @recipe -- see shinobi.dag.
* Cabs are still defined once at module level -- they describe *tasks*,
  not a specific run, so none of them bake in a run-specific MS name
  anymore; every step passes it explicitly from the recipe's own
  resolved `ms` variable (a required, no-default cab input instead).

Run it:

    ninja run examples/ninja_selfcal.py:selfcal --dryrun
    ninja run examples/ninja_selfcal.py:selfcal --init --ms0 foo.ms ...

Still true, and still worth knowing: paths are plain os.path.join()/
f-strings via out()/indir()/msdir_path()/obsinfo() -- there's no path-
substitution DSL to route them through. A couple of steps' path context
was genuinely ambiguous in the original stimela-classic script (implicit
per-cab io context); double-check those against the original cab
definitions before relying on this for a real run.

Known gap, unchanged: "init_ws" wraps an inline CASA python snippet
(stimela2's "casa-task"/"python-code" flavour). shinobi's backends only
execute "binary"-flavour cabs today -- this step is schema-complete but
will misbehave if actually run through a backend; see AGENTS.md.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from shinobi.backends import get_backend
from shinobi.decorators import cab, recipe
from shinobi.recipe import call
from shinobi.schema import CabDef, ParamSchema

# ------------------------------------------------------------------ cabs ---
# Defined once, at module level: these describe *tasks*, not a specific
# run. wsclean/cubical/pybdsm below use a permissive schema built from
# the shared *_opts dicts the original recipe used, instead of typing out
# every parameter by hand -- but the MS name is always a required,
# no-default input, passed explicitly per step (see selfcal() below),
# since it varies per run.


def _infer_dtype(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, (list, tuple)):
        return "list:str"
    return "str"


def cab_from_defaults(
    cab_name: str, command: str, cab_image: str, defaults: dict[str, Any], **extra_inputs: ParamSchema
) -> CabDef:
    inputs = {k: ParamSchema(dtype=_infer_dtype(v), default=v) for k, v in defaults.items()}
    inputs.update(extra_inputs)
    return CabDef(name=cab_name, command=command, image=cab_image, inputs=inputs)


@cab("mstransform", image="quay.io/stimela/casa:1.7.1")
def casa_mstransform(
    vis: str,
    outputvis: str,
    datacolumn: str = "corrected",
    field: str = "",
    correlation: str = "",
    spw: str = "",
    usewtspectrum: bool = False,
    keepflags: bool = True,
    nthreads: int = 1,
    docallib: bool = False,
    callib: str = "",
):
    """CASA mstransform task."""


@cab("listobs", image="quay.io/stimela/casa:1.7.1")
def casa_listobs(vis: str, listfile: str, overwrite: bool = False):
    """CASA listobs task."""


@cab("msutils", image="quay.io/stimela/base:1.2.4")
def msutils(
    msname: str,
    command: str,
    display: bool = True,
    outfile: str = "",
    col1: str = "",
    col2: str = "",
    subtract: bool = False,
):
    """msutils -- summary/sumcols commands."""


@cab("flagdata", image="quay.io/stimela/casa:1.7.1")
def casa_flagdata(
    mode: str,
    msname: str = "",
    vis: str = "",  # this cab is called with either msname= or vis= below
    autocorr: bool = False,
    flagbackup: bool = True,
    spw: str = "",
):
    """CASA flagdata task."""


@cab("flagmanager", image="quay.io/stimela/casa:1.7.1")
def casa_flagmanager(vis: str, mode: str, versionname: str = "", merge: str = ""):
    """CASA flagmanager task."""


@cab("python", image="quay.io/stimela/casa:1.7.1", flavour="casa-task")
def casa_script(vis: str, script: str):
    """Inline CASA python snippet. Schema-only for now -- see module docstring."""


breizorro = CabDef(
    name="breizorro",
    command="breizorro",
    image="breizorro:latest",
    inputs={
        "restored-image": ParamSchema(dtype="File", required=True),
        "boxsize": ParamSchema(dtype="int"),
        "threshold": ParamSchema(dtype="float"),
        "outfile": ParamSchema(dtype="File"),
    },
    outputs={"outfile": ParamSchema(dtype="File")},
)

# Shared cubical/wsclean/pybdsm parameters, straight from the original
# recipe -- minus "data-ms"/"msname", which are always passed explicitly
# per step since they vary per run (see cab_from_defaults() calls below).
cal_opts: dict[str, Any] = {
    "data-column": "DATA",
    "model-pa-rotate": True,
    "dist-ncpu": 32,
    "log-memory": True,
    "out-mode": "sc",
    "out-plots": True,
    "dist-max-chunks": 16,
    "out-casa-gaintables": False,
    "weight-column": "WEIGHT_SPECTRUM",
    "montblanc-dtype": "float",
    "out-overwrite": True,
    "madmax-enable": True,
    "madmax-threshold": 7.0,
    "madmax-plot": False,
    "madmax-estimate": "diag",
    "log-boring": False,
}

im_opts: dict[str, Any] = {
    "local-rms": True,
    "local-rms-window": 128,
    "niter": 400000,
    "size": 6000,
    "scale": 1.3,
    "auto-threshold": 3,
    "channels-out": 5,
    "join-channels": True,
    "fit-spectral-pol": 1,
    "multiscale": True,
    "multiscale-scales": [0, 1, 3, 5],
    "weight": "briggs -0.1",
    "mgain": 0.8,
    "column": "CORRECTED_DATA",
}

extract_opts: dict[str, Any] = {
    "thresh_pix": 20,
    "thresh_isl": 5,
    "adaptive_rms_box": True,
    "format": "fits",
    "port2tigger": True,
    "ncores": 32,
}

cubical = cab_from_defaults(
    "cubical",
    "cubical",
    "quay.io/stimela2/cubical:latest",
    cal_opts,
    **{
        "data-ms": ParamSchema(dtype="MS", required=True),
        "out-name": ParamSchema(dtype="str", required=True),
        "model-list": ParamSchema(dtype="str", required=True),
        "sol-jones": ParamSchema(dtype="str"),
        "sol-term-iters": ParamSchema(dtype="list:int"),
        "out-apply-solver-flags": ParamSchema(dtype="bool"),
        "flags-save": ParamSchema(dtype="str"),
        "flags-apply": ParamSchema(dtype="str"),
        # per-Jones-term params -- cubical accepts one family of these per
        # solvable term (g1-*, g-*, ...), so the exact set can't be fully
        # enumerated ahead of time; these are just the ones this recipe uses.
        "g1-solvable": ParamSchema(dtype="bool"),
        "g1-type": ParamSchema(dtype="str"),
        "g1-time-int": ParamSchema(dtype="int"),
        "g1-freq-int": ParamSchema(dtype="int"),
        "g-solvable": ParamSchema(dtype="bool"),
        "g-type": ParamSchema(dtype="str"),
        "g-time-int": ParamSchema(dtype="int"),
        "g-freq-int": ParamSchema(dtype="int"),
        "g-update-type": ParamSchema(dtype="str"),
        "g-xfer-from": ParamSchema(dtype="File"),
    },
)

wsclean = cab_from_defaults(
    "wsclean",
    "wsclean",
    "quay.io/stimela/wsclean:1.8.0",
    im_opts,
    msname=ParamSchema(dtype="MS", required=True),
    name=ParamSchema(dtype="str", required=True),
    **{
        "fits-mask": ParamSchema(dtype="File"),
        "auto-mask": ParamSchema(dtype="float"),
        "no-dirty": ParamSchema(dtype="bool"),
    },
)

pybdsm = cab_from_defaults(
    "pybdsm",
    "pybdsm",
    "quay.io/stimela/pybdsf:latest",
    extract_opts,
    image=ParamSchema(dtype="File", required=True),
    outfile=ParamSchema(dtype="str", required=True),
)


def _update(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """An updated copy of `defaults` -- `defaults` itself untouched."""
    updated = copy.deepcopy(defaults)
    updated.update(overrides)
    return updated


# ---------------------------------------------------------------- recipe ---

_JOB_NAMES = [
    "flag_MW",
    "im0",
    "cal1",
    "mask1",
    "im1",
    "cal2",
    "mask2",
    "im2",
    "sf3",
    "cal3",
    "mask3",
    "im3",
    "cal4",
    "restore-flags",
    "aplcal",
    "im4",
    "fresid",
    "im5",
    "unflag",
]


def _selected_jobs(start: int, end: int, skip: str) -> set[str]:
    """Which of _JOB_NAMES to actually run: 1-based `start`/`end` (`end`
    0 means no limit) select a range, then `skip` (comma-separated
    1-based indices/ranges into the *full* list, or "all") removes from
    it.
    """
    if skip == "all":
        return set()
    selected = set(_JOB_NAMES[start - 1 : end or None])
    for chunk in skip.split(",") if skip else []:
        if "-" in chunk:
            lo, hi = (int(x) for x in chunk.split("-"))
            selected -= set(_JOB_NAMES[lo - 1 : hi])
        elif chunk:
            selected.discard(_JOB_NAMES[int(chunk) - 1])
    return selected


@recipe()
def selfcal(
    ms0: str = "1767335776_sdp_l0.ms",
    rawmsdir: str = "/home/tanitarh/MWproject/rawdata",
    gaindir: str = "/home/tanitarh/MWproject/gains-crosscal",
    applyspec: str = "applyspec.txt",
    target: str = "CHI-OPH",
    mslabel: str = "line",
    msdir: str = "msdir",
    input_dir: str = "absolute path to input folder",
    outdir: str = "output-MAY15",
    start: int = 1,
    end: int = 0,  # 0 means no limit
    skip: str = "",  # comma-separated 1-based indices/ranges, or "all"
    init: bool = False,
    reset: bool = False,
    runtime: str = "apptainer",
):
    """MeerKAT selfcal: split off the target, then four rounds of
    image -> mask -> calibrate, finishing with a residual check.
    """
    backend = get_backend(runtime, workdir=os.getcwd())

    prefix = f"{Path(ms0).stem}-{target}"
    ms = f"{prefix}-{mslabel}.ms"
    obsinfo_dir = os.path.join(outdir, "obsinfo")
    legacy_flags = "ninja_legacy"  # flagset preserving incoming flags across selfcal

    def get_name(n: int) -> str:
        return f"{prefix}-{n}"

    def out(name: str) -> str:
        return os.path.join(outdir, name)

    def indir(name: str) -> str:
        return os.path.join(input_dir, name)

    def msdir_path(name: str) -> str:
        return os.path.join(msdir, name)

    def obsinfo(name: str) -> str:
        return os.path.join(obsinfo_dir, name)

    def run(label: str, cab_obj: CabDef, **params: Any) -> None:
        print(f"### {label}")
        result = call(cab_obj, backend, **params)
        if not result.success:
            raise RuntimeError(f"'{label}' failed:\n{result.stderr}")

    # -------------------------------------------------------- init/reset ---

    if init:
        print("Splitting MS and initialising flags and weights")
        run(
            "split:: Split data for selfcal",
            casa_mstransform,
            vis=os.path.join(rawmsdir, ms0),
            outputvis=msdir_path(ms),
            datacolumn="corrected",
            field=target,
            correlation="XX,YY",
            spw="*:1.418734071768GHz~1.422077431768GHz",
            usewtspectrum=True,
            keepflags=True,
            nthreads=32,
            docallib=True,
            callib=os.path.join(gaindir, applyspec),
        )
        run(
            "info1:: Get observation info",
            casa_listobs,
            vis=msdir_path(ms),
            listfile=obsinfo(f"{prefix}-obsinfo.txt"),
            overwrite=True,
        )
        run(
            "info2:: Get observation information as a json file",
            msutils,
            msname=msdir_path(ms),
            command="summary",
            display=False,
            outfile=obsinfo(f"{prefix}-summary.json"),
        )
        run(
            "flag_pp:: Flag auto correlations",
            casa_flagdata,
            msname=msdir_path(ms),
            mode="manual",
            autocorr=True,
            flagbackup=False,
        )

    if init or reset:
        run(
            "uflag_spw:: Unflag all spws",
            casa_flagdata,
            msname=msdir_path(ms),
            mode="unflag",
            spw="*",
            flagbackup=False,
        )
        run(
            "leg-flags:: Save incoming flags",
            casa_flagmanager,
            vis=msdir_path(ms),
            mode="save",
            versionname=legacy_flags,
            merge="replace",
        )
        run(
            "init_ws:: Setting uniform weights",
            casa_script,
            vis=msdir_path(ms),
            script=f"""
vis = os.path.join(os.environ['MSDIR'], '{ms}')
initweights(vis=vis, wtmode='ones', dowtsp=True)
""",
        )

    if skip == "all":
        return

    # -------------------------------------------------------------- jobs ---

    selected = _selected_jobs(start, end, skip)

    def job(job_name: str, label: str, cab_obj: CabDef, **params: Any) -> None:
        if job_name not in selected:
            print(f"Skipping job: {job_name}")
            return
        run(label, cab_obj, **params)

    job(
        "flag_MW",
        "flag_MW:: Flag MW before selfcal",
        casa_flagdata,
        msname=msdir_path(ms),
        mode="manual",
        spw="*:1420.38~1420.65MHz",  # bright lines that would bias selfcal gains
    )

    # Uses an external input mask
    job(
        "im0",
        "im0:: Image for initial model",
        wsclean,
        **_update(
            im_opts,
            {
                "msname": ms,
                "name": get_name(0),
                "column": "DATA",
                "fits-mask": indir("initalmask_t10_b125.fits"),
            },
        ),
    )

    job(
        "cal1",
        "cal1::f-slope calibration -> corr-data",
        cubical,
        **_update(
            cal_opts,
            {
                "data-ms": ms,
                "out-name": get_name(1),
                "model-list": "MODEL_DATA",
                "sol-jones": "g1",
                "sol-term-iters": [50],
                "g1-solvable": True,
                "g1-type": "f-slope",
                "g1-time-int": 20,
                "g1-freq-int": 256,
            },
        ),
    )

    mask1 = "breizorro_mask_1.fits"
    job(
        "mask1",
        "mask1:: Mask for image 1 from image 0",
        breizorro,
        **{
            "restored-image": out(f"{get_name(0)}-MFS-image.fits"),
            "boxsize": 60,
            "threshold": 7,
            "outfile": mask1,
        },
    )

    job(
        "im1",
        "im1:: Round 1 image",
        wsclean,
        **_update(
            im_opts,
            {
                "msname": ms,
                "name": get_name(1),
                "column": "CORRECTED_DATA",
                "fits-mask": out(mask1),
                "auto-threshold": 2,
            },
        ),
    )

    job(
        "cal2",
        "cal2:: Gain -> CORRECTED_DATA",
        cubical,
        **_update(
            cal_opts,
            {
                "data-ms": ms,
                "out-name": get_name(2),
                "model-list": "MODEL_DATA",
                "out-mode": "sc",
                "sol-jones": "g",
                "sol-term-iters": [50],
                "g-solvable": True,
                "g-type": "complex-diag",
                "g-time-int": 20,
                "g-freq-int": 256,
                "g-update-type": "complex-diag",
            },
        ),
    )

    mask2 = "breizorro_mask_2.fits"
    job(
        "mask2",
        "mask2:: Mask for image 2 from image 1",
        breizorro,
        **{
            "restored-image": out(f"{get_name(1)}-MFS-image.fits"),
            "boxsize": 65,
            "threshold": 7,
            "outfile": mask2,
        },
    )

    job(
        "im2",
        "im2::Round 2 image",
        wsclean,
        **_update(im_opts, {"msname": ms, "name": get_name(2), "fits-mask": out(mask2), "auto-threshold": 1}),
    )

    model_points_prefix = f"{get_name(2)}-skymodel"
    job(
        "sf3",
        "sf3::Extract point sources",
        pybdsm,
        **_update(
            extract_opts,
            {
                "image": out(f"{get_name(2)}-MFS-image.fits"),
                "thresh_pix": 7,
                "thresh_isl": 3,
                "outfile": f"{model_points_prefix}.fits",
            },
        ),
    )

    lsm_from_im2 = f"{model_points_prefix}.lsm.html"
    job(
        "cal3",
        "cal3:: Gain -> CORR_RES",
        cubical,
        **_update(
            cal_opts,
            {
                "data-ms": ms,
                "out-name": get_name(3),
                "model-list": lsm_from_im2,
                "out-mode": "sr",
                "sol-jones": "g",
                "sol-term-iters": [50],
                "g-solvable": True,
                "g-type": "complex-diag",
                "g-time-int": 43,
                "g-freq-int": 256,
            },
        ),
    )

    mask3 = "breizorro_mask_3.fits"
    job(
        "mask3",
        "mask3:: Mask for image 3 using image 2",
        breizorro,
        **{
            "restored-image": out(f"{get_name(2)}-MFS-image.fits"),
            "boxsize": 61,
            "threshold": 5,
            "outfile": mask3,
        },
    )

    job(
        "im3",
        "im3::Round 3 image",
        wsclean,
        **_update(im_opts, {"msname": ms, "name": get_name(3), "fits-mask": out(mask3), "auto-threshold": 1}),
    )

    # model-list = MODEL_DATA plus the point-source sky model extracted from im2
    final_model = f"MODEL_DATA+{out(lsm_from_im2)}"
    job(
        "cal4",
        "cal4:: G -> CORR_RES",
        cubical,
        **_update(
            cal_opts,
            {
                "data-ms": ms,
                "out-name": get_name(4),
                "out-mode": "sr",
                "model-list": final_model,
                "sol-jones": "g",
                "sol-term-iters": [50],
                "g-solvable": True,
                "g-type": "complex-2x2",
                "g-time-int": 43,
                "g-freq-int": 256,
                "g-update-type": "complex-diag",
            },
        ),
    )

    job(
        "restore-flags",
        "restore-flags:: Restore incoming flags",
        casa_flagmanager,
        vis=msdir_path(ms),
        mode="restore",
        versionname=legacy_flags,
        merge="replace",
    )

    job(
        "aplcal",
        "aplcal:: Apply previous gains to unflagged DATA column",
        cubical,
        **_update(
            cal_opts,
            {
                "data-ms": ms,
                "out-name": get_name(5),
                "out-mode": "ar",
                "out-plots": False,
                "out-apply-solver-flags": False,
                "model-list": final_model,
                "madmax-enable": False,  # internal flagging would re-flag the bright lines
                "flags-save": "0",
                "sol-jones": "g",
                "g-solvable": False,
                "g-type": "complex-2x2",
                "g-time-int": 43,
                "g-freq-int": 256,
                "g-update-type": "complex-diag",
                "flags-apply": legacy_flags,
                "g-xfer-from": out(f"{get_name(4)}-G-field_0-ddid_None.parmdb"),
            },
        ),
    )

    job(
        "im4",
        "im4::Image final selfcal residual",
        wsclean,
        **_update(
            im_opts,
            {
                "msname": ms,
                "name": get_name(4),
                "channels-out": 3,
                "auto-threshold": 1,
                "auto-mask": 5,
                "multiscale": False,
                "local-rms-window": 64,
                "fit-spectral-pol": 1,
            },
        ),
    )

    job(
        "fresid",
        "fresid:: Subtract what's left after imaging final selfcal residual",
        msutils,
        command="sumcols",
        msname=msdir_path(ms),
        col1="CORRECTED_DATA",
        col2="MODEL_DATA",
        subtract=True,
    )

    # This final image should be a noisy residual with no emission left. If
    # it still shows emission, the selfcal strategy needs revisiting.
    job(
        "im5",
        "im5::Final ninja residual map",
        wsclean,
        **_update(
            im_opts,
            {
                "niter": 0,
                "msname": ms,
                "name": get_name(5),
                "auto-threshold": 0.5,
                "auto-mask": 3,
                "local-rms-window": 64,
                "no-dirty": True,
            },
        ),
    )

    job(
        "unflag",
        "unflag:: Unflag all channels",
        casa_flagdata,
        vis=msdir_path(ms),
        mode="unflag",
        spw="*",
        flagbackup=False,
    )
