"""Ninja selfcal recipe -- shinobi conversion of a stimela-classic recipe
(https://github.com/SpheMakh/ninja/blob/319cc37/ninja-recipe.py).

What changed structurally, and why:

* stimela.Recipe(...).add(...) / .run(subset) -> plain Python. A "step" is
  just a (name, cab, params) record in a list; running a chosen subset is
  a list slice/filter -- which is exactly what the original script was
  already doing internally (jobs[START:END], removing skipped indices).
  There's no recipe engine here to hand a subset of names to.

* spf(...) and the "file:output"/"file:input"/"file:msfile" path-suffix
  convention -> plain os.path.join()/f-strings via a few local helper
  functions (out(), indir(), msdir_path()). Paths are just Python strings
  in shinobi; there's no reason to route them through a mini
  substitution language to say which base directory they're relative to.

* JOB_TYPE="singularity" -> get_backend("apptainer", workdir=...).

* Cabs are defined inline via shinobi.decorators.cab / a small
  defaults-dict helper for this self-contained example. In a real
  deployment, wsclean/breizorro/cubical are also available as real
  cult-cargo cab definitions (shinobi.loaders.cultcargo.load_file) and
  should be loaded from there instead of redeclared here.

Known gap: the original "casa_script" step runs an inline CASA Python
snippet (stimela2 would call this the "casa-task"/"python-code" flavour).
shinobi's backends currently only execute "binary"-flavour cabs -- other
flavours are schema-only for now (see AGENTS.md). That step is converted
faithfully as a CabDef but will raise if actually run through a backend
today.

Path resolution note: the original recipe leaned on stimela classic's
implicit per-parameter directory context (a bare filename resolves
against a cab's default io context -- e.g. "msfile" -- unless suffixed
with :input/:output, and recipe.add(..., output=, input=, msdir=) could
override that context per step). shinobi doesn't replicate that inference
-- every path below is constructed explicitly. Where the original's
implicit context was genuinely ambiguous from the script alone, that's
called out in a comment; double-check those against the original cab
definitions before relying on this for a real run.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from shinobi.backends import get_backend
from shinobi.decorators import cab
from shinobi.recipe import call
from shinobi.schema import CabDef, ParamSchema

# ---------------------------------------------------------------- config ---
# Same knobs as the original script (there they were set via `-g NAME=value`
# on the stimela CLI; here they're just Python -- edit them directly, or
# wire up argparse/click if you want CLI overrides).

INPUT = "absolute path to input folder"  # must exist and contain all input files

RAWMSDIR = "/home/tanitarh/MWproject/rawdata"
MS0 = "1767335776_sdp_l0.ms"

GAINDIR = "/home/tanitarh/MWproject/gains-crosscal"
APPLYSPEC = "applyspec.txt"
TARGET = "CHI-OPH"

PREFIX = f"1767335776_sdp_l0-{TARGET}"
MSLABEL = "line"
MSDIR = "msdir"
OUTPUT = "output-MAY15"  # use the same output dir when running caracal (esp. the line worker)

DRYRUN = False
INIT = False  # delete + regenerate the MS from MS0, then reset flags/weights
RESET = False  # just reset flags/weights

START = 1  # 1-based, like the original
END: int | None = None
SKIP: str | None = None  # comma-separated step numbers/ranges, or "all"

# Channels/frequencies with bright lines that would bias selfcal gains --
# flagged before selfcal, unflagged again before applying the final gains
FLAG_SPW = "*:1420.38~1420.65MHz"

MS = f"{PREFIX}-{MSLABEL}.ms"
obsinfo_dir = os.path.join(OUTPUT, "obsinfo")
LOGDIR = Path(OUTPUT) / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)
STAMP = f"{datetime.now():%m-%d}"  # noqa: DTZ005
LOGFILE = LOGDIR / f"log-{PREFIX}_{STAMP}.txt"

legacy_flags = "ninja_legacy"  # flagset used to preserve incoming flags across selfcal


def get_name(n: int) -> str:
    return f"{PREFIX}-{n}"


def out(name: str) -> str:
    return os.path.join(OUTPUT, name)


def indir(name: str) -> str:
    return os.path.join(INPUT, name)


def msdir_path(name: str) -> str:
    return os.path.join(MSDIR, name)


def obsinfo(name: str) -> str:
    return os.path.join(obsinfo_dir, name)


def _update(orig: dict, opts: dict) -> dict:
    """Same helper the original recipe used: an updated copy, original untouched."""
    updated = copy.deepcopy(orig)
    updated.update(opts)
    return updated


backend = get_backend("apptainer", workdir=os.getcwd())


# ------------------------------------------------------------------ cabs ---
# Defined inline for a self-contained example. wsclean/cubical/pybdsm below
# use a permissive schema built from the same shared *_opts dicts the
# original recipe used, instead of typing out every parameter by hand.


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
    vis: str = "",  # the original recipe calls this cab with either msname= or vis=
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
    """Inline CASA python snippet. Schema-only for now -- shinobi's
    backends only execute "binary"-flavour cabs; see the module docstring.
    """


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

# Common cubical/wsclean/pybdsm parameters, straight from the original recipe
cal_opts: dict[str, Any] = {
    "data-ms": MS,
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
    "msname": MS,
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
        "out-name": ParamSchema(dtype="str", required=True),
        "model-list": ParamSchema(dtype="str", required=True),
        "sol-jones": ParamSchema(dtype="str"),
        "sol-term-iters": ParamSchema(dtype="list:int"),
    },
)

wsclean = cab_from_defaults(
    "wsclean",
    "wsclean",
    "quay.io/stimela/wsclean:1.8.0",
    im_opts,
    name=ParamSchema(dtype="str", required=True),
    **{"fits-mask": ParamSchema(dtype="File"), "auto-mask": ParamSchema(dtype="float"), "no-dirty": ParamSchema(dtype="bool")},
)

pybdsm = cab_from_defaults(
    "pybdsm",
    "pybdsm",
    "quay.io/stimela/pybdsf:latest",
    extract_opts,
    image=ParamSchema(dtype="File", required=True),
    outfile=ParamSchema(dtype="str", required=True),
)


# ----------------------------------------------------------------- steps ---


@dataclass
class Step:
    name: str
    cab: CabDef
    params: dict[str, Any]
    label: str


inits: list[Step] = []
resets: list[Step] = []
jobs: list[Step] = []


def declare(bucket: list[Step], name: str, cab_def: CabDef, params: dict[str, Any], label: str) -> None:
    bucket.append(Step(name=name, cab=cab_def, params=params, label=label))


declare(
    inits,
    "split",
    casa_mstransform,
    {
        "vis": os.path.join(RAWMSDIR, MS0),
        "outputvis": msdir_path(MS),
        "datacolumn": "corrected",
        "field": TARGET,
        "correlation": "XX,YY",
        "spw": "*:1.418734071768GHz~1.422077431768GHz",
        "usewtspectrum": True,
        "keepflags": True,
        "nthreads": 32,
        "docallib": True,
        "callib": os.path.join(GAINDIR, APPLYSPEC),
    },
    "split:: Split data for selfcal",
)

# ---- prelims ---------

declare(
    inits,
    "info1",
    casa_listobs,
    {"vis": msdir_path(MS), "listfile": obsinfo(f"{PREFIX}-obsinfo.txt"), "overwrite": True},
    "info1:: Get observation info",
)

declare(
    inits,
    "info2",
    msutils,
    {
        "msname": msdir_path(MS),
        "command": "summary",
        "display": False,
        "outfile": obsinfo(f"{PREFIX}-summary.json"),
    },
    "info2:: Get observation information as a json file",
)

# -------------------------- Iter 1 -------------------------------------

declare(
    inits,
    "flag_pp",
    casa_flagdata,
    {"msname": msdir_path(MS), "mode": "manual", "autocorr": True, "flagbackup": False},
    "flag_pp:: Flag auto correlations",
)

declare(
    resets,
    "uflag_spw",
    casa_flagdata,
    {"msname": msdir_path(MS), "mode": "unflag", "spw": "*", "flagbackup": False},
    "uflag_spw:: Unflag all spws",
)

declare(
    resets,
    "leg-flags",
    casa_flagmanager,
    {"vis": msdir_path(MS), "mode": "save", "versionname": legacy_flags, "merge": "replace"},
    "leg-flags:: Save incoming flags",
)

declare(
    resets,
    "init_ws",
    casa_script,
    {
        "vis": msdir_path(MS),
        "script": f"""
vis = os.path.join(os.environ['MSDIR'],'{MS}')
initweights(vis=vis, wtmode='ones', dowtsp=True)
""",
    },
    "init_ws:: Setting uniform weights",
)

declare(
    jobs,
    "flag_MW",
    casa_flagdata,
    {"msname": msdir_path(MS), "mode": "manual", "spw": FLAG_SPW},
    "flag_MW:: Flag MW before selfcal",
)

# Uses external input mask
declare(
    jobs,
    "im0",
    wsclean,
    _update(im_opts, {"name": get_name(0), "column": "DATA", "fits-mask": indir("initalmask_t10_b125.fits")}),
    "im0:: Image for initial model",
)

declare(
    jobs,
    "cal1",
    cubical,
    _update(
        cal_opts,
        {
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
    "cal1::f-slope calibration -> corr-data",
)

# ----------------------------------- Iter 2 -------------------------------------

mask1 = "breizorro_mask_1.fits"
declare(
    jobs,
    "mask1",
    breizorro,
    {
        "restored-image": out(f"{get_name(0)}-MFS-image.fits"),
        "boxsize": 60,
        "threshold": 7,
        "outfile": mask1,
    },
    "mask1:: Mask for image 1 from image 0",
)

declare(
    jobs,
    "im1",
    wsclean,
    _update(
        im_opts,
        {
            "msname": MS,
            "name": get_name(1),
            "column": "CORRECTED_DATA",
            "fits-mask": out(mask1),
            "auto-threshold": 2,
        },
    ),
    "im1:: Round 1 image",
)

declare(
    jobs,
    "cal2",
    cubical,
    _update(
        cal_opts,
        {
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
    "cal2:: Gain -> CORRECTED_DATA",
)

# ----------------------------------- Iter 3 -------------------------------------

mask2 = "breizorro_mask_2.fits"
declare(
    jobs,
    "mask2",
    breizorro,
    {
        "restored-image": out(f"{get_name(1)}-MFS-image.fits"),
        "boxsize": 65,
        "threshold": 7,
        "outfile": mask2,
    },
    "mask2:: Mask for image 2 from image 1",
)

declare(
    jobs,
    "im2",
    wsclean,
    _update(im_opts, {"msname": MS, "name": get_name(2), "fits-mask": out(mask2), "auto-threshold": 1}),
    "im2::Round 2 image",
)

model_points_prefix = f"{get_name(2)}-skymodel"
declare(
    jobs,
    "sf3",
    pybdsm,
    _update(
        extract_opts,
        {
            "image": out(f"{get_name(2)}-MFS-image.fits"),
            "thresh_pix": 7,
            "thresh_isl": 3,
            "outfile": f"{model_points_prefix}.fits",
        },
    ),
    "sf3::Extract point sources",
)

lsm_from_im2 = f"{model_points_prefix}.lsm.html"
declare(
    jobs,
    "cal3",
    cubical,
    _update(
        cal_opts,
        {
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
    "cal3:: Gain -> CORR_RES",
)

# ----------------------------------- Iter 4 -------------------------------------

mask3 = "breizorro_mask_3.fits"
declare(
    jobs,
    "mask3",
    breizorro,
    {
        "restored-image": out(f"{get_name(2)}-MFS-image.fits"),
        "boxsize": 61,
        "threshold": 5,
        "outfile": mask3,
    },
    "mask3:: Mask for image 3 using image 2",
)

declare(
    jobs,
    "im3",
    wsclean,
    _update(im_opts, {"msname": MS, "name": get_name(3), "fits-mask": out(mask3), "auto-threshold": 1}),
    "im3::Round 3 image",
)

# model-list = MODEL_DATA plus the point-source sky model extracted from im2
final_model = f"MODEL_DATA+{out(lsm_from_im2)}"
declare(
    jobs,
    "cal4",
    cubical,
    _update(
        cal_opts,
        {
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
    "cal4:: G -> CORR_RES",
)

declare(
    jobs,
    "restore-flags",
    casa_flagmanager,
    {"vis": msdir_path(MS), "mode": "restore", "versionname": legacy_flags, "merge": "replace"},
    "restore-flags:: Restore incoming flags",
)

declare(
    jobs,
    "aplcal",
    cubical,
    _update(
        cal_opts,
        {
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
    "aplcal:: Apply previous gains to unflagged DATA column",
)

# ---------------------- Apply gains to DATA ----------------------

declare(
    jobs,
    "im4",
    wsclean,
    _update(
        im_opts,
        {
            "msname": MS,
            "name": get_name(4),
            "channels-out": 3,
            "auto-threshold": 1,
            "auto-mask": 5,
            "multiscale": False,
            "local-rms-window": 64,
            "fit-spectral-pol": 1,
        },
    ),
    "im4::Image final selfcal residual",
)

# -------------------- Subtract whatever emission is left behind --------------------

declare(
    jobs,
    "fresid",
    msutils,
    {
        "command": "sumcols",
        "msname": msdir_path(MS),
        "col1": "CORRECTED_DATA",
        "col2": "MODEL_DATA",
        "subtract": True,
    },
    "fresid:: Subtract what's left after imaging final selfcal residual",
)

# This final image should be a noisy residual with no emission left. If it
# still shows emission, the selfcal strategy needs revisiting.
declare(
    jobs,
    "im5",
    wsclean,
    _update(
        im_opts,
        {
            "niter": 0,
            "msname": MS,
            "name": get_name(5),
            "auto-threshold": 0.5,
            "auto-mask": 3,
            "local-rms-window": 64,
            "no-dirty": True,
        },
    ),
    "im5::Final ninja residual map",
)

declare(
    jobs,
    "unflag",
    casa_flagdata,
    {"vis": msdir_path(MS), "mode": "unflag", "spw": "*", "flagbackup": False},
    "unflag:: Unflag all channels",
)


# ------------------------------------------------------------- execution ---

njobs = len(jobs)
job_names = [j.name for j in jobs]

skipus: list[int] = []
if SKIP and SKIP != "all":
    for chunk in SKIP.split(","):
        if "-" in chunk:
            rstart, rend = (int(x) for x in chunk.split("-"))
            skipus += list(range(rstart, rend + 1))
        else:
            skipus.append(int(chunk))

run_jobs = jobs[START - 1 : END]
for skip in skipus:
    victim = jobs[skip - 1]
    if victim in run_jobs:
        print(f"Skipping job: {victim.name}\n")
        run_jobs.remove(victim)

if DRYRUN:
    table = Table(title="Ninja Step Indices")
    for name in job_names:
        table.add_column(name)
    table.add_row(*(str(i) for i in range(1, njobs + 1)))
    Console().print(table)
    raise SystemExit(f"### Dryrun ### \n### Jobs: {[j.name for j in run_jobs]}")


def run_steps(steps: list[Step]) -> None:
    for step in steps:
        print(f"### {step.label}")
        result = call(step.cab, backend, **step.params)
        if not result.success:
            raise RuntimeError(f"step '{step.name}' failed:\n{result.stderr}")


if INIT:
    print("Splitting MS and initialising flags and weights")
    run_steps(inits + resets)
elif RESET:
    print("Resetting flags and weights")
    run_steps(resets)

if SKIP == "all":
    raise SystemExit(0)

print(f"############### NINJA:: Running jobs {[j.name for j in run_jobs]}")
run_steps(run_jobs)
