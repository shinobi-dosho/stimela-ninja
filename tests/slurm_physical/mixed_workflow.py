"""Tiny ordinary mixed graph for physical local/Slurm acceptance.

Requires python-casacore and numpy in the shared worker environment.
"""

from pathlib import Path
import json
from pydantic import BaseModel
from shinobi import DatasetAccess, MSv2, Recipe, SlurmPlacement, pystep


class Options(BaseModel):
    fail: bool = False


class Dataset(BaseModel):
    ms: MSv2


class BranchInputs(Dataset):
    fail: bool = False


class Value(BaseModel):
    value: int


class Values(BaseModel):
    value: list[int]


class Report(BaseModel):
    report: Path


@pystep(dataset_accesses=[DatasetAccess(field="ms", mode="create")])
def create(ms: MSv2) -> Dataset:
    import casacore.tables as tables
    import numpy as np

    with tables.default_ms(str(ms)) as table:
        table.addcols(tables.maketabdesc([tables.makearrcoldesc("DATA", 0j, ndim=2)]))
        table.addrows(2)
        table.putcol("SCAN_NUMBER", np.ones(2, dtype=np.int32))
    return Dataset(ms=ms)


@pystep(dataset_accesses=[DatasetAccess(field="ms", mode="read")])
def read(ms: MSv2, index: int, fail: bool = False) -> Value:
    import casacore.tables as tables
    import time

    import socket

    print(f"start host={socket.gethostname()} slice={index}", flush=True)  # noqa: T201 -- live acceptance marker
    time.sleep(2 if index == 0 else 30)
    if fail and index == 1:
        raise RuntimeError("requested mixed smoke failure")
    print(f"end host={socket.gethostname()} slice={index}", flush=True)  # noqa: T201 -- live acceptance marker
    with tables.table(str(ms), ack=False) as table:
        return Value(value=int(table.getcell("SCAN_NUMBER", 0)) + index)


@pystep()
def gather(report: Path, values: list[list[int]]) -> Report:
    report.write_text(json.dumps({"values": values}) + "\n")
    return Report(report=report)


branch = Recipe(name="branch", inputs_model=BranchInputs, outputs_model=Values, max_workers=2)
branch.add_step("create", create, ms=branch.inputs.ms)
branch.add_step("read", read, ms=branch.outputs.create.ms, scatter=["index"], index=[0, 1], fail=branch.inputs.fail)
branch.set_output("value", branch.outputs.read.value)
recipe = Recipe(name="mixed-smoke", inputs_model=Options, outputs_model=Report, max_workers=2, cache=True)
recipe.add_step(
    "cases",
    branch,
    placement=SlurmPlacement(sbatch_opts={"partition": "dev", "cpus-per-task": "8", "mem": "1G", "time": "00:05:00"}, max_parallel=2),
    scatter=["ms"],
    ms=[Path("a/data.ms"), Path("b/data.ms")],
    fail=recipe.inputs.fail,
)
recipe.add_step("report", gather, report=Path("report.json"), values=recipe.outputs.cases.value)
recipe.set_output("report", recipe.outputs.report.report)
