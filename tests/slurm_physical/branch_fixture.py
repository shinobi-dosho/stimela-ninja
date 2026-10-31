"""Tiny native-Casacore tasks executed inside an Apptainer branch allocation."""

from pathlib import Path

from pydantic import BaseModel

from shinobi import Cab, DatasetAccess, MeasurementSetV2, Recipe
from shinobi.steps.schema import OutputRef, ParamMeta


class Empty(BaseModel):
    pass


class CreateInputs(BaseModel):
    script: str
    ms: MeasurementSetV2


class MSOutput(BaseModel):
    ms: MeasurementSetV2


class ReadInputs(BaseModel):
    script: str
    ms: MeasurementSetV2
    report: str
    delay: float


class ReadOutput(BaseModel):
    artifact: Path


READ = """import json, socket, sys, time
from pathlib import Path
from casacore.tables import table
ms, report, delay = sys.argv[1:]
start = time.time()
print('BRANCH_READER_STARTED ' + Path(report).name, flush=True)
with table(ms, ack=False) as value:
    scans = value.getcol('SCAN_NUMBER').tolist()
time.sleep(float(delay))
Path(report).write_text(json.dumps(dict(scans=scans, host=socket.gethostname(), start=start, end=time.time())))
"""


def build(image: str, scan: int = 7, first_delay: float = 0, second_delay: float = 0):
    create = Cab(
        name="create-ms",
        command="/usr/bin/python3 -c",
        image=image,
        inputs_model=CreateInputs,
        outputs_model=MSOutput,
        field_meta={"script": ParamMeta(positional_head=True), "ms": ParamMeta(positional=True, write_path=True)},
        dataset_accesses=[DatasetAccess(field="ms", mode="create", allow_schema_change=True, allow_row_count_change=True)],
    )
    read = Cab(
        name="read-ms",
        command="/usr/bin/python3 -c",
        image=image,
        inputs_model=ReadInputs,
        outputs_model=ReadOutput,
        field_meta={
            "script": ParamMeta(positional_head=True),
            "ms": ParamMeta(positional=True),
            "report": ParamMeta(positional=True, write_path=True),
            "delay": ParamMeta(positional=True),
            "artifact": ParamMeta(implicit="{report}"),
        },
        dataset_accesses=[DatasetAccess(field="ms", mode="read")],
    )
    script = (
        "import casacore.tables as t,numpy as n,sys;"
        "m=t.default_ms(sys.argv[1]);"
        "m.addcols(t.maketabdesc([t.makearrcoldesc('DATA',0j,ndim=2)]));"
        f"m.addrows(4);m.putcol('SCAN_NUMBER',n.full(4,{scan},dtype=n.int32));m.close()"
    )
    recipe = Recipe(name="branch", inputs_model=Empty, outputs_model=Empty, cache=True, sandbox=True, max_workers=1)
    recipe.add_step("create", create, script=script, ms="data/noise.ms")
    recipe.add_step(
        "read", read, scatter=["report", "delay"], script=READ, ms=OutputRef(step="create", field="ms"), report=["first.json", "second.json"], delay=[first_delay, second_delay]
    )
    return recipe
