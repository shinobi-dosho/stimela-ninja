"""Production leaf/cache integration for an MS plus a continued model file."""

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from shinobi import DatasetAccess, DatasetColumns, MSv2
from shinobi.derived import DerivedRead
from shinobi.products import FamilySpec, ProductFamily
from shinobi.steps.dispatch import _dispatch
from shinobi.steps.schema import Cab, Mutability, ParamMeta, Policies
from tests._dataset_fixtures import ROWS, attempts, make_ms, scans

pytest.importorskip("casacore.tables")
pytest.importorskip("numpy")


class Inputs(BaseModel):
    ms: MSv2
    prefix: str = "image"
    by: int = 1
    fail: int = 0
    script: str = """from pathlib import Path
import sys
import numpy as np
from casacore.tables import table
ms,prefix,by,fail=sys.argv[1:]
with table(ms, readonly=False, ack=False) as t:
    t.putcol('SCAN_NUMBER',t.getcol('SCAN_NUMBER')+int(by))
p=Path(prefix+'-model.fits')
p.write_text(str(int(p.read_text())+int(by)))
if int(fail): raise RuntimeError('tool failed')
"""


class Outputs(BaseModel):
    model: ProductFamily[Path] | None = None


def cab():
    family = FamilySpec(coordinates={}, rules=[{"path": "{prefix}-model.fits", "accept_existing": True}])
    read = FamilySpec(coordinates={}, rules=[{"path": "{prefix}-model.fits", "required": True}])
    return Cab(
        name="continue",
        command=sys.executable,
        inputs_model=Inputs,
        outputs_model=Outputs,
        policies=Policies(prefix="-"),
        input_mutability={"ms": Mutability.MUTABLE},
        field_meta={
            "script": ParamMeta(nom_de_guerre="c"),
            "ms": ParamMeta(positional=True),
            "prefix": ParamMeta(positional=True),
            "by": ParamMeta(positional=True),
            "fail": ParamMeta(positional=True),
            "model": ParamMeta(family=family),
        },
        derived_reads={"model": DerivedRead(member="file", family=read)},
        dataset_accesses=[DatasetAccess(field="ms", mode="write", columns=DatasetColumns(write=("SCAN_NUMBER",)))],
    )


@pytest.mark.parametrize("sandbox", [False, True])
def test_continuation_resume_changed_params_and_joint_rollback(tmp_path, monkeypatch, sandbox):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "ownership.json"))
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(tmp_path / "sandboxes"))
    ms = make_ms(tmp_path / "obs.ms")
    model = tmp_path / "image-model.fits"
    model.write_text("0")
    kwargs = {"ms": ms, "cache": True, "cache_dir": str(tmp_path / "cache"), "sandbox": sandbox}
    first = _dispatch(cab(), None, **kwargs)
    assert first.success and not first.cached
    assert model.read_text() == "1" and scans(ms) == [2] * ROWS
    assert _dispatch(cab(), None, **kwargs).cached
    second = _dispatch(cab(), None, by=2, **kwargs)
    assert not second.cached and model.read_text() == "2" and scans(ms) == [3] * ROWS
    failed = _dispatch(cab(), None, by=3, fail=1, **kwargs)
    assert not failed.success
    assert model.read_text() == "2" and scans(ms) == [3] * ROWS
    leaf = attempts(tmp_path)[-1].leaves[0]
    assert len(leaf.mutations) == 1 and len(leaf.auxiliary_mutations) == 1
