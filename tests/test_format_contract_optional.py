"""Format declarations and unavailable-inspector behavior without optional stacks."""

import builtins
import subprocess
import sys
import textwrap

import pytest

from shinobi.fits import FITSStatus, inspect_fits_image, inspect_fits_table


def test_declarations_work_without_optional_imports():
    code = textwrap.dedent("""
        import builtins
        import json
        import sys
        from pathlib import Path

        original_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name.split('.')[0] in {'astropy', 'casacore'}:
                raise AssertionError(f'declaration imported optional dependency: {name}')
            return original_import(name, *args, **kwargs)

        builtins.__import__ = blocked_import

        from pydantic import BaseModel
        from shinobi import MSv2, FITSImage, FITSTable

        class Inputs(BaseModel):
            ms: MSv2
            image: FITSImage
            table: FITSTable

        paths = {'ms': 'future.ms', 'image': 'future-image.fits', 'table': 'future-table.fits'}
        model = Inputs(**paths)
        assert model.model_dump() == {name: Path(value) for name, value in paths.items()}
        assert json.loads(model.model_dump_json()) == paths
        properties = Inputs.model_json_schema()['properties']
        for name, kind, profile in (
            ('ms', 'measurement-set-v2', 'msv2-structural/v1'),
            ('image', 'fits-image', 'fits-image/v1'),
            ('table', 'fits-table', 'fits-table/v1'),
        ):
            assert properties[name]['format'] == 'path'
            assert properties[name]['x-shinobi-dataset'] == {'kind': kind, 'profile': profile}
        assert not any(name.split('.')[0] in {'astropy', 'casacore'} for name in sys.modules)
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("inspect", [inspect_fits_image, inspect_fits_table])
@pytest.mark.parametrize("error", [ImportError("No module named 'astropy'"), OSError("shared library cannot be loaded")])
def test_fits_inspector_unavailable_is_actionable(tmp_path, monkeypatch, inspect, error):
    candidate = tmp_path / "candidate.fits"
    candidate.touch()
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name.split(".")[0] == "astropy":
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    descriptor = inspect(candidate)
    assert descriptor.status is FITSStatus.INSPECTOR_UNAVAILABLE
    assert str(error) in descriptor.message
    assert "install stimela-ninja[fits] in the inspection environment" in descriptor.message
