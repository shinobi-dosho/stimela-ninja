"""Tests for the `ninja cabs list`/`ninja cabs show <name>` CLI verbs --
name-resolved, provider-agnostic siblings of the path-based `ninja cab
<file> <name>` command, backed by `shinobi.cabs`.
"""

from click.testing import CliRunner
from pydantic import BaseModel

from shinobi import cli
from shinobi.exceptions import CabLoadError
from shinobi.steps.schema import Cab


class Inputs(BaseModel):
    pass


class Outputs(BaseModel):
    pass


def test_cabs_list_groups_output_by_provider(monkeypatch):
    import shinobi.cabs as cabs_module

    monkeypatch.setattr(cabs_module, "list_cabs", lambda: {"dosho": ["cubical", "wsclean"]})
    result = CliRunner().invoke(cli.main, ["cabs", "list"])
    assert result.exit_code == 0
    assert "dosho:" in result.output
    assert "cubical" in result.output
    assert "wsclean" in result.output


def test_cabs_list_errors_cleanly_when_nothing_installed(monkeypatch):
    import shinobi.cabs as cabs_module

    monkeypatch.setattr(cabs_module, "list_cabs", lambda: {})
    result = CliRunner().invoke(cli.main, ["cabs", "list"])
    assert result.exit_code != 0
    assert "no shinobi.cabs providers installed" in result.output


def test_cabs_show_prints_cab_schema_json(monkeypatch):
    import shinobi.cabs as cabs_module

    cab = Cab(name="wsclean", command="wsclean", inputs_model=Inputs, outputs_model=Outputs)

    def _get(name: str) -> Cab:
        if name != "wsclean":
            raise KeyError(name)
        return cab

    monkeypatch.setattr(cabs_module, "get", _get)
    result = CliRunner().invoke(cli.main, ["cabs", "show", "wsclean"])
    assert result.exit_code == 0
    assert '"name": "wsclean"' in result.output


def test_cabs_show_unknown_cab_errors_cleanly(monkeypatch):
    import shinobi.cabs as cabs_module

    def _raise(name):
        raise CabLoadError(f"no such cab {name!r} in any shinobi.cabs provider (none installed)")

    monkeypatch.setattr(cabs_module, "get", _raise)
    result = CliRunner().invoke(cli.main, ["cabs", "show", "unknown"])
    assert result.exit_code != 0
    assert "no such cab" in result.output
