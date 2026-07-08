"""Tests for `shinobi.cabs` -- resolving a `Cab` by name across installed
`shinobi.cabs`-entry-point providers (e.g. `dosho`), without shinobi ever
hand-importing a specific provider's modules.
"""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from shinobi import cabs
from shinobi.exceptions import CabLoadError
from shinobi.steps.schema import Cab


class Inputs(BaseModel):
    pass


class Outputs(BaseModel):
    pass


def _make_cab(name: str) -> Cab:
    return Cab(name=name, command=name, inputs_model=Inputs, outputs_model=Outputs)


class FakeProviderModule:
    def __init__(self, cab_names: list[str]):
        self._cabs = {name: _make_cab(name) for name in cab_names}

    def get(self, name: str) -> Cab:
        return self._cabs[name]

    def list_cabs(self) -> list[str]:
        return list(self._cabs)


def _fake_entry_point(provider_name: str, module: FakeProviderModule):
    return SimpleNamespace(name=provider_name, load=lambda: module)


def _patch_entry_points(monkeypatch, eps: list):
    monkeypatch.setattr(cabs, "entry_points", lambda group: eps)


@pytest.fixture
def single_provider(monkeypatch):
    module = FakeProviderModule(["wsclean", "cubical"])
    ep = _fake_entry_point("dosho", module)
    _patch_entry_points(monkeypatch, [ep])
    return module


def test_get_resolves_cab_from_the_installed_provider(single_provider):
    cab = cabs.get("wsclean")
    assert cab.name == "wsclean"


def test_get_raises_cab_load_error_for_unknown_cab(single_provider):
    with pytest.raises(CabLoadError, match="unknown-cab"):
        cabs.get("unknown-cab")


def test_get_raises_clear_error_when_no_providers_installed(monkeypatch):
    _patch_entry_points(monkeypatch, [])
    with pytest.raises(CabLoadError, match="none installed"):
        cabs.get("wsclean")


def test_list_cabs_groups_by_provider(single_provider):
    assert cabs.list_cabs() == {"dosho": ["cubical", "wsclean"]}


def test_get_tries_providers_in_name_order_first_match_wins(monkeypatch):
    a = FakeProviderModule(["shared"])
    b = FakeProviderModule(["shared"])
    ep_b = _fake_entry_point("b-provider", b)
    ep_a = _fake_entry_point("a-provider", a)
    # deliberately registered out of order -- resolver must sort by name
    _patch_entry_points(monkeypatch, [ep_b, ep_a])
    resolved = cabs.get("shared")
    assert resolved is a._cabs["shared"]


def test_get_falls_through_to_next_provider_if_first_lacks_the_cab(monkeypatch):
    a = FakeProviderModule(["only-in-a"])
    b = FakeProviderModule(["only-in-b"])
    ep_a = _fake_entry_point("a-provider", a)
    ep_b = _fake_entry_point("b-provider", b)
    _patch_entry_points(monkeypatch, [ep_a, ep_b])
    resolved = cabs.get("only-in-b")
    assert resolved is b._cabs["only-in-b"]
