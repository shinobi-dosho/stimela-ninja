"""Tests for `shinobi.cabs` -- resolving a `Cab` by name across installed
`shinobi.cabs`-entry-point providers (e.g. `dosho`), without shinobi ever
hand-importing a specific provider's modules.
"""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from shinobi import cabs
from shinobi.exceptions import CabLoadError
from shinobi.steps import pystep
from shinobi.steps.schema import Cab, StepRef


class Inputs(BaseModel):
    pass


class Outputs(BaseModel):
    pass


def _make_cab(name: str) -> Cab:
    return Cab(name=name, command=name, inputs_model=Inputs, outputs_model=Outputs)


def _make_pystep(name: str) -> StepRef:
    def _fn(x: int = 0) -> None:
        return None

    ref = pystep(name=name)(_fn)
    return ref


class FakeProviderModule:
    def __init__(self, cab_names: list[str]):
        self._cabs = {name: _make_cab(name) for name in cab_names}

    def get(self, name: str) -> Cab:
        return self._cabs[name]

    def list_cabs(self) -> list[str]:
        return list(self._cabs)


class FakePystepProviderModule:
    """A `shinobi.cabs` provider vending `StepRef`s (e.g. CASA-task
    pysteps) instead of `Cab`s -- the resolver must not care which shape
    a provider entry is.
    """

    def __init__(self, names: list[str]):
        self._entries = {name: _make_pystep(name) for name in names}

    def get(self, name: str) -> StepRef:
        return self._entries[name]

    def list_cabs(self) -> list[str]:
        return list(self._entries)


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


def test_get_resolves_a_stepref_backed_pystep_provider_entry(monkeypatch):
    module = FakePystepProviderModule(["listobs"])
    ep = _fake_entry_point("dosho", module)
    _patch_entry_points(monkeypatch, [ep])
    resolved = cabs.get("listobs")
    assert isinstance(resolved, StepRef)
    assert resolved.name == "listobs"


def test_list_cabs_works_across_mixed_cab_and_pystep_providers(monkeypatch):
    cab_provider = FakeProviderModule(["wsclean"])
    pystep_provider = FakePystepProviderModule(["listobs"])
    ep_cabs = _fake_entry_point("a-cabs", cab_provider)
    ep_psteps = _fake_entry_point("b-psteps", pystep_provider)
    _patch_entry_points(monkeypatch, [ep_cabs, ep_psteps])
    assert cabs.list_cabs() == {"a-cabs": ["wsclean"], "b-psteps": ["listobs"]}
