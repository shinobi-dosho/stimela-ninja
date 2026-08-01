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


# --------------------------------------------------------------------------
# Document-shaped providers (`get_document`)
# --------------------------------------------------------------------------

_YAML_CAB_DOC = """\
cabs:
  breizorro:
    command: breizorro
    inputs:
      threshold:
        dtype: float
        default: 6.5
    outputs:
      outfile:
        dtype: File
"""

_CLASSIC_DOC = """\
{"task": "flagms", "binary": "flagms", "base": "stimela/flagms",
 "parameters": [{"name": "msname", "dtype": "str", "required": true, "info": "MS"}]}
"""


class FakeDocumentProvider:
    """The shape this protocol exists for: a provider that hands back text
    and never imports shinobi to build anything.
    """

    def __init__(self, docs: dict[str, tuple[str, str]]):
        self._docs = docs

    def get_document(self, name: str) -> tuple[str, str]:
        return self._docs[name]

    def list_cabs(self) -> list[str]:
        return list(self._docs)


def test_build_document_yaml_cab_selects_by_name():
    cab = cabs.build_document("yaml_cab", _YAML_CAB_DOC, "breizorro")
    assert cab.name == "breizorro"
    assert cab.command == "breizorro"
    assert "threshold" in cab.inputs_model.model_fields


def test_build_document_yaml_cab_without_a_name_when_unambiguous():
    """A single-cab document needs no name -- the common case for a
    provider that ships one file per cab."""
    assert cabs.build_document("yaml_cab", _YAML_CAB_DOC).name == "breizorro"


def test_build_document_stimela_classic():
    cab = cabs.build_document("stimela-classic", _CLASSIC_DOC, "flagms")
    assert cab.name == "flagms"
    assert "msname" in cab.inputs_model.model_fields


def test_build_document_rejects_an_unknown_dialect():
    with pytest.raises(CabLoadError, match="unknown cab dialect 'toml-ish'"):
        cabs.build_document("toml-ish", "irrelevant")


def test_build_document_rejects_a_name_the_document_does_not_define():
    with pytest.raises(CabLoadError, match="not 'wsclean'"):
        cabs.build_document("yaml_cab", _YAML_CAB_DOC, "wsclean")


def test_get_resolves_through_a_document_provider(monkeypatch):
    module = FakeDocumentProvider({"breizorro": ("yaml_cab", _YAML_CAB_DOC)})
    _patch_entry_points(monkeypatch, [_fake_entry_point("datadosho", module)])
    cab = cabs.get("breizorro")
    assert isinstance(cab, Cab)
    assert cab.command == "breizorro"


def test_get_still_resolves_object_providers(monkeypatch):
    """The change is additive: a provider exposing only `get` is untouched."""
    module = FakeProviderModule(["wsclean"])
    _patch_entry_points(monkeypatch, [_fake_entry_point("dosho", module)])
    assert cabs.get("wsclean").name == "wsclean"


def test_document_and_object_providers_coexist(monkeypatch):
    """The split dosho is heading for: binary cabs as data, pysteps as code."""
    data = FakeDocumentProvider({"breizorro": ("yaml_cab", _YAML_CAB_DOC)})
    code = FakePystepProviderModule(["casa-listobs"])
    _patch_entry_points(
        monkeypatch,
        [_fake_entry_point("a-data", data), _fake_entry_point("b-code", code)],
    )
    assert isinstance(cabs.get("breizorro"), Cab)
    assert isinstance(cabs.get("casa-listobs"), StepRef)


class MixedProvider(FakeDocumentProvider):
    """One provider serving most cabs as data and the rest as objects --
    `get_document` is tried first, and a KeyError there falls through to
    `get` rather than to the next provider."""

    def get(self, name: str) -> StepRef:
        if name != "a-pystep":
            raise KeyError(name)
        return _make_pystep(name)

    def list_cabs(self) -> list[str]:
        return [*self._docs, "a-pystep"]


def test_get_document_miss_falls_through_to_get_on_the_same_provider(monkeypatch):
    module = MixedProvider({"breizorro": ("yaml_cab", _YAML_CAB_DOC)})
    _patch_entry_points(monkeypatch, [_fake_entry_point("mixed", module)])
    assert isinstance(cabs.get("breizorro"), Cab)
    assert isinstance(cabs.get("a-pystep"), StepRef)
    with pytest.raises(CabLoadError, match="no such cab 'nope'"):
        cabs.get("nope")
