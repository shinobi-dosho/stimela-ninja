"""Installed dosho provider contract, exercised by the cross-repository CI job."""

import pytest

from shinobi import cabs

dosho = pytest.importorskip("dosho")


def test_every_dosho_entry_resolves_through_the_provider():
    names = set(dosho.list_cabs())
    assert names
    assert names == set(cabs.list_cabs()["dosho"])

    for name in sorted(names):
        assert cabs.get(name).name == name
