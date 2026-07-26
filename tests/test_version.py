from __future__ import annotations

import tomllib
from pathlib import Path

import shinobi

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _declared_version() -> str:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def test_version_matches_pyproject():
    """The exported version is the one being released.

    `shinobi.__version__` feeds `ninja version` and the `shinobi_version`
    field of every run manifest, so a stale value misattributes results to a
    release that did not produce them. It used to be a literal in
    `shinobi/__init__.py` and sat three releases behind; deriving it from the
    installed distribution fixes that, and this pins the derivation to the
    version actually declared for the build.
    """
    assert shinobi.__version__ == _declared_version()


def test_version_is_not_the_uninstalled_fallback():
    """Guard the `PackageNotFoundError` branch from passing for the wrong reason.

    If the test run ever imports `shinobi` off a bare source tree rather than
    the installed project, `__version__` degrades to the sentinel and the
    comparison above would be testing nothing.
    """
    assert shinobi.__version__ != "0+unknown"
