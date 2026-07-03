import pytest

from shinobi.backends.native import NativeBackend


@pytest.fixture
def native():
    return NativeBackend()
