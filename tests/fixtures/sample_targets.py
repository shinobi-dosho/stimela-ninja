"""Tiny @cab and @recipe targets used by tests/test_cli.py, and as a
concrete example of the 'path/to/file.py:name' target syntax `ninja run`
expects.
"""

from shinobi.decorators import cab, recipe
from shinobi.schema import ParamSchema


@cab("/bin/echo")
def greet(text: str = "hi"):
    """Echo TEXT back."""


@cab("/bin/echo")
def greet_image(restored_image: str):
    """A cab with an underscored param, to check CLI flag round-tripping."""


@cab("/bin/false")
def fail():
    """A cab that always fails, for testing nonzero-exit propagation."""


@recipe()
def double(n: int) -> int:
    """Doubles n and prints it."""
    print(n * 2)
    return n * 2


@cab("/bin/echo", outputs={"path": ParamSchema(dtype="File")})
def make_file(name: str = "out.txt"):
    """Pretend to create a file, for --dryrun dependency-chain testing."""


@cab("/bin/echo")
def use_file(path: str):
    """Consume a path from a previous step -- used to exercise real
    dependency detection during --dryrun.
    """


@recipe()
def chained():
    """Calls make_file then use_file, threading the former's output into
    the latter -- so --dryrun should detect a real dependency edge.
    """
    from shinobi.backends import get_backend
    from shinobi.recipe import call

    backend = get_backend("native")
    result = call(make_file, backend)
    call(use_file, backend, path=result.path)
