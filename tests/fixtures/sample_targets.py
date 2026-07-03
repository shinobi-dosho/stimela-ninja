"""Tiny @cab and @recipe targets used by tests/test_cli.py, and as a
concrete example of the 'path/to/file.py:name' target syntax `ninja run`
expects.
"""

from shinobi.decorators import cab, recipe


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
