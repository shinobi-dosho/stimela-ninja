from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Result:
    """The outcome of running a cab: exit status, captured console output,
    and any values extracted by wranglers. Recipes pass these between steps
    directly as Python objects -- no string substitution involved.
    """

    cab_name: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    outputs: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def __getattr__(self, name: str) -> Any:
        try:
            return self.outputs[name]
        except KeyError:
            raise AttributeError(
                f"Result for '{self.cab_name}' has no output '{name}' "
                f"(available: {sorted(self.outputs)})"
            ) from None
