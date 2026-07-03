"""Turn a cab's schema + user-supplied parameter values into a command line."""

from __future__ import annotations

from typing import Any

from shinobi.exceptions import ParameterError, UnsupportedFlavourError
from shinobi.schema import CabDef

# Flavours whose `command` is a real executable name, safe to hand to
# subprocess as argv[0]. Everything else (cult-cargo's "python-code",
# "casa-task", etc.) has a `command` that's inline source or a dotted
# reference to a function -- not something to run, let alone eval()/exec().
_EXECUTABLE_FLAVOURS = {"binary"}


def resolve_params(cab: CabDef, params: dict[str, Any]) -> dict[str, Any]:
    """Merge user-supplied params with implicit values and defaults, and
    check that all required inputs are present.
    """
    resolved: dict[str, Any] = {}
    for name, schema in cab.inputs.items():
        if schema.implicit is not None:
            if name in params:
                raise ParameterError(
                    f"{cab.name}: '{name}' is implicit and cannot be set by the caller"
                )
            resolved[name] = schema.implicit
        elif name in params:
            resolved[name] = params[name]
        elif schema.default is not None:
            resolved[name] = schema.default
        elif schema.required:
            raise ParameterError(f"{cab.name}: missing required parameter '{name}'")

    unknown = set(params) - set(cab.inputs)
    if unknown:
        raise ParameterError(f"{cab.name}: unknown parameter(s) {sorted(unknown)}")

    return resolved


def _format_value(value: Any, policies) -> str | None:
    if isinstance(value, bool):
        raise TypeError("bool values are handled by the caller, not _format_value")
    if isinstance(value, (list, tuple)):
        return policies.list_sep.join(str(v) for v in value)
    return str(value)


def build_argv(cab: CabDef, resolved: dict[str, Any]) -> list[str]:
    """Build a full argv (starting with the cab's command) from an already
    resolve_params()-ed parameter dict, according to the cab's policies.
    """
    if cab.flavour not in _EXECUTABLE_FLAVOURS:
        raise UnsupportedFlavourError(
            f"cab '{cab.name}' has flavour '{cab.flavour}', which shinobi doesn't "
            f"execute (only {sorted(_EXECUTABLE_FLAVOURS)} today) -- its `command` "
            f"is not an executable name and must not be run as one"
        )

    argv: list[str] = [cab.command]
    policies = cab.policies

    for name, schema in cab.inputs.items():
        if name not in resolved:
            continue
        value = resolved[name]
        if value is None:
            continue

        arg_name = policies.arg_name(cab.param_name(name, schema))

        if isinstance(value, bool):
            if value:
                argv.append(arg_name)
            continue

        if isinstance(value, (list, tuple)) and policies.repeat_list:
            for item in value:
                argv.append(arg_name)
                argv.append(str(item))
            continue

        argv.append(arg_name)
        argv.append(_format_value(value, policies))

    return argv


def build_args(cab: CabDef, params: dict[str, Any]) -> list[str]:
    """Convenience wrapper: resolve_params() + build_argv() in one call."""
    return build_argv(cab, resolve_params(cab, params))
