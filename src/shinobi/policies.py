"""Turn a cab's schema + resolved parameter values into a command line.

Operates on the step-model `Cab`: the parameter *values* come from an
already-validated `inputs_model` instance (or the prepared dict dispatch
builds from it), while per-field naming/implicit metadata comes from the
cab's `field_meta`, dynamically-named params from `input_patterns`, and
arg formatting from `policies`.
"""

from __future__ import annotations

from typing import Any

from shinobi.exceptions import UnsupportedFlavourError
from shinobi.steps.schema import Cab

# Flavours whose `command` is a real executable name, safe to hand to
# subprocess as argv[0]. Everything else (cult-cargo's "python-code",
# "casa-task", ...) has a `command` that is inline source or a dotted
# function reference -- not something to run, let alone eval()/exec().
_EXECUTABLE_FLAVOURS = {"binary"}


def _format_value(value: Any, policies) -> str:
    if isinstance(value, (list, tuple)):
        if policies.repeat == "[]":
            return "[" + ",".join(str(v) for v in value) + "]"
        return policies.list_sep.join(str(v) for v in value)
    if isinstance(value, bool) and policies.key_value:
        return "true" if value else "false"
    return str(value)


def _emit_arg(argv: list[str], policies, arg_name: str, value: Any) -> None:
    if policies.key_value:
        # hydra-style single token, e.g. "input_ms.data_column=DATA" or
        # "solver.terms=[K,G]" -- never a bare flag, even for a bool.
        argv.append(f"{arg_name}={_format_value(value, policies)}")
        return

    if isinstance(value, bool):
        if value:
            argv.append(arg_name)
            if policies.explicit_true:
                argv.append("true")
        elif policies.explicit_false:
            argv.append(arg_name)
            argv.append("false")
        return

    if isinstance(value, (list, tuple)) and policies.repeat_list:
        for item in value:
            argv.append(arg_name)
            argv.append(str(item))
        return

    argv.append(arg_name)
    argv.append(_format_value(value, policies))


def build_argv(cab: Cab, resolved: dict[str, Any]) -> list[str]:
    """Build a full argv (starting with the cab's command) from a resolved
    parameter dict, according to the cab's policies and field metadata.

    Rejects any non-"binary" flavour before building argv -- so a
    non-executable `command` can never reach subprocess as argv[0] (see
    AGENTS.md, "Never eval()/exec() a cab's command").
    """
    if cab.flavour not in _EXECUTABLE_FLAVOURS:
        raise UnsupportedFlavourError(
            f"cab '{cab.name}' has flavour '{cab.flavour}', which shinobi doesn't "
            f"execute (only {sorted(_EXECUTABLE_FLAVOURS)} today) -- its `command` "
            f"is not an executable name and must not be run as one"
        )

    # A subcommand-style command (e.g. "simms telsim") is more than one
    # argv token -- split it so subprocess execs the real binary, not a
    # literal (and nonexistent) file named "simms telsim".
    argv: list[str] = cab.command.split()
    policies = cab.policies
    declared = set(cab.inputs_model.model_fields)
    positionals: list[str] = []

    for name in cab.inputs_model.model_fields:
        meta = cab.field_meta.get(name)
        if meta is not None and meta.implicit is not None:
            value: Any = meta.implicit
        elif name in resolved:
            value = resolved[name]
        else:
            continue
        if value is None:
            continue
        repeat_as_tokens = meta is not None and meta.repeat_as_tokens and isinstance(value, (list, tuple))
        if meta is not None and meta.positional:
            if repeat_as_tokens:
                positionals.extend(str(item) for item in value)
            else:
                positionals.append(_format_value(value, policies))
            continue
        if repeat_as_tokens:
            # One flag occurrence, then each item as its own bare token --
            # e.g. wsclean's "-size 4096 4096"/"-weight briggs 0", not
            # "-size 4096,4096" (one token, which the tool can't parse).
            argv.append(policies.arg_name(cab.param_name(name)))
            argv.extend(str(item) for item in value)
            continue
        _emit_arg(argv, policies, policies.arg_name(cab.param_name(name)), value)

    # pattern-matched (dynamically-named) params, e.g. K.type/G.type
    for name, value in resolved.items():
        if name in declared or value is None:
            continue
        meta = cab.match_pattern(name)
        if meta is None:
            continue
        arg = meta.nom_de_guerre or name
        _emit_arg(argv, policies, policies.arg_name(arg), value)

    # Positional args (e.g. simms' "ms") come last, in field-declaration
    # order -- matches how tools that mix flags with one positional arg
    # are actually invoked (flags first, bare value last).
    argv.extend(positionals)
    return argv
