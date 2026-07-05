"""`@shinobi.pystep`: turn a plain, type-hinted Python function into a step
without hand-writing pydantic `inputs_model`/`outputs_model` classes.

`inputs_model` is derived from the function's own parameters (via
`inspect.signature` + `typing.get_type_hints` -- not `param.annotation`
directly, since every module in this codebase uses
`from __future__ import annotations`, making raw annotations lazy strings).
`outputs_model` is derived from its return-type annotation: a `BaseModel`
subclass is used directly (the function must return an instance of it); no
annotation or `-> None` means no outputs, and the function must return
`None`. Any other return annotation is rejected at decoration time -- there
is no auto-wrapping of a bare scalar/dict return into an invented field
name, since that would be exactly the kind of implicit magic this project
avoids elsewhere.

This builds a bare `Scope` (not a `Cab`, not a `Recipe`) and wraps the
function in an adapter that returns its own `StepResult` directly, never
calling `ctx.run()` -- see `Scope`/`StepRef`'s docstrings in `schema.py` for
why a bare `Scope` is a real, supported shape, not a special case bolted on
here. `@shinobi.step` (`decorator.py`), by contrast, never introspects the
decorated function's signature at all -- `scope.inputs_model` is the schema
authority there. Use `@shinobi.pystep` when you have a plain function and no
external tool; use `@shinobi.step` when you have an existing `Cab`/`Recipe`.

**Container execution**: when `image=` is set and a container backend is
resolved, the function runs inside the container instead of in-process.
The function's source module is mounted into the container via
`inspect.getfile()`, and a generated runner script handles invocation.
Native runs call the function in-process (the original behaviour).

For container-only imports (e.g. CASA tasks that don't exist on the host),
use `ctx.import_func()` to avoid linter warnings:

    @shinobi.pystep(image="quay.io/stimela/casa:latest")
    def flagdata(ctx, vis: Path, mode: str = "manual") -> FlagdataOutputs:
        flagdata_fn = ctx.import_func("flagdata", "casatasks")
        flagdata_fn(vis=str(vis), mode=mode)
        return FlagdataOutputs(...)

This is cleaner than `from casatasks import flagdata` which triggers linter
errors when the module isn't installed on the host.

Caveat: `typing.get_type_hints` resolves annotations against the function's
own module globals, so any `BaseModel` used in the signature or return type
must be defined at module level, not inside another function.

v1 always deep-copies every input before calling the function (the `Scope`
default, `Mutability.IMMUTABLE` for every field) -- there is no per-parameter
mutability override yet; add one if a real need surfaces.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, get_type_hints

from pydantic import BaseModel, create_model

from shinobi.results import StepResult
from shinobi.steps.schema import Scope, StepRef

if TYPE_CHECKING:
    from shinobi.steps.dispatch import ExecContext

_UNSUPPORTED_KINDS = (
    inspect.Parameter.VAR_POSITIONAL,
    inspect.Parameter.VAR_KEYWORD,
    inspect.Parameter.POSITIONAL_ONLY,
)


def _pascal(func_name: str) -> str:
    return "".join(word.capitalize() for word in func_name.split("_") if word)


def _inputs_model_from_signature(func: Callable) -> tuple[type[BaseModel], bool]:
    """Derive the inputs model from `func`'s signature.

    Returns `(inputs_model, wants_ctx)`. If the first parameter is named
    `ctx` it is treated as the execution-context injection point (matching
    `@shinobi.step`'s convention) rather than an input field: it is skipped
    when building the model and needs no type hint. The adapter then calls
    `func(ctx, **inputs)`.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.items())
    wants_ctx = bool(params) and params[0][0] == "ctx"
    if wants_ctx:
        params = params[1:]
    hints = get_type_hints(func)
    fields: dict[str, tuple[Any, Any]] = {}
    for pname, param in params:
        if param.kind in _UNSUPPORTED_KINDS:
            raise TypeError(
                f"pystep {func.__name__!r}: parameter {pname!r} is "
                f"{param.kind.description} -- only plain positional-or-keyword "
                "parameters (with a real type hint) are supported"
            )
        if pname not in hints:
            raise TypeError(
                f"pystep {func.__name__!r}: parameter {pname!r} has no type "
                "hint -- every parameter needs one so its inputs_model can be "
                "derived from the signature"
            )
        required = param.default is inspect.Parameter.empty
        fields[pname] = (hints[pname], ... if required else param.default)
    return create_model(f"{_pascal(func.__name__)}Inputs", **fields), wants_ctx


def _outputs_model_from_return(func: Callable) -> tuple[type[BaseModel], bool]:
    hints = get_type_hints(func)
    ret = hints.get("return")
    if ret is None or ret is type(None):
        return create_model(f"{_pascal(func.__name__)}Outputs"), True
    if isinstance(ret, type) and issubclass(ret, BaseModel):
        return ret, False
    raise TypeError(
        f"pystep {func.__name__!r}: return type {ret!r} isn't a BaseModel "
        "subclass (or None) -- declare a BaseModel and return an instance "
        "of it, rather than a bare scalar/dict/list, so outputs stay "
        "explicitly named and typed"
    )


# A minimal, dependency-free stand-in for ExecContext, injected into the
# runner when the function takes a leading `ctx`. Shinobi itself is not
# assumed to be installed inside the container, so we cannot import the real
# ExecContext; this shim provides just the container-relevant surface
# (`import_func`), mirroring ExecContext.import_func.
_CTX_SHIM = '''
class _Ctx:
    def import_func(self, func, module=None):
        if module is None:
            import builtins
            return getattr(builtins, func)
        import importlib
        return getattr(importlib.import_module(module), func)


ctx = _Ctx()
'''

# `inputs_path` and the script's own path are host paths that are
# identity-bind-mounted into the container (see build_container_argv), so the
# same absolute path is valid on both sides -- no fixed `/shinobi_io` mount.
_RUNNER_TEMPLATE = '''\
import json
import sys

sys.path.insert(0, {source_dir!r})

from {module_path} import {func_name}

with open({inputs_path!r}) as f:
    inputs = json.load(f)
{ctx_shim}
result = {func_name}({ctx_arg}**inputs)

if result is not None:
    if hasattr(result, "model_dump"):
        result = result.model_dump(mode="json")
    print(json.dumps(result))
'''


def _run_pystep_container(
    scope: Scope,
    func: Callable,
    outputs_model: type[BaseModel],
    is_empty: bool,
    wants_ctx: bool,
    ctx: ExecContext,
    backend_name: str,
) -> StepResult:
    """Execute a pystep's function inside a container.

    Mounts the function's source module directory and a temp directory
    containing a runner script and serialized inputs. The runner imports
    the function, calls it with the inputs, and prints JSON output. When
    the function takes a leading `ctx`, a minimal context shim is injected.
    """
    from shinobi.backends.container import build_container_argv

    module_path = func.__module__
    func_name = func.__qualname__

    # The runner imports `func` via its dotted module path, so the directory
    # put on sys.path must be the package *root*, not the file's own parent.
    # For a top-level module (no dots) that is the parent dir; for a package
    # module like `pkg.sub.mod` it is that many levels above the file.
    source_file = Path(inspect.getfile(func)).resolve()
    source_root = source_file.parent
    for _ in range(module_path.count(".")):
        source_root = source_root.parent
    source_dir = str(source_root)

    # Derive the container inputs the same way the in-process path does, so
    # the two never diverge: prepare_inputs() applies mutability handling,
    # and re-validating through the model yields a JSON-safe dump.
    prepared = ctx.prepare_inputs()
    inputs_json = ctx.scope.inputs_model(**prepared).model_dump(mode="json")

    with tempfile.TemporaryDirectory(prefix="shinobi_pystep_") as tmpdir:
        io_dir = Path(tmpdir) / "io"
        io_dir.mkdir()

        inputs_path = io_dir / "inputs.json"
        inputs_path.write_text(json.dumps(inputs_json))

        runner_path = io_dir / "runner.py"
        runner_path.write_text(
            _RUNNER_TEMPLATE.format(
                source_dir=source_dir,
                module_path=module_path,
                func_name=func_name,
                inputs_path=str(inputs_path),
                ctx_shim=_CTX_SHIM if wants_ctx else "",
                ctx_arg="ctx, " if wants_ctx else "",
            )
        )

        workdir = os.getcwd()
        extra_dirs = [str(io_dir), source_dir]

        inner_argv = ["python3", str(runner_path)]

        full_argv = build_container_argv(
            backend_name,
            ctx.scope,
            inner_argv,
            prepared,
            workdir,
            extra_dirs=extra_dirs,
            image_override=ctx.scope.image,
        )

        proc = subprocess.run(
            full_argv,
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            try:
                outputs: BaseModel = outputs_model()
            except Exception:
                outputs = outputs_model.model_construct()
            return StepResult(
                name=scope.name,
                returncode=proc.returncode,
                outputs=outputs,
                inputs=ctx.inputs,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )

        if is_empty:
            outputs: BaseModel = outputs_model()
        else:
            output_data = {}
            if proc.stdout.strip():
                try:
                    output_data = json.loads(proc.stdout.strip().splitlines()[-1])
                except json.JSONDecodeError:
                    pass
            try:
                outputs = outputs_model(**output_data)
            except Exception:
                outputs = outputs_model.model_construct(**output_data)

        return StepResult(
            name=scope.name,
            returncode=0,
            outputs=outputs,
            inputs=ctx.inputs,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )


def _make_adapter(
    func: Callable, outputs_model: type[BaseModel], is_empty: bool, wants_ctx: bool
) -> Callable[[ExecContext], StepResult]:
    def _adapter(ctx: ExecContext) -> StepResult:
        from shinobi.steps.dispatch import _CONTAINER_BACKENDS

        backend_name = ctx.resolve_backend_name()
        if ctx.scope.image and backend_name in _CONTAINER_BACKENDS:
            return _run_pystep_container(
                ctx.scope, func, outputs_model, is_empty, wants_ctx, ctx, backend_name
            )

        prepared = ctx.prepare_inputs()
        ret = func(ctx, **prepared) if wants_ctx else func(**prepared)
        if is_empty:
            if ret is not None:
                raise TypeError(
                    f"pystep {func.__name__!r} has no declared outputs (no return "
                    f"annotation, or -> None) but returned {type(ret).__name__!r} "
                    "instead of None"
                )
            outputs: BaseModel = outputs_model()
        else:
            if not isinstance(ret, outputs_model):
                raise TypeError(
                    f"pystep {func.__name__!r} must return {outputs_model.__name__!r}, "
                    f"got {type(ret).__name__!r}"
                )
            outputs = ret
        return StepResult(
            name=ctx.scope.name,
            returncode=0,
            outputs=outputs,
            inputs=ctx.inputs,
            stdout="",
            stderr="",
        )

    return _adapter


def pystep(
    *,
    name: str | None = None,
    info: str | None = None,
    image: str | None = None,
    backend: str | None = None,
    **params: Any,
) -> Callable[[Callable], StepRef]:
    """Decorate (or directly call on an existing function, matching
    `@shinobi.step`'s precedent: `pystep()(existing_func)`) a plain,
    type-hinted function to turn it into a `StepRef`. See the module
    docstring for the schema-derivation and outputs rules.

    `image` enables container execution: when set and a container backend
    is resolved, the function runs inside the specified container image
    instead of in-process. The function's source module is mounted into
    the container so it can be imported by the runner script.

    `backend` sets the default backend for this step (same as on any
    `Scope`). With `image`, this is typically a container backend name
    like ``"docker"`` or ``"apptainer"``.

    `**params` are per-call constants, same as `@shinobi.step`.
    """

    def decorator(func: Callable) -> StepRef:
        inputs_model, wants_ctx = _inputs_model_from_signature(func)
        outputs_model, is_empty = _outputs_model_from_return(func)
        adapter = _make_adapter(func, outputs_model, is_empty, wants_ctx)
        step_name = name or func.__name__
        scope = Scope(
            name=step_name,
            info=info if info is not None else inspect.getdoc(func),
            inputs_model=inputs_model,
            outputs_model=outputs_model,
            image=image,
            backend=backend,
        )
        return StepRef(name=step_name, step=scope, func=adapter, params=params)

    return decorator
