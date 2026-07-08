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
The runner is invoked as ``python3`` -- container images must provide it
(every official Python image does; a bare ``python`` would fail). Native
runs call the function in-process (the original behaviour).

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
import pickle
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, get_type_hints

from pydantic import BaseModel, create_model

from shinobi.backends._stream import run_streaming
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


def _ctx_shim() -> str:
    """A minimal, dependency-free stand-in for ExecContext, injected into
    the runner when the function takes a leading `ctx`. Shinobi itself is
    not assumed to be installed inside the container, so we cannot import
    the real ExecContext; instead the shim's `import_func` body is lifted
    verbatim from the real method with `inspect.getsource`, so the two
    cannot drift. The method body relies on the runner's module-level
    `importlib` import plus the `builtins` import added here; its
    annotations stay unevaluated thanks to the runner's
    `from __future__ import annotations`.
    """
    from shinobi.steps.dispatch import ExecContext

    return (
        "import builtins\n\n\nclass _Ctx:\n"
        + inspect.getsource(ExecContext.import_func)
        + "\n\nctx = _Ctx()\n"
    )


# All paths in the runner (`inputs_path`, `outputs_path`, the script's own
# path) are host paths that are identity-bind-mounted into the container
# (see build_container_argv), so the same absolute path is valid on both
# sides -- no fixed `/shinobi_io` mount. Inputs travel as a pickle so
# pydantic-coerced values (Path, datetime, ...) arrive in the container as
# the same types the in-process path passes; the result is written to the
# outputs file rather than stdout, so the function is free to print.
_RUNNER_TEMPLATE = '''\
from __future__ import annotations

import importlib
import json
import pickle
import sys

sys.path.insert(0, {source_dir!r})

_obj = importlib.import_module({module_path!r})
for _part in {qualname_parts!r}:
    _obj = getattr(_obj, _part)
{func_name} = _obj

with open({inputs_path!r}, "rb") as f:
    inputs = pickle.load(f)
{ctx_shim}
result = {func_name}({ctx_arg}**inputs)

if result is not None and hasattr(result, "model_dump"):
    result = result.model_dump(mode="json")
with open({outputs_path!r}, "w") as f:
    json.dump(result, f)
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
    containing a runner script, pickled inputs, and the outputs file. The
    runner imports the function, calls it with the same objects the
    in-process path would pass, and writes the JSON result to the outputs
    file. When the function takes a leading `ctx`, a context shim is
    injected.
    """
    from shinobi.backends.container import build_container_argv

    if "<locals>" in func.__qualname__:
        raise TypeError(
            f"pystep {func.__name__!r}: a function defined inside another "
            "function has no importable module path, so it cannot run in a "
            "container"
        )

    module_path = func.__module__
    source_file = Path(inspect.getfile(func)).resolve()
    # A function defined in a directly-run script has __module__ ==
    # '__main__', which inside the container would name the runner itself;
    # import it by its file name instead. The runner inserts source_dir at
    # sys.path[0], so this file shadows any same-named module elsewhere on
    # the path -- two __main__ scripts with the same stem in different dirs
    # never collide because each pystep run mounts only its own source_dir.
    if module_path == "__main__":
        module_path = source_file.stem

    # The runner imports `func` via its dotted module path, so the directory
    # put on sys.path must be the package *root*, not the file's own parent.
    # For a top-level module (no dots) that is the parent dir; for a package
    # module like `pkg.sub.mod` it is that many levels above the file.
    source_root = source_file.parent
    for _ in range(module_path.count(".")):
        source_root = source_root.parent
    source_dir = str(source_root)

    # Same objects the in-process path passes: prepare_inputs() applies
    # mutability handling on top of pydantic-coerced values. They travel by
    # pickle (protocol 4, not 5, so containers on Python 3.4-3.7 can
    # unpickle them -- protocol 5 requires 3.8+) so e.g. Path-typed inputs
    # stay Paths inside the container.
    prepared = ctx.prepare_inputs()

    with tempfile.TemporaryDirectory(prefix="shinobi_pystep_") as tmpdir:
        io_dir = Path(tmpdir) / "io"
        io_dir.mkdir()

        inputs_path = io_dir / "inputs.pkl"
        inputs_path.write_bytes(pickle.dumps(prepared, protocol=4))

        outputs_path = io_dir / "outputs.json"

        runner_path = io_dir / "runner.py"
        runner_path.write_text(
            _RUNNER_TEMPLATE.format(
                source_dir=source_dir,
                module_path=module_path,
                qualname_parts=func.__qualname__.split("."),
                func_name=func.__name__,
                inputs_path=str(inputs_path),
                outputs_path=str(outputs_path),
                ctx_shim=_ctx_shim() if wants_ctx else "",
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
        )

        run = run_streaming(full_argv, label=ctx._cache_path or scope.name, stream=ctx._stream)

        if run.returncode != 0:
            # Best-effort outputs on the failure path: try a full
            # construction (fills defaults), fall back to model_construct
            # (skips validation -- required fields stay unset). Both are
            # acceptable because success=False; the point is to carry
            # stdout/stderr out, not to report valid outputs.
            try:
                outputs: BaseModel = outputs_model()
            except Exception:
                outputs = outputs_model.model_construct()
            return StepResult(
                name=scope.name,
                returncode=run.returncode,
                outputs=outputs,
                inputs=ctx.inputs,
                stdout=run.stdout,
                stderr=run.stderr,
            )

        # Exit 0 means the runner ran to completion, and it always writes
        # the outputs file -- so a missing/unreadable one is a broken
        # contract. Fail loudly (mirroring the TypeErrors the in-process
        # adapter raises) rather than fabricating outputs.
        try:
            output_data = json.loads(outputs_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise TypeError(
                f"pystep {func.__name__!r}: container run exited 0 but left "
                f"no readable outputs file ({exc})"
            ) from exc

        if is_empty:
            if output_data is not None:
                raise TypeError(
                    f"pystep {func.__name__!r} has no declared outputs (no "
                    "return annotation, or -> None) but returned "
                    f"{type(output_data).__name__!r} instead of None"
                )
            outputs = outputs_model()
        else:
            if not isinstance(output_data, dict):
                raise TypeError(
                    f"pystep {func.__name__!r} must return "
                    f"{outputs_model.__name__!r}, got "
                    f"{type(output_data).__name__!r} from the container"
                )
            outputs = outputs_model(**output_data)

        return StepResult(
            name=scope.name,
            returncode=0,
            outputs=outputs,
            inputs=ctx.inputs,
            stdout=run.stdout,
            stderr=run.stderr,
        )


def _make_adapter(
    func: Callable, outputs_model: type[BaseModel], is_empty: bool, wants_ctx: bool
) -> Callable[[ExecContext], StepResult]:
    def _adapter(ctx: ExecContext) -> StepResult:
        # Check the cheap field first: resolving the backend name can fall
        # through to a config-file read, which image-less pysteps (the
        # common case) should never pay on every call.
        if ctx.scope.image:
            from shinobi.backends.container import CONTAINER_RUNTIMES

            backend_name = ctx.resolve_backend_name()
            if backend_name in CONTAINER_RUNTIMES:
                return _run_pystep_container(
                    ctx.scope,
                    func,
                    outputs_model,
                    is_empty,
                    wants_ctx,
                    ctx,
                    backend_name,
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
        # `adapter` is a generic closure defined once in this module --
        # every pystep's adapter has identical source text. Anything that
        # wants the *actual* decorated function (e.g. `shinobi.cache`'s
        # cache-key identity, which hashes a pystep's own source so
        # editing its implementation invalidates cached results) needs
        # this standard `__wrapped__` pointer to see past the adapter.
        adapter.__wrapped__ = func
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
