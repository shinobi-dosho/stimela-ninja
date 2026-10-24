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

**Out-of-process execution**: when `image=` is set and a container backend is
resolved, the function runs inside the container instead of in-process; when
`venv=` is set and the `venv` backend is resolved, it runs under that venv's
interpreter (see `shinobi.backends.venv`). Either way the function's source
*file* is mounted/read via `inspect.getfile()`, and a generated runner script
loads it as an isolated module and invokes it. The runner never imports the
framework stack (shinobi/pydantic). For a container it also stubs the target's
*own* package (so the container's Python need not be ABI-compatible with the
host's compiled wheels); for a venv it does *not* -- importing the venv's real
packages is the point. The host rebuilds the typed outputs from the child's
JSON. Native runs call the function in-process (the original behaviour).

For container-only imports (e.g. CASA tasks that don't exist on the host),
use `ctx.import_callable()` to avoid linter warnings:

    @shinobi.pystep(image="quay.io/stimela/casa:latest")
    def flagdata(ctx, vis: Path, mode: str = "manual") -> FlagdataOutputs:
        flagdata_fn = ctx.import_callable("flagdata", "casatasks")
        flagdata_fn(vis=str(vis), mode=mode)
        return FlagdataOutputs(...)

This is cleaner than `from casatasks import flagdata` which triggers linter
errors when the module isn't installed on the host. It is named for what a
pystep actually asks for -- a class (`casacore.tables.table`) and a numpy
`ufunc` are as common here as a plain function -- and it rejects a
non-callable, which is how a dotted path split at the wrong point announces
itself. `ctx.import_func()` is the old name, kept as an alias.
`ctx.import_module()` is the companion for when the body wants a module
rather than one of its attributes (`np = ctx.import_module("numpy")`).

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
import logging
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from pydantic import BaseModel

from shinobi.backends._stream import TeardownIncomplete, display_label, run_streaming
from shinobi.config import AppConfig
from shinobi.dataset_access import DatasetAccess
from shinobi.fits_access import FITSAccess
from shinobi.exceptions import BackendError, CabRunError, ParameterError
from shinobi.results import StepResult, explain_returncode
from shinobi.sandbox import (
    ProductCapture,
    absolutize_path_inputs,
    clear_stale_outputs,
    create_sandbox,
    discard_sandbox,
    harvest_outputs,
    prepare_output_parents,
    observe_prepared_parents,
    prune_unused_parents,
    relativize_path_outputs,
)
from shinobi.steps.schema import ParamMeta, Scope, StepRef, validate_declared_writes

if TYPE_CHECKING:
    from shinobi.steps.dispatch import ExecContext

logger = logging.getLogger(__name__)


def _inputs_model_from_signature(func, write_paths=()):
    try:
        from dosho.validation.pyfunc import inputs_model_from_signature
    except ImportError as exc:
        raise ImportError("This Shinobi source requires matching Dosho sources with neutral function schema support") from exc
    return inputs_model_from_signature(func, write_paths, meta_class=ParamMeta)


def _outputs_model_from_return(func):
    from dosho.validation.pyfunc import outputs_model_from_return

    return outputs_model_from_return(func)


def _ctx_shim() -> str:
    """A minimal, dependency-free stand-in for ExecContext, injected into
    the runner when the function takes a leading `ctx`. Shinobi itself is
    not assumed to be installed inside the container, so we cannot import
    the real ExecContext; instead the shim's import methods are lifted
    verbatim from the real ones with `inspect.getsource`, so the two cannot
    drift. Their bodies rely on the runner's module-level `importlib` import
    plus the `builtins` and `ModuleType` imports added here; their
    annotations stay unevaluated thanks to the runner's `from __future__
    import annotations`.

    All three import methods are lifted, `import_callable` included even
    though `import_func` is only its alias -- an alias whose target is
    missing is an `AttributeError` inside the image, which is the one place
    it cannot be debugged.
    """
    from shinobi.steps.dispatch import ExecContext

    methods = "\n".join(inspect.getsource(m) for m in (ExecContext.import_callable, ExecContext.import_func, ExecContext.import_module))
    return "import builtins\nfrom types import ModuleType\n\n\nclass _Ctx:\n" + methods + "\nctx = _Ctx()\n"


# All paths in the runner (`inputs_path`, `outputs_path`, the script's own
# path, the target `source_file`) are host paths that are identity-bind-mounted
# into the container (see build_container_argv), so the same absolute path is
# valid on both sides -- no fixed `/shinobi_io` mount. Inputs travel as a
# pickle so pydantic-coerced values (Path, datetime, ...) arrive in the
# container as the same types the in-process path passes; the result is written
# to the outputs file rather than stdout, so the function is free to print.
#
# The target module is loaded from its file directly (not imported by its
# dotted package path), so its package `__init__` never runs and the host
# site-packages is never placed on the container's `sys.path` -- the host's
# compiled wheels (e.g. `pydantic_core`, built for the host's cpXY) cannot be
# loaded by the container's own (possibly different) Python otherwise. Imports
# of whatever prefixes are in `_STUB_PREFIXES` (always the framework packages
# `shinobi`/`pydantic`; for a container run, the target's own top-level package
# too) are intercepted by a stub finder installed at the front of
# `sys.meta_path`, so none of them -- nor their compiled deps -- ever load
# in-child. This honours the ctx-shim's "shinobi is not assumed installed
# in the container" design: the function is *defined* against dependency-free
# stubs and *runs* using only stdlib plus whatever it pulls in via
# `ctx.import_callable`/`ctx.import_module` (real, container-provided packages
# like `casatasks`). The
# function returns a stubbed `*Outputs` (kwargs stored as attrs); its plain
# dict is written out and the host -- which has real pydantic -- rebuilds the
# typed, validated model from it (see `_run_pystep_container`).
_RUNNER_TEMPLATE = '''\
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import json
import pickle
import sys
import types


class _Any:
    """Permissive stand-in: every attribute access and call yields another
    _Any, so a stubbed module's arbitrary members resolve to something
    harmless at import time (they are never used for real work in-container)."""

    def __getattr__(self, name):
        return _ANY

    def __call__(self, *args, **kwargs):
        return _ANY


_ANY = _Any()


class _StubModel:
    """Dependency-free stand-in for a pydantic BaseModel: stores keyword
    arguments as attributes and dumps them back as a plain dict. The host
    (which has real pydantic) rebuilds the typed *Outputs model from that
    dict, so no compiled pydantic ever loads inside the container."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self, *args, **kwargs):
        return dict(self.__dict__)


def _stub_attr(name):
    if name.startswith("__") and name.endswith("__"):
        # Dunders must raise, exactly as a real module does for ones it does
        # not define. Returning _ANY for `__file__` in particular poisons
        # `inspect.getmodule`, which scans sys.modules guarded only by
        # `hasattr(module, "__file__")` and then calls os.path.splitext on
        # whatever it finds -- so ANY container-side library that introspects
        # the module table dies on `expected str, bytes or os.PathLike
        # object, not _Any`. astropy does this at import time, which takes
        # out every simms and casacore-adjacent pystep.
        raise AttributeError(name)
    if name == "BaseModel":
        return _StubModel
    if name == "pystep":
        # `@shinobi.pystep(...)` used as a decorator: return a factory whose
        # decorator leaves the function unchanged, so the module-level name
        # stays the plain function rather than a StepRef.
        return lambda *a, **k: (lambda func: func)
    return _ANY


_STUB_PREFIXES = {stub_prefixes!r}


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []  # mark as a package so `from x.y import z` resolves
        mod.__getattr__ = _stub_attr
        return mod

    def exec_module(self, module):
        pass


class _StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _STUB_PREFIXES:
            return importlib.machinery.ModuleSpec(fullname, _StubLoader())
        return None


sys.meta_path.insert(0, _StubFinder())

_SOURCE_ROOT = {source_root!r}
if _SOURCE_ROOT is not None:
    sys.path.insert(0, _SOURCE_ROOT)

_spec = importlib.util.spec_from_file_location("_shinobi_pystep_target", {source_file!r})
_module = importlib.util.module_from_spec(_spec)
sys.modules["_shinobi_pystep_target"] = _module
_spec.loader.exec_module(_module)

_obj = _module
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
    json.dump(result, f, default=str)
'''


@dataclass
class _Launch:
    """How to launch a pystep runner out-of-process, plus the provenance the
    resulting `StepResult` should carry. `argv` is the full command; `env` is
    the process environment (`None` inherits, e.g. for a container runtime
    whose isolation comes from the runtime, not the env); `cwd` is the working
    directory to run in (`None` = process cwd -- containers set their workdir
    via a runtime flag instead, venvs need the sandbox dir here); `provenance`
    is splatted into `StepResult` (the container path fills image fields, the
    venv path fills venv fields).
    """

    argv: list[str]
    env: dict[str, str] | None
    cwd: str | None
    provenance: dict[str, Any] = field(default_factory=dict)
    # Runtime-specific teardown for a container ninja's signals cannot reach
    # (docker/podman); None everywhere else. See `_stream.run_streaming`.
    stop: Callable[[], None] | None = None


class _ContainerLauncher:
    """Runs the pystep runner inside a container runtime. Stubs the target's
    own package (its ABI-locked wheels can't load against a possibly-mismatched
    in-container interpreter)."""

    stub_target_package = True

    def __init__(self, backend_name: str, ctx: ExecContext):
        self.backend_name = backend_name
        self.ctx = ctx

    def build(self, runner_path: Path, workdir: str, extra_dirs: list[str], run_prepared: dict[str, Any]) -> _Launch:
        from shinobi.backends.container import build_container_argv, container_stopper, new_container_name

        container_name = new_container_name()
        dataset_plan = self.ctx.dataset_backend_plan(self.backend_name, run_prepared)
        full_argv, image_digest = build_container_argv(
            self.backend_name,
            self.ctx.scope,
            ["python3", str(runner_path)],
            run_prepared,
            workdir,
            extra_dirs=extra_dirs,
            dataset_mounts=[(str(mount.source), mount.writable) for mount in dataset_plan.mounts] if dataset_plan is not None else None,
            run_as_host_user=(self.ctx._config or AppConfig.load()).backend.run_as_host_user,
            pin=self.ctx._pin,
            container_name=container_name,
        )
        return _Launch(
            argv=full_argv,
            env=None,
            cwd=None,  # docker gets its workdir via --workdir, runs in host cwd
            provenance={"image": self.ctx.scope.image, "image_digest": image_digest, "containerized": True},
            # Ignored for the apptainer-likes (they need no handle); the
            # difference is decided inside `container_stopper`, not here.
            stop=container_stopper(self.backend_name, container_name),
        )

    def failure_hint(self, stderr: str) -> str | None:
        """Translate a cgroup-delegation failure, if that is what this was
        (see `backends.container.cgroup_failure_hint`)."""
        from shinobi.backends.container import cgroup_failure_hint

        return cgroup_failure_hint(stderr, self.backend_name)


class _VenvLauncher:
    """Runs the pystep runner with a virtualenv's own interpreter. Does *not*
    stub the target's own package -- importing the venv's real packages is the
    whole point (see `backends.venv`)."""

    stub_target_package = False

    def __init__(self, venv: Path, backend_name: str, ctx: ExecContext):
        self.venv = venv
        self.backend_name = backend_name
        self.ctx = ctx

    def build(self, runner_path: Path, workdir: str, extra_dirs: list[str], run_prepared: dict[str, Any]) -> _Launch:
        from shinobi.backends.venv import venv_digest, venv_env

        return _Launch(
            argv=[str(self.venv / "bin" / "python"), str(runner_path)],
            env=venv_env(self.venv),
            cwd=workdir,  # subprocess has no --workdir; sandboxing depends on this
            provenance={"venv": str(self.venv), "venv_digest": venv_digest(self.venv) if self.ctx._pin else None},
        )


def _run_pystep_subprocess(
    scope: Scope,
    func: Callable,
    outputs_model: type[BaseModel],
    is_empty: bool,
    wants_ctx: bool,
    ctx: ExecContext,
    launcher: "_ContainerLauncher | _VenvLauncher",
) -> StepResult:
    """Execute a pystep's function out-of-process, in a container or a venv.

    Writes a temp directory with a runner script, pickled inputs, and the
    outputs file. The runner loads the function's source file as an isolated
    module (always stubbing the framework packages; the target's own package
    too for containers, but not for venvs -- see `launcher.stub_target_package`),
    calls the function with the same objects the in-process path would pass,
    and writes the JSON result to the outputs file. The host rebuilds the typed
    outputs model from that JSON. When the function takes a leading `ctx`, a
    context shim is injected. Only the launch (interpreter, env, cwd, mounts)
    and the recorded provenance differ between the two backends.
    """
    if "<locals>" in func.__qualname__:
        raise TypeError(f"pystep {func.__name__!r}: a function defined inside another function has no importable module path, so it cannot run out-of-process")

    source_file = Path(inspect.getfile(func)).resolve()
    # Offload workers attach the import root of the immutable staged source
    # tree to the reconstructed callable. The ordinary local path leaves
    # this unset and retains the historical isolated-module/stub behaviour.
    staged_code_root_value = getattr(func, "__shinobi_staged_code_root__", None)
    staged_code_root = Path(staged_code_root_value).resolve() if staged_code_root_value else None

    # The runner loads this file as an isolated module, stubbing the framework
    # packages (shinobi, pydantic) always, and the target's own top-level
    # package only for containers (see _RUNNER_TEMPLATE and the launcher). A
    # function defined in a directly-run script has __module__ == '__main__'
    # and no importable package.
    stub_prefixes = {"shinobi", "pydantic"}
    if func.__module__.split(".")[0] != "dosho":
        stub_prefixes.add("dosho")
    if launcher.stub_target_package and staged_code_root is None and func.__module__ != "__main__":
        stub_prefixes.add(func.__module__.split(".")[0])

    # Same objects the in-process path passes: prepare_inputs() applies
    # mutability handling on top of pydantic-coerced values. They travel by
    # pickle (protocol 4, not 5, so containers on Python 3.4-3.7 can
    # unpickle them -- protocol 5 requires 3.8+) so e.g. Path-typed inputs
    # stay Paths in the child interpreter.
    prepared = ctx.prepare_inputs()

    # Sandboxed run (shinobi.sandbox): the child's workdir is a private scratch
    # dir, with path-typed inputs anchored back at the workspace -- the
    # containerized-cab counterpart in `dispatch._run_cab` documents the
    # scheme. `prepared` (the caller's original values) is kept for harvest.
    workspace = os.getcwd()
    sandbox_dir: Path | None = None
    precreated: list[Path] = []
    run_prepared = prepared
    dataset_resources = ctx.workflow_dataset_resources()
    planned = ctx._planned_inputs
    run_planned = planned
    if dataset_resources or any(meta.family for meta in scope.field_meta.values()):
        from shinobi.ownership import validate_contained_execution

        validate_contained_execution(scope, prepared, workspace=Path(workspace), dataset_resources=dataset_resources, planned_inputs=planned)
    if ctx._sandbox_root is not None:
        sandbox_dir = create_sandbox(ctx._sandbox_root, ctx._cache_path or scope.name, **({"dataset_resources": dataset_resources} if dataset_resources else {}))
        run_prepared = absolutize_path_inputs(scope, prepared, Path(workspace))
        if planned is not None:
            run_planned = planned.with_inputs(absolutize_path_inputs(scope, planned.inputs, Path(workspace)), workspace=sandbox_dir)
    # Same pre-run replacement a cab gets (`dispatch._run_cab`): a declared
    # output the child writes straight to its destination must not still hold
    # the last run's product when the function starts.
    run_cwd = sandbox_dir if sandbox_dir is not None else Path(workspace)
    if dataset_resources or any(meta.family for meta in scope.field_meta.values()):
        from shinobi.ownership import validate_contained_execution

        validate_contained_execution(scope, run_prepared, workspace=run_cwd, dataset_workspace=Path(workspace), dataset_resources=dataset_resources, planned_inputs=run_planned)
    validate_declared_writes(scope, run_prepared, run_cwd, error_type=BackendError if isinstance(launcher, _ContainerLauncher) else ParameterError)
    if ctx._clear_outputs:
        clear_stale_outputs(scope, run_prepared, Path(workspace), sandboxed=sandbox_dir is not None)
    created = prepare_output_parents(scope, run_prepared, run_cwd)
    prepared_observations = observe_prepared_parents(created)
    precreated = [path for path in created if sandbox_dir is not None and path.resolve().is_relative_to(sandbox_dir)]

    capture = (
        ProductCapture(scope, run_prepared, Path(workspace), sandbox_dir, prepared_observations=prepared_observations)
        if getattr(ctx, "_product_captures", None) is not None
        else None
    )

    # Not `with TemporaryDirectory(...)`: on an interrupt whose child could
    # not be confirmed stopped, this directory must **stay**. It holds the
    # runner and the pickled inputs, and it is bind-mounted into the
    # container -- deleting it under a process still using it does not stop
    # that process, it just makes what it produces incomplete, which is the
    # exact failure this whole path exists to prevent (a real interrupted run
    # lost a 13 GB measurement set that way). `TeardownIncomplete` is the one
    # exception that skips the cleanup; everything else, including an ordinary
    # KeyboardInterrupt that *did* stop the child, cleans up as before.
    tmpdir = tempfile.mkdtemp(prefix="shinobi_pystep_")
    keep_tmpdir = False
    try:
        io_dir = Path(tmpdir) / "io"
        io_dir.mkdir()

        inputs_path = io_dir / "inputs.pkl"
        inputs_path.write_bytes(pickle.dumps(run_prepared, protocol=4))

        outputs_path = io_dir / "outputs.json"

        runner_path = io_dir / "runner.py"
        runner_path.write_text(
            _RUNNER_TEMPLATE.format(
                stub_prefixes=stub_prefixes,
                source_root=str(staged_code_root) if staged_code_root is not None else None,
                source_file=str(source_file),
                qualname_parts=func.__qualname__.split("."),
                func_name=func.__name__,
                inputs_path=str(inputs_path),
                outputs_path=str(outputs_path),
                ctx_shim=_ctx_shim() if wants_ctx else "",
                ctx_arg="ctx, " if wants_ctx else "",
            )
        )

        workdir = str(sandbox_dir) if sandbox_dir is not None else workspace
        # A normal local pystep needs only its entry directory. An offloaded
        # pystep needs the complete frozen import root: its captured helpers
        # are deliberately part of the executable identity.
        extra_dirs = [str(io_dir), str(staged_code_root or source_file.parent)]

        launch = launcher.build(runner_path, workdir, extra_dirs, run_prepared)

        run = run_streaming(
            launch.argv, label=display_label(ctx._cache_path) if ctx._cache_path else scope.name, stream=ctx._stream, cwd=launch.cwd, env=launch.env, stop=launch.stop
        )

        if run.returncode != 0:
            stderr_tail = (run.stderr or "").strip()
            detail = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
            # A containerized pystep hits the same cgroup-delegation wall a cab
            # does, and gets the same unreadable message; explain it here too.
            explain = getattr(launcher, "failure_hint", None)
            hint = explain(run.stderr or "") if explain else None
            if hint:
                detail = f"\n{hint}{detail}"
            if sandbox_dir is not None:
                raise CabRunError(f"pystep '{scope.name}' failed (returncode {explain_returncode(run.returncode)}); its sandbox is kept for post-mortem at {sandbox_dir}{detail}")
            raise CabRunError(f"pystep '{scope.name}' failed (returncode {explain_returncode(run.returncode)}){detail}")

        # Exit 0 means the runner ran to completion, and it always writes
        # the outputs file -- so a missing/unreadable one is a broken
        # contract. Fail loudly (mirroring the TypeErrors the in-process
        # adapter raises) rather than fabricating outputs.
        try:
            output_data = json.loads(outputs_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise TypeError(f"pystep {func.__name__!r}: subprocess run exited 0 but left no readable outputs file ({exc})") from exc

        if is_empty:
            if output_data is not None:
                raise TypeError(f"pystep {func.__name__!r} has no declared outputs (no return annotation, or -> None) but returned {type(output_data).__name__!r} instead of None")
            outputs = outputs_model()
        else:
            if not isinstance(output_data, dict):
                raise TypeError(f"pystep {func.__name__!r} must return {outputs_model.__name__!r}, got {type(output_data).__name__!r} from the subprocess")
            outputs = outputs_model(**output_data)

        capture_outputs = outputs
        if sandbox_dir is not None:
            outputs = relativize_path_outputs(scope, outputs, Path(workspace))
            prune_unused_parents(precreated, prepared_observations)
            harvest_outputs(scope, outputs, run_prepared, sandbox_dir, Path(workspace), _capture=capture)
            discard_sandbox(sandbox_dir)

        result = StepResult(
            name=scope.name,
            returncode=0,
            outputs=outputs,
            inputs=ctx.inputs,
            stdout=run.stdout,
            stderr=run.stderr,
            kind="pyfunc",
            backend=launcher.backend_name,
            sandboxed=sandbox_dir is not None,
            sandbox_path=str(sandbox_dir) if sandbox_dir is not None and sandbox_dir.exists() else None,
            resources=scope.resources,
            **launch.provenance,
        )
        if capture is not None:
            ctx._product_captures[id(result)] = (result, capture.finish(capture_outputs))
        return result
    except TeardownIncomplete:
        keep_tmpdir = True
        logger.error(
            "pystep '%s': keeping %s -- the process using it could not be stopped, so removing it would only hide a writer that is still running.",
            scope.name,
            tmpdir,
        )
        raise
    finally:
        if not keep_tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


@dataclass(frozen=True)
class PystepCallable:
    """A declared Python computation, distinct from an orchestration function.

    ``__wrapped__`` preserves the cache's source identity. The explicit type
    lets a compiler recognize pysteps without guessing from a closure name
    or accepting every function carrying a ``__wrapped__`` attribute.
    """

    __wrapped__: Callable
    _run: Callable[[ExecContext], StepResult]
    is_empty: bool
    wants_ctx: bool

    @property
    def __name__(self) -> str:
        return self._run.__name__

    def __call__(self, ctx: ExecContext) -> StepResult:
        return self._run(ctx)


def _make_adapter(func: Callable, outputs_model: type[BaseModel], is_empty: bool, wants_ctx: bool) -> PystepCallable:
    def _adapter(ctx: ExecContext) -> StepResult:
        from shinobi.backends import registered_backend_names
        from shinobi.steps.dispatch import get_step_backend

        backend_name = ctx.resolve_backend_name()
        # Reject removed or misspelled backends before falling through to a
        # local Python call. The existing resolver also accepts instance
        # overrides; registered classes need no eager construction here.
        if backend_name not in registered_backend_names():
            get_step_backend(backend_name)
        # An explicit venv backend honours backend.venv.default even when
        # the scope names no virtualenv of its own.
        if ctx.scope.image or ctx.scope.venv or ctx.scope.backend == "venv":
            from shinobi.backends.container import CONTAINER_RUNTIMES

            # The resolved backend name decides, so a scope carrying both
            # `image` and `venv` is well-defined.
            if backend_name in CONTAINER_RUNTIMES and ctx.scope.image:
                return _run_pystep_subprocess(ctx.scope, func, outputs_model, is_empty, wants_ctx, ctx, _ContainerLauncher(backend_name, ctx))
            if backend_name == "venv":
                from shinobi.backends.venv import resolve_venv

                venv = resolve_venv(ctx.scope.venv)  # raises if declared-but-missing
                if venv is not None:
                    return _run_pystep_subprocess(ctx.scope, func, outputs_model, is_empty, wants_ctx, ctx, _VenvLauncher(venv, backend_name, ctx))
                # No venv declared anywhere: fall through to in-process, warning
                # (selecting `venv` is an opt-in to isolation, so a silent no-op
                # would surprise -- matches the cab backend's native fallback).
                import warnings

                warnings.warn(
                    f"pystep '{ctx.scope.name}' selected the venv backend but no venv is declared (on the step or via backend.venv.default) -- running in-process",
                    stacklevel=2,
                )

        prepared = ctx.prepare_inputs()
        # In-process: no sandbox (os.chdir is process-global, see the module
        # docstring), so every declared output is written straight to its
        # destination and a re-run needs the previous product cleared.
        if ctx._clear_outputs:
            clear_stale_outputs(ctx.scope, prepared, Path.cwd(), sandboxed=False)
        capture = ProductCapture(ctx.scope, prepared, Path.cwd()) if getattr(ctx, "_product_captures", None) is not None else None
        ret = func(ctx, **prepared) if wants_ctx else func(**prepared)
        if is_empty:
            if ret is not None:
                raise TypeError(f"pystep {func.__name__!r} has no declared outputs (no return annotation, or -> None) but returned {type(ret).__name__!r} instead of None")
            outputs: BaseModel = outputs_model()
        else:
            if not isinstance(ret, outputs_model):
                raise TypeError(f"pystep {func.__name__!r} must return {outputs_model.__name__!r}, got {type(ret).__name__!r}")
            outputs = ret
        result = StepResult(
            name=ctx.scope.name,
            returncode=0,
            outputs=outputs,
            inputs=ctx.inputs,
            stdout="",
            stderr="",
            kind="pyfunc",  # ran in-process; no container -> backend/image left None
        )
        if capture is not None:
            ctx._product_captures[id(result)] = (result, capture.finish(outputs))
        return result

    return PystepCallable(func, _adapter, is_empty, wants_ctx)


def pystep(
    *,
    name: str | None = None,
    info: str | None = None,
    image: str | None = None,
    venv: str | None = None,
    backend: str | None = None,
    cache: bool | None = None,
    cache_dir: str | None = None,
    sandbox: bool | None = None,
    harvest: list[str] | None = None,
    write_paths: Sequence[str] | None = None,
    dataset_accesses: Sequence[DatasetAccess] | None = None,
    fits_accesses: Sequence[FITSAccess] | None = None,
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

    `venv` enables virtualenv execution: when set (a path or a key into
    `backend.venv.envs`) and the `venv` backend is resolved, the function
    runs under that venv's own interpreter, importing the venv's real
    packages (unlike the container path, the target's own package is *not*
    stubbed). A scope may carry both `image` and `venv`; the resolved
    backend name decides which is used.

    `backend` sets the default backend for this step (same as on any
    `Scope`). With `image`, this is typically a container backend name
    like ``"docker"`` or ``"apptainer"``; with `venv`, ``"venv"``.

    `cache`/`cache_dir` set this step's cache policy, with the same precedence
    as the corresponding fields on any `Scope`. In particular,
    ``cache=False`` lets a function whose products depend on run-time choices
    opt out when one static output schema cannot describe them safely.

    `sandbox`/`harvest` opt this step into sandboxed execution and declare
    extra keep-globs (see `shinobi.sandbox` and the fields on `Scope`).
    Sandboxing applies to the out-of-process paths (container and venv) --
    an in-process run ignores it (`os.chdir` is process-global, and recipes
    run steps on a thread pool), executing in the caller's cwd as always.

    `write_paths` names the parameters that are *destinations this function
    creates* rather than data it reads (`ParamMeta.write_path`) -- the one
    thing no signature can express, since `outputvis: Path` returned as the
    output and `vis: Path` rewritten in place are the same annotation. Only
    a named parameter has its stale product cleared before a re-run
    (`sandbox.clear_stale_outputs`); everything else is treated as the
    caller's data and left alone.

    `dataset_accesses` attaches the same serializable MSv2 access contracts a
    binary `Cab` carries on `Scope`; it does not inspect a path while the
    function is decorated.

    `**params` are per-call constants, same as `@shinobi.step`.
    """

    def decorator(func: Callable) -> StepRef:
        """Turn `func` into a `StepRef` with a schema derived from its signature.

        Args:
            func: A type-hinted plain function (or a function taking `ctx`
                as its first parameter).

        Returns:
            A `StepRef` wrapping `func` behind a generated adapter.
        """
        inputs_model, wants_ctx, input_meta = _inputs_model_from_signature(func, write_paths or ())
        outputs_model, is_empty = _outputs_model_from_return(func)
        unknown = sorted(set(write_paths or ()) - set(inputs_model.model_fields))
        if unknown:
            raise TypeError(f"pystep {func.__name__!r}: write_paths names {unknown}, which {'is' if len(unknown) == 1 else 'are'} not a parameter of the function")
        # `**params` accepts any name, so a *keyword this decorator does not
        # have* lands there silently instead of failing -- which is how a cab
        # written against a newer shinobi (`write_paths=[...]`) installs
        # against an older one and quietly declares nothing. A per-call
        # constant only means something for a parameter the function has, so
        # anything else is a mistake worth naming at decoration time.
        stray = sorted(set(params) - set(inputs_model.model_fields))
        if stray:
            raise TypeError(
                f"pystep {func.__name__!r}: {stray} {'is' if len(stray) == 1 else 'are'} neither a parameter of the "
                f"function nor an option of @shinobi.pystep -- a per-call constant has to name a parameter "
                f"(and a keyword this version does not know would otherwise be swallowed here silently)"
            )
        adapter = _make_adapter(func, outputs_model, is_empty, wants_ctx)
        # `adapter` is a generic closure defined once in this module --
        # every pystep's adapter has identical source text. Anything that
        # wants the *actual* decorated function (e.g. `shinobi.cache`'s
        # cache-key identity, which hashes a pystep's own source so
        # editing its implementation invalidates cached results) needs
        # this standard `__wrapped__` pointer to see past the adapter.
        step_name = name or func.__name__
        scope = Scope(
            name=step_name,
            info=info if info is not None else inspect.getdoc(func),
            inputs_model=inputs_model,
            outputs_model=outputs_model,
            image=image,
            venv=venv,
            backend=backend,
            cache=cache,
            cache_dir=cache_dir,
            sandbox=sandbox,
            harvest=harvest or [],
            field_meta=dict(input_meta),
            dataset_accesses=list(dataset_accesses or ()),
            fits_accesses=list(fits_accesses or ()),
        )
        return StepRef(name=step_name, step=scope, func=adapter, params=params)

    return decorator
