"""Experimental M1 plan protocol; no scheduler submission or worker execution.

A bundle is a frozen declaration, not evidence that any step ran. Reading
one reconstructs only framework schemas, never imports its captured code.
Paths remain relative to the recorded shared workspace, not the bundle dir.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import JsonValue, ValidationError, model_validator

from shinobi import __version__
from shinobi.backends.venv import resolve_venv
from shinobi.config import AppConfig
from shinobi.graph import check_offloadable
from shinobi.offload._codec import BundleError, ModelSpec, WireModel, pack, unpack
from shinobi.offload.code import CodeBundle, capture_code
from shinobi.storage import sync_directory_chain as _sync_directory_chain
from shinobi.steps.dispatch import _prepare_inputs
from shinobi.steps.pyfunc import PystepCallable
from shinobi.steps.schema import Cab, InputRef, LoopIteration, OutputRef, Recipe, Scope, StepRef


def fingerprint(model: WireModel) -> str:
    """Canonical content identity, independent of dictionary insertion order."""
    data = json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def write_new(path: Path, model: WireModel) -> Path:
    """Publish a complete, fsynced JSON file without replacing another record.

    The temporary inode and final link live on the same shared filesystem.
    Readers see no final file until its content is complete. A crash before
    publication leaves no committed record, never an inferred success.
    All parent entries are synced before returning, including directories
    created by this call or by submission staging. Sync failures propagate.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".record-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(model.model_dump_json(indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # atomic no-clobber publication
        _sync_directory_chain(path.parent)
    finally:
        os.unlink(temporary)
    return path


class ScopeSpec(WireModel):
    kind: Literal["cab", "pyfunc", "recipe"]
    inputs: ModelSpec
    outputs: ModelSpec
    settings: dict[str, JsonValue]

    @classmethod
    def capture(cls, scope: Scope) -> ScopeSpec:
        kinds = {Cab: "cab", Scope: "pyfunc", Recipe: "recipe"}
        if type(scope) not in kinds:
            raise BundleError(f"custom scope {type(scope).__name__!r} cannot be frozen")
        return cls(
            kind=kinds[type(scope)],
            inputs=ModelSpec.capture(scope.inputs_model),
            outputs=ModelSpec.capture(scope.outputs_model),
            settings={k: pack(v) for k, v in scope.model_dump(mode="python", exclude={"inputs_model", "outputs_model", "steps", "output_wiring", "max_workers"}).items()},
        )

    def restore(self) -> Scope:
        constructor = {"cab": Cab, "pyfunc": Scope, "recipe": Recipe}[self.kind]
        forbidden = {"inputs_model", "outputs_model", "steps", "output_wiring", "max_workers"}
        unknown = self.settings.keys() - (constructor.model_fields.keys() - forbidden)
        if unknown:
            raise BundleError(f"unsupported scope settings: {sorted(unknown)}")
        return constructor(inputs_model=self.inputs.restore(), outputs_model=self.outputs.restore(), **{k: unpack(v) for k, v in self.settings.items()})


class Binding(WireModel):
    """An explicit recipe-input or producing-step output address."""

    field: str
    step: str | None = None

    @classmethod
    def capture(cls, ref: InputRef | OutputRef) -> Binding:
        return cls(field=ref.field, step=ref.step if isinstance(ref, OutputRef) else None)

    def restore(self) -> InputRef | OutputRef:
        return InputRef(field=self.field) if self.step is None else OutputRef(step=self.step, field=self.field)


class FrozenStep(WireModel):
    name: str
    scope: ScopeSpec
    params: JsonValue
    wiring: dict[str, Binding | tuple[Binding, ...]]
    after: tuple[str, ...]
    loop: LoopIteration | None = None
    backend: Literal["native", "docker", "podman", "apptainer", "venv"]
    tool_venv: str | None = None
    image_digest: str | None = None
    code: CodeBundle | None = None
    pystep_is_empty: bool | None = None
    pystep_wants_ctx: bool | None = None

    def declaration(self) -> StepRef:
        """Reconstruct graph data only. A pystep here is NOT executable."""
        return StepRef(
            name=self.name,
            step=self.scope.restore(),
            params=unpack(self.params),
            wiring={k: [b.restore() for b in v] if isinstance(v, tuple) else v.restore() for k, v in self.wiring.items()},
            after=list(self.after),
            loop=self.loop,
        )


class RecipeBundle(WireModel):
    schema_version: Literal[1] = 1
    worker_protocol: Literal[1] = 1
    shinobi_version: str = __version__
    workspace: str
    recipe: ScopeSpec
    inputs: JsonValue
    config: dict[str, JsonValue]
    steps: tuple[FrozenStep, ...]
    output_wiring: dict[str, Binding]
    max_workers: int | None = None

    @model_validator(mode="after")
    def _validate_plan(self) -> RecipeBundle:
        if not Path(self.workspace).is_absolute() or self.recipe.kind != "recipe":
            raise BundleError("a bundle needs an absolute shared workspace and a recipe root")
        recipe = self.declaration()
        for value in self.config.values():
            unpack(value)  # also validate tagged/finite configuration on read
        # Do not apply legacy eligibility: declarations intentionally have
        # no live Python adapters, including for captured pysteps.
        from shinobi.graph import build_graph

        build_graph(recipe)
        _prepare_inputs(recipe, unpack(self.inputs))
        for step in self.steps:
            if (step.scope.kind == "pyfunc") != (step.code is not None):
                raise BundleError(f"step {step.name!r}: pystep code and scope kind disagree")
            if (step.code is not None) != (step.pystep_is_empty is not None and step.pystep_wants_ctx is not None):
                raise BundleError(f"step {step.name!r}: pystep adapter metadata and code disagree")
            if step.scope.kind == "recipe":
                raise BundleError("nested recipes are not supported by the worker bundle")
            if step.backend == "venv" and (not step.tool_venv or not Path(step.tool_venv).is_absolute()):
                raise BundleError(f"step {step.name!r}: venv execution needs a resolved shared tool environment")
            image = unpack(step.scope.settings["image"]) if "image" in step.scope.settings else None
            if step.image_digest is not None and (not image or step.backend not in ("docker", "podman", "apptainer") or not step.image_digest.startswith("sha256:")):
                raise BundleError(f"step {step.name!r}: image digest does not describe its container execution")
            if step.code is not None and step.backend != "venv" and (step.backend == "native" or not image):
                raise BundleError(f"step {step.name!r}: a pystep must execute in an image or venv, never in-process")
        return self

    def declaration(self) -> Recipe:
        recipe = self.recipe.restore()
        if not isinstance(recipe, Recipe):
            raise BundleError("bundle root is not a Recipe")
        recipe.steps = [step.declaration() for step in self.steps]
        recipe.output_wiring = {k: b.restore() for k, b in self.output_wiring.items()}
        recipe.max_workers = self.max_workers
        return recipe

    @property
    def digest(self) -> str:
        return fingerprint(self)

    @classmethod
    def read(cls, path: Path) -> RecipeBundle:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def stage(self, root: Path) -> Path:
        """Create a unique submission directory; never name it from user steps.

        No runtime or image is resolved here yet. Submission preparation
        must provision the matching worker and verify tool environments.
        """
        # Revalidate after any accidental edits to nested Python dictionaries.
        snapshot = type(self).model_validate_json(self.model_dump_json())
        digest = snapshot.digest  # validate the full identity before any writes
        root.mkdir(parents=True, exist_ok=True)
        workflow_id = uuid4()
        directory = root / str(workflow_id)
        directory.mkdir()
        for index, step in enumerate(snapshot.steps):
            if step.code is not None:
                step.code.write(directory / "code" / str(index))
        write_new(directory / "bundle.json", snapshot)
        write_new(directory / "submission.json", Submission(workflow_id=workflow_id, bundle_digest=digest))
        return directory


class Submission(WireModel):
    schema_version: Literal[1] = 1
    workflow_id: UUID
    bundle_digest: str
    worker_protocol: Literal[1] = 1
    worker_version: str = __version__


def freeze_recipe(
    recipe: Recipe, inputs: dict[str, Any], *, config: AppConfig, workspace: Path, code_roots: tuple[Path, ...] = (), include_modules: tuple[str, ...] = ()
) -> RecipeBundle:
    """Freeze declarations without writing files, running tools or resolving pins.

    Configuration is explicit so compilation never consults a new ambient
    config. All schemas are checked before input validation can run a user
    validator or default factory. Wired values are fully validated later,
    when their producers have committed; known inputs are validated here.
    """
    check_offloadable(recipe, worker=True)
    root = ScopeSpec.capture(recipe)
    scopes = [ScopeSpec.capture(ref.step) for ref in recipe.steps]
    prepared = _prepare_inputs(recipe, inputs)
    steps = []
    for ref, spec in zip(recipe.steps, scopes):
        backend = ref.step.backend or recipe.backend or config.backend.default
        if backend not in ("native", "docker", "podman", "apptainer", "venv"):
            raise BundleError(f"step {ref.name!r}: unsupported worker tool backend {backend!r}")
        venv = resolve_venv(ref.step.venv, config) if backend == "venv" else None
        if backend == "venv" and venv is None:
            raise BundleError(f"step {ref.name!r}: no tool venv declared; native fallback is forbidden")
        known = dict(ref.params)
        deferred = set()
        for name, source in ref.wiring.items():
            refs = source if isinstance(source, list) else [source]
            if any(isinstance(r, OutputRef) for r in refs):
                deferred.add(name)
                known.pop(name, None)
            else:
                values = [prepared[r.field] for r in refs]
                known[name] = values if isinstance(source, list) else values[0]
        try:
            ref.step.inputs_model(**known)
        except ValidationError as exc:
            # A not-yet-produced value may be missing, but no other input
            # failure is deferred. Aliases are intentionally reported by
            # their pydantic location, rather than guessed from names.
            if any(e["type"] != "missing" or len(e["loc"]) != 1 or e["loc"][0] not in deferred for e in exc.errors()):
                raise BundleError(f"step {ref.name!r}: invalid known inputs: {exc}") from exc
        code = capture_code(ref.func.__wrapped__, roots=code_roots, include=include_modules) if isinstance(ref.func, PystepCallable) else None
        pystep = ref.func if isinstance(ref.func, PystepCallable) else None
        steps.append(
            FrozenStep(
                name=ref.name,
                scope=spec,
                params=pack(ref.params),
                wiring={k: tuple(Binding.capture(b) for b in v) if isinstance(v, list) else Binding.capture(v) for k, v in ref.wiring.items()},
                after=tuple(ref.after),
                loop=ref.loop.model_copy(deep=True) if ref.loop else None,
                backend=backend,
                tool_venv=str(venv) if venv else None,
                code=code,
                pystep_is_empty=pystep.is_empty if pystep else None,
                pystep_wants_ctx=pystep.wants_ctx if pystep else None,
            )
        )
    return RecipeBundle(
        workspace=str(workspace.resolve()),
        recipe=root,
        inputs=pack(prepared),
        config={k: pack(v) for k, v in config.model_dump(mode="python").items()},
        steps=tuple(steps),
        output_wiring={k: Binding.capture(v) for k, v in recipe.output_wiring.items()},
        max_workers=recipe.max_workers,
    )
