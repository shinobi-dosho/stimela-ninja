"""Immutable mixed-worker evidence. Monitoring walks records, never result trees.

References use invocation UUIDs and content hashes; logical addresses are identity,
not filesystem paths. Only explicit ``result(full=True)`` materializes descendants
and concatenated logs. Conversion of legacy records is deliberately offline/eager.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, ValidationError, model_validator

from shinobi.offload._codec import BundleError, ModelSpec, WireModel, unpack
from shinobi.offload.bundle import write_new_stream
from shinobi.offload.records import AttemptRecord

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
LEGACY_LIMIT = 8 * 1024 * 1024


class Artifact(WireModel):
    sha256: Digest
    size: int = Field(ge=0, strict=True)


class Reference(Artifact):
    owner: UUID
    address: tuple[str, ...] = ()


class Edge(WireModel):
    name: str
    reference: Reference


class LegacySource(Artifact):
    schema_version: Literal[6] = 6


class Record(AttemptRecord):
    schema_version: Literal[7] = 7
    address: tuple[str, ...] = ()
    schemas: tuple[Artifact, Artifact] | None = None
    edges: tuple[Edge, ...] = ()
    # A gather envelope owns its summary, but its one unit supplies descendants.
    transparent: bool = False
    stdout_artifact: Artifact | None = None
    stderr_artifact: Artifact | None = None
    stdout_children: bool = False
    stderr_children: bool = False
    manifest: Artifact | None = None
    legacy_source: LegacySource | None = None

    @model_validator(mode="after")
    def _closed_node(self):
        if self.children is not None or self.result_models is not None or self.stdout or self.stderr:
            raise BundleError("v7 records cannot embed children, schemas or logs")
        if self.observation is not None and (self.observation.steps or self.schemas is None):
            raise BundleError("v7 observation must be shallow and have typed schemas")
        if len({edge.name for edge in self.edges}) != len(self.edges):
            raise BundleError("duplicate child edge name")
        if self.transparent and len(self.edges) != 1:
            raise BundleError("transparent gather requires exactly one unit")
        for stream in ("stdout", "stderr"):
            if getattr(self, stream + "_children") and getattr(self, stream + "_artifact") is not None:
                raise BundleError("stream cannot contain both own and concatenated diagnostics")
        return self


class RootSummary(WireModel):
    """Compact dependency evidence retained after a root's payload is released."""

    workflow_id: UUID
    attempt_id: UUID
    step_path: str
    bundle_digest: str
    state: Literal["running", "succeeded", "failed", "cached", "skipped", "unknown"]
    source_attempts: dict[str, UUID] = Field(default_factory=dict)

    @property
    def committed(self):
        return self.state in ("succeeded", "cached", "skipped")


def digest_file(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
            size += len(data)
    return Artifact(sha256=digest.hexdigest(), size=size)


def check_file(path, artifact):
    if digest_file(path) != Artifact(sha256=artifact.sha256, size=artifact.size):
        raise BundleError(f"content digest or size mismatch: {path}")


def read_terminal(path, *, legacy_limit=None):
    """Prefer a bound migration sidecar; refuse large legacy files before parse."""
    sidecar = path.with_name("final.v7.json")
    if sidecar.exists():
        record = Record.model_validate_json(sidecar.read_bytes())
        if record.address:
            raise BundleError("terminal sidecar must have the root logical address")
        if record.legacy_source is None:
            raise BundleError("migration sidecar lacks original source binding")
        check_file(path, record.legacy_source)
        return record
    # Writers put schema_version first. Arbitrary legacy ordering is still
    # supported below the cap; oversized files cannot bypass it by a prefix.
    with path.open("rb") as stream:
        prefix = stream.read(4096)
    is_v7 = re.match(rb'\s*\{\s*"schema_version"\s*:\s*7\s*[,}]', prefix) is not None
    if legacy_limit is not None and path.stat().st_size > legacy_limit and not is_v7:
        raise BundleError("oversized legacy result; run python -m shinobi.offload.mixed convert-records --submission PATH on a provisioned host")
    data = path.read_bytes()
    version = json.loads(data).get("schema_version")
    record = (Record if version == 7 else AttemptRecord).model_validate_json(data)
    if isinstance(record, Record) and record.address:
        raise BundleError("terminal record must have the root logical address")
    return record


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self._schemas = OrderedDict()
        self.roots = {}
        self._verified = {}
        self._identities = {}
        self._active = set()
        self._artifacts = set()

    def artifact_path(self, kind, artifact):
        return self.directory / ("schemas" if kind == "schema" else "artifacts/logs") / (artifact.sha256 + (".json" if kind == "schema" else ""))

    def put_bytes(self, kind, data):
        artifact = Artifact(sha256=hashlib.sha256(data).hexdigest(), size=len(data))
        path = self.artifact_path(kind, artifact)
        try:
            write_new_stream(path, lambda stream: stream.write(data))
        except FileExistsError:
            check_file(path, artifact)
        return artifact

    def schema(self, artifact):
        key = (artifact.sha256, artifact.size)
        if key not in self._schemas:
            path = self.artifact_path("schema", artifact)
            data = path.read_bytes()
            self._check_bytes(data, artifact)
            self._schemas[key] = ModelSpec.model_validate_json(data).restore()
            if len(self._schemas) > 16:
                self._schemas.popitem(last=False)
        self._schemas.move_to_end(key)
        return self._schemas[key]

    @staticmethod
    def _check_bytes(data, artifact):
        if len(data) != artifact.size or hashlib.sha256(data).hexdigest() != artifact.sha256:
            raise BundleError("content digest or size mismatch")

    def node_path(self, reference):
        root = self.directory / "results" / str(reference.owner)
        if not reference.address:
            sidecar = root / "final.v7.json"
            return sidecar if sidecar.exists() else root / "final.json"
        return root / "nodes" / (reference.sha256 + ".json")

    def read_one(self, reference):
        path = self.node_path(reference)
        data = path.read_bytes()
        self._check_bytes(data, reference)
        record = Record.model_validate_json(data)
        if (record.attempt_id, record.address) != (reference.owner, reference.address):
            raise BundleError("referenced node has wrong owner or logical address")
        if record.legacy_source is not None:
            check_file(path.with_name("final.json"), record.legacy_source)
        return record

    def root_reference(self, invocation):
        path = self.directory / "results" / str(invocation) / "final.json"
        sidecar = path.with_name("final.v7.json")
        artifact = digest_file(sidecar if sidecar.exists() else path)
        return Reference(owner=invocation, **artifact.model_dump())

    def verify(self, record, *, reference=None):
        """Validate each unique reachable node once, retaining compact identities."""
        identity = (record.attempt_id, record.address)
        signature = (reference.sha256, reference.size) if reference else (hashlib.sha256(record.model_dump_json().encode()).hexdigest(), len(record.model_dump_json().encode()))
        if identity in self._active:
            raise BundleError("cycle in result evidence")
        prior = self._identities.setdefault(identity, signature)
        if prior != signature:
            # A root loaded directly and later by reference has different
            # serialization whitespace. Register roots with their file refs.
            raise BundleError("contradictory references to one logical node")
        if identity in self._verified:
            return
        self._active.add(identity)
        try:
            if record.observation is not None:
                for artifact, values in zip(record.schemas, (record.observation.inputs, record.observation.outputs), strict=True):
                    model = self.schema(artifact)
                    self.typed_values(model, values, partial=not record.committed)
            for stream in ("stdout", "stderr"):
                artifact = getattr(record, stream + "_artifact")
                if artifact is not None and (artifact.sha256, artifact.size) not in self._artifacts:
                    check_file(self.artifact_path("log", artifact), artifact)
                    self._artifacts.add((artifact.sha256, artifact.size))
            if record.manifest is not None:
                check_file(self.directory / "results" / str(record.attempt_id) / "manifest.json", record.manifest)
            for edge in record.edges:
                if edge.reference.owner == record.attempt_id:
                    if edge.reference.address != (*record.address, edge.name):
                        raise BundleError("local child reference has wrong logical address")
                elif edge.reference.address:
                    raise BundleError("cross-invocation references must select unit roots")
                key = (edge.reference.owner, edge.reference.address)
                if key in self._verified:
                    if self._identities[key] != (edge.reference.sha256, edge.reference.size):
                        raise BundleError("contradictory references to one logical node")
                    if record.committed and not self._verified[key]:
                        raise BundleError("successful composite contains unsuccessful child")
                    continue
                child = self.read_one(edge.reference)
                if (child.workflow_id, child.bundle_digest) != (record.workflow_id, record.bundle_digest):
                    raise BundleError("child belongs to another workflow or bundle")
                if record.committed and not child.committed:
                    raise BundleError("successful composite contains unsuccessful child")
                self.verify(child, reference=edge.reference)
            self._verified[identity] = record.committed
            if not record.address:
                self.roots[record.attempt_id] = RootSummary.model_validate(record.model_dump(include=set(RootSummary.model_fields)))
        finally:
            self._active.remove(identity)

    @staticmethod
    def typed_values(model, values, *, partial=False):
        decoded = {key: unpack(value) for key, value in values.items()}
        try:
            return model.model_validate(decoded)
        except ValidationError as exc:
            if not partial or any(error["type"] != "missing" for error in exc.errors()):
                raise
            return model.model_construct(**decoded)

    def result(self, record, scope, *, full=False):
        """Restore own typed I/O; full reconstruction is explicitly eager."""
        scope = scope.model_copy(update={"inputs_model": self.schema(record.schemas[0]), "outputs_model": self.schema(record.schemas[1])})
        result = AttemptRecord.result(record, scope)
        if full:
            children = {edge.name: self.result(self.read_one(edge.reference), scope, full=True) for edge in record.edges}
            result.sub_results = next(iter(children.values())).sub_results if record.transparent else children or None
            for stream in ("stdout", "stderr"):
                artifact = getattr(record, stream + "_artifact")
                value = (
                    self.artifact_path("log", artifact).read_bytes().decode("utf-8")
                    if artifact
                    else "\n".join(getattr(child, stream) for child in children.values() if getattr(child, stream))
                    if getattr(record, stream + "_children")
                    else ""
                )
                setattr(result, stream, value)
        return result

    def capture(self, result, *, address=(), edges=None, transparent=False, **identity):
        """Publish local descendants bottom-up; return the unpublished own node."""
        own = AttemptRecord.from_result(result, composite=True, recursive=False, **identity)
        children = result.sub_results or {}
        if edges is None:
            edges = []
            for name, child in children.items():
                child_identity = {**identity, "step_path": name, "dataset_lifecycle": None}
                node = self.capture(child, address=(*address, name), **child_identity)
                edges.append(Edge(name=name, reference=self.put_node(node)))
        values = own.model_dump(exclude={"schema_version", "children", "result_models", "stdout", "stderr"})
        schemas = tuple(self.put_bytes("schema", model.model_dump_json().encode()) for model in own.result_models)
        for stream in ("stdout", "stderr"):
            text = getattr(result, stream)
            joined = "\n".join(getattr(child, stream) for child in children.values() if getattr(child, stream))
            concatenate = bool(edges) and (text == joined or transparent)
            values[stream + "_children"] = concatenate
            values[stream + "_artifact"] = self.put_bytes("log", text.encode()) if text and not concatenate else None
        return Record(**values, address=address, schemas=schemas, edges=tuple(edges), transparent=transparent)

    def put_node(self, record):
        data = record.model_dump_json().encode()
        reference = Reference(owner=record.attempt_id, address=record.address, sha256=hashlib.sha256(data).hexdigest(), size=len(data))
        if not record.address:
            raise BundleError("terminal records require explicit publication")
        path = self.node_path(reference)
        try:
            write_new_stream(path, lambda stream: stream.write(data))
        except FileExistsError:
            check_file(path, reference)
        return reference

    def manifest(self, path, record):
        """Stream the ordinary provenance tree without accumulating descendants."""
        from datetime import datetime, timezone
        from shinobi import __version__
        from shinobi.provenance import _jsonable

        pinned = True

        def write(stream):
            def emit(value):
                stream.write(json.dumps(value, allow_nan=False, separators=(",", ":")).encode())

            def node(current, name=None):
                nonlocal pinned
                observation = current.observation
                if observation is None:
                    return
                pinned = pinned and (not observation.containerized or observation.image_digest is not None) and observation.venv is None
                values = observation.model_dump(mode="json", exclude={"steps", "inputs", "outputs"})
                if name is not None:
                    values["name"] = name
                values["inputs"] = _jsonable(self.typed_values(self.schema(current.schemas[0]), observation.inputs, partial=not current.committed))
                values["outputs"] = _jsonable(self.typed_values(self.schema(current.schemas[1]), observation.outputs, partial=not current.committed))
                data = json.dumps(values, allow_nan=False, separators=(",", ":")).encode()
                stream.write(data[:-1] + b',"steps":[')
                edges = current.edges
                if current.transparent:
                    edges = self.read_one(edges[0].reference).edges
                first = True
                for edge in edges:
                    child = self.read_one(edge.reference)
                    if child.observation is None:
                        continue
                    if not first:
                        stream.write(b",")
                    node(child, edge.name)
                    first = False
                stream.write(b"]}")

            header = {
                "schema_version": 1,
                "shinobi_version": __version__,
                "stimela_version": None,
                "cab_repo_commit": None,
                "target": None,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "backend": "slurm",
                "returncode": 0 if record.committed else 1,
            }
            stream.write(json.dumps(header, separators=(",", ":")).encode()[:-1] + b',"root":')
            node(record)
            stream.write(b',"pinned":')
            emit(pinned)
            stream.write(b"}")

        write_new_stream(path, write)
        return digest_file(path)

    def convert(self, old, source):
        """Convert an already parsed v6 tree without reconstructing StepResults."""

        def node(record, address):
            edges = tuple(Edge(name=name, reference=self.put_node(node(child, (*address, name)))) for name, child in record.children.items())
            values = record.model_dump(exclude={"schema_version", "children", "result_models", "stdout", "stderr", "observation"})
            observation = record.observation.model_copy(update={"steps": []}) if record.observation else None
            schemas = tuple(self.put_bytes("schema", model.model_dump_json().encode()) for model in record.result_models)
            for stream in ("stdout", "stderr"):
                text = getattr(record, stream)
                joined = "\n".join(getattr(child, stream) for child in record.children.values() if getattr(child, stream))
                concatenate = bool(edges) and text == joined
                values[stream + "_children"] = concatenate
                values[stream + "_artifact"] = self.put_bytes("log", text.encode()) if text and not concatenate else None
            return Record(**values, address=address, observation=observation, schemas=schemas, edges=edges)

        return node(old, ()).model_copy(update={"legacy_source": source})
