CASA and MeasurementSet declarations
====================================

``CasaTable`` and ``MeasurementSetV2`` are explicit, versioned annotations
for dataset paths:

.. code-block:: python

   from pydantic import BaseModel

   from shinobi import MeasurementSetV2


   class Inputs(BaseModel):
       ms: MeasurementSetV2


They remain ordinary :class:`pathlib.Path` values.  Constructing or serializing
``Inputs`` does not touch the filesystem and does not import
``python-casacore``.  This is important for output paths which do not exist
yet, recipe loading on a login node, and remote execution.  Generated JSON
Schema carries the declaration in ``x-shinobi-dataset`` while the field value
still serializes as a path string.

The current named profiles are ``casa-table/v1`` and
``msv2-structural/v1``.  The profile name is part of the declaration, so a
future, stronger contract will not silently change what an existing annotation
means.  The older cab-loader dtype ``MS`` is unchanged and continues to map to
a plain ``Path``.

Explicit structural inspection
------------------------------

Inspection is a separate operation:

.. code-block:: python

   from shinobi.datasets import DatasetStatus, inspect_measurement_set_v2

   observed = inspect_measurement_set_v2("observation.ms")
   if observed.status is not DatasetStatus.VALID:
       raise RuntimeError(observed.message)

Install the optional inspector in the environment which performs inspection:

.. code-block:: console

   pip install "stimela-ninja[casacore]"

For a source checkout managed by uv, use
``uv sync --group dev --group measurement-set``.  The development group is
also the environment used for real-table lifecycle and recovery acceptance
tests; those tests must not silently skip during Measurement Set work.

The result is a serializable :class:`~shinobi.datasets.DatasetDescriptor`.
Statuses distinguish a missing path, a non-directory, an unavailable
inspector, an arbitrary directory/non-CASA table, a CASA table which is not an
MS, an incomplete MSv2, an unsupported version or metadata size, and a valid
dataset.  Diagnostics distinguish unreadable required subtables from readable
subtables whose required MSv2 columns are missing.  Optional subtables such as
``SOURCE`` and optional or custom columns remain legal.  The structural MSv2
profile also requires at least one primary visibility-data column: ``DATA``,
``FLOAT_DATA``, or ``LAG_DATA``.  Derived columns such as ``CORRECTED_DATA``
do not replace that collective requirement.

Inspection imports ``python-casacore`` lazily and requests only structural
metadata: column names, keyword names, row count, and ``MS_VERSION``.  Casacore
may materialize complete name lists before Shinobi can count them; Shinobi
retains and accepts metadata only within the configured limits, applying the
same limits to the main table and required subtables.  Inspection never reads
column cells or scans visibility data.  Within those limits, main-table names
are retained including optional and custom columns.

Dataset declarations nested in Pydantic models, sequences, mappings, and
unions are discovered recursively.  Diagnostic paths use ``[]`` for a
sequence item and ``.*`` for a mapping value.

Physical dataset closure
------------------------

After structural validation, :func:`~shinobi.dataset_closure.resolve_dataset_closure`
can produce a separate, serializable observation of the physical tables that
must travel together.  Its versioned ``msv2-dataset-closure/v1`` profile
contains the canonical main table plus the actual keyword-referenced mandatory
and supported optional subtables.  Symlink aliases and duplicate member
references collapse to one resource.  A subtable may be outside the MS
directory and shared by several MS roots, but every resource must remain in
the explicit storage namespace.

Resolution supports bounded, ordinary directory-backed tables using known
local storage managers.  It refuses reference tables, virtual concatenations,
opaque managers, escaped or cyclic references, and resources which change
while being observed, with distinct diagnostics.  The serializable result
records the managers, exact table-member files, and explicit requirements for
copy, mount, stage, materialize, and restore operations.  It contains
structural metadata only and retains no table handles.  Resolution rechecks
identities and metadata observations around its reads, but ordinary filesystem
inspection is not an atomic snapshot: a consumer must provide a cooperative
immutable or snapshot boundary and revalidate the observation before acting
on it.

Declarative task access
-----------------------

An atomic scope can state how a field is used with a serializable access
contract.  The same declaration works for binary cabs and Python steps:

.. code-block:: python

   from shinobi import Cab, DatasetAccess, DatasetColumns, DatasetMode

   flagger = Cab(
       name="flag",
       command="flag-tool",
       inputs_model=Inputs,
       outputs_model=Outputs,
       dataset_accesses=[
           DatasetAccess(
               field="ms",
               mode=DatasetMode.WRITE,
               columns=DatasetColumns(read=("DATA",), write=("FLAG",)),
           )
       ],
   )

Modes are ``read``, ``write`` and ``create``.  A declaration targets ``MAIN``
or one supported keyword-referenced MSv2 subtable and can record columns read,
written, created or removed.  Column creation/removal requires
``allow_schema_change=True``; row-count and keyword changes likewise require
their explicit ``allow_row_count_change`` and ``allow_keyword_change`` flags.
Selections are bounded data (half-open row ranges and finite standard ID
sets), not TaQL or executable expressions.

``field`` and ``root_field`` must name direct path/MS-compatible fields.
Containers, nested models and scalar fields are rejected at definition time;
scatter and nested dataset recipes have their separate bounded refusals below.
An explicit ``read`` declaration must also agree with the ordinary filesystem
schema.  A same-named input/output, ``MUTABLE`` path or ``write_path`` marker
on the same resolved dataset is a write and makes a contradictory ``read``
contract fail closed.

Resolution is explicit.  :func:`shinobi.dataset_access.plan_recipe_accesses`
links each declaration to the canonical root and every physical resource in
its :class:`~shinobi.dataset_closure.DatasetClosure`.  The serializable
:class:`~shinobi.dataset_access.ResolvedDatasetAccess` contains paths and
reasons only, never casacore handles.  Aliases therefore converge, a subtable
and its parent MS share one association, and two MS roots which reference one
external subtable conflict through that shared closure resource.

Unknown columns and omitted access metadata conservatively mean the whole
dataset.  The resolved record distinguishes ``unknown-columns``,
``undeclared`` and ``unknown-path`` fallbacks.  A path which is not statically
known must provide a ``reservation`` path envelope or planning refuses it.
An unresolved ``OutputRef`` remains unknown even when the consumer input has a
default; only its reservation may cover that interval.  A statically named
``create`` followed by a wired reader is planned through one provisional root
identity before the product exists.  ``create`` means a new dataset: an
existing target is refused unless the run names the step in ``--overwrite``
(see "Contained local mutation").
Column detail is currently validation and provenance only: it does **not**
permit concurrent writers, even when they name different columns.  Read/read
access may overlap; every write or create is ordered against all overlapping
access.  Inferred edges carry inspectable reasons such as
``write-after-read: observation.ms, MAIN.FLAG`` and never invent ``OutputRef``
lineage.

Resolved records validate their redundant public fields on deserialization:
the declaration, mode, known-path state, root/resources, column state and
fallback must agree.  Dataset read claims are retained by the shared ownership
registry as reads, so a separately planned writer conflicts with them even
though read/read workflows remain compatible.  The contained execution route
described below holds that claim across backend execution; unsupported routes
are refused and never dispatch an unclaimed reader.

The first planner is deliberately bounded: dataset-bearing scatter and nested
recipes are refused with a diagnostic rather than partially planned.  Flatten
those steps (including bounded loop expansions) so every access has one
declared graph node.

Contained read execution
------------------------

``contained-native-msv2-read/v1`` is the first executable strict-dataset
capability.  It accepts one ordinary, directory-backed MSv2 closure read by a
direct atomic ``Cab`` or ``@pystep``, or by atomic leaves in one flat recipe,
through a backend route proven by the separate ``msv2-backend-route/v1``
capability profile. Recipe boundary models may carry the same annotation;
access declarations still belong on their leaves.

Before dispatch, Shinobi resolves the complete access plan and structural
closure observation, acquires a shared read claim for every canonical closure
resource, then resolves and observes again under that claim.  A changed plan
or observation is a refusal before the backend starts.  Pure readers may hold
compatible claims concurrently.  A reader may still produce ordinary
non-dataset reports or images through the existing output lifecycle, but a
generic write overlapping the MSv2 closure is refused.  Every generic path
must be concrete at the ownership boundary and covered by the claim; runtime
``Path`` returns, path-valued ``OutputRef`` inputs, output default factories,
glob-selected products, and generic in-place mutation are bounded refusals.
An unannotated generic path may not alias the strict closure.  Mixed
read/output claims retain the conservative exclusive workspace authority. Any
overlapping writer remains excluded through the same workspace ownership and
shared registry protocol; there is no second dataset lock.

The boundary covers the complete flat recipe.  Every effective backend,
including one selected by a leaf without a dataset annotation, must have a
tested route capability.  This stops an ordinary sibling from receiving the
claimed MS through an unannotated ``Path`` while executing in a namespace the
outer lifecycle cannot observe.

``native`` and ``venv`` execute in the lifecycle controller's filesystem
namespace.  ``docker``, ``podman``, ``apptainer`` and ``singularity`` augment
their schema-derived mounts with every resolved canonical closure. Each root
starts read-only and only roots this leaf declares ``write`` access to become
read-write. A declaring ``create`` leaf mounts the target's existing parent
read-write; a missing parent, or an absent create root seen by any other leaf,
is refused. Thus an unannotated container leaf receives all existing claimed
roots read-only, preventing its broad workspace mount from bypassing the
boundary or borrowing another leaf's mutation rights. The host lifecycle
still performs inspection, claims and recovery. A closure member outside the
contained root is refused rather than partially mounted. Mount contradictions
are likewise refused rather than weakening either contract.

Docker and Podman qualify only as local routes. An explicit remote
``DOCKER_HOST``, non-default Docker context, remote ``CONTAINER_HOST`` or named
Podman connection makes the route ``unavailable``: a daemon resolves bind
sources in its own host namespace, which might contain different bytes at the
same spelling.

``slurm`` and ``kubernetes`` are currently ``unavailable`` for strict dataset
execution.  Submitting from a host does not prove that a Slurm compute node
sees the same storage namespace, and a Kubernetes ``hostPath`` spelling does
not prove that the scheduled node names the same storage.  Neither route yet
runs validation and recovery at the execution authority.  This is a
pre-launch refusal, not a fallback to an ordinary ``Path``.  A remote CLI
launch delegates planning to the remote ``ninja`` process, which acquires its
own claim and selects a capability in the remote namespace. ``backend="remote"``
is not a step backend and is refused; use the CLI's remote launch option.

After the backend returns or raises, Shinobi observes again while the claim is
still held.  A changed dataset, or an observation which can no longer be
established, raises
:class:`~shinobi.exceptions.DatasetLifecycleViolationError`.  A backend
failure is re-raised unchanged only when the read-only postcondition still
matches the baseline.  Cache entries, snapshot success transitions, and the
top-level run manifest are held until this postcondition succeeds, including
for recipe leaves without dataset annotations.  A failed postcondition cannot
publish a reusable success.  Cleanup or lifecycle-store failures are attached
to the primary backend exception rather than replacing it.  The lease is
released on every path; a partial metadata release restores the root claim so
an operator can inspect and retry it without opening an overlap window.

Snapshot crash recovery is mutation and therefore never runs over the MSv2
under a shared read claim.  The lifecycle may reconcile only explicitly
claimed writable non-dataset paths.  If the closure itself has a pending
mutation marker, execution is refused until a writer or operator with
exclusive authority performs recovery.

Every attempt durably records its versioned capability, resolved accesses,
the backend route decisions, claim, pre/post observations, phase events,
reason and terminal outcome below
``.shinobi/dataset-attempts`` in the launch workspace.  The public
:class:`shinobi.DatasetLifecycleAttempt` and
:class:`shinobi.DatasetLifecyclePhase` models describe those records; use
:func:`shinobi.read_dataset_attempt` to read and validate one file.
Successful runs end in ``committed``; pre-execution policy/claim failures in
``refused``; execution or postcondition failures in ``failed``.

Contained mutation
------------------

``contained-native-msv2-mutation/v1`` extends the same route to ``write`` and
``create`` access with **exact** recovery.  A workflow in which any leaf
writes or creates a strict dataset runs this lifecycle instead of the read
one; leaves that only read are checked within it.  It accepts one or more
pairwise-disjoint contained ordinary MSv2 roots (a ``create`` target is
planned as absent), under the same flat-recipe and tested-backend boundary.

Exact recovery reuses the Tier 1 mutation-chain snapshots
(``shinobi.snapshots``) in a *strict* policy.  Tier 1's default promise is
"never worse than an uncached run": a state it cannot name or restore
degrades to running against live disk with a warning.  A strict dataset
promises its exact predecessor instead, so every such degradation is a
refusal **before** the tool launches -- a missing or unaffordable snapshot, a
predecessor that predates a write the journal could not name, two journal
histories for one tree (an alias), a dataset whose root changed outside the
journal, a predecessor whose structure differs from the one recorded for its
state name, or a mutated field Tier 1 would decline (list-valued, scattered or
wired to a keyless producer).  Because states are named by the writing step's
cache key, a writing leaf caches automatically, whatever the configured
default.  Only an *explicit* ``cache=False`` refuses the workflow: the call
argument (``ninja run --no-cache``), the writer's own ``cache`` or an
enclosing recipe's.  Snapshots must not be ``off``, and every writer must
journal into the workflow's cache directory.  Each attempt's cache decision
says when caching was enabled automatically.  A field wired from the
top-level recipe's own input is a boundary dataset, exactly as if it had been
passed to the step directly.

The workflow takes an **exclusive** claim over every written closure and
performs crash recovery for those datasets under it: an in-flight marker left
by an interrupted strict step is decided by that step's own success oracle
(below), and an unvouched successor is rolled back before anything runs.
Datasets the workflow only reads stay under a shared claim and a pending
marker on one is still refused.

Each strict leaf then:

#. records its cache decision -- ``miss``, ``hit``, or ``rejected-hit`` -- with
   the identity used and, per dataset field, the fingerprint coverage (a
   written dataset contributes only its path string; a wired one its producer
   lineage; an unwired one its per-file path/mtime/size fingerprint).  A skip
   cache hit on a writer is reused only when the journal vouches for the live
   dataset -- same structure *and* same member files as the recorded head --
   and that head *descends* from the state the step produced.  This catches
   a same-key re-run of an upstream writer, which moves the dataset back
   while every downstream key still matches;
#. refuses, before anything else, a dataset whose table files changed since
   the journal recorded its head (an in-place write outside the pipeline,
   which moves neither the structure nor the root directory's ctime), rather
   than reusing it or restoring over it;
#. restores and verifies its exact predecessor, snapshots it, marks the
   dataset in flight, and observes it structurally;
#. runs the tool.  A non-zero exit or exception returns the dataset to an
   exact, trusted state **immediately** rather than leaving the partial write
   for the next run: its predecessor when the step ran against the head it
   found, or that head when the step had first restored an older predecessor
   (a mid-chain re-run must not leave the workspace behind the complete state
   it found -- crash reconciliation does the same).  A ``create`` target is
   removed;
#. validates its declared postconditions.  Exit status zero is not success:
   a writer may change the MAIN row count, the schema (MAIN columns,
   subtables, closure membership) and MAIN keyword names only where its
   ``DatasetAccess`` permits, may add or remove only its declared columns
   when its columns are known, and may touch only the tables it declares
   (checked against each table's own backing files).  A ``create`` must
   produce a valid contained MSv2 carrying its declared columns.  A reader
   must leave the dataset identical.  A writer that breaks its contract is
   rolled back and raises
   :class:`~shinobi.exceptions.DatasetLifecycleViolationError`;
#. snapshots the successor, commits it in the journal with its structural
   signature and parent state, and only then writes its ``committed`` leaf
   record.  That record is the in-flight marker's success oracle; the reusable
   cache index is written after it.

Which *cells* a writer changed is not observable structurally, so declared
column writes are recorded, not verified.  Every mutation record keeps four
identities apart: the physical snapshot directories, the structural signature
(``msv2-structural-signature/v1``: row count, column/keyword/subtable names,
closure membership, table files and storage managers -- explicitly **not**
cell values) together with the member fingerprint
(``msv2-member-fingerprint/v1``: each table file's relative path, size and
mtime), the step's cache key, and the journal state names that give
provenance lineage.  The signature refuses an incompatible tree and the
fingerprint catches a write through the filesystem; neither is a content
hash, and neither is evidence that visibility data are scientifically
unchanged.  A failed mutation's record also names the state it left on disk
(``restored_state``), separately from the one it consumed.  A future MSv4/Zarr
reusable state is a separate identity and never replaces the native
predecessor snapshot as the rollback source.

Mutation attempts are schema version 2 records: the version 1 fields plus
``absent_roots``, crash ``recovery`` notes and one ``leaves`` entry per strict
step with its accesses, cache decision, pre/post observations and, per
written dataset, a :class:`~shinobi.DatasetMutationRecord`
(predecessor/successor state, signature and snapshot, and outcome:
``committed``, ``rolled-back`` (the predecessor is back), ``pre-run-restored``
(the head the run found is back, after a mid-chain re-run failed),
``absent-restored``, ``untrusted`` when a rollback itself failed and the
dataset stays marked for recovery, or ``refused``).

Re-creating a dataset
~~~~~~~~~~~~~~~~~~~~~

A ``create`` step refuses a target that already exists, so re-running a
pipeline that simulates its own MS fails at planning with a pointer to the
opt-in:

.. code-block:: console

   $ ninja run pipeline.py:sim --ms obs.ms --overwrite simulate

``--overwrite STEP`` (repeatable; ``overwrite_steps=["simulate"]`` from
Python) deletes the existing ``create`` targets of STEP and invalidates the
cached results of STEP and everything downstream of it -- through wiring,
``after`` and access-hazard edges -- before the workflow plans.  The deletion
runs under its own short exclusive claim, after reconciling any interrupted
strict mutation on the target, and drops the target's journal history (its
snapshots stay for ``ninja cache evict``).  The workflow then plans and
claims as usual; a target recreated by someone else in between is refused,
not overwritten.  The attempt record lists each ``overwrites`` entry.

Because the path comes from a parameter, overwrite deletes only a CASA table
(a directory containing ``table.dat``), and refuses a symlink (its resolved
path is the link's target, which the caller never named) or a path containing
the workspace or cache directory.  These checks run again under the claim,
immediately before deletion.  A symlinked *parent* directory is followed, as
``rm -r`` would: ``data/obs.ms`` with ``data`` linked to scratch storage names
the table on scratch storage.  It names steps that ``create`` a strict
``MeasurementSetV2`` only.  A target whose own parameters include
``overwrite`` keeps its ``--overwrite`` flag; name the step with
``--overwrite-step STEP`` instead.  ``--overwrite`` is refused with
``--dryrun``.

The boundary is intentionally narrow.  Unresolved or runtime-selected generic
products, generic in-place mutation, a strict write addressed through a
subtable path rather than its MS root, several fields writing one root in one
step, replacing an existing ``create`` target without ``--overwrite``,
external closure members (one
directory rename cannot restore a closure spanning several roots),
unsupported/opaque closure shapes, nested dataset recipes, dataset scatter,
orchestration functions, manual ``Scope`` routes, unproven scheduler/storage
namespaces, and detached/offloaded execution remain strict refusals.
Dry-run and compilation may still plan dataset-bearing workflows, but legacy
and worker submission retain their planning-only marker and refuse before
scheduler records, ownership claims, logs, or ``sbatch``.  Use ``Path`` or the
legacy loader dtype ``MS`` when one of those unsupported execution routes is
required.
