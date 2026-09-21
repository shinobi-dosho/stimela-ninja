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

For a source checkout managed by uv, use ``uv sync --extra casacore``.

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
existing target is refused because replacement needs a separate lifecycle
policy.
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
though read/read workflows remain compatible.  Current strict execution is
refused before acquiring such a claim; no unclaimed reader is dispatched.

The first planner is deliberately bounded: dataset-bearing scatter and nested
recipes are refused with a diagnostic rather than partially planned.  Flatten
those steps (including bounded loop expansions) so every access has one
declared graph node.

Execution status
----------------

Strict dataset annotations and access contracts are currently declaration,
inspection and planning APIs, not an execution-lifecycle contract.  A scope
carrying either is refused before local execution or Slurm submission.
Dry-run, the planning API, pure legacy Slurm compilation, and worker-bundle
freezing/preparation may resolve closures and show inferred order, but do not
dispatch.  Dataset-bearing legacy and worker workflows are marked
planning-only; both submission functions refuse before creating scheduler
records, ownership claims, logs, or calling ``sbatch``.  This guard is
deliberate: accepting the field as a plain path would suggest that validation,
staging, ownership, recovery, and provenance all enforce the structural
contract when they do not yet.

Use ``Path`` or the legacy loader dtype ``MS`` for executable cabs until that
lifecycle support ships.
