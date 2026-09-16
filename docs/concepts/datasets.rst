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
``SOURCE`` and optional or custom columns remain legal.

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

Execution status
----------------

Strict dataset annotations are currently a declaration and inspection API,
not an execution-lifecycle contract.  A scope carrying one is refused before
local or offloaded execution.  This guard is deliberate: accepting the field
as a plain path would suggest that validation, staging, ownership, recovery,
and provenance all enforce the structural contract when they do not yet.

Use ``Path`` or the legacy loader dtype ``MS`` for executable cabs until that
lifecycle support ships.
