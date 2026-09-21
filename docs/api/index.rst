API reference
=============

This reference is generated from the source docstrings. The most useful names
are re-exported from the top-level :mod:`shinobi` package and documented here;
supporting types live in their home modules below.

Top-level package
-----------------

.. automodule:: shinobi
   :members:
   :imported-members:

Schema helpers
--------------

Supporting types used when defining cabs, not re-exported at the top level.

.. autoclass:: shinobi.steps.schema.ParamMeta
   :members:

.. autoclass:: shinobi.steps.schema.ParamPattern
   :members:

.. autoclass:: shinobi.steps.schema.ParamSegment
   :members:

.. autoclass:: shinobi.steps.schema.Policies
   :members:

.. autoclass:: shinobi.steps.schema.ScatterSpec
   :members:

.. autofunction:: shinobi.steps.schema.path_fields

.. autofunction:: shinobi.steps.schema.declared_output_dirs

.. autofunction:: shinobi.steps.schema.declared_output_paths

.. autofunction:: shinobi.steps.schema.readonly_path_fields

.. autofunction:: shinobi.steps.schema.mutated_path_fields

.. autofunction:: shinobi.steps.schema.paths_overlap

Datasets
--------

.. currentmodule:: shinobi.datasets

.. py:data:: CasaTable

   A path annotated with the versioned ``casa-table/v1`` declaration.

.. py:data:: MeasurementSetV2

   A path annotated with the versioned ``msv2-structural/v1`` declaration.

.. autoclass:: shinobi.datasets.DatasetType

.. autoclass:: shinobi.datasets.InspectionLimits

.. autoclass:: shinobi.datasets.DatasetDescriptor
   :members:

.. autoclass:: shinobi.datasets.DatasetKind
   :members:

.. autoclass:: shinobi.datasets.DatasetStatus
   :members:

.. autoclass:: shinobi.dataset_closure.DatasetClosure
   :members:

.. autoclass:: shinobi.dataset_closure.ClosureRequirement
   :members:

.. autoclass:: shinobi.dataset_closure.ClosureCapabilities

.. autoclass:: shinobi.dataset_closure.ClosureResource

.. autoclass:: shinobi.dataset_closure.ClosureStatus
   :members:

.. autofunction:: shinobi.dataset_closure.resolve_dataset_closure

.. autoclass:: shinobi.dataset_access.DatasetAccess

.. autoclass:: shinobi.dataset_access.DatasetMode
   :members:

.. autoclass:: shinobi.dataset_access.DatasetFallback
   :members:

.. autoclass:: shinobi.dataset_access.DatasetTable
   :members:

.. autoclass:: shinobi.dataset_access.DatasetColumns

.. autoclass:: shinobi.dataset_access.DatasetSelection

.. autoclass:: shinobi.dataset_access.ResolvedDatasetAccess
   :members:

.. autoclass:: shinobi.dataset_access.RecipeAccessPlan

.. autofunction:: shinobi.dataset_access.resolve_scope_dataset_accesses

.. autofunction:: shinobi.dataset_access.plan_recipe_accesses

.. autofunction:: shinobi.datasets.dataset_fields

.. autofunction:: shinobi.datasets.dataset_declarations

.. autofunction:: shinobi.datasets.inspect_dataset

.. autofunction:: shinobi.datasets.inspect_casa_table

.. autofunction:: shinobi.datasets.inspect_measurement_set_v2

Execution
---------

.. autoclass:: shinobi.results.BackendRun
   :members:

.. autoclass:: shinobi.results.StepResult
   :members:

.. autofunction:: shinobi.steps.dispatch.register_step_backend

.. autofunction:: shinobi.steps.dispatch.get_step_backend

.. automodule:: shinobi.wranglers
   :members:

.. automodule:: shinobi.steps.loops
   :members:


Graphs and resources
--------------------

.. automodule:: shinobi.graph
   :members:

.. automodule:: shinobi.resources
   :members:

Backends
--------

.. automodule:: shinobi.backends
   :members:

Building argv
-------------

.. automodule:: shinobi.policies
   :members:

Loaders
-------

.. automodule:: shinobi.loaders.yaml_cab
   :members:

.. automodule:: shinobi.loaders.worker_schema
   :members:

.. automodule:: shinobi.loaders.stimela_classic
   :members:

.. automodule:: shinobi.loaders
   :members:

Configuration
-------------

.. automodule:: shinobi.config
   :members:

Caching
-------

.. autoclass:: shinobi.cache.ProvenanceKey
   :members:

Exceptions
----------

.. automodule:: shinobi.exceptions
   :members:
   :show-inheritance:
