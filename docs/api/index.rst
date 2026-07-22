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

Execution
---------

.. autoclass:: shinobi.results.BackendRun
   :members:

.. autoclass:: shinobi.results.StepResult
   :members:

.. autofunction:: shinobi.steps.dispatch.register_step_backend

.. autofunction:: shinobi.steps.dispatch.get_step_backend

.. automodule:: shinobi.steps.loops
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

.. automodule:: shinobi.loaders.cultcargo
   :members:

.. automodule:: shinobi.loaders.stimela_classic
   :members:

.. automodule:: shinobi.loaders
   :members:

Configuration
-------------

.. automodule:: shinobi.config
   :members:
