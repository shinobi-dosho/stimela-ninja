Configuration
=============

Settings are layered via `pydantic-settings
<https://docs.pydantic.dev/latest/concepts/pydantic_settings/>`_. From lowest
to highest precedence:

#. built-in defaults,
#. a config file,
#. environment variables (``SHINOBI_*``),
#. explicit CLI overrides.

The model is :class:`shinobi.config.AppConfig`.

Settings
--------

.. code-block:: yaml

    # ~/.shinobi/config.yml
    backend:
      default: native      # default backend when none is specified
    execution:
      max_workers: 1       # concurrent recipe steps (1 = sequential)
    log:
      dir: "."             # log output directory
      level: INFO          # log level

``execution.max_workers`` defaults to ``1``: parallelism is opt-in. At ``1``
the scheduler reproduces exact declaration-order execution and no ``MUTABLE``
input can be shared across concurrently-running steps.

Config file location
---------------------

By default ``ninja`` reads ``~/.shinobi/config.yml`` if it exists. Point it at
a different file with the global ``--config`` option:

.. code-block:: console

    $ ninja --config ./my-config.yml run myrecipe.py:selfcal --ms data.ms

Environment variables
----------------------

Every setting can be overridden with a ``SHINOBI_``-prefixed environment
variable. Nested fields use a double-underscore delimiter:

.. code-block:: console

    $ export SHINOBI_BACKEND__DEFAULT=docker
    $ export SHINOBI_EXECUTION__MAX_WORKERS=4
    $ export SHINOBI_LOG__LEVEL=DEBUG

Loading config in Python
------------------------

.. code-block:: python

    from shinobi.config import AppConfig

    config = AppConfig.load()                       # defaults + file + env
    config = AppConfig.load("my-config.yml")        # explicit file
    config = AppConfig.load(backend={"default": "docker"})  # CLI-style override
