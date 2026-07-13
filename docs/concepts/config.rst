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
      default: native            # default backend when none is specified
      run_as_host_user: true     # docker/podman: run as host uid:gid, not root
    execution:
      max_workers: 1             # concurrent recipe steps (1 = sequential)
    log:
      dir: "."                   # log output directory
      level: INFO                # log level
      stream: true                # live-echo running cabs' stdout/stderr
    cache:
      enabled: false              # step-level result caching, off by default
      dir: ".shinobi/cache"       # cache directory
    provenance:
      enabled: false              # image pinning + run manifests, off by default
      dir: ".shinobi/runs"        # where run manifests are written

``execution.max_workers`` defaults to ``1``: parallelism is opt-in. At ``1``
the scheduler reproduces exact declaration-order execution and no ``MUTABLE``
input can be shared across concurrently-running steps. Raising it lets
independent recipe branches run concurrently -- see the execution model in
:doc:`recipes`. A recipe can also set its own ``max_workers``, overriding this
default.

``backend.run_as_host_user`` (docker/podman only, default ``True``) adds
``--user uid:gid`` plus ``HOME=<workdir>`` so bind-mounted outputs come out
owned by the invoking host user instead of root. It's a no-op for
``apptainer``, which already runs as the host user; set it to ``False`` for
images that need to run as root. See :doc:`backends`.

``log.stream`` (default ``True``) live-echoes a running cab's stdout/stderr
as it runs (native/container backends only); set to ``False`` to restore the
old behavior of a silent run followed by one dump of captured output at the
end. Overridable per-invocation with ``ninja run --quiet``.

``cache.enabled`` turns on step-level result caching: a step with an
unchanged cache key is skipped and its prior result reused. It's off by
default and must also be opted into per-step or per-recipe via ``Scope.cache``
-- see ``shinobi.cache``. ``ninja run --cache-dir``/``--no-cache`` override
this per invocation.

``provenance.enabled`` turns on reproducible-run provenance: container images
are digest-pinned before running (pin-then-run) and a run manifest is written
per top-level run under ``provenance.dir``. It's off by default because
pinning changes how containers execute. ``ninja run --provenance``/
``--no-provenance`` override this per invocation. See :doc:`provenance`.

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
