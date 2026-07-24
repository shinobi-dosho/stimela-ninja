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
      venv:                      # settings for the `venv` backend
        default: null            #   venv used when a step declares none (path or a name below)
        envs: {}                 #   name -> venv path, so recipes/config refer to a venv by name
    execution:
      max_workers: 1             # concurrent recipe steps (1 = sequential)
      resources:                 # machine budget steps are admitted against
        cpus: auto               #   auto | unbounded | a number
        memory: auto             #   auto | unbounded | e.g. "250GiB"
    log:
      dir: "."                   # log output directory
      file: null                  # run-log filename (null = file logging off)
      level: INFO                # log level
      stream: true                # live-echo running cabs' stdout/stderr
    cache:
      enabled: false              # step-level result caching, off by default
      dir: ".shinobi/cache"       # cache directory
    provenance:
      enabled: false              # image pinning + run manifests, off by default
      dir: ".shinobi/runs"        # where run manifests are written
    sandbox:
      enabled: false              # per-step sandbox execution, off by default
      dir: ".shinobi/work"        # scratch root for per-step sandbox dirs

``execution.max_workers`` defaults to ``1``: parallelism is opt-in. At ``1``
the scheduler reproduces exact declaration-order execution and no ``MUTABLE``
input can be shared across concurrently-running steps. Raising it lets
independent recipe branches run concurrently -- see the execution model in
:doc:`recipes`. A recipe can also set its own ``max_workers``, overriding this
default.

``execution.resources`` is the total budget the scheduler admits work against
when steps declare what they cost (see :doc:`recipes`). It is only consulted if
something actually declares a footprint, so the default costs nothing.

``auto`` detects the real limit, and detection is **cgroup-aware**: it walks
the whole cgroup ancestor chain and takes the tightest limit at any level. That
matters more than it sounds. A fair-share memory quota is usually set several
levels above the cgroup a process actually runs in, so reading only the leaf
finds no limit, falls back to ``/proc/meminfo``, and reports the host's full
memory -- which is precisely how a tool ends up sizing itself for a machine it
is not allowed to fill, and getting killed for it. Set an explicit value to
override, or ``unbounded`` to stop constraining that dimension. Note ``null``
is *not* the way to spell "unbounded"; elsewhere in this file ``null`` means
"unset, fall back", and it is not quietly inverted here.

``backend.run_as_host_user`` (docker/podman only, default ``True``) adds
``--user uid:gid`` plus ``HOME=<workdir>`` so bind-mounted outputs come out
owned by the invoking host user instead of root. It's a no-op for
``apptainer``, which already runs as the host user; set it to ``False`` for
images that need to run as root. See :doc:`backends`.

``backend.venv`` configures the ``venv`` backend. ``backend.venv.default`` is
the venv a ``venv``-backend step uses when it declares none of its own (a path,
or a key into ``envs``); ``null`` means no default, so such a step falls back
to native. ``backend.venv.envs`` maps short names to venv paths, letting a
recipe or config name a venv (``venv: myenv``) instead of a machine-specific
absolute path. Both are reachable via the environment as
``SHINOBI_BACKEND__VENV__DEFAULT`` / ``SHINOBI_BACKEND__VENV__ENVS``. A venv is
a deployment concern, so it lives here or on a ``Scope`` in Python -- never in a
shared cab repo. See :doc:`backends` and :doc:`provenance`.

``log.stream`` (default ``True``) live-echoes a running cab's stdout/stderr
as it runs (native/container backends only); set to ``False`` to restore the
old behavior of a silent run followed by one dump of captured output at the
end. Overridable per-invocation with ``ninja run --quiet``.

``log.file`` (default ``None`` = off) turns on the run-log file, written to
``log.dir/log.file`` and filtered at ``log.level``. Every step -- cab,
pystep, recipe, and each recipe sub-step under its dotted label (e.g.
``selfcal.image``) -- is logged exactly once, regardless of backend:
lifecycle records (``starting`` / ``finished`` / ``failed`` / ``cache hit``)
and the step's captured stdout/stderr at ``INFO``, failures and exceptions at
``ERROR``, and the resolved backend plus full argv at ``DEBUG``. Output is
logged from the captured text after each step completes, so the log is
complete even for non-streaming backends and unaffected by ``--quiet``. All
three settings are overridable per-invocation with the global
``ninja --log-file/--log-dir/--log-level`` options.

Programmatic runs never write a log file (shinobi's modules only emit
through the ``shinobi.*`` logger hierarchy); attach your own handler to
``logging.getLogger("shinobi")`` instead.

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

``sandbox.enabled`` turns on per-step sandbox execution: each
subprocess-backed step runs with its cwd inside a private scratch directory
under ``sandbox.dir``, and on success only declared outputs are moved back to
the workspace -- auxiliary droppings (tool logfiles etc.) are deleted with the
scratch dir. It's off by default. ``ninja run --sandbox``/``--no-sandbox``
override this per invocation. See :doc:`sandbox`.

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
