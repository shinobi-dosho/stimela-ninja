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
      enforce_resources: auto    # backend-side limits: auto | always | never
    log:
      dir: "."                   # log output directory
      file: null                  # run-log filename (null = file logging off)
      level: INFO                # log level
      stream: true                # live-echo running cabs' stdout/stderr
      capture_head_lines: 5000    # lines kept from the start of each stream
      capture_tail_lines: 5000    # lines kept from the end of each stream
    cache:
      enabled: false              # step-level result caching, off by default
      dir: ".shinobi/cache"       # cache directory
      content_sample: false       # sample file extents into boundary fingerprints
      snapshots:
        mode: auto                # auto | copy | off -- mutation-chain snapshots
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

``execution.enforce_resources`` controls whether a *backend* turns a
declaration into a real limit. Admission control is unaffected by it and
always applies -- this is only about ``--cpus``/``--memory`` and friends.

``auto`` (the default) emits only what this host can actually enforce.
Enforcement by a rootless runtime (apptainer, rootless podman) needs the
relevant cgroup controller **delegated** to your systemd user session, and
delegation is per controller: a session with ``memory pids`` and no ``cpu``
can hold every step to its memory declaration, but cannot apply ``--cpus`` at
all. Under ``auto`` each dimension is decided on its own, so such a host
enforces memory and warns once that CPU limits were dropped -- rather than
failing every step that declares both. Check what you have with::

    cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/cgroup.controllers

``always`` emits everything declared and lets the runtime fail loudly if it
cannot apply it (the behaviour before this was per-dimension). ``never`` emits
nothing, leaving the scheduler's admission control as the only consumer of a
declaration -- for a site that wants planning without backend enforcement.
Docker is unaffected by the probe either way: its limits are applied by a root
daemon, whose reach your session's delegation says nothing about. So is a
Slurm job script, which is compiled here but runs on a compute node this host
cannot inspect. See :doc:`backends`.

``backend.run_as_host_user`` (docker/podman only, default ``True``) adds
``--user uid:gid`` plus ``HOME=<workdir>`` so bind-mounted outputs come out
owned by the invoking host user instead of root. A *rootless* ``podman``
gets the ``HOME`` half only -- it already runs the container as the invoking
user, and ``--user`` there names an unmapped subuid inside the user
namespace, which would make every write to a bind mount fail. It's a no-op
for ``apptainer``, which already runs as the host user; set it to ``False``
for images that need to run as root. See :doc:`backends`.

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
logged from the captured text after each step completes, so the log covers
non-streaming backends too and is unaffected by ``--quiet`` -- within the
capture limits below. All three settings are overridable per-invocation with
the global ``ninja --log-file/--log-dir/--log-level`` options.

``log.capture_head_lines`` / ``log.capture_tail_lines`` (default ``5000``
each) bound how much of a cab's stdout and stderr is held in memory. Radio
tools are not shy: a single wsclean or CASA step can emit hundreds of MB of
progress chatter, and every line of it would otherwise be retained for the
whole run -- once in the ``BackendRun``, again in the ``StepResult``, and
again in the recipe-level aggregate. Beyond the limits the *middle* of each
stream is replaced by a single ``... [shinobi] N lines elided ...`` marker,
keeping both ends: the banner, the resolved parameters and the early failures
at the top, the result, the summary and the traceback at the bottom.

Two things are deliberately exempt. Live echo (``log.stream``) is never
capped -- it is a side channel, not a buffer, so a chatty tool still scrolls
past in full. And **lines matching the cab's own wranglers are retained
wherever they occur**, so capping costs readability but not output values;
if even those overflow, the step warns rather than silently returning an
unset output. Set either limit to ``0`` to drop that end entirely.

Programmatic runs never write a log file (shinobi's modules only emit
through the ``shinobi.*`` logger hierarchy); attach your own handler to
``logging.getLogger("shinobi")`` instead.

``cache.enabled`` turns on step-level result caching: a step with an
unchanged cache key is skipped and its prior result reused. It's off by
default and must also be opted into per-step or per-recipe via ``Scope.cache``
-- see ``shinobi.cache``. ``ninja run --cache-dir``/``--no-cache`` override
this per invocation.

``cache.snapshots.mode`` controls mutation-chain snapshots, which ride on
caching being enabled (see ``shinobi.snapshots``). When several steps rewrite
one measurement set in turn -- split, then flag, then calibrate -- the cache
can say *which* state of that MS a step consumed, but on its own it cannot
put that state back. So re-running a middle step executes it against the
chain's final state, and resuming after a step died mid-rewrite executes it
against its own half-written output. Both produce wrong results that look
finished. With snapshots on, each state is saved under a name and restored
before the step that needs it re-runs.

* ``auto`` (default): use the cheapest copy the filesystem supports. On XFS
  with ``reflink=1``, Btrfs, or ZFS with block cloning enabled, a snapshot
  shares blocks with the original and costs almost nothing; elsewhere it is a
  full copy, and a snapshot that would not fit is refused loudly rather than
  half-written. ``ninja cache check`` reports what was chosen for each
  filesystem and why.
* ``copy``: always take full copies. For measuring the real cost, or for a
  filesystem whose clone support you don't trust.
* ``off``: the escape hatch -- no snapshots, no journal, no restores, leaving
  exactly the plain skip-cache behaviour described above.

Anything the snapshot layer cannot name -- a scattered or many-valued mutated
field, a path produced by an uncached step, a snapshot that could not fit --
is left alone with a warning, and runs exactly as it would with snapshots
off. It never rolls back something it cannot justify.

.. important::

   With rollback in play, **declaring in-place mutation stops being an
   optimisation and becomes a correctness requirement.** It used to only
   affect whether a step re-ran needlessly. Now, a step that rewrites a
   measurement set another step also uses must say so -- by declaring that
   path as an output, or marking the input ``mutable`` -- and, if a later
   step depends on that rewrite, must be **wired** ahead of it so the graph
   records the ordering. Two steps that touch one path with no edge between
   them are independent as far as shinobi is concerned, and a rollback of
   one can discard the other's work without anything noticing.

``cache.content_sample`` adds a bounded sample (the first and last 4 KiB of
each file) to the fingerprints of *boundary* inputs -- raw data you supplied,
which shinobi did not produce. It exists for one case: two different datasets
that happen to have the same layout and sizes, with mtimes preserved by
``cp -a`` or ``tar -x``, which would otherwise fingerprint identically. It
does **not** detect an intermediate edited behind shinobi's back: rewriting a
column in the middle of a table changes neither the file's size nor its first
and last pages. That case is undetectable by design -- the declared graph is
the truth. Off by default, because turning it on changes every boundary-input
key, so the first run afterwards recomputes that whole layer.

Cache tooling
~~~~~~~~~~~~~

``ninja cache check`` reports anything the cache cannot vouch for: steps
interrupted mid-run, paths whose content is not vouched for, quarantined
trees left by a crash, journal/manifest disagreements, and the clone
capability chosen per filesystem. It only reads.

``ninja cache invalidate <step-path>`` forgets a step's cached result *and*
rolls its snapshots back, then forces the rest of its chain to re-run. Use it
when a step exited zero but wrote something wrong -- dropping the cache entry
alone would leave that output snapshotted and reachable.

``ninja cache evict --bytes N`` frees snapshot space, dropping unreachable
states first and never one a live chain still needs.

``ninja clean --cache`` removes the cache directory and the snapshot journal
with it, and refuses while a quarantined tree is outstanding -- the journal
is the only thing that explains what such a tree was set aside for. Pass
``--force`` to remove both.

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
