Backends
========

A **backend** takes a cab and a resolved argv and runs it somewhere. Backends
are pluggable executors; every one shells out to the relevant CLI rather than
using a Python SDK, and every one *blocks* until the job finishes. A backend
returns a raw :class:`~shinobi.results.BackendRun` (return code, stdout,
stderr); the dispatch layer wrangles that into the schema-aware
:class:`~shinobi.results.StepResult` a step call yields. There is no async
mode -- recipes are plain Python.

Available backends
------------------

``native``
    Runs the command as a local subprocess. No isolation; the command must be
    on ``PATH``.

``venv``
    Runs the command inside an existing Python virtualenv -- the venv's ``bin``
    is prepended to ``PATH``, ``VIRTUAL_ENV`` is set, and
    ``PYTHONHOME``/``PYTHONPATH`` are cleared (what ``activate`` does, minus the
    shell). A bare command is rewritten to the venv's own copy when it has one,
    so a genuinely missing tool fails loudly rather than falling through to a
    host binary of the same name. This is *weaker* than a container -- no
    filesystem namespace, no OS-level pin -- and a **complement** to the
    container backends, not a replacement: it covers the pip-installable half of
    a pipeline (quartical, tricolour, breizorro, ...) without a container
    runtime, while tools that need a full image (wsclean, casa, aoflagger) still
    use one. The venv is named on the step (``venv=`` on a ``Scope`` or
    ``@pystep``) or in :doc:`config` (``backend.venv``); it is **not** read from
    cab YAML, since a venv path is machine-specific. If the ``venv`` backend is
    selected but no venv is declared anywhere, the step runs natively (with a
    warning). ``@pystep`` functions run under the venv's own interpreter and
    import its real packages. See :doc:`provenance` for why a venv run is always
    reported *unpinned*. Not supported by :doc:`../offloading` (an offloaded
    venv step is refused, not silently run without its venv). A ``--dryrun`` of
    a venv cab prints the plain (un-rewritten) argv; the venv resolution happens
    only at run time.

``docker`` / ``podman`` / ``apptainer``
    Runs the cab's ``image`` in a container. Bind mounts are derived from the
    cab's own schema -- every ``File``/``MS``-dtype parameter contributes its
    parent directory as a mount, so inputs and outputs are visible inside the
    container. The output side is read the same way, so that a tool whose
    output stem is a plain ``str`` parameter (the usual shape -- see
    :ref:`declaring-where-a-tool-writes`) still has its products land on the
    host: whatever directory a ``File``-dtype output or a ``harvest`` pattern
    resolves to is mounted read-write when it is absolute. A directory that
    does not exist yet contributes its nearest existing ancestor, so a tool
    that creates its own output tree behaves as it would natively; an
    absolute output path with no existing ancestor at all is refused before
    the run rather than written into the container and lost. When a write
    target lands in a directory an input marked ``writable: false`` made
    read-only, both declarations still hold: the directory is mounted
    read-write and the read-only input is re-asserted ``:ro`` at its own path
    inside it, so the tool can write its product but not touch that input.
    For ``docker``/``podman``, the container runs as the invoking
    host user (not root) by default, so bind-mounted outputs come out
    host-owned -- see ``backend.run_as_host_user`` in :doc:`config`.
    ``apptainer`` already runs as the host user, so this is a no-op there.
    With :doc:`provenance` enabled, the image is digest-pinned before running
    (``repo@sha256:...``); by default it runs by its original tag.

``slurm``
    Submits the command as a batch job via ``sbatch`` and tracks it with
    ``sacct``.

``kubernetes``
    Runs the command as a batch ``Job`` via ``kubectl``. Mounts carry the same
    read-only classification as the container backends, but the nested-mount
    case above is **refused** rather than emitted: whether a kubelet shadows a
    nested ``readOnly`` volumeMount the way docker, podman and apptainer shadow a
    nested ``:ro`` bind is unverified, and a silent failure there would hand
    the step write access to an input the cab declared read-only.

Choosing a backend
------------------

A backend can be selected in several places, in increasing order of specificity:

* the global default in configuration (see :doc:`config`);
* the ``--backend`` option on the ``ninja`` command line;
* a per-step ``backend=`` on :func:`@shinobi.step <shinobi.step>`;
* a ``backend=`` override passed to ``ctx.run()`` inside an orchestration
  function.

.. code-block:: python

    @step(wsclean, backend="native")
    def image(ctx):
        return ctx.run()

Getting a backend directly
--------------------------

:func:`shinobi.backends.get_backend` returns a backend instance by name, if you
need one outside the CLI:

.. code-block:: python

    from shinobi.backends import get_backend

    backend = get_backend("native")

Resource limits
----------------

When a step declares a footprint (see :doc:`recipes`), what happens to that
declaration depends on the backend -- and the difference is the difference
between a soft scheduling hint and a real limit:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Backend
     - What a declaration does
   * - ``docker`` / ``podman``
     - Emits ``--cpus`` / ``--memory``. Enforced by the container runtime: a
       runaway is killed inside its own cgroup rather than eating memory its
       siblings are using.
   * - ``apptainer``
     - Emits ``--cpus`` / ``--memory``, same spelling and same effect
       (verified: ``--memory 256M`` really does produce a cgroup scope with
       ``memory.max=268435456``). Needs cgroup delegation -- cgroups v2 under
       systemd -- and delegation is **per controller**, so shinobi emits only
       the dimensions this session can enforce and warns about the rest (see
       below).
   * - ``kubernetes``
     - Sets container ``resources.requests`` and ``resources.limits`` to the
       same values, so the cluster reserves exactly what was declared. If no
       node can satisfy the request the pod would sit ``Pending`` forever, so
       shinobi surfaces that as an error instead of waiting.
   * - ``slurm``
     - Emits ``#SBATCH --cpus-per-task`` / ``--mem``, rounded up. Any
       explicitly configured ``sbatch_opts`` win over the derived values.
   * - ``native`` / ``venv``
     - **Nothing.** There is no container and no cgroup, so the declaration is
       honoured only by shinobi's own admission control -- it decides whether
       to *start* the step and cannot constrain it afterwards. If a native
       step's declaration is wrong, nothing catches it.

The same values also feed the local scheduler's admission control, so a step's
declaration is used consistently whether it gates a thread pool or a cluster
allocation.

Partial cgroup delegation
~~~~~~~~~~~~~~~~~~~~~~~~~~

A rootless runtime can only apply the controllers systemd delegated to your
user session, and a shared HPC box very often delegates ``memory pids`` and
not ``cpu`` -- which is precisely where you most want enforcement and least
able to change the delegation policy. Handing such a host ``--cpus`` makes it
refuse to create the container at all::

    FATAL:   container creation failed: while applying cgroups config: ...
    .../apptainer-3628201.scope/cpu.max: no such file or directory

so a single unenforceable dimension used to take down the enforcement of the
other, and the run with it. Under the default ``execution.enforce_resources:
auto`` shinobi probes the delegated set and emits each dimension on its own
merits: memory limits are still applied, and one warning per run says which
flags were dropped and why. A run that fails this way anyway (under
``always``, or on a runtime whose reach could not be probed) gets the cause
and the remedy spelled out instead of the raw ``openat2`` message. See
:doc:`config` for the setting and how to inspect your own delegation.

Verification status
--------------------

The ``native`` and container backends were verified against a real
``quay.io/stimela/wsclean`` image, and ``kubernetes`` against a real ``kind``
cluster (``hostPath`` volumes there only work if the node running the pod
has the path -- fine for a single-node dev cluster or nodes with shared
storage, not a general multi-node cluster without a shared filesystem, which
would need ``PersistentVolumeClaim``\ s instead).

The ``slurm`` step backend has no live test yet -- it's covered only by
tests that mock the ``sbatch``/``sacct`` calls
(``tests/test_slurm_backend.py``), not proven against a real scheduler.
Verify against a real cluster before relying on it. The separate
compile-and-offload Slurm path (:doc:`../offloading`) *does* have live
single-node coverage; this step backend doesn't share it.
