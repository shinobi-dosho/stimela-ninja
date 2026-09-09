Backends
========

A **backend** takes a cab and a resolved argv and runs it somewhere. Backends
are pluggable executors; every one shells out to the relevant CLI rather than
using a Python SDK, and every one *blocks* until the job finishes. A backend
returns a raw :class:`~shinobi.results.BackendRun` (return code, stdout,
stderr); the dispatch layer wrangles that into the schema-aware
:class:`~shinobi.results.StepResult` a step call yields. There is no fire-and-forget backend mode: recipe concurrency is coordinated
by the declared-DAG scheduler.

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

``docker`` / ``podman`` / ``apptainer`` / ``singularity``
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
    target -- or simply another input -- lands in a directory an input marked
    ``writable: false`` made read-only, both declarations still hold: the
    directory is mounted read-write and the read-only input is re-asserted
    ``:ro`` at its own path inside it, so the write lands but that input stays
    untouchable. A cab that declares something writable *inside* a
    ``writable: false`` input is refused instead -- no arrangement of mounts
    satisfies both. Any input can say the word: a declared field marks itself
    with ``writable: false`` in the schema, and a pattern-matched
    (dynamically-named) input -- which has no declared field at all -- marks
    the attr it matches, the same place that attr's ``dtype`` lives.
    For ``docker``/``podman``, the container runs as the invoking
    host user (not root) by default, so bind-mounted outputs come out
    host-owned -- see ``backend.run_as_host_user`` in :doc:`config`.
    ``apptainer`` already runs as the host user, so this is a no-op there.
    ``singularity`` is the same backend under the project's former name: it
    behaves exactly as ``apptainer`` but invokes the ``singularity`` binary,
    which is the only one present on some HPC sites. Pick whichever your
    cluster actually installs.
    With :doc:`provenance` enabled, the image is digest-pinned before running
    (``repo@sha256:...``); by default it runs by its original tag.

``slurm``
    Submits the command as a batch job via ``sbatch`` and tracks it with
    ``sacct``.

``kubernetes``
    Runs the command as a batch ``Job`` via ``kubectl``. Mounts carry the same
    read-only classification and nested shadowing as the
    container backends: a writable parent is mounted first and the protected
    input is re-asserted as a nested ``readOnly`` volumeMount. This behavior is
    live-verified on a real kubelet. ``namespace`` is required, and pods run
    with the caller's uid/gid, privilege escalation disabled, and all
    capabilities dropped.

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

``ninja --backend`` is checked against the list above before anything runs, so
a name that isn't a backend is rejected on the spot and told what the
alternatives are. This matters more than it looks: a ``@pystep`` with an
``image`` containerises only when the resolved backend *is* a container
runtime, and otherwise runs the function in the calling process -- so an
unrecognised name used to run a containerised step on the host and fail on
whatever the image was supposed to provide.

Interrupting a run
------------------

Ctrl-C stops the work, not just the client -- for every backend that runs the
work *here* (``native``, ``venv``, and the four container runtimes). Each
step's process gets its own session, so the interrupt goes to ninja alone,
and ninja then tears the step down deliberately: SIGTERM, a grace period,
SIGKILL, and it does not return until the process and its process group are
confirmed gone. ``SIGTERM`` and ``SIGHUP`` do the same thing as Ctrl-C, which
matters because giving the child its own session is exactly what stopped a
dropped ssh connection from reaching it by accident. During a parallel recipe
the same teardown reaches *every* in-flight step -- an interrupt is delivered
only to the main thread, while the workers sit blocked on children that would
otherwise never hear about it.

``slurm`` and ``kubernetes`` are the exception, and are **not** covered: they
hand work to a scheduler and poll it, so what would need stopping is a job on
a cluster rather than a child of this process. Interrupting a run that has
submitted one leaves the job running, exactly as it did before any of this
existed; cancel it with ``scancel`` or ``kubectl delete job`` yourself.

Docker and podman need more than signals, and this is worth knowing if you
ever inspect a run by hand: their containers are **not** descendants of the
client. containerd-shim or conmon owns them, so killing the client (or its
whole process group) leaves the container running. ninja therefore names
every container it starts ``shinobi-<random>`` and stops it with
``<engine> rm -f``. A container left over from something that bypassed this
is findable and removable directly::

    docker rm -f $(docker ps -qf name=shinobi-)

The ordering matters more than it sounds. A step's workspace -- the sandbox,
and the temporary directory holding a ``@pystep``'s runner and its I/O -- is
cleaned up as the run unwinds, and deleting those while the tool still has
them open does not stop the tool. It keeps writing into files that no longer
have names, and the product it leaves behind is silently incomplete: on one
real interrupted run, a 13 GB measurement set collapsed to 102 MB the moment
the orphaned writer's file descriptors closed. Teardown finishing *before*
cleanup starts is what prevents that.

The same reasoning guards the other direction. Before clearing a stale output
so a tool can rewrite it (see ``execution.clear_stale_outputs`` in
:doc:`config`), ninja checks whether any live process is still **writing**
that path, and refuses with the offending pids rather than deleting it. That
is the case a leaked container from an *earlier* run produces, where teardown
never got the chance to run at all. Readers are only warned about, not
refused: deleting a path out from under a reader is ordinary POSIX -- it
keeps the old inode -- and failing a run because a viewer or a ``tail -f``
had the previous product open would be its own bug.

Know what that check cannot see, because all three make it report *fewer*
holders than exist: it reads ``/proc``, so it is local-only and blind to a
job still running on a cluster node; it cannot read another user's
processes, which includes a rootful docker container's payload under the
default ``backend.run_as_host_user`` settings; and it looks at file
descriptors, not memory mappings, so a casacore table mapped and then closed
does not appear. It is a safety net, not a guarantee.

If a process cannot be killed -- almost always one blocked in uninterruptible
I/O, or a container runtime that will not release it -- ninja says so, names
the pid, and leaves that step's workspace in place rather than cleaning up
around something still running. The run fails rather than reporting a clean
cancellation, because something is still going.

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
   * - ``apptainer`` / ``singularity``
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

The ``native``, ``docker``, ``podman``, and ``apptainer`` paths were
verified with real runtimes and bind-mounted host data, and ``kubernetes``
against a real ``kind`` cluster (``hostPath`` volumes there only work if the
node running the pod
has the path -- fine for a single-node dev cluster or nodes with shared
storage, not a general multi-node cluster without a shared filesystem, which
would need ``PersistentVolumeClaim``\ s instead).

Both Slurm paths -- the ``slurm`` step backend and the separate
compile-and-offload path (:doc:`../offloading`) -- were verified against a
real Slurm cluster. What differs is how much of that a test run re-checks:
the offload path has automated live coverage (``tests/test_slurm_live.py``
against the throwaway cluster in ``tests/slurm_live/``, skipped unless it
is up), while the step backend's automated tests mock the
``sbatch``/``sacct`` calls (``tests/test_slurm_backend.py``). Both live
setups are single-node, so multi-node scheduling and cross-node shared
storage remain unproven either way.
