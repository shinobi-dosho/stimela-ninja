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

``docker`` / ``podman`` / ``apptainer``
    Runs the cab's ``image`` in a container. Bind mounts are derived from the
    cab's own schema -- every ``File``/``MS``-dtype parameter contributes its
    parent directory as a mount, so inputs and outputs are visible inside the
    container. For ``docker``/``podman``, the container runs as the invoking
    host user (not root) by default, so bind-mounted outputs come out
    host-owned -- see ``backend.run_as_host_user`` in :doc:`config`.
    ``apptainer`` already runs as the host user, so this is a no-op there.
    With :doc:`provenance` enabled, the image is digest-pinned before running
    (``repo@sha256:...``); by default it runs by its original tag.

``slurm``
    Submits the command as a batch job via ``sbatch`` and tracks it with
    ``sacct``.

``kubernetes``
    Runs the command as a batch ``Job`` via ``kubectl``.

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

Verification status
--------------------

The ``native`` and container backends were verified against a real
``quay.io/stimela/wsclean`` image, and ``kubernetes`` against a real ``kind``
cluster. The ``slurm`` backend was **not** live-verified -- no cluster was
available in the development environment. See ``AGENTS.md`` in the repository
for what that means in practice.
