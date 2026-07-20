stimela-ninja (Stimela 3.0)
===========================

A spiritual successor to `Stimela classic
<https://github.com/ratt-ru/Stimela-classic>`_, built around the same core
philosophy: **robust and flexible simplicity for reproducible radio
astronomy pipelines**.

Recipes are plain Python. A step is a function call; a step's output is a
Python value you wire into the next call. There is no YAML
expression/substitution language, no alias-propagation system, and no stacked
config libraries -- control flow is just Python, and it doesn't need
reinventing.

.. code-block:: python

    from pydantic import BaseModel

    from shinobi import Cab, Recipe, step


    class ImageInputs(BaseModel):
        ms: str = "obs.ms"
        prefix: str = "img"


    class ImageOutputs(BaseModel):
        restored: str | None = None


    wsclean = Cab(
        name="wsclean",
        command="wsclean",
        image="quay.io/stimela/wsclean:latest",
        inputs_model=ImageInputs,
        outputs_model=ImageOutputs,
    )


    @step(wsclean, backend="native")
    def image(ctx):
        """Image the visibilities. A near-empty body auto-runs the cab."""
        return ctx.run()

Run it straight from the command line -- the step's schema becomes the CLI
options, no entrypoint script required::

    ninja run myrecipe.py:image --ms data.ms --prefix out

Architecture
------------

- **Cabs** (``shinobi.Cab``) -- a typed, backend-agnostic description of
  an atomic task: an inputs/outputs schema (pydantic models) plus *policies*
  for turning parameters into a CLI invocation. Define one directly in Python,
  or load one from existing `cult-cargo
  <https://github.com/caracal-pipeline/cult-cargo>`_ YAML
  (``shinobi.loaders.cultcargo``) -- that schema format is good design and
  is reused as-is, including its ``_include`` (file composition) and ``_use``
  (dotted-path deep-merge) mechanisms, verified against real upstream cab
  files. The ``=config.x.y`` expression language and package-scoped includes
  are deliberately not implemented -- see the module docstring and, for the
  security rationale behind the package-scoped-include restriction,
  ``SECURITY.md``.

- **Steps** (``shinobi.step``, ``shinobi.pystep``) -- a step binds an
  orchestration function to a scope. ``@shinobi.step`` decorates a function
  with an existing ``Cab``/``Recipe``; its body receives an ``ExecContext``
  (``ctx``) and calls ``ctx.run()`` to execute. ``@shinobi.pystep`` turns a
  plain, type-hinted Python function into a step, deriving its schema from the
  signature -- no external tool, no hand-written models.

- **Backends** (``shinobi.backends``) -- pluggable executors, all shelling
  out to the relevant CLI rather than a Python SDK: ``native`` (subprocess),
  ``docker``/``podman``/``apptainer``, ``slurm`` (``sbatch``/``sacct``),
  ``kubernetes`` (``kubectl``, batch ``Job``\ s). Every backend blocks until
  the job finishes and returns a ``BackendRun`` -- no async mode, steps are
  scheduled by dispatch, not left to fire-and-forget. Container/cluster
  backends derive bind mounts from the cab's own schema (File/MS-dtype
  params get their parent dir mounted). ``native``/container backends were
  verified against a real ``quay.io/stimela/wsclean`` image; ``kubernetes``
  against a real ``kind`` cluster; the ``slurm`` step backend has no live
  test yet (covered only by mocked-CLI tests) -- see
  ``docs/concepts/backends.rst`` for the full verification status.

- **Recipes** (``shinobi.Recipe``) -- just Python. A ``Recipe`` composes
  steps, wiring one step's output into the next either declaratively (via
  ``StepRef``/``InputRef``/``OutputRef``, or the ``recipe.inputs`` /
  ``recipe.outputs`` proxies and ``add_step``) or through an orchestration
  function whose body is ordinary Python.

- **Config** (``shinobi.config.AppConfig``) -- layered settings via
  pydantic-settings: built-in defaults < config file < env vars
  (``SHINOBI_*``) < explicit overrides.

CLI
---

Every ``Cab``, ``Recipe``, or ``@shinobi.step``-decorated function can be run
directly, without writing a Python entrypoint script -- its signature/schema
becomes CLI options automatically::

    ninja run myrecipe.py:image --ms data.ms --prefix out
    ninja run myrecipe.py:selfcal --ms data.ms

``ninja run <target>`` resolves ``<target>`` (``path/to/file.py:name`` or a
dotted module path) to the ``Cab``, ``Recipe``, or ``StepRef`` it names and
dispatches it with the parsed options.

Add ``--dryrun`` to see the execution graph a recipe would produce, without
running anything::

    $ ninja run myrecipe.py:selfcal --ms data.ms --dryrun
    [ image ]
        |
        v
    [ mask ]

Nothing is executed to produce this: a ``Recipe`` is a declared graph -- a
list of steps plus their ``InputRef``/``OutputRef`` wiring, built once when
the recipe module runs -- and ``--dryrun`` simply renders that graph. The
same validation (``shinobi.graph.build_graph``) backs both the renderer and
the real executor, so a cyclic or mis-wired recipe is rejected identically
either way, and the diagram never disagrees with what a real run would do.
Steps that share the same declared dependencies render on one row (a
**fan-out**); a step fed by several upstream outputs is a **fan-in**.

A purely-declarative recipe (no orchestration functions, no MUTABLE inputs,
only paths crossing between steps) can be **offloaded** to a cluster with
``ninja compile``, which emits linked ``sbatch`` scripts and, with
``--submit``, hands the workflow off and detaches::

    ninja compile myrecipe.py:pipe --target /scratch/made.ms --submit
    ninja status /scratch/.shinobi/pipe/handle.json

See ``docs/design.rst`` for the design philosophy behind the declared-DAG
model and what's deliberately left out.

Status
------

Early scaffolding. Interfaces above are real and tested (``pytest``), but this
is not yet ready to run real pipelines.

Installation
------------

Once published to PyPI::

    pip install stimela-ninja

Until then, install the latest from GitHub::

    pip install git+https://github.com/SpheMakh/stimela-ninja.git

This installs the ``ninja`` command and the importable ``shinobi`` package.

Documentation
-------------

Full documentation is built with Sphinx and hosted on Read the Docs. Build it
locally with::

    uv sync --group docs
    uv run sphinx-build -b html docs docs/_build/html

Development
-----------

.. code-block:: bash

    uv venv .venv && uv pip install -e . --group dev
    .venv/bin/pytest
    .venv/bin/ruff check src tests
