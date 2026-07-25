Migrating from CARACal / Stimela 2
==================================

If you have a CARACal pipeline, the thing to understand first is what this
project asks of you and what it doesn't.

**Your cab definitions carry over.** cult-cargo YAML is loaded as-is,
including ``_include`` and ``_use`` (:doc:`concepts/loaders`). Two
exceptions are documented below.

**Your worker configuration does not port automatically.** A CARACal worker
is a YAML schema plus a Python module that reads it; shinobi has no worker
concept and no ``enable:`` machinery. What replaces both is a function that
builds a :class:`~shinobi.Recipe`. There is no converter, and this page is
not pretending otherwise -- it is a mapping guide, and the port is manual.

The honest summary: you are trading a declarative worker config for Python
you write yourself. That is a real cost, paid once per worker, and it buys
you ordinary control flow instead of a config language.

.. note::

   `caracal2 <https://github.com/caracal-pipeline/caracal2>`_ is the
   reference port and the most useful thing to read alongside this page. It
   keeps the CARACal worker *schemas* verbatim and rebuilds only the worker
   bodies as shinobi recipes -- which is the migration path this page
   describes, done for real across a dozen workers.

.. _migration-package-roots:

First: package-scoped ``_include`` needs ``package_roots``
----------------------------------------------------------

This is a **breaking change** and it will be the first thing you hit.

A schema that pulls in a shared base by package name --

.. code-block:: yaml

    libs:
      _include: (caracal.schemas)caracal_base.yaml

-- used to be resolved by *importing* ``caracal.schemas`` to find its
directory. shinobi no longer does that: importing a package named by a
config file executes its ``__init__.py``, which is arbitrary code execution
from data (see ``SECURITY.md``). Both loaders now resolve the dotted name
against an explicit mapping the caller supplies:

.. code-block:: python

    from pathlib import Path
    from shinobi.loaders.worker_schema import load_worker_schema

    import caracal.schemas

    SCHEMA_DIR = Path(caracal.schemas.__file__).parent
    ROOTS = {"caracal": SCHEMA_DIR.parent}

    schema = load_worker_schema(SCHEMA_DIR / "transform_schema.yaml", package_roots=ROOTS)

Without it you get a ``ConfigLoadError`` naming the package and telling you
what to pass.

This is not a weakening of the rule. The point is that *shinobi* must not
import a name it read out of a YAML file; your own code importing your own
package, deliberately, is fine. One entry covers a whole tree --
``resolve_package_root`` matches the longest registered prefix and descends
the remainder as subdirectories, so ``{"caracal": ...}`` resolves
``(caracal.schemas)`` and ``(caracal.anything.else)`` alike.

The same applies to ``shinobi.loaders.cultcargo.load_file``/``loads``, which
have always worked this way.

Two cab-level limitations to check first
----------------------------------------

Before porting anything, check your cab library against these -- both are
deliberate, and both are better discovered now than halfway through.

``dynamic_schema`` is not resolved
    A cab whose real schema comes from a Python function
    (``dynamic_schema: some.module.make_schema``) loads with a warning and
    whatever static ``inputs:``/``outputs:`` it happens to have, which may
    be **incomplete**. Real cult-cargo ``wsclean.yml``, ``cubical.yml`` and
    ``quartical.yml`` all use it. Hand-authored full ports of those three
    live in `dosho <https://github.com/shinobi-dosho/dosho>`_; prefer them
    over the cult-cargo originals.

Only ``binary`` cabs execute
    Code-carrying flavours -- ``flavour: python``, inline source -- are
    refused with ``UnsupportedFlavourError`` rather than run.
    ``msutils.copycol`` and ``bdsf.catalog`` are the ones CARACal users hit.
    dosho has native equivalents for the common cases.

Worker to Recipe, side by side
-------------------------------

Take CARACal's ``transform`` worker: split and average an MS, optionally
re-centre it, producing a new MS for downstream workers.

The configuration you write is unchanged -- shinobi loads that schema and
validates against it exactly as before:

.. code-block:: yaml

    transform:
      enable: true
      field: target
      tag: cal
      split_field:
        enable: true
        col: corrected
        time_avg: '8s'
        chan_avg: 4
      changecentre:
        enable: false

What changes is the worker body. In CARACal, ``enable:`` flags are consumed
by a worker module that appends stimela recipe steps as it walks the config.
In shinobi, they are just ``if`` statements in a function that returns a
``Recipe``:

.. code-block:: python

    import shinobi
    from shinobi import Recipe
    from dosho.cabs.casatasks import fixvis, mstransform


    def build_recipe(config) -> Recipe:
        """`config` is an instance of the schema's generated inputs model."""
        recipe = Recipe(
            name=schema.name,
            info=schema.info,
            inputs_model=schema.inputs_model,
            outputs_model=schema.outputs_model,
        )

        # The worker-level `enable:` flag is an early return, not a framework
        # concept. A disabled worker is an empty recipe.
        if not config.enable:
            return recipe

        current_ms = recipe.inputs.ms

        # Each segment's own `enable:` is an ordinary branch. The step is
        # only added to the graph when the branch is taken -- there is no
        # "declared but skipped" state to reason about.
        if config.split_field.enable:
            seg = config.split_field
            recipe.add_step(
                "split_field",
                mstransform,
                vis=current_ms,
                outputvis=produced_ms_name,
                field=recipe.inputs.field,
                datacolumn=seg.col,
                timeaverage=seg.time_avg not in ("", "0s", None),
                timebin=seg.time_avg or "0s",
                chanaverage=(seg.chan_avg or 1) > 1,
                chanbin=seg.chan_avg or 1,
            )
            current_ms = recipe.outputs.split_field.outputvis

        if config.changecentre.enable:
            recipe.add_step("changecentre", fixvis, vis=current_ms, phasecenter=config.changecentre.ra_dec)
            current_ms = recipe.outputs.changecentre.vis

        recipe.set_output("ms", current_ms)
        return recipe

The ``current_ms`` variable is doing the work that CARACal's
label/alias-propagation machinery does: each enabled segment rebinds it, so
the next step wires from whatever actually ran. That is the whole
substitution mechanism, and it is a local variable.

A second worker: ``flag``
-------------------------

``transform`` shows the shape. ``flag`` shows what it looks like at length,
and it is the more typical case: a dozen independent flagging segments, each
with its own ``enable:``, all operating on the same MS.

The config is again unchanged:

.. code-block:: yaml

    flag:
      enable: true
      field: calibrators
      flag_autocorr:
        enable: true
      flag_quack:
        enable: true
        interval: 8.0
        mode: beg
      flag_shadow:
        enable: true
        tol: 0.0
      flag_spw:
        enable: true
        chans: '*:850~900MHz'

and every segment collapses to the same four lines:

.. code-block:: python

    if config.flag_autocorr.enable:
        recipe.add_step("flag_autocorr", flagdata, vis=current_ms, mode="manual", autocorr=True, field=selected_field, flagbackup=False)
        current_ms = recipe.outputs.flag_autocorr.vis

    if config.flag_quack.enable:
        recipe.add_step(
            "flag_quack",
            flagdata,
            vis=current_ms,
            mode="quack",
            quackinterval=config.flag_quack.interval,
            quackmode=config.flag_quack.mode,
            field=selected_field,
            flagbackup=False,
        )
        current_ms = recipe.outputs.flag_quack.vis

    if config.flag_shadow.enable:
        recipe.add_step("flag_shadow", flagdata, vis=current_ms, mode="shadow", tolerance=config.flag_shadow.tol, field=selected_field, flagbackup=False)
        current_ms = recipe.outputs.flag_shadow.vis

There is no cleverness to find here, and that is the point -- porting a
worker is mechanical once the pattern is in hand. Three things are worth
noticing:

**"Executed in the same order in which they are given below."** CARACal's
``flag`` schema has to promise that in prose, because the order lives in the
framework. Here the order *is* the order you wrote the ``if`` statements in,
and ``current_ms`` threading through them is what makes it real rather than
conventional.

**A segment can depend on more than its own flag.** ``flag_time`` only runs
when it was enabled *and* actually given a range:

.. code-block:: python

    if config.flag_time.enable and config.flag_time.timerange:
        ...

In a config language that is a second schema rule someone has to express and
enforce. In Python it is ``and``.

.. _migration-inplace-outputs:

**Pass the mutated MS through as an output.** ``flagdata`` flags the MS *in
place* -- but its step declares ``vis`` as an **output** as well as an input,
so each segment wires from the previous one's ``vis`` rather than all of them
naming the same file independently.

Do this. It is the single most valuable habit to carry into a port, because
it turns an in-place mutation into a real edge in the graph:

* the scheduler orders the segments because they are genuinely dependent,
  not because they happen to be declared in that sequence;
* ``ninja run --dryrun`` shows the chain;
* the recipe stays offloadable to a cluster, where declaration order means
  nothing and only edges do (:doc:`offloading`).

The alternative -- every segment taking ``vis=recipe.inputs.ms`` and relying
on in-place mutation -- also works locally at the default
``max_workers: 1``, and shinobi will derive the ordering for you when
compiling to Slurm. But you get a graph that *says* the steps are
independent when they are not, and you are relying on inference where you
could simply have stated the dependency.

The mapping, item by item
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - CARACal / Stimela 2
     - shinobi
   * - A worker (schema + module)
     - A function returning a :class:`~shinobi.Recipe`
   * - ``enable: true`` on a worker
     - ``if not config.enable: return recipe``
   * - ``enable: true`` on a segment
     - An ``if`` around ``recipe.add_step(...)``
   * - Alias / label propagation between steps
     - A Python variable rebound as steps are added
   * - ``_include: (pkg)file.yaml``
     - Same syntax, but see :ref:`package_roots <migration-package-roots>`
   * - ``_use: dotted.path``
     - Unchanged
   * - ``=config.x.y`` substitution
     - Not implemented; it is ordinary Python here
   * - ``dynamic_schema:``
     - Not resolved -- use a dosho port
   * - ``flavour: python`` cabs
     - Refused -- use a dosho port
   * - Worker ordering in the config file
     - Declaration order, plus real wiring edges (:doc:`concepts/recipes`)
   * - Running the pipeline
     - ``ninja run recipe.py:name`` (:doc:`cli`)

Keeping your schemas
--------------------

You do not have to give up the schema files. ``load_worker_schema`` turns a
scabha-dialect worker schema into pydantic models
(:doc:`concepts/loaders`), so the config surface, its defaults, its
``choices`` and its ``info`` strings stay exactly where they are -- and you
get CLI options generated from them for free.

That is the incremental path, and the one caracal2 took: keep every schema,
port one worker body at a time, and leave the rest of the pipeline alone
while you do it.

What has no equivalent yet
--------------------------

Stated plainly, so you can judge before committing:

* **No resume-from-step.** The opt-in cache (:doc:`concepts/provenance`)
  approximates it by skipping steps whose inputs are unchanged, but there is
  no ``--start-from`` flag.
* **No per-step retry/backoff** for flaky cluster jobs.
* **No aggregated run report artifact** beyond the run manifest.
* **Offload assumes a shared filesystem.** There is no data-staging story
  for clusters without one (:doc:`offloading`).
* **Scatter is not offloadable** -- scattered recipes run locally.
