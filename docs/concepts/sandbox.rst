Sandboxed execution
===================

Radio-astronomy tools are messy neighbours: they drop logfiles, ``*.last``
files, and scratch products into whatever directory they run in. Sandboxed
execution keeps the workspace clean by running each step with its working
directory inside a **private scratch directory**; when the step succeeds,
only its *declared* outputs are moved back to the workspace, and the scratch
directory -- with all the junk -- is deleted.

This is an allowlist, not a blocklist. There is no per-tool inventory of
junk to sweep up: anything the step didn't declare as an output simply does
not survive. "Fully-defined I/O" is enforced by construction.

It is **opt-in and off by default**. Enable it per invocation, per scope,
per call, or in config -- the same precedence chain as caching (call-time
argument > the scope's own ``sandbox`` > the enclosing recipe's > config):

.. code-block:: console

    $ ninja run myrecipe.py:selfcal --ms data.ms --sandbox

.. code-block:: python

    cab = Cab(name="wsclean", ..., sandbox=True)      # on the scope
    result = my_step(ms="data.ms", sandbox=True)      # or per call

.. code-block:: yaml

    # ~/.shinobi/config.yml
    sandbox:
      enabled: true
      dir: ".shinobi/work"

How a sandboxed step runs
-------------------------

#. A fresh scratch directory is created under ``sandbox.dir`` (default
   ``.shinobi/work``, relative to the invocation cwd) and becomes the
   tool's working directory -- the subprocess ``cwd`` for the native
   backend, the ``-w``/``--pwd`` workdir for container backends.
#. **Inputs are never copied in.** Path-typed inputs (the same fields that
   drive container bind mounts) are rewritten to absolute paths anchored at
   the workspace, so the tool reads -- and, for ``MUTABLE`` inputs like a
   measurement set, writes -- the caller's real files in place.
#. **Parent directories of relative outputs are pre-created** inside the
   sandbox -- from declared output values (including resolved ``implicit``
   templates) and the literal directory prefix of each ``harvest`` glob.
   Tools generally don't ``mkdir -p`` their own output stems (wsclean's
   ``-name img/run1``, ragavi's ``htmlname``), so without this a relative
   output like ``plots/gain.html`` that works in the workspace would crash
   in the empty sandbox.
#. The tool runs; relative outputs land inside the sandbox.
#. On success, declared outputs are **harvested**: moved (by rename -- the
   scratch root lives on the workspace's filesystem precisely so this is
   never a copy) back to the workspace at their declared relative paths,
   parents before anything nested inside them. Pre-created directories the
   tool never wrote into are removed first, so only what the tool actually
   produced comes back. Everything else is deleted with the sandbox.
#. On failure, nothing is harvested and the sandbox is deliberately *kept*
   for post-mortem; a warning reports its path. ``ninja clean`` removes
   leftover sandboxes (it targets ``sandbox.dir`` by default).

What survives the sandbox
-------------------------

Two declarations feed the harvest allowlist:

* every **path-typed output field**, at its resolved value (including
  ``implicit`` templates like ``"{prefix}-MFS-image.fits"``); an absolute
  output path bypasses the sandbox entirely -- the tool writes it straight
  to its declared destination;
* the scope's ``harvest`` globs, for dynamically-named output families that
  can't be enumerated as literal fields. Patterns are resolved against the
  step's own inputs; a pattern that resolves absolute is skipped (the tool
  wrote those files straight to its absolute destination, same as an
  absolute declared output), and one that resolves to a ``..`` escape is
  skipped with a warning:

.. code-block:: python

    wsclean = Cab(
        name="wsclean", ...,
        sandbox=True,
        harvest=["{prefix}-*.fits"],   # the per-band/interval image family
    )

.. _clearing-stale-outputs:

Re-running replaces the previous product
----------------------------------------

Harvest gives a relative output more than tidiness: the tool writes into a
fresh scratch dir and the new product is *moved over* the destination, so a
re-run replaces what the last run left. An output the tool writes straight to
its destination -- an absolute path, a **path-typed input** naming a
destination (which sandboxing anchors at the workspace), or any output at all
when sandboxing is off -- never went through that. It landed on top of the
previous run's product. CASA-family tools (``mstransform``, ``split``,
``importuvfits``) check the output for existence and refuse, failing the step
until a human moves the product aside; tools that append or merge instead
succeed and produce something corrupt.

So before a step runs, shinobi deletes the previous run's product from each
declared output path the tool writes directly
(``sandbox.clear_stale_outputs``, on by default, off via
``execution.clear_stale_outputs``). Each removal is logged with the path and
the declaration it came from.

Two things are never cleared:

* **Paths the step reads.** Compared resolved and by containment, since an MS
  is a directory: an output resolving inside a declared input, or a declared
  output directory that *holds* one, is the caller's data either way.
* **``harvest``/``scratch`` glob matches** -- the tool chose those names at
  run time, so they can collide with workspace data the step knows nothing
  about. Only declared output *fields* are cleared. Same reason harvest
  refuses to replace an undeclared directory.

That leaves one shape the framework cannot read off the declarations. An
output field that echoes a same-named input looks identical whether the tool
*created* that path or *rewrote the caller's data* in it:

.. code-block:: python

    # both: one name on inputs_model and on outputs_model
    mstransform(vis=..., outputvis="obs_mst.ms") -> outputvis   # created here
    flagdata(vis="obs.ms")                       -> vis         # the caller's MS

The default reading is the safe one -- an echoed path is the caller's data and
is never deleted -- and a cab whose output really is its own product says so
once, in its definition:

.. code-block:: python

    Cab(name="mstransform", ...,
        field_meta={"outputvis": ParamMeta(write_path=True)})

    @shinobi.pystep(image=CASA6, write_paths=["outputvis"])
    def mstransform(vis: Path, outputvis: Path) -> MstransformOutputs: ...

``write_path`` changes nothing about how the value is handled (its other,
older job is declaring that a *string* input is an output stem, so a cab that
names no write target for it fails to build); it is simply the declaration of
which of those two a path is. Note the
cost of clearing before the tool starts: a re-run that then fails leaves
neither product. That is unavoidable -- a tool that refuses to overwrite has
already failed by the time anything could harvest -- and Tier 1 snapshots
restore it when they are enabled.

Sandboxed state and cache portability
-------------------------------------

The ``StepResult`` records whether a step ran sandboxed in its
``sandboxed`` boolean field. The flag is carried through the step cache and
written into the run manifest's ``StepRecord`` so provenance records can
show which steps executed inside a scratch directory.

Because sandboxing rewrites relative path inputs to absolute workspace paths
before the tool runs, the same step can produce absolute or relative output
paths depending on whether sandboxing was enabled. Before a result is cached,
path-typed outputs are normalized back to workspace-relative paths, so a
later cache hit is valid regardless of whether the original run used a
sandbox. Absolute outputs requested explicitly by the caller are left
unchanged.

Limits, by design
-----------------

* **In-process pysteps are never sandboxed** -- ``os.chdir`` is
  process-global and recipes run steps on a thread pool. Containerized
  pysteps (``@shinobi.pystep(image=...)``, e.g. CASA tasks) sandbox fine,
  and they are the messiest offenders anyway.
* **Junk written next to an input escapes.** A tool that drops
  ``<ms>.flagversions`` beside the measurement set writes into the real
  workspace, because the MS necessarily lives there. If such a by-product
  matters, declare it as an output.
* The ``slurm``/``kubernetes`` backends accept and ignore the sandbox cwd
  (the job runs in a remote/pod working directory shinobi can't scope), so
  a sandboxed step degrades gracefully to an unsandboxed run there.
* Concurrent steps each get their own sandbox, so parallel recipe steps
  can no longer see each other's droppings at all.

Where possible, prefer *prevention* too: a cab whose tool can be told not
to write a logfile at all (``--no-log-file`` flags, CASA's
``casalog.setlogfile``) should bake that into its definition -- stdout and
stderr are always captured on the ``StepResult`` regardless, so console
output is never lost.
