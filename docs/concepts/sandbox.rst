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
  wrote those files straight to their absolute destination, same as an
  absolute declared output), and one that resolves to a ``..`` escape is
  skipped with a warning:

.. code-block:: python

    wsclean = Cab(
        name="wsclean", ...,
        sandbox=True,
        harvest=["{prefix}-*.fits"],   # the per-band/interval image family
    )

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
