Provenance
==========

Provenance makes a run **reproducible**: it pins every container image to a
content digest before running, and writes a static manifest recording exactly
what ran -- resolved inputs and outputs, the backend, and the pinned image
digest of each step.

It is **opt-in and off by default**, because pinning changes how containers
execute (see :ref:`pin-then-run` below). Enable it per invocation, per call,
or in config:

.. code-block:: console

    $ ninja run myrecipe.py:selfcal --ms data.ms --provenance

.. code-block:: python

    result = my_step(ms="data.ms", provenance=True)   # StepRef call

.. code-block:: yaml

    # ~/.shinobi/config.yml
    provenance:
      enabled: true
      dir: ".shinobi/runs"

When enabled, two things happen together: images are digest-pinned, and one
run manifest is written per top-level run.

.. _pin-then-run:

Pin-then-run
------------

With provenance on, a container image is resolved to its registry digest
*before* it runs, and the executed reference is rewritten to
``repo@sha256:...``. What executes is therefore exactly what the manifest
records -- a floating tag like ``:latest`` can't drift between the record and
the run.

This is a genuine behaviour change, which is why it is opt-in:

* the run executes ``repo@sha256:...`` instead of ``repo:tag``;
* it needs a registry round-trip to resolve the digest before running;
* if your local ``:latest`` differs from the registry's, the registry image
  is what runs.

With provenance **off** (the default), images run by their original tag with
no registry round-trip -- exactly as a plain container run always has.

The digest is resolved best-effort, in order:

#. a built-in, dependency-free registry API client (reads the manifest digest
   over HTTPS; uses credentials from ``~/.docker/config.json`` for private
   repositories);
#. ``skopeo inspect``, if installed;
#. ``docker buildx imagetools inspect``, for the docker/podman runtimes.

A local ``.sif`` file is content-hashed directly. An image that can't be
resolved (a local-only build, an offline host) runs **unpinned**, and the
manifest reports that honestly rather than inventing a digest.

The run manifest
----------------

One manifest is written per top-level run to ``provenance.dir``
(default ``.shinobi/runs``), named ``<name>.<utc-timestamp>.<pid>.run.json``.
It freezes the resolved run as a tree of steps:

.. code-block:: json

    {
      "schema_version": 1,
      "shinobi_version": "0.1.0b1",
      "target": "myrecipe.py:selfcal",
      "generated_at": "2026-07-13T14:07:50Z",
      "backend": "docker",
      "returncode": 0,
      "root": {
        "name": "image",
        "kind": "cab",
        "backend": "docker",
        "image": "quay.io/stimela/wsclean:latest",
        "image_digest": "sha256:6baf435...",
        "containerized": true,
        "sandboxed": false,
        "inputs": { "ms": "data.ms", "prefix": "out" },
        "outputs": { "image": "out-MFS-image.fits" },
        "steps": []
      },
      "pinned": true
    }

A recipe's sub-steps appear, in declaration order, under ``steps``.

``pinned``
    ``true`` only when every step that ran *inside a container* resolved to a
    digest **and** no step ran in a venv. A containerized step that couldn't be
    pinned -- including a Slurm job running under apptainer -- makes it
    ``false``, as does any ``venv`` step. A consumer can refuse to treat a
    manifest with ``pinned: false`` as reproducible.

    Provenance durability is **tiered**, and ``pinned`` is a claim about
    container image digests, not cross-machine reproducibility in general:

    * ``docker`` / ``podman`` / ``apptainer`` with provenance enabled --
      **durable**: the exact image digest that ran is recorded and re-run.
    * ``venv`` -- a **version-parity record only**: ``venv_digest`` is a hash of
      the venv's ``name==version`` list, which is not an OS-level pin (identical
      version lists can sit on different compiled C-extensions), so a venv step
      is always reported *unpinned*.
    * ``native`` -- **no environment provenance at all**: an ``image`` is mere
      metadata, so a native step never drags ``pinned`` false, but its vacuous
      ``pinned: true`` is *not* a reproducibility guarantee.

``image_digest``
    The ``sha256:...`` that actually ran, or ``null`` when unpinned.

``venv`` / ``venv_digest``
    The virtualenv a ``venv``-backend step ran in, and a ``sha256`` of its
    ``name==version`` distribution list (recorded only under provenance). The
    digest is informational -- a venv step is always unpinned (see ``pinned``
    above), so replaying a manifest that contains one is refused unless
    ``--allow-unpinned`` is passed, exactly like a digest-less container step.

``sandboxed``
    ``true`` when the step ran with per-step sandbox execution enabled
    (see :doc:`sandbox`), otherwise ``false``. This is recorded for
    diagnostics; it does not affect whether a manifest can be replayed.

``skipped``
    ``true`` for a :ref:`loop <declared-loops>` iteration that ran after the
    loop had already converged, so it passed the previous iteration's outputs
    through instead of doing work. This is what makes the manifest an exact
    record of how many cycles a run actually performed -- the graph, and
    ``--dryrun``, only show how many were *declared*. Distinct from
    ``cached``: nothing was looked up, and ``kind`` still reports what the
    step is.

``target``
    The CLI target string (``path/to/file.py:name`` or ``pkg.mod:name``) that
    produced the run, recorded so :ref:`ninja replay <ninja-replay>` can find
    the recipe again. ``null`` for programmatic runs (a ``StepRef`` called
    from Python) and for manifests written before the field existed.

``stimela_version`` / ``cab_repo_commit`` are reserved and currently ``null``.

Replaying a run
---------------

A manifest is not just a record -- it can be re-run:

.. code-block:: console

    $ ninja replay .shinobi/runs/selfcal.20260713T140750Z.12345.run.json

Replay loads the manifest's ``target``, forces every containerized step's
image to the recorded ``repo@sha256:...`` (an already-pinned reference passes
through pin-then-run with no registry round-trip), and re-runs with the
recorded inputs on the recorded backend. The replay is itself a provenance
run, so it writes a fresh manifest of what it ran.

What replay guarantees -- and what it refuses:

* **Declared resources are recorded, not restored.** Each step's
  ``resources`` footprint (see :doc:`recipes`) is written into the manifest,
  because a bare ``returncode -9`` months later tells you nothing while
  ``-9`` beside ``memory=200GiB`` is a diagnosis. Replay deliberately does
  *not* re-apply it: a footprint describes the machine a run happened on, not
  the run itself, so replaying elsewhere uses that machine's own declaration.
* **Unpinned manifests are refused.** A manifest with ``pinned: false``
  cannot promise the same images run again, so replay errors, naming the
  unpinned steps; ``--allow-unpinned`` proceeds anyway, running those steps
  by their original reference.
* **A changed recipe is an error.** Manifest steps are matched to the
  recipe's steps by name; a step that has since been removed, renamed, or
  added makes replay refuse rather than run something other than what the
  manifest froze. This also means a failed or interrupted run (whose
  manifest omits never-reached steps) cannot be replayed exactly.
* **The source still matters.** The manifest pins images and inputs, not
  code: replay re-imports the target file, so it reproduces the original run
  only against the same checkout (the reserved ``cab_repo_commit`` field is
  where that will eventually be recorded). Orchestration functions
  (``@shinobi.step`` bodies) re-execute; any nondeterminism inside them is
  outside the manifest's guarantee.
* **Lossy inputs may not replay.** Non-serializable inputs (e.g. a MUTABLE
  field holding a live Python object) are stored in the manifest as strings;
  if they no longer validate against the recipe's inputs model, replay
  reports that rather than guessing.

``--target`` supplies the target for manifests that don't record one, and the
global ``ninja --backend`` flag overrides the recorded backend (e.g. when
replaying a Slurm run on a laptop with docker).

Cleaning up
-----------

Manifests accumulate under ``provenance.dir``. Remove them (and the step
cache) with :ref:`ninja clean <ninja-clean>`:

.. code-block:: console

    $ ninja clean --no-cache        # just the run manifests
    $ ninja clean --dry-run         # preview without deleting
