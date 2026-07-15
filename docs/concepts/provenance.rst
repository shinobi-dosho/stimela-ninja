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
        "inputs": { "ms": "data.ms", "prefix": "out" },
        "outputs": { "image": "out-MFS-image.fits" },
        "steps": []
      },
      "pinned": true
    }

A recipe's sub-steps appear, in declaration order, under ``steps``.

``pinned``
    ``true`` only when every step that ran *inside a container* resolved to a
    digest. A native step (whose ``image`` is mere metadata) never counts
    against it; a containerized step that couldn't be pinned -- including a
    Slurm job running under apptainer -- makes it ``false``. A consumer can
    refuse to treat a manifest with ``pinned: false`` as reproducible.

``image_digest``
    The ``sha256:...`` that actually ran, or ``null`` when unpinned.

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
