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

``stimela_version`` / ``cab_repo_commit`` are reserved and currently ``null``.

Cleaning up
-----------

Manifests accumulate under ``provenance.dir``. Remove them (and the step
cache) with :ref:`ninja clean <ninja-clean>`:

.. code-block:: console

    $ ninja clean --no-cache        # just the run manifests
    $ ninja clean --dry-run         # preview without deleting
