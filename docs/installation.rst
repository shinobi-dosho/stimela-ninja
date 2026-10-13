Installation
============

Requirements
------------

* Python 3.11 or newer
* One or more execution backends available on ``PATH`` for the cabs you run
  (for example ``wsclean``, or a container runtime such as ``docker`` /
  ``podman`` / ``apptainer``, or a ``slurm`` cluster).

From PyPI
---------

.. code-block:: console

    $ pip install stimela-ninja

From GitHub
-----------

To install the latest development version:

.. code-block:: console

    $ pip install git+https://github.com/shinobi-dosho/stimela-ninja.git

Either way, this installs:

* the ``ninja`` command-line tool, and
* the importable ``shinobi`` package.

.. note::

   The distribution is named ``stimela-ninja`` on PyPI, the command is
   ``ninja``, and the import name is ``shinobi``.

For development
---------------

Explicit MSv2 state operations currently require a source checkout and
``uv sync --group measurement-set``. The group tracks
``msutils[exact-native]`` from the upstream Git ``main`` branch, while
``uv.lock`` records the resolved commit and complete stack for reproducible
local installs. Refresh upstream with ``uv lock --upgrade-package msutils``
then ``uv sync --locked --group measurement-set``. The direct Git reference
lives in the unpublished development group because PyPI rejects it in wheel
metadata. Nominal msutils versions alone do not establish preservation v2
capability; state operations require its public APIs and upstream Git
provenance and record the exact installed commit.

The other qualified packages remain pinned: xarray-ms, xarray, arcae,
numcodecs, and Zarr (3.1.6 on Python 3.11 and 3.3.0 on newer Python).
Measurement Set CI refreshes msutils main on every integration run and hourly
after this workflow is merged, making upstream regressions visible promptly.
See :doc:`concepts/states` for the profile and local Linux requirement.

The project uses `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: console

    $ git clone https://github.com/shinobi-dosho/stimela-ninja.git
    $ cd stimela-ninja
    $ uv sync --group dev
    $ .venv/bin/pytest
    $ .venv/bin/ruff check .
    $ .venv/bin/ruff format --check .

``uv.lock`` is committed, so ``uv sync`` gives you the same dependency versions
CI tests against (it runs every job with ``--locked``). Change
``pyproject.toml`` and you must re-run ``uv lock`` and commit both; the repo's
pre-commit hook and CI each reject the mismatch. See ``CONTRIBUTING.md``.

To build the documentation locally:

.. code-block:: console

    $ uv sync --group docs
    $ uv run sphinx-build -b html docs docs/_build/html
    $ open docs/_build/html/index.html
