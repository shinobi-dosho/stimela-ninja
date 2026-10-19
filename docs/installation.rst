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
``uv sync --group measurement-set``. The group requires
``msutils[exact-native]>=3.1.0``, the first release with the preservation v2
public APIs, while ``uv.lock`` records the resolved release and complete stack
for reproducible local installs. Refresh it with
``uv lock --upgrade-package msutils`` then
``uv sync --locked --group measurement-set``. The group is unpublished, so
the optional Measurement Set stack stays out of a plain install. State
operations check that msutils supplies the preservation v2 public APIs and
accept two provenances: an ordinary release install, which records the msutils
version, or a Git install of the upstream repository, which records the exact
commit. Local paths, editable installs and other repositories are refused.

The other qualified packages remain pinned: xarray-ms, xarray, arcae,
numcodecs, and Zarr (3.1.6 on Python 3.11 and 3.3.0 on newer Python).
Measurement Set CI refreshes to the latest msutils release on every
integration run and daily, making regressions in a new release visible
promptly.
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
