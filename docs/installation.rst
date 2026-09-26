Installation
============

Requirements
------------

* Python 3.11 or newer
* One or more execution backends available on ``PATH`` for the cabs you run
  (for example ``wsclean``, or a container runtime such as ``docker`` /
  ``podman`` / ``apptainer``, or a ``slurm`` / ``kubernetes`` cluster).

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

Explicit MSv2 state operations need ``pip install '.[state]'`` in a
development checkout, or ``uv sync --group measurement-set``. This pins
xarray-ms, xarray, Zarr, arcae, numcodecs and ``msutils[exact-native]`` at
commit ``91675064fbe466369a775286a8e35fccb591d883``. Nominal msutils 3.0.0
alone does not establish preservation v2 capability. Zarr is 3.1.6 on Python
3.11 and 3.3.0 on newer Python; the lock supplies the complete tested stack.
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
