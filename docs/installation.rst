Installation
============

Requirements
------------

* Python 3.10 or newer
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

    $ pip install git+https://github.com/SpheMakh/stimela-ninja.git

Either way, this installs:

* the ``ninja`` command-line tool, and
* the importable ``shinobi`` package.

.. note::

   The distribution is named ``stimela-ninja`` on PyPI, the command is
   ``ninja``, and the import name is ``shinobi``.

For development
---------------

The project uses `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: console

    $ git clone https://github.com/SpheMakh/stimela-ninja.git
    $ cd stimela-ninja
    $ uv venv .venv && uv pip install -e . --group dev
    $ .venv/bin/pytest
    $ .venv/bin/ruff check src tests

To build the documentation locally:

.. code-block:: console

    $ uv sync --group docs
    $ uv run sphinx-build -b html docs docs/_build/html
    $ open docs/_build/html/index.html
