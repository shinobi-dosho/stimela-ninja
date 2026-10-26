Installation
============

Requirements
------------

* Python 3.11 or newer
* One or more execution backends available on ``PATH`` for the cabs you run
  (for example ``wsclean``, or a container runtime such as ``docker`` /
  ``podman`` / ``apptainer``, or a ``slurm`` cluster).

Apptainer is the default backend. Install its ``apptainer`` executable on
``PATH`` for containerized cabs, or select another backend with
``ninja --backend NAME`` or ``backend.default`` in :doc:`concepts/config`.

Install Shinobi
---------------

This documentation assumes Shinobi and Dosho 0.2.0b1 or newer:

.. code-block:: console

    $ pip install "stimela-ninja>=0.2.0b1"

This installs:

* the ``ninja`` command-line tool;
* the importable ``shinobi`` package; and
* ``dosho[validation]>=0.2.0b1,<0.3`` for cab schemas, parameter validation
  and argv formatting. Pydantic 2.14 or newer is required.

Dosho is a mandatory dependency, so its catalogue is available automatically.
Dosho also supports catalogue access without Shinobi; runnable cabs and pysteps
require the framework. See :doc:`design_cab_boundary` for the ownership contract.

.. note::

   The distribution is named ``stimela-ninja`` on PyPI, the command is
   ``ninja``, and the import name is ``shinobi``.

Optional format contracts
-------------------------

``MSv2``, ``FITSImage`` and ``FITSTable`` annotations are available in the base
package. Declaring them does not import an inspector or inspect files. To
execute their strict contracts, install the corresponding extra in the
environment performing lifecycle inspection:

.. code-block:: console

    $ pip install "stimela-ninja[ms]>=0.2.0b1"
    $ pip install "stimela-ninja[fits]>=0.2.0b1"
    $ pip install "stimela-ninja[ms,fits]>=0.2.0b1"

The extras name the supported formats. Their current implementations use
``python-casacore`` for Measurement Sets and Astropy for FITS; those packages
are implementation dependencies, while the annotations specify versioned
format profiles. A declared strict contract refuses execution if its inspector
is unavailable. See :doc:`concepts/datasets` for the supported profiles and
execution routes. A strict annotation opts a field into enforcement; installing
an extra alone does not make ordinary ``Path`` fields strict.

For Slurm compute workers, provision the extra in the shared worker interpreter
and in preparation/finalization environments that inspect files. Installing it
only in a tool image does not provide the controller or worker inspector.
Tool containers remain free to use other FITS libraries.

From a source checkout, use ``uv sync --extra ms`` or
``uv sync --extra fits``. The broader ``measurement-set`` development
group below also supplies the MS state export and reconstruction stack.

For development
---------------

Explicit MSv2 state operations currently require a source checkout and
``uv sync --group measurement-set``. The group requires
``msutils[exact-native]>=3.1.0`` for the preservation v2 public APIs, while
``uv.lock`` records the resolved release and complete stack
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
    $ uv run sphinx-build -E -n -W --keep-going -b html docs docs/_build/html
    $ open docs/_build/html/index.html
