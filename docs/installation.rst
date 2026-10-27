Installation
============

You need **Python 3.11 or newer**. These instructions use a Linux or macOS
terminal. You can install Stimela Ninja and explore its tools before setting
up containers or a cluster.

Install Stimela Ninja
---------------------

Create an environment for your pipeline work, then activate it. This keeps
its Python packages separate from your other projects:

.. code-block:: console

    $ python3 -m venv .venv
    $ source .venv/bin/activate
    $ pip install "stimela-ninja>=0.2.0b1"

If you already have a Python environment for this work, activate it and run
just the last command. The examples in these docs use Shinobi 0.2.0b1
and Dosho 0.2.0b2 or newer.

The installation includes the ``ninja`` command, the ``shinobi`` Python
package, and Dosho's ready-made tool definitions. Python dependencies are
installed automatically. The package to install is ``stimela-ninja``;
the unrelated package named ``shinobi`` on PyPI is a different project.

Check the installation
----------------------

.. code-block:: console

    $ ninja --help
    $ ninja cabs list
    $ ninja cabs show wsclean

These commands show help, list available tools, and display WSClean's
parameters. They do not run an astronomy tool or need any data.

If your terminal cannot find ``ninja``, activate the environment again with
``source .venv/bin/activate``. You also need to activate it when opening a
new terminal.

Prepare to run a tool
---------------------

Dosho contains tool definitions; the astronomy software itself is supplied
separately. Choose how you want to run it:

* **In a container:** Apptainer is the default. Its ``apptainer`` command
  must be available in your terminal. A cab with a container image can then
  use that image to run the tool.
* **Already installed on your computer:** select ``native`` with
  ``ninja --backend native run ...``. The tool's command must be available
  in your terminal.
* **Another container system or a cluster:** see :doc:`concepts/backends`
  for Docker, Podman and cluster setup, supported workflows and verification
  status.

You can change the default in :doc:`concepts/config`.
Once your tools are available, follow :doc:`quickstart` to build a small
imaging and masking pipeline.

When a recipe checks Measurement Sets or FITS files
---------------------------------------------------

Some recipes check the contents and structure of their data before and after
running a tool. Install the matching extra when a recipe requires these
checks. For Measurement Sets:

.. code-block:: console

    $ pip install "stimela-ninja[ms]>=0.2.0b1"

For FITS images and tables:

.. code-block:: console

    $ pip install "stimela-ninja[fits]>=0.2.0b1"

For both:

.. code-block:: console

    $ pip install "stimela-ninja[ms,fits]>=0.2.0b1"

The imaging and masking quickstart needs ``fits``. Recipes using strict
Measurement Set definitions from Dosho may need both extras.
A recipe that declares these checks will refuse to run if the required
inspection package is missing. Installing an extra does not turn on checks
for every file; the recipe's definitions select them.

Install extras in the environment that runs the checks. For remote Slurm
workflows, this includes the shared worker interpreter and the environments
that prepare and finalize the run. Installing them only inside an astronomy
tool's container is not enough. See :doc:`concepts/datasets` for supported
workflows and :doc:`concepts/states` for Measurement Set state operations.

For contributors
----------------

Source checkout setup, tests, documentation builds and Measurement Set state
development are covered in :doc:`contributing`.
