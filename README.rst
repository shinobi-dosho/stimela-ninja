stimela-ninja (Stimela 3.0)
===========================

Stimela 3.0 is a Python framework for reproducible radio astronomy pipelines.
It builds on `Stimela classic <https://github.com/ratt-ru/Stimela-classic>`_
with typed task inputs and outputs, Python recipes, and execution on local
machines, containers, and computing clusters.

Use it to compose radio astronomy tools into workflows, run independent tasks
concurrently, cache results, and record provenance for reproducible runs.
`dosho <https://github.com/shinobi-dosho/dosho>`_ provides ready-to-use task
definitions for tools such as WSClean, CASA, QuartiCal, and simms.

Academic attribution
--------------------

Please acknowledge this project and its contributors when using the work
in research, and cite the associated publications and software release
where applicable. This is a scholarly request, not an additional licence
condition.
Citation information can be found in `CITATION.md <CITATION.md>`_.

Installation
------------

Requires Python 3.11 or newer::

    pip install stimela-ninja

This installs the ``ninja`` command and the importable ``shinobi`` package.
For the companion task library::

    pip install dosho

External tools must be installed locally or available through the container
or cluster backend you use. See the `installation guide
<https://stimela-ninja.readthedocs.io/en/latest/installation.html>`_.

Quick start
-----------

Save this minimal Python step as ``myrecipe.py``:

.. code-block:: python

    from shinobi import pystep


    @pystep()
    def greet(name: str = "world") -> None:
        print(f"Hello, {name}!")

Run it from the command line; the function parameters become CLI options::

    ninja run myrecipe.py:greet --name astronomer

For an imaging workflow that combines multiple tools, follow the
`pipeline tutorial <https://stimela-ninja.readthedocs.io/en/latest/quickstart.html>`_.

Documentation
-------------

* `User guide <https://stimela-ninja.readthedocs.io/en/latest/>`_
* `Command-line reference <https://stimela-ninja.readthedocs.io/en/latest/cli.html>`_
* `Execution backends <https://stimela-ninja.readthedocs.io/en/latest/concepts/backends.html>`_
* `Migrating from CARACal or Stimela 2 <https://stimela-ninja.readthedocs.io/en/latest/migration.html>`_
* `Examples <https://github.com/shinobi-dosho/stimela-ninja/tree/main/examples>`_

Contributing
------------

Bug reports and contributions are welcome. See `CONTRIBUTING.md
<https://github.com/shinobi-dosho/stimela-ninja/blob/main/CONTRIBUTING.md>`_
for development setup and testing, and use the `issue tracker
<https://github.com/shinobi-dosho/stimela-ninja/issues>`_ for bugs and feature
requests. Report security issues as described in `SECURITY.md
<https://github.com/shinobi-dosho/stimela-ninja/blob/main/SECURITY.md>`_.

License
-------

Apache License 2.0 — see `LICENSE <LICENSE>`_ and `NOTICE <NOTICE>`_.
