Stimela Ninja
=============

**Stimela Ninja** helps you turn radio astronomy processing tasks into
repeatable pipelines. Connect tools such as WSClean, CASA and QuartiCal in a
Python recipe, then run it from the command line.

Preview the steps before running, reuse cached results, and keep a record
of how a run was made. Tools can run on your computer, in containers, or on
a computing cluster. `Dosho <https://github.com/shinobi-dosho/dosho>`_ is
included and provides ready-made tool definitions, called **cabs**.

Academic attribution
--------------------

Please acknowledge this project and its contributors when using the work
in research, and cite the associated publications and software release
where applicable. This is a scholarly request, not an additional licence
condition.
Citation information can be found in `CITATION.cff <CITATION.cff>`_.

Installation
------------

Requires Python 3.11 or newer::

    pip install "stimela-ninja>=0.2.0b1"

This includes the ``ninja`` command, the ``shinobi`` Python package, and
Dosho. Install ``stimela-ninja`` even though Python examples import
``shinobi``. Python dependencies are installed automatically.

The `installation guide
<https://stimela-ninja.readthedocs.io/en/latest/installation.html>`_ walks
through creating a Python environment, checking the installation, and
setting up astronomy tools. Apptainer is the default way to run tools in
containers; you can also use software installed on your computer.

Quick start
-----------

Save this minimal Python step as ``myrecipe.py``:

.. code-block:: python

    from shinobi import pystep


    @pystep()
    def greet(name: str = "world") -> None:
        print(f"Hello, {name}!")

Run it from the command line::

    ninja --backend native run myrecipe.py:greet --name astronomer

This prints ``Hello, astronomer!``. The ``native`` option runs the example
on your computer, so you can try it without containers or astronomy data.
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
