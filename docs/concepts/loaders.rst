Loaders
=======

You do not have to define cabs in Python. ``shinobi`` reuses existing cab
definitions from two established formats, each producing the same
:class:`~shinobi.Cab` objects you would build by hand.

cult-cargo YAML
---------------

`cult-cargo <https://github.com/caracal-pipeline/cult-cargo>`_'s schema format
is good design and is reused as-is. :func:`shinobi.loaders.cultcargo.load_file`
reads a YAML file and returns a ``{name: Cab}`` mapping:

.. code-block:: python

    from shinobi.loaders.cultcargo import load_file

    cabs = load_file("cabs.yml")
    wsclean = cabs["wsclean"]

Use :func:`shinobi.loaders.cultcargo.loads` to parse from a string instead of a
file.

The loader implements cult-cargo's composition mechanisms, verified against
real upstream cab files:

* ``_include`` -- file composition, and
* ``_use`` -- dotted-path deep-merge.

The ``=config.x.y`` expression language and package-scoped includes are
**deliberately not implemented** -- control flow belongs in Python, not in the
cab definition. See the module docstring and ``AGENTS.md`` for the rationale.

Stimela classic parameter files
--------------------------------

:func:`shinobi.loaders.stimela_classic.load_file` reads a Stimela classic
``parameters.json`` and returns a single :class:`~shinobi.Cab`:

.. code-block:: python

    from shinobi.loaders.stimela_classic import load_file

    cab = load_file("casa_listobs/parameters.json")

Inspecting the result
---------------------

Whichever loader you use, ``ninja cab`` dumps the resolved schema as JSON so
you can confirm how a definition was interpreted:

.. code-block:: console

    $ ninja cab cabs.yml wsclean
