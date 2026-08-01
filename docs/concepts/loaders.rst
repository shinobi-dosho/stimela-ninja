Loaders
=======

You do not have to define cabs in Python. ``shinobi`` reuses existing cab
definitions from two established formats, each producing the same
:class:`~shinobi.Cab` objects you would build by hand.

YAML cabs (the scabha dialect)
-------------------------------

shinobi's cab schema is **borrowed from scabha**, the schema library
underneath Stimela 2.0. The vocabulary is deliberately scabha's --
``inputs``/``outputs`` with ``dtype``/``required``/``default``/``info``/
``choices``, plus ``policies``, ``management.wranglers``, ``image``,
``flavour`` and ``command`` -- so loading a scabha cab is a translation, not an
interpretation. What shinobi drops is the layer *above* the cab: stimela2's
recipe, alias and expression machinery.

`cult-cargo <https://github.com/caracal-pipeline/cult-cargo>`_ is the largest
published library of cabs written in this dialect, and is what the loader is
usually pointed at -- but the dialect is scabha's, and nothing in the loader is
specific to that project.

:func:`shinobi.loaders.yaml_cab.load_file` reads a YAML file and returns a
``{name: Cab}`` mapping:

.. code-block:: python

    from shinobi.loaders.yaml_cab import load_file

    cabs = load_file("cabs.yml")
    wsclean = cabs["wsclean"]

Use :func:`shinobi.loaders.yaml_cab.loads` to parse from a string instead of a
file.

What is supported
~~~~~~~~~~~~~~~~~

Support is **deliberately partial**: the static, declarative subset is read,
and the parts that are a programming language wearing YAML are refused.

Implemented, verified against real upstream cab files:

* ``_include`` -- file composition, resolved wherever it appears in the
  document, not only at the top level;
* ``_use`` -- dotted-path deep-merge;
* package-scoped ``_include`` (``(pkg.dotted.path)file.yaml``) -- resolved
  against a **caller-supplied** ``package_roots`` mapping. shinobi never
  imports a cab package to find its data directory, which would execute
  arbitrary ``__init__.py`` code; see ``SECURITY.md``.

Not implemented, and not by omission
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each of these is a point where scabha stops describing a tool and starts
computing something:

* **Expressions and substitutions** (``=config.x.y``, ``=recipe.ms``,
  ``${...}``, ``=IFSET(...)``) -- kept as literal strings, so a value carrying
  one is visible in the built :class:`~shinobi.Cab` rather than silently
  dropped. The one templating shinobi *does* resolve is
  ``ParamMeta.implicit``, and it is plain ``str.format`` against the step's own
  validated inputs: no cross-step name resolution, no calls, no conditionals.
* **Conditionals and control flow** -- a cab is a parameter table. Branching
  over it belongs in the Python that calls the step, where it is visible to the
  reader and to the DAG.
* **Aliases and value propagation** between recipe and step level -- shinobi
  wires steps with typed :class:`~shinobi.InputRef`/:class:`~shinobi.OutputRef`
  objects, so there is nothing to propagate, and no need for the expression
  language that propagation forces into existence.
* **``dynamic_schema``** -- a dotted reference to a Python function that would
  have to be imported *and called* to produce the cab's real schema. A cab
  using it loads with a warning and whatever static ``inputs:``/``outputs:``
  it carries. See the module docstring and ``SECURITY.md``.

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
