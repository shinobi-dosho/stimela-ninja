Loaders
=======

For executable cabs, ``shinobi`` reuses definitions from two established
formats, each producing the same :class:`~shinobi.Cab` objects you would build
by hand. A related loader builds pydantic models from CARACal worker config
schemas without pretending those schemas are executable cabs.

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


Worker/config schemas
---------------------

:func:`shinobi.loaders.worker_schema.load_worker_schema` handles the related
scabha dialect used by CARACal/caracal2 worker *configuration*. It returns a
``ConfigSchema`` with generated ``inputs_model`` and ``outputs_model`` classes,
not a ``Cab``: a config schema validates values but contains no command to
dispatch.

.. code-block:: python

    from pathlib import Path

    import yaml

    from shinobi.loaders.worker_schema import load_worker_schema


    schema = load_worker_schema(
        "schemas/crosscal_schema.yaml",
        package_roots={"caracal": Path("schemas").resolve().parent},
    )
    raw = yaml.safe_load(Path("pipeline.yml").read_text())["crosscal"]
    config = schema.inputs_model.model_validate(raw)
    recipe = build_crosscal_recipe(config)

Nested schema groups become nested pydantic models. ``choices`` become
``Literal`` annotations and are enforced during validation. Both
``List[T]``/``list:T`` and nested ``Tuple[...]``/``Union[...]`` dtype forms
are understood; file-like dtypes become :class:`pathlib.Path`. Parameter names
containing hyphens or dots are sanitized to Python identifiers, with collisions
rejected rather than silently overwriting one field.

``_include`` and ``_use`` share the cab loader's resolution helpers and safety
rules. Package-scoped includes require an explicit ``package_roots`` mapping;
the loader never imports a package named by YAML. Included paths are contained
inside the registered package root, including nested include chains.

Config-supplied mapping keys with ``_each``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ordinary groups declare all their keys in the schema. An ``_each`` group
declares one value schema while allowing the config to choose the mapping
keys, which is useful for named calibration chains:

.. code-block:: yaml

    inputs:
      chains:
        _key_pattern: '^[A-Za-z][A-Za-z0-9_]*$'
        _each:
          solver:
            dtype: str
            choices: [gaincal, bandpass]
          enabled:
            dtype: bool
            default: true
        default:
          primary:
            solver: gaincal

This produces ``dict[str, ChainModel]``. Set ``_key_pattern`` whenever keys
become step names or path components, so an invalid name fails at config load
rather than much later in a run. Defaults are validated when the schema loads
and rebuilt for every model instance, so mutable mappings are never shared.
Only ``_key_pattern``, ``info``, and ``default`` may accompany ``_each``;
misspelled or inapplicable keys are errors.

The generated CLI deliberately skips ``_each`` fields: there is no fixed set
of names from which to construct options. Supply those mappings in the config
file or programmatically.

Inspecting the result
---------------------

Whichever loader you use, ``ninja cab`` dumps the resolved schema as JSON so
you can confirm how a definition was interpreted:

.. code-block:: console

    $ ninja cab cabs.yml wsclean
