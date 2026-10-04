Loaders
=======

For executable cabs, ``shinobi`` reuses definitions from two established
formats, each producing the same :class:`~shinobi.Cab` objects you would build
by hand. A related loader builds pydantic models from CARACal worker config
schemas for validating configuration.

YAML cabs (the scabha dialect)
-------------------------------

shinobi's cab schema uses the scabha dialect, also used by Stimela 2.0.
It supports the same vocabulary --
``inputs``/``outputs`` with ``dtype``/``required``/``default``/``info``/
``choices``, plus ``policies``, ``management.wranglers``, ``image``,
``flavour`` and ``command``. The loader translates static cab definitions into
shinobi objects; recipes are declared separately in Python.

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

The loader supports static cab definitions and the composition mechanisms
listed below. Expression evaluation and executable schema generation are
outside its supported scope.

Implemented, verified against real upstream cab files:

* ``_include`` -- file composition, resolved wherever it appears in the
  document, not only at the top level;
* ``_use`` -- dotted-path deep-merge;
* package-scoped ``_include`` (``(pkg.dotted.path)file.yaml``) -- resolved
  against a **caller-supplied** ``package_roots`` mapping. shinobi never
  imports a cab package to find its data directory, which would execute
  arbitrary ``__init__.py`` code; see ``SECURITY.md``.

Unsupported features
~~~~~~~~~~~~~~~~~~~~

The following features are not evaluated by the loader:

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
  objects to declare parameter sources and data dependencies explicitly.
* **``dynamic_schema``** -- a dotted reference to a Python function that would
  have to be imported *and called* to produce the cab's real schema. A cab
  using it loads with a warning and whatever static ``inputs:``/``outputs:``
  it carries. See the module docstring and ``SECURITY.md``.

Strict dataset declarations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Case-insensitive ``dtype: MSv2`` and ``dtype: CasaTab`` preserve the same
versioned dataset metadata as their Python annotations. The compatibility
spellings ``MeasurementSetV2`` and ``CasaTable`` do likewise. Legacy ``MS``
remains a plain ``Path``, including ``List[MS]``. ``MSv4`` is reserved and
unsupported, even inside a list, tuple or union.

Cab-level ``dataset_accesses`` is a list of mappings validated by
:class:`~shinobi.DatasetAccess`. For example:

.. code-block:: yaml

    cabs:
      flag:
        command: flag-tool
        inputs:
          data.ms:
            dtype: MSv2
            required: true
            mutable: true
        dataset_accesses:
          - field: data_ms
            mode: write
            table: MAIN
            columns:
              read: [DATA]
              write: [FLAG]

``field`` and ``root_field`` use literal sanitized model names: ``data.ms``
becomes ``data_ms``. The loader does not rewrite references or propagate
aliases. Malformed declarations report the cab and ``dataset_accesses[i]``.
``_include`` and ``_use`` replace access lists when overridden; they do not
append or merge individual entries. An absent or null list means no explicit
access declarations.

Omitting accesses on a mutable ``MSv2`` input is valid and conservatively
reserves a whole-dataset write. Ordinary inputs infer reads and new outputs
infer creates. An explicit access list may refine a mutable field, but cannot
declare only reads for it; construction refuses that contradiction. Path and
alias contradictions that require resolved filesystem identities are checked
by planning.
Column creation/removal requires ``allow_schema_change: true``; creating a
new dataset with ``mode: create`` alone does not require that flag.
MAIN writers may specify ``allow_subtable_change: [QUALITY_BASELINE_STATISTIC]``
to permit changes to named opaque keyword-linked tables; generic schema or
keyword permission does not permit those changes. See :doc:`datasets`.

A cab may set ``cache: true``, ``cache: false`` or ``cache: null``. Omission
and null inherit the existing cache policy; false makes the cab execute on
every run, including when a containing recipe enables caching. ``_include``
and ``_use`` preserve this setting with ordinary override precedence. A direct
call-time cache argument overrides the cab setting. Strict MSv2 writers retain
their requirement for cache-backed recovery and refuse explicit false.

Column names can be templated from the cab's own validated string inputs:

.. code-block:: yaml

    inputs:
      ms:
        dtype: MSv2
        required: true
      column:
        dtype: str
        default: MODEL_DATA
    dataset_accesses:
      - field: ms
        mode: write
        allow_schema_change: true
        columns:
          create: ["{column}"]

Templates use direct sanitized input names, with no attribute/index access,
conversion or format specification. The same syntax works on Python-authored
``DatasetColumns``. Missing or ``None`` values use whole-dataset column intent;
resolved names are validated and recorded. See :doc:`datasets` for the full
contract, including existing columns named under ``create``.

Executable YAML cabs support direct strict dataset scalars, including
optional scalars with a ``None`` default, and direct read-only input
``List[MSv2]`` (also ``list:MSv2``). Optional list fields may default to ``None``;
concrete lists must be non-empty. One base ``dataset_accesses`` read declaration
applies to every element. List outputs, explicit or inferred writes and creates,
other strict containers, mixed unions or nested models, strict dynamic patterns,
and strict ``choices`` are refused. See :doc:`datasets` for indexed records,
closure overlap refusals and the multi-root execution boundary. Worker/config schemas may retain strict composite annotations.
Legacy ``MS`` containers, patterns and choices retain their existing behavior.
``CasaTab`` can be loaded and structurally inspected, but current dispatch
supports only the bounded MSv2 lifecycle. Any ``CasaTab`` field in an atomic
input or output is refused at execution, including a subtable with
``root_field``. A strict cab must also enter through the top-level dataset
lifecycle: dispatch refuses strict declarations beneath a cache path or an
already claimed workspace rather than letting nested execution bypass claims,
validation or recovery. See :doc:`datasets` for the execution boundary.

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
