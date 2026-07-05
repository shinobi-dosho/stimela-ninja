Cabs
====

A :class:`~shinobi.Cab` is a typed, backend-agnostic description of an atomic
task -- a single command with an inputs/outputs schema and *policies* for
turning parameters into a CLI invocation. It is the fundamental unit of work: a
recipe is just cabs (and other steps) wired together.

Defining a cab in Python
-------------------------

A cab needs a ``name``, the ``command`` to run, and pydantic models describing
its inputs and outputs. An optional ``image`` names the container the command
lives in (used by the container/cluster backends).

.. code-block:: python

    from pydantic import BaseModel

    from shinobi import Cab


    class ImageInputs(BaseModel):
        ms: str = "obs.ms"
        prefix: str = "img"


    class ImageOutputs(BaseModel):
        restored: str | None = None


    wsclean = Cab(
        name="wsclean",
        command="wsclean",
        image="quay.io/stimela/wsclean:latest",
        inputs_model=ImageInputs,
        outputs_model=ImageOutputs,
    )

Fields with a default are optional; fields without one are required. The dtype
comes from the type hint.

Building models from a compact spec
-----------------------------------

Hand-writing a pydantic model per cab is verbose. The same helper the YAML
loaders use, :func:`shinobi.loaders.build_model`, builds one from a
``{name: (dtype, required, default)}`` mapping:

.. code-block:: python

    from shinobi.loaders import build_model

    inputs = build_model("MaskInputs", {"restored_image": ("File", True, None)})
    outputs = build_model("MaskOutputs", {"mask": ("File", False, None)})

``File`` and ``MS`` dtypes are meaningful beyond typing: the container and
cluster backends inspect them to decide which paths to bind-mount.

Turning parameters into argv
----------------------------

How a cab's parameters become command-line arguments is controlled by its
``policies`` and per-field ``field_meta``. For example, mark a parameter as
positional (passed as a bare argument rather than ``--flag value``) with a
:class:`~shinobi.steps.schema.ParamMeta`:

.. code-block:: python

    from shinobi.steps import ParamMeta

    touch = Cab(
        name="make",
        command="/bin/touch",
        inputs_model=build_model("TouchInputs", {"out": ("File", True, None)}),
        outputs_model=build_model("PathOutputs", {"out": ("File", False, None)}),
        field_meta={"out": ParamMeta(positional=True)},
    )

See :class:`shinobi.Cab` and :class:`shinobi.steps.schema.Policies` in the
:doc:`API reference <../api/index>` for the full set of knobs (prefixes,
repeat policies, ``nom_de_guerre`` renaming, input patterns, and output
wranglers).

Loading cabs from YAML
----------------------

You do not have to define cabs in Python. Existing `cult-cargo
<https://github.com/caracal-pipeline/cult-cargo>`_ YAML is loaded as-is -- see
:doc:`loaders`.

Inspecting a cab
----------------

The ``ninja cab`` command dumps a loaded cab's resolved schema as JSON, which
is handy for checking how a YAML definition was interpreted:

.. code-block:: console

    $ ninja cab cabs.yml wsclean
