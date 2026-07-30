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

.. _declaring-where-a-tool-writes:

Declaring where a tool writes
-----------------------------

Many tools name their output *family* with a single stem parameter -- wsclean's
``prefix``, ddfacet's ``Output-Name`` -- from which they derive a dozen actual
files. Declare that stem as a plain ``str``, not as a ``File``: a path dtype
would be rewritten to an absolute workspace path when the step runs
:doc:`sandboxed <sandbox>`, and the tool would then write its family outside
the sandbox where harvest cannot see it.

That leaves the *outputs* side to say where the products land, and it must say
so, because nothing else can:

.. code-block:: python

    wsclean = Cab(
        name="wsclean",
        command="wsclean",
        image="quay.io/stimela/wsclean:latest",
        inputs_model=build_model("In", {"prefix": ("str", True, None)}),
        outputs_model=build_model("Out", {"restored_image": ("File", False, None)}),
        field_meta={"restored_image": ParamMeta(implicit="{prefix}-MFS-image.fits")},
        harvest=["{prefix}-*.fits"],  # the rest of the family
    )

Not everything a tool writes is a product, though. A cache tree, a scratch
directory, a tool logfile: those must be *writable* -- so the container
backends have to mount them -- but they must not follow the products back out
of a sandbox into the caller's workspace. Declare those with ``scratch``,
which has the same shape as ``harvest`` and the opposite effect on rescue:

.. code-block:: python

    ddfacet = Cab(
        ...,
        harvest=["{output_name}.*"],       # products: mounted, and rescued
        scratch=["{cache_dir}/*"],         # cache: mounted, never rescued
    )

An ``implicit`` template on a ``File``-dtype output, or a ``harvest`` or
``scratch`` glob, is what tells shinobi that ``prefix`` names a write target.
All three are resolved against the step's own inputs *before* the run, and
drive real behaviour: the sandbox pre-creates the directories they imply, and
the container backends bind-mount them so a write outside the working
directory reaches the host instead of dying inside the container. A cab that
declares none of them is taken at its word -- a bare ``str`` stem is just a string, and a value pointing
somewhere no declaration mentions gets no mount.

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

Positional args come after every flagged/pattern-matched arg, in
field-declaration order -- the right spot for tools that take flags then a
trailing bare value (e.g. simms' ``ms``). Some tools instead only recognise a
positional as their very first argument (``argv[1]``), never as a trailing
leftover -- CubiCal and killMS both only look at ``sys.argv[1]`` for a parset
file. For those, use ``ParamMeta(positional_head=True)`` instead: it emits
the value as a bare argument *before* every flag. Head and tail positionals
can be mixed on the same cab; each group keeps its own field-declaration
order.

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

To look a cab up by name across installed ``shinobi.cabs`` providers (e.g.
`dosho <https://github.com/shinobi-dosho/dosho>`_) instead of pointing at
a specific YAML file, use ``ninja cabs show``/``ninja cabs list``:

.. code-block:: console

    $ ninja cabs list
    $ ninja cabs show wsclean
