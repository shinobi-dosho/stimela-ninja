Results and output wrangling
============================

Every call returns a :class:`~shinobi.results.StepResult`. Backends first
produce a schema-free :class:`~shinobi.results.BackendRun` containing an exit
status and captured streams; dispatch then validates the effective inputs,
builds the scope's declared output model, and attaches execution metadata.
Recipes return the same type as cabs and Python steps, so callers do not need a
second result API for composite work.

Reading a result
----------------

``outputs`` and ``inputs`` are pydantic model instances. Output fields are also
available directly on the result for convenience:

.. code-block:: python

    result = image(ms="obs.ms", prefix="run1")

    if not result.success:
        raise RuntimeError(result.stderr)

    assert result.restored == result.outputs.restored
    print(result.inputs.prefix)
    print(result.returncode, result.stdout, result.stderr)

``success`` means exactly ``returncode == 0``. A negative return code denotes a
signal on POSIX; command-line failure messages name the signal and identify
``SIGKILL`` as the common symptom of an exceeded memory limit.

How cab outputs are filled
--------------------------

A cab does not scrape arbitrary files after it runs. For each field in its
``outputs_model``, dispatch uses the first available source in this order:

#. a value extracted by an output wrangler;
#. a same-named effective input (the common pass-through shape for an in-place
   Measurement Set operation);
#. the reserved ``returncode``, ``stdout``, or ``stderr`` value;
#. a ``ParamMeta.implicit`` string, resolved with plain ``str.format`` against
   the cab's validated inputs;
#. the pydantic field default.

The completed mapping is validated by ``outputs_model``. A missing required
output or a value of the wrong type is therefore a ``ParameterError`` rather
than a partially populated result. Outputs that may genuinely be absent when a
tool fails should be optional or have a default, so the original non-zero exit
can still be reported as a ``StepResult``.

Implicit outputs are useful when a tool derives a predictable filename from an
input stem:

.. code-block:: python

    from shinobi.steps import ParamMeta

    image = Cab(
        ...,
        outputs_model=build_model(
            "ImageOutputs", {"restored": ("File", False, None)}
        ),
        field_meta={
            "restored": ParamMeta(implicit="{prefix}-MFS-image.fits")
        },
    )

This is deliberately only ``str.format`` over the current step's inputs. It is
not an expression language and cannot reference another step; inter-step data
flow belongs in ``InputRef``/``OutputRef`` wiring.

Extracting values from console output
-------------------------------------

A wrangler maps a regular expression to one or more actions. The implemented
action is ``PARSE_OUTPUT:<named-group>:<type>``; supported casts are ``str``,
``int``, ``float``, and ``bool`` (using Python truthiness, so any non-empty capture is true; an
unknown type falls back to ``str``):

.. code-block:: python

    class Progress(BaseModel):
        percentage: float | None = None


    flagger = Cab(
        name="flagger",
        command="flagger",
        inputs_model=Inputs,
        outputs_model=Progress,
        wranglers={
            r"Flagged: (?P<percentage>[0-9.]+)%": [
                "PARSE_OUTPUT:percentage:float"
            ]
        },
    )

Patterns are searched against every captured stdout line followed by every
captured stderr line. Later matches overwrite earlier values for the same
field. Display-only Stimela/scabha actions such as ``HIGHLIGHT``, ``SUPPRESS``,
and ``SEVERITY`` are not implemented and are ignored; malformed
``PARSE_OUTPUT`` actions are rejected.

Capture limits preserve lines matching the cab's wrangler patterns even when
the middle of a chatty stream is elided. If that retained-match safety ceiling
is itself exceeded, the run warns that a structured output may be missing. See
:doc:`config` for ``log.capture_head_lines`` and ``log.capture_tail_lines``.

Recipe results
--------------

A recipe result contains its validated recipe-level outputs and a
``sub_results`` mapping keyed by declared step name. Recipe ``stdout`` and
``stderr`` are the child streams concatenated in declaration order, regardless
of which parallel step finished first. Output wiring and the winning failure
are resolved in that same stable order.

On the first child failure the scheduler stops submitting new steps and drains
work already running. The recipe result uses the first non-zero return code by
declaration order. A worker exception is re-raised after in-flight work drains,
also choosing the first one by declaration order.

Execution metadata
------------------

``StepResult`` also records whether work was served from cache (``cached``),
short-circuited by a declared loop (``skipped``), or run in a sandbox
(``sandboxed``), plus backend, container image/digest, virtualenv digest,
resource declaration, and nested results. These fields feed cache provenance
and run manifests; :doc:`provenance` explains which of them constitute a
reproducibility guarantee.
