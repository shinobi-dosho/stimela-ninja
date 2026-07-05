Steps
=====

A **step** binds an orchestration function to a *scope* (a ``Cab``, a
``Recipe``, or a bare :class:`~shinobi.Scope`). There are two decorators for
producing one, depending on what you start from.

``@shinobi.step`` -- wrap an existing cab or recipe
---------------------------------------------------

Use :func:`@shinobi.step <shinobi.step>` when you already have a
``Cab``/``Recipe`` and want to make it runnable (and, optionally, drive it with
custom Python). The decorated function receives an
:class:`~shinobi.ExecContext` as its first argument and calls ``ctx.run()`` to
execute the underlying scope:

.. code-block:: python

    from shinobi import Cab, step

    @step(wsclean, backend="native")
    def image(ctx):
        """A near-empty body just runs the cab."""
        return ctx.run()

The decorated function's own signature is *never* introspected --
``scope.inputs_model`` is the schema authority. ``ctx.run()`` accepts a
``backend=`` override and per-call input overrides; returning its
``StepResult`` (or ``None`` to auto-run) hands control back to the engine.

``@shinobi.pystep`` -- turn a plain function into a step
--------------------------------------------------------

Use :func:`@shinobi.pystep <shinobi.pystep>` when you have an ordinary,
type-hinted Python function and no external command. Its input schema is
derived from the function's parameters, and its output schema from the return
annotation:

.. code-block:: python

    from pydantic import BaseModel

    from shinobi import pystep


    class Sum(BaseModel):
        total: float


    @pystep
    def add(a: float, b: float) -> Sum:
        return Sum(total=a + b)

A ``BaseModel`` return annotation is used directly (the function must return an
instance of it). No annotation, or ``-> None``, means the step has no outputs.
Any *other* return annotation is rejected at decoration time -- there is no
implicit wrapping of a bare scalar into an invented field name.

.. note::

   ``typing.get_type_hints`` resolves annotations against the function's own
   module globals, so any ``BaseModel`` used in the signature or return type
   must be defined at module level, not nested inside another function.

Which to use
------------

* ``@shinobi.step`` -- you have an existing ``Cab``/``Recipe`` (an external
  tool, or a composite pipeline).
* ``@shinobi.pystep`` -- you have a plain Python function and want it to
  participate as a step without hand-writing pydantic models.

Both return a :class:`~shinobi.StepRef`: a named, executable binding. There is
no global function registry -- the function travels on the ``StepRef`` itself,
so two functions over one scope never collide. A ``StepRef`` is a valid
``ninja run`` target and a valid recipe step.
