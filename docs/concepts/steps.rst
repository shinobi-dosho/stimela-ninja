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

Container-only imports: ``ctx.import_func()``
----------------------------------------------

A pystep declared with ``image=`` runs *inside* that container when a container
backend is resolved; the module defining it, however, is imported on the
**host** every time a recipe is built. That split makes a top-level import of a
tool package a problem: the package lives in the image, not in the host
environment, so ``from casacore.tables import table`` at module scope raises
``ImportError`` on the host long before the step ever runs -- and trips linters
and type checkers there for the same reason.

:meth:`ExecContext.import_func <shinobi.ExecContext.import_func>` defers the
import to execution time, where the package really exists. Give a pystep a
leading ``ctx`` parameter and resolve the names inside the body:

.. code-block:: python

    from pathlib import Path

    from pydantic import BaseModel

    from shinobi import pystep


    class PhaseCentre(BaseModel):
        ra_deg: float
        dec_deg: float
        sexagesimal: str


    @pystep(image="quay.io/stimela/casa:latest")
    def phase_centre(ctx, ms: Path, field_id: int = 0) -> PhaseCentre:
        """Read a field's phase centre from an MS and format it for humans."""
        table = ctx.import_func("table", "casacore.tables")
        SkyCoord = ctx.import_func("SkyCoord", "astropy.coordinates")

        field = table(f"{ms}::FIELD", ack=False)
        try:
            ra_rad, dec_rad = field.getcol("PHASE_DIR")[field_id][0]
        finally:
            field.close()

        coord = SkyCoord(ra=ra_rad, dec=dec_rad, unit="rad")
        return PhaseCentre(
            ra_deg=float(coord.ra.deg),
            dec_deg=float(coord.dec.deg),
            sexagesimal=coord.to_string("hmsdms"),
        )

Neither ``python-casacore`` nor ``astropy`` needs to be installed on the host:
the host only imports this module to *build* the recipe, and by then
``import_func`` has resolved nothing at all.

The signature is ``ctx.import_func(func, module=None)``:

* With ``module``, it imports that module and returns the named attribute --
  ``importlib.import_module(module)`` followed by ``getattr(module, func)``.
* Without ``module``, it looks ``func`` up in :mod:`builtins`, so
  ``ctx.import_func("print")`` and ``ctx.import_func("len")`` work.

.. important::

   ``import_func`` returns an **attribute of** a module, never a module. There
   is no one-argument form for pulling in a package: ``ctx.import_func("numpy")``
   does not import numpy, it looks for ``numpy`` in ``builtins`` and raises
   ``AttributeError: module 'builtins' has no attribute 'numpy'``. Name the
   attribute you actually want -- ``ctx.import_func("array", "numpy")``,
   ``ctx.import_func("getheader", "astropy.io.fits")`` -- or bind the module
   through one of its own attributes if you need several.

The name is historical: the returned object need not be a function. Classes
(``table`` above), and any other module attribute, resolve the same way.

.. note::

   Inside the container the runner stubs out ``shinobi``, ``pydantic`` and the
   step's own top-level package, so those never load there. Only stdlib and
   whatever the body pulls in through ``import_func`` are real -- one more
   reason tool imports belong in the body rather than at module scope.

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

A ``StepRef`` may also carry a ``scatter`` specification, so a single step can
fan out over one or more list inputs when it is part of a recipe. See
:doc:`recipes`.
