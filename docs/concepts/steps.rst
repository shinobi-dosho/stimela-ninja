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

Container-only imports: ``ctx.import_callable()`` / ``ctx.import_module()``
---------------------------------------------------------------------------

A pystep declared with ``image=`` runs *inside* that container when a container
backend is resolved; the module defining it, however, is imported on the
**host** every time a recipe is built. That split makes a top-level import of a
tool package a problem: the package lives in the image, not in the host
environment, so ``from casacore.tables import table`` at module scope raises
``ImportError`` on the host long before the step ever runs -- and trips linters
and type checkers there for the same reason.

:meth:`ExecContext.import_callable <shinobi.ExecContext.import_callable>`
defers the import to execution time, where the package really exists. Give a
pystep a leading ``ctx`` parameter and resolve the names inside the body:

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
        table = ctx.import_callable("table", "casacore.tables")
        SkyCoord = ctx.import_callable("SkyCoord", "astropy.coordinates")

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
``import_callable`` has resolved nothing at all.

The signature is ``ctx.import_callable(name, module=None)``:

* With ``module``, it imports that module and returns the named attribute --
  ``importlib.import_module(module)`` followed by ``getattr(module, func)``.
  ``module`` is the **full dotted path**, however deep, so
  ``ctx.import_callable("getheader", "astropy.io.fits")`` is the right spelling
  for an attribute of a submodule. Splitting the path at the package boundary
  -- ``ctx.import_callable("fits", "astropy.io")`` -- asks for the ``fits``
  *module* as though it were an attribute of ``astropy.io``, which is not what
  the caller meant and is rejected.
* Without ``module``, it looks ``name`` up in :mod:`builtins`, so
  ``ctx.import_callable("print")`` and ``ctx.import_callable("len")`` work.

**Callable, not function.** What a pystep asks for here is a class about as
often as a function -- ``casacore.tables.table``, ``astropy.wcs.WCS``,
``astropy.coordinates.SkyCoord`` -- and numpy hands back neither:
``numpy.log10`` is a ``ufunc`` and ``numpy.linspace`` an
``_ArrayFunctionDispatcher``. Callability is the one property they share, and
the one the body is about to rely on, so it is the one checked: a non-callable
result raises ``TypeError`` naming the fix.

That check is what makes the split point safe rather than lucky. Whether a
package boundary split raises on its own depends on the package:
``("tables", "casacore")`` and ``("fits", "astropy.io")`` raise
``AttributeError``, but ``("ndimage", "scipy")`` quietly returns the *module*
on any scipy new enough to expose its subpackages lazily -- a wrong-shaped
object the body then carries to its first call site.

``ctx.import_func`` is the former name of this method, kept as an alias so
existing pysteps keep working; it delegates, check included.

When the body wants the **module object itself** rather than one of its
attributes, use :meth:`ExecContext.import_module
<shinobi.ExecContext.import_module>`, which is plain
``importlib.import_module`` deferred to execution time in exactly the same way:

.. code-block:: python

    @pystep(image="quay.io/stimela/casa:latest")
    def clip(ctx, ms: Path) -> ClipOutputs:
        np = ctx.import_module("numpy")
        fits = ctx.import_module("astropy.io.fits")
        ...

.. important::

   ``import_callable`` returns a **callable attribute of** a module, never a
   module, and its one-argument form is a :mod:`builtins` lookup, not an
   import: ``ctx.import_callable("numpy")`` raises ``AttributeError: module
   'builtins' has no attribute 'numpy'``. Reach for
   ``ctx.import_module("numpy")`` there. The two methods are kept separate so
   each has one return type, rather than having one of them return a callable
   or a module depending on what the name happens to be.

.. note::

   Inside the container the runner stubs out ``shinobi``, ``pydantic`` and the
   step's own top-level package, so those never load there. Only stdlib and
   whatever the body pulls in through ``import_callable``/``import_module`` are
   real -- one more reason tool imports belong in the body rather than at
   module scope. All three import methods (``import_callable``,
   ``import_func``, ``import_module``) are lifted into that in-container shim
   from the real ``ExecContext``, so they behave there exactly as they do on
   the host.

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

Two further fields matter only inside a recipe:

``after``
    Step names this one must run after, with **no data flowing**. Ordinary
    wiring already orders steps, so reach for this only when a step must not
    start before another has finished yet reads nothing from it.
    :ref:`add_loop <declared-loops>` is the motivating case: an iteration's
    first step reads nothing from its own iteration, so without an ordering
    edge nothing would stop it starting before the previous iteration had
    decided whether the loop had converged.

``loop``
    Set by ``add_loop`` on the steps it generates, recording which iteration a
    step belongs to and which sentinel it should check. Bookkeeping for the
    short-circuit decision only -- the edges that make an unrolled loop a real
    DAG are ordinary ``wiring`` and ``after``, never this.
