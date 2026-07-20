Design philosophy
==================

stimela-ninja (the ``shinobi`` package) is a spiritual successor to `Stimela
classic <https://github.com/ratt-ru/Stimela-classic>`_, built in direct
reaction to `Stimela 2.0 <https://github.com/caracal-pipeline/stimela>`_'s
YAML-recipe complexity. This page is the "why" behind the architecture
described in :doc:`concepts/recipes` and the other concept pages.

Recipes are declared DAGs
--------------------------

A ``Recipe`` is a data structure, not a running program: a list of
``StepRef``\ s with explicit wiring (``InputRef``/``OutputRef``) declaring
how data flows between steps. Because the graph is data, it is statically
inspectable -- renderable and validatable before anything runs (see
``ninja run --dryrun`` in :doc:`cli`).

String-keyed step references are deliberately allowed -- the graph is
*data*, and names are its natural addressing. This is not a return to YAML
orchestration: the graph is built in Python, not parsed from a markup file,
and carries no expression language or control-flow semantics beyond the
declared edges.

What we deliberately left out
------------------------------

Stimela 2.0's YAML recipe layer grew several kinds of complexity that
stimela-ninja refuses to reintroduce:

* **A string-based expression/substitution language** for referencing other
  steps' params or outputs (Stimela 2.0's ``=recipe.ms``,
  ``{recipe.name}-{info.suffix}``). Wiring uses typed ``InputRef``/
  ``OutputRef`` objects with explicit field names instead of string
  templates -- a typo in a field name is a validation error at graph-build
  time, not a runtime string-substitution failure.
* **An alias-propagation system.** Stimela 2.0 needs multi-pass up/down
  propagation logic to keep step- and recipe-level params in sync, plus glob
  re-evaluation hacks to work around it. That entire class of problem only
  exists because YAML was the orchestration layer -- a declared Python graph
  doesn't need it.
* **A YAML-based way to express control flow.** A thin YAML-to-``Recipe``
  *compiler* for simple linear pipelines may be worth adding later, but it
  would have to compile into the same declared graph, not grow its own
  semantics.

Before adding a feature
-------------------------

Ask whether Stimela classic or Stimela 2.0 already solved the problem, and
which one solved it *simply*. If neither did, keep the new piece as small
and boring as possible -- this project refuses to adopt complexity that
isn't earning its keep.

Cab schemas are reused, the recipe layer isn't
-------------------------------------------------

The `cult-cargo <https://github.com/caracal-pipeline/cult-cargo>`_ cab
*schema* (inputs/outputs/policies/wranglers) is good design and is loaded
as-is (:doc:`concepts/loaders`) -- it's Stimela 2.0's recipe/alias layer
that gets dropped, not the cab format. Loading existing cult-cargo cab
definitions unlocks the whole existing radio-astronomy tool library instead
of requiring a rewrite.

See also
--------

* :doc:`concepts/recipes` for how the declared graph is built, validated,
  and executed.
* :doc:`security` for the threat model around loading cab definitions from
  arbitrary files.
* :doc:`contributing` for the day-to-day conventions contributors follow.
