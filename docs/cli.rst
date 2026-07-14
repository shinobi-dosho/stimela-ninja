Command-line interface
======================

The ``ninja`` command is the primary way to run cabs and recipes. It takes
global options followed by a subcommand:

.. code-block:: console

    $ ninja [--config FILE] [--backend NAME] COMMAND ...

Global options
--------------

``--config FILE``
    Path to a config file (default: ``~/.shinobi/config.yml``). See
    :doc:`concepts/config`.

``--backend NAME``
    Override the default backend for this invocation.

Targets
-------

Commands that act on a cab or recipe take a **target** of the form
``path/to/file.py:name`` or ``dotted.module.path:name``. The name must resolve
to a ``Cab``, ``Recipe``, or a ``@shinobi.step``-decorated function.

``ninja run`` -- run a target
-----------------------------

Runs a ``Cab``, ``Recipe``, or step. The target's own parameters become the
command's options -- run ``ninja run TARGET --help`` to see them.

.. code-block:: console

    $ ninja run myrecipe.py:image --ms data.ms --prefix out
    $ ninja run myrecipe.py:selfcal --ms data.ms

Add ``--dryrun`` to render the execution graph without running anything:

.. code-block:: console

    $ ninja run myrecipe.py:selfcal --ms data.ms --dryrun
    [ image ]
        |
        v
    [ mask ]

The dry run executes the recipe's real Python control flow with every cab
swapped for a no-op that records the call, so it shows the one path taken for
the given inputs -- never an untaken branch.

Add ``--cache-dir DIR`` / ``--no-cache`` to control step-level result caching
(a step must also opt in via its own ``Scope.cache``, an enclosing recipe's,
or ``AppConfig.cache.enabled`` -- these flags alone don't turn caching on):

.. code-block:: console

    $ ninja run myrecipe.py:selfcal --ms data.ms --cache-dir /scratch/cache
    $ ninja run myrecipe.py:selfcal --ms data.ms --no-cache

By default, running cabs' stdout/stderr are echoed live as they run
(native/container backends only). Add ``--quiet`` to restore the old
behavior of a silent run followed by one dump of captured output at the end;
this overrides ``AppConfig.log.stream`` for the invocation.

Add ``--provenance`` to make the run reproducible: container images are
digest-pinned before running and a run manifest is written under
``AppConfig.provenance.dir``. It's off by default (``--no-provenance`` forces
it off), and overrides ``AppConfig.provenance.enabled`` for the invocation.
See :doc:`concepts/provenance`.

.. code-block:: console

    $ ninja run myrecipe.py:selfcal --ms data.ms --provenance

Add ``--remote user@host:/path`` to launch on a remote host instead of
locally: the target file and its statically-discoverable cab deps are synced
over, then the run happens detached -- check progress with ``ninja status``.
``--add-venv/--no-add-venv`` (default: on) sources ``venv/bin/activate`` or
``.venv/bin/activate`` under the remote path first, if present.
``--include PATH`` (repeatable) syncs extra files/dirs alongside the target,
for orchestration code the static cab-dep scan can't see:

.. code-block:: console

    $ ninja run myrecipe.py:selfcal --ms data.ms --remote user@cluster:/scratch/run1
    $ ninja run myrecipe.py:selfcal --ms data.ms --remote user@cluster:/scratch/run1 --include extra_cabs.yml

``ninja cab`` -- inspect a cab schema by file
----------------------------------------------

Dumps a cab's resolved schema (as loaded from a cult-cargo style YAML file) as
JSON:

.. code-block:: console

    $ ninja cab cabs.yml wsclean

``ninja cabs`` -- look up installed cabs by name
--------------------------------------------------

Looks up cabs by name across installed ``shinobi.cabs`` providers (e.g.
``dosho``), instead of pointing at a specific YAML file:

.. code-block:: console

    $ ninja cabs list
    $ ninja cabs show wsclean

``ninja download`` -- fetch cab definitions
---------------------------------------------

Downloads cab definitions for use with the file-based ``ninja cab`` /
cult-cargo loader. ``--cult-cargo`` downloads cab definitions from GitHub;
``--dest-dir`` sets the destination (default: ``.shinobi/cabs/cultcargo``);
``--version`` picks ``latest`` (highest ``v*`` tag), a tag, a branch, or a
commit SHA:

.. code-block:: console

    $ ninja download --cult-cargo
    $ ninja download --cult-cargo --version v1.2.3 --dest-dir .shinobi/cabs/cultcargo

``ninja compile`` -- offload a recipe
-------------------------------------

Compiles a purely-declarative recipe into a cluster workflow and, with
``--submit``, hands it off and detaches. See :doc:`offloading`.

.. code-block:: console

    $ ninja compile myrecipe.py:pipe --target /scratch/made.ms --container-runtime none
    $ ninja compile myrecipe.py:pipe --target /scratch/made.ms --submit

Options: ``--engine`` (workflow engine, ``slurm`` in v1), ``--workdir``
(working directory for compiled jobs), ``--container-runtime`` (runtime to wrap
imaged cabs in; ``none`` for bare argv), and ``--submit`` (submit and detach).

.. _ninja-clean:

``ninja clean`` -- remove runtime artifacts
-------------------------------------------

Removes shinobi's runtime artifacts: run manifests (``AppConfig.provenance.dir``),
the step cache (``AppConfig.cache.dir``), and detached-run launch dirs
(``.shinobi/<recipe>/``, holding the handle file and Slurm job logs written by
``ninja compile --submit`` / ``ninja run --remote``). ``--dry-run`` previews
what would be removed without deleting.

Run manifests and the step cache are removed by default; narrow the
selection with ``--no-runs`` / ``--no-cache``. Launch dirs are the opposite:
**off by default**, opt in with ``--launches`` -- deleting one doesn't stop a
still-running detached job, but it does destroy ``ninja status``'s only local
record of it, so it isn't swept as part of a routine clean. ``--workdir DIR``
picks where to look for launch dirs (default: cwd); it has no effect on
``--runs``/``--cache``, which always come from the active config. Nothing
outside those targets is touched.

.. code-block:: console

    $ ninja clean                   # run manifests + step cache
    $ ninja clean --no-cache        # just run manifests
    $ ninja clean --dry-run         # preview
    $ ninja clean --launches        # + all detached-run launch dirs under cwd
    $ ninja clean --no-runs --no-cache --launches --workdir /scratch/run1

``ninja status`` -- check a detached run
----------------------------------------

Reports a detached offloaded run's progress from the handle file written by
``ninja compile --submit`` or ``ninja run --remote``, querying the engine
fresh (no persistent process):

.. code-block:: console

    $ ninja status /scratch/.shinobi/pipe/handle.json

``ninja version`` -- print the version
--------------------------------------

.. code-block:: console

    $ ninja version
