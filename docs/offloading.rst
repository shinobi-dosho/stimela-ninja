Offloading to a cluster
=======================

A recipe that is *purely declarative* can be compiled to a cluster workflow and
handed off, so the pipeline runs without a live ``ninja`` process babysitting
it. This is what ``ninja compile`` does.

Shared-storage worker bundles
-----------------------------

The supported shared-storage path provides a **plan and result protocol**
through ``shinobi.offload.bundle`` and ``shinobi.offload.records``, plus an
opt-in short-lived compute worker.
The legacy argv compiler remains the default. ``ninja compile --worker
--submit`` stages and submits the worker lifecycle; ``--worker`` without
``--submit`` is refused because immutable source and environment staging is a
submission-preparation side effect.

``freeze_recipe(recipe, inputs, config=config, workspace=workspace)`` builds a
version-1 ``RecipeBundle`` without writing files, launching tools, importing
captured user code or resolving network-dependent image pins. It records:

* the declared flat DAG in declaration order, including explicit wiring,
  ordering edges, recipe outputs and unrolled-loop bookkeeping;
* reconstructible input/output models, field/path metadata, command policies,
  wranglers, output declarations, harvest, scratch and resource settings;
* validated recipe inputs, step constants and the resolved configuration
  snapshot, plus each step's selected tool backend and shared tool-venv path.

Known step inputs are checked during freezing. Inputs wired from a producing
step remain references, **not fabricated values**; the worker must validate
them once that producer commits its result. Relative paths retain their
spelling and are interpreted against the recorded absolute shared workspace,
never against the submission directory. The initial deployment assumes that
workspace, cache and snapshot paths are visible identically on all nodes;
node-local staging is separate work.

The worker eligibility check explicitly recognizes the generated
``PystepCallable`` adapter, not arbitrary functions with a ``__wrapped__``
attribute. Binary cabs and image-/venv-backed pysteps have distinct execution
specifications. Arbitrary orchestration functions, nested recipes, scatter,
local/nested callables and closures are refused. A pystep cannot silently
fall back to native/in-process execution. Tool venvs must already exist at
their resolved shared paths; submission preparation must additionally verify
compute-node compatibility. They are separate from the version-pinned worker
environment and remain **unpinned**, even with a package-version digest.

For pysteps, supply explicit ``code_roots=(Path(...),)`` Python import roots.
The bundle captures the source module, package initializers and local helpers
reachable through literal imports, without importing those modules. Use
``include_modules=("package.helper",)`` for dynamically selected local helpers.
Imports outside those roots are execution-environment requirements, not files
to discover by importing an installed package. This is a source snapshot,
not serialization of a live interpreter: runtime monkey-patching, mutable
module state and dynamically generated dependencies are not supported.
The caller is responsible for declaring all dynamically selected helpers.
Captured sources are trusted executable code under the same trust boundary as
the original Python recipe; merely reading a bundle never executes them.
The callable's module address must match its entry path within the supplied
root (``pkg/mod.py`` means ``pkg.mod``; ``pkg/__init__.py`` means ``pkg``).
Alias-loaded modules and ``__main__`` callables are rejected; import the
callable by its canonical module name before freezing it.

The bundle's SHA-256 covers captured source and helper contents as well as
the declaration. ``bundle.stage(shared_root)`` creates a UUID-named submission
directory containing ``bundle.json`` and ``submission.json``. Source contents
are embedded in the bundle and can be materialized with ``CodeBundle.write``
into a fresh directory; subsequent edits/removal of the original files cannot
change them. Staging records bundle/worker protocol and software versions,
but does not yet provision a worker or pin an image. Those are submission
preparation responsibilities, not compilation side effects.

Serialization is deliberately closed: finite JSON scalars, ``Path``, lists,
tuples and string-keyed dictionaries; models built from these types, unions,
scalar ``Literal`` choices and nested data-only models; numeric/length
constraints and ``Strict``; and builtin ``list``/``dict``/``tuple`` default
factories. Framework ``ParamMeta`` is preserved explicitly. Executable default
factories, validators, serializers, custom initialization/core schemas,
recursive models, model-instance defaults (including inside containers) and unrecognized types/constraints
are rejected rather than silently weakened. Unsupported protocol versions or
unknown protocol fields also fail. No Recipe pickle, class-name import or
expression evaluation is involved.
Scope settings and the configuration snapshot use the same finite tagged
encoding as parameter values, including nested metadata. Non-finite numbers
are rejected during freezing, before any submission directory is created.

Plans versus observed execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A submission UUID identifies a workflow; a declared step name identifies a
logical node within it; a separate UUID identifies each execution attempt.
These identities are independent of scheduler job IDs. Produced-state identity
is the pair ``(cache_key, producer_field)``: cache hits and loop pass-through
must retain the original producing field, even when an output is renamed.

``AttemptRecord`` carries the existing provenance observation vocabulary,
captured streams and produced-state identities. States are ``running``,
``succeeded``, ``failed``, ``cached``, ``skipped`` and ``unknown``. Final states
require a matching leaf observation; unsuccessful or unknown attempts cannot
publish reusable state. Readers check workflow, attempt, logical step and
bundle identity before accepting a record. A missing/truncated final record
or scheduler-reported completion is **not success**.
The observation's name must also agree with the envelope's logical step.
Unlike a reporting manifest, its input/output fields contain tagged values:
strict ``Path``/tuple values, union branches and values under ``Any`` must not
be converted to strings or lists. Unsupported runtime values are rejected
before a final record can be published. ``AttemptRecord.result(scope)``
decodes these fields and validates the executable inputs/outputs.
Model instances require explicit model annotations; hiding one under ``Any``
is rejected because a data dictionary cannot recover its undeclared class.

Files are fsynced and published atomically without replacing an existing
record, using a same-filesystem hard link followed by directory fsync of the
leaf and its full ancestor chain, including newly created submission and
attempt directories. This covers parent-entry durability as well as atomic
visibility; the shared filesystem must support and honor those semantics.
Sync failures before publication propagate and cannot create a commit. If the
final link is already visible and contains the exact intended record, the worker
keeps that single terminal verdict and warns that directory durability could not
be confirmed; it never tries to publish a contradictory failure beside it.
Started/final records and requeue/publication diagnostics have separate names
per attempt. Volatile ``unknown`` status is reconstructed rather than frozen as
a write-once attempt record. This is not a multi-process cache
journal or ownership lease. Shared cache coordination follows in M2.

Worker submission and execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``prepare_worker_slurm`` is deliberately separate from ``freeze_recipe``. It
requires every executed container image to resolve to an immutable reference
on the submission host, creates a
unique submission, materializes every pystep code bundle, snapshots the
installed Shinobi worker source, fingerprints the worker interpreter's package
set, and emits one worker command per Slurm allocation. A caller may use
``provision_worker_venv`` to build a content-addressed dependency environment
from ``uv.lock`` on the login host, or name an already provisioned absolute
``--worker-python`` that is identically visible on every compute node. Compute
jobs never install or download Python packages. The staged source digest,
Python version, platform ABI tag and distribution digest are checked before a
scientific command starts. An unresolved image is refused before ``sbatch``;
compute workers never repeat mutable-tag resolution. Sites that prohibit image
downloads on compute nodes should supply shared, pre-staged ``.sif`` files (or
otherwise pre-populate their runtime storage) during provisioning.

For example, from a cluster driver whose workspace and interpreter paths are
shared with the workers::

  ninja compile recipe.py:pipeline --worker --submit \
      --workdir /data/project \
      --submission-root /data/project/.shinobi/submissions \
      --worker-python /data/worker-env/bin/python \
      --code-root /data/src

Each allocation imports a pystep only from ``<submission>/code/<index>`` and
executes exactly one declared step through the ordinary dispatch lifecycle.
That complete captured root is also mounted and placed on the child runner's
``sys.path``, so bundled package helpers resolve in image and venv subprocesses;
the original checkout is never consulted.
Binary cabs select their frozen native/container/venv tool backend; image- and
venv-backed pysteps reuse the existing out-of-process runner. A tool venv is
separate from the worker environment and remains unpinned even when its
installed-distribution digest is recorded. An image-backed pystep records both
the prepared image digest and staged Python-code digest.

M2 performs cache checks inside each allocated worker, after every declared
predecessor has committed. ``OutputRef`` wiring carries the producing cache
key and output field out of the immutable parent attempt record; ``after`` and
compiler-inferred mutation ordering carry no data provenance. The ordinary
shared cache-key and manifest implementation is used, including declared-
product validation, boundary fingerprints and partial/uncached-producer
semantics. A hit still consumes an allocation: submission-time pruning cannot
prove that its predecessors will publish reusable state.

The frozen execution identity adds the captured pystep code-bundle digest, the
submission-resolved image digest, or the resolved tool-venv path and
installed-distribution digest as applicable. Worker packaging is not part of
a scientific step key, so updating the staged Shinobi worker does not evict
unrelated branches. A venv fingerprint establishes package-version parity,
not an exact binary/OS pin; venv-backed results therefore remain unpinned even
when cached. ``ninja compile --worker --submit --cache`` and ``--no-cache``
override scope/config settings, while ``--cache-dir`` selects the shared store
without enabling it by itself. These options are refused by the legacy argv
compiler rather than silently ignored.

Mutation-declaring leaves use the same snapshot eligibility, state naming,
restore and conservative exclusion rules as local dispatch. Before a worker
cache decision, recovery is limited to the concrete mutation paths assigned to
that allocation; Slurm's inferred mutation dependencies ensure another live
worker is not writing those paths. The marker names the invocation's immutable
final attempt record as its success oracle. On success, Rule B and the journal
commit precede that record; the cache manifest is updated only afterwards, and
trash disposal plus marker clearing remain last. A cache-index failure can
therefore cost reuse but cannot revoke committed work, while a missing or
failed attempt record forces rollback and removes a same-key stale cache entry.
This ordering also distinguishes a legitimately completed uncached mutator
from a crash, so reconciliation does not silently discard its work. Loop
pass-through still publishes a current immutable attempt record and carries
the producing state's key and environment provenance without claiming a new
result. Sandboxing is always enabled. Each attempt owns
``sandboxes/<attempt-id>/...`` under the unique submission directory on the
same filesystem as the recorded workspace;
cross-filesystem scratch is refused rather than silently becoming node-local
staging. A successful tool must validate and harvest its declared outputs
before its final attempt record is committed. Failed tools and harvest errors
retain the exact shared sandbox path in their diagnostics. Harvesting several
products is ordered, but is not a filesystem transaction across all products.

Scientific-workspace ownership is deliberately coarser than the per-step
access ordering: one workflow that declares any filesystem write owns the
canonical workspace from before its first local step or ``sbatch`` call until
local completion or terminal detached finalization. Independent steps inside
that workflow still run concurrently, and different canonical workspaces do
not conflict. Resolving the workspace before naming
``.shinobi/workspace-owner.json`` makes symlink/relative spellings of one
workspace share the same authority. Bind-mount aliases cannot be discovered
portably. Shinobi derives the ownership authority from every statically known
canonical write target (MUTABLE inputs, ``write_path`` destinations and path
outputs) rather than from the launch directory. Two launches from different
workdirs that point at the same absolute or symlink-aliased dataset therefore
converge on one owner. Several targets use their common parent; a filesystem
root is refused as too broad to be a safe authority. Claims also record
resolved access paths, so overlapping
claims rooted at nested authorities conflict. The short transaction lock on
the ownership record is never held while a tool runs.

Every claim is also entered in one shared access registry (by default
``~/.shinobi/workspace-owners.json``; override with
``SHINOBI_OWNERSHIP_REGISTRY``). The registry stores the complete canonical
access set and permits disjoint workflows, but rejects any overlap where at
least one side writes. This single rendezvous makes exclusion independent of
which nested authority was acquired first and catches a workflow that writes
one tree while reading another workflow's mutation target. For detached use,
the resolved registry path is frozen in ``execution.json`` and must be on
shared storage whose distributed locks have passed the same physical storage
probe required for other shared metadata.

Detached ownership persists while jobs are queued, between allocations and
after the submitting client exits. A Slurm requeue derives a fresh attempt
UUID from the immutable planned UUID and ``SLURM_RESTART_COUNT``, publishing
that generation before execution. Downstream workers and finalization select
the newest published generation, so a started/final record from an earlier
invocation cannot validate the retry. No elapsed-time timeout steals an owner.

Use ``ninja workspace inspect --workdir PATH`` for a read-only report and
``ninja workspace reconcile --workdir PATH`` to release only an owner proven
dead by its local liveness lock or terminal Slurm accounting. Missing or
unavailable evidence is reported as uncertain and is not released. If even one
planned job lacks a durable accepted-job record, the submission is likewise
uncertain even when every recorded job is terminal: the submitter may have
crashed after ``sbatch`` accepted the omitted job. After inspecting such a
workflow and cancelling any remaining scheduler work, an operator may use
``ninja workspace release --workdir PATH --workflow-id UUID --force``. The exact workflow
identity must still match: force never releases a newer owner. The immutable
execution plan says whether ownership is required and records its canonical
access set; submitted workers require the matching ``ownership.json`` and
active owner before every allocation touches data. A missing marker and an old
queued job after operator release both fail closed. This
storage ownership is separate from cache/snapshot metadata synchronization.

Submission writes one immutable job-id record immediately after every
successful ``sbatch`` call and writes ``handle.json`` even when a later
submission fails. This makes accepted jobs recoverable and discoverable after
a partial handoff. Scientific jobs use ``afterok`` dependencies and ask
Slurm to terminate impossible dependencies. An ``afterany`` finalizer reads
the durable job/attempt records, queries accounting, and writes steps in
declaration order. A missing final worker record is ``unknown`` even when
Slurm says ``COMPLETED``; scheduler state never manufactures success. Cancelled
jobs are reported separately from missing/unknown attempts. An early
finalization while work is still active is only a volatile observation and does
not publish ``finalization.json``. Once all jobs are terminal, finalization is
written once; a successful workflow additionally publishes ``manifest.json``.
Repeating finalization returns that stable record without depending on later
``sacct`` availability.

Supported deployment workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Support is conditional on the storage and runtime being proved at the site;
it is not a claim that every NFS-like mount has the required semantics. Before
running scientific data, an operator must:

#. Mount the workspace, submission root, cache, snapshot store, ownership
   registry, container images, worker environment and tool venvs at the same
   absolute paths on the submission host and every selected compute node.
#. Run ``tests/slurm_physical/run_m2.py`` against that exact shared mount.
   The probe must demonstrate cross-node OFD-lock exclusion and lossless
   concurrent cache/journal transactions in every holder direction.
#. Provision the worker environment on the login host, either with
   ``provision_worker_venv`` from the project's locked dependencies or by
   installing a fixed interpreter at the absolute path passed to
   ``--worker-python``. Compute allocations never install dependencies.
#. Provision every declared tool venv separately and verify that its Python and
   installed packages work on each eligible node. A missing or unreadable tool
   venv is a preparation error; a worker pystep never falls back in-process.
   A venv distribution digest provides version parity, not an OS/binary pin,
   so a manifest containing one remains unpinned.
#. Pre-stage immutable ``.sif`` images where compute nodes cannot pull them.
   Submission resolves and records their content digest before any ``sbatch``.
   Image pysteps may use ``ctx.import_callable`` and ``ctx.import_module`` for
   environment-only packages such as ``casatasks``; local helpers reached by
   literal imports are captured from ``--code-root``. Name dynamic local
   helpers explicitly with ``include_modules`` when using the Python API.

Then compile and detach from a shared checkout, for example::

  export SHINOBI_OWNERSHIP_REGISTRY=/data/project/.shinobi/workspace-owners.json
  ninja compile recipe.py:pipeline --worker --submit --cache \
      --workdir /data/project \
      --submission-root /data/project/.shinobi/submissions \
      --cache-dir /data/project/.shinobi/cache \
      --worker-python /data/envs/shinobi-worker/bin/python \
      --code-root /data/src

The resulting handle is sufficient after the submitting shell exits. Use
``ninja status HANDLE`` for a fresh scheduler view. The detached ``afterany``
finalizer writes ``finalization.json`` and, only when every immutable attempt
record committed, ``manifest.json``. Re-running finalization is idempotent. A
missing result remains ``unknown`` even if Slurm reports ``COMPLETED``.

With caching enabled, an unchanged submission consumes allocations but skips
valid work inside them. A parameter, captured-code, image or tool-environment
change invalidates that step and only its data dependants. Deleted products
invalidate their producer. Mutation-declaring steps additionally use the
shared snapshot journal: a killed or requeued allocation restores the named
predecessor state before retrying, while a committed attempt survives an
interrupted cache update or final cleanup. Inspect ownership with
``ninja workspace inspect --workdir PATH``. Reconcile only when Slurm and the
durable job records prove the owner dead; uncertainty is deliberately not
permission to release it.

The initial supported boundary is a declared flat DAG of binary cabs and
image-backed or pre-provisioned venv-backed ``@pystep`` computations on shared
storage. It excludes node-local staging, dynamically sized scatter, nested
recipes and arbitrary ``@step`` orchestration functions. Those recipes remain
valid for local execution or ``ninja run --remote``; they are not silently
weakened for worker offload.

``tests/slurm_physical/run_m3.py`` is the release gate for this workflow. It
must pass from a fresh checker process after the M2 storage probe on any mount
claimed as supported. It exercises a native/image/venv mutation chain on
different nodes, overlapping independent commits, selective cache
invalidation, missing products and environments, immutable staged source, a
cancelled finalizer, representative pre-/post-oracle crashes, and real Slurm
requeues of both pystep modes. The ordinary test suite exhaustively injects the
remaining S1--S5 publication boundaries; the physical gate verifies that the
same recovery protocol works through the scheduler, subprocess/container
boundaries and the site's actual shared filesystem.

When a recipe can be offloaded
------------------------------

Offloading requires that the whole recipe be statically knowable -- the
compiler must be able to determine every job and every dependency without
running any Python. A recipe using the legacy ``ninja compile`` argv path is
eligible only when:

* it has **no orchestration functions** (nothing whose behaviour depends on
  live Python control flow),
* every step is a ``binary``-flavour ``Cab``,
* any **MUTABLE input is a path** (see below), and
* only **paths** cross between steps (an output wired into a later input must be
  a filesystem path knowable at compile time, not a wrangler-derived value).

Anything relying on live Python is rejected with an explanation. That is not
the end of the road for a cluster: see :ref:`offload-remote` below, which runs
any recipe on a remote host and has none of these restrictions.

The shared-storage ``ninja compile --worker --submit`` path admits a wider but
still declared subset: binary cabs plus image-backed or pre-provisioned
venv-backed ``@pystep`` nodes with explicitly bundled, transportable source and
typed data. It retains declared loops, but rejects arbitrary orchestration
functions, nested recipes and scatter. Ordinary non-path outputs may cross
between worker steps through committed attempt records. A path mutated in place
must remain statically resolvable so submission can derive the same mutation
ordering as the legacy compiler; an unresolved wired ``MUTABLE`` path is
refused before any job is submitted.

.. _offload-remote:

The other path: ``ninja run --remote``
--------------------------------------

``ninja compile`` is not the only way onto a cluster, and it is the more
demanding one. ``ninja run TARGET --remote user@host:/path`` rsyncs the target
and its cab dependencies to one host, starts ``ninja run`` there detached, and
gives you a handle to poll -- no scheduler in between.

The path may be relative, in which case it means what it would mean to
``rsync`` or ``ssh``: relative to the remote login shell's working directory.
It is resolved to an absolute path once, from that shell's own ``$PWD``, before
anything is built from it -- the provisioning script ``cd``\ s, so a path left
relative would be resolved a second time against the directory it had just
entered.

Which to reach for:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - ``ninja compile --submit``
     - ``ninja run --remote``
   * - Runs on
     - Many nodes, as a scheduler-managed DAG
     - One host, start to finish
   * - Recipe must be
     - Purely declarative (see above)
     - Anything ``ninja run`` can run
   * - Steps are ordered by
     - Slurm ``afterok`` dependencies
     - ninja's own scheduler, on that host

So a recipe with orchestration functions -- the kind offload rejects -- is
still perfectly runnable on the cluster's big-memory node over ``--remote``.
The cost is that one process babysits the whole pipeline there, which is
precisely what ``compile`` exists to avoid for long multi-node runs.

The environment on the far side
-------------------------------

Both paths need a Python environment on the remote host, and neither
inherits yours. ``--remote`` can build one:

.. code-block:: console

    $ ninja run myrecipe.py:selfcal --remote user@cluster:/scratch/run1 \
        --venv sync --venv-package 'caracal==2.0.1'

``--venv sync`` provisions with ``uv``, installing uv itself first if the host
has none. The environment is named by a hash of what was asked for plus the
host's architecture, libc and Python, so it is built once and reused by every
later launch, and two different hosts sharing one ``/scratch`` cannot activate
each other's. See :ref:`the --venv options <cli-remote>` for the full flag set.

Two things specific to clusters:

* **Provision from a login node.** Compute nodes commonly have no route to
  PyPI, and ``sync`` needs one. Run ``--venv sync`` once where there is egress;
  every subsequent launch can use the default ``--venv use``, which needs no
  network and never writes to the remote.
* **Provisioning executes build backends** for any source distribution
  involved, under your account. That is a real difference from a container
  image, which executes nothing when it is pulled.

``ninja compile`` has no equivalent: its jobs run whatever ``sbatch`` finds on
the node, so a container runtime (``--container-runtime``) is the reproducible
option there.

.. _offload-mutation-ordering:

In-place mutation is offloadable
--------------------------------

Self-cal pipelines rewrite one Measurement Set in place: ``flag``,
``gaincal`` and ``applycal`` each take the same MS as a plain input and
modify it. Nothing wires them together, so the declared graph sees three
*independent* steps -- run locally that is harmless, because the default
``max_workers: 1`` executes them in declaration order anyway, but handed to
a cluster as an unordered DAG they would run concurrently against the same
files.

``ninja compile`` therefore derives the missing edges itself. As it resolves
each step it records its schema-declared path accesses. A path is written when
the input is ``MUTABLE``, when the same path field appears on both the input
and output models, when ``ParamMeta.write_path`` marks it as a destination, or
when it is a statically resolvable path output. This shared ``Scope`` analysis
also covers image-backed and venv-backed pysteps: their default Python
``IMMUTABLE`` policy means the input object is copied, not that the filesystem
path is read-only. Any two steps sharing a path are ordered when **at least
one** writes it:

* mutate-then-mutate -- the second waits for the first;
* mutate-then-read -- a reader sees the finished result;
* read-then-mutate -- the writer waits for readers of the old contents.

Two steps that only *read* the same path are left parallel, which is the
whole point of offloading them.

Because this works on **resolved values**, it does not care how each step
spells the path. A step wiring the MS from a recipe input and a step naming
the same file as a literal are recognised as touching one file, as are
``./obs.ms`` and ``/data/obs.ms``, a path neither step mentions because both
take a schema default, and ``/data/obs.ms`` versus ``/data/obs.ms/CORRECTED``
-- a Measurement Set is a directory, so containment counts.

Relative paths are anchored to the workflow's declared workdir/workspace
before canonicalisation, rather than to the compiler process's current
directory.

A MUTABLE input that is *not* a path is still refused: that is a live Python
object, and no shared filesystem can carry one across a node boundary.

.. warning::

   Canonicalisation is ``Path.resolve()``, and it runs on the machine where
   ``ninja compile`` runs. Two steps reaching one MS by paths that are only
   equal *on the compute node* are therefore not recognised as sharing it,
   and no ordering edge is emitted. In practice that means a cluster where
   the submitting host and the compute nodes disagree about the filesystem:
   ``/scratch`` against ``/mnt/scratch`` under a different automount layout,
   or a symlink that resolves one way on the login node and another way on
   the node that runs the job.

   The container boundary is *not* affected -- every container backend
   identity-mounts (``-v {d}:{d}``, ``--bind {d}:{d}``, and Kubernetes
   ``mountPath == hostPath.path``), so a container-side path equals its
   host-side path by construction and comparison holds straight through.

   Closing the cross-node case needs a canonical naming the cluster itself
   agrees to, which shinobi cannot derive. Until then: give steps that share
   an MS the same spelling of its path, and prefer paths that resolve
   identically on both sides.

A :ref:`declared loop <declared-loops>` satisfies all of this: unrolling
leaves a plain dependency chain of ``Cab`` steps, and its convergence test
becomes a guard at the top of each job's script --

.. code-block:: bash

    if [ -e /scratch/converged.flag ]; then
      exit 0
    fi

-- so an iteration that runs after the loop has converged exits successfully
without doing any work, satisfying the ``afterok`` dependency so the rest of
the chain proceeds. It needs to create nothing on the way out: every path a
loop carries resolves to the same name in every iteration. A body that instead
names its outputs *per cycle* is not statically knowable and is rejected, like
anything else the compiler cannot resolve.

A minimal offloadable recipe
----------------------------

This mirrors ``examples/offload_demo.py``: two steps wired by a single
filesystem path -- ``make`` touches a file, ``use`` reads it.

.. code-block:: python

    from pathlib import Path

    from pydantic import BaseModel

    from shinobi.steps import Cab, InputRef, OutputRef, ParamMeta, Recipe, StepRef


    class PipeInputs(BaseModel):
        target: Path = Path("made.ms")


    class TouchInputs(BaseModel):
        out: Path


    class PathOutputs(BaseModel):
        out: Path | None = None


    class CatInputs(BaseModel):
        f: Path | None = None


    class OkOutputs(BaseModel):
        ok: bool = True


    make = Cab(name="make", command="/bin/touch", inputs_model=TouchInputs,
               outputs_model=PathOutputs, field_meta={"out": ParamMeta(positional=True)})
    use = Cab(name="use", command="/bin/cat", inputs_model=CatInputs,
              outputs_model=OkOutputs, field_meta={"f": ParamMeta(positional=True)})

    pipe = Recipe(
        name="pipe",
        inputs_model=PipeInputs,
        outputs_model=OkOutputs,
        steps=[
            StepRef(name="make", step=make, wiring={"out": InputRef(field="target")}),
            StepRef(name="use", step=use, wiring={"f": OutputRef(step="make", field="out")}),
        ],
        output_wiring={"ok": OutputRef(step="use", field="ok")},
    )

Because the only thing crossing between steps is a path (``make``'s ``out``
output is a passthrough of its ``out`` input, so it is known statically), the
recipe is offload-eligible.

Compile it
----------

Preview the compiled Slurm workflow without submitting anything -- no cluster
needed:

.. code-block:: console

    $ ninja compile myrecipe.py:pipe --target /scratch/made.ms --container-runtime none

This prints two ``sbatch`` scripts linked by ``--dependency=afterok``: ``make``
first, then ``use`` once ``make`` succeeds.

Or run the same recipe locally instead, driven in-process:

.. code-block:: console

    $ ninja run myrecipe.py:pipe --target /tmp/made.ms

Submit and detach
-----------------

Add ``--submit`` to hand the workflow to a real Slurm cluster and detach. A
handle file is written under ``<workdir>/.shinobi/<recipe>/handle.json``:

.. code-block:: console

    $ ninja compile myrecipe.py:pipe --target /scratch/made.ms \
        --container-runtime none --submit

Check on it later
-----------------

``ninja status`` queries the engine fresh from the handle file -- there is no
persistent process to keep alive:

.. code-block:: console

    $ ninja status /scratch/.shinobi/pipe/handle.json

``ninja runs`` does the same for *every* launch this workspace has made, in
one table, and ``ninja logs <name> --follow`` streams a ``--remote`` run's
output until it finishes. Both reconstruct state the same way and keep
nothing running locally. See :ref:`cli-runs` and :ref:`cli-logs`.

Once a run is done, remove its handle file and Slurm job logs with
``ninja clean --launches --workdir <workdir>`` (or run it from ``<workdir>``;
see :ref:`ninja-clean`) -- unlike run manifests and the step cache, this is
opt-in, since deleting a handle for a still-running detached job doesn't
stop it, but does destroy ``ninja status``'s only local record of it.

.. note::

   The Slurm compiler and step backend were verified against a real Slurm
   cluster, and ``tests/test_slurm_live.py`` keeps the offload path covered
   automatically against the throwaway cluster in ``tests/slurm_live/``. That
   cluster is single-node, so multi-node scheduling and cross-node shared
   storage are not proven by it -- check generated scripts and accounting
   behavior on your own site's scheduler. See :doc:`concepts/backends`.
