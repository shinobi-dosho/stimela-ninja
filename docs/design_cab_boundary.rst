Cab ownership and 0.2 compatibility
===================================

The v0.2.0b1 boundary separates reusable cab schema/argument machinery from
workflow execution. These are implemented contracts shared by
``stimela-ninja`` and ``dosho``; recipes remain declared Python DAGs.

Ownership and dependency
------------------------

Dosho owns cab schemas, parameter validation, argument policies, dynamic
parameter patterns and formatting validated parameters into argv. Shinobi
owns declared framework types, DAG construction and wiring, scheduling,
execution and backends, filesystem enforcement, dataset lifecycle and recovery,
and caching.

``stimela-ninja`` requires ``dosho[validation]>=0.2.0b1,<0.3`` and Pydantic
2.14 or newer. Dosho's base catalogue remains readable without Pydantic or
Shinobi. Its ``validation`` extra provides reusable Pydantic structures and a
compiler without importing Shinobi. The reverse ``dosho[run]`` dependency is
retired: install Shinobi to obtain runnable objects. Lazy ``dosho.get`` and
``dosho.define_cab`` bridges import the execution adapter on object access.

Upgrading an existing installation
----------------------------------

Shinobi 0.2.0b1 is not yet published, while Dosho 0.2.0b1 is available on
PyPI. Install the current framework source to obtain the compatible pair:

.. code-block:: console

    $ pip install --upgrade git+https://github.com/shinobi-dosho/stimela-ninja.git

This installs Dosho's validation extra automatically. Existing
``from shinobi.steps import ParamMeta, ParamPattern, ParamSegment, Policies``
imports remain supported. ``Policies`` is the shared Dosho model; the three
metadata/pattern types extend Dosho models with Shinobi's typed family
contract. Use those Shinobi extensions for framework declarations, and
``shinobi.loaders.build_model`` when dtype names include strict datasets or
product families. Plain configuration schemas can use
``dosho.validation.build_model`` without the execution framework.

Dosho's provider returns the ``dosho-cab-v1`` dialect tag. Older Shinobi
versions reject that tag rather than interpreting new catalogue schemas with
the former compiler. Runnable-object bridges also check the adapter contract
version and report explicit 0.2 installation guidance. Legacy third-party
``yaml_cab`` and ``stimela-classic`` documents remain supported.

The built-in execution default is now Apptainer. Cabs without a backend
selection need a container image and the runtime on ``PATH``. Select
``ninja --backend native run ...`` or ``backend.default: native`` for
host-installed tools. See :doc:`concepts/backends` for override precedence.

Compilation and adaptation
--------------------------

``dosho.validation.parse_document`` captures a composed static cab document;
``compile_cab`` builds its parameter models and command metadata. Shared
model-generation helpers implement sanitization, required/default/nullable
rules, choices, string and length constraints, and safe composition once.
Recipe input schemas and worker schema adapters reuse these structures.

Shinobi supplies explicit trusted dtype bindings for MSv2, CasaTab, FITS,
ProductFamily and DirectoryBundle. Recursive list/tuple/union compilation uses
that mapping; a known strict dtype without a binding is an error, never a
string fallback. Documents cannot select imports, callbacks, validators or
schema hooks. Unknown ordinary legacy dtypes retain their historical fallback.

Dosho's compiled record carries framework declaration fragments as inert JSON.
The Shinobi adapter validates them using its authoritative FamilySpec,
DatasetAccess, FITSAccess and DerivedRead models and constructs the existing
``Cab(Scope)`` class. ``Scope.field_meta`` retains its concrete typed metadata
extension. Provider lookups, YAML, classic JSON and Dosho's Python authoring
helper converge on shared compilation and adaptation. Direct Python ``Cab``
authoring remains supported.

``shinobi.policies.build_argv`` first normalizes framework product values,
then calls Dosho's formatter with its shared policy/pattern models. Unsupported
executable flavours fail before backend access, including dry runs. Dynamic
matched values retain their pass-through behavior and warnings; unmatched
extras are not emitted. Pattern matching covers the complete parameter name.

Safety and identity
-------------------

Includes retain caller-supplied filesystem roots, traversal containment and
cycle diagnostics. No command is evaluated; ``dynamic_schema`` is warned about
and never executed. Framework validation remains separate from schema parsing.

Detached workers stage both installed packages under ``worker-src`` and hash
the combined tree. The closed offload codec refuses arbitrary Python
validators, serializers, factories, subclasses and import references. Known
policy and pattern models reconstruct from data. Local trusted Python schema
validation remains supported independently of offload eligibility.

Cab cache identity is versioned independently of worker source identity. It
includes command/flavour, ordered argument declarations, metadata/patterns,
policies and Dosho schema/formatter contract versions. Implicit input values
are represented by actual formatter-produced argv tokens per field plus an
override-presence marker, preserving field boundaries, mapping order,
boolean modes and trusted scalar spelling. A cab wrapped by an orchestration
function includes both its code identity and cab argument semantics.

Access-bearing cabs additionally include versioned ``cab-access-v1`` semantics:
DatasetAccess/FITSAccess declarations and effective read-only input metadata
and patterns. Frozen bundle schema version 3 verifies these semantics on
restoration; historical bundle versions remain readable. Output-only product
declarations, non-cab identities, dataset state IDs, snapshots, worker source
digests and worker attempt IDs keep their separate contracts.

Planning and guarded readers
----------------------------

Finite selected-family path planning, optional scratch destinations,
workspace-aware access planning and restoration of module-level decorated
pysteps are supported by the execution layer. Plans never fabricate observed
family membership: execution selects from actual producer outputs. See
:doc:`concepts/products` and :doc:`offloading` for the bounded handoff rules.

Dosho supplies ``wsclean-readonly``, ``ddfacet-readonly`` and
``quartical-readonly`` with strict MSv2 reads, protected input paths and fixed
validated native controls. General writer variants remain available. Outputs,
caches, wisdom, temporary data and logs must stay outside protected inputs,
including symlink aliases. Relative destinations resolve against the declared
workspace; sibling outputs and scratch remain writable beneath a read-only
input-parent mount while nested protected MS roots remain read-only.

Flagversions backup and delete use ``DatasetAccess(mode="read",
exclusive=True, ...)`` and separately own the attached history bundle. The
reservation blocks every concurrent MS access without granting scientific
write authority or entering MS mutation snapshots. They use ``cache: false``.
Interrupted history updates may require inspection or manual repair; MS
snapshots do not recover that separate directory. See :doc:`concepts/datasets`
for lifecycle limits. Declaration and argv checks alone do not establish
native-tool concurrency or preservation safety.

Paired verification
-------------------

Dosho locks independently, without a reverse Shinobi dependency or Git source
override. Ninja's committed lock resolves the published Dosho beta from the
registry. Compatibility CI checks that locked release and fresh Dosho main.
The manual ``cab-candidate-pair.yml`` workflow accepts an exact ``dosho-ref``
and installs explicit wheel pairs, checking installed origins before testing.

Dosho's independent core job tests validation, catalogue/image metadata, CLI,
publication authorization and versions without Shinobi. Full integration and
docs run automatically on pushes and pull requests against Ninja main;
``ninja-ref`` remains available for candidate verification. The temporary
Dosho-first bootstrap gate has been removed.
