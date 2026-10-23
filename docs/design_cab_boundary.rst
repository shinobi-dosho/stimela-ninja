Cab ownership boundary: v0.2.0b1
================================

Status: approved implementation design, 23 October 2026. This migration spans
``stimela-ninja`` and ``dosho``. It introduces no recipe language.

Ownership and dependency
------------------------

Dosho owns cab schemas, parameter validation, argument policies and formatting
validated parameters into argv. Shinobi owns declared framework type contracts,
DAG construction and wiring, scheduling, execution and backends, filesystem
enforcement, dataset lifecycle and recovery, and caching.

``stimela-ninja`` requires ``dosho[validation]>=0.2.0b1,<0.3``. Dosho's base
catalogue remains readable without Pydantic or Shinobi. Its ``validation``
extra provides the reusable Pydantic structures and compiler without importing
Shinobi. The reverse ``dosho[run]`` dependency is retired: install
``stimela-ninja`` to obtain runnable objects. Lazy ``dosho.get`` and
``dosho.define_cab`` bridges may import the execution adapter on object access.

Compilation and adaptation
--------------------------

``dosho.validation.parse_document`` captures a composed static cab document;
``compile_cab`` builds its parameter models and command metadata. The shared
model-generation helpers implement sanitization, required/default/nullable
rules, choices, string constraints and safe composition once. Recipe input
schemas and worker schema adapters reuse these structures.

Shinobi supplies an explicit trusted dtype mapping for MSv2, CasaTab, FITS,
ProductFamily and DirectoryBundle. Recursive list/tuple/union compilation uses
that mapping; a known strict dtype without a binding is an error, never a
string fallback. Documents cannot select imports, callbacks, validators or
schema hooks. Unknown ordinary legacy dtypes retain their historical fallback.

Dosho's compiled record carries framework declaration fragments as inert JSON.
The Shinobi adapter validates them using its authoritative FamilySpec,
DatasetAccess, FITSAccess and DerivedRead models and constructs the existing
exact ``Cab(Scope)`` class. ``Scope.field_meta`` retains its concrete typed
metadata extension. Provider lookups, YAML, classic JSON and Python authoring
converge on shared compilation and adaptation.

Dosho formats scalar argument values using its policy/pattern models. The
``shinobi.policies.build_argv`` compatibility wrapper first normalizes framework
product values. Unsupported executable flavours fail before backend access,
including dry runs. Dynamic matched values retain existing pass-through
behavior and warnings; unmatched extras are not emitted.

Safety and identity
-------------------

Includes retain caller-supplied filesystem roots, traversal containment and
cycle diagnostics. No command is evaluated; ``dynamic_schema`` is warned about
and never executed. Framework validation remains separate from schema parsing.

Detached workers stage both installed packages under ``worker-src`` and hash
the combined tree. The closed offload codec still refuses arbitrary Python
validators, serializers, factories, subclasses and import references. Known
policy and pattern models reconstruct from data. Local trusted Python schema
validation remains supported independently of offload eligibility.

Cab cache identity is versioned independently of worker source identity. It
includes command/flavour, ordered argument declarations, metadata/patterns,
policies, and Dosho schema/formatter contract versions. Implicit input values
are represented by their actual formatter-produced argv tokens per input field plus an
override-presence marker, preserving field boundaries and
mapping order (including nested mappings), boolean modes and trusted scalar
spelling. A cab wrapped by an orchestration function includes both that code
identity and its cab argument semantics. Non-cab function identities are
unchanged. Output-only declarations remain in the separate product contract.
Bundle schema version 3 freezes and verifies those semantics for each cab. Historical bundle versions
remain readable. Non-cab cache identity, dataset state IDs, snapshots and
worker attempt IDs retain their separate contracts.

Integration and release gate
----------------------------

Both projects and citations target ``0.2.0b1``. Dosho locks independently with
no reverse Shinobi dependency or Git source override. Integration verification
installs an explicit candidate wheel pair and checks package origins; it must
not silently substitute a released framework or catalogue.

Dosho ``0.2.0b1`` is published on PyPI from merged commit
``26dfb3d2b4014f1280bcdb2857d67232874a5c33``. Its published wheel matches
the verified candidate wheel byte for byte. Ninja's registry lock now resolves
that beta and has passed ``uv lock --check``, completing the dependency lock
gate. It contains registry artifacts, with no local-path overrides or
fabricated hashes. Ninja publication still requires maintainer authorization.

Provider generation negotiation
-------------------------------

Dosho's provider returns the ``dosho-cab-v1`` dialect tag. Older Shinobi
versions reject that tag instead of silently interpreting new catalogue
schemas using the former compiler. Runnable-object bridges additionally check
``CAB_ADAPTER_CONTRACT_VERSION`` and give explicit 0.2 installation guidance.
The adapter continues to support legacy third-party ``yaml_cab`` documents.

Temporary CI bootstrap gate
---------------------------

Dosho's default core job covers validation, catalogue/image metadata, CLI,
publication authorization and version tests without Shinobi. During the
Dosho-first bootstrap only, full cab integration and docs require an explicit
compatible ``ninja-ref``. Once the companion adapter lands on Ninja main,
remove the two ``CAB_BOUNDARY_BOOTSTRAP`` job conditions and default that ref
to main, restoring automatic full coverage on every push/PR. Completing the
v0.2 release still requires this CI restoration; Ninja's registry lock gate is
complete. The temporary gate is not the intended steady-state CI policy.

Local incorporation of PR 231
-----------------------------

The approved integration includes the changes from `Ninja PR 231
<https://github.com/shinobi-dosho/stimela-ninja/pull/231>`_ at head
``d7242c0e017ef4b29672b621cb1e5c5184ab8f6e``, applied locally without merging
the pull request. These remain Shinobi-owned execution and planning contracts:
finite selected-family path planning, optional scratch-path handling, declared
workspace propagation and restoration of module-level decorated pysteps.
Plans never fabricate observed family membership; execution still selects
from actual producer outputs.

Integration additionally forwards the declared workspace through nested generic
access planning, worker preflight, FITS lifecycle snapshots and auxiliary
recovery traversal. This keeps canonical ownership paths and access dependencies
independent of the caller's current directory. Dosho schemas and formatting,
the exact Cab adapter, and staging of both packages remain unchanged.

Pydantic minimum version
------------------------

Both Shinobi and Dosho's validation extra require Pydantic 2.14 or newer.
The approved minimum avoids the context-local generic specialization identity
observed with Pydantic 2.12 when selected FITS-family handoffs cross worker
threads. No framework compatibility cache or broader type acceptance is added.
The currently tested Python 3.11–3.14 range is unchanged. The combined runtime suite
passed with Pydantic 2.14.1; release verification also checks the 2.14.0 floor.
Both Dosho's independent lock and Ninja's registry lock record this requirement.

Read-only catalogue extension
-----------------------------

The approved extension provides named WSClean, DDFacet and QuartiCal read-only
variants with fixed validated native controls and implicit argv constants.
Static composition reuses existing schemas through an explicit Dosho filesystem
root. The shared compiler supports ordinary Pydantic min/max length constraints;
QuartiCal's dynamic term matcher requires a full identifier match.

Flagversions backup and delete reserve the associated MS exclusively with a
strict READ declaration with ``exclusive: true`` and separately own the
attached history bundle. This
user-approved rule blocks all concurrent MS access during history mutation;
it preserves read-only MS mounts and never enters scientific mutation
snapshots. Interrupted sidecar writes may require inspection/manual repair;
MS snapshots do not recover the separate directory. General writers remain
available.

Cab identity conditionally adds versioned ``cab-access-v1`` semantics for
DatasetAccess/FITSAccess declarations and effective read-only input metadata
and patterns. Ordinary cabs without access contracts retain their keys.
Output-only product declarations, non-Cab identities, dataset states and worker
source digests remain separate. Frozen bundles verify these access semantics.
Native concurrency and preservation require separate measured acceptance;
passing declaration and argv tests alone does not establish native safety.

Container bind planning resolves relative declared destinations against the
workdir and uses the deepest applicable mount to decide whether a write is
already covered. Sibling outputs and scratch remain writable beneath a
read-only input-parent mount; nested protected MS roots remain read-only.
Readonly QuartiCal gain/log locations are string destinations backed by
explicit family/scratch declarations, not in-place mutable input paths.
