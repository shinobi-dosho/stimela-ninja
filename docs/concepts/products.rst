Output families and local bundles
=================================

A ``ProductFamily[Path]`` is a coordinate-labelled table of files emitted by
one step. ``ProductFamily[DirectoryBundle]`` holds local directory products,
such as an entire QuartiCal gain store. Membership can be bounded by validated
inputs or discovered from declared filename templates after execution. Both
use the ordinary execution, sandbox, ownership and cache lifecycle.

A family has two distinct states: ``resolved=False, members=()`` means unknown,
including a failed command; ``resolved=True, members=()`` means successful
resolution found no accepted products. A required family field requires a
family value, not a nonempty table. Declare ``min_members`` when nonempty
membership is part of the tool contract.

Members retain named integer or string coordinates. Integers and strings are
distinct addresses, and booleans are rejected. Coordinates describe filenames
and products, not physical frequencies, timestamps or scientific array axes.
A singleton Q image remains labelled Q even when its filename omits ``-Q``.
Sparse source IDs and saved major-cycle products remain sparse; configured
maximum counts do not prove that files were emitted.

Bounded declarations
--------------------

Python output models declare ``ProductFamily[Path]`` or
``ProductFamily[DirectoryBundle]``. ``ParamMeta.family`` accepts a ``FamilySpec``;
its ``MemberRule`` and ``AxisSpec`` children are closed Pydantic models.
The equivalent static YAML vocabulary is:

.. code-block:: yaml

   outputs:
     image:
       dtype: ProductFamily[File]
       required: true
       family:
         root: "."
         coordinates:
           time: int
           frequency: [int, str]
           polarization: str
           component: str
         rules:
           - path: "{prefix}{time}{frequency}{polarization}-image.fits"
             coordinates: {component: real}
             axes:
               time:
                 count_input: intervals_out
                 suffix: "-t{index:04d}"
                 singleton_suffix: ""
               frequency:
                 count_input: nchan
                 suffix: "-{index:04d}"
                 singleton_suffix: ""
                 aggregate: {value: mfs, suffix: "-MFS", only_multiple: true}
               polarization:
                 values_input: pol
                 tokens: [I, Q, U, V, XX, XY, YX, YY, RR, RL, LR, LL]
                 uppercase: true
                 split: csv_or_compact
                 suffix: "-{value}"
                 singleton_suffix: ""
             required: false

Counts are nonnegative integers from validated inputs. Axes may instead take
an explicit ``values`` sequence or a ``values_input`` sequence. ``include``
filters a finite set. CSV/compact tokenization uses the declared finite token
vocabulary and rejects ambiguous spellings. Compact parsing solves each suffix
once and retains at most two interpretations, bounding work even for
overlapping tokens. Defaults belong to the input
model. Expansion is limited by ``max_members`` (default 10,000; maximum
100,000), before allocating the Cartesian product.

Rules form a union. Duplicate coordinate addresses, duplicate canonical paths,
and directory-bundle ownership overlaps are errors. Constant coordinates name
components, product kinds, representations, formats or flux scales. A finite
``when`` table permits conjunctions of input-value membership, for example
``when: {mode: [Clean, Dirty], predict: [false]}``. It cannot execute expressions,
import callbacks, address other steps or change the declared DAG.

``required: true`` on a bounded rule requires evidence for every active
candidate. The default ``false`` retains only observed members. Rule and
family ``min_members`` constraints are checked before harvesting and success
publication. The example describes filename dimensions; a cab author must
separately declare which modes actually emit those products.

Discovered and bundled products
-------------------------------

.. code-block:: yaml

   family:
     root: "{output_directory}/{output_filename}_cubelets"
     coordinates: {source: int, product: str, format: str}
     rules:
       - path: "{output_filename}_{source}_spec.txt"
         captures: {source: int}
         coordinates: {product: spectrum, format: text}
       - path: "{output_filename}_{source}_cube.fits"
         captures: {source: int}
         coordinates: {product: cube, format: fits}
       - path: "{output_filename}_{source}_cube.h5"
         captures: {source: int}
         coordinates: {product: cube, format: hdf5}

Capture templates match the entire relative filename. Integer captures accept
decimal digits; string captures accept a bounded ASCII filename component
(letters, digits, underscore and hyphen, at most 128 characters). Literal text
and formatted inputs are escaped. Captures cannot cross directory components;
literal nested directories are permitted. The first version keeps captures and
bounded axes in separate rules. There is no arbitrary regex or recursive glob
language in a family declaration.

For a bundle, use ``dtype: ProductFamily[DirectoryBundle]`` and an exact rule
such as ``path: "{gain_directory}"``. The bundle root is one owned product.
Its execution inventory freezes every regular file and directory, including
empty directories. Symlinks, special nodes and failed enumeration prevent
success publication. Bundle children do not become independently owned files
or scientific partitions.

The ``root`` is a containment/search boundary and reservation, never itself a
produced directory or a target to clear. Members cannot escape it through
``..``, absolute paths or symlinks. Families may share a search parent, but
cannot claim overlapping physical products. Output members cannot overlap
input trees. Discovery namespaces are checked against every physical input
before launch, including future members and whole-family/bundle inputs.
Filename prefix proofs permit disjoint output families beside input MS trees.
Source and harvest-destination containment are rechecked against their frozen
canonical boundaries before accepting or publishing members, so a run-created
ancestor symlink cannot redirect a product into another tree. A killMS output
implicitly placed inside its input MS therefore
needs a different explicit destination.

Execution evidence and reuse
----------------------------

Family resolution runs with caching enabled or disabled. Direct candidates
and discovery matches are observed before the command runs. New or changed
files count as evidence; unchanged workspace matches do not. Fresh sandbox
products count after successful execution. Existing broad harvest patterns
cannot supply coordinates for files outside a declared member rule.

``accept_existing: true`` permits unchanged exact candidates or an exact whole
bundle following successful execution. It is refused for discovery. Accepted
existing products are protected from stale-output clearing. Relative sandbox
reuse of an existing workspace product is refused before launch because it
would require explicit staging; use direct execution or an absolute shared
output location. Discovery roots and shared output parents are never cleared
wholesale.

Nonzero commands retain unresolved families and publish no reusable success.
SoFiA's no-source exit code 8 remains a command failure. Required-member,
containment, inventory and cardinality errors are checked before harvest;
existing rollback and immutable publication ordering remain unchanged.

Wiring and selection
--------------------

Whole-family wiring consumes the typed collection. A unique selection returns
a typed file or bundle:

.. code-block:: python

   selected = result.outputs.image.select(
       time=1, frequency="mfs", polarization="Q", component="real"
   )
   reference = recipe.outputs.clean.image.select(
       time=1, frequency="mfs", polarization="Q", component="real"
   )

``OutputRef.select`` remains an ordinary dependency on its producer. Partial
selectors are permitted only when one member matches; unavailable, ambiguous,
unknown and incorrectly typed coordinates raise clear errors. A reference
cannot be selected twice. Selection supplies no runtime graph nodes or array
slicing. Whole-family argv inputs emit member paths using ordinary cab list
policies; bundle inputs emit their physical roots.

Cache and worker boundaries
---------------------------

Product contract version 2 includes normalized family declarations separately
from scientific invocation identity. Cache lookup restores the saved table;
it never rescans to replace missing members. Every recorded member and bundle
child must still exist with its recorded kind, including optional products
that were present. Additional files cannot repair deleted evidence. This uses
the existing intermediate-product existence policy, not an implicit promise of
content checksums. Boundary family inputs use typed values and physical path
fingerprints; selected upstream identities additionally contain the original
producer field and sorted typed coordinate address.

Detached workers preserve framework types with a closed codec and selection
bindings in bundle schema version 2. Arbitrary subclass validators remain
unsupported. Coordinate-bearing produced states and bundle inventories use
attempt-record version 4; historical scalar and indexed-MS records retain
their shapes and readability. See :doc:`provenance` and :doc:`../offloading`.

Supported boundaries
--------------------

Remote schemes (including already-normalized ``Path('s3:/...')``) and store
subresources such as ``gains.qc::G`` are rejected. The initial QuartiCal shape
is one local directory bundle; a trusted consumer chooses a gain group through
its storage library. Internal time/frequency/antenna/partition axes do not
become files. ``StoreRef`` and remote storage adapters are deferred.

Legacy static argv compilation rejects family declarations/selections; use
the detached worker path. Families cannot represent strict mutable MS states
or replace ``dataset_accesses``, snapshots or reusable state-store contracts.
Cab migration, image bumps and scientific-tool product completeness remain
separate downstream work. Engine fixture tests cover naming, sparse capture,
evidence, cache deletion, bundles, selection, mounts and worker transport;
real-tool conformance additionally requires supported tool versions, modes,
input dimensions and container builds.
