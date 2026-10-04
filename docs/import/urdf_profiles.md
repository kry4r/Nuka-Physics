# URDF import profiles

`LoadUrdf(path)` retains the existing permissive projection and exception-based
API. Its one intended mapping correction is the URDF default X axis for revolute,
continuous, and prismatic joints; explicit axes remain unchanged. This legacy
API does not establish semantic fidelity.

`ImportUrdf(path, UrdfImportProfile::Strict)` returns an optional scene and typed
diagnostics. The default is Strict. A rejected input has no scene. Each diagnostic
contains a stable code, kind (`Io`, `Syntax`, `Invalid`, or `Unsupported`), source
filename, XML element and line. `StrictSuccess()` establishes acceptance by this
bounded profile, not full URDF conformance or cross-engine dynamics equivalence.

`ImportUrdf(path, UrdfImportProfile::LegacyCompatible)` reports the same source
limitations while attempting the legacy projection. A scene may coexist with
diagnostics. It never reports `StrictSuccess()`. Projection failures add a separate
diagnostic and leave the scene absent. The XML document is parsed once per call.

## Strict subset

Supported inputs include a single named robot with a connected acyclic link tree,
explicit positive mass and diagonal inertia on each link, inertial origins,
box/sphere or STL/OBJ mesh geometry, visual origins, unit-length scalar joint axes and explicit lower/upper position
limits, fixed/floating/revolute/continuous/prismatic joint types. Mesh resources use
the existing local relative/file/package path resolver. Strict accepts only fields
its audited projection consumes, with numeric arity/finite/range checks, unique
names, resolvable link endpoints and singleton element checks.

The profile is deliberately conservative. For example, zero-mass links and omitted
inertial data are not accepted; this is not a claim that all such source files are
invalid URDF. Unsupported defaults or fields need a later exact mapping before
this profile can accept them. Zero or negative dimensions/masses, non-unit axes, nonfinite float inverses,
degenerate half extents and nonphysical diagonal inertia are rejected. Scaled mesh
streams are checked for finite values before a strict scene is returned. A continuous joint with authored position limits is not accepted.

These currently block strict projection rather than silently changing meaning:

- Cylinder or unknown geometry, planar/unknown joints, nonzero inertia cross terms
- Materials, transmissions, dynamics, mimic, effort/velocity limits and extensions
  outside this bounded profile, even where the legacy importer maps a subset
- Unconsumed attributes, including collision names, or unknown elements

XML comments, prefixed namespace declarations and empty default namespace declarations
are accepted. Nonempty default namespaces are rejected because this reader does not
interpret namespace-qualified URDF elements. Actual
unknown namespaced metadata is reported as `Unsupported`, not malformed XML. This
batch does not preserve arbitrary source metadata in NKS and therefore cannot
claim a lossless export of it. A diagnostic on the enclosing element gives its
XML line; attribute columns and full XML paths are not yet available.

Strict parse/validation errors occur before projection. Asset/mesh loading and
other projection exceptions are reported as `URDF_PROJECTION_FAILED` at the robot
line, with the underlying error message; they do not yet have asset-level source
locations or a dedicated asset-I/O classification.

## CLI

`cook_scene --strict-urdf input.urdf output.nks` prints diagnostics and exits nonzero
without writing output when rejected. The flag rejects non-URDF inputs. The existing
`cook_scene input output.nks` command remains legacy-compatible. NKS imports retain
the legacy path; strictness does not silently propagate into unrelated formats.

The CLI produces NKS/NKA assets. Successful serialization alone does not prove a
production simulation has been run. The regression exercises accepted source →
SceneIR → NKS → reload → production host CookScene. CUDA stepping and cross-engine
physics are separate validation requirements.

Tracked scope: https://github.com/kry4r/Nuka-Physics/issues/12
Broader multi-format roadmap: https://github.com/kry4r/Nuka-Physics/issues/11
