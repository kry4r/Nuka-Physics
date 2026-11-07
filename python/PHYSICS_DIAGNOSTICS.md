# Quantitative physics diagnostics

`nuka.diagnostics.DiagnosticSession` records production readouts from a `World`.
The session requests diagnostics before stepping, stores every environment and
physics interval, and preserves completed chunks when a run stops or fails.
Readouts do not update positions, velocities or impulses.

`Field.CONTACT_PROFILES` contains 22 float32 words per material profile. Word 1
stores the bits of `ContactProfileFlag`; reinterpret it as uint32 before editing.
`GEOMETRIC_FRICTION` requests geometric-mean mixing of sliding, spinning and rolling
friction. At equal priority, either shape can request this rule; otherwise the
higher-priority shape supplies the coefficients. The default is maximum mixing.
Authored contact pairs keep their own coefficients. Mesh and primitive contacts
use the same rule, and zero geometric-mean friction disables tangential response.
Upload modified profiles through `World.upload_field`. Contact assembly uses the
updated parameters and invalidates caches whose resolved material values changed.

Production stepping uses discrete contact constraints without CCD or displacement
truncation. Historical DAT field IDs remain reserved: scalar counters and energy
columns are zero, while fraction, history and witness arrays have no elements.
The `BEFORE_DAT_*` energy columns describe the projected state before velocity
finalization. Offline trajectory geometry audits do not participate in stepping.

## Capture

```python
from nuka.diagnostics import DiagnosticSession

metadata = {
    "expected_policy_steps": 100,
    "solver_velocity_tolerance_mps": 3e-4,
    "ogc_contact_capacity": 524288,
}
with DiagnosticSession(world, ".nuka-runs/diagnostics/my-new-tag", metadata,
                       env_count=world.env_count) as session:
    for _ in range(100):
        session.step()
```

Set controls before each `step()` and pass their values as `controls=` when
needed. Use `state_fields=` to preserve physical states. A geometry audit
requires particle positions at every physics interval and a topology artifact
whose SHA256 is recorded as `initial_geometry_sha256` in the manifest.
`record()` copies the interval after an externally issued step; call it once
per step. After creation or reset, `ENERGY_LEDGER.VALID` is zero and all dependent
diagnostic decisions remain unmeasured until a complete interval is recorded.

## Time layers and layouts

| Field | Per environment and substep | Meaning |
|---|---|---|
| `ENERGY_LEDGER` | 44 float32 values | State energies, actual work, signed losses, original residual, validity, MPM transfer and material terms |
| `PHYSICS_STAGE_METRICS` | 5 × 44 float32 values | Physical momentum, finite counts, VBD force defect, discrete momentum and substep solver residuals |
| `CONTACT_AUDIT_COUNTS` | 8 uint32 values | Contact and violation counts over active contact rows |
| `CONTACT_AUDIT_METRICS` | 7 uint64 values | Maximum audit values and their global row witnesses |
| `VBD_EFFECTIVE_DT` | One float32 per VBD vertex, latest substep | Actual BE/BDF2 step, including per-vertex restarts |
| `VBD_ELEMENTS` | Shared cooked model, sixteen uint32 words per element | Exact elastic topology and material parameters |
| `VBD_SOLVE_AUDIT` | 20 float32 values per VBD vertex, latest solve call | Final force and Newton correction; local descent direction, scale, energy change and counters |

`DiagnosticSession(..., interval_states=True)` additionally retains the committed
particle positions and velocities, MPM deformation and affine matrices and plastic
history, collidable body poses and velocities, and joint coordinates, rates,
rotations, spatial velocities and base poses after every physical interval.
The corresponding `Field.INTERVAL_*` arrays have environment, substep, owner,
component order. `INTERVAL_STATE_DT` has environment and substep axes; zero marks
an unwritten or reset interval, whose other stored values must be ignored.
Non-MPM particle matrices are identity for deformation and zero for affine rate;
their plastic strain is zero. Body poses include articulated colliders after FK.
These optional buffers are allocated on first demand and have no solver consumers.
Enabling the readout rebuilds a captured execution plan without changing physical
updates. It increases capture memory and latency and belongs in untimed validation.

The C++ pipeline benchmark accepts `--interval-output <new-directory>` for the
same readout during its quality replay. `manifest.json` describes the byte offsets,
element sizes and counts in `initial.bin`, `inputs.bin` and each policy frame of
`intervals.bin`. Inputs include the cooked model and persistent data fields; their
typed layouts come from the bound `src/nk/model/fields.yaml`. The manifest remains
`recording` on interruption and becomes `closed` after replay, with the measured
`timed_replay_bit_equal` result. This export does not establish physical acceptance.
For MPM spatial convergence, `--mpm-grid-refinement n` divides cell and particle
spacing by `n`, preserves the material domain and total mass, and increases the
contact reserve with the particle count. `--substeps` controls time refinement.

`--quality-only 1` executes that complete quality replay without a preceding timed
pass. All warmup and measured-window physical steps still run, with optional
rendering and state export. Performance samples and timed replay binding are
unmeasured in this mode; it cannot supply a performance denominator.

Stages are `BEGIN`, `FREE`, `SOLVED`, `PROJECTED`, `END`.
`FREE` follows gravity and controls. Contact impulses and work are measured at
`SOLVED`; closest-feature gaps are measured at the committed `END` state.
Python enums in `nuka.diagnostics.schema` define column order and stage units.

`EXTERNAL_WORK` includes applied world-frame body forces and COM torques, including
articulation link wrenches. Their interval impulses are paired with the mean
`BEGIN`/`SOLVED` COM velocity and angular velocity. Loads are captured before the
step consumes them; position corrections do no external work. Integration error
remains in `RESIDUAL`. This readout allocates 48 additional bytes per body and per
link only when energy diagnostics are requested.
`decode_maxima` in `nuka.diagnostics.report` decodes packed values and row IDs.
A row ID belongs to its recorded interval; slots can be reused in later steps.

MPM kinetic energy includes the affine velocity field and the full compact-kernel
covariance. Stored energy follows the authored elastic law and recoverable J2
hardening. `MPM_P2G_LOSS` and `MPM_G2P_LOSS` measure the weighted least-squares
velocity residual at the entry stencil. Transfer roundoff remains in the energy
residual. `MPM_PLASTIC_LOSS` measures energy released by return
mapping. `MPM_AFFINE_TRANSPORT` and `MPM_VOLUME_AVERAGE_WORK` are signed changes
from representation transport and volume averaging. They are reported separately
and require convergence checks. `MPM_STRESS_WORK` is internal elastic work;
`MPM_BOUNDARY_WORK` includes reactions at prescribed grid velocities, and
`MPM_VISCOUS_LOSS` measures explicit viscous stress work. Implicit viscous stress
and nonzero stress on prescribed MPM particles retain incomplete coverage flags.
These observations use private buffers allocated on first energy demand.
`POSITION_KINETIC_WORK` compares kinetic energy at the projected pose with a
private pose advanced by physical velocity alone, including articulated inertia
and conserved rigid angular momentum. `POSITION_POTENTIAL` records the associated
potential change. These numerical contributions are separate from physical loads
and require convergence checks; joint load work excludes pseudo displacement.
`CONSTRAINT_STORAGE` estimates the conservative work of MuJoCo-law compliant normal,
pyramid-edge and limit rows over their solved displacement `dx = dt J v1`.
Each env's solved active set (rows strictly inside their bounds and friction cones)
factors `C = A + R` and solves two right-hand sides: `f_d = C^-1 (-B v0)` for
reference damping, and `C^-1 K dx` for the change in spring excitation. The conservative
share `f_c = f - f_d` includes coupled loads. Its frozen-set storage estimate is
`-f_c dot dx + 0.5 dx dot C^-1 K dx`; changing loads or active sets require independent
checks, and heterogeneous stiffness need not define a global potential.
`NORMAL_LOSS`, `LIMIT_LOSS`, and the normal component of a pyramid edge retain damping
impulse work at the average physical velocity. Spring discretization error remains in
the energy residual and must shrink under timestep refinement. Throughput retains its
gross row-work accounting. An active set larger than 128 rows leaves storage unbooked
and raises `STORAGE_CAPACITY`, so its records are not covered.
`Field.ENERGY_ROW_TERMS` keeps the last solved stage per row slot as `EnergyRowTerm`
columns (kind, impulse, work, storage, rate, conservative impulse, response, compliance), so a
negative loss or storage step can be traced to its rows.
Older captures with 31, 43 or 44 energy columns remain readable; missing appended terms
are represented by NaN and keep energy coverage incomplete.

`SOLVED` retains all four solver residual classes in m/s for every physical
interval. `SOLVER_RESIDUAL_MEASURED` is one when those residuals were written.
The separate `SOLVER_RESIDUAL` field and sweep traces describe the latest solve.
Under sweep tracing, `Field.BODY_MOMENTUM_RESIDUAL` and `Field.ARTICULATION_MOMENTUM_RESIDUAL`
keep each dense block's last momentum balance as `MomentumResidual` columns: the velocity
residual at the worst DOF, its force residual and the sum of the absolute terms forming it,
the largest force residual relative to that sum over all DOFs, and the largest incident row
term. A relative residual near float epsilon is rounding of the balance terms; a larger one
is a balance the multipliers do not meet.
Capture schema 2 includes the retained interval residuals; schema 1 records
remain readable and keep their missing residual coverage unmeasured.

Request `VBD_SOLVE_AUDIT` in `state_fields` before stepping to record individual
vertex descents. Its scope is the last solve call of the latest substep, including
the verification call when the solver uses one. Static vertices remain zero;
`TOTAL_STEPS > 0` identifies recorded dynamic vertices. `ACCEPTED_STEPS`,
`REJECTED_STEPS` and `ZERO_SLOPE_STEPS` sum to `TOTAL_STEPS`; `ROUNDED_STEPS`
counts accepted moves whose stored velocity is unchanged. `LAST_ENERGY_CHANGE`
is the accepted local potential change, or zero when no trial was accepted.
Final force is measured after neighboring blocks and dual updates; the last
primal force belongs to the local step before those updates. These quantities
serve diagnosis and do not replace solver or conservation acceptance gates.
The manifest declares optional columns and units for main records and budget
replays separately. Reports emit `vertex_solver_audit.json` when present.

Energy uses joules, time uses seconds, linear momentum uses kg·m/s and angular
momentum uses kg·m²/s. VBD force-equation velocity defect uses m/s. The physical
state momentum is separate from the step's BE/BDF2 discrete momentum:
`P_tilde = sum(m * (3*v_new - v_old)/2)` for actual BDF2 vertices. Restarted BE
vertices use `m*v`. Measures from different restart states do not telescope.
Preserve `VBD_EFFECTIVE_DT` with velocity states to independently reconstruct
mixed measures using `analysis.vbd_discrete_momentum`. A global BDF2 setting or
an aggregate count cannot recover which individual vertices restarted at BE.

## Reproducible twisting cloth

```bash
export PYTHONPATH=python
export NUKA_SOLVER_VEL_TOLERANCE=3e-4
python python/bench_cloth_twist.py \
  --output .nuka-runs/diagnostics/twist-new-tag \
  --build-record .nuka-runs/builds/my-build-record
python python/physics_diagnostics.py audit .nuka-runs/diagnostics/twist-new-tag \
  --geometry .nuka-runs/diagnostics/twist-new-tag/initial.npz
python python/physics_diagnostics.py verify-particles .nuka-runs/diagnostics/twist-new-tag \
  --geometry .nuka-runs/diagnostics/twist-new-tag/initial.npz \
  --output .nuka-runs/diagnostics/twist-new-tag/particle_state_check.json
python python/physics_diagnostics.py verify-elastic .nuka-runs/diagnostics/twist-new-tag \
  --geometry .nuka-runs/diagnostics/twist-new-tag/initial.npz \
  --output .nuka-runs/diagnostics/twist-new-tag/elastic_state_check.json
python python/physics_diagnostics.py report .nuka-runs/diagnostics/twist-new-tag \
  --output .nuka-runs/diagnostics/twist-new-tag-report --plot
```

The fixture defaults to 101 × 61 vertices, 5 mm spacing and eight relative turns
over 12.8 s. It records the actual gravity, material, solver settings, binary
hashes and controller inputs. The first DAT truncation stops sampling and
preserves evidence. `--stop-after` produces an explicitly incomplete prefix.
Every output directory must be new.

The audit uses external `ipctk.TightInclusionCCD` with tolerance `1e-6`, minimum
distance `0` and conservative rescaling `1`. Default rescaling `0.8` alerts are
reported separately. Hashes, interval continuity and audit configuration must
match before the report accepts geometry evidence. Multi-substep traces require
intermediate geometry capture; end-of-policy positions cannot certify them.

## Sweep-budget replay

`session.replay_budgets(budgets, state_fields=...)` replays the next step once
per velocity-sweep budget from one checkpoint and writes
`budget_replay_step_NNNNNN.npz` with `budgets` and one readout row per budget.
Each budget is applied with `World.set_velocity_iterations`; afterwards the world
returns to the checkpoint at the production budget read from
`World.velocity_iterations()`, so the main record is unchanged. The L1 bench accepts
`--replay-step N --replay-budgets first:last` (or a comma list) and
`--replay-stop` to end the record after that step. States at consecutive budgets
give the solver's actual sweep-by-sweep trajectory for that step.

Articulated fixtures can add `JOINT_VELOCITY`, `JOINT_LIMIT_IMPULSE` (lower and
upper limit-row impulses per link) and `LINK_CONTACT_WRENCH` (contact force and
torque per link) to `state_fields=`.

## Evidence and decisions

| Artifact | Contents |
|---|---|
| `manifest.json`, `chunk_*.npz` | Axes, settings, units, stop reason and hashed raw records |
| `report.json`, `report.md` | Per-layer decisions and unchanged thresholds |
| `ledger.parquet` | Energy terms per environment and substep |
| `diagnostic_timeline.json` | First anomaly and count for each measured failure class |
| `events.json` | Largest energy residuals, force-work comparison and interval witnesses |
| `contact_audit.json` | Impulse closure, cone, normal impulse and actual VF/EE gap checks |
| `ipctk_audit.json` | External trajectory audit and separate safety alerts |
| `particle_state_check.json` | Independent float64 state formulas and actual mixed BE/BDF2 momentum |
| `elastic_state_check.json` | Independent cooked-material potential and internal-gradient sum |
| `diagnostics_env*.png` | State, residual, momentum, force and work plots |
| `perf.json` | Host step wall time and explicit GPU measurement coverage |

Run `compare source-a source-b --output new-comparison.json` through the CLI
to compare matching physical inputs at common sampled times. This reports state
differences; it does not certify a speedup. `analysis.particle_state_quantities`
independently evaluates particle K/G/P/L in float64. `predict_linear_modes`
computes linear BE/BDF2 modal energy predictions for applicable isolated models.
`verify-particles` requires a declared particle-only world, captured masses,
initial velocities and actual effective steps. It reports readout differences
without turning numerical agreement into a full physics acceptance result.

`verify-elastic` uses the captured cooked elements and actual END positions.
`constitutive.elastic_quantities` independently evaluates StVK with the area
barrier, dihedral hinges, axial springs and rod bends. Optional rates reconstruct
relative candidate geometry; explicit float32/float64 arithmetic supports
roundoff diagnosis. Degenerate geometry and unknown material kinds fail visibly.
Element words are `kind`, four vertex indices, eight float32 rest parameters,
float32 damping and two reserved words. Triangle rest parameters are row-major
inverse material edges, rest area, mu and lambda. Hinge parameters are rest angle
and stiffness; spring parameters are rest length and stiffness per length;
rod bend uses its energy coefficient. The remaining parameters are unused.

Missing coverage is unmeasured, not passed. Empty contact traces do not validate
contact behavior. Frozen-Jacobian gap estimates are auxiliary; only actual
closest-feature gaps enter the gap decision. Force residual work is never
subtracted from the original energy residual. Signed negative dissipation and
the first nonfinite stage remain visible.

## Required coverage and numeric evidence

Declare `required_physical_systems` and `required_ccd_contact_domains` in capture
metadata. Measured energy and linear momentum channels currently support `vbd`. Declare dynamic
rigid bodies, articulations, MPM and other particle systems when present; their
missing complete balances keep full acceptance unmeasured. Static boundaries
belong in the contact-domain inventory. The mesh audit covers
`particle_mesh_self`; `particle_static`, `particle_articulation` and
`articulation_self` need their own complete geometry evidence.

The geometry artifact must contain actual initial `positions` or `position`,
initial velocity, masses and topology. Material rest coordinates alone cannot
establish the first trajectory segment. A legacy `rest` array is accepted only
with the explicit metadata contract `initial_positions_equal_rest: true`.
Particle positions are read a chunk at a time through `iter_record_blocks`;
consuming the complete iterator verifies hashes, extents and continuity.
The audit independently checks initial mesh intersections and exact degenerate
edges/faces before CCD. Invalid initial environments retain their failure and
recorded segment counts; their trajectories are skipped and cannot pass.
Zero-distance CCD does not certify finite contact offsets or rotating rigid trajectories.

The original relative momentum gate and the independently recomputed
solver-tolerance closure must both pass. Newton and contact residuals need valid
packed measurements; absent readouts cannot pass as zero. Energy and stage
intervals must have the same positive finite timestep. Replay artifacts are
registered in the manifest with a SHA256, checkpoint step, requested and active
budgets, and the restored production budget. The largest replay budget must
meet the production residual tolerance; this does not waive quality at the
production budget.

Independent state and elastic checks are required report layers. Declare
nonnegative absolute measurement bounds in `independent_readout_limits` before
capture for `kinetic_j`, `gravity_j`, `mass_kg`, `linear_momentum_kg_mps`,
`angular_momentum_kg_m2ps`, `discrete_momentum_kg_mps`, `elastic_energy_j` and
`internal_gradient_sum_n`. Derive these bounds from the diagnostic arithmetic
and input scale; they measure readout accuracy and do not replace conservation
or solver thresholds. Missing bounds, hashes, environments or intervals leave
these checks unmeasured. Incorrect BE/BDF2 vertex counts fail the state check.

Analytic/convergence, reference-engine and angular balance results can enter the report through
`analytic_acceptance.json`, `reference_acceptance.json` and `angular_momentum_acceptance.json`. Define required checks
in `metadata.quantitative_acceptance` under the report layer names
`L1_analytic_and_convergence`, `reference_engines` and `L2_angular_momentum`. Each definition has a unique
`name`, a `unit` and a finite `minimum` and/or `maximum`. The corresponding result
contains the source `source_manifest_sha256`, matching `physical_input_sha256`,
and a `checks` object keyed by name. Each check stores its numeric `value`, the
same `unit`, and a nonempty list of `artifacts` with run-relative `file` and
`sha256`. The reporter recomputes decisions from values and declared bounds;
an external `passed` flag is insufficient.

Angular balance must account for external and boundary torque, contact impulse
moments and the actual integrator. State angular momentum or the force-equation
angular defect alone cannot establish conservation; missing balance evidence
keeps this required layer unmeasured.

Those producers must preserve the actual initial conditions, controls,
constitutive parameters, integrator and reference-engine versions and model
differences. A bounded shape difference alone does not establish conservation.
Modal energy predictions apply to isolated linear models; integration losses
remain separate from friction, material damping and solver defects, and are not
subtracted from the original residual.

`compare` requires closed completed captures, identical declared physical inputs
and initial particle states, valid finite readouts and the same complete physical
horizon. Supply runs in decreasing timestep order. Three runs at `h`, `h/2` and
`h/4` report pairwise position, velocity and energy RMS at their common sample
times and the observed convergence order. Zero differences are marked
`roundoff_limited`; observed order alone does not pass physical acceptance.

Full engine acceptance additionally needs the complete control/reset/state/sensor
and rendering pipeline. Performance acceptance needs five independent processes,
GPU completion timing, complete pipelines at equal physical quality, memory and
capacity accounting, and latency distributions. `perf.json` continues to mark
these requirements unmeasured when their evidence is absent.

The pipeline benchmark additionally requires active rows and positive normal
impulse in each environment's timed window for every required system pair.
Warmup impulses and timed zero-impulse rows cannot establish workload coverage.
Its `status.valid` covers the reported lifecycle, state, render, deformation and
coupling contracts; separate fields mark full physics acceptance unmeasured.

The current measured balances cover the VBD particle subsystem and particle
VF/EE contact features. General stage sampling includes bodies and articulated
links, but full articulated, rigid-load and MLS-MPM balance channels remain to
be implemented and independently validated. Analytic fixtures, reference-engine
comparisons, rendered acceptance and five-process GPU measurements are separate
required evidence; capture and reporting alone cannot pass full physics acceptance.
