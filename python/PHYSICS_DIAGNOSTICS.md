# Quantitative physics diagnostics

`nuka.diagnostics.DiagnosticSession` records production readouts from a `World`.
The session requests diagnostics before stepping, stores every environment and
physics interval, and preserves completed chunks when a run stops or fails.
Readouts do not update positions, velocities, impulses or DAT fractions.

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
| `ENERGY_LEDGER` | 32 float32 values | State energies, actual work, signed losses, original residual, validity |
| `PHYSICS_STAGE_METRICS` | 5 × 39 float32 values | Physical state momentum, finite counts, VBD force defect and discrete momentum |
| `CONTACT_AUDIT_COUNTS` | 8 uint32 values | Contact and violation counts over active contact rows |
| `CONTACT_AUDIT_METRICS` | 7 uint64 values | Maximum audit values and their global row witnesses |
| `VBD_EFFECTIVE_DT` | One float32 per VBD vertex, latest substep | Actual BE/BDF2 step, including per-vertex restarts |
| `VBD_ELEMENTS` | Shared cooked model, sixteen uint32 words per element | Exact elastic topology and material parameters |

Stages are `BEGIN`, `FREE`, `SOLVED`, `PROJECTED`, `END`.
`FREE` follows gravity and controls. Contact impulses and work are measured at
`SOLVED`; closest-feature gaps are measured at `END`, after DAT.
Python enums in `nuka.diagnostics.schema` define column order and stage units.
`decode_maxima` in `nuka.diagnostics.report` decodes packed values and row IDs.
A row ID belongs to its recorded interval; slots can be reused in later steps.

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

The current measured balances cover the VBD particle subsystem and particle
VF/EE contact features. General stage sampling includes bodies and articulated
links, but full articulated, rigid-load and MLS-MPM balance channels remain to
be implemented and independently validated. Analytic fixtures, reference-engine
comparisons, rendered acceptance and five-process GPU measurements are separate
required evidence; capture and reporting alone cannot pass full physics acceptance.
