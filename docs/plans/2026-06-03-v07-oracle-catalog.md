# v0.7 Oracle Catalog (phase p15) — exit criteria 1/2/3

**Date:** 2026-06-03 · **Branch:** v07 · **Phase:** p15
**Backs v0.7 exit criteria:** 1 (XPBD soft/cloth), 2 (PBF fluid), 3 (sparse-SDF cooker + Newton contact + analytical adjoint).

---

## ★ GATE-WORDING DEVIATION (read first — must be surfaced at the v0.7 EXIT AUDIT)

The v0.7 exit criteria are worded:

- **crit 1:** XPBD soft/cloth operational — "oracle vs **Vellum/Flex** passing".
- **crit 2:** PBF fluid + internal density — "oracle vs **Flex paper** passing".
- **crit 3:** sparse-SDF cooker + Newton contact + analytical adjoint — "oracle vs **MuJoCo-SDF**".

**Vellum, NVIDIA Flex, and MuJoCo are NOT available in this environment.** There is no external engine to run, and none was run. p15 therefore **cannot** literally execute "oracle vs Vellum/Flex/MuJoCo". This catalog meets the intent of crit 1/2/3 by a **substitute** (shipped progressively across p08/p09/p10 and inventoried here): every oracle entry below is one of

- **(a)** an **analytic / closed-form invariant** (a number derived from the math, independent of the engine), or
- **(b)** a **self-consistency / finite-difference (FD) check** (the engine's analytic derivative vs a numerical re-derivation of the *same* primal — internal consistency, NOT an independent ground truth), or
- **(c)** a number transcribed from a **cited published paper**.

**No catalog entry claims a Vellum/Flex/MuJoCo run; none happened.** Where an oracle is *self-consistency* (type b) rather than an *independent* oracle, the table says so in plain words — the project's standing discipline (e.g. the SDF adjoint's "host-FD is self-consistency, the physical test is the real evidence", and `PointDegeneracyIsDocumented`).

This deviation is to be carried to the v0.7 exit audit **exactly as** the crit-6 C-ABI camera-arm deferral (task #29) is carried: as an explicitly-stated substitution, not a silent pass. **Failure mode avoided:** a catalog that reads "✓ vs Vellum/Flex" when no Vellum/Flex ran.

**One additional crit-2 honesty flag (see §Gaps):** crit 2's named "dam-break front speed vs analytic (~4.71 m/s)" oracle is **NOT** met by a front-speed-vs-analytic match. The shipped test asserts a one-sided *energy/stability upper bound*, not a front-speed equality, and the 3D inviscid SPH front does not reproduce the 1D shallow-water (Ritter) analytic (measured + explained below). This is the one place crit wording is materially stronger than the shipped oracle.

---

## How to reproduce (build + run)

Build dir `build-cuda128`. Environment:

```
export PATH="/root/.nuka-toolchain-gcc14/bin:/opt/cuda-12.8-root/usr/local/cuda-12.8/bin:$PATH"
export CUDA_HOME=/opt/cuda-12.8-root/usr/local/cuda-12.8
export LD_LIBRARY_PATH="/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0
```

Targets: `nuka_adjoint_fd_test`, `nuka_cuda_runtime_test`, `nuka_pbf_test`,
`nuka_sparse_sdf_test`, `nuka_sdf_contact_adjoint_test`, `nuka_sdf_contact_ift_test`.

**Reading the "measured value" column.** Several oracles emit their measured value
only via `printf` (SDF IFT/adjoint), via `RecordProperty` (PBF `--gtest_output=xml`),
or *only in the EXPECT failure message* (the V3 FD `max_rel_err`; the XPBD physical
oracles emit **nothing on pass** — they assert a bound). For those that print nothing
on pass, the cell states **"PASS; asserts bound < X"** with the bound read from the
test source, plus any **closed-form value computed independently** (a pure function of
the setup constants). The V3 FD `max_rel_err` values below were reproduced by a
throwaway that links `libnuka_codegen_v3_validation.a` and calls the `Validate…Fd`
harness functions directly (the same harness the tests run). Every number in this
catalog was produced on a real run on 2026-06-03; no number is taken from prior notes.

---

## crit 1 — XPBD soft / cloth

Four codegen row classes (ids 6/7/8/9), each shipping a **genuine dispatchable per-row
reverse-mode adjoint** (`HasAdjoint(id)==true`, `dense_adjoint`). Two oracle layers:
(A) **V3 finite-difference** of the analytic adjoint vs a central-difference of the *same*
generated primal — **type (b) self-consistency**; (B) **physical/analytic invariants** —
type (a), the real physical evidence.

### (A) V3 FD adjoint — all 4 rows + the K3 coupling row (type b, self-consistency)

| Subsystem (row id) | What it validates | Basis | Measured value + tolerance | Test target |
|---|---|---|---|---|
| XPBD distance (id 6) | analytic adjoint == numerical Jacobian of the generated primal | FD self-consistency, 100 random cases | **max_rel_err = 6.104e-04** < 1e-3 · PASS | `nuka_adjoint_fd_test` → `AdjointFdXpbdDistance.MatchesNumericalJacobian` |
| XPBD bend (id 7) | "" (isometric Bergou 2006 stencil multiplier law) | FD self-consistency, 100 cases | **6.104e-04** < 1e-3 · PASS | `AdjointFdXpbdBend.MatchesNumericalJacobian` |
| XPBD volume (id 8) | "" (det-gradient multiplier law) | FD self-consistency, 100 cases | **6.104e-04** < 1e-3 · PASS | `AdjointFdXpbdVolume.MatchesNumericalJacobian` |
| XPBD shape-match (id 9) | "" (per-particle goal-projection law) | FD self-consistency, 100 cases | **2.623e-05** < 1e-3 · PASS | `AdjointFdXpbdShapeMatch.MatchesNumericalJacobian` |
| particle-particle (id 10, K3) | "" (2-particle unilateral non-penetration coupling) | FD self-consistency, 100 cases | **7.883e-05** < 1e-3 · PASS | `AdjointFdParticleParticleContact.MatchesNumericalJacobian` |

Determinism (D1) for each row: identical-seed FD is bit-identical across two runs
(`*.IsDeterministicAcrossRuns`, all PASS).

> **Honesty note (type b):** V3 FD differences the analytic adjoint against the
> *same* generated primal — it proves the derivative is internally consistent, NOT
> that the forward physics matches an external engine. The physical invariants in
> (B) are the real (analytic, type-a) evidence; the V3 FD is the gate that the
> shipped adjoint is dispatchable and correct *for that primal*.

### (B) Physical / analytic invariants (type a, the real evidence)

All run in `nuka_cuda_runtime_test` (filter `Xpbd*`). These assert physical bounds and
emit nothing on pass, so the measured cell is the **asserted bound** (+ any
independently-computed closed form). All PASS (12/12 in this group).

| Subsystem | What it validates | Basis | Measured / asserted | Test target |
|---|---|---|---|---|
| distance row — oscillation period | a stretched distance pair oscillates at the **analytic discrete period** | closed-form (type a): `T_disc = 2π·dt / acos(√γ)`, `γ = α/(α+w·dt²)` | observed zero-crossing period vs **closed-form T_discrete = 0.888834 s** (T_continuous = 0.888577 s; disc-vs-cont 0.029%); test asserts rel-err < 2% · PASS | `XpbdDistanceOracle.OscillationPeriodMatchesAnalyticDiscrete` |
| distance row — D1 forward | predict/solve/correct byte-exact across two runs | byte memcmp | pos+vel memcmp == 0 · PASS | `XpbdDistanceOracle.ForwardIsByteExactAcrossRuns` |
| bend row — isometric stencil | linear-precision conditions `Σkᵢ=0`, `Σkᵢxᵢ=0` (pin the cotangent stencil up to scale) | closed-form (type a) | both Σ within 1e-5 · PASS | `XpbdBendCloth.IsometricStencilHasLinearPrecision` |
| bend row — grad C is constant + FD | ∇C is config-independent and == host FD (catches a nonlinear regression) | host FD self-consistency | analytic vs FD rel < 1e-3, ∇C config-independent within 1e-4 · PASS | `XpbdBendCloth.BendGradientIsConstantAndMatchesFd` |
| bend row — flatten a folded flap | **stiff bend flattens** a folded flap while a **no-bend control stays folded** (the real grad-C-does-work evidence) | physical (type a); non-planarity via triple product | stiff < 0.2·initial, control > 0.8·initial · PASS | `XpbdBendCloth.StiffBendFlattensFoldedFlapWhileNoneStaysFolded` |
| bend row — drape | gravity drape stays finite, near-inextensible, D1 byte-exact | physical + D1 | edge drift < 5%, pos+vel memcmp == 0 · PASS | `XpbdBendCloth.DrapeIsFiniteNearInextensibleAndByteExact` |
| volume row — det gradient | analytic cross-product gradient == host FD; gradients sum to 0 (momentum) | host FD self-consistency + closed form | analytic vs FD rel < 1e-3, Σ∇ within 1e-4 · PASS | `XpbdVolumeTet.DeterminantGradientMatchesFdAndSumsToZero` |
| volume row — isolated volume restore | a tet compressed to 60% rest volume, **no edge constraints**, is driven back to rest volume by the volume row alone | physical (type a) | rel-err to rest volume < 1e-2 · PASS | `XpbdVolumeTet.IsolatedVolumeConstraintRestoresRestVolume` |
| volume row — block + D1 | a tet block under gravity keeps each tet's volume near rest, D1 byte-exact | physical + D1 | volume drift < 5%, pos+vel memcmp == 0 · PASS | `XpbdVolumeTet.BlockPreservesVolumeUnderGravityAndIsByteExact` |
| shape-match — polar dR/dA | analytic polar-decomposition derivative == host central-difference (fp64) + R is a proper rotation | host FD self-consistency (fp64) | max rel-err < 1e-3, det(R)=1 within 1e-9 · PASS | `XpbdShapeMatch.PolarDerivativeMatchesFdAndIsProperRotation` |
| shape-match — relaxes to rigid | a non-rigidly-deformed cluster (stretch+shear) **relaxes to a rigid transform** of the rest shape (rest pairwise distances recovered) + recovered rotation == polar(A_init) (anti-identity-guarded, non-vacuous) + D1 | physical (type a) + D1 | rest pairwise drift < 1e-2, rotation drift < 1e-2, pos+vel memcmp == 0 · PASS | `XpbdShapeMatch.NonRigidClusterRelaxesToRigidShapeAndIsByteExact` |

### K3 coupling (particle-particle) physical oracles (supports crit 1, used by v1.0 coupling)

Run in `nuka_cuda_runtime_test` (filter `ParticleParticle*`), all PASS (7/7):
separation to exactly `d_min`, mass-weighted centroid conserved under unequal masses,
soft↔fluid pair separates, asymmetric multi-pair segment-reduction, D1 two-run
byte-identical, subsystem-paths byte-identical with no cross-pairs.

**crit 1 verdict:** richly oracled by analytic invariants + FD self-consistency. The
"vs Vellum/Flex" wording is met by these substitutes; **no Vellum/Flex run exists** —
the Vellum/Houdini golden trajectory is deferred (Vellum not reproducible here).

---

## crit 2 — PBF fluid + internal density

PBF is **forward-only + internal** (master plan §3 Round 7): no codegen row, no adjoint,
no V3 FD. Oracles are analytic/physical invariants (type a) run in `nuka_pbf_test`
(14/14 PASS). `rho0` is calibrated numerically from the engine's own Poly6 kernel, so a
kernel-normalization slip is absorbed — the equilibrium `C_i = rho_i/rho0 − 1 = 0` is
independent of the kernel constant (units matter only for the deferred real-units Flex
oracle).

| Subsystem | What it validates | Basis | Measured value + tolerance | Test target |
|---|---|---|---|---|
| rest-density block | an interior rest block stays at rest after a step (no blow-up / contraction) | physical (type a); calibrated `C_i` | interior max\|C_i\| < 0.10, drift < 0.25·dx; cap-truncation == 0 · PASS | `nuka_pbf_test` → `PbfDensityIteration.RestDensityBlockStaysAtRest` |
| over-compression relaxation | an over-compressed block's interior over-density **strictly decreases** after one density projection (the load-bearing solver-correctness check) | physical (type a) | C_after < 0.9·C_before (genuine work) · PASS | `PbfDensityIteration.OverCompressedBlockRelaxesTowardRest` |
| column stability / energy bound | a column under gravity stays finite, above floor, count-invariant, peak speed under the energy scale (no energy injection) | physical bound (type a) — see §Gaps for the front-speed caveat | peak speed < `2.5·√(2·g·H0)`; finite; on floor · PASS | `PbfDensityIteration.ColumnUnderGravityStaysStableAndBounded` |
| D1 forward | predict/density-iters/finalize byte-exact across two runs | byte memcmp | pos+vel memcmp == 0 · PASS | `PbfDensityIteration.ForwardIsByteExactAcrossRuns` |
| bulk volume conservation | bulk interior density (∝ 1/volume) holds across 200 steps, no gravity | physical (type a); density proxy `V=N·m/ρ` | **drift = 0.000%** (`bulk_volume_drift_pct`=0) < 5% · PASS | `PbfVolumeConservation.BulkRestBlockHoldsVolume` |
| one-sided incompressibility | a column's interior never over-compresses beyond tolerance (projection relieves impact) | physical (type a) | **max over-density = 1.696%** (`max_interior_over_density_pct`=1696) < 15% · PASS | `PbfVolumeConservation.ColumnInteriorNeverOverCompresses` |
| XSPH viscosity (M&M 2013 eq.17) | a sheared block's velocity variance is reduced by viscous smoothing | physical (type a) | var **1.006323 → 0.925205 (−8.06%)** (`var_visc_off/on_e9`) · PASS | `PbfViscosity.ReducesVelocityVarianceUnderShear` |
| viscosity off-gate + D1 | c==0 byte-identical to no-viscosity path (skip, not scale-by-0); viscous run two-run byte-exact | byte memcmp | memcmp == 0 · PASS | `PbfViscosity.ZeroCoefficientIsByteIdenticalToNoViscosity`, `…ViscousForwardIsByteExactAcrossRuns` |
| surface-tension cohesion (Akinci 2013) | a detached blob's mean pairwise distance decreases (contracts / stays connected) | physical (type a) | mean pairwise **0.196724 → 0.192087 (−2.36%)** (`mean_pairwise_dist_0/1_e6`) · PASS | `PbfSurfaceTension.CohesionContractsBlob` |
| cohesion off-gate + D1 | γ==0 byte-identical to no-tension path; cohesive run two-run byte-exact | byte memcmp | memcmp == 0 · PASS | `PbfSurfaceTension.ZeroGammaIsByteIdenticalToNoTension`, `…CohesiveForwardIsByteExactAcrossRuns` |
| V2 particle-count invariant | count exactly conserved over 300 steps with viscosity + cohesion both ON; all finite | engine self-check (type a) | count invariant, all finite · PASS | `PbfV2Invariant.ParticleCountConservedWithPolishOn` |

**Cited published-paper numbers (type c):** none transcribed. The Macklin–Müller
"Position Based Fluids" (2013) and Akinci et al. (2013) cohesion are cited as the
*method/equation* sources (the XSPH eq.17 and the cohesion spline are transcribed
**equations**, not numeric magnitudes). The engine calibrates `rho0` internally and the
cohesion spline is max-normalized to 1 (so γ is an O(1) demonstrative knob, not a
physical surface-tension magnitude — documented in `pbf_world` headers), so a real-units
PBF/Flex *numeric* paper value is not reproducible here; transcribing one would not be a
number the engine can actually hit. Honest non-transcription rather than a forced number.

**crit 2 verdict:** well-oracled by analytic/physical invariants; "vs Flex paper"
wording met by these substitutes; **no Flex run exists** — the NVIDIA Flex
ball-into-water golden is deferred (Flex not reproducible here). See §Gaps for the
front-speed wording caveat (the one materially-weaker-than-wording item).

---

## crit 3 — sparse-SDF cooker + Newton contact + analytical adjoint

Three oracle clusters: the **cooker** (analytic primitive SDFs), the **system-level
IFT** d/dM,d/dJ (independent Eigen oracle — the strongest, type b-but-independent), and
the **envelope-theorem depth adjoint** (FD re-running the descent).

### Cooker + sampler (analytic, type a) — `nuka_sparse_sdf_test` (10/10 PASS)

| Subsystem | What it validates | Basis | Measured value + tolerance | Test target |
|---|---|---|---|---|
| box SDF at grid nodes | cooked node values == analytic box SDF to float epsilon (cube mesh IS the box, no tessellation error — exact gate) | closed-form box SDF (type a) | **node maxerr = 5.34e-08** (0.0001% of voxel) · PASS | `nuka_sparse_sdf_test` → `SparseSdfKnownShapes.BoxAgreesWithAnalyticalToFloatEpsilon` |
| sphere SDF near surface | cooked node values == analytic sphere SDF (tessellation-limited) | closed-form sphere SDF (type a) | **node maxerr = 1.064e-03** (1.773% of voxel) · PASS | `…SphereAgreesWithAnalyticalNearSurface` |
| capsule SDF near surface | cooked node values == analytic capsule SDF | closed-form capsule SDF (type a) | **node maxerr = 5.251e-04** (1.313% of voxel) · PASS | `…CapsuleAgreesWithAnalyticalNearSurface` |
| sampler sign + gradient | the shipped `sparse_sdf_sample` returns correct sign and gradient on a sphere | closed-form (type a) | sign + gradient correct · PASS | `…SphereSamplerSignAndGradient` |
| thin-shell two-sided | a thin panel is resolved both sides, gradient flips across it | physical (type a) | gradient flips · PASS | `SparseSdfThinShell.PanelResolvedBothSidesGradientFlips` |
| narrow-band memory | narrow-band storage is a fraction of dense | invariant | band ≪ dense · PASS | `SparseSdfMemory.NarrowBandIsFractionOfDense` |
| cook determinism / dedup | repeated cooks byte-identical; identical meshes share one SDF; sign correct on the convex-piece cook path | byte memcmp + invariant | byte-identical; dedup; sign correct · PASS | `SparseSdfDeterminism.*`, `SparseSdfDedup.*` |

### System-level IFT d/dM, d/dJ — `nuka_sdf_contact_ift_test` (6/6 PASS)

This is the §4.B acceptance: it perturbs the actual `M⁻¹`/`J` entries, **re-solves the
global constraint system to convergence with an INDEPENDENT Eigen LDLT/FullPivLU oracle**
(not the module's own assembly/solve), and central-differences λ — so a transpose/index
bug in either side surfaces. This is the **strongest crit-3 oracle: an independent
re-solve, not pure self-consistency.**

| Subsystem | What it validates | Basis | Measured value + tolerance | Test target |
|---|---|---|---|---|
| Delassus assembly vs Eigen | module `A = J M⁻¹ Jᵀ` == independent Eigen assembly; module solve == Eigen LDLT | **independent Eigen oracle** | **assemble max abs diff = 3.553e-15** (scale 4.732e+01) < 1e-9·scale; solve < 1e-9 · PASS | `nuka_sdf_contact_ift_test` → `SdfContactIft.AssembleDelassusMatchesEigen` |
| d/dM (perturb M⁻¹ + re-solve) | analytic IFT `dλ = A⁻¹(−J dM⁻¹ Jᵀ λ)` == FD of independent re-solve | independent Eigen re-solve FD | **max rel-err = 2.937e-09** < 1e-3 · PASS | `SdfContactIft.DMatchesPerturbAndResolve` |
| d/dJ (perturb J + re-solve) | full IFT (dA both terms + dr=−dJ·qdot_free) == FD re-solve | independent Eigen re-solve FD | **max rel-err = 2.277e-09** < 1e-3 · PASS | `SdfContactIft.DJMatchesPerturbAndResolve` |
| d/dJ drop-dr guard | dropping the dr term **must break** the match (proves the FD is system-level, not a per-row dA-only check) | discriminating guard | full **2.577e-11** < 1e-3; dr-dropped **1.000e+00** > 1e-2 · PASS | `SdfContactIft.DJDropDrTermFailsAsExpected` |
| coupled multi-slot (n=6, dof=8) | genuinely coupled Delassus (two engaged slots), d/dM + d/dJ | independent Eigen re-solve FD | d/dM **4.207e-06**, d/dJ **1.041e-07**, both < 1e-3 · PASS | `SdfContactIft.CoupledMultiSlotDMandDJ` |
| D1 | d/dM, d/dJ directional derivatives byte-identical across two runs | byte memcmp | memcmp == 0 · PASS | `SdfContactIft.D1TwoRunByteExact` |

### Envelope-theorem depth adjoint + SDF sample/gradient — `nuka_sdf_contact_adjoint_test` (7/7 PASS)

The FD **re-runs the Newton descent** (so p* re-optimizes) to confirm the envelope claim
`dV/dθ = ∂f/∂θ|_{p* frozen}` within tolerance (not merely that the sampler is
differentiable).

| Subsystem | What it validates | Basis | Measured value + tolerance | Test target |
|---|---|---|---|---|
| depth / d(cell value) | `∂depth/∂cellvalue = −w[corner]`; cell value does not move p* → envelope error exactly 0 (machine-precise) | analytic + FD re-run (type a/b) | **max rel-err = 1.079e-06** < 1e-3 · PASS | `nuka_sdf_contact_adjoint_test` → `SdfContactAdjoint.DepthVsCellValueIsMachinePrecise` |
| depth / d(translation) | `∂depth/∂t = +grad_world`; FD re-runs the descent → carries the envelope O(residual) | analytic + FD re-run | on-axis **abs-err = 1.071e-03** (cooked-gradient discretization-limited; forward residual 1.485e-07; off-axis ~1.6e-6); test asserts within tol · PASS | `SdfContactAdjoint.DepthVsTranslationEnvelopeWithinTolerance` |
| normalize-Jacobian building block | the `n/|n|` Jacobian is exact (witness-free) | closed-form (type a) | **max abs err = 5.960e-08** · PASS | `SdfContactAdjoint.NormalizeJacobianIsExact` |
| normal-θ degeneracy (documented) | the normal-vs-θ gradient is **degenerate** (witness tracks the surface across the flat valley) → **NO normal-θ gradient shipped**, demonstrated empirically | documented degeneracy (mirrors `PointDegeneracy`) | re-converged FD ≈ 0 vs frozen-p* partial ~0.25/2.07 → diverge (degenerate) · PASS | `SdfContactAdjoint.NormalVsCellGradientIsDegenerate`, `…NormalVsTranslationIsDegenerate` |
| contact-point degeneracy (documented) + implicit forward-depth | the witness point is non-unique along the flat valley (two seeds → same depth/normal, different point); the equal depth also implicitly checks the **forward** penetration against the analytic box-sphere value | documented degeneracy + closed-form (type a) | depth1=depth2=**0.0999** ≈ analytic **0.1** (unit cube face at x=0.5, sphere x=0.7 r=0.3 → surface x=0.4 → pen = 0.1), point.x 0.45 vs 0.46 · PASS | `SdfContactAdjoint.PointDegeneracyIsDocumented` |
| D1 | depth/cellvalue + normal/cellgrad byte-identical across two runs | byte memcmp | memcmp == 0 · PASS | `SdfContactAdjoint.D1TwoRunByteExact` |

> **Honesty note (crit-3 scope, mirrors p08-close notes):** the analytical adjoint
> ships **depth** d/dM,d/dJ + the envelope depth gradient. It does **NOT** ship a
> **normal-θ** gradient (the witness tracks the surface across the flat valley — the
> frozen-p* model over-states it; empirically falsified and documented, not fabricated)
> nor a contact-point gradient (non-unique witness). The v0.7 gate never backprops
> through SDF contact (default `ContactFree` stop-grad). SDF-contact reverse-mode
> wiring + normal-sensitivity is explicitly OPEN, re-scoped to v1.0 (tasks #23/#24,
> consumer = the v1.0 coupled diff-sim demo). This is carried OPEN to the exit audit.

**crit 3 verdict:** well-oracled. The cooker has analytic primitive oracles; the IFT
d/dM,d/dJ has an **independent Eigen oracle** (the strongest of the three crits); the
envelope depth adjoint is FD-confirmed by re-running the descent. The "vs MuJoCo-SDF"
wording is met by these analytic + independent-Eigen substitutes; **no MuJoCo run
exists.** The normal-θ / point gradient and the SDF-contact backward wiring are
explicitly OPEN (v1.0), exactly as recorded since p08.

---

## Gaps + how filled

**Surveyed coverage verdict:** all three crits are richly oracled; **no crit is left
without ANY oracle.** Per the p15 discipline (prefer cataloging; fill only genuine
crit-bare gaps; don't over-build), **no new test was added.** The catalog itself is the
deliverable for the well-covered subsystems.

### Gap 1 (the one real wording-vs-oracle gap) — crit-2 dam-break front speed

crit-2's named oracle is "dam-break front speed vs analytic (~4.71 m/s)". **No shipped
test measures a front speed against an analytic value.** The closest test
(`PbfDensityIteration.ColumnUnderGravityStaysStableAndBounded`) asserts a one-sided
**energy/stability upper bound** `peak_speed < 2.5·√(2·g·H0)`, and the "~2·√(2·g·H0)"
dam-break formula appears only as a **comment** (the "4.71 m/s" of the crit wording does
not correspond to any quantity in the current test — likely an earlier scene). A front
speed *upper bound* is materially weaker than a front-speed *equality* oracle.

**How investigated (empirically, not assumed).** A throwaway probe collapsed a tall
water column on a frictionless floor (inviscid PBF, released from rest) and tracked the
leading-front x over time:

For the probe column (H0 = 0.975 m):

- Measured **terminal front speed ≈ 3.22 m/s** (stable to 4 sig figs across t=0.58–1.25 s).
- Ritter shallow-water leading front `2·√(g·H0)` = **6.19 m/s**.
- The comment formula `2·√(2·g·H0)` = **8.75 m/s**.

The measured 3.22 m/s lies **below both** analytic conventions (~2× below Ritter), so this
is not a "which analytic formula" error — there is **no convention under which the 3D
inviscid free-surface SPH front reproduces the 1D shallow-water value.** (Ritter assumes
hydrostatic depth-averaged infinite-width flow; the 3D SPH column has a finite-width free
surface, and on a frictionless floor the lead particle coasts ballistically — so a
"terminal front speed" is a boundary-condition artifact, not a hydrodynamic front law.)

**Decision (advisor-ratified):** do **NOT** add a front-speed-vs-analytic test.
Forcing it would require a non-physics-derived ~2× tolerance (tuning-to-pass), and a
"terminal front = 3.22 m/s" self-consistency test would enshrine the engine's own output
as an "oracle" — validating nothing external, redundant with the existing stability
bound, and actively misleading. Instead this gap is **filled by documentation** (this
section + the catalog deviation statement): the front-speed-vs-analytic oracle is
**non-constructible in this environment** and is **substituted by the energy/stability
bound + deferred to the Flex ball-into-water golden** (Flex not reproducible here). This
mirrors the project's `PointDegeneracyIsDocumented` discipline (empirically demonstrate
the thing is degenerate/non-constructible rather than punt silently).

### Gap 2 (already-tracked, not new) — external goldens deferred

Vellum (crit 1), NVIDIA Flex (crit 2), MuJoCo-SDF (crit 3), and a Houdini cloth golden
are all **not reproducible in this environment** and were never run. They were deferred
to p15 by p09/p10 — and p15's resolution is the **gate-wording deviation** at the top of
this doc: the literal external-engine comparison is replaced by analytic + FD + cited-
equation substitutes. This is a substitution, surfaced at exit audit, not a fill.

### Gap 3 (already-tracked, not new) — SDF-contact normal/point gradient + backward wiring

Carried OPEN since p08-close → re-scoped to v1.0 (tasks #23/#24). The depth adjoint
ships; the normal-θ gradient does not (empirically degenerate, documented); SDF-contact
reverse wiring is off by default (v0.7 never backprops it). Closer = v1.0 coupled
diff-sim demo. Carried OPEN to exit audit.

---

## Lint / build

- All 6 oracle targets build clean and pass: `nuka_adjoint_fd_test` (XPBD+pp 10/10),
  `nuka_cuda_runtime_test` (Xpbd*+ParticleParticle* 19/19), `nuka_pbf_test` (14/14),
  `nuka_sparse_sdf_test` (10/10), `nuka_sdf_contact_adjoint_test` (7/7),
  `nuka_sdf_contact_ift_test` (6/6).
- **No source or test files were added or changed by p15** (cataloging only), so the
  build/lint surface is unchanged from the pushed batch-5 state. `tools/lint/physics_smell.py`
  → exit 0.

---

## Exit-audit restatement (verbatim line for the v0.7 exit audit)

> **crit 1/2/3 oracle deviation (p15):** Vellum, NVIDIA Flex, and MuJoCo are not
> available in this environment; **no external-engine comparison was run.** The crit
> 1/2/3 wording "oracle vs Vellum/Flex/MuJoCo-SDF" is met by a substitute oracle catalog
> (`docs/plans/2026-06-03-v07-oracle-catalog.md`): analytic/closed-form invariants + FD
> self-consistency + an **independent Eigen LDLT/FullPivLU oracle** for the SDF-contact
> system IFT + cited method equations. Every catalog number was reproduced on a real run
> (2026-06-03). This substitution is surfaced here exactly as the crit-6 C-ABI camera-arm
> deferral (task #29). **Two materially-weaker-than-wording items carried OPEN:** (1)
> crit-2's "dam-break front speed vs analytic (~4.71 m/s)" is **not** met by a
> front-speed equality — the shipped oracle is a one-sided energy/stability bound; the 3D
> inviscid SPH front (measured 3.22 m/s) does not reproduce the 1D shallow-water analytic
> (6.19 m/s), so the front-speed oracle is non-constructible here and is deferred to the
> Flex golden; (2) crit-3's SDF-contact analytical adjoint ships the **depth** gradient
> only — no normal-θ / contact-point gradient (empirically degenerate, documented) and
> SDF-contact reverse wiring is off by default (v0.7 never backprops it); re-scoped to
> v1.0 (tasks #23/#24).
