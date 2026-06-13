# G1 — Batched Union World: implementation brief

> Read-only design recon for gate **G1** of
> `docs/specs/2026-06-10-h1-whole-body-rl-grasp-spec.md` (§G1, §4, §6).
> Deliverable: extend `BatchedUnifiedWorld` with a UNION mode that mirrors the N=1
> oracle `UnifiedCoResidentStepper::StepStandGrasp`. NO production edits made here.

All file:line anchors are against HEAD `26170fb`.

---

## 0. TL;DR

- The oracle `StepStandGrasp` (`src/runtime/coresident/unified_coresident_stepper.cpp:1363-1792`)
  already emits the exact union we must batch: foot-sphere×ground-plane +
  fingertip-sphere×cup-hull + cup-proxy-box×table-plane, on a **floating-base** H1
  (`base_dof_=6`), one movable cup (`body_count=1`), all rows in ONE `UnifiedSolve`.
- The batched world's grasp branch (`ResolveBatchedGraspContact`,
  `batched_unified_world.cpp:727-1280`) already has 95% of the union machinery:
  env-major device replication (P2.4b), batched ABA / CRBA / chain-J, GPU
  sphere×hull narrowphase, the bijection invariant `total_body_count + art_index`,
  env-major M⁻¹/qdot tiling, the scatter guard. It is **fixed-base, finger×cup only**.
- The union mode = add (a) feet×ground rows, (b) cup-proxy-box×table rows, both on
  PROVEN handlers (sphere×plane / box×plane host-emitted like P2.2), into the SAME
  per-env append→one-solve loop, AND (c) flip the proto to **floating base**
  (`base_dof_=6` already supported by the existing prefix-sum pack/scatter and the
  `IntegrateFloatingBase*` kernels).
- **HARD G0 DEPENDENCY:** the whole-body H1 is **51 DOF** (confirmed
  `test_h1_bridge_spike.cpp:7`, `test_h1_costand_transfer.cpp:25`). The CRBA factor
  (`articulation_contacts.cu:451-452`) and the contact-solver `dof_to_link` loops
  (`articulation_contacts.cu:700-712,1022-1040`, `articulation_contacts.hpp:294`)
  silently clamp at `kMaxFactorDof = kMaxContactSolverDof = 18`. G1 contact work is
  **dishonest until G0 lands**. G1 must be built/tested AFTER G0 raises the caps to
  ≥64 with loud-fail. (G0 is a separate parallel agent per the task.)

---

## 1. Oracle anatomy — `StepStandGrasp` (the N=1 union reference)

Impl: `unified_coresident_stepper.cpp:1363-1792`. Config struct `StandGraspConfig`:
`unified_coresident_stepper.hpp:282-314`. Routed from `Step()` at `:355`. Constructed
at `:312-342` (sets `box_state_ = stand_grasp.cup_state`, `dof_stride_`, `root_link_`,
`base_dof_` from the proto root joint type, uploads `drive_torque`/`drive_force_limits`
once, `table_enabled_ = stand_grasp.has_table`).

### 1.1 Exact stage order

| stage | code | what |
|---|---|---|
| 0 | `:1369-1371` | `LaunchApplyTorqueDriveKernels(view, grip_torque_dev_, grip_limits_dev_)` — legs(RL)+arm(reach)+finger(grip) drive → `tau`. |
| 1/2 | `:1374` | `FeatherstoneAba::ComputeAccelerations(view, gravity_z_)` (reads tau; also leaves `link_xup` current for CRBA). |
| 3 | `:1377-1383` | `IntegrateVelocity` + `IntegrateFloatingBaseVelocity` (base half) + cup gravity kick `box_state_.linear_velocity.z += g*dt`; capture `cup_vz_pre_contact`. |
| 4 | `:1386-1388` | `live = host_proto_`; `DownloadArticulationState`; `poses = DownloadWorldPoses(...)` (UpdateWorldLinkPoses + sync + copy). |
| 5 | `:1391-1396` | build `cup_hull` ConvexHullView (mesh-local verts + live cup pose via `amf::BuildPrimFrame`). |
| 6a (DIRECT-EMIT pairs) | `:1399-1443` | FK foot/finger world centers; build `drive_pairs` directly — **NO broadphase**: per foot `{a=ArticulationLink/ChainJ/handle, b=MakeGroundRef(ground.broadphase_id)}`; per finger `{a=ArticulationLink/ChainJ/handle, b=RigidBody/RigidInvMass/cup.broadphase_body_id}`; if `emit_table` (`has_table && table_enabled_ && have_cup`) `{a=MakeBoxRef(cup_table_id), b=MakeGroundRef(table_broadphase_id)}`. `cup_table_id` = proxy id (7001) if any proxy half-extent>0 else the hull id (7000). |
| 6b (narrowphase) | `:1448-1505` | host `ShapeResolver` lambda resolves each handle → Sphere(foot/finger via `FootSpherePrim(center,r)`), Box(proxy via `BoxPrimXYZ(proxy_pose, half)` with `proxy_pose.position = cup_pose.position + cup_pose.rotation.Rotate(proxy_offset)`), ConvexHull(cup, geom=&cup_hull), Plane(ground/table via `GroundPrim(height)`, +Z normal in `frame.cy`). Then `BuildContactManifolds(drive_pairs, resolve, &manifolds)`. |
| 7 | `:1525-1543` | `EmitCompliantContactRows(manifolds, inputs{vel=0,invweight=1,dt,condim=3}, &rows, &sides)`; per-row friction **by category**: cup&static→`table_mu`, else cup→`finger_mu`, else→`foot_mu` (`:1540-1542`). |
| 8 (minv) | `:1548-1549` | `minv = InverseInertia(context_, live, gravity_z_, dof_stride_)` → dof_stride² M⁻¹ via CRBA `ComputeArticulationInertiaM`/`FactorArticulationInertiaM`. |
| 9 (qdot pack) | `:1554-1560` | flat `qdot[dof_stride]`: base DOFs `[0..base_dof_)` from `live.link_velocity[root_link_].v[i]`; each scalar joint at `DofIndexOf(leg)` from `live.qdot[leg]`. `qdot_before` saved. |
| 9b (row wiring) | `:1578-1654` | `kArtIndex=0`, `body_count=1`. Per row classify by `(react,react)`: **cup×table** (`:1593-1602`) cup-side→body 0, static→`kInvalidBodyIndex`, no J, `any_static_row=true`; **foot×ground** (`:1603-1628`) `j_foot=JacobianForRowBody`, `chain_j=FootChainJ(live,poses,foot_link,contact_point,foot_dir,dof_stride)`, slot append, foot-side body→`body_count+kArtIndex`, static→kInvalid, `art_refs[r]={kArtIndex,slot}`; **finger×cup** (`:1629-1652`) same shape, cup-side→body 0, finger-side→`body_count+kArtIndex`. `handle_to_link` (`:1565-1573`) maps broadphase handle→real device link (feet then fingertips, disjoint). |
| 10 (solve) | `:1657-1677` | `bodies={box_state_}`; `cfg{vel_it=64,pos_it=0,slop=0,baumgarte=0}`; `SolveContext` with `art_refs`, `chain_jacobians` (nullptr if empty), `inertia_m_inv=&minv`, `qdot=&qdot`, `dof_stride`; `UnifiedSolve(ctx,cfg)`. |
| report | `:1680-1768` | re-walk rows by category → fill report (see §1.2). `box_state_=bodies[0]`; `box_dv_norm`; `cup_dvz_impulse = cup_mass*(vz_after - cup_vz_pre_contact)`; `qdot_delta_l1`. |
| scatter | `:1771-1781` | inverse prefix-sum: base DOFs→`live.link_velocity[root_link_].v[i]`, joints→`live.qdot[leg]`; `CopyFromHost` link_velocity + qdot; `Synchronize`. |
| 11 (position) | `:1785-1790` | `view = device_.View()`; `IntegratePosition` + `IntegrateFloatingBasePose` + `IntegrateBoxPosition` (cup symplectic-Euler, `:1794-1807`). |

Contact-free early returns (`:1512-1522`, `:1544-1546`) integrate position and return
without scatter (the "BITE" free-fall path).

### 1.2 Report fields filled (`CoResidentStepReport`, hpp:89-143)

- foot/stand: `foot_normal_impulse_sum` (Σ foot normal λ), `foot_normal_rows`,
  `ground_pair_found`, `ground_row_count`, `ground_depth`, `ground_lambda` (`:1739-1744`).
- grasp: `finger_contacts`, `cup_vertical_impulse` (Σ cup-side λ·j_z over all finger
  rows), `finger_vimpulse_normal`/`finger_vimpulse_friction` split, `lambda`
  (max finger normal λ), `contact_depth` (`:1746-1751`).
- table: `table_row_count`, `table_lambda`, `table_vertical_impulse` (`:1753-1755`).
- cup: `cup_vz`, `cup_z`, `box_dv_norm`, `cup_dvz_impulse`, `qdot_delta_l1`.

### 1.3 Cup-table proxy mechanics

`cup_table_proxy_half` (x,y≈cup footprint, z thin), `cup_table_proxy_offset` (box
center in cup body frame, so the box bottom is flush with the hull bottom),
`cup_table_proxy_id`=7001 (distinct from hull id 7000). Used **only** in the cup×table
pair. Both ids resolve to `RigidInvMass` on the **same** cup BodyState (body 0), so the
row wiring maps both to the same mass. A box×plane (C3b `BoxPlane`) is the FAITHFUL flat-
bottom rest contact and sidesteps the hull-vs-plane coplanar-face instability (named
engine debt). Zero half-extents → proxy unused, table pair falls back to the hull id.
Resolver branch: `:1469-1481` (proxy box at live pose), authoring example in
`test_h1_cup_sequence_demo.cpp:368-370`.

### 1.4 Key helpers (oracle anon namespace, `cpp:44-212`)

`DownloadWorldPoses` (`:69`), `FootChainJ` (`:88`, uploads a temp device + one
`ComputeContactChainJacobians` launch per call — **the per-row storm** the batched world
already eliminated), `InverseInertia` (`:120`, CRBA), `FootSpherePrim` (`:154`),
`BoxPrim` (`:161`), `BoxPrimXYZ` (`:173`), `GroundPrim` (`:183`, +Z in `frame.cy`),
`MakeBoxRef` (`:193`), `MakeGroundRef` (`:204`).

---

## 2. Batched world anatomy — what union mode reuses vs adds

### 2.1 `BatchedSceneTemplate` (hpp:76-146) current fields

`bodies_per_env` (rigid SoA, leading env-major block); P2.2 ground: `has_ground`,
`box_half_extent`, `ground_height`; P2.3 grasp: `has_grasp`, `gripper_proto`,
`fingertips`, `cup`, `cup_local_index`, `grip_torque`, `drive_force_limits`,
`friction_mu`, `condim`; A5a: `reset_jitter_x/y`.

### 2.2 The grasp branch — `ResolveBatchedGraspContact` (`cpp:727-1280`) stage order

- **stages 0-3 BATCHED** (`:737-756`): `LaunchApplyTorqueDriveKernels(view,
  action_torque_dev_, grip_limits_dev_)` (note: drives from the **action** buffer, not
  the constant grip), `ComputeAccelerations`, `IntegrateVelocity`,
  `IntegrateFloatingBaseVelocity` — ONE launch each over `env_device_` (env-major).
- **stage 4 batched FK** (`:758-787`): ONE `UpdateWorldLinkPoses`→`world_pose_dev_`→ host
  download `all_poses` + refresh device `link_pose` (so chain-J reads FK geometry); ONE
  consolidated `DownloadArticulationState`→`all_live`.
- **cup gravity kick + report reset** (`:789-800`): per env `cup.linear_velocity.z +=
  g*dt`; capture `cup_vz_pre_contact[e]`.
- **per-env host slice** (`:810-838`): slice `all_live`/`all_poses` into `env_live[e]` /
  `env_poses[e]`; fill `fingertip_world_host_` (obs).
- **batched GPU narrowphase** (`:886-958`): gather per-(env×fingertip) `GraspSphereInput`
  + per-env `cup_frames` (host-baked `BuildPrimFrame`) + per-slot side refs → ONE
  `LaunchGraspSphereHullNarrowphase` → bucket into `env_manifolds[e]` (non-empty slots,
  fingertip order). Replaces the per-env host `BuildContactManifolds`.
- **per-env row wiring** (`:960-1084`): per env append `EmitCompliantContactRows
  (condim_)` (`row_assembly` tag), stamp `friction_mu_`, pack `qdot` slice **at**
  `art_index*dof_stride_` (env-major placement, NOT append — the P2.3b named-debt fix),
  classify finger×cup rows, set body indices (`cup_body_index = BodyIndex(e,cup_local)`,
  finger key `total_body_count + art_index`), GATHER chain-J inputs `cj_link`
  (GLOBAL link `art_index*base_link_count_ + finger_link`), `cj_point`, `cj_dir` into a
  per-slot list, set `art_refs[r] = {art_index, slot}`. **The bijection invariant**
  (`:1018-1031`): the finger key and the art_refs art_index are set as a pair per env.
- **batched CRBA M⁻¹** (`:1101-1124`): ONE `ComputeArticulationInertiaM` +
  `FactorArticulationInertiaM` over `env_device_` → env-major `minv` (`crba_minv` tag).
- **batched chain-J** (`:1126-1159`): ONE `ComputeContactChainJacobians` over the
  gathered `cj_*` slots → `chain_jacobians[slot*dof_stride]` (`chain_jacobian` tag).
- **stage 10 solve** (`:1161-1185`): ONE `UnifiedSolve` over concatenated rows + full
  env-major `bodies_` (`row_solver` tag).
- **report** (`:1187-1235`): per env Σ cup-side λ·j_z, max λ, per-fingertip normal
  impulse bucket, `cup_dvz_impulse`, `cup_vz`.
- **scatter** (`:1237-1278`, `scatter_integrate` tag part 1/2, guarded to envs with rows).

The position integrate is in `Step()` (`:560-585`, `scatter_integrate` part 2/2): ONE
batched `IntegratePosition` + `IntegrateFloatingBasePose` over `env_device_` + per-body
`IntegrateBodyPosition`.

### 2.3 env-major machinery (P2.4b)

`env_device_` = ONE `ArticulationDeviceBuffers` of `env_count_` replicated grippers
(`ReplicateArticulationHostState(gripper_proto_, env_count_)`, `cup.cpp:235-245`),
`articulation_count == env_count_`, replica e's links `[e*base_link_count_, ...)`.
Asserted at `:243-245`. Persistent scratch `world_pose_dev_`, `crba_composite_`,
`crba_m_`, `crba_m_inv_` (`:302-316`). The M⁻¹ tile @ `art_index*dof_stride^2`, qdot
slice @ `art_index*dof_stride`.

### 2.4 GPU sphere×hull narrowphase launcher

`src/collision/gpu/narrowphase_grasp.hpp:53-61` `LaunchGraspSphereHullNarrowphase`:
one thread per (env×fingertip) slot, runs `cvx::SphereHull(sphere=fingertip,
cup=hull, sphere_is_a=true)`, slot-indexed output, fully stamped (a/b/manifold_key +
merged material) modulo ULP-scale host-vs-device SphereHull delta. `GraspSphereInput`
= `{center, radius, env_index}`.

### 2.5 SetActions / dof_to_link_ / ResetEnvs / ExportObsState / perf

- `SetActions` (`:350-369`): env-major DOF-indexed host actions →
  `action_torque_host_[e*base_link_count_ + dof_to_link_[d]]` → ONE upload to
  `action_torque_dev_`. `dof_to_link_` built `:279-287` (inverse of `DofIndexOf`).
- `ResetEnvs` (`:448-521`): per-env cup restore + jitter; ONE batched download/modify/
  upload of `env_device_` slices to the proto open config.
- `ExportObsState` (`:374-443`): ONE bulk download of q/qdot/(link_velocity if
  base_dof_>0) → env-major pack; fingertip world pos from `fingertip_world_host_`;
  per-finger normal impulse from the last report.
- `DofIndexOf` (`:523-533`), `IntegrateBodyPosition` (`:535-549`).
- perf tags (`:313-314`): `pose_download, artic_download, crba_minv, chain_jacobian,
  aba_integrate, narrowphase, row_assembly, row_solver, scatter_integrate, obs_export`.
  Baselines: N=32 ~4836 eps, N=1024 ~10,183 eps (`test_..._perf.cpp:605-629`).

### 2.6 The ground branch — `ResolveBatchedGroundContact` (`cpp:610-716`)

Pure-rigid box×plane: per env DIRECT-EMIT `{box=RigidInvMass, ground=StaticNull}`,
host `BuildContactManifolds` (C3b BoxPlane), `EmitCompliantContactRows(condim=1)`,
overwrite body indices (`BodyIndex(e,0)` / `kInvalidBodyIndex`), ONE `UnifiedSolve`
with `ctx.articulation` default (no art arm). **This is the exact template for the
union's cup-proxy-box×table rows** (box-side becomes `BodyIndex(e,cup_local)` and the
solve is the shared union solve, not a separate one).

### 2.7 Reuse vs add table

| union piece | reuse | add |
|---|---|---|
| batched ABA/CRBA/chain-J/FK over env_device_ | ✅ verbatim | — |
| GPU sphere×hull (finger×cup) | ✅ verbatim | — |
| finger×cup row wiring + bijection invariant | ✅ verbatim | — |
| env-major minv/qdot tiling, scatter guard | ✅ verbatim | — |
| cup gravity kick, position integrate, ExportObs/Reset/SetActions | ✅ (extend obs for base/feet) | — |
| **foot×ground rows** (sphere×plane) | the row-class wiring shape from oracle `:1603-1628` | feet config fields + foot narrowphase + foot chain-J slots (feet are ALSO `ArticulationChainJ` → SAME gather list as fingers) |
| **cup-proxy-box×table rows** (box×plane) | `ResolveBatchedGroundContact` emission shape | proxy/table config fields + box×plane host narrowphase appended into the SAME union rows + cup-side body index |
| **floating base** | `base_dof_` prefix-sum pack/scatter already generic; `IntegrateFloatingBase*` kernels per-articulation | flip proto to floating cook; nothing in the layout changes (base_pose already tiled by Replicate) |

---

## 3. The union-mode design

### 3.1 Additive `BatchedSceneTemplate` fields (mirror `StandGraspConfig`)

```
// --- union (feet + table) — inert unless has_union (or reuse has_grasp + has_feet/has_table) ---
bool                              has_feet      = false;
std::vector<CoResidentFootSphere> feet;                 // authored ankle spheres (toe/heel/ankle)
CoResidentGround                  ground;                // static +Z plane
float                             foot_mu       = 0.8f;
bool                              has_table     = false;
float                             table_height  = 0.0f;
float                             table_mu      = 0.6f;
uint32_t                          table_broadphase_id = 8500u;
math::Vec3                        cup_table_proxy_half{};   // 0 → use hull id
math::Vec3                        cup_table_proxy_offset{};
uint32_t                          cup_table_proxy_id  = 7001u;
// (gripper_proto becomes a FLOATING-BASE whole-body H1; no new flag strictly needed —
//  base_dof_ is derived from the proto root joint type, already at cpp:222-223.)
```

The existing `has_grasp` path stays byte-exact when `has_feet=false && has_table=false`
(no feet/table pairs emitted). Recommendation: gate on `has_feet`/`has_table` rather than
a new `has_union` flag, so the increments compose (G1b adds feet, G1c adds table).

### 3.2 Step restructure — ONE union contact phase per step

Keep the single-solve structure of `ResolveBatchedGraspContact`. The per-env loop builds
THREE row classes into the **same** shared `rows`/`sides`/`art_refs` and the **same**
gather lists, then ONE batched CRBA + ONE batched chain-J + ONE `UnifiedSolve`:

1. **Feet×ground (sphere×plane):** for each env, for each foot sphere, DIRECT-EMIT
   `{a=ArticulationLink/ChainJ/foot.broadphase_handle, b=MakeGroundRef(ground.broadphase_id)}`,
   narrowphase to a sphere×plane manifold, `EmitCompliantContactRows(condim_)`, stamp
   `foot_mu`. Wire body indices: foot-side→`total_body_count + art_index` (the SAME
   synthetic articulation key as fingers — feet and fingers are the same articulation e),
   ground→`kInvalidBodyIndex`. **GATHER foot chain-J into the SAME `cj_link`/`cj_point`/
   `cj_dir` slot list** (cj_link = `art_index*base_link_count_ + foot_link`); set
   `art_refs[r] = {art_index, slot}`. Because feet and fingers share the env's single
   articulation key, same-env foot+finger rows SERIALIZE on env e's qdot tile (correct,
   no race) and cross-env parallelize — the bijection invariant extends UNCHANGED.

2. **Finger×cup (sphere×hull):** EXACTLY as today (the GPU launcher), wired into the
   same shared buffers/gather list, slot indices continue the running counter.

3. **Cup-proxy-box×table (box×plane):** if `has_table` && cup has geometry, per env
   DIRECT-EMIT `{a=MakeBoxRef(cup_table_id), b=MakeGroundRef(table_broadphase_id)}`,
   host `BuildContactManifolds` box×plane at the live proxy pose
   (`proxy_pos = cup.position + cup.orientation.Rotate(proxy_offset)`),
   `EmitCompliantContactRows(condim_)`, stamp `table_mu`. Wire: cup-side→
   `BodyIndex(e,cup_local)`, table→`kInvalidBodyIndex`, **no chain-J / no art_refs**
   (both sides are RigidInvMass/StaticNull). This is `ResolveBatchedGroundContact`'s
   exact shape with the cup body index instead of body 0, appended into the union rows.

**Bijection invariant extension:** the synthetic articulation key
`total_body_count + art_index` (`cpp:880`,`:1031`) already serves ANY ChainJ row of env
e. Feet rows reuse it verbatim. The cup-side rows (finger×cup and cup×table both) carry
`BodyIndex(e,cup_local)` — disjoint across envs. No new keying needed; the invariant
`total_body_count + e` is unchanged, it just now keys more rows per env.

**Per-env row-range bookkeeping** (`env_row_range`, `cpp:884`): must now span foot +
finger + table rows for the per-category report split (foot_normal_impulse vs
cup_vertical_impulse vs table_vertical_impulse). Classify in the report walk by
`(react,react)` exactly as oracle `:1689-1737`.

### 3.3 Narrowphase plan per pair (PROVEN handlers only)

- **finger×cup:** the existing GPU `LaunchGraspSphereHullNarrowphase` — keep verbatim.
- **foot×ground (sphere×plane):** **recommend host emission** (per-env
  `BuildContactManifolds` sphere×plane in the per-env loop, like the oracle and like the
  ground branch). Rationale grounded in the perf structure: there are only ~4 feet
  spheres/env vs ~30 fingertips, the host sphere×plane is trivial, and the chain-J/CRBA/
  solve costs dominate (perf tags show `narrowphase` collapsed after P2.4c moved the
  HEAVY sphere×hull to GPU; sphere×plane is far cheaper). A trivial GPU sphere×plane
  kernel (mirroring the grasp launcher) is a NAMED optimization to defer to G1's gate-(d)
  throughput finding — only build it if the N=1024 measurement shows the host foot
  narrowphase is a measurable fraction. **Honesty note:** if foot narrowphase goes to a
  GPU kernel, the byte-exact bar must be handled like P2.4c (wire the SAME launcher into
  the parity ref → device-vs-device byte-exact; see §5).
- **cup-proxy-box×table (box×plane):** **host emission** like P2.2
  (`ResolveBatchedGroundContact`), C3b BoxPlane. One box×plane per env, trivial.

### 3.4 Floating-base proto: what changes

The H1.2 scene was **Fixed-root** (`gripper_proto` = `LoadH1Fixed`, `base_dof_=0`,
`test_batched_h1_hand_grasp.cpp:432`). The union scene is **floating-base**
(`LoadFloating`, `h1_demo_shared.hpp:88-101`; ~51 DOF, base_dof_=6). Verified-already-
generic pieces:

- **qdot layout / pack / scatter:** `base_dof_` is derived from the proto root joint
  type at construction (`cpp:222-223`); the pack (`cpp:1009-1015`) and scatter
  (`cpp:1259-1266`) already fill `[0..base_dof_)` from `live.link_velocity[root].v[i]`
  and joints at `DofIndexOf(leg)`. For base_dof_=6 this reproduces base@[0..5], joints
  after — BYTE-FOR-BYTE the oracle's pack (`:1554-1560`). No code change.
- **IntegrateFloatingBaseVelocity/Pose:** already called unconditionally in the grasp
  branch (`cpp:754`) and Step (`:574`); the kernels (`featherstone_aba.cu:635-696`) are
  one block per articulation and **no-op when the root is not FloatingBase**, so they
  were inert in H1.2 and ACTIVATE automatically for the floating proto. Per-articulation
  batched → env-major correct.
- **env_device_ replication:** `ReplicateArticulationHostState` tiles `base_pose`
  per-articulation (`articulation_state.cpp:439-441`) — each env's base pose starts
  identical (from the proto) and evolves independently. Correct for floating base.
- **SetActions:** the action width `dof_stride_` now includes the 6 base DOFs.
  `dof_to_link_[d]` for `d<base_dof_` maps to the root link; the RL action mask should
  zero the base columns (you cannot torque a free base). Either (a) leave base action
  slots at 0 (the policy never writes them), or (b) make `ActionDim()` report only the
  ACTUATED DOF and offset by base_dof_. **Recommend (a)** for parity simplicity — the
  drive kernel applies tau per link; a 0 on the floating root is a no-op (the root has no
  joint actuator). Confirm `LaunchApplyTorqueDriveKernels` ignores the FloatingBase root
  (it indexes per joint; a FloatingBase joint has no scalar tau slot to apply — verify in
  G1a).
- **scatter guard:** unchanged. The guard (`cpp:1252-1254`) skips envs with no rows; a
  floating-base env that established feet contact (always, when standing) will have rows
  every step, so the guard rarely fires for stand envs — fine.

**G0 BITE:** the 51-DOF proto's CRBA factor (`articulation_contacts.cu:451-452`) and
contact `dof_to_link` loops cap at 18. Without G0, the M⁻¹ tile and the contact-solver
working vectors silently lose all coupling past DOF 18 → the feet/finger reactions into
the legs/arm are garbage. **G1 cannot be honestly validated until G0 raises the caps.**

---

## 4. Parity-test design (mirror `test_batched_h1_hand_grasp.cpp`)

New TU `tests/coresident/test_batched_h1_union_world.cpp` (additive). Build the IDENTICAL
whole-body scene on BOTH the oracle (`StandGraspConfig` → `UnifiedCoResidentStepper`) and
the batched world (union template → `BatchedUnifiedWorld`); run lockstep with identical
per-link drive computed from each side's own downloaded state.

### 4.1 Scene construction (reuse the cup-sequence-demo authoring)

- **Proto:** `nuka::demo::LoadFloating(kFullMjcf)` (whole-body floating H1, ~51 DOF;
  `h1_demo_shared.hpp:88`). Zero qdot/link_velocity, identity base rotation
  (`test_h1_cup_sequence_demo.cpp:222-226`). Bent stance
  (`kStanceHipPitch=-0.40, kStanceKnee=0.70, kStanceAnkle=-0.30`,
  `h1_demo_shared.hpp:208-210`).
- **Feet:** authored ankle spheres (`CoResidentFootSphere`), toe/heel per ankle:
  `kFootSphereR=0.025, kFootBottomZ=-0.055, kFootToeX=0.10, kFootHeelX=-0.10`
  (`h1_demo_shared.hpp:212-215`), handles from 12000+ (disjoint from cup 7000 / ground
  8000 / table 8500 / fingers 9000+), authoring loop `test_h1_cup_sequence_demo.cpp:247-256`.
  Seat the base z so foot bottoms sit at `kGroundZ - kRestPen` (2 mm, `:259-267`).
  `ground.height=0`, `foot_mu=0.8`, `condim=3` (`:269-271`).
- **Fingers + cup:** the H1.1 30-sphere finger-only wrap + 10.9 cm cup. Reuse the H1.2
  constants: `kSxy=kSz=1.8` (10.9 cm), `kMu=0.8`, `kKp=4, kKd=0.4, kCloseOffset=0.18`,
  `kRealArmature=0.1, kRealDamping=1.0`, `kWrapRadius=0.006`, `kCupMass=0.2`
  (`test_batched_h1_hand_grasp.cpp:142-143,209,404-410,469`). NOTE: the cup-sequence demo
  uses the SAME wrap/cup helpers via `nuka::demo::*`; prefer those so the demo and parity
  scene share one authoring path.
- **Table:** proxy box at the cup footprint (`cup_table_proxy_half` from the scaled hull
  half-extents, `cup_table_proxy_offset = 0`, `cup_table_proxy_id=7001`,
  `test_h1_cup_sequence_demo.cpp:368-370`), `table_height` = cup-bottom rest z,
  `table_broadphase_id=8500`, `table_mu=0.6`.
- **Drive:** ONE per-link `drive_torque` vector overwritten each step
  (`StandGraspConfig::drive_torque`, `:310-311`) — legs PD-hold/RL-stand, arm reach-PD,
  fingers grip-PD. For parity, drive BOTH sides from their own downloaded state via the
  H1.2 `DrivePdOracle`/`DrivePdBatched` pattern (`test_batched_h1_hand_grasp.cpp:529-566`),
  extended to drive leg + finger links. The batched action width includes base_dof_=6
  zeroed columns (§3.4).

### 4.2 Gate structure (mirror H1.2 Gate 1 / Gate 2)

- **Gate (a) parity** (≥300 contact-rich steps): per the H1.2 reformulated bar — the
  table-free + table-supported 30-contact-on-shared-links grasp is CHAOTIC at the FP
  floor, so HARD-assert the leading FP-floor window (cup pos/vel ≤1e-6, gripper qdot
  ≤5e-6, contact-count match through the window) + a chaos control (two same-code oracles,
  1e-7 m IC nudge, must amplify to the same order). Run ≥300 steps with the table
  removed mid-run (`SetTableEnabled(false)` on the oracle / the batched analog) to exercise
  feet-loaded + fingers-closing + cup-on-table-then-lifted. The spec allows ≤1e-5 with a
  written justification (§G1) — reuse the H1.2 chaos-control justification verbatim.
- **Gate (b) independence** (N=8): the H1.2 Gate-2 pattern — per-env byte-exact vs its own
  N=1 run, adjacent envs differ, MIXED held/no-contact (env-major tile-gap + scatter
  guard), D1 two-run memcmp. `SnapshotEnv` must extend to base_pose (floating) + leg qdot.
- **Gate (c) determinism:** the D1 sub-test of Gate (b).
- **Gate (d) trainability:** the H1.2 throughput sweep extended to N∈{32,256,1024}, drive =
  ONE bulk `SetActions` + `Step` (the steady-state RL seam, NOT `DrivePdBatched` whose
  per-env `DownloadGripper` is a test artifact). Compute the <24h-per-100M-env-steps bar:
  100e6 / eps seconds. Report as a MEASURED lower bound; if it fails, name the optimization
  increment (likely a multi-block solver or device-resident host-narrowphase moves).

---

## 5. Risk list (ordered, with mitigations)

1. **G0 18-DOF cap (BLOCKER).** 51-DOF whole-body silently truncates M⁻¹/J/solver past
   DOF 18 (`articulation_contacts.cu:451-452,700-712,1022-1040`, `hpp:294`). **Mitigation:**
   G1 contact work is gated on G0 landing (caps ≥64, loud-fail). Until then, only G1a
   (floating-base parity, NO contact) is honestly testable. Verify the union parity test
   FAILS on pre-G0 HEAD (proves the truncation) and GREENS after — this is the G0 gate
   test the spec asks for, naturally produced by G1's scene.
2. **Floating-base batched replication is UNTESTED for contact.** H1.2 proved env-major
   for a Fixed root; the floating root activates `IntegrateFloatingBase*` and the base
   prefix-sum lanes per env. **Mitigation:** G1a (floating-base parity, NO contact) is a
   standalone increment that isolates this before any union rows.
3. **Oracle host-narrowphase vs batched GPU-narrowphase ULP delta vs the byte-exact bar.**
   The finger×cup GPU launcher differs from the oracle host SphereHull at ULP scale; the
   30-contact grasp is chaotic. **Mitigation:** follow P2.4c/H1.2 — wire the SAME launcher
   into a standalone reference for a device-vs-device byte-exact check; the A2-vs-oracle
   gate uses the FP-floor-window + chaos-control justification (1e-6/5e-6 in-window, full
   run reported). If foot narrowphase ever moves to GPU, apply the same pattern.
4. **Contact-count variability across envs (row bucketing / env-major tile gaps).** With
   feet+fingers+table, the per-env row count varies a lot; a no-contact env (cup dropped,
   foot airborne) leaves tile gaps. **Mitigation:** the env-major PLACEMENT (write minv/qdot
   AT `e*stride`, never append; `cpp:848-861`) + the scatter guard (`:1252-1254`) already
   handle this; the G1 MIXED sub-test (Gate b/c) must include a foot-airborne env, not just
   cup-dropped.
5. **51-DOF factor cost per step at N=1024 (throughput).** CRBA factor + chain-J scale with
   dof_stride² and dof_stride×contacts; 51 DOF is ~4× the H1.2 fixed-hand dof_stride and the
   union adds ~4 feet rows + table rows. The N=1024 baseline (~10k eps) may drop. **Mitigation:**
   gate (d) is a MEASURED lower bound; if <24h/100M fails, name a device-resident-solver or
   batched-host-narrowphase increment BEFORE training (per spec §G1(d)).
6. **Drive injection for legs+arm+fingers simultaneously + the floating base.** The action
   buffer now spans base(6, zeroed) + ~45 actuated DOF; mis-mapping a DOF column to the wrong
   link corrupts the drive. **Mitigation:** the `dof_to_link_` build (`cpp:279-287`) is
   generic; verify base columns map to the root and that `LaunchApplyTorqueDriveKernels`
   no-ops the FloatingBase root joint (G1a check). Reuse the H1.2 `DofIndexOf` equivalence.
7. **Cup-table proxy id collision / fall-back.** The proxy id (7001) must stay distinct from
   the hull id (7000) and the feet/ground/table handles. **Mitigation:** reuse the
   demo's disjoint handle map (cup 7000 / proxy 7001 / ground 8000 / table 8500 / fingers
   9000+ / feet 12000+; `test_h1_cup_sequence_demo.cpp:244-247`).
8. **Report-field category split.** With three row classes the per-env report must split
   foot/finger/table impulses correctly (the gate force-balance numbers). **Mitigation:**
   classify by `(react,react)` exactly as oracle `:1689-1737`; extend
   `BatchedGraspEnvReport` (hpp:152-169) with the foot/table fields the oracle reports.

---

## 6. Increment decomposition (smallest independently-testable commits)

Each `[skip ci]`, additive TU preferred, full suite green after each. **All contact
increments gated on G0.**

- **G1a — floating-base batched articulation parity, NO contact.** Flip a template proto
  to `LoadFloating` (51-DOF), `has_feet=has_table=has_grasp=false` (or a no-contact grasp
  with the cup far away). Gate: N=1 byte-exact base_pose/q/qdot vs an oracle stepper on the
  IDENTICAL floating proto over ≥100 steps with leg PD drive; N=8 independence + D1. Proves
  the floating root + base prefix-sum lanes + `IntegrateFloatingBase*` batch correctly,
  ISOLATED from any union-row work. (Does NOT need G0 — no contact factor path.)
- **G1b — feet×ground union at floating base (no cup/table).** Add `has_feet` + foot
  sphere×plane host narrowphase + foot chain-J slots into the shared solve. Gate: N=1
  byte-exact vs an oracle `StandConfig` stepper (`StepStand`, `:1126`) on the IDENTICAL
  feet/ground scene over ≥300 steps (feet loaded); N=8 + D1; foot_normal_impulse balances
  m·g·dt. **Gated on G0** (foot chain-J into 51-DOF legs hits the cap).
- **G1c — + finger×cup (the full grasp on the floating base).** Enable the existing grasp
  branch on the floating proto + feet, cup held by friction. Gate: N=1 byte-exact vs the
  oracle (a `StandGraspConfig` with `has_table=false`) — feet + fingers + cup, ≥300 steps;
  N=8 MIXED (foot-airborne + cup-dropped); D1. This is the headline G1 parity. **Gated on G0.**
- **G1d — + cup-proxy-box×table (the lift choreography).** Add `has_table` + box×plane host
  narrowphase + `SetTableEnabled` toggle. Gate: N=1 byte-exact vs the oracle full
  `StandGraspConfig` (`has_table=true`) with the table removed mid-run; table_vertical_impulse
  carries load then →0, finger impulse closes the triangle. The COMPLETE union parity (a/b/c).
- **G1e — throughput gate (d) + the trainability verdict.** Extend the perf TU to the
  union scene N∈{32,256,1024}, ONE bulk SetActions + Step, MEASURED eps → 100M-env-step
  wall-clock → the <24h bar. If RED, this commit names (does not yet build) the optimization
  increment. Pure instrumentation (physics-neutral).
- **(optional) G1f — GPU foot sphere×plane kernel.** Only if G1e measures the host foot
  narrowphase as a non-trivial fraction. Device-vs-device byte-exact bar (P2.4c pattern).

Sequencing note: G1a can start immediately (no G0 dep). G1b→G1d wait on G0. G1e/G1f after
the union physics is parity-green.

---

## 7. Locked decisions honored (spec §4)

ONE world (extend `BatchedUnifiedWorld`); PROVEN handlers only (sphere×plane / sphere×hull /
box×plane); feet = authored ankle spheres; ONE env class with config-gated stages; legacy
`BatchedArticulatedWorld`/`world_stepper` untouched; no hull×box, no mesh NP, no new EPA;
no golden regeneration; production edits confined to `src/runtime/coresident/*` + the
articulation kernels G0 names.
