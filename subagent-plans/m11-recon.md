# M11 ULTRACODE RECON — Authoritative Construction Plan

> EXECUTION BASELINE. An implementer follows this with NO further investigation.
> Repo: /root/Nuka-Physics · branch master · HEAD dcd40c8 (M9 COMPLETE+REVIEWED+PUSHED).
> Synthesized from 6 facet findings (viewer, interop, RT, collision/sdf, buffer-legacy, exit-gate).
> Milestone order (owner-ruled): M8 → M8.5 → M9 → **M11 (this)** → M10 (RL, LAST).

---

## 1. M11 Scope & Mission

M11 is the **last refactor milestone** before M10 (RL). It is **render/UI + the deferred buffer/CUDA-homing
endgame** — six work-buckets:

| # | Bucket | Essence |
|---|--------|---------|
| **A** | **buffer_legacy FULL SWEEP (T13)** | Migrate all **65** `buffer_legacy.hpp` includers (collision/diffsim/sdf/sensor/c_abi/core-diag/rt/articulation) to phi v2 `BufferI`, ONE stream-binding decision, ONE D1 re-gate, then DELETE `buffer_legacy.hpp` + legacy phi `{OwnedStream,DeviceContext,UploadVector,StreamView}` (123 consumers, fast-follow). **FOUNDATIONAL.** |
| **B** | **collision/SDF cluster finalize** | Re-assert the **SDF-tier forward gate** (carry-forward a, salvage from `fcae2a6`); keep narrowphase_dispatch/contact_stream_driver/sdf_contact as oracle infra (carry-forward b); delete redundant `sdf_device_world`; migrate SAP-oracle off buffer_legacy. |
| **C** | **RT homing + RtBackendI (§3.10)** | Extract `render::RtBackendI`; LINE-FAITHFULLY home `src/rt/*.cu` → `src/phi/backend_cuda/rt/`; add `src/render/rt_adapter.{hpp,cpp}` (`RenderWorldToTwoLevelScene`); RT offscreen path stays a **local D1/CI gate**. |
| **D** | **render_scene delete + old-renderer disposition** | Delete `render_scene.{hpp,cpp}` (plan L499 deferred) + retarget/remove the dead `RenderSceneDebugOverlayVulkan` path + 2 tests. |
| **E** | **CUDA↔Vulkan interop publisher** | Add `CudaVulkanInteropPublisher` behind the existing `PosePublisher` seam; device scatter via `launch.cuh` in `backend_cuda`. NOT locally verifiable (CUDA-GPU vs lavapipe-CPU). |
| **F** | **ImGui viewer delta** | Add drive-slider panel (writes `FieldId::DriveTarget`), entity picker + **DRAG ENTITY / MoveEntity** (writes LIVE nk Data `BodyPose`/`BasePose`), fix viewer debt. NOT a greenfield (M8.5 viewer exists). |

### Highest directive (overrides everything)
**ONE general/universal physics solver. NO per-case physics special-casing; NO dedicated special world type.**
M11 is pure render/UI/buffer-plumbing — **no task introduces a solver branch or special world**. The two risk
surfaces:
- **Viewer MoveEntity/DRAG** MUST write through the GENERAL path: viewer → `command_queue` (MoveEntity variant)
  → `nk::Data::UploadField(BodyPose/BasePose)`. **NEVER** through `nuka_scene_set_local` (authoring-only,
  needs re-cook) and **NEVER** a "grab the cup" demo shim.
- **Interop publisher** MUST implement the SAME generic `PosePublisher::Publish(world, env, render_world)`
  reading the SAME general `nk::World`/`Data` device buffers — no per-demo binding, no world-type switch.

The collision/SDF cluster is **explicitly allowed** (collision infra / IFT diffsim / test oracle, NOT a special
world — audited, no demo special-casing). Drive-slider panel MUST be a **generic per-DOF editor**, never a
hardcoded grasp choreography.

### D1 byte-exact discipline (non-negotiable)
- `tests/oracle/golden/**` OWNER-PROTECTED, NEVER regenerated. Red golden = real regression, never "rebaseline".
- `examples/scenes/h1_cup_table.nks/.nka` physics-gate-frozen — viewer edits are RUNTIME `Data` writes, **never
  persisted back to the .nks**.
- Kernel relocation must be LINE-FAITHFUL (bit-identical), proven by adjoint-FD / union_cook_golden /
  determinism gates.
- **The async-stream change in the buffer sweep is the single biggest D1 risk in M11.**

### What M11 is NOT (M10 boundary)
M11 does **NOT** touch: any RL (`train_go2_ppo`/`train_h1_grasp_ppo`/catch-eval), go2 PD-stance, per-link
**wrench CONTROL** (the readout op `sensor/contact_wrench.cu` is M3-homed and fine), batched autoreset, base-pose
control, control-modes, or the SG-spec re-narration. Carry-forward ledger verdicts:

| Carry-forward | Verdict | Why |
|---|---|---|
| (a) SDF-tier forward gate re-assert | **M11** | Rides the SDF cluster; salvage `fcae2a6:tests/collision/test_sdf_tier_wired.cpp` verbatim. |
| (b) collision host-narrowphase migrate-vs-keep | **M11** | KEEP as oracle (8 surviving gates); only BufferI type-swap. |
| (c) Newton-on-SDF IFT family | **M11** | KEEP infra; rides buffer sweep transitively. |
| (d) buffer_legacy + OSC `ArticulationDeviceState` homing | **M11** | The atomic sweep + single stream decision. |
| (e) rt/ + render_scene + DLPACK-Vulkan interop | **M11** | Buckets C/D/E. |

---

## 2. Cross-Facet Dependency DAG & Phase Order

The buffer sweep (Phase 0) is the **base of the DAG**: collision-sdf, RT, and the diffsim/IFT/OSC families are
all among the 65 buffer_legacy consumers. No deletion completes until the sweep lands. Phase boundaries =
`cmake --build build-cuda128 -j` links green before the next deletion.

```
Phase 0  BUF atomic sweep  ─────────────┬──────────────┬───────────────┐
(foundational)                          │              │               │
   │ (BUF-12 single D1 re-gate)         │              │               │
   ▼                                    ▼              ▼               ▼
Phase 1  collision/SDF finalize    Phase 2  RT     Phase 3  render   (Phase 4/5 leaves)
   │  (COL-1 SDF-tier gate,        homing (RT-*)   _scene delete +
   │   COL-5 delete sdf_dev_world)      │          old vk_renderer
   │                                    │          (RT-0, RT-6)
   └────────────────┬───────────────────┘               │
                    ▼                                    ▼
              Phase 4  interop publisher (INT-*)   ── leaf, build-link only
                    │
                    ▼
              Phase 5  viewer delta (VIEW-*)       ── leaf
                    │
                    ▼
              Phase 6  EXIT GATE (EXIT-1) + ultracode review
```

### Phase 0 — buffer_legacy ATOMIC SWEEP (FOUNDATIONAL, locally verifiable)
**Tasks:** BUF-0, BUF-1, BUF-2, BUF-3, BUF-4, BUF-5, BUF-6, BUF-7, BUF-8, BUF-9, BUF-10, BUF-11, **BUF-12 (THE
single D1 re-gate)**, BUF-13 (delete buffer_legacy.hpp), BUF-14 (fast-follow: delete DeviceContext/OwnedStream).
**Gated by:** BUF-12 — `ctest -L full && -L perf` green + every `tests/oracle/golden/**` byte-identical after the
async-stream rebind.
**Locally verifiable:** YES (pure CUDA compute on this box). **HARD RULE:** never leave a TU including both
`buffer.hpp` and `buffer_legacy.hpp` (duplicate `nuka::phi::Buffer` = compile error, which *enforces* atomicity).
Do NOT delete `buffer_legacy.hpp` until ALL 65 consumers migrate.

### Phase 1 — collision/SDF cluster finalize (locally verifiable)
**Tasks:** COL-1 (SDF-tier gate re-assert), COL-2 (special-world audit), COL-3 (doc retarget), COL-4
(sdf_contact.cu dead-kernel decision), **COL-5 (delete sdf_device_world — lands INSIDE the BUF sweep window)**,
COL-6 (SAP-oracle migrate — INSIDE BUF sweep), COL-7 (remaining test buffer_legacy re-point — INSIDE BUF sweep),
COL-8 (§3.10 no-op-ify audit).
**Gated by:** COL-1's `BoxOnBoxMatchesAnalyticalNormalAndPenetration` + `TwoRunByteIdenticalManifold`; the SAP
`D1TwoRunByteExactPairList` post-migration.
**Locally verifiable:** YES. **Sequencing note:** COL-5/6/7 are buffer_legacy consumers → they execute as part
of the Phase-0 sweep's single stream-binding decision, not as a separate sweep. COL-1/2/3/4/8 are pure-host /
audit and can run before or alongside.

### Phase 2 — RT homing + RtBackendI (locally verifiable for offscreen D1)
**Tasks:** RT-2 (split scene-desc Tier-A vs device Tier-B headers), RT-CMAKE (homed target keeps `--fmad=false`),
RT-3 (LINE-FAITHFUL move 3 `.cu` + 6 `.cuh` → `backend_cuda/rt/`), RT-1 (define `RtBackendI`), RT-4
(`rt_adapter` = `RenderWorldToTwoLevelScene`), RT-5 (NEW offscreen RenderWorld→RtBackendI D1 image gate), **RT-7
(RT buffer migration to BufferI — INSIDE the BUF sweep)**.
**Gated by:** `nuka_rt_two_level_test` + `nuka_rt_scene_render_test` + `nuka_rt_traversal_test` memcmp==0 after
the move + RT-5's 6-AOV two-run memcmp==0.
**Locally verifiable:** YES — RT offscreen is pure CUDA compute → image buffer; the rt tests are in the
`if(CMAKE_CUDA_COMPILER)` region OUTSIDE the `NK_BUILD_VULKAN` block. "RT still D1" is a HARD local gate.
**Depends on Phase 0** (RT outputs go via BufferI; RT-7 rides the sweep).

### Phase 3 — render_scene delete + old vulkan_renderer (locally verifiable, build-correctness)
**Tasks:** RT-0 (confirm deletion safety — owner gate DEC-1), RT-6 (delete `render_scene.{hpp,cpp}`; remove
`RenderSceneDebugOverlayVulkan`/`RenderSceneVulkan` + helpers from `vulkan_renderer`; retarget the 2 tests).
**Gated by:** git-grep zero `render_scene`/`RenderScene` survivors; `nuka_render` + `nuka_render_test` +
`nuka_perf_test` build green; lint 0.
**ORDERING HAZARD:** `vulkan_renderer.hpp` `#include`s `render_scene.hpp` → delete/re-point vulkan_renderer FIRST,
THEN render_scene. Do not delete render_scene before its sole consumer.
**Locally verifiable:** YES (compile/link + grep). Independent of RT kernels but shares the render dir; run after
`rt_adapter` exists so the render lib still has a tracer path.

### Phase 4 — CUDA↔Vulkan interop publisher (LEAF; NOT locally verifiable)
**Tasks:** INT-1 (`InteropScatterI` backend-agnostic interface), INT-3 (`World::Backend()` accessor), INT-4
(exportable transform SSBO in raster+present renderers), INT-5 (instanced vertex-shader variant), INT-2 (CUDA
scatter `.cu` + `cudaImportExternalMemory`), INT-6 (`CudaVulkanInteropPublisher` app layer, zero-CUDA), INT-7
(Vulkan↔CUDA external semaphore), INT-8 (wire into `viewer_main` with fallback; recorder stays host-download),
INT-9 (build-link + lint redline acceptance gate).
**Gated by:** build+link into `nuka_phi2`/`nuka_render`/`nuka_runtime_app`; lint 0; the fallback branch taken on
this lavapipe box. Real zero-copy = **OWNER-VERIFIED on a display machine**.
**Locally verifiable:** build-link + host-fallback only (INT-2/INT-7 are `local_verifiable:false`).
**Depends on Phase 0** (external-memory buffer alloc via BufferI; reads `World::Backend()` main stream the sweep
binds).

### Phase 5 — ImGui viewer delta (LEAF; offscreen D1 + host round-trip local, interaction owner-verified)
**Tasks:** VIEW-1 (camera screen→world ray), VIEW-2 (drive-slider panel → `DriveTarget`), VIEW-3 (entity picker →
SceneGraph selection), VIEW-4 (MoveEntity handler → LIVE `BodyPose`/`BasePose`), VIEW-5 (fix
vk-imgui-stale-renderpass), VIEW-6 (StepOnce one-frame-lag + optional keysyms), VIEW-7 (extend
`viewer_frame_smoke` GATE-B).
**Gated by:** GATE-B `viewer_frame_smoke` (offscreen draw-data D1: CmdLists>0 + two-run byte-identical) + host
`UploadField`/`DownloadField` round-trip asserts (runnable on this box). Real interaction = OWNER-VERIFIED.
**Locally verifiable:** offscreen smoke + host Data round-trips YES; live drag/slider NO.
**Depends on:** `command_queue` + Scene/Data (M8/M9, done). Independent of buffer sweep (uses `Data::UploadField`,
not buffer_legacy).

### Phase 6 — Exit gate + ultracode review (locally verifiable except display items)
**Tasks:** EXIT-1 (author + run the exit-gate command set), plus the cross-cutting governance tasks BUF-PLAN
(stream decision), SEQ-PHASE-DAG (this DAG), LINT-KEEPGREEN, COMPLIANCE-AUDIT, M10-BOUNDARY-GUARD, TESTREG.
**Gated by:** §6 command set all green / labeled.

---

## 3. Per-Facet Task Tables

### 3.0 The single atomic-sweep boundary (buffer_legacy ↔ RT ↔ collision/sdf reconciliation)

These three facets **share `buffer_legacy.hpp`**. There is **ONE atomic sweep** with **ONE stream-binding
decision** and **ONE D1 re-gate (BUF-12)**. Reconciliation:
- **Buffer facet (A) OWNS the sweep mechanics**: BUF-0 (decide stream), BUF-1 (v2 transfer helpers), BUF-3..14.
- **RT facet (C) RT-7** (migrate `rt/*.cu` buffers) is **executed inside the sweep**, not separately. RtBackendI's
  BufferI output contract (RT-1) is the pre-built seam.
- **collision/sdf facet (B) COL-5/6/7** (delete sdf_device_world, migrate SAP oracle, re-point test includes) are
  **executed inside the sweep**.
- Single `buffer_legacy.hpp` deletion (BUF-13) happens exactly once, when the LAST consumer (across A+B+C) is
  migrated. **Coordinate so collision/RT do not re-touch the same signatures the buffer facet threads.**
- The diffsim SOURCE files `diffsim/sdf_contact_ift.{cpp,hpp}` + `collision/sdf_contact_adjoint.{cpp,hpp}` carry
  **NO** buffer_legacy (pure host fp64) — only their TEST files + the diffsim SIBLINGS (ift_runner/kkt/tape/etc.)
  migrate.

### 3.A Facet — buffer_legacy FULL SWEEP (T13)

| Task | Title | Files | Risk | Gate | Local |
|---|---|---|---|---|---|
| **BUF-0** | Lock the single stream-binding decision + write contract | docs/plans/2026-06-11-...md | low | Decision recorded + reviewer sign-off (no build) | Y |
| **BUF-1** | v2 host transfer helpers (`buffer_transfer.hpp`) + `Backend→BufferType` accessor | src/phi/buffer_transfer.hpp | low | header compiles standalone (nvcc + g++-10) | Y |
| **BUF-2** | Delete dead c_abi `BufferRecord`/`BufferTable` (VERIFIED zero callers) | src/c_abi/{internal.hpp,handle_table.hpp,error.cpp} | low | `git grep BufferRecord\|BufferTable src/`==0; nuka_c_abi links | Y |
| **BUF-3** | Migrate HUB #1 `articulation_state.{hpp,cpp}` (29 buffers→v2; Upload/Download take `Backend*,Device*`; KEEP byte counts/order EXACT) | src/runtime/articulation/articulation_state.{hpp,cpp} | **high** | nuka_articulation recompiles; featherstone_oracle_harness compiles against new sig | Y |
| **BUF-4** | Resolve `articulation_types.cuh` duplication (collapse vs keep — owner DEC) | src/phi/backend_cuda/ops/articulation_types.cuh, articulation_state.hpp | med | static_asserts pass; nk arena seam test compiles | Y |
| **BUF-5** | Migrate diffsim solver leaves (sparse_solver_backend/cg, kkt_builder) | src/diffsim/{sparse_solver_backend.hpp,sparse_solver_cg.{hpp,cu},kkt_builder.{hpp,cu}} | **high** | kkt_build (A byte-symmetry) + cg_vs_dense byte-exact PASS | Y |
| **BUF-6** | Migrate diffsim orchestration (tape, checkpoint, recompute, backward_runner, ift_runner, osc_adjoint, step_backward) | src/diffsim/{tape,checkpoint,recompute_orchestrator,backward_runner,ift_runner,osc_adjoint,step_backward}.* | **high** | aba_reverse/ift_backward/multistep/osc_adjoint byte-exact incl D1TwoRunByteExact | Y |
| **BUF-7** | Migrate c_abi diffsim production seam (5 allocs → BufferAlloc; build runners with v2 Backend*/Device*) | src/c_abi/diffsim.cpp | **high** | nuka_c_abi links; c_abi diffsim ctest byte-exact | Y |
| **BUF-8** | Migrate collision tree (broadphase_lbvh, candidate_pair, cross_system_query, particle_uniform_grid, gpu/broadphase, particle_candidate_pairs) | src/collision/{...},src/collision/gpu/broadphase.{cuh,cu} | med | collision determinism + lbvh/sap pair-set + cross-system PASS | Y |
| **BUF-9** | Migrate `sdf_device_world` + `sdf_contact_ift` (owner DEC: migrate-in-place vs Model-fold — but see COL-5 = DELETE) | src/runtime/sdf/sdf_device_world.*, src/diffsim/sdf_contact_ift.* | med | sdf_contact_ift + sdf_device_upload PASS *(superseded by COL-5 delete)* | Y |
| **BUF-10** | Migrate invariants_gpu + rt/*.cu + sensor kernels (`MakeDefaultDeviceContext`→`InitBestDevice`+`DeviceInitBackend`) | src/core/diagnostics/invariants_gpu.cu, src/rt/{bvh_ray_traversal,scene_render,two_level_render}.cu | med | rt offscreen smoke recompiles+runs; invariants_gpu recompiles | Y |
| **BUF-11** | Migrate ~40 test harnesses to v2 + `Backend*` (featherstone_oracle_harness = golden producer FIRST; foot_chain_jacobian fixture; rewrite test_capabilities as v2 smoke) | tests/{oracle,diffsim,diffsim/solver,runtime,constraint,solver,collision,perf,sensor,phi}/... (40 files) | **high** | all affected ctest recompile+PASS; no TU includes both buffer headers | Y |
| **BUF-12** | **THE single D1 re-gate** (owner goldens + determinism + perf) | — | **high** | `ctest -L full && -L perf` green; every golden byte-identical | Y |
| **BUF-13** | Delete `buffer_legacy.hpp` + `backend_cuda/buffer.cu` (remove from nuka_phi) | src/phi/buffer_legacy.hpp, src/phi/backend_cuda/buffer.cu, src/CMakeLists.txt | med | `git grep buffer_legacy`==0; full repo links; lint 0 | Y |
| **BUF-14** | Fast-follow: migrate remaining DeviceContext/OwnedStream/StreamView consumers (123-65≈51 residual) + delete legacy stream/context types (owner DEC: 2-stage) | src/phi/{device_context,owned_stream,stream_view,stream}.*, backend_cuda/owned_stream.cu, src/CMakeLists.txt, c_abi/{device,internal} | **high** | `git grep OwnedStream\|DeviceContext\|StreamView`==0; ctest -L full green (no-op D1); lint 0 | Y |

Full migrate file list (BUF-8/11) — collision: broadphase_lbvh.hpp, candidate_pair.hpp, cross_system_query.hpp,
particle_uniform_grid.hpp, gpu/broadphase.{cuh,cu}, particle_candidate_pairs.cu, rigid_candidate_pairs.{cu,hpp},
lbvh_refit.{cu,cuh}. core-diag: invariants_gpu.{cu,cuh}, invariants.{cpp,hpp}. diffsim/solver:
{gmres,minres,ilu0_preconditioner,triangular_solve}.{cu,hpp}. c_abi: noise.cpp. Tests (40): featherstone_oracle_harness;
test_aba_reverse_fd, test_ift_vs_tape_backward, test_multistep_backward, test_osc_adjoint, test_kkt_build,
test_cg_vs_dense + diffsim/solver/{block_jacobi_tuning,cg_vs_dense,cg_convergence_rate,gmres_nonsymmetric,
ilu0_factorization,ilu0_vs_jacobi_convergence,minres_indefinite,minres_vs_dense_indefinite};
runtime/{articulation_contact_rows,articulation_contacts,articulation_inertia_m,contact_chain_jacobian,
dof_above18_honesty,sdf_device_upload}; constraint/test_reaction_providers; solver/{foot_chain_jacobian.hpp,
foot_box_coresidence,foot_box_mjx_parity,foot_ground_mjx_parity,foot_ground_subsume}; collision/{link_aabb.cu,
cross_system_matrix,cross_system_query,gjk_epa_convex,lbvh_filtered_pairs,lbvh_vs_sap_pair_set,
link_rigid_candidate_pairs,particle_grid_correctness,particle_grid_determinism}; perf/{1m_particles_grid_perf,
lbvh_50k_bodies}; sensor/{n1_gaussian.cu,n1_poisson.cu}; phi/test_capabilities (delete-or-rewrite).

### 3.B Facet — collision/SDF cluster

| Task | Title | Files | Risk | Gate | Local |
|---|---|---|---|---|---|
| **COL-1** | Re-assert SDF-tier forward gate (salvage `git show fcae2a6:tests/collision/test_sdf_tier_wired.cpp` VERBATIM) | tests/collision/test_sdf_tier_wired.cpp, tests/CMakeLists.txt | low | `ctest -R nuka_narrowphase_dispatch_test` — BoxOnBox + TwoRunByteIdenticalManifold green | Y |
| **COL-2** | Highest-directive audit: cluster is NOT a special world | — (git-grep) | low | `git grep -niE 'h1\|cup\|grasp\|union\|go2\|kitchen'` over cluster → zero physics special-case hits | Y |
| **COL-3** | Retarget `sdf_contact.hpp` doc-comment off `SdfDeviceWorld` → Model sdf_* tables; fix model.hpp stale comment | src/collision/sdf_contact.hpp, src/nk/model/model.hpp | low | lint clean; nuka_collision+nuka_nk build unchanged (comment-only) | Y |
| **COL-4** | Decide dead `sdf_contact.cu` device kernel disposition (owner DEC; default KEEP whole file) | src/collision/sdf_contact.cu | med | if kept: no-op build green; if kernel deleted: nuka_collision_gpu + adjoint/dispatch tests green | Y |
| **COL-5** | DELETE redundant `sdf_device_world.{cu,hpp}` + its test (INSIDE buffer sweep window; coverage-fold guard FIRST) | src/runtime/sdf/sdf_device_world.{cu,hpp}, tests/runtime/test_sdf_device_upload.cu, tests/CMakeLists.txt, src/CMakeLists.txt | med | nuka_runtime_gpu green; `git grep SdfDeviceWorld src/ tests/`==0; ctest -L full + SDF precision oracle green | Y |
| **COL-6** | Migrate SAP-broadphase oracle `gpu/broadphase.{cu,cuh}` + its test off buffer_legacy (KEPT oracle, not delete) | src/collision/gpu/broadphase.{cu,cuh}, tests/collision/test_lbvh_vs_sap_pair_set.cpp | **high** | `ctest -R nuka_lbvh_vs_sap_test` — all 4 + D1TwoRunByteExactPairList byte-identical | Y |
| **COL-7** | Re-point remaining cluster TEST buffer_legacy includes (test_sdf_contact_ift_fd) in the sweep | tests/diffsim/test_sdf_contact_ift_fd.cpp | med | ift/adjoint/contact_stream/analytical/dispatch tests green; cluster `git grep buffer_legacy tests/`==0 | Y |
| **COL-8** | §3.10 audit: cluster needs NO op-ify / interface-ify (out of lint redline scope; M5 op already covers runtime path) | — | low | physics_smell.py rc=0 over cluster; written §3.10 verdict | Y |

KEEP (no action, confirm-only): narrowphase_dispatch.hpp, contact_stream_driver.{hpp,cpp} (oracle for 8 gates),
sdf_contact.hpp (`find_sdf_contact_newton`), sdf_contact_adjoint.{cpp,hpp}, diffsim/sdf_contact_ift.{cpp,hpp},
sparse_sdf_query.cuh, narrowphase_sdf.cu (M5 runtime op), convex_narrowphase.hpp, analytical_manifold.hpp,
codegen/generated/{rigid,featherstone}_sdf_contact_forward.cu (GENERATED — never hand-edit), test_narrowphase_dispatch,
test_contact_stream_driver, test_sdf_contact_adjoint_fd, test_sdf_contact_ift_fd (the gate kept; only its include migrates).

### 3.C Facet — RT path tracer + RtBackendI homing

| Task | Title | Files | Risk | Gate | Local |
|---|---|---|---|---|---|
| **RT-0** | Confirm render_scene deletion safety + overlay disposition (owner DEC-1) | render_scene.{hpp,cpp}, vulkan_renderer.{hpp,cpp} | low | git-grep zero live external callers | Y |
| **RT-2** | Split RT headers Tier-A (interface-visible, CUDA-free: camera/framebuffer/material + scene-desc PODs) vs Tier-B (device .cuh) | src/rt/{two_level_render,scene_render,bvh_ray_traversal,camera,framebuffer,material}.hpp, prim_id.cuh | med | Tier-A pass physics_smell.py zero CUDA; tests/rt/* compile | Y |
| **RT-CMAKE** | Homed RT target keeps `--fmad=false` (the #1 D1 trap); do NOT fold into nuka_phi2 (default `--fmad=true`) | src/CMakeLists.txt | **high** | `grep --fmad=false build-cuda128/.../<rt target>/flags.make`; rt memcmp==0 | Y |
| **RT-3** | LINE-FAITHFUL `git mv` 3 `.cu` + 6 `.cuh` → `src/phi/backend_cuda/rt/`; include-path-only diff; implement RtBackendI concrete backend | src/phi/backend_cuda/rt/{two_level_render,scene_render,bvh_ray_traversal}.cu + {bvh_traverse_impl,intersect_primitives,ray_box,shading,instance_transform,prim_id}.cuh | **high** | nuka_rt_{two_level,scene_render,traversal}_test memcmp==0; `git show --stat` = include-only | Y |
| **RT-1** | Define `src/render/rt_backend.hpp` — `RtBackendI` vtable (build_blas/build_tlas/trace/free; output via phi v2 BufferI; CUDA-token-free) | src/render/rt_backend.hpp | med | physics_smell.py zero CUDA; compiles under g++ | Y |
| **RT-4** | Author `src/render/rt_adapter.{hpp,cpp}` = `RenderWorldToTwoLevelScene` (host, CUDA-free) | src/render/rt_adapter.{hpp,cpp} | med | physics_smell.py==0; compiles into nuka_render | Y |
| **RT-5** | NEW offscreen D1 gate: RenderWorld→rt_adapter→RtBackendI→image, dispatch-vs-replay 6-AOV memcmp + adapter-vs-oracle parity | tests/rt/test_render_world_rt.cpp, tests/CMakeLists.txt, tests/rt/test_h1_cup_pbr_render.cpp | med | `ctest -R rt_render_world` — 6-AOV two-run memcmp==0 + parity + coverage>0 | Y |
| **RT-6** | Delete `render_scene.{hpp,cpp}`; remove RenderScene-overlay path + retarget 2 tests (Phase 3) | render_scene.{hpp,cpp}, vulkan_renderer.{hpp,cpp}, src/CMakeLists.txt, tests/render/test_vulkan_backend.cpp, tests/perf/test_vulkan_renderscene_timing.cpp | med | git-grep zero render_scene/RenderScene; render+render_test+perf_test build; lint 0 | Y |
| **RT-7** | RT buffer migration to phi v2 BufferI (executed WITH the BUF sweep, single D1 re-gate) | src/phi/backend_cuda/rt/{two_level_render,scene_render,bvh_ray_traversal}.cu | **high** | rt tests + RT-5 gate memcmp==0 post-sweep | Y |

KEEP (oracle/D1, include-path update only): tests/rt/{test_two_level_render, test_scene_render,
test_bvh_ray_traversal, test_two_level_bench}.cpp. Tier-A POD headers (camera/framebuffer/material) stay in src/rt
(owner DEC: src/rt becomes an interface-types dir; only `.cu`/`.cuh` relocate).

### 3.D Facet — render_scene delete + old vulkan_renderer
Folded into **RT-0 + RT-6** above (Phase 3). Decision DEC-1: recommend **delete BOTH** `render_scene.{hpp,cpp}`
and the RenderScene-consuming overlay wrappers (`RenderSceneDebugOverlayVulkan`/`RenderSceneVulkan` +
BuildRenderSceneCommands/ToVulkanCommand/MeshColor/FindMaterial), keep the live `VulkanDebugDrawCommand`-list
overlay; retarget `test_vulkan_backend.cpp` + `test_vulkan_renderscene_timing.cpp` to feed command lists directly.
Also untangle `vulkan_offscreen_types.hpp` (it `#include`s vulkan_renderer.hpp — keep only the offscreen-report
types the M8 raster path needs).

### 3.E Facet — CUDA↔Vulkan interop publisher

| Task | Title | Files | Risk | Gate | Local |
|---|---|---|---|---|---|
| **INT-1** | Define backend-agnostic `InteropScatterI` interface + external-memory descriptor (pure C++, zero CUDA/Vulkan token; fd=plain int) | src/phi/backend_cuda/interop/cuda_vk_scatter.hpp | low | header compiles standalone (no cuda/vulkan include); grep tokens==0 | Y |
| **INT-3** | Expose `phi::Backend* World::Backend()` (narrow, redline-safe — phi::Backend is not a CUDA token) | src/nk/pipeline/world.hpp | low | lint src/nk/** green; nuka_nk builds | Y |
| **INT-4** | Exportable per-instance transform SSBO in raster+present renderers (OPT-IN, default-off; VK_KHR_external_memory_fd; zero CUDA) | src/render/raster/vulkan_{raster,present}_renderer.{hpp,cpp} | **high** | lint render zero CUDA; offscreen parity/raster_smoke byte-identical (interop OFF default) | Y |
| **INT-5** | Instanced vertex-shader variant reading transform from SSBO by gl_InstanceIndex (2nd pipeline, interop-only) | src/render/raster/shaders/mesh_instanced.vert + vulkan_{raster,present}_renderer.cpp | med | SPIR-V compiles; default pipeline (D1 pixels) untouched | Y |
| **INT-2** | CUDA scatter kernel + `cudaImportExternalMemory` (`.cu` OUTSIDE redline; reuse `math::Transform` HD operator*; LaunchCuda; no float atomics) | src/phi/backend_cuda/interop/cuda_vk_scatter.cu | **high** | compiles+links into nuka_phi2; CANNOT run here (build-only) | **N** |
| **INT-6** | `CudaVulkanInteropPublisher` (app layer, ZERO CUDA token; includes only the backend-agnostic seam) | src/runtime/app/cuda_vulkan_interop.{hpp,cpp} | med | lint runtime/app green; nuka_runtime_app links; physics_smell.py whole-repo | Y |
| **INT-7** | Vulkan↔CUDA external semaphore (timeline/binary; signal after scatter, wait before draw) | src/phi/backend_cuda/interop/cuda_vk_scatter.{cu,hpp}, vulkan_present_renderer.cpp | **high** | compiles+links; owner-verified on display | **N** |
| **INT-8** | Wire interop into `viewer_main` with graceful fallback to HostDownloadPublisher; recorder stays host-download | src/runtime/app/viewer/viewer_main.cpp | med | viewer_main links; fallback branch taken on lavapipe; frame_loop_smoke green | Y |
| **INT-9** | Build-link + lint redline acceptance gate (smoke TU asserts clean degrade to "interop unavailable") | tests/scenario/, tests/CMakeLists.txt | low | ctest builds+links interop smoke (fallback); physics_smell.py green; redline grep==0 | Y |

KEEP: pose_publisher.{hpp,cpp} (the swap seam — interop is additive behind it), systems.hpp (sole Publish call
site, MUST NOT change), simulation.{hpp,cpp} (loop unchanged; publisher selection at construction),
render_world.hpp (per-instance binding source), dlpack_table.hpp (descriptor pattern to mirror, do not modify),
cuda_internal.cuh / launch.cuh / readout.cu (the sanctioned stream + launch + scatter idiom). recorder.cpp =
KEEP on HostDownloadPublisher (protects the D1 mp4 gate).

### 3.F Facet — ImGui viewer delta

| Task | Title | Files | Risk | Gate | Local |
|---|---|---|---|---|---|
| **VIEW-1** | Camera screen→world ray + drag-plane unproject (shared picker math; pure host) | src/runtime/app/viewer/camera_controller.{hpp,cpp} | low | recompile nuka_viewer; center-ray + known-plane-hit host assert (camera_ray smoke) | Y |
| **VIEW-2** | Drive-slider panel writing `FieldId::DriveTarget` into nk Data (GENERIC per-DOF; env-major offset) | imgui_layer.{hpp,cpp}, viewer_main.cpp, simulation.hpp | med | build+link; extend viewer_frame_smoke fixed-state CmdLists>0 + two-run byte-identical; lint green | Y |
| **VIEW-3** | Entity picker: ray vs RenderWorld AABBs → SceneGraph selection (retain SceneIR/graph post-cook) | viewer_main.cpp, imgui_layer.{hpp,cpp} | med | build+link; Entity-panel recording with fixed selection in smoke (CmdLists>0 + D1) | Y |
| **VIEW-4** | MoveEntity handler: drag writes LIVE nk Data `BodyPose`/`BasePose` (NOT SceneIR); replace systems.hpp:93 stub | viewer_main.cpp, systems.hpp, simulation.hpp | **high** | NON-interactive scenario unit: push MoveEntity for cup → FramePublish → DownloadField BodyPose == requested (host, runnable); lint green | Y |
| **VIEW-5** | Fix vk-imgui-stale-renderpass-after-recreate (handle PresentFrameResult::Recreated) | viewer_main.cpp, src/render/imgui/nuka_imgui.{hpp,cpp} | med | build+link nuka_viewer+imgui+present; code-path + compile gate; live resize owner-verified | Y |
| **VIEW-6** | StepOnce one-frame-lag fix; (optional, defer) camera evdev→xcb-keysyms | viewer_main.cpp, camera_controller.cpp | low | host unit: Push(StepOnce)+FramePublish → exactly one step same-frame; keysyms owner-only | Y |
| **VIEW-7** | Extend GATE-B `viewer_frame_smoke` to cover full M11 viewer (D1 offscreen + MoveEntity/DriveTarget round-trips) | tests/scenario/viewer_frame_smoke.cpp | low | `ctest -R viewer_frame_smoke` all asserts green; lint green | Y |

KEEP: command_queue.hpp (MoveEntity payload pre-built M8 — viewer just starts PUSHING), pose_publisher.{hpp,cpp}
(viewer keeps HostDownloadPublisher), simulation.cpp (offscreen Frame untouched), scene_graph.hpp (editor
selection seam, plan L247), c_abi/scene.cpp (`nuka_scene_set_local` — DO NOT route live drag through it),
data.hpp + field_ids.hpp (the write seams, generated — never hand-edit), render_world.hpp, src/CMakeLists.txt
(no new viewer TUs — keep panels inline in imgui_layer.cpp), banned_patterns.yaml (hard zero-CUDA constraint).

### 3.G Facet — exit-gate / §3.10 / sequencing / compliance (governance)

| Task | Title | Files | Risk | Gate | Local |
|---|---|---|---|---|---|
| **EXIT-1** | Author the M11 exit-gate command set + register as closing checklist | docs/architecture/m11-render-gates.md, plan | low | doc lists each command + LOCAL vs OWNER label; review-completeness | Y |
| **BUF-PLAN** | Pin the single stream-binding decision (= BUF-0; bind to existing per-device stream → pure type swap) | c_abi/internal.hpp, c_abi/device.cpp | **high** | design review; proof = post-sweep re-gate byte-exact | Y |
| **SEQ-PHASE-DAG** | Lock the phase-order DAG (this §2); each boundary links green before next deletion | subagent-plans/m11-recon.md | med | DAG review; `cmake --build` green at each boundary | Y |
| **LINT-KEEPGREEN** | Verify engine zero-CUDA redline stays green after rt_adapter+interop+viewer land (NO new scope; only keep green) | banned_patterns.yaml, allowlist.yaml | low | physics_smell.py 0 violations whole-repo | Y |
| **COMPLIANCE-AUDIT** | 最高指令 audit: MoveEntity + interop write the GENERAL path; no special world | command_queue.hpp, cuda_vulkan_interop.cpp, imgui_layer.cpp | med | git-grep MoveEntity→command_queue+Data; publisher = PosePublisher no world-switch | Y |
| **M10-BOUNDARY-GUARD** | Confirm M11 touches no M10 item (RL/PD-stance/wrench-control/autoreset/control-mode/SG-spec) | subagent-plans/m11-recon.md | low | `git grep -lE 'autoreset\|control_mode\|PPO\|train_' src/` shows no M11 edits; ledger review | Y |
| **TESTREG** | Register new/re-pointed tests in 5-category exe + label scheme (rt D1 stays in CUDA region OUTSIDE NK_BUILD_VULKAN) | tests/CMakeLists.txt | med | ctest -L full lists new tests; rt D1 in default build; viewer/interop only in build-viewer | Y |

---

## 4. D1 Byte-Exact Risk Register (consolidated)

| # | Risk | Mitigation | Must re-pass |
|---|---|---|---|
| **R1 (THE #1)** | **ASYNC-STREAM REBIND.** Legacy `CopyFromHost/ToHost` = `cudaMemcpy` on DEFAULT stream (sync barrier); v2 `BufferUpload` = `cudaMemcpyAsync` on `CudaBackendMainStream` (async, no barrier). c_abi today runs TWO streams (`DeviceRecord.owned_stream` vs `backend->main`); migration collapses diffsim onto backend->main, changing the stream golden-producing kernels run on. | Bind v2 ops to the EXISTING per-device context stream (BUF-PLAN) → pure type swap, identical launch stream + sync points, preserved reduction order. Rebind EVERY migrated consumer's kernels to ONE stream. NEVER mix legacy-default + v2-backend in one TU. | union_cook_golden==0.0; cook_settle_determinism memcmp==0; featherstone_{go2,go2_floating,h1}_random_sample goldens; nuka_diffsim_{aba_reverse,ift_backward,multistep_backward,osc_adjoint,kkt_build}_test; sdf_contact_ift; cg/minres/gmres/ilu0; foot_* MJX parity; particle-grid determinism; `ctest -L perf` redlines (union≤5ms, grasp≥11k eps, Go2 4096≤~1µs, 1M-particles/50k-LBVH) |
| **R2** | **CROSS-STREAM SPLIT COLLAPSE in diffsim.** Inter-phase `cudaStreamSynchronize` between orchestration (legacy stream) and the op-ified backward op (backend->main) must survive the collapse. | Every `context_.stream.Synchronize()` → `BackendSynchronize(backend)`; checkpoint/recompute async copies → backend-main copies; verify happens-before vs the backward op. | ift_vs_tape; multistep; D1TwoRunByteExact |
| **R3** | **GOLDEN PRODUCTION PATH on migrated harness.** featherstone_oracle_harness produces OWNER-PROTECTED goldens; after migration runs on backend->main not default stream. | Kernel bodies untouched (only stream+handle change); fixed-order warp reductions + zero float atomics → stream-invariant. Red = real bug, STOP, do not rebaseline. | featherstone_*_random_sample.bin |
| **R4** | **FMAD TRAP (RT, highest RT risk).** nuka_phi2 (backend_cuda/ops) = nvcc default `--fmad=true`; homed RT MUST keep `--fmad=false` (paired with host oracle `-ffp-contract=off`). Folding RT into nuka_phi2 silently diverges every RT golden. | Dedicated RT lib (`nuka_phi2_rt`) or per-source `--fmad=false`; verify build-cuda128 flags.make. | nuka_rt_two_level_test depth/normal/color/albedo/uv/prim memcmp==0 |
| **R5** | **RT INCLUDE-PATH-ONLY MOVE.** Shared HD headers (ray_box/intersect_primitives/shading/instance_transform/prim_id/camera) `#include`d VERBATIM by CPU oracle TUs for host==device bit-exactness. Any op reorder / cast / FMA-introducing refactor breaks parity. | `git mv` + `git show --stat` proving body-identical; existing two-run + oracle memcmp gates catch drift. | rt offscreen D1 gates |
| **R6** | **PrimId TOTAL-ORDER.** D1-by-construction tie-break = `PackPrimId` packing instance HIGH (atomic-free, one writer/pixel). | `kInstanceBits/kPrimBits/kMaxInstances/kMaxBlasPrims` move bit-identical; never rebalance. | prim AOV memcmp |
| **R7** | **OSC ArticulationDeviceState SURFACE CHANGE.** Re-pointing the buffer member type changes the kernel-PARAM struct feeding D1 adjoint-FD. | Keep field order/types/alignment identical (only handle type changes); cook-fidelity contract "Model bytes == UploadArticulationState bytes" preserved (size_t math, empty→zero-byte skip-copy). | osc_adjoint; aba_reverse; ift_vs_tape; union_cook_golden |
| **R8** | **KKT lazy-alloc survival.** kkt_builder's size-guarded lazy `cudaMalloc` is deliberately hot_path-lint-excluded; naive per-Run BufferI alloc trips `hot_path_cuda_malloc` + changes timing. | Keep allocate-once-reuse cache in cuda_buffer.cu/v2 allocator. | test_kkt_build + perf |
| **R9** | **FLAG-BUFFER read-after-write.** ift_runner `indef_flag_/nonsym_flag_` read host-side after a detector kernel; v2 async upload doesn't sync, raw read sees stale. | Route every host-observable read through `BufferDownload` (which syncs); audit ift_runner.cu / sparse_solver_cg.cu. | ift D1 gates |
| **R10** | **render_scene/vulkan_renderer DELETION ORDER.** `vulkan_renderer.hpp` `#include`s `render_scene.hpp` → deleting render_scene first = compile break. | Phase 3: delete/re-point vulkan_renderer + 2 tests FIRST, then render_scene; link green between. | nuka_render build |
| **R11** | **INTEROP DEVICE SCATTER in zero-CUDA scope.** `cuda_vulkan_interop.cpp` lives in src/runtime/app/** (redline) — raw kernel/cuda* = lint RED. | Scatter routes through `InteropScatterI` → `.cu` in `backend_cuda/interop/` via launch.cuh; app TU includes ONLY the backend-agnostic seam. | physics_smell.py; redline grep==0 |
| **R12** | **GATE-B viewer byte-exact composite.** Offscreen `viewer_frame_smoke` asserts two-run memcmp==0; new Drive/Entity panels MUST record from a FIXED ViewerUiState, no time/animation widget. | Fixed drive_targets/selected entity; no time-seeded widgets; reuse 4-frame dock warm-up. | viewer_frame_smoke |
| **R13** | **VIEWER WRITES vs frozen .nks.** Interactive UploadField writes are RUNTIME Data edits, NOT .nks edits. Implementer must NEVER persist viewer edits back (set_local+Save would). | Live drag uses Data::UploadField only; never call set_local in the drag path. | union_cook_golden; h1_grasp_lift (cook from on-disk .nks, unaffected) |
| **R14** | **INTEROP BYTE-EXACT TRANSFORM.** Scatter kernel `world_xform = fk * cached_visual_local` must be bit-identical to HostDownloadPublisher host composition. | `math::Transform::operator*` is `__host__ __device__` — kernel calls the EXACT SAME operator; device TransformToMatrix line-faithful to host column-major. Forbid reimplementing the math. | (interop not in any golden; correctness = owner display) |
| **R15** | **cook-real-normals parse-vn (M8/M8.5 named debt, D1-SENSITIVE .nka byte change).** Carried to M11 — any `.nka` normal-parse change is a D1 byte risk for the render-data goldens. | Verify before any render-data touch; if untouched in M11 (recommend), no action; if touched, re-gate the offscreen render parity golden. | render_physics_parity offscreen pixels |
| **R16** | **sdf_device_world deletion coverage loss.** test_sdf_device_upload is the only cooked-table→device round-trip assertion. | COL-5 coverage-fold guard: confirm/add Model sdf_cell_* round-trip + cell-count assertion BEFORE deletion. | SDF precision oracle; M5 op smoke |

---

## 5. §3.10 Homing Completeness

The engine zero-CUDA-token redline (src/nk, src/scene, src/render, src/runtime/app — banned_patterns.yaml
L147-154) is **ALREADY GREEN** (git-grep finds zero `<<<`/`cuda*(`/`<cuda_runtime>`/`phi/backend_cuda` tokens and
zero bare `.cu` in those scopes). So §3.10's M11 row is **NOT "clean up engine CUDA" (done)** — it is exactly:

| §3.10 M11 row item | Today | M11 action | New home | Redline status |
|---|---|---|---|---|
| **RT backend interface (RtBackendI)** | RT kernels live in non-interface `src/rt/` | RT-1/2/3: extract `render::RtBackendI` (CUDA-free interface) + LINE-FAITHFUL home `.cu`/`.cuh` | impl → `src/phi/backend_cuda/rt/` (NEW); interface → `src/render/rt_backend.hpp` + `rt_adapter.{hpp,cpp}` | render interface ZERO CUDA |
| **CUDA↔Vulkan interop (device scatter via launch.cuh)** | none | INT-1/2/6: `InteropScatterI` interface + `.cu` scatter | impl → `src/phi/backend_cuda/interop/` (NEW); interface/publisher → `src/runtime/app/cuda_vulkan_interop.{hpp,cpp}` (ZERO CUDA) | runtime/app ZERO CUDA |

**Redline ADDITIONS:** NONE needed (render + runtime/app already covered). The M11 job is to **KEEP it green** when
rt_adapter/rt_backend/interop/viewer land. Verify: (a) rt_adapter + rt_backend.hpp zero CUDA tokens; (b)
cuda_vulkan_interop.cpp routes the device scatter behind the interface (not a raw kernel); (c) NO allowlist.yaml
entry added for any engine-scope CUDA (an entry there = breach). Optionally add a generated-header rule for
`backend_cuda/rt/**` if codegen ever emits there (not in M11 scope).

**Out-of-redline (CUDA allowed, no interface-ify needed):** src/collision, src/runtime/sdf, src/diffsim — the
collision/SDF/IFT cluster needs NO interface-ification for the redline. §3.10 "is-it-an-op" test: the RUNTIME SDF
narrowphase IS already an op (`NkOp::NarrowphaseSdf`, M5); the Newton-witness IFT family is a host oracle /
differentiable reference (not a per-step op) → stays as infra. Host narrowphase_dispatch + contact_stream_driver
are host test oracles, not device ops.

---

## 6. M11 Exit Gate

Run from REPO ROOT with `CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64`.
Superset of M9's gate. **NEVER reconfigure cmake** (build-cuda128 has the correct CUDA 12.8).

| Gate | Command | Pass | Where |
|---|---|---|---|
| **G-grep-buf** | `git grep -lE "buffer_legacy\|\b(OwnedStream\|DeviceContext\|UploadVector\|StreamView)\b" src/ tests/` + `git ls-files src/phi/buffer_legacy.hpp` | both EMPTY | **LOCAL** |
| **G-grep-scene** | `git grep -lE "render_scene\|render::RenderScene\|BuildRenderScene\|RenderSceneVulkan\|RenderSceneDebugOverlayVulkan" src/ tests/` | ZERO (only intentional survivors) | **LOCAL** |
| **G-grep-sdfworld** | `git grep -lE "SdfDeviceWorld\|sdf_device_world" src/ tests/` | ZERO (except deletion-note comments) | **LOCAL** |
| **G-lint** | `/root/miniconda3/bin/python tools/lint/physics_smell.py` | 0 violations (rt_adapter + cuda_vulkan_interop show ZERO CUDA tokens) | **LOCAL** |
| **G-ctest-full** | `ctest --test-dir build-cuda128 -L full` | green | **LOCAL** |
| **G-ctest-perf** | `ctest --test-dir build-cuda128 -L perf` | green (union≤5ms, grasp≥11k eps, Go2 4096≤~1µs, 1M-particles/50k-LBVH unchanged) | **LOCAL** |
| **G-rtD1** | `ctest -R "nuka_rt_(two_level\|scene_render\|traversal\|render_world)"` | 6-AOV two-run memcmp==0 (RT-via-RtBackendI) | **LOCAL** |
| **G-golden** | union_cook_golden==0.0 + cook_settle_determinism byte-identical + adjoint-FD/IFT/OSC/kkt/sparse-solver family ALL green AFTER async-stream sweep; every `tests/oracle/golden/**` byte-identical | green | **LOCAL** |
| **G-sdftier** | `ctest -R nuka_narrowphase_dispatch_test` (incl salvaged BoxOnBox) | green | **LOCAL** |
| **G-buildlink** | build-viewer (NK_BUILD_VULKAN_VALIDATION=ON) compiles+links viewer delta + cuda_vulkan_interop + rt_adapter; non-device branches run (interop fallback, MoveEntity/DriveTarget host round-trips, viewer_frame_smoke offscreen D1) | links + non-device asserts green | **LOCAL** (build-link + offscreen) |
| **G-display** | m8-render-gates §5 checklist: window opens / robot animates / drag-entity edits scene / interop zero-D2H / RT beauty on real NVIDIA | manual | **OWNER-ON-DISPLAY** (CUDA-GPU + llvmpipe box cannot run) |

---

## 7. OWNER DECISIONS NEEDED (consolidated, deduplicated)

> Surfaced via AskUserQuestion before implementation (mirrors M9's 4 rulings). CONTROLLER-RESOLVABLE items marked
> with the call made under existing owner rulings (strict-dominance / never-ruled).

**OD-1 — Interop CUDA placement** (the publisher path is named in src/runtime/app/ but that's a ZERO-CUDA scope).
Options: (A) interface in `cuda_vulkan_interop.cpp` zero-CUDA + scatter kernel/`cudaImportExternalMemory` in NEW
`src/phi/backend_cuda/interop/*.cu`; (B) relax the lint redline for this TU; (C) move whole publisher under
backend_cuda. → **CONTROLLER-RESOLVABLE = (A).** §3.10 literally says "device scatter kernel via launch.cuh" (=
backend_cuda); the redline must never break (owner memory + plan explicit); matches the RT facet split exactly.
B is a redline breach (reject).

**OD-2 — Transform SSBO render path** (true zero-copy needs a per-instance transform SSBO + instanced
vertex-shader; current path uses per-draw push constants in host RAM). Options: (A) approve opt-in DEVICE_LOCAL
SSBO + `mesh_instanced.vert`, default-off so D1 oracle byte-identical; (B) reject transform zero-copy (fails the
firm "Vulkan renders REAL physics device state" directive); (C) defer interop to a later milestone. →
**RECOMMEND (A).** The render↔physics-shared-memory directive is FIRM; opt-in/default-off preserves every D1 gate.
(Owner-firm directive exists → surface for confirmation, do not self-resolve.)

**OD-3 — DLPack-Vulkan literalness.** Options: (A) use `VK_KHR_external_memory_fd` + `cudaImportExternalMemory` as
the real bridge, expose through a DLPack(kDLVulkan-shaped) descriptor for API consistency; (B) literal kDLVulkan
DLPack transport (immature); (C) skip DLPack framing, raw fd only. → **CONTROLLER-RESOLVABLE = (A).** DLPack
describes bytes, does not move them; dlpack_table.hpp already defines the device-descriptor contract — mirror its
shape, actual import is external-memory-fd. Honors directive intent without betting on immature tooling.

**OD-4 — Recorder interop.** Options: (A) recorder stays HostDownloadPublisher (device-independent byte-exact
mp4); (B) recorder swaps to interop when available. → **CONTROLLER-RESOLVABLE = (A).** The recorder is the D1
video gate — must stay device-independent; interop is a viewer-only optimization. (Under D1 discipline,
strict-dominance.)

**OD-5 — Viewer MoveEntity persistence.** Options: live-only (drag perturbs running sim, lost on reset); live +
explicit "Commit to Scene" (set_local + re-cook); live + continuous write-back. → **CONTROLLER-RESOLVABLE =
live-only.** Plan L509 literal requirement ("MoveEntity → write Data"); re-cook risks frozen-.nks discipline; not
asked for.

**OD-6 — Drag velocity zeroing.** Options: zero velocity on drag (teleport-feel, stable); keep velocity
(throw-feel); UI toggle. → **RECOMMEND zero velocity** (intuitive "place here"; one extra UploadField). Minor UX —
surface briefly or let controller take zero-velocity.

**OD-7 — Drive panel granularity.** Options: flat per-DOF sliders (generic); named groups from cooked
name→dof map; both. → **CONTROLLER-RESOLVABLE = flat per-DOF, optionally labeled.** A GENERIC editor (highest
directive: no hardcoded grasp choreography). Naming is cosmetic; write path stays generic UploadField(DriveTarget).

**OD-8 — Camera evdev→xcb-keysyms debt.** Options: fix now; defer. → **CONTROLLER-RESOLVABLE = defer.** Blocks no
M11 gate (MMB-pan works), adds an xcb-keysyms dep; keep documented caveat.

**OD-9 — render_scene + old vulkan_renderer disposition (DEC-1).** Options: (A) delete BOTH overlay wrappers +
render_scene, keep command-list overlay, retarget 2 tests; (B) retarget RenderSceneDebugOverlayVulkan to consume
RenderWorld; (C) keep render_scene as POD-only header. → **RECOMMEND (A).** BuildRenderScene fully dead; overlay
wrappers zero non-test callers; cleanest plan-L499 deletion. (Surface — it deletes test coverage.)

**OD-10 — RT scene-desc POD home (DEC-2).** Options: (A) keep Tier-A PODs in src/rt (becomes interface-types dir,
only `.cu`/`.cuh` move); (B) move into new src/render/rt_types.hpp, delete src/rt; (C) define in rt_backend.hpp. →
**CONTROLLER-RESOLVABLE = (A).** Tier-A headers already CUDA-free (NUKA_RT_HD = nothing under g++); minimizes
churn + keeps git history.

**OD-11 — RT lib target (DEC-3).** Options: (A) own lib `nuka_phi2_rt` with `--fmad=false`; (B) append `.cu` to
nuka_phi2 with per-source `--fmad=false`; (C) keep `nuka_rt_gpu` name, retarget source paths. → **RECOMMEND (A)
or (C).** A separate target makes `--fmad=false` unmissable; B is fragile (one missed source = silent D1 break).
(Surface A-vs-C; reject B.)

**OD-12 — RtBackendI output contract (DEC-4).** Options: (A) caller-provided phi v2 Buffer* outputs (6 AOV); (B)
backend allocates + returns handles; (C) backend returns host Framebuffer. → **CONTROLLER-RESOLVABLE = (A).** Brief
explicitly says "output via phi v2 BufferI"; makes the interop facet a natural future consumer; keep a thin
Framebuffer convenience for tests.

**OD-13 — sdf_contact.cu dead device kernel (COL-4).** Options: KEEP whole file untouched (carry-forward c "defer
with rt"); DELETE only dead `__global__` find_sdf_contact_kernel (keep host AppendSdf* row builders); DELETE whole
file (large blast radius — reject). → **RECOMMEND KEEP whole file untouched in M11.** SDF-IFT family deferred "with
rt"; RL-contact-gradient is M10; dead kernel costs nothing; removing risks the row-class registry link. (Surface;
option 2 is safe middle if owner wants tidiness.)

**OD-14 — M9 plan delete-list #13/#14 vs ledger.** Options: (1) KEEP narrowphase_dispatch/sdf_contact/adjoint/ift
as oracle+diffsim infra, DELETE only redundant sdf_device_world (per ledger + 8 surviving gates); (2) execute
literal M9 delete-list (orphans 8 gates — NOT viable). → **CONTROLLER-RESOLVABLE = (1).** The ledger (owner ruling
#4) overrides the plan delete-list, which predates the M9 RT/buffer slip.

**OD-15 — Salvaged SDF-tier gate home (COL).** Options: fold into nuka_narrowphase_dispatch_test; fresh
nuka_sdf_tier_wired_test; fold into nuka_oracle_test. → **CONTROLLER-RESOLVABLE = fold into
nuka_narrowphase_dispatch_test.** Already links the right libs + `#include`s narrowphase_dispatch.hpp; co-located
with the routing sibling; avoids exe proliferation.

**OD-16 — Stream-binding mechanism (BUF, the D1 anchor).** Options: (1) thread `Backend*`/`Device*`→BufferType*
through ~40 call sites (wide, shallow, correct stream); (2) `BufferTypeForDevice()` global bridge (cheap, risks
wrong stream); (3) hybrid. → **RECOMMEND (1).** The only choice that keeps the stream correct + explicit; the
bridge risks a wrong-stream D1 violation that silently corrupts goldens; matches how nk::World already takes
Device*+Backend*. (Surface — it is the #1 D1 anchor.)

**OD-17 — ArticulationDeviceBuffers hub (now test-only).** Options: (1) migrate the struct to v2 in src/runtime
(reusable test-oracle harness); (2) move into a tests/ fixture, delete from src/runtime; (3) keep + DEPRECATED. →
**CONTROLLER-RESOLVABLE = (1) with a "test-oracle harness, no production consumer" banner.** Relocating now
multiplies the diff across ~15 test files for no D1 benefit; a later milestone can move it.

**OD-18 — articulation_types.cuh duplication (BUF DEC-3).** Options: (1) collapse to single definition
(collision gone post-sweep); (2) keep duplicate; (3) nkops as single home. → **CONTROLLER-RESOLVABLE = (1).** The
cuh docstring states the duplication was transitional pending the legacy delete; byte-identical PODs, no new risk.
(Defer to fast-follow + named debt if de-risking.)

**OD-19 — sdf_device_world disposition (BUF DEC-4 ↔ COL-5).** Options: (1) migrate-in-place to v2; (2) execute the
M9-intended delete/Model-fold; (3) migrate-in-place now, fold later. → **CONTROLLER-RESOLVABLE = (2) DELETE** (per
COL-5: upload duty ALREADY in nk::Model — model.hpp sdf_cell_* + cook_to_model.cpp:610-635 + narrowphase_sdf.cu
LoadSdfGrid; sole consumers are one test + a doc-comment). It is REDUNDANT, not just legacy-coupled — delete in the
sweep window with the COL-5 coverage-fold guard, do not migrate-in-place. (This RESOLVES the apparent
buffer-facet "migrate-in-place" vs collision-facet "delete" conflict in favor of DELETE.)

**OD-20 — Atomicity boundary for legacy stream types (BUF DEC-5).** Options: (1) one mega-commit (buffer_legacy +
DeviceContext + OwnedStream together — unreviewably large); (2) two-stage within M11 (BUF-13 buffers atomic, then
BUF-14 DeviceContext fast-follow); (3) buffer-atomic only, defer DeviceContext past M11 (violates carry-forward
d). → **RECOMMEND (2) two-stage within M11.** The buffer migration is forced atomic by the name collision;
DeviceContext has ~51 consumers that don't touch buffer_legacy and migrate in a tightly-following no-op-D1 commit.
(Surface — it sets the commit granularity.)

---

## 8. Open Questions / Investigate-Further (read-only could not resolve)

1. **Existing entity→cooked-row map post-cook?** Does `CookToModelResult.scene_map` expose entity→Body/Base row,
   or must the MoveEntity handler derive it from `RenderInstance.pose_source.row`? (Recommend pose_source.row —
   already baked from CookedRef; confirm populated for the movable cup (Body) + H1 base (Base) in h1_cup_table.)
2. **Viewer SceneIR/SceneGraph retention.** viewer_main currently cooks a SceneIR then discards it; VIEW-3/VIEW-4
   need it retained (Ecs/graph) for SceneGraph::SetSelected + entity resolution. Confirm no lifetime issue (cook
   moves model into World but scene_map/Ecs remain).
3. **Present window-resize event.** Does XcbWindowSurface deliver a Resize event the viewer should honor, or is
   `OUT_OF_DATE` on next acquire the only trigger? Affects how thoroughly VIEW-5 closes the present-resize debt.
4. **Viewer Reset semantics.** Should drive-slider edits reset on ViewerControl::Reset, and does Reset mean
   re-cook from .nks vs re-seed initial q/qdot/poses? (Today Reset only re-frames the camera — owner-policy.)
5. **N1 sensor noise kernel home.** buffer_transfer docstring listed `cuda_sensors.cu` (GONE) as a UploadVector
   consumer — confirm where the Gaussian/Poisson kernels live now (inlined in the `.cu` test or a surviving
   sensor/gpu TU) so BUF-10/11 covers the kernel side, not just the test.
6. **step_backward legacy buffer?** Confirm step_backward.hpp/step_backward_host.cpp hold no direct legacy buffer
   (not in the 65-includer grep — likely clean, route only ArticulationDeviceState views into the op-ified
   diffsim_backward.cu).
7. **ScopedDeviceGuard outside DeviceContext?** `git grep ScopedDeviceGuard` before BUF-14 deletes
   device_context.hpp — if used elsewhere it needs a v2 home (small device-set/restore RAII).
8. **MakeDefaultDeviceContext determinism.** Can rt/*.cu + sensor tests replace it with process-wide
   `InitBestDevice()+DeviceInitBackend`? For offscreen RT D1 the device choice must be deterministic — verify
   InitBestDevice picks the same device every run on this box.
9. **Python wheel linkage.** Confirm python/ does not transitively require the deleted nuka_phi legacy sources
   (should link c_abi → nk::World, not the legacy stepper). Grep found ZERO buffer_legacy in python/.
10. **contact_stream_driver double CMake membership.** It appears at src/CMakeLists.txt 1116 + 1187 (m9-recon T14
    flag) — confirm the duplicate-source was cleaned in M9 or persists; quick CMake audit before COL touches it.
11. **5-exe collapse end-state.** ~76 legacy per-file exes still coexist with the 5 aggregate exes. Is full
    collapse an M11 expectation or is the hybrid the accepted end-state? Affects where new gates ultimately live.
12. **lavapipe vkGetMemoryFdKHR.** Does the lavapipe ICD report VK_KHR_external_memory_fd UNSUPPORTED cleanly (so
    the fallback to HostDownloadPublisher triggers) or throw mid-construction? The fallback must catch BOTH
    missing-extension and runtime export failure for the build-link gate's fallback assertion to be meaningful.
13. **VkMemoryDedicatedAllocateInfo.** Does the present transform-SSBO VkDeviceMemory need dedicated allocation
    for export on real NVIDIA drivers? Cannot test on lavapipe; add defensively (harmless when not required).
14. **build-viewer reconfigure.** Does build-viewer (NK_BUILD_VULKAN_VALIDATION=ON) need reconfiguration to pick
    up cuda_vulkan_interop.cpp + rt_adapter + viewer delta, or are they added via existing globs? A reconfigure
    must NOT touch build-cuda128 (its CUDA-12.8 cache must never be reconfigured).
15. **NarrowphaseSdf op live wiring.** Does the M5 op get in-pipeline pair-stream wiring (SdfPairDev) in M11, or
    stay a standalone precision-oracle launcher (currently early-returns Status::Ok for every family)? Affects
    whether find_sdf_contact_newton's runtime relevance is permanently oracle-only (relevant to OD-13).
16. **m11-render-gates.md companion.** Recommend producing it (like m8-render-gates.md) recording the build-viewer
    recipe + §5 on-display checklist additions for interop + drag-entity, as the single owner reproduction doc.

---

## 9. RULINGS LOCKED (2026-06-14) — all 20 OD resolved; implementer obeys these

3 OWNER-CONFIRMED via AskUserQuestion (all = recommended option); 17 CONTROLLER-RESOLVED under existing
owner rulings (D1 discipline / highest directive / strict-dominance / never-ruled). Implement to these:

| OD | Ruling | Source |
|---|---|---|
| OD-1 | Interop CUDA placement = **(A)** zero-CUDA app `cuda_vulkan_interop.cpp` + scatter/import in NEW `src/phi/backend_cuda/interop/*.cu` | CONTROLLER (redline) |
| **OD-2** | Zero-copy render = **(A) FULL ZERO-COPY NOW** — opt-in, default-OFF device-local transform SSBO + instanced vertex shader + CUDA scatter via `cudaImportExternalMemory`/external-memory-fd. D1 offscreen gates stay byte-identical (default-off). | **OWNER ✓** |
| OD-3 | DLPack literalness = **(A)** real bridge = `VK_KHR_external_memory_fd` + `cudaImportExternalMemory`, expose a DLPack(kDLVulkan-shaped) descriptor for API consistency | CONTROLLER (confirmed by OD-2) |
| OD-4 | Recorder = **host-download** (the D1 mp4 gate stays device-independent) | CONTROLLER (D1) |
| OD-5 | MoveEntity persistence = **live-only** (Data writes; never persist to .nks) | CONTROLLER |
| OD-6 | Drag velocity = **zero velocity** on drag (place-here feel) | CONTROLLER |
| OD-7 | Drive panel = **flat per-DOF, optionally labeled** (GENERIC editor, no hardcoded choreography) | CONTROLLER (highest directive) |
| OD-8 | Camera evdev→xcb-keysyms = **defer** (documented caveat; blocks no gate) | CONTROLLER |
| **OD-9** | render_scene = **(A) DELETE BOTH** render_scene + RenderSceneDebugOverlayVulkan/RenderSceneVulkan wrappers, keep command-list overlay, retarget 2 tests | **OWNER ✓** |
| OD-10 | RT scene-desc POD home = **(A)** keep Tier-A PODs in `src/rt` (interface-types dir); only `.cu`/`.cuh` relocate | CONTROLLER |
| OD-11 | RT lib target = **(A)** dedicated lib `nuka_phi2_rt` with `--fmad=false` (unmissable; reject B per-source) | CONTROLLER (D1 fmad) |
| OD-12 | RtBackendI output = **(A)** caller-provided phi v2 `Buffer*` (6 AOV) + thin Framebuffer convenience for tests | CONTROLLER (brief) |
| OD-13 | `sdf_contact.cu` dead kernel = **KEEP whole file untouched** (SDF-IFT deferred "with rt"; dead kernel harmless) | CONTROLLER |
| OD-14 | M9 plan delete-list #13/#14 = **ledger wins** (KEEP narrowphase_dispatch/sdf_contact/adjoint/ift as oracle+diffsim infra) | CONTROLLER (ruling #4) |
| OD-15 | Salvaged SDF-tier gate home = **fold into `nuka_narrowphase_dispatch_test`** | CONTROLLER |
| **OD-16** | Buffer stream-binding = **(1) THREAD `Backend*`/`Device*`** through ~40 sites = pure type swap, correct stream, goldens survive | **OWNER ✓** |
| OD-17 | `ArticulationDeviceBuffers` hub = **(1) migrate-in-place** in src/runtime + "test-oracle harness, no production consumer" banner | CONTROLLER |
| OD-18 | `articulation_types.cuh` dup = **(1) collapse** to single definition (collision gone post-sweep; byte-identical PODs) | CONTROLLER |
| OD-19 | `sdf_device_world` = **(2) DELETE** (upload duty already in nk::Model; COL-5 coverage-fold guard first) — resolves buffer-vs-collision conflict toward DELETE | CONTROLLER |
| OD-20 | Atomicity = **(2) two-stage within M11** — BUF-13 buffers atomic, then BUF-14 DeviceContext/OwnedStream/StreamView fast-follow (no-op-D1 commit) | CONTROLLER |

**Execution mode (controller):** Phase 0 buffer sweep = tightly-coupled byte-level stream/kernel work →
**single-flight Opus max**, incremental commits per consumer-subtree (each leaves build green; NO TU ever
includes both `buffer.hpp` + `buffer_legacy.hpp`; `buffer_legacy.hpp` deleted only when last consumer migrates;
single BUF-12 D1 re-gate before BUF-13). Leaf phases (interop / viewer / rt_adapter) + the M11-end review =
**ultracode workflow**. Every commit `[skip ci]`, git-lfs on PATH, push deferred to batch end.
