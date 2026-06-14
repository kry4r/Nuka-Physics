# M11-end Adversarial Review — Synthesis

Increment: `dcd40c8..c660777` (11 commits, Phase 0–5)
Branch: master · Reviewed: 2026-06-14
Method: 6 dimensions × per-finding adversarial skeptics (2 each). A finding is CONFIRMED if ≥1 skeptic
returned `refuted=false`. Synthesis spot-verified BUF-F1 (candidate_pair.cu:110–122), VIEW-F2
(systems.cpp:103–105), and the four `GoldensMatchCudaAba` test names (featherstone_oracle_harness.cpp:413/460/560/621).

---

## 1. Verdict Summary (per dimension)

| Dimension | Findings | Confirmed | Refuted | Status |
|---|---|---|---|---|
| buffer-stream-d1 | 2 (BUF-F1, BUF-F2) | 2 | 0 | findings (both med, RAII leaks) |
| rt-homing | 2 (RT-F1, RT-F2) | 1 (RT-F2) | 1 (RT-F1) | finding (RT-F2 doc-only, low) |
| interop-r14-redline | 2 (INT-F1, INT-F2) | 2 | 0 | findings (both med, device-path) |
| viewer-directive | 4 (VIEW-F1..F4) | 3 (F1, F2, F3) | 1 (F4) | findings (3 med) |
| deletions-coverage | 1 (DEL-1) | 0 | 1 | CLEAN |
| directive-310-exitgate | 2 (EGH-F1, EGH-F2) | 1 (EGH-F2) | 1 (EGH-F1) | finding (EGH-F2 med, doc/gate-hygiene) |

CLEAN dimensions: **deletions-coverage** (DEL-1 refuted by both skeptics — the deleted kNoSdf→kOutsideBand
assertion was test-only scaffolding over a dead spike container; production narrowphase_sdf.cu has no
kNoSdf branch and the real "no contact" path stays device-tested).

Dimensions with confirmed findings: buffer-stream-d1, rt-homing, interop-r14-redline, viewer-directive,
directive-310-exitgate.

**No HIGH findings. No non-negotiable (D1 byte-exact / one-solver directive / zero-CUDA redline /
frozen-asset / lost-coverage / M10 boundary) is broken.** All confirmed findings are med/low:
error-path robustness, latent device-path correctness on the not-locally-runnable interop, viewer
UI reachability/correctness, and doc/gate-hygiene.

---

## 2. CONFIRMED Findings (ordered by severity)

All confirmed findings are **MED** except RT-F2 (LOW). No HIGH.

| id | sev | files | defect | impact | suggested fix |
|---|---|---|---|---|---|
| INT-F2 | med | phi/backend_cuda/interop/cuda_vk_scatter.cu; runtime/app/cuda_vulkan_interop.cpp; runtime/app/simulation.hpp | Interop scatter launches on a private `cudaStreamNonBlocking` stream (cuda_vk_scatter.cu:213) reading the live FK buffers the World wrote on `cb->main` (cuda_backend.cu:233); only sync is `cudaStreamSynchronize(stream_)` (:313) on the SCATTER stream — no event ordering it after the FK write. World::Backend() seam exists (world.hpp:112) for exactly this but is unused. | Cross-stream read-after-write race → torn/stale transforms on a real NVIDIA run; defeats R14 "bit-identical to host". Not locally runnable (lavapipe falls back to host path), no D1 golden. | Record a `cudaEvent` on the World stream (via World::Backend()) and `cudaStreamWaitEvent(stream_, event)` before the scatter launch, OR run scatter on `cb->main` directly. Must land before the owner's first NVIDIA-box run. |
| INT-F1 | med | render/raster/interop_transform_ssbo.{cpp,hpp}; render/raster/shaders/mesh_instanced.vert; render/raster/vulkan_present_renderer.cpp; runtime/app/cuda_vulkan_interop.cpp; runtime/app/viewer/viewer_main.cpp | Phase-4 producer (SSBO + scatter + compiled shader) shipped, but NO renderer creates the instanced pipeline or binds the exported SSBO descriptor. `InteropTransformSsbo::SetLayout/DescriptorSet/Buffer` (hpp:69–71) have zero consumers; present renderer still draws `inst.world_xform` (vulkan_present_renderer.cpp:1025, unmodified in M11); Publish leaves world_xform stale (cuda_vulkan_interop.cpp:102). | If interop activates, scatter writes mat4s no pipeline reads while present draws frozen bind poses → every instance freezes, masked by an "interop ACTIVE" log. Commit overstates "2nd instanced pipeline delivered". Default-OFF; D1/redline/frozen-asset intact. | (a) Wire the instanced pipeline into vulkan_present_renderer (load mesh_instanced.vert.spv, bind SSBO at set 1, draw by gl_InstanceIndex when publisher==interop), OR (b) downgrade the commit/shader/log claims to "scatter+SSBO+shader delivered; instanced draw deferred" and add a NAMED-DEBT banner in interop_transform_ssbo.hpp + viewer_main.cpp. |
| VIEW-F3 | med | runtime/app/viewer/viewer_main.cpp; runtime/app/systems.cpp; runtime/app/pose_publisher.cpp | Drag builds MoveEntity from `inst.world_xform` (= fk·cached_visual_local, pose_publisher.cpp:99) and ApplyMoveEntity writes it straight into FieldId::BodyPose (systems.cpp:68). When cached_visual_local ≠ identity (visual geom offset from body node — normal in MJCF/USD), the BodyPose gets the visual rotation/offset, and next frame the publisher re-multiplies cvl → double-applied offset + corrupted orientation. | Body-pose corruption + drift for any movable body with an offset visual node. Benign for the cup (identity cvl) so owner-verified case + the move_entity gate (identity instance) both pass. | Write `cmd.world_xform * Inverse(cached_visual_local)` into BodyPose (or pass the physics-frame transform from the drag site). Add an offset-visual-node case to viewer_move_entity.cpp. |
| VIEW-F1 | med | render/render_world.cpp; runtime/app/viewer/viewer_main.cpp; runtime/app/systems.cpp | Advertised floating-base teleport is UI-unreachable. ResolvePoseSource never emits Kind::Base (render_world.cpp:108–137; `git grep` zero `kind = ...Base`); floating root resolves to Kind::Link; PickInstance only accepts Body||Base (viewer_main.cpp:187–189) so the Link root is unpickable. Both the Kind::Base case and the Link→Base reroute (systems.cpp:77, 97–122) are dead via UI. | Phase-5 commit "floating-base teleports implemented" is an over-claim; only free rigid bodies (cup) drag. No D1/directive/redline/frozen-asset/coverage break (general command path honored; dead branches never had coverage). | (a) Emit Kind::Base for the floating root (joint_type[root]==FloatingBase) so the picker+Base branch fire, OR (b) accept root-link instances in PickInstance, OR (c) downgrade the docstring/commit to "Body-only; floating-base reroute present but not yet UI-reachable". |
| VIEW-F2 | med | runtime/app/systems.cpp | Floating detection uses `dof_count >= 6` heuristic (systems.cpp:103–105) instead of `joint_type[root]==FloatingBase`. A fixed-base 6+-DOF arm satisfies all predicates → root drag would falsely route to UploadField(BasePose) + zero root LinkVelocity. Correct discriminator FloatingBase==3 (articulation_state.hpp:32) is ignored. | Latent Data-write-correctness bug for programmatic MoveEntity on a Link entity. (One skeptic refuted as physically-inert/UI-unreachable; CONFIRMED because the other skeptic verified the heuristic is genuinely wrong and the move_entity gate never exercises it.) Currently masked by VIEW-F1. | Replace with `floating = root < art.joint_type.size() && art.joint_type[root] == static_cast<uint8_t>(ArticulationJointType::FloatingBase);`. Same predicate gates the Base branch if VIEW-F1(a) is taken. |
| BUF-F1 | med | collision/candidate_pair.cu | `d_keys`/`d_pairs` are bare `phi::Buffer*` from UploadVectorV2 (candidate_pair.cu:110–111); `BufferFree(d_keys)` is at :121, only reached after the `CheckCuda` throw at :118; `d_pairs` ownership transfers to the result only on success (:122). On a `stable_sort_by_key` launch-failure throw, BOTH leak. Legacy used stack-RAII value `phi::Buffer` that auto-freed on unwind. Lone holdout — every sibling collision file adopted the OwnedBuffer RAII helper. | Device-memory leak on the GPU error path; exception-safety regression introduced by the buffer sweep. Happy path byte-identical → no D1/golden impact. | Wrap in the file-local OwnedBuffer RAII (`Adopt(UploadVectorV2(...))`), `Release()` d_pairs into the CandidatePairStream result on success; wrapper frees both on any throw. |
| BUF-F2 | med | collision/particle_candidate_pairs.cu | Both Build* fns allocate bare `phi::Buffer*` (d_body_ids :200, d_out :203/:269) freed only on the happy path (:218–219/:283). Two real pre-free throw points: `CheckCuda` launch (:213/:278) and `std::vector survivors(total)` bad_alloc (:216/:281) → bare GPU handles leak. Legacy used stack-RAII; siblings in the same sweep adopted OwnedBuffer. (Finding overstated 2 of 4 paths — BufferDownload swallows errors, BuildAndDedup runs after free — but the core leak is confirmed.) | Device-memory leak on GPU error / OOM path; exception-safety regression. Happy-path bytes + goldens unaffected. | Wrap d_body_ids and d_out in OwnedBuffer/Adopt so they free on every exit incl. the CheckCuda throw and the survivors bad_alloc path. |
| EGH-F2 | med | tests/oracle/featherstone_oracle_harness.cpp; subagent-plans/m11-recon.md | Buffer-sweep D1 re-gate used `ctest -L full -E GoldensMatchCudaAba`. The regex matches FOUR tests (:413/460/560/621) but only #206 RandomSampleGoldensMatchCudaAba is the documented carry-forward known-fail. It also silently excludes 3 GREEN byte-exact nk-vs-legacy gates (#274/275/276) — the ones most sensitive to the async-stream sweep. (One skeptic refuted as cosmetic since the running #277 NkWorldGo2Stand5s/#278 NkWorldBatchedContact byte-exact cover the same path; CONFIRMED because the other skeptic verified the over-match + gate-report dishonesty are real.) | Headline "NNN/NNN green" structurally omits 3 green gates and reads as hiding 4 ABA fails. D1 is NOT actually unverified (#277/#278 cover the same nk articulation+buffer path), so no unique coverage lost → med not high. | Anchor the exclusion to the exact known-fail: `-E 'FeatherstoneOracle\.RandomSampleGoldensMatchCudaAba$'`. At minimum, the exit-gate writeup must state the 3 byte-exact nk gates were run separately and passed. |
| RT-F2 | low | render/rt_backend.hpp; phi/backend_cuda/rt/rt_backend_cuda.cpp | OD-12 docstring says Trace writes the 6 AOVs into caller device buffers "(no host copy)" (rt_backend.hpp:18–19, 94–97), but the only impl does rt::RenderFrame (device→host download) then UploadAov (host→device) — a full device→host→device roundtrip (rt_backend_cuda.cpp:76–83). Impl comment honestly admits it. | Doc/contract overstatement only. RT-5 gate asserts device==host byte-equality and passes (same bytes uploaded back); no golden/D1/redline/frozen-asset impact. | Correct the OD-12 wording: state the CUDA backend currently produces AOVs via the host RenderFrame path and uploads them (host roundtrip), with true device-resident zero-copy Trace flagged as named debt. Or write kernel AOVs directly into caller device buffers. |

---

## 3. REFUTED / False-Alarm List

- **DEL-1** (low, deletions-coverage) — REFUTED by both skeptics. The deleted `sdf_index==kNoSdf → kOutsideBand`
  assertion lived inside the TEST's own SampleSdfKernel, not in any shipping kernel (deleted sdf_device_world.cu
  had no device sampler at all). Production narrowphase_sdf.cu has no kNoSdf branch (kNoSdf bodies are filtered
  out upstream at cook time), and the real out-of-band "no contact" path stays device-tested by
  SdfPrecisionOracle.SeparatedSampleEmitsNoContact. No production coverage lost.
- **RT-F1** (med, rt-homing) — REFUTED by both skeptics. Foundational claim is factually false: the RT TUs
  allocate via `DeviceBufferType(InitBestDevice())` whose backend is nullptr, so BufferUpload/Download resolve
  `CudaBackendMainStream(nullptr)` = stream 0 — the SAME NULL/default stream as the RT kernels. No cross-stream
  race; the OwnedBuffer "stream-0 / byte+ordering identical to legacy" comment is ACCURATE. Finding conflated
  DeviceBufferType(device)[stream 0] with BackendDeviceBufferType(backend)[cb->main]; RT never uses the latter.
- **VIEW-F4** (low, viewer-directive) — Split (one confirm / one refute). One skeptic confirmed the readout is
  always 0; the other refuted it as PRE-EXISTING (field, "lit pixels" row, and missing assignment all exist at
  baseline dcd40c8 — the Phase-5 diff never touches non_bg_pixels) and purely cosmetic. Treated as NOT-an-M11-defect:
  it is a pre-existing cosmetic blemish, not introduced by this increment; excluded from confirmed actions.
- **EGH-F1** (med, directive-310-exitgate) — REFUTED by both skeptics. Core premise false: pre-M11, the
  test_caller_stream.py path (World.create_from_scene→step_n→buffer_view) NEVER ran on desc->cuda_stream — only
  the legacy diffsim/noise paths did, and the test never calls those. nk::World already ran on the backend's own
  cb->main stream before AND after M11; the test path is byte-unchanged by M11 and its terminal d2h imposes
  ordering. The nuka_ext.cpp docstring WAS updated in M11, and the test isn't wired into any ctest/CI gate. Only
  residue: nuka.h:38 `cuda_stream` not doc-flagged as ignored — a pre-existing cosmetic doc gap, not an M11 break.

---

## 4. Recommended Fix-Forward Actions (controller)

Priority order (device-correctness before the owner's NVIDIA-box run; then robustness; then doc/gate hygiene):

1. **INT-F2 (do before any real NVIDIA-box run):** In `CudaVkScatter::Scatter`, order the scatter after the
   physics FK write — record a `cudaEvent` on the World's `cb->main` stream (use the existing `World::Backend()`
   seam) and `cudaStreamWaitEvent(stream_, event)` before launching ScatterTransformsKernel; or run the scatter
   on `cb->main`. Add an owner-verify note (not locally runnable).
2. **INT-F1:** Close the producer→consumer loop OR be honest. Preferred: wire the instanced pipeline into
   `vulkan_present_renderer` (load `mesh_instanced.vert.spv`, bind `interop_ssbo.DescriptorSet()` at set 1, draw
   by `gl_InstanceIndex` when the interop publisher is active). If deferring, downgrade the Phase-4 commit message,
   the shader header, and the viewer "interop ACTIVE" log to "scatter+SSBO+shader delivered; instanced draw
   deferred" and add NAMED-DEBT banners in `interop_transform_ssbo.hpp` and `viewer_main.cpp`.
3. **VIEW-F3:** In the drag write path, write `cmd.world_xform * Inverse(cached_visual_local)` into
   `FieldId::BodyPose` (or pass the physics-frame transform from the drag site so ApplyMoveEntity does not need
   the cvl lookup). Add an offset-visual-node case to `viewer_move_entity.cpp`.
4. **VIEW-F2:** Replace the `dof_count >= 6` heuristic in `systems.cpp:103–105` with
   `root < art.joint_type.size() && art.joint_type[root] == static_cast<uint8_t>(ArticulationJointType::FloatingBase)`.
5. **VIEW-F1:** Make the floating-base teleport reachable — emit `Kind::Base` for the floating root in
   `ResolvePoseSource` (then the picker + Base branch fire), OR accept root-link instances in `PickInstance`.
   If neither is taken now, downgrade the Phase-5 "floating-base teleports implemented" claim to
   "Body-only; floating-base reroute present but not yet UI-reachable" and flag as named debt.
   (VIEW-F1 + VIEW-F2 are best fixed together — emit Kind::Base via the FloatingBase joint-type check.)
6. **BUF-F1:** Wrap `d_keys`/`d_pairs` in `candidate_pair.cu` in the file-local OwnedBuffer RAII
   (`Adopt(UploadVectorV2(...))`), `Release()` d_pairs into the result on success — matching the sibling
   collision files; wrapper frees both on any throw.
7. **BUF-F2:** Wrap `d_body_ids` and `d_out` in `particle_candidate_pairs.cu` in OwnedBuffer/Adopt so they free
   on the CheckCuda-launch throw and the `survivors(total)` bad_alloc path.
8. **EGH-F2:** Narrow the D1 re-gate exclusion to `-E 'FeatherstoneOracle\.RandomSampleGoldensMatchCudaAba$'`
   so the 3 green byte-exact nk gates are counted; or explicitly document in the exit-gate writeup that they were
   run separately and passed.
9. **RT-F2:** Fix the OD-12 docstring in `rt_backend.hpp` to state the CUDA backend currently produces AOVs via
   the host RenderFrame path and uploads them (host roundtrip), with device-resident zero-copy Trace as named debt.

---

## 5. Overall VERDICT

**SOUND_WITH_FIXES.**

Zero HIGH findings — no non-negotiable is broken. D1 byte-exactness holds (the RT homing is genuinely stream-0
same-stream, the buffer sweep happy paths are byte-identical, and the masked-gate count is covered by the running
#277/#278 byte-exact nk gates). The one-solver directive, zero-CUDA redline, frozen assets, and the M10 boundary
are all respected. 9 confirmed med/low findings: two RAII device-leak regressions on GPU error paths, the
incomplete interop consumer + its cross-stream race (the two highest-value items), three viewer correctness/
reachability bugs, one over-broad gate-exclusion, and one RT doc overstatement.

**Single biggest risk:** the Phase-4 CUDA↔Vulkan interop path (INT-F1 + INT-F2). It is the only increment that is
not locally runnable, gated solely on build-link + lavapipe fallback, yet shipped with commit/shader/log language
asserting a working zero-copy render. As written, the first real NVIDIA-box activation would render frozen bind
poses (no SSBO consumer) over torn/stale transforms (uncoordinated non-blocking scatter stream). Both must be
fixed — or the claims loudly downgraded to named debt — before the owner's hardware verification, so the first
real run is correct rather than silently wrong behind an "interop ACTIVE" banner.
