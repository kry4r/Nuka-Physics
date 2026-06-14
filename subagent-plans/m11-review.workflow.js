export const meta = {
  name: 'm11-end-review',
  description: 'M11-end ultracode adversarial review: 6 facet finders -> per-finding skeptic refute -> synthesis writes m11-review.md',
  phases: [
    { title: 'Review', detail: '6 adversarial facet finders over the M11 increment' },
    { title: 'Verify', detail: 'per-finding skeptics try to REFUTE each claim' },
    { title: 'Synthesize', detail: 'keep confirmed findings, write m11-review.md, return digest' },
  ],
}

const RANGE = 'dcd40c8..c660777' // the full M11 increment (Phase 0-5)

const SHARED = `
You are an ADVERSARIAL reviewer of milestone M11 of the nk-core platform refactor of Nuka-Physics
(repo /root/Nuka-Physics, branch master). The M11 increment is git range ${RANGE} (11 commits, Phase 0-5).
Inspect it with "git log/show/diff ${RANGE} -- <path>", read the working-tree files, and run "git grep".
You MAY build/test to verify a suspicion, but PREFER static inspection (a heavy ctest may contend the GPU;
if you build, use build-cuda128 / build-viewer, NEVER reconfigure cmake). Your job is to FIND REAL PROBLEMS,
not to praise. Default to skepticism.

== WHAT M11 DID (the 6 buckets) ==
- Phase 0 (6 commits 63ae3a5..67de9a2): buffer_legacy -> phi-v2 atomic sweep + delete DeviceContext/
  OwnedStream/StreamView. Design: legacy CopyFromHost = sync memcpy on default stream 0; v2 BufferUpload =
  async on the buffer's backend stream. Claimed D1-safe because every migrated consumer's transfers + kernels
  stay on ONE stream (stream-0 DeviceBufferType for the legacy/oracle path = byte+ordering identical; the v2
  Buffer is allocation+base-pointer only, transfers/kernels unchanged). c_abi DeviceRecord owned_stream -> stream 0.
- Phase 1 (4b8d234): re-asserted the SDF-tier forward gate (salvaged 5 SdfTierWired tests from fcae2a6 into
  test_narrowphase_dispatch.cpp) + collision/SDF audits.
- Phase 2 (8a5a200): homed the self-written RT path tracer src/rt -> src/phi/backend_cuda/rt (line-faithful
  git mv, include-path-only) behind RtBackendI (src/render, zero-CUDA) + rt_adapter; dedicated nuka_phi2_rt lib
  with --fmad=false; RT-5 offscreen D1 gate. RT output via phi-v2 BufferI (6 AOV).
- Phase 3 (c985e5c): deleted render_scene.{hpp,cpp} + the dead RenderScene overlay wrappers; kept the live
  command-list overlay; retargeted tests. (rt::RenderScene is a DIFFERENT, untouched thing.)
- Phase 4 (92b68be): CUDA<->Vulkan zero-copy interop publisher (CudaVulkanInteropPublisher : PosePublisher),
  opt-in default-OFF. InteropScatterI (CUDA-free seam) + cuda_vk_scatter.cu (scatter + cudaImportExternalMemory
  + external semaphore, in backend_cuda/interop) + exportable transform SSBO + mesh_instanced.vert. NOT
  locally runnable (CUDA-GPU vs lavapipe-CPU); gate = build-link + graceful fallback to HostDownloadPublisher.
- Phase 5 (c660777): ImGui viewer delta -- entity pick + drag -> MoveEntity -> live nk::Data BodyPose/BasePose
  write (never .nks) + zero velocity; generic per-DOF drive-slider panel -> FieldId::DriveTarget; swapchain-
  recreate ImGui rebuild; StepOnce same-frame. Extended GATE-B + new host tests.

== THE NON-NEGOTIABLES (a violation of any is a HIGH finding) ==
1. D1 BYTE-EXACT: tests/oracle/golden/** never regenerated; kernel/transform ports LINE-FAITHFUL. The buffer
   sweep's async-stream change is the #1 D1 risk -- hunt for a consumer whose transfers and kernels ended up on
   DIFFERENT streams, a dropped sync, a host-read of a v2 buffer without BufferDownload's sync, or a golden-
   producing kernel that silently moved streams. The RT relocation must be include-path-ONLY (no math/order/FMA
   change); --fmad=false must be ACTIVE on the homed lib. The interop scatter transform (R14) must be bit-faithful.
2. HIGHEST DIRECTIVE: ONE general physics solver, NO per-case special-casing, NO dedicated special world type.
   Hunt for any demo-keyed branch (h1/cup/grasp/union/go2/kitchen) in a solver/contact/publisher/viewer path,
   any MoveEntity/drive write that bypasses the general command_queue->systems->Data path, any interop publisher
   that switches on world type.
3. ZERO-CUDA REDLINE: src/nk, src/scene, src/render (interface layer), src/runtime/app must have ZERO CUDA token
   (#include phi/backend_cuda, <cuda_runtime>, <<<>>>, cuda*()). Hunt for a leak introduced by RtBackendI/
   rt_adapter/interop/viewer.
4. FROZEN ASSETS: examples/scenes/h1_cup_table.nks/.nka unmodified; viewer edits are Data-only (R13).
5. COVERAGE: a deleted test's unique coverage must be preserved (sdf_device_world->COL-5 fold; render_scene
   tests->retarget; the buffer/stream test reworks). Hunt for SILENTLY LOST coverage.
6. M10 BOUNDARY: M11 must not touch RL / go2 PD-stance / per-link wrench-control / autoreset / control-mode / SG-spec.

== OUTPUT ==
Return the schema object. Each finding must be CONCRETE (file:line + the specific defect + why it breaks a
non-negotiable) and carry a suggested fix. Do NOT invent findings to seem thorough -- an empty findings list is
a valid, honest result if the increment is clean in your dimension. Severity: high = breaks a non-negotiable
(D1/directive/redline/frozen-asset/lost-coverage); med = correctness/robustness bug not caught by a gate; low =
doc/comment/cosmetic/named-debt.
`

const FINDING_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['dimension', 'summary', 'findings'],
  properties: {
    dimension: { type: 'string' },
    summary: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'title', 'severity', 'files', 'claim', 'evidence', 'suggested_fix'],
        properties: {
          id: { type: 'string', description: 'e.g. BUF-F1' },
          title: { type: 'string' },
          severity: { type: 'string', enum: ['high', 'med', 'low'] },
          files: { type: 'array', items: { type: 'string' } },
          claim: { type: 'string', description: 'the specific defect' },
          evidence: { type: 'string', description: 'file:line + what proves it' },
          suggested_fix: { type: 'string' },
        },
      },
    },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['finding_id', 'refuted', 'reasoning', 'impact_if_real'],
  properties: {
    finding_id: { type: 'string' },
    refuted: { type: 'boolean', description: 'true = NOT a real problem (false alarm); false = CONFIRMED real' },
    reasoning: { type: 'string', description: 'the independent check you ran/read to refute or confirm' },
    impact_if_real: { type: 'string', enum: ['breaks-D1', 'breaks-directive', 'breaks-redline', 'frozen-asset', 'lost-coverage', 'correctness', 'cosmetic', 'none'] },
  },
}

const DIMENSIONS = [
  { key: 'buffer-stream-d1', label: 'review:buffer', prompt: `DIMENSION 1 -- BUFFER/STREAM SWEEP D1 FIDELITY (Phase 0, the #1 risk). Inspect every migrated consumer (articulation/diffsim/collision/sensor/rt/invariants/c_abi). Hunt for: (a) a consumer whose v2 buffer transfers and its kernels ended up on DIFFERENT streams (stream-0 buffer vs a kernel still on some other stream) -> race / nondeterminism; (b) a dropped/missing sync where legacy had a sync barrier; (c) a host-side read of a v2 Buffer that does NOT go through BufferDownload (which syncs) -> stale read; (d) the c_abi owned_stream->stream-0 rebind changing happens-before with the backend->main backward op; (e) any RAII leak (Buffer* not freed, double-free, grow-realloc not freeing old); (f) a golden-producing kernel that silently changed stream. Read git diff ${RANGE} for src/runtime/articulation/articulation_state.*, src/diffsim/*, src/c_abi/diffsim.cpp, src/collision/*, src/phi/backend.hpp + buffer_transfer_v2.hpp.\n\n${SHARED}` },
  { key: 'rt-homing', label: 'review:rt', prompt: `DIMENSION 2 -- RT HOMING FIDELITY (Phase 2). Verify the 9 git-mv'd files (src/phi/backend_cuda/rt/*) differ from their src/rt originals by INCLUDE PATHS ONLY -- use git show ${RANGE} / git diff -M to confirm no math/order/cast/FMA-introducing change; check prim_id.cuh bit-packing is bit-identical. Confirm --fmad=false is ACTIVE on nuka_phi2_rt (grep build flags) and it is NOT folded into nuka_phi2. Confirm src/render/rt_backend.hpp + rt_adapter.{hpp,cpp} have ZERO CUDA tokens. Check the RtBackendI BufferI-output contract (OD-12) and the no-CUDA fallback (factory returns null cleanly). Hunt for a behavioral change in the homed kernels or a redline leak.\n\n${SHARED}` },
  { key: 'interop-r14-redline', label: 'review:interop', prompt: `DIMENSION 3 -- INTEROP REDLINE + R14 BYTE-EXACT (Phase 4). Confirm src/runtime/app/cuda_vulkan_interop.{hpp,cpp} + src/phi/interop_scatter.hpp + src/nk/pipeline/world.hpp + src/render/raster/interop_transform_ssbo.* have ZERO CUDA token (the redline) -- all CUDA must be in src/phi/backend_cuda/interop only. R14: read cuda_vk_scatter.cu's transform math and compare it op-for-op against HostDownloadPublisher::Publish (pose_publisher.cpp) -- is world_xform = fk * cached_visual_local reproduced bit-faithfully (same quat rotate/mul/normalize sequence, --fmad=false, no atomics, same env-major offset)? Flag any divergence. Confirm default-OFF really leaves the offscreen render path's pixels unchanged (the instanced pipeline is a separate opt-in). Check the fallback is robust (both missing-extension AND runtime-export-failure -> HostDownloadPublisher). Confirm the recorder stays host-download (OD-4).\n\n${SHARED}` },
  { key: 'viewer-directive', label: 'review:viewer', prompt: `DIMENSION 4 -- VIEWER CORRECTNESS + HIGHEST DIRECTIVE (Phase 5). Confirm MoveEntity (systems.cpp ApplyMoveEntity) writes through the GENERAL path (command_queue -> systems -> nk::Data::UploadField), never a demo-specific shim, never nuka_scene_set_local/Save (R13 -- frozen .nks). Confirm the drive-slider panel is a GENERIC per-DOF editor (no hardcoded grasp/choreography). Verify the entity->Data-row resolution (pose_source.{kind,row}) is honest -- the interior-articulation-link no-op claim: is it correct that interior links are FK outputs and only Body/floating-base teleports are valid? Check GATE-B (viewer_frame_smoke) records from a FIXED ViewerUiState with NO time/animation-seeded widget (R12 two-run byte-identical). Check the velocity-zeroing (OD-6) is applied to the right field. Hunt for a directive violation or a Data-write that could corrupt state.\n\n${SHARED}` },
  { key: 'deletions-coverage', label: 'review:deletions', prompt: `DIMENSION 5 -- DELETIONS SAFETY + COVERAGE (Phase 0/1/3). For each deletion verify no live consumer was orphaned AND no unique test coverage was silently lost: (a) render_scene.{hpp,cpp} + the RenderScene overlay wrappers (Phase 3) -- confirm zero live (non-test) consumer + the 2 tests' coverage retargeted or honestly dropped; (b) sdf_device_world (Phase 0/COL-5) -- confirm its upload-roundtrip + cell-count coverage was genuinely folded into test_gjk_epa_convex (SdfPrecisionOracle.CookedTableDeviceRoundTripAndCellCount), not just deleted; (c) the buffer_legacy/DeviceContext deletes -- confirm the reworked/deleted phi type-tests (test_capabilities, test_device_context_per_handle, test_stream_view_external) didn't drop coverage that isn't elsewhere; (d) BUF-2 dead BufferRecord/BufferTable -- confirm truly zero callers. Read git diff ${RANGE} for the deleted files + their test reworks.\n\n${SHARED}` },
  { key: 'directive-310-exitgate', label: 'review:governance', prompt: `DIMENSION 6 -- HIGHEST-DIRECTIVE SWEEP + §3.10 + EXIT-GATE HONESTY + M10 BOUNDARY. (a) git grep the WHOLE M11 increment for demo-keyed special-casing (h1|cup|grasp|union|go2|kitchen) in any non-test, non-comment solver/contact/publisher/render path -> any hit is a directive violation. (b) §3.10: confirm the zero-CUDA redline (src/nk, src/scene, src/render, src/runtime/app) is intact after M11 (RtBackendI/rt_adapter/interop/viewer added no CUDA token); confirm no new bare-.cu was left in the engine layer. (c) EXIT-GATE HONESTY: is the milestone genuinely green or is something masked? Scrutinize the "4 slow ABA goldens excluded via -E GoldensMatchCudaAba" -- is that a legitimate pre-existing exclusion or does it hide an M11 regression? Are any gates SKIP-stubbed that should run? (d) M10 BOUNDARY: confirm M11 touched NO RL/PD-stance/wrench-control/autoreset/control-mode/SG-spec. (e) carry-forward ledger: confirm the M9->M11 contracts (SDF-tier gate, collision/sdf kept as oracle, buffer sweep, rt/render_scene/interop) were all actually delivered.\n\n${SHARED}` },
]

phase('Review')
log(`M11-end review: ${DIMENSIONS.length} adversarial finders over ${RANGE}`)

const results = await pipeline(
  DIMENSIONS,
  (d) => agent(d.prompt, { label: d.label, phase: 'Review', schema: FINDING_SCHEMA, model: 'opus' }),
  (review, d) => {
    if (!review || !review.findings || review.findings.length === 0) {
      return { dimension: d.key, findings: [], verified: [] }
    }
    // Verify phase: 2 independent skeptics per finding, each tries to REFUTE.
    return parallel(
      review.findings.flatMap((f) => [0, 1].map((k) => () =>
        agent(
          `You are skeptic #${k + 1} for an M11 review finding. Try to REFUTE it -- prove it is NOT a real problem. ` +
          `Independently inspect the code/diff/gates. Default to refuted=true ONLY if you genuinely cannot confirm the defect; ` +
          `set refuted=false ONLY if you CONFIRM it is a real violation. Finding:\n` +
          JSON.stringify(f, null, 1) + `\n\n${SHARED}`,
          { label: `verify:${f.id}#${k + 1}`, phase: 'Verify', schema: VERDICT_SCHEMA, model: 'opus' }
        )
      ))
    ).then((verdicts) => ({ dimension: d.key, findings: review.findings, verified: verdicts.filter(Boolean) }))
  }
)

phase('Synthesize')
const flat = results.filter(Boolean)
const synthPrompt = `You are the SYNTHESIS agent for the M11-end adversarial review. Below are 6 dimensions, each with
its findings + per-finding skeptic verdicts (2 skeptics each; refuted=true means false-alarm). A finding is
CONFIRMED if at least one skeptic set refuted=false (real). Reconcile, dedupe, and WRITE
/root/Nuka-Physics/subagent-plans/m11-review.md with: (1) verdict summary (clean dimensions vs findings),
(2) CONFIRMED findings table (id, severity, files, defect, impact, suggested fix) ordered by severity,
(3) REFUTED/false-alarm list (one line each, why dismissed), (4) recommended fix-forward actions for the
controller (each confirmed finding -> a concrete fix), (5) overall VERDICT (SOUND / SOUND_WITH_FIXES /
NEEDS_WORK) + the single biggest risk. Be honest -- do not inflate or hallucinate; if all dimensions are clean,
say SOUND with zero confirmed findings.

${SHARED}

The 6 dimensions with findings + verdicts:
${JSON.stringify(flat, null, 1)}

After writing the file, RETURN a concise digest: total findings, confirmed count by severity, the VERDICT, the
one-line list of CONFIRMED findings, and the single biggest risk. Your returned text is the controller's digest.`

const digest = await agent(synthPrompt, { label: 'synthesis', phase: 'Synthesize', model: 'opus' })
log('M11-end review synthesis complete -> subagent-plans/m11-review.md')
return { dimensions: flat.length, digest }
