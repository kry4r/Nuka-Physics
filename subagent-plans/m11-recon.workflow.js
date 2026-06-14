export const meta = {
  name: 'm11-recon',
  description: 'M11 ultracode recon: 6 facet investigators -> synthesis writes subagent-plans/m11-recon.md (the M11 construction plan)',
  phases: [
    { title: 'Recon', detail: '6 parallel facet investigators read the codebase + plan; emit structured findings' },
    { title: 'Synthesize', detail: 'one agent reconciles all 6 facets, writes m11-recon.md, surfaces owner decisions' },
  ],
}

// ---------------------------------------------------------------------------
// SHARED CONTEXT injected into every facet prompt. Recon is READ-ONLY: no code
// edits, no commits, no builds that mutate state. Each facet produces an
// EXECUTABLE construction-plan slice for M11.
// ---------------------------------------------------------------------------
const SHARED = `
You are a RECON investigator for milestone M11 of the nk-core platform refactor of the Nuka-Physics
engine (repo root /root/Nuka-Physics, branch master, HEAD dcd40c8 = M9 COMPLETE+REVIEWED+PUSHED).
This is READ-ONLY investigation. DO NOT edit code, DO NOT commit, DO NOT run builds. You MAY read
any file, run "git grep" / "git log" / "git show", and inspect commit history. Your output is a
structured slice of the M11 construction plan that an implementer will execute next.

==== BINDING DOCUMENTS (read the relevant parts) ====
- Plan: docs/plans/2026-06-11-nk-core-platform-refactor.md
  * M9 section lines 495-501 (its DELETE list line 499 DEFERRED render_scene.{hpp,cpp} + buffer_legacy.hpp
    + legacy phi::{OwnedStream,DeviceContext,UploadVector...} to M11 -- they were NOT deleted in M9).
  * M11 section lines 507-509: viewer (scene tree / playback / env select / camera / drive sliders /
    DRAG ENTITY via MoveEntity command -> write Data) + cuda_vulkan_interop.cpp (CudaVulkanInteropPublisher:
    register Vulkan transform buffer as CUDA external memory, device scatter kernel via launch.cuh) +
    src/render/rt_adapter.{hpp,cpp} (RenderWorldToTwoLevelScene, BLAS once / TLAS per frame, self-written
    NO OptiX) + src/render/rt_backend.hpp (RtBackendI: BLAS/TLAS build, trace dispatch, output via BufferI),
    CUDA impl homes to src/phi/backend_cuda/rt/, render interface layer ZERO CUDA token.
    GATE: display-machine interactive run; RT still D1; offscreen path still CI gate.
  * §3.10 CUDA-homing master table lines 358-372: every CUDA-coupled subsystem must be extracted to a
    backend-agnostic interface with impl in src/phi/backend_cuda; src/nk, src/scene, src/render (interface
    layer), src/runtime/app keep a ZERO-CUDA-token lint redline. M11 row = RT backend interface (RtBackendI)
    + CUDA<->Vulkan interop (device scatter kernel via launch.cuh).

==== HIGHEST DIRECTIVE (overrides everything) ====
"通用的物理求解引擎，禁止物理出现 case 特惠，禁止出现专门的 uniworld 等东西，现在都是一个物理世界求解."
= ONE general/universal physics solver. FORBIDDEN: per-case/per-demo physics special-casing; any dedicated
special world type. Behavior = imported scene data + per-scene control script, NOT world type. Any M11 work
that would (re)introduce a special-cased path is a VIOLATION -- flag it.

==== D1 BYTE-EXACT DISCIPLINE (non-negotiable) ====
- Goldens under tests/oracle/golden/** are OWNER-PROTECTED and NEVER regenerated. A red golden = a real
  regression, never "rebaseline".
- examples/scenes/h1_cup_table.nks/.nka are physics-gate-frozen -- do not modify.
- Any kernel relocation/port must be LINE-FAITHFUL (bit-identical), proven by existing adjoint-FD /
  union_cook_golden / determinism gates. The async-stream change in the buffer sweep is the single biggest
  D1 risk in M11 (see buffer facet).

==== BUILD / VERIFY ENV (for your task estimates, not to run now) ====
- build dir: build-cuda128 (CUDA 12.8 /opt/cuda-12.8-root, g++-10, Makefiles). NEVER reconfigure cmake
  (wrong CUDA 11.3 on default path). Gates run from REPO ROOT with:
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64
- lint: /root/miniconda3/bin/python tools/lint/physics_smell.py  (zero-CUDA-token redlines live here)
- The Vulkan render flag NK_BUILD_VULKAN_VALIDATION is OFF in build-cuda128 and is NEVER reconfigured.
  M8/M8.5 established the "recompile the gated TU + link the already-built archive" verification paradigm
  for render code that can't run on this box.

==== LOCAL NON-VERIFIABILITY (critical for gate design) ====
This is a headless server: compute GPU = real CUDA device; the only Vulkan ICD = llvmpipe/lavapipe (CPU).
Therefore:
- The on-screen interactive viewer and the CUDA<->Vulkan zero-copy interop CANNOT be run/verified locally
  (CUDA-GPU memory cannot be shared with a CPU Vulkan device). Their gate = build + link + offscreen
  draw-data smoke; real interaction/interop is owner-verified on a display machine.
- The RT path tracer's OFFSCREEN path CAN and MUST be a local D1/CI gate (it's pure CUDA compute -> image
  buffer). RT "still D1" is a hard requirement.

==== WHAT M8/M8.5 ALREADY BUILT (so M11 viewer/render is a DELTA, not greenfield) ====
- src/runtime/app/{command_queue.hpp, pose_publisher.{hpp,cpp}, systems.{hpp,cpp}, simulation.{hpp,cpp}}
  (M8 frame loop; PosePublisher abstraction + HostDownloadPublisher; the interop publisher is the planned swap).
- src/runtime/app/viewer/{viewer_main.cpp, imgui_layer.{hpp,cpp}, camera_controller.{hpp,cpp}} (M8.5 docked viewer).
- src/render/imgui (vendored ImGui v1.92.8-docking + nuka_imgui + ApplyNukaTheme custom teal #2DD4BF theme),
  src/render/window (XcbWindowSurface / VK_KHR_xcb_surface present), vulkan present renderer + swapchain.
- src/render/{render_world.{hpp,cpp}, vulkan_renderer.{hpp,cpp}, raster/, shaders/} = M8 offscreen raster +
  M8.5 full Cook-Torrance GGX PBR + lighting UBO + ACES tonemap (offscreen + present share the renderer).
- M8/M8.5 named debt carried to M11 (verify each): cook-real-normals parse-vn (D1-SENSITIVE .nka byte change),
  present-resize relies only on OUT_OF_DATE, vk-imgui-stale-renderpass-after-recreate, base-posepath-dead
  (the M11 interop publisher is what first lights the dead PoseSource::Base path), render_finished per-image
  semaphore variant, camera evdev->formal xcb-keysyms, vk-raster-render-exception-leak RAII.

==== M9 CARRY-FORWARD LEDGER -> M11 (owner-confirmed during M9) ====
(a) SDF-TIER FORWARD GATE RE-ASSERT: M9 deleted tests/collision/test_sdf_tier_wired.cpp but KEPT the SDF
    forward infrastructure (narrowphase_dispatch + find_sdf_contact_newton). M11 MUST re-assert the gate
    BoxOnBoxMatchesAnalyticalNormalAndPenetration -- salvage it VERBATIM from commit
    fcae2a6:tests/collision/test_sdf_tier_wired.cpp. Partial backstop today = test_sdf_contact_adjoint_fd.
(b) collision host-narrowphase (narrowphase_dispatch.hpp + contact_stream_driver) is LIVE (8 surviving
    solver/union gates + analytical_manifold/narrowphase_dispatch/contact_stream_driver tests). It is NOT a
    forbidden special world -- it's collision infra / a test oracle. M11 disposition = decide migrate-vs-keep.
(c) sdf_contact.{cu,hpp} + sdf_contact_adjoint.{cpp,hpp} + sdf_device_world.{cu,hpp} + diffsim/sdf_contact_ift
    + codegen/generated/{featherstone,rigid}_sdf_contact_forward.cu = the LIVE v0.7 Newton-on-SDF contact-
    gradient (IFT) family; deferred from M9 "with rt". These all #include buffer_legacy -> couple tightly to
    the buffer sweep.
(d) buffer_legacy (T13, deferred from M9 in full): non-RT + RT consumers migrate to phi v2 BufferI as ONE
    ATOMIC SWEEP with a SINGLE stream-binding decision and a SINGLE D1 re-gate, then delete buffer_legacy.hpp
    + legacy phi::{OwnedStream,DeviceContext,UploadVector,StreamView,...}. The IFT/KKT/OSC/sparse-CG/SDF-IFT
    families (diffsim/*) were deferred here too. OSC API still uses legacy ArticulationDeviceState -> M11.
(e) rt/ + render_scene + DLPACK-Vulkan interop (this milestone).

==== OUTPUT CONTRACT ====
Return the structured object the schema demands. Be concrete and file-level. For every file you name, give
its disposition (migrate / delete / keep / new / op-ify / interface-ify) and WHY. Order tasks by dependency.
Call out every D1 byte-exact risk and every owner-level decision (with a recommendation). This is the plan an
implementer will follow with NO further investigation -- so be exhaustive and precise, not high-level.
`

const FACET_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['facet', 'summary', 'files_inventory', 'tasks', 'd1_risks',
             'owner_decisions_needed', 'cross_facet_dependencies', 'open_questions'],
  properties: {
    facet: { type: 'string' },
    summary: { type: 'string', description: '3-8 sentence executive summary of this facet of M11' },
    files_inventory: {
      type: 'array',
      description: 'Every relevant file with its M11 disposition',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['path', 'role', 'disposition', 'why'],
        properties: {
          path: { type: 'string' },
          role: { type: 'string', description: 'what it currently is / does' },
          disposition: { type: 'string', enum: ['migrate', 'delete', 'keep', 'new', 'op-ify', 'interface-ify', 'investigate'] },
          why: { type: 'string' },
        },
      },
    },
    tasks: {
      type: 'array',
      description: 'Dependency-ordered executable tasks for this facet',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'title', 'detail', 'files_touched', 'risk', 'depends_on', 'local_verifiable', 'gate'],
        properties: {
          id: { type: 'string', description: 'e.g. M11-BUF-1' },
          title: { type: 'string' },
          detail: { type: 'string', description: 'concrete how-to: interfaces, signatures, migration mechanics' },
          files_touched: { type: 'array', items: { type: 'string' } },
          risk: { type: 'string', enum: ['low', 'med', 'high'] },
          depends_on: { type: 'array', items: { type: 'string' }, description: 'task ids (this or other facets)' },
          local_verifiable: { type: 'boolean', description: 'can the gate run on this headless lavapipe box?' },
          gate: { type: 'string', description: 'exact verification (ctest target / build+link / offscreen smoke / lint / git-grep)' },
        },
      },
    },
    d1_risks: { type: 'array', items: { type: 'string' }, description: 'byte-exact / golden / async-stream risks + mitigation' },
    owner_decisions_needed: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['question', 'options', 'recommendation'],
        properties: {
          question: { type: 'string' },
          options: { type: 'array', items: { type: 'string' } },
          recommendation: { type: 'string' },
        },
      },
    },
    cross_facet_dependencies: { type: 'array', items: { type: 'string' }, description: 'how this facet couples to the other 5' },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
}

const FACETS = [
  {
    key: 'viewer-imgui-delta',
    label: 'facet:viewer',
    prompt: `FACET 1 -- ImGui VIEWER DELTA (M8.5-done vs M11-required).
Scope: the interactive on-screen viewer. M8.5 already built src/runtime/app/viewer/{viewer_main,imgui_layer,
camera_controller} + src/render/imgui + src/render/window + present renderer. M11 plan line 509 requires the
viewer to expose: scene tree, playback (pause/step/play), env selection, camera, DRIVE SLIDERS (write PD
targets into Data), and DRAG ENTITY (a MoveEntity command -> writes Data / Scene).
INVESTIGATE: read all of src/runtime/app/viewer/*, src/runtime/app/command_queue.hpp, imgui_layer.{hpp,cpp},
the SceneNode editor seam (graph/scene_graph.hpp has Selected()/SetSelected() per plan line 247), and how the
frame loop (simulation.cpp/systems.cpp) routes commands. Determine PRECISELY what M8.5 already implements vs
the M11 delta. Identify the MoveEntity command path (command_queue -> a system -> writes Data body/link pose
or Scene local transform), the drive-slider path (write PD target into nk Data), env-selection, scene-tree
widget, playback transport. Note the M8/M8.5 viewer named debt (vk-imgui-stale-renderpass-after-recreate,
present-resize OUT_OF_DATE, camera evdev->xcb-keysyms, out-of-band StepOnce) and which M11 must fix.
Gate model: build+link + offscreen imgui draw-data smoke (NOT locally runnable interactively). Output the
file-level delta task list.\n\n${SHARED}`,
  },
  {
    key: 'cuda-vulkan-interop',
    label: 'facet:interop',
    prompt: `FACET 2 -- CUDA<->VULKAN INTEROP (CudaVulkanInteropPublisher, DLPACK zero-copy).
Scope: M11 plan line 509 + §3.10 line 372 + the firm render-physics-shared-memory requirement: render must
share GPU memory with physics (DLPACK-VULKAN zero-copy interop) so Vulkan renders the REAL physics state.
M8 deliberately left a PosePublisher abstraction (src/runtime/app/pose_publisher.{hpp,cpp}, with
HostDownloadPublisher) as the no-rework swap point; M8.5 review noted PoseSource::Base is the dead path the
M11 interop publisher first lights.
INVESTIGATE: read pose_publisher.{hpp,cpp}, systems.cpp (how poses flow Data->render instance world_xform),
render_world.{hpp,cpp} (the transform/instance buffer the interop must register), launch.cuh (the only
<<<>>> wrapper -- the device scatter kernel must go through it), and how a Vulkan vertex/transform buffer
would be registered as CUDA external memory (cudaImportExternalMemory / VK_KHR_external_memory). Design the
CudaVulkanInteropPublisher: it implements the same PosePublisher interface but scatters device poses directly
into a Vulkan-owned buffer via a CUDA kernel, NO host download. CRITICAL CONSTRAINTS: (1) src/runtime/app
keeps the ZERO-CUDA-TOKEN lint redline -> the scatter kernel + external-memory import must live in
backend_cuda (or a phi seam), the app layer only calls a backend-agnostic publisher. (2) This is NOT locally
verifiable (CUDA-GPU vs lavapipe-CPU cannot share memory) -> gate = build+link only; the host-download path
stays the working/CI path. Output the interface design + file-level task list + the exact lint-redline-safe
placement of the CUDA scatter + external-memory import.\n\n${SHARED}`,
  },
  {
    key: 'rt-backend-homing',
    label: 'facet:rt',
    prompt: `FACET 3 -- RT PATH TRACER + RtBackendI HOMING + render_scene disposition (§3.10 RT row).
Scope: M11 plan line 509 + §3.10 line 371. Today src/rt/ holds a self-written CUDA path tracer:
bvh_ray_traversal.{cu,hpp}, scene_render.{cu,hpp}, two_level_render.{cu,hpp}, camera.hpp, framebuffer.hpp,
material.hpp. M11 must: (1) define src/render/rt_backend.hpp = RtBackendI backend-agnostic interface (BLAS
build once / TLAS build per frame / trace dispatch / output buffer via phi v2 BufferI); (2) home the CUDA
impl to src/phi/backend_cuda/rt/ (self-written, NO OptiX, unchanged algorithm -- LINE-FAITHFUL move);
(3) src/render/rt_adapter.{hpp,cpp} = RenderWorldToTwoLevelScene (consume the SAME RenderWorld the raster
path uses); (4) the render interface layer must end with ZERO CUDA token; (5) render_scene.{hpp,cpp} (plan
line 499 marks for deletion -- confirm it has no live consumer, vs render::BuildRenderScene in render_world);
(6) RT output is STILL D1 -- the offscreen RT path is a local CI/D1 gate (this box CAN run CUDA RT to an
image buffer).
INVESTIGATE: read all of src/rt/*, render_scene.{hpp,cpp}, render_world.{hpp,cpp} (what RenderWorld exposes
for RT), grep current consumers of src/rt/* and render_scene, and check src/rt/* buffer_legacy includes
(they couple to the buffer sweep). Determine the exact RtBackendI vtable, what stays interface vs backend,
how RenderWorld feeds geometry/material/camera/lights to the path tracer, and the offscreen RT D1 gate
(image-buffer byte compare across dispatch vs replay). Output interface design + line-faithful homing task
list + render_scene disposition + the offscreen D1 RT gate definition.\n\n${SHARED}`,
  },
  {
    key: 'collision-sdf-cluster',
    label: 'facet:collision-sdf',
    prompt: `FACET 4 -- COLLISION/SDF CLUSTER (M9 carry-forward (a),(b),(c)).
Scope: the LIVE collision + SDF-contact-gradient estate deferred from M9. Files: src/collision/
narrowphase_dispatch.hpp + the contact_stream_driver (host narrowphase driver, 8 surviving solver/union
gates); src/collision/sdf_contact.{cu,hpp}, sdf_contact_adjoint.{cpp,hpp}; src/runtime/sdf/sdf_device_world.
{cu,hpp}; src/diffsim/sdf_contact_ift.{cpp,hpp}; src/codegen/generated/{featherstone,rigid}_sdf_contact_
forward.cu; the broadphase SAP oracle (tests/collision/test_lbvh_vs_sap_pair_set.cpp O(n^2) reference);
src/phi/backend_cuda/ops/narrowphase_sdf.cu (already op-ified in M5 -- the RUNTIME SDF narrowphase).
THREE jobs: (1) DISPOSITION each file -- which migrate to phi v2 ops/interfaces (per §3.10), which stay host
test-oracle/IFT infra, which delete. Distinguish the RUNTIME SDF contact (narrowphase_sdf.cu, op-ified) from
the DIFFERENTIABLE Newton-on-SDF IFT family (sdf_contact_ift / sdf_contact_adjoint / codegen). (2) SDF-TIER
FORWARD GATE RE-ASSERT: salvage test BoxOnBoxMatchesAnalyticalNormalAndPenetration VERBATIM from commit
fcae2a6:tests/collision/test_sdf_tier_wired.cpp (use git show), find where it should now live (tests/oracle
or tests/collision), and what it asserts about narrowphase_dispatch/find_sdf_contact_newton. (3) Confirm none
of this is a forbidden special world (it is collision infra / IFT diffsim / test oracle -- per highest
directive that is allowed; flag if any path special-cases a demo).
INVESTIGATE: read narrowphase_dispatch.hpp, contact_stream_driver, sdf_contact*, sdf_device_world*,
sdf_contact_ift*, the two codegen sdf_contact_forward.cu, and run git show fcae2a6:tests/collision/
test_sdf_tier_wired.cpp. Note every buffer_legacy include (coordinate with the buffer facet -- these migrate
together). Output disposition table + the gate-salvage task + migration tasks ordered against the buffer
sweep.\n\n${SHARED}`,
  },
  {
    key: 'buffer-legacy-sweep',
    label: 'facet:buffer',
    prompt: `FACET 5 -- buffer_legacy FULL SWEEP (T13, the biggest + riskiest M11 task).
Scope: migrate ALL consumers of src/phi/buffer_legacy.hpp + legacy phi::{OwnedStream,DeviceContext,
UploadVector,StreamView,...} to phi v2 BufferI/BufferType/DeviceI, as ONE ATOMIC SWEEP, then DELETE
buffer_legacy.hpp. There are ~65 includers today (run: git grep -l buffer_legacy -- src/ python/ tests/).
KNOWN STRUCTURE (from the M9 T13 read-only recon, agent a4c27e -- verify it):
- Two HUBS force atomic migration: src/runtime/articulation/articulation_state.hpp (~30 buffers) and
  src/phi/buffer_transfer.hpp, plus their ~25/~30 consumer trees. The NAME COLLISION is the forcing function:
  buffer_legacy.hpp declares 'class Buffer' and buffer.hpp declares 'struct Buffer' at the SAME FQN
  (nuka::phi::Buffer) -> they cannot coexist in one TU -> no leaf-by-leaf migration; the hub + its tree go
  in one shot.
- ASYNC-STREAM D1 RISK (the single biggest D1 risk in M11): legacy CopyFromHost/ToHost are DEFAULT-STREAM
  SYNCHRONOUS memcpy; v2 BufferUpload/Download are BACKEND-STREAM ASYNC (a different stream than each
  consumer's DeviceContext stream). A correct migration must rebind consumers to the backend stream -> that
  changes the stream the golden-producing kernels run on -> the FULL determinism golden set must be re-gated
  for D1 after the sweep. This is why it is ONE decision + ONE re-gate, done once (not split non-RT vs RT).
- BRIDGE TRUTH: there is NO device-keyed BufferType registry; each non-RT consumer holds only a legacy
  DeviceContext. Getting a v2 BufferType requires either threading Backend*/BufferType* through call sites
  (wide but shallow ripple incl. ~40 test harnesses) OR a BufferTypeForDevice() bridge (cheap but
  wrong-stream). Pick and justify.
- DEAD CODE to delete in the same sweep: c_abi BufferRecord/BufferTable (verified dead: BufferTable() zero
  calls; buffer.cpp goes through nuka_world_get_buffer_view/FieldPtr). Watch the internal.hpp ->
  buffer_legacy include vs diffsim.cpp transitive dependency.
- The IFT/KKT/OSC/sparse-CG/SDF-IFT families (src/diffsim/*: ift_runner, kkt_builder, sparse_solver_backend,
  recompute_orchestrator, checkpoint, backward_runner, tape, sdf_contact_ift) + OSC legacy ArticulationDevice
  State API all sit on buffer_legacy and migrate in this sweep. rt/*.cu also include buffer_legacy.
INVESTIGATE: enumerate ALL 65 includers and CLASSIFY them (hub / hub-consumer / diffsim-IFT / sdf / rt /
collision / sensor / c_abi-dead / test-harness). Read articulation_state.hpp + buffer_transfer.hpp + buffer.hpp
+ buffer_legacy.hpp + backend_cuda/buffer.cu to nail the exact v2 API the migration targets. Produce the
TOPOLOGICAL migration order (hubs first or leaves first?), the stream-binding DECISION (with the D1
implication spelled out), the single D1 re-gate plan (which golden/ctest targets must re-pass), and the
buffer_legacy.hpp deletion task. This facet's tasks gate the RT and collision/sdf facets (they share
buffer_legacy). Be exhaustive: list every includer with its disposition.\n\n${SHARED}`,
  },
  {
    key: 'exitgate-sequencing-310',
    label: 'facet:exitgate',
    prompt: `FACET 6 -- §3.10 HOMING COMPLETENESS + M11 EXIT GATE + CROSS-FACET SEQUENCING + 最高指令 compliance.
Scope: the cross-cutting / closing facet.
INVESTIGATE & DELIVER:
(1) §3.10 COMPLETENESS: grep the engine side (src/nk, src/scene, src/render, src/runtime/app) for any
    remaining BARE CUDA (.cu files or CUDA tokens) that M11 must home to backend_cuda per the zero-CUDA-token
    redline. Confirm the lint redline list (tools/lint/physics_smell.py) and what M11 adds to it (render
    interface layer, runtime/app). Identify the OSC legacy ArticulationDeviceState API (M11 homing per
    buffer facet) and any other M1-era "placeholder/bare .cu" still in the engine layer.
(2) M11 EXIT GATE definition: write the precise exit-gate command set, analogous to M9's. Candidates:
    git-grep ZERO for buffer_legacy + legacy phi types (OwnedStream/DeviceContext/UploadVector/StreamView);
    ctest -L full && -L perf green; lint whole-repo (incl. new zero-CUDA redlines for render-interface +
    runtime/app); offscreen RT D1 byte gate; build+link for the non-runnable viewer/interop/RT-present;
    union_cook_golden 0.0 + determinism goldens re-passed after the buffer async-stream re-gate. Specify
    which gates are LOCALLY RUNNABLE vs OWNER-VERIFIED-ON-DISPLAY.
(3) CROSS-FACET SEQUENCING: produce the dependency DAG across all 6 facets. The buffer_legacy sweep is
    foundational (RT/collision-sdf/diffsim sit on it). Viewer + interop are leaf (depend on render + publisher).
    Propose the M11 task phase order (e.g. Phase 0 buffer sweep + D1 re-gate -> Phase 1 collision/sdf + RT
    homing -> Phase 2 render_scene delete + rt_adapter -> Phase 3 interop publisher -> Phase 4 viewer delta ->
    Phase 5 exit gate + ultracode review). Flag any ordering where a deletion would break the build mid-phase.
(4) 最高指令 COMPLIANCE: confirm NO M11 task reintroduces a special-cased physics path or dedicated world type.
    The RT/viewer/interop are render/UI, not physics -- but verify the interop publisher and any new scene-edit
    command write through the GENERAL Scene/Data path, not a demo-specific shim.
(5) M10 BOUNDARY: confirm what M11 must NOT do (all RL + go2 PD-stance + per-link wrench + batched autoreset
    /base-pose/control-modes belong to M10, the LAST milestone). Flag anything in the carry-forward ledger
    that is actually M10 not M11.
Read docs/plans/...refactor.md (M9/M10/M11 + §3.10), tools/lint/physics_smell.py, tests/CMakeLists.txt (T14
label scheme fast/full/perf -- where do new M11 tests register), and the M8 render-gate doc
docs/architecture/m8-render-gates.md. Output the §3.10 completeness table + exit-gate command set + the
cross-facet phase DAG + compliance verdict.\n\n${SHARED}`,
  },
]

phase('Recon')
log('M11 recon: dispatching 6 facet investigators (viewer, interop, RT, collision/sdf, buffer-legacy, exit-gate/sequencing)')

const findings = await parallel(
  FACETS.map((f) => () =>
    agent(f.prompt, { label: f.label, phase: 'Recon', schema: FACET_SCHEMA, model: 'opus' })
  )
)

const ok = findings.filter(Boolean)
log(`M11 recon: ${ok.length}/${FACETS.length} facets returned`)

phase('Synthesize')
const synthPrompt = `You are the SYNTHESIS agent for the M11 ultracode recon of the Nuka-Physics nk-core refactor.
You have 6 facet findings (JSON below). Reconcile them into ONE authoritative construction plan and WRITE it
to /root/Nuka-Physics/subagent-plans/m11-recon.md (use the Write tool). This file becomes the EXECUTION
BASELINE for M11 implementation -- an implementer will follow it with no further investigation.

${SHARED}

The 6 facet findings (viewer, interop, RT, collision/sdf, buffer-legacy, exit-gate/sequencing):
${JSON.stringify(ok, null, 1)}

WRITE m11-recon.md with these sections (markdown):
1. ## M11 Scope & Mission -- the 6 work-buckets, the highest-directive + D1 framing, what M11 is NOT (M10 boundary).
2. ## Cross-Facet Dependency DAG & Phase Order -- the definitive ordered phase plan (buffer sweep foundational
   first; RT/collision-sdf homing; render_scene delete + rt_adapter; interop publisher; viewer delta; exit gate
   + ultracode review). For each phase list its tasks (ids), what gates it, and whether locally verifiable.
3. ## Per-Facet Task Tables -- one subsection per facet, the dependency-ordered tasks with files_touched, risk,
   gate, local_verifiable. Reconcile overlaps (esp. buffer-legacy vs RT vs collision/sdf -- they share
   buffer_legacy; state the single atomic-sweep boundary).
4. ## D1 Byte-Exact Risk Register -- consolidated, esp. the buffer async-stream re-gate (the #1 risk) and the
   RT-still-D1 offscreen gate and any cook-real-normals parse-vn D1-sensitive item. For each: risk + mitigation
   + which golden/ctest must re-pass.
5. ## §3.10 Homing Completeness -- the table of remaining bare-CUDA / interface-ification + the zero-CUDA-token
   redline additions (render interface layer, runtime/app).
6. ## M11 Exit Gate -- the precise command set (git-grep zero for buffer_legacy + legacy phi types; ctest -L
   full/-L perf; lint; offscreen RT D1; build+link for non-runnable viewer/interop/present; determinism goldens
   re-passed). Mark each LOCAL vs OWNER-ON-DISPLAY.
7. ## OWNER DECISIONS NEEDED -- consolidated, deduplicated list of every owner-level decision across facets,
   each with options + a controller recommendation. (These will be surfaced to the owner via AskUserQuestion
   before implementation, mirroring M9's 4 rulings.) If a decision is clearly controller-resolvable under
   existing owner rulings (strict-dominance / never-ruled), mark it CONTROLLER-RESOLVABLE with the call instead.
8. ## Open Questions / Investigate-Further -- anything a facet could not resolve read-only.

Be exhaustive and file-level. Do NOT lose any task or risk from any facet. After writing the file, RETURN a
concise summary: total task count, the phase order, the count + one-line list of OWNER DECISIONS NEEDED, and
the single biggest risk. Your returned text is NOT shown to the user directly -- it is the controller's digest.`

const synthSummary = await agent(synthPrompt, { label: 'synthesis', phase: 'Synthesize', model: 'opus' })

log('M11 recon synthesis complete -> subagent-plans/m11-recon.md')
return { facets_returned: ok.length, synthesis: synthSummary }
