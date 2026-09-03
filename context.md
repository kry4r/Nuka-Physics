# Root Cause: Island Byte-Identity Nondeterminism (READ-ONLY scout)

## Verdict

The dynamic island solve (BuildSolveIslands/SolveRowsBlockIsland) is NOT the source of
the flakiness. The REAL root cause is UPSTREAM of the islands entirely:

**RANK 1 (root cause): the broadphase candidate-pair stream has a run-to-run
NONDETERMINISTIC SLOT ORDER, and that order flows unchanged into contact slots,
row slots, GS sweep order, and the warm-start cache — i.e., the float summation
order itself differs between ANY two simulations, including two STATIC-schedule
runs.**

Chain of evidence (all exact):

1. Pair emission order is atomicAdd race order:
   - `src/phi/backend_cuda/ops/broadphase.cu:365` —
     `const uint32_t slot = atomicAdd(&out_count[env], 1u);` inside
     `EnvQueryPairsKernel` (kernel starts broadphase.cu:314; op entry
     `OpLbvhQueryPairs` broadphase.cu:531-657).
   - Each concurrent leaf-thread that finds an overlap claims the NEXT free slot;
     CUDA warp/block scheduling decides who wins. With >=2 overlapping pairs in an
     env (3 mutually-overlapping dog trunks at spawn give 2-3 pairs), the pair ->
     slot assignment differs BETWEEN RUNS OF THE SAME BINARY.
   - There is NO downstream sort/canonicalization of the emitted pair stream:
     pipeline op order is BuildAabbs -> LbvhBuild -> LbvhQueryPairs ->
     Narrowphase* -> AssembleRows (`src/nk/pipeline/pipeline.cpp:285-301,357,499`);
     nothing sorts `candidate_pairs`. (The radix sorts at broadphase.cu:83/606 are
     LBVH Morton construction / particle grid, NOT the pair output.)

2. Contact slots inherit the pair slot:
   - Narrowphase writes ucontact_* into the SAME slot index
     (`src/phi/backend_cuda/ops/narrowphase_prims.cu:239,251`;
     `narrowphase_sdf.cu:317`; `narrowphase_heightfield.cu:642`).

3. Row slots inherit the contact slot:
   - `src/phi/backend_cuda/ops/assemble_rows.cu:~985-1006` (`ContactRowOffsets`):
     `base = env*rows_per_env (+ ...) slot * kPdRowsPerSlot`, normal/tangent rows
     at fixed offsets inside the slot footprint. So a given physical pair's rows
     land at whatever slot the broadphase race handed out THIS step.

4. The solve's float arithmetic is order-dependent over those slots:
   - Gauss-Seidel is sequential in ascending row-slot order:
     static: `SolveRowsBlockIslandKernel`, compaction into `order_sh`
     (solve_rows.cu:1424-1450) preserves schedule/slot order, one warp per row,
     `__syncthreads()` between colors;
     dynamic: each row its own color, `__syncwarp()` between rows
     (solve_rows.cu:1460-1508); scalar path solve_rows.cu:1050-1112 iterates
     `row_order[seg_off + r]` ascending.
   - Each `lambda[gslot] += ...` / velocity apply feeds the NEXT row's `jv`
     (`ComputeSlimRowVelocity`, solve_rows.cu:398-455). Changing visitation order
     of interacting rows changes the rounding — classic non-associativity, exactly
     the observed ONE-ULP-per-field signature after 160 steps x (32 vel + 28 pos)
     iters of warm-started feedback.

5. The warm-start cache adds MORE slot-order sensitivity:
   - `PrepareContactWarmStartKernel` dedups/matches by scanning `point_index`
     order (`src/phi/backend_cuda/ops/assemble_rows.cu:1005-1070`) and
     `RebuildContactCacheKernel`'s free-rank compaction
     (`old_keep[i]` rank scan, assemble_rows.cu:1190-1260) assigns snapshot
     sources by point_index rank — both are functions of the (nondeterministic)
     slot ordering.

CONSEQUENCE: the test compares TWO INDEPENDENT 160-step simulations (dyn run then
static run, `tests/scenario/island_byte_identity.cpp:172-175,196-198`). EACH roll
its own dice on pair-slot order. Even a bit-perfect island implementation fails
this memcmp whenever the two runs happen to order overlapping pairs differently —
matching "flaky EVEN ALONE (2 pass / 1 fail)" and "passes alone, fails inside the
full suite" (GPU load changes warp scheduling).

**RANK 2 (latent, currently arithmetic-neutral): nondeterministic ORDERS inside the
dynamic path that do not yet touch float sums but would if accumulation ever became
cross-component:**
- Component id reservation by atomicAdd: `build_islands.cu:164`
  (`atomicAdd(island_count, 1u)` in `EmitIslandsKernel`) — island INDEX order
  varies per run (partition and per-island row order do NOT; see below).
- Per-component tile-list compaction by atomicAdd: `solve_rows.cu:1303`
  (`tile_list[atomicAdd(&tile_cnt_sh,1)] = t`) — tile load/scatter order varies,
  but every write is to a distinct slot (no accumulation), so bit-neutral today.
- Scalar-island thread mapping `island = blockIdx.x + threadIdx.x*gridDim.x`
  (solve_rows.cu:1063-1070) — scheduling-dependent, but components are
  state-disjoint so no numeric effect.
These are NOT the observed bug; do not "fix" them for this symptom.

**RANK 3 (cleared): island DISCOVERY is deterministic in everything that matters.**
- `UnionRowsKernel`/`Unite` (build_islands.cu:56-72): lock-free smaller-root-wins
  union with CAS-retry yields the SAME partition and min-index roots regardless of
  interleaving; `FindReadonly` runs after a kernel boundary.
- The stable cub radix sort (build_islands.cu:225-240) groups rows per component
  ASCENDING and stably — identical to the static single-island sweep order.
- Components are state-disjoint by construction (artic/body/particle coupling keys
  + friction-group anchor, build_islands.cu:96-131), so concurrent blocks never
  accumulate into shared floats; there is NO float atomicAdd anywhere in the solve
  path (grep confirms: only uint32 atomicOr/atomicAdd for flags/counters/lists).

## Minimal GENERAL fix (one physics path, no scene hacks)

Canonicalize the broadphase output ONCE, generically:
after `EnvQueryPairsKernel`, stable-sort the per-env pair stream by the packed
canonical key `(min(a,b)<<32)|max(a,b)` (already emitted canonically i<j,
broadphase.cu:357-360) with `cub::DeviceRadixSort::SortPairs` into the existing
scratch-arena pattern used by ParticleGridBuild (broadphase.cu:592-615) — or have
AssembleRows consume a sorted copy. Then:
- contact slot -> pair becomes a pure function of body ids,
- row slots, GS sweep order, and the warm-start point_index ranks all become
  deterministic,
- BOTH schedules read identical row streams and the byte-identity test measures
  what it was meant to measure (island scheduling only).
This is one general D1 anchor consistent with the repo's stated determinism model
("integer-only atomics, radix stable_sort, sorted compact output",
CMakeLists.txt:1071); the pair-emission atomicAdd is the one place that violates it.

Residual risk: until pairs are canonicalized, ANY byte-identity A/B test that runs
two separate simulations is measuring broadphase scheduling noise, not the feature
under test.

## Start Here
Open `src/phi/backend_cuda/ops/broadphase.cu` lines 314-400 (`EnvQueryPairsKernel`)
and add the canonicalizing radix sort in `OpLbvhQueryPairs` (lines 531-657).

## Files Retrieved
1. `src/phi/backend_cuda/ops/build_islands.cu` (1-251) — union-find + radix island build; cleared.
2. `src/phi/backend_cuda/ops/solve_rows.cu` (60-1852) — SolveRowsBlockIsland/warp/scalar paths; order-sensitive GS confirmed; no float atomics.
3. `src/phi/backend_cuda/ops/broadphase.cu` (254-400, 525-661) — ROOT CAUSE: atomicAdd pair-slot emission, no output sort.
4. `src/phi/backend_cuda/ops/assemble_rows.cu` (600-850, 990-1290) — slot->row mapping; order-sensitive warm-start match/rebuild.
5. `src/nk/pipeline/pipeline.cpp` (280-640) — op wiring proves no pair canonicalization stage exists.
6. `tests/scenario/island_byte_identity.cpp` (1-216) — two independent sims compared; env-var forcing via `_putenv_s` read once at World Build (pipeline.cpp:601).

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Ranked root causes with file:line evidence in context.md: broadphase.cu:365 atomicAdd pair-slot emission -> assemble_rows.cu ContactRowOffsets slot->row mapping -> order-dependent GS in solve_rows.cu; island discovery (build_islands.cu) explicitly cleared."
    }
  ],
  "changedFiles": [
    "context.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "grep/find/read inspection of broadphase.cu, build_islands.cu, solve_rows.cu, assemble_rows.cu, pipeline.cpp, island_byte_identity.cpp",
      "result": "passed",
      "summary": "Read-only trace of pair->contact->row slot flow; confirmed no pair-output sort and no float atomics in the solve path"
    }
  ],
  "validationOutput": [
    "Static analysis only (READ-ONLY task); no builds or tests executed"
  ],
  "residualRisks": [
    "Fix unverified empirically: implementing the pair-stream radix sort requires a follow-up editing+test pass",
    "If any other family (UnionCsr) emits pairs through a different unsorted atomicAdd path, it needs the same canonicalization audit"
  ],
  "noStagedFiles": true,
  "diffSummary": "No source changes; context.md findings document written per runtime output-path override",
  "reviewFindings": [
    "blocker: src/phi/backend_cuda/ops/broadphase.cu:365 - candidate pair slot assigned by atomicAdd race with no downstream canonicalization; makes row/GS summation order run-to-run nondeterministic (root cause of flaky byte-identity)",
    "minor: src/phi/backend_cuda/ops/assemble_rows.cu:1190-1260 - warm-start rebuild free-rank compaction depends on nondeterministic point_index order (secondary order sensitivity)",
    "info: build_islands.cu:164 and solve_rows.cu:1303 atomic orders are arithmetic-neutral today (disjoint components, distinct write slots) - do not change for this symptom"
  ],
  "manualNotes": "Test instrumentation note: [isi-dbg] prints in island_byte_identity.cpp are temporary per task brief; removal belongs to main agent. Static-vs-dynamic env var is read once at World Build (pipeline.cpp:601) via getenv after _putenv_s - correct usage in the test.",
  "notes": ""
}
```

## Follow-up verification

The pair-stream canonicalization is implemented in `src/phi/backend_cuda/ops/broadphase.cu`: emitted body pairs are snapshotted, sorted by global canonical pair key with CUB radix sort, and scattered back into the rigid candidate slots. The World sizes `pair_sort_scratch` before graph capture. `PairClampCountsKernel` bounds its thread index by `env_count`, and the op rejects sort sizes outside CUB's signed-int item-count contract instead of falling back to nondeterministic ordering.

Validation on the Windows CUDA build (`build-win`, CUDA 13.2):
- `nuka_phi2` and `nuka_scenario_test` build successfully.
- `IslandByteIdentity.*`: 2 tests passed across 3 repeated runs.
- Candidate stream, LBVH filtered-pair, and link-rigid candidate-pair suites: 17 tests passed.
- Remaining compiler output is the repository's existing C4819 encoding warnings and CUDA remarks.

