#pragma once
// ---------------------------------------------------------------------------
// nk::SolveSchedule — the device-resident island/color row schedule (plan §3.4).
//
// Built ONCE on the host at World construction (and Reset, were the topology to
// change — it cannot: fixed capacities) over the MAX-CAPACITY row slots of the
// Model's union contact template, then uploaded device-resident as the three
// Model fields {island_row_offsets, island_color_segments, row_order}. At
// runtime the SolveRowsBlockIsland kernel walks the triple with ZERO host
// participation and ZERO re-coloring; per-slot active flags (the watermark)
// mask empty slots.
//
// ALGORITHMS: BuildRowColorPartitions / BuildRowComponentPartitions migrated
// from src/solver/gpu/row_scheduler.cu (the legacy per-step host scheduler —
// the 140-200 ms/step pathology this milestone retires), re-keyed from
// constraint::RowBuffers to the slim ScheduleRow conflict description. The
// greedy-lowest-color-by-row-index coloring, the dense slot remap, the
// union-find component pass, and the (color, intra-color) segment emission are
// SEMANTICALLY IDENTICAL — for the worst-case (all-slots-active) row set the
// emitted execution order equals what the legacy scheduler produced for the
// same emission order, which is what makes the nk trajectory track the legacy
// coresident union world at the FP floor.
//
// ★ THE WORST-CASE-SLOT VALIDITY INVARIANT (the heart of plan replayability).
// The schedule is computed once with EVERY row slot assumed live. At runtime
// any SUBSET of slots is active. The schedule stays valid for every subset
// because:
//   (1) Coloring: deactivating rows only REMOVES edges from the conflict graph;
//       a proper coloring of the supergraph is a proper coloring of every
//       subgraph. No two ACTIVE rows in one color segment ever share mutable
//       state, whatever the subset.
//   (2) Order: within the interacting set, rows execute in ascending
//       (color, slot) order. The legacy per-step scheduler colored the
//       COMPACTED emission greedily by index, which also yields ascending
//       emission order within each mutually-conflicting subset — so the
//       relative update order of every pair of INTERACTING active rows is the
//       same under both schedules; rows in different components/disjoint state
//       may interleave differently, which cannot change any float.
//   (3) Islands: the worst-case connected components are a REFINEMENT bound —
//       an active subset's true components are sub-unions of the worst-case
//       islands, so running one block per worst-case island serializes at
//       least as much as required (never less). Inactive rows early-exit.
// Hence one fixed schedule serves every step — the CUDA-graph plan never needs
// recapture (plan §3.2 "拓扑不变永不重捕获").
// ---------------------------------------------------------------------------

#include <cstdint>
#include <vector>

namespace nuka::nk {

class Model;

// One row's conflict description (the slim re-key of the legacy RowBuffers +
// sides + art_refs inputs): up to 2 shared-mutable-state keys (dense body /
// articulation-tile keys; ~0u = absent, the legacy kInvalidBodyIndex) plus the
// lambda-group anchor (TotalNormalLambda reads [group_first, ...) — kept as a
// component edge by construction, exactly like the legacy pass 4).
struct ScheduleRow {
    uint32_t key_a = ~0u;
    uint32_t key_b = ~0u;
    uint32_t group_first = ~0u;  // global row index of the group anchor
    uint32_t artic = ~0u;        // articulation index (qdot tile) or ~0u (side A)
    // S4 (general contact pipeline Phase 1B): a row whose side B is ALSO an
    // articulation (artic x artic, e.g. two co-resident dogs colliding) carries a
    // SECOND articulation key here so the union-find merges BOTH sides' islands
    // into one component (they share mutable qdot state and must serialize). ~0u
    // for a single-artic row (foot / artic x rigid / artic x static) — identical
    // to the legacy single-key behaviour, so the Union schedule is
    // unchanged at K==1.
    uint32_t artic_b = ~0u;      // second articulation index (side B) or ~0u
};

struct SolveScheduleResult {
    // row_order: global row ids grouped per island, ascending (color, intra).
    std::vector<uint32_t> row_order;
    // segments: {row_order offset, count} u32 pairs, contiguous per island.
    std::vector<uint32_t> color_segments;
    // islands: {segment offset, segment count, flags, env} u32 quads.
    // flags bit0 = the island contains articulation rows (owns the qdot
    // pack-tile scatter for its env).
    std::vector<uint32_t> islands;
    uint32_t island_count = 0;
    uint32_t segment_count = 0;
};

class SolveSchedule {
public:
    // Build the worst-case schedule for `model` (union / pair-driven family; a
    // contact-free model yields an empty schedule) and write the triple +
    // counts into the model's schedule_* host tables (staged into the Model
    // device buffer by Model::UploadTo). Deterministic: same model -> byte-same
    // schedule.
    static void Build(Model* model);

    // The migrated legacy algorithms (exposed for the schedule equivalence
    // testing seam; semantics == row_scheduler.cu Build*Partitions).
    static SolveScheduleResult Partition(const std::vector<ScheduleRow>& rows,
                                         const std::vector<uint32_t>& row_env);
};

// L1-c: UnionSlotRowBases (the union-slot row-layout contract) was DELETED with
// the UnionCsr path. The PairDriven schedule derives its row layout directly
// from ModelCapacities::max_rows_per_env (see SolveSchedule::Build).

}  // namespace nuka::nk
