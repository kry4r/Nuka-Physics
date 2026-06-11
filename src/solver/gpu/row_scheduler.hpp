#pragma once
// ---------------------------------------------------------------------------
// nuka::solver::gpu::row_scheduler -- deterministic CSR row coloring
// ---------------------------------------------------------------------------

#include "constraint/contact_row_sides.hpp"
#include "constraint/row_articulation_refs.hpp"
#include "constraint/row_buffers.hpp"

#include <cstdint>
#include <vector>

namespace nuka::solver::gpu {

struct RowColorRange {
    uint32_t row_offset = 0u;
    uint32_t row_count = 0u;
};

struct RowColorPartitions {
    std::vector<uint32_t> row_indices;
    std::vector<RowColorRange> color_ranges;

    uint32_t ColorCount() const {
        return static_cast<uint32_t>(color_ranges.size());
    }
};

RowColorPartitions BuildRowColorPartitions(const constraint::RowBuffers& rows);
bool ValidateNoSharedBodiesPerColor(const constraint::RowBuffers& rows,
                                    const RowColorPartitions& partitions);

// ---------------------------------------------------------------------------
// Connected-component repartition of a colored row set (throughput increment,
// spec gate G1(d)). Two rows INTERACT during the sweep iff they share mutable
// solver state; the full shared-state relation (a superset of the coloring's
// shared-VALID-body conflict relation) is:
//   1. a shared valid body index            (BodyState velocity/position writes)
//   2. a shared articulation art_index      (the qdot tile both sides apply into)
//   3. a shared ParticleInvMass handle      (the particle velocity both write)
//   4. the same contact lambda group        (TotalNormalLambda reads the group's
//      normal-row lambdas; group rows share bodies anyway -- kept for safety)
// Components are the transitive closure of that relation: rows in DIFFERENT
// components touch disjoint memory, so their relative execution order cannot
// change any float. Each component's rows are listed in ascending (color,
// intra-color partition position) order and split into SEGMENTS at color
// boundaries: executing a component's segments in order, with the rows inside
// one segment in parallel, reproduces the global color sweep's arithmetic on
// that component's data EXACTLY (same ops, same order, same operands) -- which
// is what lets the solve kernel run one component per CUDA block instead of
// the whole row set on a single block, byte-identically.
// ---------------------------------------------------------------------------

struct RowComponentRange {
    uint32_t segment_offset = 0u;  // into RowComponentPartitions::segments
    uint32_t segment_count = 0u;
    uint32_t row_offset = 0u;      // into RowComponentPartitions::row_indices
    uint32_t row_count = 0u;
};

struct RowComponentPartitions {
    // Rows grouped by component; within a component ascending (color, intra-color
    // position). Segments address contiguous same-color runs of row_indices.
    std::vector<uint32_t> row_indices;
    std::vector<RowColorRange> segments;
    std::vector<RowComponentRange> components;

    uint32_t ComponentCount() const {
        return static_cast<uint32_t>(components.size());
    }
};

// `colors` MUST be the partition BuildRowColorPartitions produced for `rows`.
// `sides` / `art_refs` are the optional per-row streams the unified solve
// threads to the kernel (nullptr => that key class contributes no edges, which
// matches the kernel: without the stream the corresponding reaction arm never
// fires, so no shared state exists through it).
RowComponentPartitions BuildRowComponentPartitions(
    const constraint::RowBuffers& rows,
    const RowColorPartitions& colors,
    const constraint::ContactRowSides* sides,
    uint32_t sides_count,
    const constraint::RowArticulationRefs* art_refs,
    uint32_t art_refs_count);

} // namespace nuka::solver::gpu
