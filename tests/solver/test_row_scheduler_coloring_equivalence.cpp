// ---------------------------------------------------------------------------
// Row-coloring O(rows*K) rewrite EQUIVALENCE gate.
//
// row_scheduler.cu was rewritten from O(rows^2) (greedy graph coloring that
// scanned ALL prior rows per row via RowsConflict, plus an O(rows_per_color^2)
// validator) to O(rows*K) using a body->colors map. The single-block GPU solve
// kernel and every solver caller depend on the EXACT partition, so the rewrite
// must be BYTE-IDENTICAL.
//
// This TU keeps a TEST-LOCAL COPY of the OLD quadratic functions
// (*ReferenceQuadratic) and asserts the production functions
// (nuka::solver::gpu::BuildRowColorPartitions / ValidateNoSharedBodiesPerColor)
// produce ARRAY-EQUAL output (row_indices element-wise, color_ranges
// {row_offset,row_count} element-wise, ColorCount) over a battery:
//   - N-disjoint-K-clique synthetic grasp rows at N in {1,8,32,256,1024};
//   - seeded randomized row sets (small body range, including kInvalidBodyIndex
//     on side a only / b only / both, and body_a==body_b), many seeds;
//   - dense single-clique (ColorCount==rows) and all-invalid (ColorCount==1).
//
// Plain host C++ (no CUDA): links nuka_solver (which carries row_scheduler.cu).
// ---------------------------------------------------------------------------

#include "constraint/row.hpp"
#include "constraint/row_buffers.hpp"
#include "solver/gpu/row_scheduler.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

using nuka::constraint::kInvalidBodyIndex;
using nuka::constraint::RowBuffers;
using nuka::solver::gpu::BuildRowColorPartitions;
using nuka::solver::gpu::RowColorPartitions;
using nuka::solver::gpu::ValidateNoSharedBodiesPerColor;

namespace {

// ---------------------------------------------------------------------------
// TEST-LOCAL COPY of the OLD O(rows^2) algorithm -- byte-for-byte the pre-rewrite
// row_scheduler.cu (RowsConflict + BuildRowColorPartitions + ValidateNoShared-
// BodiesPerColor). This is the reference the production rewrite must match.
// ---------------------------------------------------------------------------

bool RowsConflictReference(const RowBuffers& rows, uint32_t lhs, uint32_t rhs) {
    const auto a = rows.BodiesForRow(lhs);
    const auto b = rows.BodiesForRow(rhs);
    const uint32_t lhs_bodies[2] = {a.body_a, a.body_b};
    const uint32_t rhs_bodies[2] = {b.body_a, b.body_b};
    for (const uint32_t lhs_body : lhs_bodies) {
        if (lhs_body == kInvalidBodyIndex) {
            continue;
        }
        for (const uint32_t rhs_body : rhs_bodies) {
            if (lhs_body == rhs_body) {
                return true;
            }
        }
    }
    return false;
}

RowColorPartitions BuildRowColorPartitionsReferenceQuadratic(const RowBuffers& rows) {
    RowColorPartitions partitions;
    const uint32_t row_count = rows.RowCount();
    if (row_count == 0u) {
        return partitions;
    }

    std::vector<uint32_t> row_colors(row_count, 0u);
    uint32_t color_count = 0u;

    for (uint32_t row = 0; row < row_count; ++row) {
        std::vector<uint8_t> used(color_count, 0u);
        for (uint32_t prior = 0; prior < row; ++prior) {
            if (RowsConflictReference(rows, row, prior)) {
                used[row_colors[prior]] = 1u;
            }
        }

        uint32_t color = 0u;
        while (color < color_count && used[color] != 0u) {
            ++color;
        }
        if (color == color_count) {
            ++color_count;
        }
        row_colors[row] = color;
    }

    partitions.color_ranges.resize(color_count);
    for (uint32_t color = 0; color < color_count; ++color) {
        partitions.color_ranges[color].row_offset =
            static_cast<uint32_t>(partitions.row_indices.size());
        for (uint32_t row = 0; row < row_count; ++row) {
            if (row_colors[row] == color) {
                partitions.row_indices.push_back(row);
            }
        }
        partitions.color_ranges[color].row_count =
            static_cast<uint32_t>(partitions.row_indices.size()) -
            partitions.color_ranges[color].row_offset;
    }

    return partitions;
}

bool ValidateNoSharedBodiesPerColorReferenceQuadratic(
    const RowBuffers& rows, const RowColorPartitions& partitions) {
    for (const auto& range : partitions.color_ranges) {
        const uint32_t end = range.row_offset + range.row_count;
        for (uint32_t lhs_index = range.row_offset; lhs_index < end; ++lhs_index) {
            for (uint32_t rhs_index = lhs_index + 1u; rhs_index < end; ++rhs_index) {
                if (RowsConflictReference(rows,
                                          partitions.row_indices[lhs_index],
                                          partitions.row_indices[rhs_index])) {
                    return false;
                }
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// The equivalence assertion: production output == reference output, array-equal,
// and both validators agree on the production partition (and the reference one).
// ---------------------------------------------------------------------------
void ExpectByteIdentical(const RowBuffers& rows, const char* label) {
    const RowColorPartitions prod = BuildRowColorPartitions(rows);
    const RowColorPartitions ref = BuildRowColorPartitionsReferenceQuadratic(rows);

    ASSERT_EQ(prod.ColorCount(), ref.ColorCount()) << label << ": ColorCount differs";
    ASSERT_EQ(prod.row_indices.size(), ref.row_indices.size())
        << label << ": row_indices size differs";
    for (size_t i = 0; i < ref.row_indices.size(); ++i) {
        ASSERT_EQ(prod.row_indices[i], ref.row_indices[i])
            << label << ": row_indices[" << i << "] differs";
    }
    ASSERT_EQ(prod.color_ranges.size(), ref.color_ranges.size())
        << label << ": color_ranges size differs";
    for (size_t i = 0; i < ref.color_ranges.size(); ++i) {
        EXPECT_EQ(prod.color_ranges[i].row_offset, ref.color_ranges[i].row_offset)
            << label << ": color_ranges[" << i << "].row_offset differs";
        EXPECT_EQ(prod.color_ranges[i].row_count, ref.color_ranges[i].row_count)
            << label << ": color_ranges[" << i << "].row_count differs";
    }

    // Both validators must agree, on both the production AND reference partition
    // (they are identical by the assertions above, so the boolean must match too).
    const bool prod_valid = ValidateNoSharedBodiesPerColor(rows, prod);
    const bool ref_valid =
        ValidateNoSharedBodiesPerColorReferenceQuadratic(rows, ref);
    EXPECT_EQ(prod_valid, ref_valid) << label << ": validator boolean differs";
    // A correct greedy coloring is always conflict-free per color.
    EXPECT_TRUE(prod_valid) << label << ": production partition self-conflicts";
}

// ---------------------------------------------------------------------------
// Synthetic N-disjoint-K-clique rows (mirrors the perf test's BuildSyntheticGrasp-
// Rows): per env e, K rows all share body_a = n_envs + e and body_b = e.
// ColorCount == K for every N.
// ---------------------------------------------------------------------------
RowBuffers BuildSyntheticGraspRows(uint32_t n_envs, uint32_t k) {
    RowBuffers rows;
    const nuka::constraint::RowJacobian6 zero_j{};
    for (uint32_t e = 0u; e < n_envs; ++e) {
        const uint32_t finger_key = n_envs + e;
        const uint32_t cup_key = e;
        for (uint32_t f = 0u; f < k; ++f) {
            rows.AddRow(nuka::constraint::Row{}, {finger_key, cup_key}, zero_j, zero_j);
        }
    }
    return rows;
}

// A single body shared by ALL rows -> a dense clique -> ColorCount == row count.
RowBuffers BuildDenseSingleClique(uint32_t row_count) {
    RowBuffers rows;
    const nuka::constraint::RowJacobian6 zero_j{};
    for (uint32_t r = 0u; r < row_count; ++r) {
        rows.AddRow(nuka::constraint::Row{}, {0u, kInvalidBodyIndex}, zero_j, zero_j);
    }
    return rows;
}

// Every row has only invalid bodies -> no conflicts -> ColorCount == 1.
RowBuffers BuildAllInvalid(uint32_t row_count) {
    RowBuffers rows;
    const nuka::constraint::RowJacobian6 zero_j{};
    for (uint32_t r = 0u; r < row_count; ++r) {
        rows.AddRow(nuka::constraint::Row{},
                    {kInvalidBodyIndex, kInvalidBodyIndex}, zero_j, zero_j);
    }
    return rows;
}

// Seeded random rows: small body range so cliques actually form; deliberately
// injects kInvalidBodyIndex on side a only / b only / both, and body_a==body_b.
RowBuffers BuildRandomRows(uint32_t seed, uint32_t row_count, uint32_t body_range) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> body_dist(0u, body_range - 1u);
    // 0..7: shape selector. Weighted toward both-valid but with a meaningful slice
    // of invalid-side and a==b cases (the exact patterns the map can get wrong).
    std::uniform_int_distribution<uint32_t> shape_dist(0u, 9u);

    RowBuffers rows;
    const nuka::constraint::RowJacobian6 zero_j{};
    for (uint32_t r = 0u; r < row_count; ++r) {
        const uint32_t a = body_dist(rng);
        const uint32_t b = body_dist(rng);
        uint32_t body_a = a;
        uint32_t body_b = b;
        switch (shape_dist(rng)) {
            case 0u:  // invalid on side a only
                body_a = kInvalidBodyIndex;
                break;
            case 1u:  // invalid on side b only
                body_b = kInvalidBodyIndex;
                break;
            case 2u:  // both invalid
                body_a = kInvalidBodyIndex;
                body_b = kInvalidBodyIndex;
                break;
            case 3u:
            case 4u:  // body_a == body_b (same valid body on both sides)
                body_b = body_a;
                break;
            default:  // both distinct-ish valid (a,b as drawn)
                break;
        }
        rows.AddRow(nuka::constraint::Row{}, {body_a, body_b}, zero_j, zero_j);
    }
    return rows;
}

}  // namespace

// ---------------------------------------------------------------------------

TEST(RowSchedulerColoringEquivalence, EmptyRows) {
    RowBuffers rows;
    ExpectByteIdentical(rows, "empty");
    const auto prod = BuildRowColorPartitions(rows);
    EXPECT_EQ(prod.ColorCount(), 0u);
    EXPECT_TRUE(prod.row_indices.empty());
}

TEST(RowSchedulerColoringEquivalence, SyntheticGraspCliquesSweep) {
    const uint32_t kNs[] = {1u, 8u, 32u, 256u, 1024u};
    const uint32_t kK = 10u;  // condim=3 grasp: 5 rows/contact x 2 fingertips.
    for (uint32_t n : kNs) {
        const RowBuffers rows = BuildSyntheticGraspRows(n, kK);
        const std::string label = "synthetic_N" + std::to_string(n);
        ExpectByteIdentical(rows, label.c_str());
        // Decisive structural invariant: one color per clique member, reused
        // across all envs -> ColorCount == K for every N.
        const auto prod = BuildRowColorPartitions(rows);
        EXPECT_EQ(prod.ColorCount(), kK) << label << ": expected K colors";
        EXPECT_EQ(rows.RowCount(), n * kK) << label;
    }
}

TEST(RowSchedulerColoringEquivalence, DenseSingleClique) {
    for (uint32_t rc : {1u, 7u, 64u, 300u}) {
        const RowBuffers rows = BuildDenseSingleClique(rc);
        const std::string label = "dense_clique_" + std::to_string(rc);
        ExpectByteIdentical(rows, label.c_str());
        const auto prod = BuildRowColorPartitions(rows);
        EXPECT_EQ(prod.ColorCount(), rc)
            << label << ": dense clique must use one color per row";
    }
}

TEST(RowSchedulerColoringEquivalence, AllInvalidNoConflicts) {
    for (uint32_t rc : {1u, 5u, 128u, 500u}) {
        const RowBuffers rows = BuildAllInvalid(rc);
        const std::string label = "all_invalid_" + std::to_string(rc);
        ExpectByteIdentical(rows, label.c_str());
        const auto prod = BuildRowColorPartitions(rows);
        EXPECT_EQ(prod.ColorCount(), 1u)
            << label << ": no conflicts -> a single color";
    }
}

TEST(RowSchedulerColoringEquivalence, RandomizedBattery) {
    // Many seeds x a few (row_count, body_range) shapes. Small body ranges force
    // cliques; the shape selector guarantees invalid-side + a==b rows appear.
    struct Shape { uint32_t rows; uint32_t body_range; };
    const Shape shapes[] = {
        {64u, 4u}, {200u, 8u}, {300u, 3u}, {256u, 16u}, {500u, 6u}, {400u, 32u},
    };
    bool saw_self_loop = false;
    bool saw_invalid = false;
    for (uint32_t seed = 1u; seed <= 60u; ++seed) {
        for (const Shape& s : shapes) {
            const RowBuffers rows = BuildRandomRows(seed, s.rows, s.body_range);
            const std::string label = "rand_seed" + std::to_string(seed) + "_rows" +
                                      std::to_string(s.rows) + "_range" +
                                      std::to_string(s.body_range);
            ExpectByteIdentical(rows, label.c_str());
            // Confirm the battery actually exercises the trap cases at least once.
            for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
                const auto bp = rows.BodiesForRow(r);
                if (bp.body_a != kInvalidBodyIndex && bp.body_a == bp.body_b) {
                    saw_self_loop = true;
                }
                if (bp.body_a == kInvalidBodyIndex || bp.body_b == kInvalidBodyIndex) {
                    saw_invalid = true;
                }
            }
        }
    }
    EXPECT_TRUE(saw_self_loop) << "battery never produced a body_a==body_b row";
    EXPECT_TRUE(saw_invalid) << "battery never produced an invalid-body row";
}
