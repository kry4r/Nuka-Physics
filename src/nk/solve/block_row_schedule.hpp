#pragma once

#include <algorithm>
#include <cstdint>

#include "core/checked_size.hpp"

namespace nuka::nk {

enum class BlockSolveFailure : uint32_t {
    None, InvalidMass, InvalidRow, InvalidEquation, Factorization,
    InvalidDirection, InvalidPotential, InvalidCandidate
};

enum class BlockSolveFailureEquationColumn : uint32_t {
    Mass, ForceX, ForceY, ForceZ,
    HessianXX, HessianYY, HessianZZ, HessianXY, HessianXZ, HessianYZ,
    SnapshotX, SnapshotY, SnapshotZ, FreeX, FreeY, FreeZ,
    Iteration, DominantRow, DominantPenalty, DominantResponseA, DominantResponseB,
    DominantResidualNormal, DominantDualNormal, DominantCurvatureTrace, Count
};

inline constexpr uint32_t kBlockSolveFailureEquationColumnCount =
    static_cast<uint32_t>(BlockSolveFailureEquationColumn::Count);
static_assert(sizeof(double) == 2u * sizeof(uint32_t), "failure equations require binary64 storage");

// CSR ownership and each block's first failure within a policy step share a checked layout.
struct BlockRowScheduleLayout {
    uint64_t owners;
    uint64_t rows;
    uint64_t incidence_capacity;

    uint64_t CountsWord() const { return 0u; }
    uint64_t OffsetsWord() const { return CheckedAdd(owners, 1u); }
    uint64_t IncidenceWord() const { return CheckedProduct({2u, CheckedAdd(owners, 1u)}); }
    uint64_t ActiveRowsWord() const { return CheckedAdd(IncidenceWord(), incidence_capacity); }
    uint64_t ActiveCountWord() const { return CheckedAdd(ActiveRowsWord(), rows); }
    uint64_t FailureReasonsWord() const { return CheckedAdd(ActiveCountWord(), 1u); }
    uint64_t FailureRowsWord() const { return CheckedAdd(FailureReasonsWord(), owners); }
    uint64_t FailureSubstepsWord() const { return CheckedAdd(FailureRowsWord(), owners); }
    uint64_t FailureEquationsWord() const {
        return CheckedAlignUp(CheckedAdd(FailureSubstepsWord(), owners), 2u);
    }
    uint64_t Words() const {
        return CheckedAdd(FailureEquationsWord(), CheckedProduct({owners, kBlockSolveFailureEquationColumnCount, 2u}));
    }
};

inline BlockRowScheduleLayout MakeBlockRowScheduleLayout(
    uint64_t owners, uint64_t rows, uint32_t max_point_terms) {
    return {owners, rows, CheckedProduct({2u, rows, std::max(1u, max_point_terms)})};
}

}  // namespace nuka::nk
