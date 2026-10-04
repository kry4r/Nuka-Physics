// Independent islands retain ordered row updates and separate real/pseudo velocities.

#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cub/block/block_scan.cuh>

#include "constraint/coulomb_contact.hpp"
#include "nk/material/mpm_constitutive.hpp"
#include <algorithm>
#include <cstring>
#include <limits>

#include "math/cuda_vec_ops.cuh"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/launch_grid.cuh"
#include "phi/backend_cuda/ops/island_schedule.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/union_types.cuh"
#include "phi/backend_cuda/ops/vertex_blocks.cuh"

namespace nuka::phi {

namespace {

using namespace ::nuka::phi::nkops;

namespace mg = ::nuka::math::gpu;

__forceinline__ __device__ float Dot3(math::Vec3 a, math::Vec3 b) {
    return mg::Dot(a, b);
}

// Small impulse increments retain their rounding error across ordered row updates.
__device__ void AddVelocity(float& value, float* error, float delta) {
    if (error == nullptr) { value += delta; return; }
    const float adjusted = __fsub_rn(delta, *error);
    const float sum = __fadd_rn(value, adjusted);
    *error = __fsub_rn(__fsub_rn(sum, value), adjusted);
    value = sum;
}

__device__ void AddVelocity(math::Vec3& value, math::Vec3* error, math::Vec3 delta) {
    AddVelocity(value.x, error != nullptr ? &error->x : nullptr, delta.x);
    AddVelocity(value.y, error != nullptr ? &error->y : nullptr, delta.y);
    AddVelocity(value.z, error != nullptr ? &error->z : nullptr, delta.z);
}

struct VelocityErrorView {
    math::Vec3* body_linear = nullptr;
    math::Vec3* body_angular = nullptr;
    math::Vec3* particle = nullptr;
    math::Vec3* grid = nullptr;
    float* qdot = nullptr;

    __device__ math::Vec3* Point(uint32_t kind, uint32_t index) const {
        math::Vec3* base = kind == kNkSideGrid ? grid : particle;
        return base != nullptr ? base + index : nullptr;
    }
    __device__ math::Vec3* Linear(uint32_t index) const {
        return body_linear != nullptr ? body_linear + index : nullptr;
    }
    __device__ math::Vec3* Angular(uint32_t index) const {
        return body_angular != nullptr ? body_angular + index : nullptr;
    }
};

__device__ void ApplyPointImpulse(const NkRowSide& side, float delta,
                                  PointMassView points, VelocityErrorView error,
                                  uint32_t lane = 0u, uint32_t width = 1u) {
    for (uint32_t i = lane; i < points.Count(side); i += width) {
        const auto term = points.At(side, i);
        const auto* inverse_mass = points.InverseMass(term.kind);
        auto* velocity = points.Velocity(term.kind);
        if (velocity != nullptr && inverse_mass != nullptr && inverse_mass[term.index] > 0.0f) {
            AddVelocity(velocity[term.index], error.Point(term.kind, term.index),
                        points.Respond(term.kind, term.index, term.jacobian,
                                       inverse_mass[term.index], delta));
            points.RecordImpulse(term.kind, term.index, term.jacobian * delta);
        }
    }
}

// Independent warps screen consecutive rows before the ordered updates.
constexpr uint32_t kUnionIslandBlockSize = 64u;
constexpr uint32_t kPairDrivenIslandBlockSize = 32u;
constexpr uint32_t kScalarIslandBlockSize = 32u;
constexpr uint32_t kScalarIslandGridBlocks = 64u;
// An owner shared by more rows of a warp batch is mass-split across the rows solved with it.
// Islands no larger than this keep one thread, which gives the same ordered result.
constexpr uint32_t kSplitShare = 8u;
constexpr uint64_t kNoOwner = ~0ull;
// A colored row spreads at most two owners over each lane of its warp.
constexpr uint32_t kOwnerKeysPerRow = 64u;
constexpr uint32_t kOwnerEmpty = ~0u;
constexpr uint32_t kOwnerGroupKind = 7u;
// Live rows sharing no written state take one color and sweep together across the grid.
// Rows the palette cannot place run in island order instead.
constexpr uint32_t kColorPalette = 128u;
constexpr uint32_t kColorWords = kColorPalette / 32u;
constexpr uint32_t kColorEpochs = 16u;
constexpr uint32_t kColorSlots = kColorPalette * kColorEpochs;
constexpr uint32_t kColorRounds = 256u;
constexpr uint32_t kRecolorPasses = 1u;
constexpr uint32_t kColorBlockSize = 256u;
constexpr uint32_t kColorBlocksPerSm = 2u;
constexpr uint32_t kColorGridLimit = 1024u;
constexpr uint32_t kColorPending = ~0u;
constexpr uint32_t kColorOverflow = ~1u;
constexpr uint32_t kColorChain = ~2u;
// Control words: live and chain-island totals, the color count, then rotating round flags.
constexpr uint32_t kControlLive = 0u;
constexpr uint32_t kControlValidRows = 1u;
constexpr uint32_t kControlChainIslands = 2u;
constexpr uint32_t kControlColors = 3u;
constexpr uint32_t kControlPending = 4u;
constexpr uint32_t kControlChanged = 8u;
constexpr uint32_t kControlOverflow = 12u;
constexpr uint32_t kControlArticRows = 14u;
constexpr uint32_t kControlScheduleChanged = 15u;
constexpr uint32_t kControlHubRows = 16u;
constexpr uint32_t kControlIdleViolations = 17u;
constexpr uint32_t kControlWords = 18u;
// A dynamic body in more active rows than this is a hub: it owns none of them, stays frozen
// within a sweep, and a Schur step couples its rows as it couples an articulation's.
constexpr uint32_t kHubRows = 64u;
constexpr uint32_t kHubTriangle = 21u;
constexpr uint32_t kHubItem = kHubTriangle + 6u;
// Rows read articulation velocity frozen over a sweep; a Schur step per articulation then
// couples them. Its entry sums split into at most this many fixed-order groups.
constexpr uint32_t kSchurGroups = 128u;
constexpr uint32_t kSchurTriangle = kMaxArticulationDof * (kMaxArticulationDof + 1u) / 2u;
constexpr uint32_t kSchurTrianglePerThread =
    (kSchurTriangle + kColorBlockSize - 1u) / kColorBlockSize;
// Right-side rows each lane of the substituting warp holds.
constexpr uint32_t kSchurLaneRows = (kMaxArticulationDof + 31u) / 32u;
// Island roots keep their build flags in cc_parent; the top bit marks a root with live rows.
constexpr uint32_t kIslandNeedsSolve = 1u << 31u;

struct IslandActivityView {
    const uint32_t* sorted_roots = nullptr;
    uint32_t* needs_solve = nullptr;

    __device__ bool Active(const IslandRecord& island) const {
        return needs_solve == nullptr ||
               (needs_solve[sorted_roots[island.seg_off]] & kIslandNeedsSolve) != 0u;
    }
};

// A compact row stores both articulation tiles and one dynamic side.
// Rows with two dynamic sides read their complete NkRow.
struct SlimRow {
    uint32_t flags;
    uint32_t group_local;   // group_first - env_row_base
    uint32_t group_cnt;
    uint32_t code;          // kSlim* bits
    uint32_t dyn_index;
    float rhs, R, lower, upper, mu;
    float jl[3];            // the dynamic side's linear jacobian
    float ja[3];            // ... angular (rigid r x j augment)
    // Environment-local articulation tiles; ~0u denotes a non-articulation side.
    uint32_t a_tile;        // side-A artic env-local tile index (or ~0u).
    uint32_t b_tile;        // side-B artic env-local tile index (or ~0u).
};
static_assert(sizeof(SlimRow) == 18 * sizeof(float), "SlimRow must be 72 B");
constexpr uint32_t kSlimAArt = 1u << 0;
constexpr uint32_t kSlimBArt = 1u << 1;
constexpr uint32_t kSlimHasDyn = 1u << 2;
constexpr uint32_t kSlimDynIsB = 1u << 3;
constexpr uint32_t kSlimDynParticle = 1u << 4;
constexpr uint32_t kSlimDynGrid = 1u << 6;
constexpr uint32_t kSlimDynEndpoint = 1u << 7;
constexpr uint32_t kSlimPointMass = kSlimDynParticle | kSlimDynGrid | kSlimDynEndpoint;
constexpr uint32_t kSlimFallback = 1u << 5;  // two dynamic sides: read NkRow

// Lanes fetch adjacent words once, then broadcast the row fields needed by the solve.
__forceinline__ __device__ uint32_t RowWordWarp(const NkRow* rows, uint32_t slot, uint32_t lane) {
    static_assert(sizeof(NkRow) == 32u * sizeof(uint32_t));
    uint32_t word;
    memcpy(&word, reinterpret_cast<const unsigned char*>(rows + slot) + lane * sizeof(word), sizeof(word));
    return word;
}

__forceinline__ __device__ NkRow BroadcastRowWarp(uint32_t word) {
    NkRow result;
    #pragma unroll
    for (uint32_t i = 0u; i < sizeof(NkRow) / sizeof(word); ++i) {
        const uint32_t value = __shfl_sync(0xffffffffu, word, i);
        memcpy(reinterpret_cast<unsigned char*>(&result) + i * sizeof(word), &value, sizeof(value));
    }
    return result;
}

__forceinline__ __device__ NkRow LoadRowWarp(const NkRow* rows, uint32_t slot, uint32_t lane) {
    return BroadcastRowWarp(RowWordWarp(rows, slot, lane));
}

__device__ NkRowSide SlimPointSide(const SlimRow& row) {
    NkRowSide side;
    side.kind = (row.code & kSlimDynEndpoint) ? kNkSidePointEndpoint :
                (row.code & kSlimDynGrid) ? kNkSideGrid : kNkSideParticle;
    side.index = row.dyn_index;
    side.jlin = {row.jl[0], row.jl[1], row.jl[2]};
    return side;
}

// Union islands cache row data and Jacobians; PairDriven islands read global rows.
// Both allocate velocity tiles, with separate pseudo tiles for the position solve.
inline size_t IslandSharedBytes(uint32_t rows_per_env, uint32_t dof_stride,
                                uint32_t qdot_floats, bool cache_jw, bool pos_pass) {
    if (!cache_jw) return sizeof(float) * qdot_floats * (pos_pass ? 3u : 2u);
    const uint64_t jw = 2ull * rows_per_env * dof_stride;
    const uint64_t slim = sizeof(SlimRow) * rows_per_env;
    return sizeof(float) *
               (2ull * qdot_floats +                                   // velocity and rounding error
                3ull * rows_per_env +                                   // lambda+meff+damping
                jw) +                                                   // J + w (union only)
           slim +                                                       // slim (union only)
           sizeof(uint32_t) * 3ull * rows_per_env;                      // order+segs
}

// Compact row construction retains global row identity and environment-local tiles.
__device__ inline SlimRow MakeSlimRow(const NkRow& row, uint32_t env_row_base,
                                      uint32_t env_artic_base) {
    SlimRow sr;
    sr.flags = row.flags;
    sr.group_local = row.group_first - env_row_base;
    sr.group_cnt = row.group_normal_count;
    sr.rhs = row.rhs;
    sr.R = row.compliance_alpha;
    sr.lower = row.lower;
    sr.upper = row.upper;
    sr.mu = row.mu;
    uint32_t code = 0u;
    sr.a_tile = ~0u;
    sr.b_tile = ~0u;
    if (row.a.kind == kNkSideArtic) {
        code |= kSlimAArt;
        sr.a_tile = (row.a.index >= env_artic_base) ? (row.a.index - env_artic_base) : 0u;
    }
    if (row.b.kind == kNkSideArtic) {
        code |= kSlimBArt;
        sr.b_tile = (row.b.index >= env_artic_base) ? (row.b.index - env_artic_base) : 0u;
    }
    const bool a_dyn = row.a.kind == kNkSideRigid || PointMassView::IsPointSide(row.a.kind);
    const bool b_dyn = row.b.kind == kNkSideRigid || PointMassView::IsPointSide(row.b.kind);
    sr.dyn_index = 0u;
    sr.jl[0] = sr.jl[1] = sr.jl[2] = 0.0f;
    sr.ja[0] = sr.ja[1] = sr.ja[2] = 0.0f;
    if (a_dyn) {
        code |= kSlimHasDyn;
        if (row.a.kind == kNkSideParticle) code |= kSlimDynParticle;
        if (row.a.kind == kNkSideGrid) code |= kSlimDynGrid;
        if (row.a.kind == kNkSidePointEndpoint) code |= kSlimDynEndpoint;
        sr.dyn_index = row.a.index;
        sr.jl[0] = row.a.jlin.x; sr.jl[1] = row.a.jlin.y; sr.jl[2] = row.a.jlin.z;
        sr.ja[0] = row.a.jang.x; sr.ja[1] = row.a.jang.y; sr.ja[2] = row.a.jang.z;
        if (b_dyn) code |= kSlimFallback;
    } else if (b_dyn) {
        code |= kSlimHasDyn | kSlimDynIsB;
        if (row.b.kind == kNkSideParticle) code |= kSlimDynParticle;
        if (row.b.kind == kNkSideGrid) code |= kSlimDynGrid;
        if (row.b.kind == kNkSidePointEndpoint) code |= kSlimDynEndpoint;
        sr.dyn_index = row.b.index;
        sr.jl[0] = row.b.jlin.x; sr.jl[1] = row.b.jlin.y; sr.jl[2] = row.b.jlin.z;
        sr.ja[0] = row.b.jang.x; sr.ja[1] = row.b.jang.y; sr.ja[2] = row.b.jang.z;
    }
    sr.code = code;
    return sr;
}

__device__ void AddBlockVelocity(float& value, float* error,
                                 float normal, float tangent_first, float tangent_second,
                                 math::Vec3 impulse) {
    if (impulse.x != 0.0f) AddVelocity(value, error, normal * impulse.x);
    if (impulse.y != 0.0f) AddVelocity(value, error, tangent_first * impulse.y);
    if (impulse.z != 0.0f) AddVelocity(value, error, tangent_second * impulse.z);
}

__device__ void AddBlockVelocity(math::Vec3& value, math::Vec3* error,
                                 math::Vec3 normal, math::Vec3 tangent_first,
                                 math::Vec3 tangent_second, math::Vec3 impulse) {
    if (impulse.x != 0.0f) AddVelocity(value, error, normal * impulse.x);
    if (impulse.y != 0.0f) AddVelocity(value, error, tangent_first * impulse.y);
    if (impulse.z != 0.0f) AddVelocity(value, error, tangent_second * impulse.z);
}

// Compliant velocity projection preserves side A/B order with cooperative DOF reductions.
// Effective mass and inverse-mass Jacobians are assembled before the solve.
__device__ void ApplySlimImpulse(
    const SlimRow& sr, uint32_t gslot, uint32_t j_row, uint32_t wlane, float delta,
    uint32_t env_artic_base, float* qdot_sh,
    const NkRow* __restrict__ urows,
    const float* J_sh, const float* w_sh, const float* J_b_sh,
    const float* w_b_sh, math::Vec3* body_lin_vel,
    math::Vec3* body_ang_vel, const float* body_inv_mass,
    const math::SymmetricMat3* body_world_inv_inertia, PointMassView point_masses, uint32_t dof_stride,
    VelocityErrorView error = {}, const NkRow* prepared = nullptr) {
    const uint32_t code = sr.code;
    const uint32_t a_tile = (sr.a_tile == ~0u) ? 0u : sr.a_tile;
    const uint32_t b_tile = (sr.b_tile == ~0u) ? 0u : sr.b_tile;
    for (int side = 0; side < 2; ++side) {
        const bool art = side == 0 ? (code & kSlimAArt) != 0u
                                   : (code & kSlimBArt) != 0u;
        const bool dyn = (code & kSlimHasDyn) &&
                         ((side == 1) == ((code & kSlimDynIsB) != 0u));
        if (art) {
            const uint32_t tile = (side == 0) ? a_tile : b_tile;
            const float* w = side == 0
                                 ? w_sh + static_cast<size_t>(j_row) * dof_stride
                                 : ((w_b_sh != nullptr) ? w_b_sh : w_sh) +
                                       static_cast<size_t>(j_row) * dof_stride;
            float* const qd = qdot_sh + static_cast<size_t>(tile) * dof_stride;
            for (uint32_t r = wlane; r < dof_stride; r += 32u)
                AddVelocity(qd[r], error.qdot != nullptr ?
                    error.qdot + static_cast<size_t>(tile) * dof_stride + r : nullptr, w[r] * delta);
        } else if (dyn) {
            if (code & kSlimPointMass) {
                ApplyPointImpulse(SlimPointSide(sr), delta, point_masses, error, wlane, warpSize);
            } else if (wlane == 0u) {
                const float im = body_inv_mass[sr.dyn_index];
                if (im > 0.0f) {
                    const math::Vec3 angular_response = body_world_inv_inertia[sr.dyn_index].Multiply(
                        {sr.ja[0], sr.ja[1], sr.ja[2]});
                    math::Vec3& v = body_lin_vel[sr.dyn_index];
                    math::Vec3& w = body_ang_vel[sr.dyn_index];
                    AddVelocity(v, error.Linear(sr.dyn_index),
                                math::Vec3{sr.jl[0], sr.jl[1], sr.jl[2]} * (im * delta));
                    AddVelocity(w, error.Angular(sr.dyn_index), angular_response * delta);
                }
            }
        } else if ((code & kSlimFallback) && side == 1) {
            const NkRow row = prepared != nullptr ? *prepared : LoadRowWarp(urows, gslot, wlane);
            if (PointMassView::IsPointSide(row.b.kind)) {
                ApplyPointImpulse(row.b, delta, point_masses, error, wlane, warpSize);
            } else if (row.b.kind == kNkSideRigid && wlane == 0u) {
                const float im = body_inv_mass[row.b.index];
                if (im > 0.0f) {
                    const math::Vec3 angular_response = body_world_inv_inertia[row.b.index].Multiply(row.b.jang);
                    math::Vec3& v = body_lin_vel[row.b.index];
                    math::Vec3& w = body_ang_vel[row.b.index];
                    AddVelocity(v, error.Linear(row.b.index), row.b.jlin * (im * delta));
                    AddVelocity(w, error.Angular(row.b.index), angular_response * delta);
                }
            }
        }
        __syncwarp();
    }
}

// A caller that already summed the dynamic point side passes that sum as point_velocity.
template <bool cooperative = false>
__device__ float ComputeSlimRowVelocity(
    const SlimRow& sr, uint32_t gslot, uint32_t j_row,
    const float* J, const float* J_b, const float* qdot,
    const NkRow* __restrict__ urows,
    const math::Vec3* __restrict__ body_lin_vel,
    const math::Vec3* __restrict__ body_ang_vel,
    PointMassView point_masses, uint32_t dof_stride, uint32_t lane = 0u,
    const NkRow* prepared = nullptr, const float* point_velocity = nullptr) {
    const uint32_t code = sr.code;
    const uint32_t a_tile = sr.a_tile == ~0u ? 0u : sr.a_tile;
    const uint32_t b_tile = sr.b_tile == ~0u ? 0u : sr.b_tile;
    float art_jv_a = 0.0f;
    if (code & kSlimAArt) {
        const float* const row_j = J + static_cast<size_t>(j_row) * dof_stride;
        const float* const qd = qdot + static_cast<size_t>(a_tile) * dof_stride;
        if constexpr (cooperative) {
            for (uint32_t r = lane; r < dof_stride; r += warpSize) art_jv_a += row_j[r] * qd[r];
            art_jv_a = WarpSum(art_jv_a);
        } else {
            for (uint32_t r = 0u; r < dof_stride; ++r) art_jv_a += row_j[r] * qd[r];
        }
    }
    float art_jv_b = 0.0f;
    if (code & kSlimBArt) {
        const float* const row_j = (J_b != nullptr ? J_b : J) +
                                   static_cast<size_t>(j_row) * dof_stride;
        const float* const qd = qdot + static_cast<size_t>(b_tile) * dof_stride;
        if constexpr (cooperative) {
            for (uint32_t r = lane; r < dof_stride; r += warpSize) art_jv_b += row_j[r] * qd[r];
            art_jv_b = WarpSum(art_jv_b);
        } else {
            for (uint32_t r = 0u; r < dof_stride; ++r) art_jv_b += row_j[r] * qd[r];
        }
    }
    float dyn_jv = 0.0f;
    if (code & kSlimHasDyn) {
        const math::Vec3 jl{sr.jl[0], sr.jl[1], sr.jl[2]};
        if (code & kSlimPointMass) {
            if (point_velocity != nullptr) dyn_jv = *point_velocity;
            else if constexpr (cooperative) dyn_jv = point_masses.RowVelocityWarp(SlimPointSide(sr), lane);
            else dyn_jv = point_masses.RowVelocity(SlimPointSide(sr));
        } else {
            const math::Vec3 ja{sr.ja[0], sr.ja[1], sr.ja[2]};
            dyn_jv = Dot3(jl, body_lin_vel[sr.dyn_index]) +
                     Dot3(ja, body_ang_vel[sr.dyn_index]);
        }
    }
    float jv = 0.0f;
    if (code & kSlimAArt) jv += art_jv_a;
    else if ((code & kSlimHasDyn) && !(code & kSlimDynIsB)) jv += dyn_jv;
    if (code & kSlimBArt) jv += art_jv_b;
    else if ((code & kSlimHasDyn) && (code & kSlimDynIsB)) jv += dyn_jv;
    if (code & kSlimFallback) {
        NkRow row;
        if (prepared != nullptr) row = *prepared;
        else if constexpr (cooperative) row = LoadRowWarp(urows, gslot, lane);
        else row = urows[gslot];
        if (PointMassView::IsPointSide(row.b.kind)) {
            if constexpr (cooperative) jv += point_masses.RowVelocityWarp(row.b, lane);
            else jv += point_masses.RowVelocity(row.b);
        } else if (row.b.kind == kNkSideRigid) {
            jv += Dot3(row.b.jlin, body_lin_vel[row.b.index]) +
                  Dot3(row.b.jang, body_ang_vel[row.b.index]);
        }
    }
    return jv;
}

// Both tangent rows read the same owners, so each velocity is loaded once.
__device__ math::Vec3 ComputeSlimTangentVelocity(
    const SlimRow& first, const SlimRow& second,
    uint32_t first_slot, uint32_t second_slot, uint32_t first_j, uint32_t second_j,
    const float* J, const float* J_b, const float* qdot, const NkRow* urows,
    const math::Vec3* body_linear, const math::Vec3* body_angular,
    PointMassView points, uint32_t dofs, uint32_t lane,
    const NkRow* prepared = nullptr) {
    float first_velocity = 0.0f;
    float second_velocity = 0.0f;
    for (uint32_t side_index = 0u; side_index < 2u; ++side_index) {
        const bool artic = side_index == 0u ? (first.code & kSlimAArt) != 0u
                                            : (first.code & kSlimBArt) != 0u;
        const bool dynamic = (first.code & kSlimHasDyn) &&
            ((side_index == 1u) == ((first.code & kSlimDynIsB) != 0u));
        if (artic) {
            const uint32_t tile = side_index == 0u ? first.a_tile : first.b_tile;
            const float* source = side_index == 0u || J_b == nullptr ? J : J_b;
            const float* velocity = qdot + static_cast<size_t>(tile) * dofs;
            const float* first_row = source + static_cast<size_t>(first_j) * dofs;
            const float* second_row = source + static_cast<size_t>(second_j) * dofs;
            for (uint32_t k = lane; k < dofs; k += warpSize) {
                const float value = velocity[k];
                first_velocity += first_row[k] * value;
                second_velocity += second_row[k] * value;
            }
        } else if (dynamic) {
            if (first.code & kSlimPointMass) {
                const auto first_side = SlimPointSide(first);
                const auto second_side = SlimPointSide(second);
                for (uint32_t i = lane; i < points.Count(first_side); i += warpSize) {
                    const auto a = points.At(first_side, i);
                    const auto b = points.At(second_side, i);
                    const auto* velocity = points.Velocity(a.kind);
                    if (velocity == nullptr) continue;
                    const math::Vec3 value = velocity[a.index];
                    first_velocity += a.jacobian.Dot(value);
                    second_velocity += b.jacobian.Dot(value);
                }
            } else if (lane == 0u) {
                const uint32_t index = first.dyn_index;
                const math::Vec3 linear = body_linear[index];
                const math::Vec3 angular = body_angular[index];
                first_velocity += math::Vec3{first.jl[0], first.jl[1], first.jl[2]}.Dot(linear) +
                    math::Vec3{first.ja[0], first.ja[1], first.ja[2]}.Dot(angular);
                second_velocity += math::Vec3{second.jl[0], second.jl[1], second.jl[2]}.Dot(linear) +
                    math::Vec3{second.ja[0], second.ja[1], second.ja[2]}.Dot(angular);
            }
        } else if ((first.code & kSlimFallback) && side_index == 1u) {
            const NkRowSide first_side = prepared != nullptr
                ? prepared[1].b : LoadRowWarp(urows, first_slot, lane).b;
            const NkRowSide second_side = prepared != nullptr
                ? prepared[2].b : LoadRowWarp(urows, second_slot, lane).b;
            if (PointMassView::IsPointSide(first_side.kind)) {
                for (uint32_t i = lane; i < points.Count(first_side); i += warpSize) {
                    const auto a = points.At(first_side, i);
                    const auto b = points.At(second_side, i);
                    const auto* velocity = points.Velocity(a.kind);
                    if (velocity == nullptr) continue;
                    const math::Vec3 value = velocity[a.index];
                    first_velocity += a.jacobian.Dot(value);
                    second_velocity += b.jacobian.Dot(value);
                }
            } else if (first_side.kind == kNkSideRigid && lane == 0u) {
                const math::Vec3 linear = body_linear[first_side.index];
                const math::Vec3 angular = body_angular[first_side.index];
                first_velocity += first_side.jlin.Dot(linear) + first_side.jang.Dot(angular);
                second_velocity += second_side.jlin.Dot(linear) + second_side.jang.Dot(angular);
            }
        }
    }
    return {0.0f, WarpSum(first_velocity), WarpSum(second_velocity)};
}

__device__ bool ContactFrictionActive(const NkRow& normal, float response, float damping,
                                     float dt, float velocity, float impulse) {
    if (!(fmaxf(normal.mu, normal.friction_secondary) > 0.0f)) return false;
    const float scale = 1.0f / (1.0f + damping * dt);
    const float residual = normal.rhs * dt * scale - velocity - normal.compliance_alpha * scale * impulse;
    return constraint::ProjectedContactNormal(response, residual, impulse) > 0.0f;
}

__device__ bool ContactFrictionActive(const NkRow& normal, float damping, float dt,
                                     float velocity, float impulse) {
    return ContactFrictionActive(normal, normal.contact_response.xx, damping, dt, velocity, impulse);
}

// Coulomb rows share owners, so their three reactions update each velocity once.
__device__ void ApplySlimContactImpulse(
    const SlimRow& normal, const SlimRow& tangent_first, const SlimRow& tangent_second,
    uint32_t normal_slot, uint32_t tangent_first_slot, uint32_t tangent_second_slot,
    uint32_t normal_j, uint32_t tangent_first_j, uint32_t tangent_second_j,
    uint32_t lane, math::Vec3 impulse, uint32_t env_artic_base, float* qdot,
    const NkRow* urows, const float* minv_j, const float* minv_j_b,
    math::Vec3* body_linear, math::Vec3* body_angular, const float* body_inv_mass,
    const math::SymmetricMat3* body_inv_inertia, PointMassView points,
    uint32_t dofs, VelocityErrorView error, const NkRow* prepared = nullptr) {
    const SlimRow rows[3] = {normal, tangent_first, tangent_second};
    const uint32_t slots[3] = {normal_slot, tangent_first_slot, tangent_second_slot};
    const uint32_t jacobians[3] = {normal_j, tangent_first_j, tangent_second_j};
    for (uint32_t side_index = 0u; side_index < 2u; ++side_index) {
        const bool artic = side_index == 0u ? (normal.code & kSlimAArt) != 0u
                                            : (normal.code & kSlimBArt) != 0u;
        const bool dynamic = (normal.code & kSlimHasDyn) &&
            ((side_index == 1u) == ((normal.code & kSlimDynIsB) != 0u));
        if (artic) {
            const uint32_t tile = side_index == 0u ? normal.a_tile : normal.b_tile;
            float* velocity = qdot + static_cast<size_t>(tile) * dofs;
            const float* source = side_index == 0u || minv_j_b == nullptr ? minv_j : minv_j_b;
            for (uint32_t k = lane; k < dofs; k += warpSize) {
                float value = velocity[k];
                float* destination_error = error.qdot != nullptr ?
                    error.qdot + static_cast<size_t>(tile) * dofs + k : nullptr;
                AddBlockVelocity(value, destination_error,
                    source[static_cast<size_t>(jacobians[0]) * dofs + k],
                    source[static_cast<size_t>(jacobians[1]) * dofs + k],
                    source[static_cast<size_t>(jacobians[2]) * dofs + k], impulse);
                velocity[k] = value;
            }
        } else if (dynamic) {
            if (normal.code & kSlimPointMass) {
                const auto normal_side = SlimPointSide(rows[0]);
                const auto first_side = SlimPointSide(rows[1]);
                const auto second_side = SlimPointSide(rows[2]);
                for (uint32_t i = lane; i < points.Count(normal_side); i += warpSize) {
                    const auto a = points.At(normal_side, i);
                    const auto b = points.At(first_side, i);
                    const auto c = points.At(second_side, i);
                    const float* inverse_mass = points.InverseMass(a.kind);
                    auto* velocity = points.Velocity(a.kind);
                    if (velocity == nullptr || inverse_mass == nullptr || !(inverse_mass[a.index] > 0.0f)) continue;
                    math::Vec3 value = velocity[a.index];
                    const float m = inverse_mass[a.index];
                    AddBlockVelocity(value, error.Point(a.kind, a.index),
                        points.Respond(a.kind, a.index, a.jacobian, m),
                        points.Respond(a.kind, a.index, b.jacobian, m),
                        points.Respond(a.kind, a.index, c.jacobian, m), impulse);
                    points.RecordBlockImpulse(a.kind, a.index, a.jacobian, b.jacobian, c.jacobian, impulse);
                    velocity[a.index] = value;
                }
            } else if (lane == 0u) {
                const uint32_t index = normal.dyn_index;
                const float inverse_mass = body_inv_mass[index];
                if (inverse_mass > 0.0f) {
                    math::Vec3 linear = body_linear[index];
                    math::Vec3 angular = body_angular[index];
                    const math::Vec3 jl[3] = {
                        {rows[0].jl[0], rows[0].jl[1], rows[0].jl[2]},
                        {rows[1].jl[0], rows[1].jl[1], rows[1].jl[2]},
                        {rows[2].jl[0], rows[2].jl[1], rows[2].jl[2]}};
                    const math::Vec3 ja[3] = {
                        {rows[0].ja[0], rows[0].ja[1], rows[0].ja[2]},
                        {rows[1].ja[0], rows[1].ja[1], rows[1].ja[2]},
                        {rows[2].ja[0], rows[2].ja[1], rows[2].ja[2]}};
                    AddBlockVelocity(linear, error.Linear(index), jl[0] * inverse_mass,
                        jl[1] * inverse_mass, jl[2] * inverse_mass, impulse);
                    AddBlockVelocity(angular, error.Angular(index), body_inv_inertia[index].Multiply(ja[0]),
                        body_inv_inertia[index].Multiply(ja[1]),
                        body_inv_inertia[index].Multiply(ja[2]), impulse);
                    body_linear[index] = linear;
                    body_angular[index] = angular;
                }
            }
        } else if ((normal.code & kSlimFallback) && side_index == 1u) {
            const NkRowSide sides[3] = {
                prepared != nullptr ? prepared[0].b : LoadRowWarp(urows, slots[0], lane).b,
                prepared != nullptr ? prepared[1].b : LoadRowWarp(urows, slots[1], lane).b,
                prepared != nullptr ? prepared[2].b : LoadRowWarp(urows, slots[2], lane).b};
            if (PointMassView::IsPointSide(sides[0].kind)) {
                for (uint32_t i = lane; i < points.Count(sides[0]); i += warpSize) {
                    const auto a = points.At(sides[0], i);
                    const auto b = points.At(sides[1], i);
                    const auto c = points.At(sides[2], i);
                    const float* inverse_mass = points.InverseMass(a.kind);
                    auto* velocity = points.Velocity(a.kind);
                    if (velocity == nullptr || inverse_mass == nullptr || !(inverse_mass[a.index] > 0.0f)) continue;
                    math::Vec3 value = velocity[a.index];
                    const float m = inverse_mass[a.index];
                    AddBlockVelocity(value, error.Point(a.kind, a.index),
                        points.Respond(a.kind, a.index, a.jacobian, m),
                        points.Respond(a.kind, a.index, b.jacobian, m),
                        points.Respond(a.kind, a.index, c.jacobian, m), impulse);
                    points.RecordBlockImpulse(a.kind, a.index, a.jacobian, b.jacobian, c.jacobian, impulse);
                    velocity[a.index] = value;
                }
            } else if (sides[0].kind == kNkSideRigid && lane == 0u) {
                const uint32_t index = sides[0].index;
                const float inverse_mass = body_inv_mass[index];
                if (inverse_mass > 0.0f) {
                    math::Vec3 linear = body_linear[index];
                    math::Vec3 angular = body_angular[index];
                    AddBlockVelocity(linear, error.Linear(index), sides[0].jlin * inverse_mass,
                        sides[1].jlin * inverse_mass, sides[2].jlin * inverse_mass, impulse);
                    AddBlockVelocity(angular, error.Angular(index),
                        body_inv_inertia[index].Multiply(sides[0].jang),
                        body_inv_inertia[index].Multiply(sides[1].jang),
                        body_inv_inertia[index].Multiply(sides[2].jang), impulse);
                    body_linear[index] = linear;
                    body_angular[index] = angular;
                }
            }
        }
        __syncwarp();
    }
}

// The three axis rows of a contact side share its state, so one pass reads each
// velocity once; each axis sums in the order a single-axis pass would.
__device__ math::Vec3 ComputePreparedSideVelocities(
    const NkRow* rows, uint32_t side_index, uint32_t j_row, uint32_t j_stride,
    uint32_t env_artic_base, const float* J, const float* J_b, const float* qdot,
    const math::Vec3* body_linear, const math::Vec3* body_angular,
    PointMassView points, uint32_t dofs, uint32_t lane) {
    const NkRowSide sides[3] = {
        side_index == 0u ? rows[0].a : rows[0].b,
        side_index == 0u ? rows[1].a : rows[1].b,
        side_index == 0u ? rows[2].a : rows[2].b};
    float result[3] = {0.0f, 0.0f, 0.0f};
    if (sides[0].kind == kNkSideArtic) {
        const float* source = side_index == 1u && J_b != nullptr ? J_b : J;
        const float* velocity = qdot + static_cast<size_t>(sides[0].index - env_artic_base) * dofs;
        for (uint32_t k = lane; k < dofs; k += warpSize) {
            const float value = velocity[k];
            #pragma unroll
            for (uint32_t axis = 0u; axis < 3u; ++axis)
                result[axis] += source[static_cast<size_t>(j_row + axis * j_stride) * dofs + k] * value;
        }
        return {WarpSum(result[0]), WarpSum(result[1]), WarpSum(result[2])};
    }
    if (PointMassView::IsPointSide(sides[0].kind)) {
        const uint32_t count = points.Count(sides[0]);
        for (uint32_t i = count == 1u ? 0u : lane; i < count; i += warpSize) {
            PointMassView::Contribution terms[3];
            #pragma unroll
            for (uint32_t axis = 0u; axis < 3u; ++axis) terms[axis] = points.At(sides[axis], i);
            const auto* velocity = points.Velocity(terms[0].kind);
            if (velocity == nullptr) continue;
            const math::Vec3 value = velocity[terms[0].index];
            #pragma unroll
            for (uint32_t axis = 0u; axis < 3u; ++axis) result[axis] += terms[axis].jacobian.Dot(value);
        }
        if (count == 1u) return {result[0], result[1], result[2]};
        return {WarpSum(result[0]), WarpSum(result[1]), WarpSum(result[2])};
    }
    if (sides[0].kind == kNkSideRigid && lane == 0u) {
        const math::Vec3 linear = body_linear[sides[0].index];
        const math::Vec3 angular = body_angular[sides[0].index];
        #pragma unroll
        for (uint32_t axis = 0u; axis < 3u; ++axis)
            result[axis] = sides[axis].jlin.Dot(linear) + sides[axis].jang.Dot(angular);
    }
    return {__shfl_sync(0xffffffffu, result[0], 0u), __shfl_sync(0xffffffffu, result[1], 0u),
            __shfl_sync(0xffffffffu, result[2], 0u)};
}

// A side the apply path never writes carries no aliasing: its impulse is skipped,
// so rows meeting only there still commute.
__device__ inline bool PreparedSideWritable(const NkRowSide& s, const float* body_inv_mass) {
    if (s.kind == kNkSideStatic) return false;
    if (s.kind == kNkSideRigid)
        return body_inv_mass == nullptr || body_inv_mass[s.index] > 0.0f;
    return true;
}

__device__ inline uint32_t SideOwnerCount(const NkRowSide& side, PointMassView points,
                                          const float* body_inv_mass) {
    if (!PreparedSideWritable(side, body_inv_mass)) return 0u;
    return PointMassView::IsPointSide(side.kind) ? points.Count(side) : 1u;
}

__device__ inline void SideOwner(const NkRowSide& side, uint32_t term, PointMassView points,
                                 uint32_t& kind, uint32_t& index) {
    kind = side.kind;
    index = side.index;
    if (side.kind != nk::kNkSidePointEndpoint) return;
    const nk::PointEndpointTerm& entry = points.terms[points.ranges[side.index].first + term];
    kind = entry.kind;
    index = entry.index;
}

// A lane row packed for its one-thread solve: every value the sweep reads besides velocities
// and the impulse, in one 128-byte line. Point terms follow side a, then side b.
constexpr uint32_t kLaneRowTerms = 4u;
constexpr uint32_t kLaneGridBit = 1u << 31u;
struct alignas(16) LaneRecord {
    uint32_t slot;
    uint32_t counts;  // side a terms, side b terms << 8
    float implicit_mass;
    float rhs;         // damped rhs * dt
    float compliance;  // damped R
    float lower;
    float upper;
    float meff;
    uint32_t index[kLaneRowTerms];  // kLaneGridBit marks a grid node
    float inverse_mass[kLaneRowTerms];
    float jacobian[kLaneRowTerms][3];
    uint32_t pad[4];
};
static_assert(sizeof(LaneRecord) == 128u, "a lane record fills one line");

// Views of solve_color_scratch. Dense owners index bodies, particles, grid nodes, then row groups.
struct ColorScratch {
    LaneRecord* lane = nullptr;        // per color row position: lane row records
    uint32_t* used = nullptr;          // kColorWords per owner: colors taken this epoch
    uint32_t* tent = nullptr;          // kColorWords per owner: this round's picks
    uint32_t* dup = nullptr;           // kColorWords per owner: picks seen twice
    uint32_t* claim = nullptr;         // per owner: highest priority among duplicate picks
    uint32_t* pos_color = nullptr;     // per live position
    uint32_t* pos_tent = nullptr;      // per live position: (round << 8) | color
    uint32_t* color_rows = nullptr;
    uint32_t* color_start = nullptr;   // bounds of the non-empty colors
    uint32_t* color_cursor = nullptr;  // kColorSlots
    uint32_t* color_tail = nullptr;    // kColorSlots: next free warp-row slot, descending
    uint32_t* color_pair_count = nullptr; // kColorSlots: pair rows per color
    uint32_t* color_pair_tail = nullptr;  // kColorSlots: next free pair-row slot, descending
    uint32_t* color_slot = nullptr;    // per non-empty color: its palette slot
    uint32_t* color_lane = nullptr;    // per non-empty color: end of its lane rows
    uint32_t* color_pair = nullptr;    // per non-empty color: first of its pair rows
    uint32_t* color_position_count = nullptr; // kColorSlots: rows a position sweep moves
    uint32_t* color_position = nullptr; // per non-empty color: rows a position sweep moves
    uint32_t* chain_rows = nullptr;
    uint32_t* chain_excl = nullptr;    // per live position, then the total
    uint32_t* chain_islands = nullptr;
    uint32_t* block_count = nullptr;   // kColorGridLimit
    uint32_t* control = nullptr;       // kControlWords
    float* qdot_error = nullptr;       // per articulation DOF
    uint32_t* artic_rows = nullptr;    // live rows with an articulation side, in live order
    uint32_t* artic_excl = nullptr;    // per live position, then the total
    uint32_t* artic_entries = nullptr; // (slot << 2) | sides, grouped per articulation
    uint32_t* artic_range = nullptr;   // per articulation: first and end entry
    float* artic_start = nullptr;      // per slot axis: impulse the articulation already holds
    float* artic_pending = nullptr;    // per slot axis: Schur correction owed to the other sides
    float* artic_step = nullptr;       // per articulation DOF: the Schur step
    float* artic_partial = nullptr;    // per group item: upper triangle of S, then J^T dlambda
    float* artic_summed = nullptr;     // per articulation: the group items summed
    uint32_t* hub_count = nullptr;     // per body: active rows naming it
    float* owner_inv_mass = nullptr;   // per body: inverse mass, zero for a hub
    uint32_t* hub_rows = nullptr;      // live rows with a hub side, in live order
    uint32_t* hub_excl = nullptr;      // per live position, then the total
    uint32_t* hub_entries = nullptr;   // (slot << 2) | sides, grouped per hub
    uint32_t* hub_range = nullptr;     // per body: first and end entry
    float* hub_start = nullptr;        // per slot axis: impulse the hub already holds
    float* hub_system = nullptr;       // per body: upper triangle of S, then J^T dlambda
    float* hub_step = nullptr;         // per body: the Schur step
    uint32_t bodies = 0u;
    uint32_t particles = 0u;
    uint32_t grid = 0u;
    uint32_t rows = 0u;
    uint32_t artic_dofs = 0u;
    uint32_t articulations = 0u;
    uint32_t dofs = 0u;
};

__host__ __device__ inline uint32_t TriangleSize(uint32_t n) { return n * (n + 1u) / 2u; }

// Group items of one Schur phase; every item stores a triangle and a vector.
__host__ __device__ inline uint32_t SchurGroups(uint32_t articulations, uint32_t blocks) {
    if (articulations == 0u) return 0u;
    const uint32_t groups = blocks / articulations;
    return groups < 1u ? 1u : (groups > kSchurGroups ? kSchurGroups : groups);
}

// Lays the scratch out from base; a null base only counts the words.
uint64_t BindColorScratch(const SolveRowsBlockIslandParams& p, uint32_t* base, ColorScratch* out) {
    const uint64_t rows = uint64_t{p.rows_per_env} * p.env_count;
    const uint64_t owners = uint64_t{p.total_body_count} + p.total_particle_count +
                            p.total_grid_count + rows;
    const uint64_t items =
        uint64_t{p.articulation_count} * SchurGroups(p.articulation_count, kColorGridLimit);
    uint64_t words = 0u;
    auto take = [&](uint64_t count) {
        uint32_t* const at = base != nullptr ? base + words : nullptr;
        words += count;
        return at;
    };
    auto take_float = [&](uint64_t count) { return reinterpret_cast<float*>(take(count)); };
    ColorScratch s;
    static_assert(sizeof(LaneRecord) % sizeof(uint32_t) == 0u);
    s.lane = reinterpret_cast<LaneRecord*>(take(rows * (sizeof(LaneRecord) / sizeof(uint32_t))));
    s.used = take(kColorWords * owners);
    s.tent = take(kColorWords * owners);
    s.dup = take(kColorWords * owners);
    s.claim = take(owners);
    s.pos_color = take(rows);
    s.pos_tent = take(rows);
    s.color_rows = take(rows);
    s.color_start = take(kColorSlots + 1u);
    s.color_cursor = take(kColorSlots);
    s.color_tail = take(kColorSlots);
    s.color_pair_count = take(kColorSlots);
    s.color_pair_tail = take(kColorSlots);
    s.color_slot = take(kColorSlots);
    s.color_lane = take(kColorSlots);
    s.color_pair = take(kColorSlots);
    s.color_position_count = take(kColorSlots);
    s.color_position = take(kColorSlots);
    s.chain_rows = take(rows);
    s.chain_excl = take(rows + 1u);
    s.chain_islands = take(rows);
    s.block_count = take(kColorGridLimit);
    s.control = take(kControlWords);
    s.qdot_error = take_float(uint64_t{p.articulation_count} * p.max_dof);
    s.artic_rows = take(rows);
    s.artic_excl = take(rows + 1u);
    s.artic_entries = take(2u * rows);
    s.artic_range = take(2u * uint64_t{p.articulation_count});
    s.artic_start = take_float(3u * rows);
    s.artic_pending = take_float(3u * rows);
    s.artic_step = take_float(uint64_t{p.articulation_count} * p.max_dof);
    s.artic_partial = take_float(items * (TriangleSize(p.max_dof) + p.max_dof));
    s.artic_summed =
        take_float(uint64_t{p.articulation_count} * (TriangleSize(p.max_dof) + p.max_dof));
    s.hub_count = take(p.total_body_count);
    s.owner_inv_mass = take_float(p.total_body_count);
    s.hub_rows = take(rows);
    s.hub_excl = take(rows + 1u);
    s.hub_entries = take(2u * rows);
    s.hub_range = take(2u * uint64_t{p.total_body_count});
    s.hub_start = take_float(3u * rows);
    s.hub_system = take_float(uint64_t{kHubItem} * p.total_body_count);
    s.hub_step = take_float(6u * uint64_t{p.total_body_count});
    s.bodies = p.total_body_count;
    s.particles = p.total_particle_count;
    s.grid = p.total_grid_count;
    s.rows = static_cast<uint32_t>(rows);
    s.artic_dofs = p.articulation_count * p.max_dof;
    s.articulations = p.articulation_count;
    s.dofs = p.max_dof;
    if (out != nullptr) *out = s;
    return words;
}

__device__ inline uint32_t DenseOwner(const ColorScratch& s, uint32_t kind, uint32_t index) {
    uint32_t base = 0u;
    uint32_t limit = 0u;
    if (kind == kNkSideRigid) {
        limit = s.bodies;
    } else if (kind == kNkSideParticle) {
        base = s.bodies;
        limit = s.particles;
    } else if (kind == kNkSideGrid) {
        base = s.bodies + s.particles;
        limit = s.grid;
    } else if (kind == kOwnerGroupKind) {
        base = s.bodies + s.particles + s.grid;
        limit = s.rows;
    }
    return index < limit ? base + index : kOwnerEmpty;
}

// Lane j holds a row's owners j and j + 32. Owner 0 is the row's friction group; a row with
// an owner outside the dense ranges is reported unpacked.
struct LaneOwners {
    uint32_t first = kOwnerEmpty;
    uint32_t second = kOwnerEmpty;
    bool packed = true;
};

// Articulation sides read frozen velocities during a sweep, so they are never row owners.
__device__ bool HasArticulationSide(const NkRow& row) {
    return row.a.kind == kNkSideArtic || row.b.kind == kNkSideArtic;
}

__device__ inline bool IsHubSide(const NkRowSide& side, const ColorScratch& s) {
    return side.kind == kNkSideRigid && side.index < s.bodies && s.hub_count[side.index] > kHubRows;
}

// Articulation rows between heavy sides run in the chain: frozen rows corrected only on their
// active set leave sliding friction uncoupled. Point-mass rows color with their own masses.
__device__ inline bool IsSequentialArticRow(const NkRow& row, const ColorScratch& s) {
    if (IsHubSide(row.a, s) || IsHubSide(row.b, s)) return false;
    const bool a = row.a.kind == kNkSideArtic, b = row.b.kind == kNkSideArtic;
    if (a == b) return a;
    return !PointMassView::IsPointSide((a ? row.b : row.a).kind);
}

// Rows the articulation Schur step couples; a block normal carries its tangents' impulses.
__device__ inline bool IsSchurRow(const NkRow& row, const ColorScratch& s) {
    return (row.flags & nk::nk_row_flags::kActive) &&
           !(row.flags & nk::nk_row_flags::kBlockTangent) && HasArticulationSide(row) &&
           !IsSequentialArticRow(row, s);
}

// Rows a hub's Schur step couples; a block normal carries its tangents' impulses.
__device__ inline bool IsHubRow(const NkRow& row, const ColorScratch& s) {
    return (row.flags & nk::nk_row_flags::kActive) &&
           !(row.flags & nk::nk_row_flags::kBlockTangent) &&
           (IsHubSide(row.a, s) || IsHubSide(row.b, s));
}

// Rows a position sweep projects; the others return at once, so a color without any is skipped.
__device__ inline bool IsPositionRow(const NkRow& row) {
    return (row.flags & nk::nk_row_flags::kActive) &&
           !(row.flags & (nk::nk_row_flags::kFriction | nk::nk_row_flags::kBlockTangent |
                          nk::nk_row_flags::kVelocityOnly));
}

// A material block is solved by half a warp, so a warp takes two blocks of a color at once.
constexpr uint32_t kPairRowWidth = 16u;
__device__ inline bool IsPairRow(const NkRow& row) {
    return (row.flags & nk::nk_row_flags::kActive) &&
           (row.flags & nk::nk_row_flags::kMaterialBlock);
}

// A scalar row whose sides hold at most kLaneRowTerms point terms in all is solved by one
// thread from its lane record.
__device__ inline bool IsLaneRow(const NkRow& row, PointMassView points) {
    constexpr uint32_t kWide = nk::nk_row_flags::kBlockNormal | nk::nk_row_flags::kBlockTangent |
                               nk::nk_row_flags::kMaterialBlock | nk::nk_row_flags::kFriction;
    if (!(row.flags & nk::nk_row_flags::kActive) || (row.flags & kWide)) return false;
    const auto terms = [&](const NkRowSide& side) {
        return PointMassView::IsPointSide(side.kind) ? points.Count(side)
             : side.kind == kNkSideStatic ? 0u : kLaneRowTerms + 1u;
    };
    return terms(row.a) + terms(row.b) <= kLaneRowTerms;
}

__device__ inline void PackLaneRecord(const NkRow& row, uint32_t slot, float meff, float damping,
                                      float dt, PointMassView points, LaneRecord& out) {
    const float damping_scale = 1.0f / (1.0f + damping * dt);
    out.slot = slot;
    out.meff = meff;
    out.implicit_mass = meff / (1.0f - meff * row.compliance_alpha * (1.0f - damping_scale));
    out.rhs = row.rhs * dt * damping_scale;
    out.compliance = row.compliance_alpha * damping_scale;
    out.lower = row.lower;
    out.upper = row.upper;
    uint32_t term = 0u;
    uint32_t counts = 0u;
    for (uint32_t side_index = 0u; side_index < 2u; ++side_index) {
        const NkRowSide& side = side_index == 0u ? row.a : row.b;
        const uint32_t count = PointMassView::IsPointSide(side.kind) ? points.Count(side) : 0u;
        for (uint32_t i = 0u; i < count; ++i, ++term) {
            const auto c = points.At(side, i);
            const float* inverse_mass = points.InverseMass(c.kind);
            const bool moves = points.Velocity(c.kind) != nullptr;
            out.index[term] = c.index | (c.kind == kNkSideGrid ? kLaneGridBit : 0u);
            out.inverse_mass[term] = moves && inverse_mass != nullptr ? inverse_mass[c.index] : 0.0f;
            out.jacobian[term][0] = moves ? c.jacobian.x : 0.0f;
            out.jacobian[term][1] = moves ? c.jacobian.y : 0.0f;
            out.jacobian[term][2] = moves ? c.jacobian.z : 0.0f;
        }
        counts |= count << (8u * side_index);
    }
    out.counts = counts;
}

// SolveDynamicRowScalar for a packed lane row: every velocity loads at once, and a repeated
// point continues from its earlier term. Returns whether the impulse moved past tolerance.
__device__ bool SolveLaneRecord(const LaneRecord& r, float* __restrict__ lambda,
                                PointMassView points, VelocityErrorView error,
                                bool apply_cached, float tolerance) {
    const uint32_t a_count = r.counts & 0xffu;
    const uint32_t count = a_count + (r.counts >> 8u);
    math::Vec3* target[kLaneRowTerms] = {};
    math::Vec3* rounding[kLaneRowTerms] = {};
    math::Vec3 velocity[kLaneRowTerms];
    math::Vec3 carry[kLaneRowTerms];
    #pragma unroll
    for (uint32_t k = 0u; k < kLaneRowTerms; ++k) {
        velocity[k] = carry[k] = math::Vec3{};
        if (k >= count) continue;
        const uint32_t kind = (r.index[k] & kLaneGridBit) ? kNkSideGrid : kNkSideParticle;
        const uint32_t index = r.index[k] & ~kLaneGridBit;
        math::Vec3* const base = points.Velocity(kind);
        target[k] = base != nullptr ? base + index : nullptr;
        rounding[k] = error.Point(kind, index);
        if (target[k] != nullptr) velocity[k] = *target[k];
        if (rounding[k] != nullptr) carry[k] = *rounding[k];
    }
    const float old = lambda[r.slot];
    float delta = old;
    if (!apply_cached) {
        float side_a = 0.0f, side_b = 0.0f;
        #pragma unroll
        for (uint32_t k = 0u; k < kLaneRowTerms; ++k) {
            if (k >= count) continue;
            const math::Vec3 j{r.jacobian[k][0], r.jacobian[k][1], r.jacobian[k][2]};
            if (k < a_count) side_a += j.Dot(velocity[k]);
            else side_b += j.Dot(velocity[k]);
        }
        const float jv = side_a + side_b;
        const float increment = r.implicit_mass * (r.rhs - jv - r.compliance * old);
        const float next = fminf(fmaxf(old + increment, r.lower), r.upper);
        lambda[r.slot] = next;
        delta = next - old;
    }
    if (delta == 0.0f) return false;
    bool dirty[kLaneRowTerms] = {};
    #pragma unroll
    for (uint32_t k = 0u; k < kLaneRowTerms; ++k) {
        if (k >= count || target[k] == nullptr || !(r.inverse_mass[k] > 0.0f)) continue;
        const math::Vec3 j{r.jacobian[k][0], r.jacobian[k][1], r.jacobian[k][2]};
        const uint32_t kind = (r.index[k] & kLaneGridBit) ? kNkSideGrid : kNkSideParticle;
        const math::Vec3 impulse =
            points.Respond(kind, r.index[k] & ~kLaneGridBit, j, r.inverse_mass[k], delta);
        points.RecordImpulse(kind, r.index[k] & ~kLaneGridBit, j * delta);
        bool merged = false;
        #pragma unroll
        for (uint32_t m = 0u; m < k; ++m) {
            if (merged || r.index[m] != r.index[k]) continue;
            AddVelocity(velocity[m], rounding[m] != nullptr ? &carry[m] : nullptr, impulse);
            dirty[m] = merged = true;
        }
        if (merged) continue;
        AddVelocity(velocity[k], rounding[k] != nullptr ? &carry[k] : nullptr, impulse);
        dirty[k] = true;
    }
    #pragma unroll
    for (uint32_t k = 0u; k < kLaneRowTerms; ++k) {
        if (!dirty[k]) continue;
        *target[k] = velocity[k];
        if (rounding[k] != nullptr) *rounding[k] = carry[k];
    }
    const float step = fabsf(delta);
    return tolerance <= 0.0f ? true : (r.meff > 0.0f ? step / r.meff : step) > tolerance;
}

__device__ LaneOwners LoadLaneOwners(const NkRow& row, PointMassView points,
                                     const float* body_inv_mass, const ColorScratch& s,
                                     uint32_t lane) {
    const uint32_t a_owners = row.a.kind == kNkSideArtic
        ? 0u : SideOwnerCount(row.a, points, body_inv_mass);
    const uint32_t b_owners = row.b.kind == kNkSideArtic
        ? 0u : SideOwnerCount(row.b, points, body_inv_mass);
    const uint32_t owners = 1u + a_owners + b_owners;
    LaneOwners result;
    result.packed = owners <= kOwnerKeysPerRow;
    for (uint32_t j = lane; result.packed && j < owners; j += warpSize) {
        uint32_t kind = kOwnerGroupKind;
        uint32_t index = row.group_first;
        if (j != 0u) {
            const bool side_a = j - 1u < a_owners;
            SideOwner(side_a ? row.a : row.b, side_a ? j - 1u : j - 1u - a_owners, points,
                      kind, index);
        }
        const uint32_t owner = DenseOwner(s, kind, index);
        if (owner == kOwnerEmpty) result.packed = false;
        if (j < warpSize) result.first = owner;
        else result.second = owner;
    }
    result.packed = __all_sync(0xffffffffu, result.packed);
    return result;
}

template <typename Visit>
__device__ inline void VisitLaneOwners(const LaneOwners& owners, Visit&& visit) {
    if (owners.first != kOwnerEmpty) visit(owners.first);
    if (owners.second != kOwnerEmpty) visit(owners.second);
}

// While live rows are colored, the lane record area holds each lane row's owners column by
// column over positions, then its owner count; zero marks a row colored by a warp.
constexpr uint32_t kLaneOwners = kLaneRowTerms + 1u;
struct LaneOwnerTable {
    uint32_t* words = nullptr;
    uint32_t stride = 0u;

    __device__ explicit LaneOwnerTable(const ColorScratch& s)
        : words(reinterpret_cast<uint32_t*>(s.lane)), stride(s.rows) {}
    __device__ uint32_t& Count(uint32_t position) const {
        return words[size_t{kLaneOwners} * stride + position];
    }
    __device__ uint32_t& Owner(uint32_t k, uint32_t position) const {
        return words[size_t{k} * stride + position];
    }
};
static_assert(sizeof(LaneRecord) >= (kLaneOwners + 1u) * sizeof(uint32_t));

// One thread walks the owners LoadLaneOwners spreads over a warp, in the same order; an owner
// outside the dense ranges is visited as kOwnerEmpty.
template <typename Visit>
__device__ inline void VisitRowOwners(const NkRow& row, PointMassView points,
                                      const float* body_inv_mass, const ColorScratch& s,
                                      Visit&& visit) {
    const uint32_t a_owners = row.a.kind == kNkSideArtic
        ? 0u : SideOwnerCount(row.a, points, body_inv_mass);
    const uint32_t b_owners = row.b.kind == kNkSideArtic
        ? 0u : SideOwnerCount(row.b, points, body_inv_mass);
    for (uint32_t j = 0u; j < 1u + a_owners + b_owners; ++j) {
        uint32_t kind = kOwnerGroupKind;
        uint32_t index = row.group_first;
        if (j != 0u) {
            const bool side_a = j - 1u < a_owners;
            SideOwner(side_a ? row.a : row.b, side_a ? j - 1u : j - 1u - a_owners, points,
                      kind, index);
        }
        visit(DenseOwner(s, kind, index));
    }
}

// A bijective integer mix, so distinct row slots keep distinct coloring priorities.
__device__ inline uint32_t MixSlot(uint32_t x) {
    x = (x ^ 61u) ^ (x >> 16u);
    x *= 9u;
    x ^= x >> 4u;
    x *= 0x27d4eb2du;
    x ^= x >> 15u;
    return x;
}

// The first color free at every owner, scanning the palette cyclically from start.
__device__ inline uint32_t FirstFreeColor(const uint32_t (&taken)[kColorWords], uint32_t start) {
    const uint32_t first_word = start / 32u;
    const uint32_t shift = start % 32u;
    for (uint32_t k = 0u; k <= kColorWords; ++k) {
        const uint32_t word = (first_word + k) % kColorWords;
        uint32_t free = ~taken[word];
        if (k == 0u) free &= ~0u << shift;
        if (k == kColorWords) free &= (1u << shift) - 1u;
        if (free != 0u) return word * 32u + static_cast<uint32_t>(__ffs(static_cast<int>(free))) - 1u;
    }
    return kColorPalette;
}

__device__ inline void StageRowWarp(const NkRow* rows, uint32_t slot, NkRow* staged, uint32_t lane) {
    uint32_t word;
    memcpy(&word, reinterpret_cast<const unsigned char*>(rows + slot) + lane * sizeof(word), sizeof(word));
    memcpy(reinterpret_cast<unsigned char*>(staged) + lane * sizeof(word), &word, sizeof(word));
    __syncwarp();
}

__device__ void ApplyPreparedContactSide(
    const NkRow* rows, uint32_t side_index, uint32_t j_row, uint32_t j_stride,
    uint32_t env_artic_base,
    uint32_t lane, math::Vec3 impulse, float* qdot, const float* minv_j,
    const float* minv_j_b, math::Vec3* body_linear, math::Vec3* body_angular,
    const float* body_inv_mass, const math::SymmetricMat3* body_inv_inertia,
    PointMassView points, uint32_t dofs, VelocityErrorView error) {
    const NkRowSide sides[3] = {
        side_index == 0u ? rows[0].a : rows[0].b,
        side_index == 0u ? rows[1].a : rows[1].b,
        side_index == 0u ? rows[2].a : rows[2].b};
    const NkRowSide& normal = sides[0];
    if (normal.kind == kNkSideArtic) {
        const uint32_t tile = normal.index - env_artic_base;
        float* velocity = qdot + static_cast<size_t>(tile) * dofs;
        const float* source = side_index == 1u && minv_j_b != nullptr ? minv_j_b : minv_j;
        for (uint32_t k = lane; k < dofs; k += warpSize) {
            float value = velocity[k];
            float* destination_error = error.qdot != nullptr
                ? error.qdot + static_cast<size_t>(tile) * dofs + k : nullptr;
            AddBlockVelocity(value, destination_error,
                source[static_cast<size_t>(j_row) * dofs + k],
                source[static_cast<size_t>(j_row + j_stride) * dofs + k],
                source[static_cast<size_t>(j_row + 2u * j_stride) * dofs + k], impulse);
            velocity[k] = value;
        }
    } else if (PointMassView::IsPointSide(normal.kind)) {
        for (uint32_t i = lane; i < points.Count(normal); i += warpSize) {
            const auto a = points.At(sides[0], i);
            const auto b = points.At(sides[1], i);
            const auto c = points.At(sides[2], i);
            const float* inverse_mass = points.InverseMass(a.kind);
            auto* velocity = points.Velocity(a.kind);
            if (velocity == nullptr || inverse_mass == nullptr || !(inverse_mass[a.index] > 0.0f))
                continue;
            math::Vec3 value = velocity[a.index];
            const float m = inverse_mass[a.index];
            AddBlockVelocity(value, error.Point(a.kind, a.index),
                points.Respond(a.kind, a.index, a.jacobian, m),
                points.Respond(a.kind, a.index, b.jacobian, m),
                points.Respond(a.kind, a.index, c.jacobian, m), impulse);
            points.RecordBlockImpulse(a.kind, a.index, a.jacobian, b.jacobian, c.jacobian, impulse);
            velocity[a.index] = value;
        }
    } else if (normal.kind == kNkSideRigid && lane == 0u) {
        const uint32_t index = normal.index;
        const float inverse_mass = body_inv_mass[index];
        if (inverse_mass > 0.0f) {
            math::Vec3 linear = body_linear[index];
            math::Vec3 angular = body_angular[index];
            AddBlockVelocity(linear, error.Linear(index), sides[0].jlin * inverse_mass,
                sides[1].jlin * inverse_mass, sides[2].jlin * inverse_mass, impulse);
            AddBlockVelocity(angular, error.Angular(index),
                body_inv_inertia[index].Multiply(sides[0].jang),
                body_inv_inertia[index].Multiply(sides[1].jang),
                body_inv_inertia[index].Multiply(sides[2].jang), impulse);
            body_linear[index] = linear;
            body_angular[index] = angular;
        }
    }
    __syncwarp();
}

struct PreparedContactStep {
    math::Vec3 impulse;
    math::Vec3 delta;
    bool changed;
    bool significant;
};

template <bool precomputed_response>
__device__ PreparedContactStep ContactBlockStep(
    const NkRow& normal, const NkRow& first, const NkRow& second,
    math::Vec3 old, float damping, math::Vec3 velocity, float dt,
    float tangent_response, float vel_tolerance) {
    const float old_normal = old.x;
    const float old_first = old.y;
    const float old_second = old.z;
    const float damping_scale = 1.0f / (1.0f + damping * dt);
    const math::Vec3 residual{
        normal.rhs * dt * damping_scale - velocity.x -
            normal.compliance_alpha * damping_scale * old_normal,
        first.rhs * dt - velocity.y - first.compliance_alpha * old_first,
        second.rhs * dt - velocity.z - second.compliance_alpha * old_second};
    PreparedContactStep step;
    if constexpr (precomputed_response) {
        step.impulse = constraint::ProjectedCoulombStepWithResponse(
            normal.contact_response, residual, {old_normal, old_first, old_second},
            normal.mu, normal.friction_secondary, tangent_response);
    } else {
        step.impulse = constraint::ProjectedCoulombStep(
            normal.contact_response, residual, {old_normal, old_first, old_second},
            normal.mu, normal.friction_secondary);
    }
    step.delta = {step.impulse.x - old_normal, step.impulse.y - old_first,
                  step.impulse.z - old_second};
    step.changed = step.delta.x != 0.0f || step.delta.y != 0.0f || step.delta.z != 0.0f;
    // An impulse correction times its diagonal response is the velocity it moves.
    const float response = fmaxf(normal.contact_response.xx, tangent_response);
    const float largest = fmaxf(fabsf(step.delta.x), fmaxf(fabsf(step.delta.y), fabsf(step.delta.z)));
    step.significant = vel_tolerance > 0.0f ? (largest * response > vel_tolerance) : step.changed;
    return step;
}

// One lane's term of a point side, read once so the apply reuses what the velocity pass loaded.
struct PointLaneTerm {
    uint32_t kind = nk::kNkSideStatic;
    uint32_t index = 0u;
    math::Vec3 jacobian[3];
    math::Vec3 velocity;
    math::Vec3 error;
    float inverse_mass = 0.0f;
};

// Axis velocities of a point side whose terms fit one per lane; sums match the uncached pass.
template <uint32_t axes>
__device__ void LoadPointSideVelocities(
    const NkRowSide (&sides)[axes], nk::PointEndpointRange range, PointMassView points,
    VelocityErrorView error, uint32_t lane, PointLaneTerm& term, float (&result)[axes]) {
    #pragma unroll
    for (uint32_t axis = 0u; axis < axes; ++axis) result[axis] = 0.0f;
    const uint32_t i = range.count == 1u ? 0u : lane;
    if (i < range.count) {
        if (sides[0].kind == nk::kNkSidePointEndpoint) {
            const nk::PointEndpointTerm& entry = points.terms[range.first + i];
            term.kind = entry.kind;
            term.index = entry.index;
            #pragma unroll
            for (uint32_t axis = 0u; axis < axes; ++axis)
                term.jacobian[axis] = entry.TransposeMultiply(sides[axis].jlin);
        } else {
            term.kind = sides[0].kind;
            term.index = sides[0].index;
            #pragma unroll
            for (uint32_t axis = 0u; axis < axes; ++axis) term.jacobian[axis] = sides[axis].jlin;
        }
        const math::Vec3* velocity = points.Velocity(term.kind);
        const float* inverse_mass = points.InverseMass(term.kind);
        const math::Vec3* compensation = error.Point(term.kind, term.index);
        if (velocity != nullptr) {
            term.velocity = velocity[term.index];
            #pragma unroll
            for (uint32_t axis = 0u; axis < axes; ++axis)
                result[axis] += term.jacobian[axis].Dot(term.velocity);
        }
        if (inverse_mass != nullptr) term.inverse_mass = inverse_mass[term.index];
        if (compensation != nullptr) term.error = *compensation;
    }
    if (range.count == 1u) return;
    #pragma unroll
    for (uint32_t axis = 0u; axis < axes; ++axis) result[axis] = WarpSum(result[axis]);
}

// A scalar impulse on a lane's cached term, rounded as ApplyPointImpulse rounds it.
__device__ void ApplyPointLaneImpulse(const PointLaneTerm& term, uint32_t count, uint32_t lane,
                                      float delta, PointMassView points) {
    math::Vec3* velocity = points.Velocity(term.kind);
    if (lane >= count || velocity == nullptr || points.InverseMass(term.kind) == nullptr ||
        !(term.inverse_mass > 0.0f))
        return;
    math::Vec3 value = term.velocity;
    AddVelocity(value, nullptr,
                points.Respond(term.kind, term.index, term.jacobian[0], term.inverse_mass, delta));
    points.RecordImpulse(term.kind, term.index, term.jacobian[0] * delta);
    velocity[term.index] = value;
}

// No concurrent row writes this lane's term, so its loaded state is still current.
__device__ void ApplyPointSideTerm(const PointLaneTerm& term, uint32_t count, uint32_t lane,
                                   math::Vec3 impulse, PointMassView points,
                                   VelocityErrorView error) {
    math::Vec3* velocity = points.Velocity(term.kind);
    if (lane >= count || velocity == nullptr || points.InverseMass(term.kind) == nullptr ||
        !(term.inverse_mass > 0.0f))
        return;
    math::Vec3* compensation = error.Point(term.kind, term.index);
    math::Vec3 value = term.velocity;
    math::Vec3 residue = term.error;
    AddBlockVelocity(value, compensation != nullptr ? &residue : nullptr,
        points.Respond(term.kind, term.index, term.jacobian[0], term.inverse_mass),
        points.Respond(term.kind, term.index, term.jacobian[1], term.inverse_mass),
        points.Respond(term.kind, term.index, term.jacobian[2], term.inverse_mass), impulse);
    points.RecordBlockImpulse(term.kind, term.index, term.jacobian[0], term.jacobian[1],
                              term.jacobian[2], impulse);
    velocity[term.index] = value;
    if (compensation != nullptr) *compensation = residue;
}

// A Schur correction left on a deferred row reaches its other sides at the row's next visit.
__device__ void ApplyContactPending(
    const NkRow* rows, float* pending, uint32_t lane, math::Vec3* body_linear,
    math::Vec3* body_angular, const float* body_inv_mass,
    const math::SymmetricMat3* body_inv_inertia, PointMassView points, VelocityErrorView error) {
    const math::Vec3 delta{pending[0], pending[1], pending[2]};
    if (delta.x == 0.0f && delta.y == 0.0f && delta.z == 0.0f) return;
    __syncwarp();
    if (lane == 0u) pending[0] = pending[1] = pending[2] = 0.0f;
    for (uint32_t side = 0u; side < 2u; ++side)
        if ((side == 0u ? rows[0].a.kind : rows[0].b.kind) != kNkSideArtic)
            ApplyPreparedContactSide(rows, side, 0u, 0u, 0u, lane, delta, nullptr, nullptr, nullptr,
                body_linear, body_angular, body_inv_mass, body_inv_inertia, points, 0u, error);
}

// One warp solves a prepared contact; axis k reads J row j_row + k * j_stride. With a pending
// slot the articulation sides stay frozen and only the other sides are written.
__device__ bool SolvePreparedContactBlockWarp(
    uint32_t gslot, uint32_t env_artic_base, uint32_t lane, const NkRow* urows, NkRow* rows,
    float* lambda, const float* row_damping, const float* J, const float* minv_j,
    const float* J_b, const float* minv_j_b, float* qdot,
    math::Vec3* body_linear, math::Vec3* body_angular,
    const float* body_inv_mass, const math::SymmetricMat3* body_inv_inertia,
    PointMassView points, uint32_t dofs, float dt, float vel_tolerance,
    VelocityErrorView error, uint32_t j_row, uint32_t j_stride, bool tangents_staged,
    float* articulation_pending = nullptr) {
    const uint32_t count = rows[0].group_normal_count;
    const uint32_t point = (gslot - rows[0].group_first) % count;
    const uint32_t normal_slot = rows[0].group_first + point;
    const uint32_t tangent_first_slot = normal_slot + count;
    const uint32_t tangent_second_slot = normal_slot + 2u * count;
    const NkRowSide side_a = rows[0].a;
    const bool cached = PointMassView::IsPointSide(side_a.kind);
    const bool deferred = articulation_pending != nullptr;
    nk::PointEndpointRange range{0u, 1u};
    if (cached && side_a.kind == nk::kNkSidePointEndpoint) range = points.ranges[side_a.index];
    const math::Vec3 old{lambda[normal_slot], lambda[tangent_first_slot],
                         lambda[tangent_second_slot]};
    const float damping = row_damping[normal_slot];
    if (!tangents_staged) {
        uint32_t first_word;
        uint32_t second_word;
        memcpy(&first_word, reinterpret_cast<const unsigned char*>(urows + tangent_first_slot) +
                            lane * sizeof(first_word), sizeof(first_word));
        memcpy(&second_word, reinterpret_cast<const unsigned char*>(urows + tangent_second_slot) +
                             lane * sizeof(second_word), sizeof(second_word));
        memcpy(reinterpret_cast<unsigned char*>(rows + 1) + lane * sizeof(first_word), &first_word,
               sizeof(first_word));
        memcpy(reinterpret_cast<unsigned char*>(rows + 2) + lane * sizeof(second_word),
               &second_word, sizeof(second_word));
        __syncwarp();
    }
    if (deferred)
        ApplyContactPending(rows, articulation_pending, lane, body_linear, body_angular,
                            body_inv_mass, body_inv_inertia, points, error);
    const NkRowSide sides_a[3] = {rows[0].a, rows[1].a, rows[2].a};
    PointLaneTerm term;
    const bool fits = cached && range.count <= warpSize;
    float point_velocity[3];
    if (fits) LoadPointSideVelocities(sides_a, range, points, error, lane, term, point_velocity);
    const math::Vec3 a = fits
        ? math::Vec3{point_velocity[0], point_velocity[1], point_velocity[2]}
        : ComputePreparedSideVelocities(rows, 0u, j_row, j_stride, env_artic_base, J, J_b, qdot,
              body_linear, body_angular, points, dofs, lane);
    const math::Vec3 b = ComputePreparedSideVelocities(rows, 1u, j_row, j_stride, env_artic_base,
        J, J_b, qdot, body_linear, body_angular, points, dofs, lane);
    const float tangent_response = constraint::CoulombTangentSpectralResponse(
        rows[0].contact_response, rows[0].mu, rows[0].friction_secondary);
    const PreparedContactStep step = ContactBlockStep<true>(rows[0], rows[1], rows[2],
        old, damping, {a.x + b.x, a.y + b.y, a.z + b.z}, dt, tangent_response, vel_tolerance);
    __syncwarp();
    if (lane == 0u) {
        lambda[normal_slot] = step.impulse.x;
        lambda[tangent_first_slot] = step.impulse.y;
        lambda[tangent_second_slot] = step.impulse.z;
    }
    if (step.changed) {
        if (fits) {
            ApplyPointSideTerm(term, range.count, lane, step.delta, points, error);
            __syncwarp();
        } else if (!deferred || rows[0].a.kind != kNkSideArtic) {
            ApplyPreparedContactSide(rows, 0u, j_row, j_stride, env_artic_base, lane, step.delta,
                qdot, minv_j, minv_j_b, body_linear, body_angular, body_inv_mass,
                body_inv_inertia, points, dofs, error);
        }
        if (!deferred || rows[0].b.kind != kNkSideArtic)
            ApplyPreparedContactSide(rows, 1u, j_row, j_stride, env_artic_base, lane, step.delta,
                qdot, minv_j, minv_j_b, body_linear, body_angular, body_inv_mass,
                body_inv_inertia, points, dofs, error);
    }
    return step.significant;
}

// Every lane of every staged row can name a distinct point mass, so a batch always fits.
constexpr uint32_t kChainCacheEntries = kColorBlockSize;
constexpr unsigned long long kChainCacheEmpty = ~0ull;
constexpr uint32_t kChainLaneTerm = 1u << 31u;
constexpr uint32_t kChainLaneVelocity = 1u << 30u;
constexpr uint32_t kChainLaneError = 1u << 29u;
constexpr uint32_t kChainLaneEntry = 0xffffu;
constexpr uint32_t kChainNoPointSide = 2u;
constexpr uint32_t kChainRowScalars = 5u;
static_assert((kChainCacheEntries & (kChainCacheEntries - 1u)) == 0u, "entries probe by mask");

// One chain batch's point masses stay shared until every ordered row has applied its update.
// Per row: point side, old impulses, damping, tangent response.
struct ChainPointCache {
    unsigned long long* key = nullptr;
    float* velocity = nullptr;
    float* error = nullptr;
    float* inverse_mass = nullptr;
    uint32_t* lane = nullptr;
    float* jacobian = nullptr;
    uint32_t* shape = nullptr;
    float* scalar = nullptr;
};

// Stages a block normal's point side and old impulses; false when a point side is not cacheable.
__device__ bool StageChainPoints(const NkRow* rows, uint32_t gslot, uint32_t row, uint32_t lane,
                                 ChainPointCache cache, PointMassView points,
                                 VelocityErrorView error, const float* lambda,
                                 const float* row_damping) {
    const NkRow& normal = rows[0];
    const bool point_a = PointMassView::IsPointSide(normal.a.kind);
    const bool point_b = PointMassView::IsPointSide(normal.b.kind);
    if (normal.flags & nk::nk_row_flags::kBlockTangent) return true;
    if (!(normal.flags & nk::nk_row_flags::kBlockNormal)) return !point_a && !point_b;
    if (point_a && point_b) return false;
    const uint32_t side = point_a ? 0u : (point_b ? 1u : kChainNoPointSide);
    const NkRowSide sides[3] = {
        side == 0u ? rows[0].a : rows[0].b,
        side == 0u ? rows[1].a : rows[1].b,
        side == 0u ? rows[2].a : rows[2].b};
    const uint32_t count = side == kChainNoPointSide ? 0u : points.Count(sides[0]);
    if (count > warpSize) return false;
    uint32_t word = 0u;
    if (lane < count) {
        uint32_t kind = sides[0].kind;
        uint32_t index = sides[0].index;
        math::Vec3 jacobian[3] = {sides[0].jlin, sides[1].jlin, sides[2].jlin};
        if (kind == nk::kNkSidePointEndpoint) {
            const nk::PointEndpointTerm& entry = points.terms[points.ranges[index].first + lane];
            kind = entry.kind;
            index = entry.index;
            #pragma unroll
            for (uint32_t axis = 0u; axis < 3u; ++axis)
                jacobian[axis] = entry.TransposeMultiply(sides[axis].jlin);
        }
        const unsigned long long key = (static_cast<unsigned long long>(kind) << 32u) | index;
        uint32_t at = MixSlot(index ^ (kind << 28u)) & (kChainCacheEntries - 1u);
        unsigned long long prior = atomicCAS(cache.key + at, kChainCacheEmpty, key);
        while (prior != kChainCacheEmpty && prior != key) {
            at = (at + 1u) & (kChainCacheEntries - 1u);
            prior = atomicCAS(cache.key + at, kChainCacheEmpty, key);
        }
        const math::Vec3* velocity = points.Velocity(kind);
        const float* inverse_mass = points.InverseMass(kind);
        const math::Vec3* compensation = error.Point(kind, index);
        if (prior == kChainCacheEmpty) {
            const math::Vec3 value = velocity != nullptr ? velocity[index] : math::Vec3{};
            const math::Vec3 residue = compensation != nullptr ? *compensation : math::Vec3{};
            cache.velocity[3u * at] = value.x;
            cache.velocity[3u * at + 1u] = value.y;
            cache.velocity[3u * at + 2u] = value.z;
            cache.error[3u * at] = residue.x;
            cache.error[3u * at + 1u] = residue.y;
            cache.error[3u * at + 2u] = residue.z;
            cache.inverse_mass[at] = velocity != nullptr && inverse_mass != nullptr
                ? inverse_mass[index] : 0.0f;
        }
        word = at | kChainLaneTerm | (velocity != nullptr ? kChainLaneVelocity : 0u) |
               (compensation != nullptr ? kChainLaneError : 0u);
        float* const staged = cache.jacobian + 9u * (row * warpSize + lane);
        #pragma unroll
        for (uint32_t axis = 0u; axis < 3u; ++axis) {
            staged[3u * axis] = jacobian[axis].x;
            staged[3u * axis + 1u] = jacobian[axis].y;
            staged[3u * axis + 2u] = jacobian[axis].z;
        }
    }
    cache.lane[row * warpSize + lane] = word;
    if (lane == 0u) {
        const uint32_t normal_count = normal.group_normal_count;
        const uint32_t normal_slot = normal.group_first + (gslot - normal.group_first) % normal_count;
        float* const scalar = cache.scalar + row * kChainRowScalars;
        scalar[0] = lambda[normal_slot];
        scalar[1] = lambda[normal_slot + normal_count];
        scalar[2] = lambda[normal_slot + 2u * normal_count];
        scalar[3] = row_damping[normal_slot];
        scalar[4] = constraint::CoulombTangentSpectralResponse(
            normal.contact_response, normal.mu, normal.friction_secondary);
        cache.shape[row] = count | (side << 16u);
    }
    return true;
}

__device__ inline math::Vec3 ChainLaneJacobian(const ChainPointCache& cache, uint32_t term,
                                               uint32_t axis) {
    const float* staged = cache.jacobian + 9u * term + 3u * axis;
    return {staged[0], staged[1], staged[2]};
}

// A staged chain contact: its point side and old impulses come from the batch cache, and the
// sums, step and side order match SolvePreparedContactBlockWarp for the same staged responses.
__device__ bool SolveChainContactBlockWarp(
    uint32_t gslot, uint32_t row, uint32_t env_artic_base, uint32_t lane, const NkRow* rows,
    const ChainPointCache& cache, float* lambda, const float* J, const float* minv_j,
    const float* J_b, const float* minv_j_b, float* qdot, math::Vec3* body_linear,
    math::Vec3* body_angular, const float* body_inv_mass,
    const math::SymmetricMat3* body_inv_inertia, PointMassView points, uint32_t dofs, float dt,
    float vel_tolerance, VelocityErrorView error, uint32_t j_row) {
    const uint32_t shape = cache.shape[row];
    const uint32_t count = shape & 0xffffu;
    const uint32_t point_side = shape >> 16u;
    const uint32_t term = row * warpSize + (count == 1u ? 0u : lane);
    const uint32_t word = cache.lane[term];
    const uint32_t at = word & kChainLaneEntry;
    const float* scalar = cache.scalar + row * kChainRowScalars;
    math::Vec3 side_velocity[2];
    #pragma unroll
    for (uint32_t side = 0u; side < 2u; ++side) {
        if (side != point_side) {
            side_velocity[side] = ComputePreparedSideVelocities(rows, side, j_row, 1u,
                env_artic_base, J, J_b, qdot, body_linear, body_angular, points, dofs, lane);
            continue;
        }
        float result[3] = {0.0f, 0.0f, 0.0f};
        if (word & kChainLaneVelocity) {
            const math::Vec3 value{cache.velocity[3u * at], cache.velocity[3u * at + 1u],
                                   cache.velocity[3u * at + 2u]};
            #pragma unroll
            for (uint32_t axis = 0u; axis < 3u; ++axis)
                result[axis] += ChainLaneJacobian(cache, term, axis).Dot(value);
        }
        side_velocity[side] = count == 1u
            ? math::Vec3{result[0], result[1], result[2]}
            : math::Vec3{WarpSum(result[0]), WarpSum(result[1]), WarpSum(result[2])};
    }
    const math::Vec3 a = side_velocity[0];
    const math::Vec3 b = side_velocity[1];
    const PreparedContactStep step = ContactBlockStep<true>(rows[0], rows[1], rows[2],
        {scalar[0], scalar[1], scalar[2]}, scalar[3], {a.x + b.x, a.y + b.y, a.z + b.z}, dt,
        scalar[4], vel_tolerance);
    __syncwarp();
    if (lane == 0u) {
        const uint32_t normal_count = rows[0].group_normal_count;
        const uint32_t normal_slot =
            rows[0].group_first + (gslot - rows[0].group_first) % normal_count;
        lambda[normal_slot] = step.impulse.x;
        lambda[normal_slot + normal_count] = step.impulse.y;
        lambda[normal_slot + 2u * normal_count] = step.impulse.z;
    }
    if (!step.changed) return step.significant;
    #pragma unroll
    for (uint32_t side = 0u; side < 2u; ++side) {
        if (side != point_side) {
            // Deferred articulation sides take the impulse at the Schur commit instead.
            if ((side == 0u ? rows[0].a.kind : rows[0].b.kind) == kNkSideArtic &&
                (side == 0u ? minv_j : minv_j_b) == nullptr) continue;
            ApplyPreparedContactSide(rows, side, j_row, 1u, env_artic_base, lane, step.delta,
                qdot, minv_j, minv_j_b, body_linear, body_angular, body_inv_mass,
                body_inv_inertia, points, dofs, error);
            continue;
        }
        const float inverse_mass =
            lane < count && (word & kChainLaneTerm) ? cache.inverse_mass[at] : 0.0f;
        if (inverse_mass > 0.0f) {
            const bool compensated = (word & kChainLaneError) != 0u;
            math::Vec3 value{cache.velocity[3u * at], cache.velocity[3u * at + 1u],
                             cache.velocity[3u * at + 2u]};
            math::Vec3 residue{cache.error[3u * at], cache.error[3u * at + 1u],
                               cache.error[3u * at + 2u]};
            const uint32_t kind = static_cast<uint32_t>(cache.key[at] >> 32u);
            const uint32_t index = static_cast<uint32_t>(cache.key[at]);
            AddBlockVelocity(value, compensated ? &residue : nullptr,
                points.Respond(kind, index, ChainLaneJacobian(cache, term, 0u), inverse_mass),
                points.Respond(kind, index, ChainLaneJacobian(cache, term, 1u), inverse_mass),
                points.Respond(kind, index, ChainLaneJacobian(cache, term, 2u), inverse_mass),
                step.delta);
            points.RecordBlockImpulse(kind, index, ChainLaneJacobian(cache, term, 0u),
                                      ChainLaneJacobian(cache, term, 1u),
                                      ChainLaneJacobian(cache, term, 2u), step.delta);
            cache.velocity[3u * at] = value.x;
            cache.velocity[3u * at + 1u] = value.y;
            cache.velocity[3u * at + 2u] = value.z;
            if (compensated) {
                cache.error[3u * at] = residue.x;
                cache.error[3u * at + 1u] = residue.y;
                cache.error[3u * at + 2u] = residue.z;
            }
        }
        __syncwarp();
    }
    return step.significant;
}

// Cacheable batches access each point exclusively through the cache during their row sweep.
__device__ void CommitChainPoints(const ChainPointCache& cache, PointMassView points,
                                  VelocityErrorView error) {
    for (uint32_t at = threadIdx.x; at < kChainCacheEntries; at += blockDim.x) {
        const unsigned long long key = cache.key[at];
        if (key == kChainCacheEmpty || !(cache.inverse_mass[at] > 0.0f)) continue;
        const uint32_t kind = static_cast<uint32_t>(key >> 32u);
        const uint32_t index = static_cast<uint32_t>(key);
        points.Velocity(kind)[index] = {cache.velocity[3u * at], cache.velocity[3u * at + 1u],
                                       cache.velocity[3u * at + 2u]};
        math::Vec3* residue = error.Point(kind, index);
        if (residue != nullptr)
            *residue = {cache.error[3u * at], cache.error[3u * at + 1u], cache.error[3u * at + 2u]};
    }
}

// The projected impulse of an active row's next step, and the velocity that step moves.
struct IdleRowStep {
    float impulse;
    float velocity;
};

__device__ IdleRowStep IdleRowStepWarp(
    const NkRow* urows, const NkRow& row, uint32_t slot, uint32_t env_row_base,
    uint32_t env_artic_base, const float* lambda, const float* row_meff, const float* row_damping,
    const float* chain_jacobian, const float* chain_jacobian_b, const float* qdot,
    const math::Vec3* body_lin_vel, const math::Vec3* body_ang_vel,
    PointMassView point_masses, uint32_t dof_stride, float dt, uint32_t lane) {
    const SlimRow sr = MakeSlimRow(row, env_row_base, env_artic_base);
    const float jv = ComputeSlimRowVelocity<true>(sr, slot, slot,
        chain_jacobian, chain_jacobian_b, qdot, urows, body_lin_vel,
        body_ang_vel, point_masses, dof_stride, lane, &row);
    const float old_impulse = lambda[slot];
    const float damping_scale = 1.0f / (1.0f + row_damping[slot] * dt);
    const float residual = row.rhs * dt * damping_scale - jv -
                           row.compliance_alpha * damping_scale * old_impulse;
    if (row.flags & nk::nk_row_flags::kBlockNormal) {
        const float impulse =
            constraint::ProjectedContactNormal(row.contact_response.xx, residual, old_impulse);
        return {impulse, fabsf(impulse - old_impulse) * row.contact_response.xx};
    }
    const float effective_mass = row_meff[slot];
    const float implicit_mass = effective_mass /
        (1.0f - effective_mass * row.compliance_alpha * (1.0f - damping_scale));
    const float impulse =
        fminf(fmaxf(old_impulse + implicit_mass * residual, row.lower), row.upper);
    const float step = fabsf(impulse - old_impulse);
    return {impulse, effective_mass > 0.0f ? step / effective_mass : step};
}

// Velocity test of an active row whose own and block impulses are all zero.
__device__ bool RowVelocityNeedsSolveWarp(
    const NkRow* urows, const NkRow& row, uint32_t slot, uint32_t env_row_base,
    uint32_t env_artic_base, const float* lambda, const float* row_meff, const float* row_damping,
    const float* chain_jacobian, const float* chain_jacobian_b, const float* qdot,
    const math::Vec3* body_lin_vel, const math::Vec3* body_ang_vel,
    PointMassView point_masses, uint32_t dof_stride, float dt, uint32_t lane) {
    return IdleRowStepWarp(urows, row, slot, env_row_base, env_artic_base, lambda, row_meff,
        row_damping, chain_jacobian, chain_jacobian_b, qdot, body_lin_vel, body_ang_vel,
        point_masses, dof_stride, dt, lane).impulse != 0.0f;
}

__device__ bool RowNeedsVelocitySolveWarp(
    const NkRow* urows, uint32_t slot, uint32_t env_row_base, uint32_t env_artic_base,
    const float* lambda, const float* row_meff, const float* row_damping,
    const float* chain_jacobian, const float* chain_jacobian_b, const float* qdot,
    const math::Vec3* body_lin_vel, const math::Vec3* body_ang_vel,
    PointMassView point_masses, uint32_t dof_stride, float dt, uint32_t lane,
    const NkRow* prepared = nullptr) {
    const uint32_t flags = prepared != nullptr ? prepared->flags : urows[slot].flags;
    if (!(flags & nk::nk_row_flags::kActive)) return false;
    if (lambda[slot] != 0.0f ||
        (flags & (nk::nk_row_flags::kFriction | nk::nk_row_flags::kMaterialBlock |
                  nk::nk_row_flags::kVertexBlock))) return true;
    const NkRow row = prepared != nullptr ? *prepared : LoadRowWarp(urows, slot, lane);
    if ((row.flags & nk::nk_row_flags::kBlockNormal) &&
        (lambda[slot + row.group_normal_count] != 0.0f ||
         lambda[slot + 2u * row.group_normal_count] != 0.0f)) return true;
    return RowVelocityNeedsSolveWarp(urows, row, slot, env_row_base, env_artic_base, lambda,
        row_meff, row_damping, chain_jacobian, chain_jacobian_b, qdot, body_lin_vel, body_ang_vel,
        point_masses, dof_stride, dt, lane);
}

template <bool cooperative>
__device__ void UpdateSpeculativePenetration(
    uint32_t slot, uint32_t env_row_base, uint32_t env_artic_base,
    float* row_penetration, const NkRow* urows,
    const float* chain_jacobian, const float* chain_jacobian_b, const float* qdot,
    const math::Vec3* body_linear, const math::Vec3* body_angular,
    PointMassView points, uint32_t dof_stride, float dt, uint32_t lane = 0u) {
    if (!(urows[slot].flags & nk::nk_row_flags::kSpeculative)) return;
    NkRow row;
    if constexpr (cooperative) row = LoadRowWarp(urows, slot, lane);
    else row = urows[slot];
    const SlimRow slim = MakeSlimRow(row, env_row_base, env_artic_base);
    const float velocity = ComputeSlimRowVelocity<cooperative>(slim, slot, slot,
        chain_jacobian, chain_jacobian_b, qdot, urows, body_linear, body_angular,
        points, dof_stride, lane);
    if (lane == 0u) row_penetration[slot] = fmaxf(row_penetration[slot] - dt * velocity, 0.0f);
}

struct ScalarStepResult {
    float impulse;
    float delta;
};

__device__ __forceinline__ float ScalarRowIncrement(const SlimRow& row, float effective_mass,
                                                     float damping, float old_impulse,
                                                     float jv, float dt) {
    const float rhs_v = row.rhs * dt;
    const float damping_scale = 1.0f / (1.0f + damping * dt);
    const float implicit_mass = effective_mass /
        (1.0f - effective_mass * row.R * (1.0f - damping_scale));
    return implicit_mass * (rhs_v * damping_scale - jv -
                            row.R * damping_scale * old_impulse);
}

__device__ __forceinline__ ScalarStepResult ScalarRowStep(float old_impulse,
                                                            float increment, float lower,
                                                            float upper) {
    const float impulse = fminf(fmaxf(old_impulse + increment, lower), upper);
    return {impulse, impulse - old_impulse};
}

struct PositionStepResult {
    float impulse;
    float delta;
};

__device__ PositionStepResult PositionRowStep(float depth, float effective_mass,
                                              float old_impulse, float jv, float beta,
                                              float slop, float dt, float max_velocity) {
    const float bias = fminf(beta * fmaxf(depth - slop, 0.0f) / dt, max_velocity);
    const float impulse = fmaxf(old_impulse + effective_mass * (bias - jv), 0.0f);
    return {impulse, impulse - old_impulse};
}

// Factors the active rows of a stress block, deviator rows first and the head last, so the
// leading five rows alone factor the deviator; inactive rows become identity.
__device__ __forceinline__ void FactorMaterialBlock(const float (&A)[nk::kMpmStressRowsPerCell]
                                                                  [nk::kMpmStressRowsPerCell],
                                                    uint32_t rows,
                                                    float (&L)[nk::kMpmStressRowsPerCell]
                                                              [nk::kMpmStressRowsPerCell],
                                                    float (&inverse_pivot)
                                                        [nk::kMpmStressRowsPerCell]) {
    constexpr uint32_t n = nk::kMpmStressRowsPerCell;
    #pragma unroll
    for (uint32_t i = 0u; i < n; ++i) {
        const uint32_t p = (i + 1u) % n;
        #pragma unroll
        for (uint32_t j = 0u; j <= i; ++j) {
            const uint32_t q = (j + 1u) % n;
            L[i][j] = ((rows >> p) & (rows >> q) & 1u) ? A[p][q] : (i == j ? 1.0f : 0.0f);
        }
    }
    #pragma unroll
    for (uint32_t j = 0u; j < n; ++j) {
        float diagonal = L[j][j];
        #pragma unroll
        for (uint32_t k = 0u; k < j; ++k) diagonal -= L[j][k] * L[j][k];
        inverse_pivot[j] = rsqrtf(fmaxf(diagonal, 1.0e-30f));
        #pragma unroll
        for (uint32_t i = j + 1u; i < n; ++i) {
            float value = L[i][j];
            #pragma unroll
            for (uint32_t k = 0u; k < j; ++k) value -= L[i][k] * L[j][k];
            L[i][j] = value * inverse_pivot[j];
        }
    }
}

// Solves the leading `count` factored rows for b in block order; other entries return 0.
template <uint32_t count>
__device__ __forceinline__ void SolveMaterialFactor(const float (&L)[nk::kMpmStressRowsPerCell]
                                                                  [nk::kMpmStressRowsPerCell],
                                                    const float (&inverse_pivot)
                                                        [nk::kMpmStressRowsPerCell],
                                                    const float (&b)[nk::kMpmStressRowsPerCell],
                                                    uint32_t rows,
                                                    float (&x)[nk::kMpmStressRowsPerCell]) {
    constexpr uint32_t n = nk::kMpmStressRowsPerCell;
    float y[n];
    #pragma unroll
    for (uint32_t i = 0u; i < count; ++i) {
        const uint32_t p = (i + 1u) % n;
        float value = (rows >> p) & 1u ? b[p] : 0.0f;
        #pragma unroll
        for (uint32_t k = 0u; k < i; ++k) value -= L[i][k] * y[k];
        y[i] = value * inverse_pivot[i];
    }
    #pragma unroll
    for (uint32_t ii = count; ii > 0u; --ii) {
        const uint32_t i = ii - 1u;
        float value = y[i];
        #pragma unroll
        for (uint32_t k = i + 1u; k < count; ++k) value -= L[k][i] * y[k];
        y[i] = value * inverse_pivot[i];
    }
    #pragma unroll
    for (uint32_t i = 0u; i < n; ++i) {
        const uint32_t p = (i + 1u) % n;
        x[p] = i < count && ((rows >> p) & 1u) ? y[i] : 0.0f;
    }
}

// Sums over the kWidth lanes of mask; every lane receives the total.
template <uint32_t kWidth>
__device__ __forceinline__ float GroupSum(float value, uint32_t mask) {
    #pragma unroll
    for (uint32_t offset = kWidth / 2u; offset > 0u; offset /= 2u)
        value += __shfl_xor_sync(mask, value, offset, kWidth);
    return value;
}

// The kWidth lanes of mask solve a cell's stress block as one 6x6 system, then bound the pressure
// and scale the deviator into its yield cone; lane counts from 0 within the group.
template <uint32_t kWidth>
__device__ bool SolveMaterialBlockGroup(uint32_t slot, const NkRow& head, const NkRow* urows,
                                        float* lambda, PointMassView points, float dt,
                                        bool apply_cached, float vel_tolerance,
                                        VelocityErrorView error, uint32_t lane, uint32_t mask) {
    constexpr uint32_t n = nk::kMpmStressRowsPerCell;
    // Rows of one color share no point, so the velocities a lane reads stay its own until the
    // apply; each lane holds its first terms in registers and revisits only terms past them.
    constexpr uint32_t kHeld = (48u + kWidth - 1u) / kWidth;
    const uint32_t count = points.Count(head.a);
    float old[n], compliance[n], rhs[n];
    #pragma unroll
    for (uint32_t k = 0u; k < n; ++k) {
        old[k] = lambda[slot + k];
        compliance[k] = k == 0u ? head.compliance_alpha : urows[slot + k].compliance_alpha;
        rhs[k] = k == 0u ? head.rhs : urows[slot + k].rhs;
    }
    PointMassView::Contribution held[kHeld];
    math::Vec3 held_velocity[kHeld], held_error[kHeld];
    float held_mass[kHeld];
    #pragma unroll
    for (uint32_t c = 0u; c < kHeld; ++c) {
        const uint32_t i = lane + c * kWidth;
        held[c] = i < count ? points.At(head.a, i)
                            : PointMassView::Contribution{nk::kNkSideGrid, ~0u, {}};
    }
    #pragma unroll
    for (uint32_t c = 0u; c < kHeld; ++c) {
        const float* inverse_mass = points.InverseMass(held[c].kind);
        const math::Vec3* velocity = points.Velocity(held[c].kind);
        const math::Vec3* residue = error.Point(held[c].kind, 0u);
        const bool live = held[c].index != ~0u && velocity != nullptr;
        held_mass[c] = live && inverse_mass != nullptr ? fmaxf(inverse_mass[held[c].index], 0.0f)
                                                       : 0.0f;
        held_velocity[c] = live ? velocity[held[c].index] : math::Vec3{};
        held_error[c] = live && residue != nullptr ? residue[held[c].index] : math::Vec3{};
    }
    const auto apply = [&](const float (&delta)[n]) {
        float tensor[6];
        nk::material::MpmStressBlockTensor(delta, tensor);
        const math::SymmetricMat3 step{tensor[0], tensor[1], tensor[2], tensor[3], tensor[4],
                                       tensor[5]};
        #pragma unroll
        for (uint32_t c = 0u; c < kHeld; ++c) {
            if (!(held_mass[c] > 0.0f)) continue;
            math::Vec3* residue = error.Point(held[c].kind, held[c].index);
            math::Vec3 value = held_velocity[c];
            AddVelocity(value, residue != nullptr ? &held_error[c] : nullptr,
                        step.Multiply(held[c].jacobian) * held_mass[c]);
            points.RecordImpulse(held[c].kind, held[c].index, step.Multiply(held[c].jacobian));
            points.Velocity(held[c].kind)[held[c].index] = value;
            if (residue != nullptr) *residue = held_error[c];
        }
        for (uint32_t i = lane + kHeld * kWidth; i < count; i += kWidth) {
            const auto term = points.At(head.a, i);
            const float* inverse_mass = points.InverseMass(term.kind);
            math::Vec3* velocity = points.Velocity(term.kind);
            if (velocity == nullptr || inverse_mass == nullptr || !(inverse_mass[term.index] > 0.0f))
                continue;
            AddVelocity(velocity[term.index], error.Point(term.kind, term.index),
                        step.Multiply(term.jacobian) * inverse_mass[term.index]);
            points.RecordImpulse(term.kind, term.index, step.Multiply(term.jacobian));
        }
        __syncwarp(mask);
    };
    if (apply_cached) {
        apply(old);
        return false;
    }
    float moment[6]{}, response[6]{};
    const auto gather = [&](math::Vec3 g, math::Vec3 v, float m) {
        moment[0] += v.x * g.x; moment[1] += v.y * g.y; moment[2] += v.z * g.z;
        moment[3] += 0.5f * (v.x * g.y + v.y * g.x);
        moment[4] += 0.5f * (v.x * g.z + v.z * g.x);
        moment[5] += 0.5f * (v.y * g.z + v.z * g.y);
        response[0] += m * g.x * g.x; response[1] += m * g.y * g.y; response[2] += m * g.z * g.z;
        response[3] += m * g.x * g.y; response[4] += m * g.x * g.z; response[5] += m * g.y * g.z;
    };
    #pragma unroll
    for (uint32_t c = 0u; c < kHeld; ++c)
        gather(held[c].jacobian, held_velocity[c], held_mass[c]);
    for (uint32_t i = lane + kHeld * kWidth; i < count; i += kWidth) {
        const auto term = points.At(head.a, i);
        const float* inverse_mass = points.InverseMass(term.kind);
        const math::Vec3* velocity = points.Velocity(term.kind);
        if (velocity == nullptr) continue;
        gather(term.jacobian, velocity[term.index],
               inverse_mass != nullptr ? fmaxf(inverse_mass[term.index], 0.0f) : 0.0f);
    }
    #pragma unroll
    for (uint32_t c = 0u; c < 6u; ++c) {
        moment[c] = GroupSum<kWidth>(moment[c], mask);
        response[c] = GroupSum<kWidth>(response[c], mask);
    }
    float jv[n], A[n][n];
    nk::material::MpmStressBlockRates(moment, jv);
    nk::material::MpmStressBlockResponse(response, A);
    // Every lane solves the same small system, so no broadcast is needed.
    float residual[n];
    uint32_t rows = 0u;
    #pragma unroll
    for (uint32_t k = 0u; k < n; ++k) {
        const bool active = k == 0u ? head.lower != head.upper : compliance[k] < FLT_MAX;
        rows |= active ? 1u << k : 0u;
        A[k][k] += active ? compliance[k] : 0.0f;
        residual[k] = active ? rhs[k] * dt - jv[k] - compliance[k] * old[k] : 0.0f;
    }
    float L[n][n], inverse_pivot[n], delta[n];
    FactorMaterialBlock(A, rows, L, inverse_pivot);
    SolveMaterialFactor<n>(L, inverse_pivot, residual, rows, delta);
    float next[n];
    #pragma unroll
    for (uint32_t k = 0u; k < n; ++k) next[k] = old[k] + delta[k];
    const float bounded = fminf(fmaxf(next[0], head.lower), head.upper);
    if ((rows & 1u) && bounded != next[0]) {
        // A bound holds the pressure; the deviator then solves against that pressure step.
        const float fixed = bounded - old[0];
        #pragma unroll
        for (uint32_t k = 1u; k < n; ++k) residual[k] -= A[k][0] * fixed;
        SolveMaterialFactor<n - 1u>(L, inverse_pivot, residual, rows, delta);
        delta[0] = fixed;
        #pragma unroll
        for (uint32_t k = 0u; k < n; ++k) next[k] = old[k] + delta[k];
    }
    next[0] = fminf(fmaxf(next[0], head.lower), head.upper);
    if (head.friction_secondary < FLT_MAX) {
        const float bound = fmaxf(fmaf(head.mu, next[0], head.friction_secondary), 0.0f);
        float norm = 0.0f;
        #pragma unroll
        for (uint32_t k = 1u; k < n; ++k) norm += next[k] * next[k];
        norm = sqrtf(norm);
        if (norm > bound) {
            const float scale = bound / norm;
            #pragma unroll
            for (uint32_t k = 1u; k < n; ++k) next[k] *= scale;
        }
    }
    bool significant = false;
    #pragma unroll
    for (uint32_t k = 0u; k < n; ++k) {
        delta[k] = next[k] - old[k];
        significant |= fabsf(delta[k]) * A[k][k] > vel_tolerance;
    }
    apply(delta);
    if (lane == 0u)
        #pragma unroll
        for (uint32_t k = 0u; k < n; ++k) lambda[slot + k] = next[k];
    __syncwarp(mask);
    return significant;
}

__device__ bool SolveUnionRowWarp(uint32_t ls,            // env-local slot
                                  uint32_t gslot,         // global slot
                                  uint32_t env_row_base,  // env's first global slot
                                  uint32_t env_artic_base,// env's first global artic
                                  uint32_t j_row,         // row index into J/w arrays
                                  uint32_t wlane,
                                  const SlimRow* slim_sh, // null => build inline
                                  float* lambda_sh,       // null => use global lambda
                                  const float* meff_sh,   // null => use global row_meff
                                  const float* damping_sh, // null => use global row_damping
                                  float* __restrict__ lambda,    // global lambda
                                  const float* __restrict__ row_meff,  // global meff
                                  const float* __restrict__ row_damping, // global damping
                                  const float* J_sh,      // shared (union) OR global (PD)
                                  const float* w_sh,
                                  const float* J_b_sh,    // side-B chain-J (or null)
                                  const float* w_b_sh,    // side-B M^-1 J^T (or null)
                                  float* qdot_sh,         // K tiles, [tile*dof+r]
                                  const NkRow* __restrict__ urows,
                                  math::Vec3* __restrict__ body_lin_vel,
                                  math::Vec3* __restrict__ body_ang_vel,
                                  const float* __restrict__ body_inv_mass,
                                  const math::SymmetricMat3* __restrict__ body_world_inv_inertia,
                                  PointMassView point_masses,
                                  uint32_t dof_stride,
                                  float dt,
                                  bool apply_cached_impulse, VelocityErrorView error,
                                  const NkRow* prepared = nullptr,
                                  float vel_tolerance = 0.0f,
                                  float* articulation_pending = nullptr) {
    const NkRow row = prepared != nullptr ? prepared[0] : LoadRowWarp(urows, gslot, wlane);
    if (row.flags & nk::nk_row_flags::kMaterialBlock)
        return SolveMaterialBlockGroup<32u>(gslot, row, urows, lambda, point_masses, dt,
                                            apply_cached_impulse, vel_tolerance, error, wlane,
                                            0xffffffffu);
    const SlimRow sr = slim_sh != nullptr ? slim_sh[ls]
        : MakeSlimRow(row, env_row_base, env_artic_base);
    const uint32_t flags = sr.flags;
    if (!(flags & nk::nk_row_flags::kActive)) {
        return false;
    }
    const uint32_t jacobian_base = gslot - j_row;
    const bool block_row =
        (flags & (nk::nk_row_flags::kBlockNormal |
                  nk::nk_row_flags::kBlockTangent)) != 0u;
    uint32_t block_normal_slot = 0u;
    uint32_t block_tangent1_slot = 0u;
    uint32_t block_tangent2_slot = 0u;
    float block_old_normal = 0.0f;
    float block_old_tangent1 = 0.0f;
    float block_old_tangent2 = 0.0f;
    float block_delta_normal = 0.0f;
    float block_delta_tangent1 = 0.0f;
    float block_delta_tangent2 = 0.0f;
    if (block_row) {
        const NkRow& block = row;
        const uint32_t count = block.group_normal_count;
        const uint32_t point = (gslot - block.group_first) % count;
        block_normal_slot = block.group_first + point;
        block_tangent1_slot = block.group_first + count + point;
        block_tangent2_slot = block.group_first + 2u * count + point;
        block_old_normal = lambda[block_normal_slot];
        block_old_tangent1 = lambda[block_tangent1_slot];
        block_old_tangent2 = lambda[block_tangent2_slot];
    }
    if (block_row && gslot != block_normal_slot) return false;
    // Deferred rows leave articulation sides frozen; a scalar row's pending Schur correction
    // reaches its other sides first. Block rows take theirs in SolvePreparedContactBlockWarp.
    const bool deferred = articulation_pending != nullptr;
    SlimRow apply_sr = sr;
    if (deferred) apply_sr.code &= ~(kSlimAArt | kSlimBArt);
    if (deferred && !block_row && !apply_cached_impulse && articulation_pending[0] != 0.0f) {
        const float pending = articulation_pending[0];
        __syncwarp();
        if (wlane == 0u) articulation_pending[0] = 0.0f;
        ApplySlimImpulse(apply_sr, gslot, j_row, wlane, pending, env_artic_base, qdot_sh,
                         urows, J_sh, w_sh, J_b_sh, w_b_sh, body_lin_vel,
                         body_ang_vel, body_inv_mass, body_world_inv_inertia,
                         point_masses, dof_stride, error, prepared);
    }

    float jv = 0.0f, block_jv_tangent1 = 0.0f, block_jv_tangent2 = 0.0f;
    if (!apply_cached_impulse) {
        jv = ComputeSlimRowVelocity<true>(
            sr, gslot, j_row, J_sh, J_b_sh, qdot_sh, urows, body_lin_vel,
            body_ang_vel, point_masses, dof_stride, wlane, &row);
        const float normal_damping = damping_sh != nullptr ? damping_sh[ls] : row_damping[gslot];
        if (block_row && ContactFrictionActive(row, normal_damping, dt, jv, block_old_normal)) {
            const NkRow tangent1_row = prepared != nullptr
                ? prepared[1] : LoadRowWarp(urows, block_tangent1_slot, wlane);
            const NkRow tangent2_row = prepared != nullptr
                ? prepared[2] : LoadRowWarp(urows, block_tangent2_slot, wlane);
            const SlimRow tangent1 = MakeSlimRow(tangent1_row, env_row_base, env_artic_base);
            const SlimRow tangent2 = MakeSlimRow(tangent2_row, env_row_base, env_artic_base);
            const math::Vec3 tangent_velocity = ComputeSlimTangentVelocity(
                tangent1, tangent2, block_tangent1_slot, block_tangent2_slot,
                prepared != nullptr ? j_row + 1u : block_tangent1_slot - jacobian_base,
                prepared != nullptr ? j_row + 2u : block_tangent2_slot - jacobian_base,
                J_sh, J_b_sh, qdot_sh,
                urows, body_lin_vel, body_ang_vel, point_masses, dof_stride, wlane,
                prepared);
            block_jv_tangent1 = tangent_velocity.y;
            block_jv_tangent2 = tangent_velocity.z;
        }
    }
    float delta = 0.0f;
    if (wlane == 0u) {
        // Union islands cache lambda and effective mass; other callers read global rows.
        const float effective_mass = meff_sh != nullptr ? meff_sh[ls]
                                                        : row_meff[gslot];
        const float damping = damping_sh != nullptr ? damping_sh[ls]
                                                    : row_damping[gslot];
        const float old_impulse = lambda_sh != nullptr ? lambda_sh[ls]
                                                       : lambda[gslot];
        if (apply_cached_impulse) {
            if (block_row && gslot == block_normal_slot) {
                block_delta_normal = block_old_normal;
                block_delta_tangent1 = block_old_tangent1;
                block_delta_tangent2 = block_old_tangent2;
            } else if (!block_row) {
                delta = old_impulse;
            }
        } else if (block_row) {
            const NkRow& normal = row;
            const NkRow& tangent1 = prepared != nullptr ? prepared[1] : urows[block_tangent1_slot];
            const NkRow& tangent2 = prepared != nullptr ? prepared[2] : urows[block_tangent2_slot];
            const float normal_damping = damping_sh != nullptr
                ? damping_sh[block_normal_slot - env_row_base]
                : row_damping[block_normal_slot];
            const PreparedContactStep step = ContactBlockStep<false>(normal, tangent1, tangent2,
                {block_old_normal, block_old_tangent1, block_old_tangent2}, normal_damping,
                {jv, block_jv_tangent1, block_jv_tangent2}, dt, 0.0f, vel_tolerance);
            const float new_normal = step.impulse.x;
            const float new_tangent1 = step.impulse.y;
            const float new_tangent2 = step.impulse.z;
            lambda[block_normal_slot] = new_normal;
            lambda[block_tangent1_slot] = new_tangent1;
            lambda[block_tangent2_slot] = new_tangent2;
            block_delta_normal = step.delta.x;
            block_delta_tangent1 = step.delta.y;
            block_delta_tangent2 = step.delta.z;
        } else {
            const float lambda_inc = ScalarRowIncrement(
                sr, effective_mass, damping, old_impulse, jv, dt);
            float lower = sr.lower;
            float upper = sr.upper;
            if (flags & nk::nk_row_flags::kFriction) {
                float total = 0.0f;
                for (uint32_t g = 0u; g < sr.group_cnt; ++g) {
                    const float lg = lambda_sh != nullptr
                                         ? lambda_sh[sr.group_local + g]
                                         : lambda[env_row_base + sr.group_local + g];
                    total += fmaxf(lg, 0.0f);
                }
                lower = 0.0f;
                upper = fmaxf(sr.mu, 0.0f) * total;
            }
            const ScalarStepResult step = ScalarRowStep(old_impulse, lambda_inc, lower, upper);
            if (lambda_sh != nullptr) lambda_sh[ls] = step.impulse;
            else lambda[gslot] = step.impulse;
            delta = step.delta;
        }
    }
    delta = __shfl_sync(0xffffffffu, delta, 0);
    if (block_row) {
        block_delta_normal = __shfl_sync(0xffffffffu, block_delta_normal, 0);
        block_delta_tangent1 = __shfl_sync(0xffffffffu, block_delta_tangent1, 0);
        block_delta_tangent2 = __shfl_sync(0xffffffffu, block_delta_tangent2, 0);
        if (block_delta_normal != 0.0f || block_delta_tangent1 != 0.0f ||
            block_delta_tangent2 != 0.0f) {
            const NkRow tangent1_row = prepared != nullptr
                ? prepared[1] : LoadRowWarp(urows, block_tangent1_slot, wlane);
            const NkRow tangent2_row = prepared != nullptr
                ? prepared[2] : LoadRowWarp(urows, block_tangent2_slot, wlane);
            const SlimRow sr_block = MakeSlimRow(tangent1_row, env_row_base, env_artic_base);
            const SlimRow sr_block_second = MakeSlimRow(tangent2_row, env_row_base, env_artic_base);
            ApplySlimContactImpulse(apply_sr, sr_block, sr_block_second,
                block_normal_slot, block_tangent1_slot, block_tangent2_slot,
                prepared != nullptr ? j_row : block_normal_slot - jacobian_base,
                prepared != nullptr ? j_row + 1u : block_tangent1_slot - jacobian_base,
                prepared != nullptr ? j_row + 2u : block_tangent2_slot - jacobian_base,
                wlane,
                {block_delta_normal, block_delta_tangent1, block_delta_tangent2},
                env_artic_base, qdot_sh, urows, w_sh, w_b_sh,
                body_lin_vel, body_ang_vel, body_inv_mass,
                body_world_inv_inertia, point_masses, dof_stride, error, prepared);
        }
    }
    if (!block_row && delta != 0.0f) {
        ApplySlimImpulse(apply_sr, gslot, j_row, wlane, delta, env_artic_base, qdot_sh,
                         urows, J_sh, w_sh, J_b_sh, w_b_sh, body_lin_vel,
                         body_ang_vel, body_inv_mass, body_world_inv_inertia,
                         point_masses, dof_stride, error, prepared);
    }
    if (vel_tolerance <= 0.0f)
        return delta != 0.0f || block_delta_normal != 0.0f ||
               block_delta_tangent1 != 0.0f || block_delta_tangent2 != 0.0f;
    // row_meff stores the row's effective mass, so delta/meff is the velocity moved.
    const float meff = meff_sh != nullptr ? meff_sh[ls] : row_meff[gslot];
    const float scalar_step = meff > 0.0f ? fabsf(delta) / meff : fabsf(delta);
    const float block_step = fmaxf(fabsf(block_delta_normal),
                                   fmaxf(fabsf(block_delta_tangent1),
                                         fabsf(block_delta_tangent2))) *
                             row.contact_response.xx;
    return scalar_step > vel_tolerance || block_step > vel_tolerance;
}

// Normal rows project penetration into separate pseudo velocities in fixed order.
// Tangent rows receive no pseudo impulse.
template <bool apply = true>
__device__ bool SolvePositionRowWarp(uint32_t gslot,
                                     uint32_t env_row_base,
                                     uint32_t env_artic_base,
                                     uint32_t j_row,
                                     uint32_t wlane,
                                     const float* __restrict__ row_meff,
                                     const float* __restrict__ row_penetration,
                                     float* __restrict__ row_pseudo_lambda,
                                     const float* J_sh,
                                     const float* w_sh,
                                     const float* J_b_sh,
                                     const float* w_b_sh,
                                     float* qdot_pseudo_sh,
                                     const NkRow* __restrict__ urows,
                                     math::Vec3* __restrict__ body_pseudo_lin,
                                     math::Vec3* __restrict__ body_pseudo_ang,
                                     const float* __restrict__ body_inv_mass,
                                     const math::SymmetricMat3* __restrict__ body_world_inv_inertia,
                                     PointMassView point_masses,
                                     uint32_t dof_stride,
                                     float beta, float slop, float dt,
                                     float baumgarte_max_velocity,
                                     const NkRow* prepared = nullptr,
                                     float* articulation_pending = nullptr) {
    const NkRow row = prepared != nullptr ? *prepared : LoadRowWarp(urows, gslot, wlane);
    const SlimRow sr = MakeSlimRow(row,
                                   env_row_base, env_artic_base);
    const uint32_t flags = sr.flags;
    if (!(flags & nk::nk_row_flags::kActive) ||
        (flags & (nk::nk_row_flags::kFriction | nk::nk_row_flags::kBlockTangent |
                  nk::nk_row_flags::kVelocityOnly))) {
        return false;
    }
    // Deferred rows keep articulation pseudo velocities frozen and first pass on their
    // pending Schur correction; a screening pass reports it as work.
    const bool deferred = articulation_pending != nullptr;
    SlimRow rest = sr;
    if (deferred) rest.code &= ~(kSlimAArt | kSlimBArt);
    if (deferred && articulation_pending[0] != 0.0f) {
        if constexpr (!apply) return true;
        const float pending = articulation_pending[0];
        __syncwarp();
        if (wlane == 0u) articulation_pending[0] = 0.0f;
        ApplySlimImpulse(rest, gslot, j_row, wlane, pending, env_artic_base, qdot_pseudo_sh,
                         urows, J_sh, w_sh, J_b_sh, w_b_sh,
                         body_pseudo_lin, body_pseudo_ang, body_inv_mass,
                         body_world_inv_inertia, point_masses,
                         dof_stride, {}, prepared);
    }

    // Row state and the point side's terms load together; the apply reuses those terms.
    float depth = 0.0f, effective_mass = 0.0f, old_imp = 0.0f;
    if (wlane == 0u) {
        depth = row_penetration[gslot];
        effective_mass = row_meff[gslot];
        old_imp = row_pseudo_lambda[gslot];
    }
    const NkRowSide point_side[1] = {SlimPointSide(sr)};
    const bool point = (sr.code & kSlimPointMass) != 0u;
    nk::PointEndpointRange range{0u, 1u};
    if (point && point_side[0].kind == nk::kNkSidePointEndpoint)
        range = point_masses.ranges[point_side[0].index];
    const bool cached = point && range.count <= warpSize;
    PointLaneTerm term;
    float point_velocity[1];
    if (cached)
        LoadPointSideVelocities(point_side, range, point_masses, {}, wlane, term, point_velocity);

    // Both reaction sides participate in geometric push-out, preserving center of mass.
    const float jv = ComputeSlimRowVelocity<true>(
        sr, gslot, j_row, J_sh, J_b_sh, qdot_pseudo_sh, urows,
        body_pseudo_lin, body_pseudo_ang, point_masses, dof_stride, wlane, prepared,
        cached ? point_velocity : nullptr);

    float delta = 0.0f;
    if (wlane == 0u) {
        const PositionStepResult step = PositionRowStep(depth, effective_mass, old_imp, jv,
            beta, slop, dt, baumgarte_max_velocity);
        if constexpr (apply) row_pseudo_lambda[gslot] = step.impulse;
        delta = step.delta;
    }
    delta = __shfl_sync(0xffffffffu, delta, 0);
    if (apply && delta != 0.0f) {
        // The point side shares no state with an articulated side, so it may go first.
        if (cached) {
            ApplyPointLaneImpulse(term, range.count, wlane, delta, point_masses);
            __syncwarp();
            rest.code &= ~kSlimHasDyn;
        }
        ApplySlimImpulse(rest, gslot, j_row, wlane, delta, env_artic_base, qdot_pseudo_sh,
                         urows, J_sh, w_sh, J_b_sh, w_b_sh,
                         body_pseudo_lin, body_pseudo_ang, body_inv_mass,
                         body_world_inv_inertia, point_masses,
                         dof_stride, {}, prepared);
    }
    return delta != 0.0f;
}

// Sides whose owner is split across a warp round (share > 1) sum their reactions here;
// the round adds each owner's total once.
struct RowSplit {
    float share[2] = {1.0f, 1.0f};
    math::Vec3 linear[2] = {};
    math::Vec3 angular[2] = {};

    __device__ bool Any() const { return share[0] > 1.0f || share[1] > 1.0f; }
};

__device__ void ApplyDynamicImpulseScalar(
    uint32_t gslot, float delta, const NkRow* __restrict__ urows,
    math::Vec3* __restrict__ body_lin_vel,
    math::Vec3* __restrict__ body_ang_vel,
    const float* __restrict__ body_inv_mass,
    const math::SymmetricMat3* __restrict__ body_world_inv_inertia,
    PointMassView point_masses, VelocityErrorView error = {}, RowSplit* split = nullptr) {
    if (delta == 0.0f) return;
    const NkRow row = urows[gslot];
    for (int side = 0; side < 2; ++side) {
        const NkRowSide& sd = side == 0 ? row.a : row.b;
        const bool shared = split != nullptr && split->share[side] > 1.0f;
        if (PointMassView::IsPointSide(sd.kind)) {
            if (!shared) {
                ApplyPointImpulse(sd, delta, point_masses, error);
                continue;
            }
            const auto term = point_masses.At(sd, 0u);
            const float* inverse_mass = point_masses.InverseMass(term.kind);
            if (inverse_mass != nullptr && inverse_mass[term.index] > 0.0f) {
                split->linear[side] += point_masses.Respond(term.kind, term.index, term.jacobian,
                                                            inverse_mass[term.index], delta);
                point_masses.RecordSharedImpulse(term.kind, term.index, term.jacobian * delta);
            }
        } else if (sd.kind == kNkSideRigid) {
            const float im = body_inv_mass[sd.index];
            if (im > 0.0f) {
                const math::Vec3 angular_response = body_world_inv_inertia[sd.index].Multiply(sd.jang);
                if (shared) {
                    split->linear[side] += sd.jlin * (im * delta);
                    split->angular[side] += angular_response * delta;
                    continue;
                }
                math::Vec3& v = body_lin_vel[sd.index];
                math::Vec3& w = body_ang_vel[sd.index];
                AddVelocity(v, error.Linear(sd.index), sd.jlin * (im * delta));
                AddVelocity(w, error.Angular(sd.index), angular_response * delta);
            }
        }
    }
}

// J M^-1 J^T of two rows through one shared rigid or single-term point owner.
__device__ float SplitSideCoupling(const NkRowSide& lhs, const NkRowSide& rhs,
                                   const float* __restrict__ body_inv_mass,
                                   const math::SymmetricMat3* __restrict__ body_world_inv_inertia,
                                   PointMassView points) {
    if (PointMassView::IsPointSide(lhs.kind)) return points.Coupling(lhs, rhs);
    if (lhs.kind != kNkSideRigid) return 0.0f;
    const float im = body_inv_mass[lhs.index];
    return im > 0.0f ? im * Dot3(lhs.jlin, rhs.jlin) +
                       Dot3(lhs.jang, body_world_inv_inertia[lhs.index].Multiply(rhs.jang))
                     : 0.0f;
}

// A split side responds as if its owner kept 1/share of its mass.
__device__ float SplitEffectiveMass(const NkRow& row, float effective_mass, const RowSplit& split,
                                    const float* __restrict__ body_inv_mass,
                                    const math::SymmetricMat3* __restrict__ body_world_inv_inertia,
                                    PointMassView points) {
    if (!(effective_mass > 0.0f)) return effective_mass;
    float response = 1.0f / effective_mass;
    for (uint32_t s = 0u; s < 2u; ++s) {
        if (split.share[s] <= 1.0f) continue;
        const NkRowSide& side = s == 0u ? row.a : row.b;
        response += (split.share[s] - 1.0f) *
                    SplitSideCoupling(side, side, body_inv_mass, body_world_inv_inertia, points);
    }
    return 1.0f / response;
}

// An island without articulation DOFs uses one thread for the same ordered row solve.
__device__ void SolveDynamicRowScalar(
    uint32_t gslot, uint32_t env_row_base, uint32_t env_artic_base,
    const NkRow* __restrict__ urows, float* __restrict__ lambda,
    const float* __restrict__ row_meff,
    const float* __restrict__ row_damping,
    math::Vec3* __restrict__ body_lin_vel,
    math::Vec3* __restrict__ body_ang_vel,
    const float* __restrict__ body_inv_mass,
    const math::SymmetricMat3* __restrict__ body_world_inv_inertia,
    PointMassView point_masses, float dt,
    bool apply_cached_impulse, VelocityErrorView error, RowSplit* split = nullptr) {
    const SlimRow sr = MakeSlimRow(urows[gslot], env_row_base, env_artic_base);
    const uint32_t flags = sr.flags;
    if (!(flags & nk::nk_row_flags::kActive)) return;
    const bool block_row =
        (flags & (nk::nk_row_flags::kBlockNormal |
                  nk::nk_row_flags::kBlockTangent)) != 0u;
    uint32_t block_normal_slot = 0u;
    uint32_t block_tangent1_slot = 0u;
    uint32_t block_tangent2_slot = 0u;
    float block_old_normal = 0.0f;
    float block_old_tangent1 = 0.0f;
    float block_old_tangent2 = 0.0f;
    if (block_row) {
        const NkRow& block = urows[gslot];
        const uint32_t count = block.group_normal_count;
        const uint32_t point = (gslot - block.group_first) % count;
        block_normal_slot = block.group_first + point;
        block_tangent1_slot = block.group_first + count + point;
        block_tangent2_slot = block.group_first + 2u * count + point;
        block_old_normal = lambda[block_normal_slot];
        block_old_tangent1 = lambda[block_tangent1_slot];
        block_old_tangent2 = lambda[block_tangent2_slot];
    }
    if (block_row && gslot != block_normal_slot) return;

    const bool split_row = split != nullptr && split->Any() && !apply_cached_impulse;
    math::SymmetricMat3 response{};
    if (block_row && !apply_cached_impulse) {
        response = urows[block_normal_slot].contact_response;
        // Each split side adds (share - 1) copies of its own block response.
        for (uint32_t s = 0u; split_row && response.xx > 0.0f && s < 2u; ++s) {
            const float extra = split->share[s] - 1.0f;
            if (extra <= 0.0f) continue;
            const NkRowSide& n = s == 0u ? urows[block_normal_slot].a : urows[block_normal_slot].b;
            const NkRowSide& t1 = s == 0u ? urows[block_tangent1_slot].a : urows[block_tangent1_slot].b;
            const NkRowSide& t2 = s == 0u ? urows[block_tangent2_slot].a : urows[block_tangent2_slot].b;
            const auto k = [&](const NkRowSide& l, const NkRowSide& r) {
                return extra * SplitSideCoupling(l, r, body_inv_mass, body_world_inv_inertia, point_masses);
            };
            response.xx += k(n, n);
            response.yy += k(t1, t1);
            response.zz += k(t2, t2);
            response.xy += k(n, t1);
            response.xz += k(n, t2);
            response.yz += k(t1, t2);
        }
    }
    const float jv = apply_cached_impulse ? 0.0f : ComputeSlimRowVelocity(
        sr, gslot, gslot, nullptr, nullptr, nullptr, urows, body_lin_vel,
        body_ang_vel, point_masses, 0u);
    float block_jv_tangent1 = 0.0f;
    float block_jv_tangent2 = 0.0f;
    if (block_row && !apply_cached_impulse &&
        ContactFrictionActive(urows[gslot], response.xx, row_damping[gslot], dt, jv, block_old_normal)) {
        const SlimRow tangent1 = MakeSlimRow(urows[block_tangent1_slot],
                                             env_row_base, env_artic_base);
        const SlimRow tangent2 = MakeSlimRow(urows[block_tangent2_slot],
                                             env_row_base, env_artic_base);
        block_jv_tangent1 = ComputeSlimRowVelocity(
            tangent1, block_tangent1_slot, block_tangent1_slot, nullptr, nullptr,
            nullptr, urows, body_lin_vel, body_ang_vel, point_masses, 0u);
        block_jv_tangent2 = ComputeSlimRowVelocity(
            tangent2, block_tangent2_slot, block_tangent2_slot, nullptr, nullptr,
            nullptr, urows, body_lin_vel, body_ang_vel, point_masses, 0u);
    }

    const float effective_mass = split_row && !block_row
        ? SplitEffectiveMass(urows[gslot], row_meff[gslot], *split, body_inv_mass,
                             body_world_inv_inertia, point_masses)
        : row_meff[gslot];
    const float old_impulse = lambda[gslot];
    float delta = 0.0f;
    float block_delta_normal = 0.0f;
    float block_delta_tangent1 = 0.0f;
    float block_delta_tangent2 = 0.0f;
    if (apply_cached_impulse) {
        if (block_row && gslot == block_normal_slot) {
            block_delta_normal = block_old_normal;
            block_delta_tangent1 = block_old_tangent1;
            block_delta_tangent2 = block_old_tangent2;
        } else if (!block_row) {
            delta = old_impulse;
        }
    } else if (block_row) {
        const NkRow& normal = urows[block_normal_slot];
        const NkRow& tangent1 = urows[block_tangent1_slot];
        const NkRow& tangent2 = urows[block_tangent2_slot];
        const float normal_damping = row_damping[block_normal_slot];
        const float damping_scale = 1.0f / (1.0f + normal_damping * dt);
        const float residual_n = normal.rhs * dt * damping_scale - jv -
                                 normal.compliance_alpha * damping_scale * block_old_normal;
        const float residual_t1 = tangent1.rhs * dt - block_jv_tangent1 -
                                  tangent1.compliance_alpha * block_old_tangent1;
        const float residual_t2 = tangent2.rhs * dt - block_jv_tangent2 -
                                  tangent2.compliance_alpha * block_old_tangent2;
        const auto projected = constraint::ProjectedCoulombStep(response,
            {residual_n, residual_t1, residual_t2},
            {block_old_normal, block_old_tangent1, block_old_tangent2}, normal.mu, normal.friction_secondary);
        const float new_normal = projected.x, new_tangent1 = projected.y, new_tangent2 = projected.z;
        lambda[block_normal_slot] = new_normal;
        lambda[block_tangent1_slot] = new_tangent1;
        lambda[block_tangent2_slot] = new_tangent2;
        block_delta_normal = new_normal - block_old_normal;
        block_delta_tangent1 = new_tangent1 - block_old_tangent1;
        block_delta_tangent2 = new_tangent2 - block_old_tangent2;
    } else {
        const float rhs_v = sr.rhs * dt;
        const float damping_scale = 1.0f / (1.0f + row_damping[gslot] * dt);
        const float implicit_mass = effective_mass /
            (1.0f - effective_mass * sr.R * (1.0f - damping_scale));
        const float lambda_inc =
            implicit_mass * (rhs_v * damping_scale - jv -
                             sr.R * damping_scale * old_impulse);
        float lower = sr.lower;
        float upper = sr.upper;
        if (flags & nk::nk_row_flags::kFriction) {
            float total = 0.0f;
            for (uint32_t g = 0u; g < sr.group_cnt; ++g) {
                total += fmaxf(lambda[env_row_base + sr.group_local + g], 0.0f);
            }
            lower = 0.0f;
            upper = fmaxf(sr.mu, 0.0f) * total;
        }
        const float new_impulse =
            fminf(fmaxf(old_impulse + lambda_inc, lower), upper);
        lambda[gslot] = new_impulse;
        const float d = new_impulse - old_impulse;
        delta = d;
    }
    if (block_row) {
        ApplyDynamicImpulseScalar(block_normal_slot, block_delta_normal, urows,
                                  body_lin_vel, body_ang_vel, body_inv_mass,
                                  body_world_inv_inertia, point_masses, error, split);
        ApplyDynamicImpulseScalar(block_tangent1_slot, block_delta_tangent1, urows,
                                  body_lin_vel, body_ang_vel, body_inv_mass,
                                  body_world_inv_inertia, point_masses, error, split);
        ApplyDynamicImpulseScalar(block_tangent2_slot, block_delta_tangent2, urows,
                                  body_lin_vel, body_ang_vel, body_inv_mass,
                                  body_world_inv_inertia, point_masses, error, split);
    } else {
        ApplyDynamicImpulseScalar(gslot, delta, urows, body_lin_vel, body_ang_vel,
                                  body_inv_mass, body_world_inv_inertia,
                                  point_masses, error, split);
    }
}

// No-articulation subset of SolvePositionRowWarp, with the same two-sided
// reaction as the scalar velocity solve.
__device__ void SolvePositionRowScalar(
    uint32_t gslot, uint32_t env_row_base, uint32_t env_artic_base,
    const float* __restrict__ row_meff,
    const float* __restrict__ row_penetration,
    float* __restrict__ row_pseudo_lambda,
    const NkRow* __restrict__ urows,
    math::Vec3* __restrict__ body_pseudo_lin,
    math::Vec3* __restrict__ body_pseudo_ang,
    const float* __restrict__ body_inv_mass,
    const math::SymmetricMat3* __restrict__ body_world_inv_inertia,
    PointMassView point_masses,
    float beta, float slop, float dt, float baumgarte_max_velocity, RowSplit* split = nullptr) {
    const SlimRow sr = MakeSlimRow(urows[gslot], env_row_base, env_artic_base);
    const uint32_t flags = sr.flags;
    if (!(flags & nk::nk_row_flags::kActive) ||
        (flags & (nk::nk_row_flags::kFriction | nk::nk_row_flags::kBlockTangent |
                  nk::nk_row_flags::kVelocityOnly))) return;
    const float jv = ComputeSlimRowVelocity(
        sr, gslot, gslot, nullptr, nullptr, nullptr, urows,
        body_pseudo_lin, body_pseudo_ang, point_masses, 0u);
    const float depth = row_penetration[gslot];
    const float bias =
        fminf(beta * fmaxf(depth - slop, 0.0f) / dt, baumgarte_max_velocity);
    const float effective_mass = split != nullptr && split->Any()
        ? SplitEffectiveMass(urows[gslot], row_meff[gslot], *split, body_inv_mass,
                             body_world_inv_inertia, point_masses)
        : row_meff[gslot];
    const float old_imp = row_pseudo_lambda[gslot];
    const float new_imp = fmaxf(old_imp + effective_mass * (bias - jv), 0.0f);
    row_pseudo_lambda[gslot] = new_imp;
    const float d = new_imp - old_imp;
    const float delta = d;
    if (delta == 0.0f) return;

    ApplyDynamicImpulseScalar(gslot, delta, urows, body_pseudo_lin, body_pseudo_ang,
                              body_inv_mass, body_world_inv_inertia,
                              point_masses, {}, split);
}

// A lane's writable owners: its row's two sides, then the friction group a non-block row reads.
struct ScalarBatchShared {
    uint64_t owner[3][32];
    float linear[2][32][3];
    float angular[2][32][3];

    __device__ void Store(float (&at)[3], math::Vec3 v) { at[0] = v.x; at[1] = v.y; at[2] = v.z; }
    __device__ static math::Vec3 Load(const float (&at)[3]) { return {at[0], at[1], at[2]}; }
};

__device__ inline uint64_t ScalarSideOwner(const NkRowSide& side, PointMassView points,
                                           const float* __restrict__ body_inv_mass) {
    if (side.kind == kNkSideRigid)
        return body_inv_mass[side.index] > 0.0f ? (uint64_t{kNkSideRigid} << 32u) | side.index
                                                 : kNoOwner;
    if (!PointMassView::IsPointSide(side.kind) || points.Count(side) == 0u) return kNoOwner;
    const auto term = points.At(side, 0u);
    return (uint64_t{term.kind} << 32u) | term.index;
}

// A warp sweeps an island in equal batches of at most 32 rows. Each row waits for the earlier
// rows of its batch that share an unsplit owner, so those owners see the island's ordered updates.
template <typename Solve>
__device__ void SweepScalarIslandWarp(
    const IslandRecord& rec, const uint32_t* __restrict__ row_order,
    const NkRow* __restrict__ urows, const float* __restrict__ body_inv_mass,
    PointMassView points, math::Vec3* body_lin, math::Vec3* body_ang, VelocityErrorView error,
    ScalarBatchShared& sh, uint32_t lane, Solve&& solve) {
    const uint32_t batches = (rec.seg_cnt + warpSize - 1u) / warpSize;
    const uint32_t width = (rec.seg_cnt + batches - 1u) / batches;
    const uint32_t lower = (1u << lane) - 1u;
    for (uint32_t base = 0u; base < rec.seg_cnt; base += width) {
        const bool valid = lane < width && base + lane < rec.seg_cnt;
        const uint32_t slot = valid ? row_order[rec.seg_off + base + lane] : 0u;
        uint64_t owner[3] = {kNoOwner, kNoOwner, kNoOwner};
        if (valid) {
            const NkRow& row = urows[slot];
            owner[0] = ScalarSideOwner(row.a, points, body_inv_mass);
            owner[1] = ScalarSideOwner(row.b, points, body_inv_mass);
            if (!(row.flags & (nk::nk_row_flags::kBlockNormal | nk::nk_row_flags::kBlockTangent)))
                owner[2] = (uint64_t{kOwnerGroupKind} << 32u) | row.group_first;
        }
        for (uint32_t k = 0u; k < 3u; ++k) sh.owner[k][lane] = owner[k];
        __syncwarp();
        uint32_t pending = __ballot_sync(~0u, valid);
        // Lanes sharing each side's owner, then every lane an unsplit owner orders this one after.
        uint32_t same[2] = {0u, 0u}, conflict = 0u;
        for (uint32_t m = 0u; m < warpSize; ++m) {
            const uint64_t a = sh.owner[0][m], b = sh.owner[1][m];
            for (uint32_t s = 0u; s < 2u; ++s)
                if (owner[s] != kNoOwner && (a == owner[s] || b == owner[s])) same[s] |= 1u << m;
            if (owner[2] != kNoOwner && sh.owner[2][m] == owner[2]) conflict |= 1u << m;
        }
        bool split[2];
        for (uint32_t s = 0u; s < 2u; ++s) {
            split[s] = static_cast<uint32_t>(__popc(same[s])) > kSplitShare;
            if (!split[s]) conflict |= same[s];
        }
        conflict &= lower;
        while (pending != 0u) {
            const bool ready = ((pending >> lane) & 1u) != 0u && (conflict & pending) == 0u;
            const uint32_t round = __ballot_sync(~0u, ready);
            RowSplit rs;
            for (uint32_t s = 0u; s < 2u; ++s)
                if (split[s]) rs.share[s] = static_cast<float>(__popc(same[s] & round));
            if (ready) solve(slot, rs);
            for (uint32_t s = 0u; s < 2u; ++s) {
                sh.Store(sh.linear[s][lane], rs.linear[s]);
                sh.Store(sh.angular[s][lane], rs.angular[s]);
            }
            __syncwarp();
            // The first round row of a split owner adds every reaction on it in lane order.
            for (uint32_t s = 0u; ready && s < 2u; ++s) {
                uint32_t rows = same[s] & round;
                if (rs.share[s] <= 1.0f || (rows & lower) != 0u) continue;
                math::Vec3 linear{}, angular{};
                for (; rows != 0u; rows &= rows - 1u) {
                    const uint32_t m = static_cast<uint32_t>(__ffs(static_cast<int>(rows))) - 1u;
                    const uint32_t t = sh.owner[0][m] == owner[s] ? 0u : 1u;
                    linear += ScalarBatchShared::Load(sh.linear[t][m]);
                    angular += ScalarBatchShared::Load(sh.angular[t][m]);
                }
                const uint32_t kind = static_cast<uint32_t>(owner[s] >> 32u);
                const uint32_t index = static_cast<uint32_t>(owner[s]);
                if (kind == kNkSideRigid) {
                    AddVelocity(body_lin[index], error.Linear(index), linear);
                    AddVelocity(body_ang[index], error.Angular(index), angular);
                } else if (math::Vec3* velocity = points.Velocity(kind)) {
                    AddVelocity(velocity[index], error.Point(kind, index), linear);
                }
            }
            __syncwarp();
            pending &= ~round;
        }
    }
}

__global__ void SolveRowsScalarIslandsKernel(
    const NkRow* __restrict__ urows, float* __restrict__ lambda,
    const float* __restrict__ row_meff,
    const float* __restrict__ row_damping,
    math::Vec3* __restrict__ body_lin_vel,
    math::Vec3* __restrict__ body_ang_vel,
    const float* __restrict__ body_inv_mass,
    const math::SymmetricMat3* __restrict__ body_world_inv_inertia,
    PointMassView point_masses,
    const uint32_t* __restrict__ islands,
    const uint32_t* __restrict__ row_order,
    float* __restrict__ row_penetration,
    float* __restrict__ row_pseudo_lambda,
    math::Vec3* __restrict__ body_pseudo_lin,
    math::Vec3* __restrict__ body_pseudo_ang,
    math::Vec3* __restrict__ particle_pseudo_vel,
    math::Vec3* __restrict__ grid_pseudo_vel,
    const uint32_t* __restrict__ island_count_dev,
    IslandActivityView activity,
    uint32_t rows_per_env, uint32_t artics_per_env,
    uint32_t vel_iters, uint32_t pos_iters,
    float pos_beta, float pos_slop, float dt,
    float baumgarte_max_velocity, bool apply_cached_impulses, VelocityErrorView error) {
    const uint32_t live_islands = *island_count_dev;
    __shared__ ScalarBatchShared batch_shared[kScalarIslandBlockSize / 32u];
    ScalarBatchShared& sh = batch_shared[threadIdx.x / 32u];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32u;
    const uint32_t warps = gridDim.x * blockDim.x / 32u;
    const PointMassView pseudo_points = point_masses.Pseudo(particle_pseudo_vel, grid_pseudo_vel);
    const auto velocity_row = [&](const IslandRecord& rec, uint32_t slot, bool cached,
                                  RowSplit* split) {
        SolveDynamicRowScalar(slot, rec.env * rows_per_env,
                              rec.env * (artics_per_env == 0u ? 1u : artics_per_env), urows,
                              lambda, row_meff, row_damping, body_lin_vel, body_ang_vel,
                              body_inv_mass, body_world_inv_inertia, point_masses, dt, cached,
                              error, split);
    };
    const auto position_row = [&](const IslandRecord& rec, uint32_t slot, RowSplit* split) {
        SolvePositionRowScalar(slot, rec.env * rows_per_env,
                               rec.env * (artics_per_env == 0u ? 1u : artics_per_env), row_meff,
                               row_penetration, row_pseudo_lambda, urows, body_pseudo_lin,
                               body_pseudo_ang, body_inv_mass, body_world_inv_inertia,
                               pseudo_points, pos_beta, pos_slop, dt, baumgarte_max_velocity,
                               split);
    };
    const auto begin_position = [&](const IslandRecord& rec, uint32_t first, uint32_t step) {
        for (uint32_t r = first; r < rec.seg_cnt; r += step) {
            const uint32_t slot = row_order[rec.seg_off + r];
            row_pseudo_lambda[slot] = 0.0f;
            UpdateSpeculativePenetration<false>(slot, rec.env * rows_per_env,
                rec.env * (artics_per_env == 0u ? 1u : artics_per_env), row_penetration, urows,
                nullptr, nullptr, nullptr, body_lin_vel, body_ang_vel, point_masses, 0u, dt);
        }
    };
    // Each lane takes one island of its warp's 32; islands wider than kSplitShare rows are
    // then swept by the whole warp.
    for (uint32_t first = warp * warpSize; first < live_islands; first += warps * warpSize) {
        const uint32_t island = first + lane;
        IslandRecord rec{0u, 0u, 0u, 0u};
        bool solve = island < live_islands;
        if (solve) {
            rec = reinterpret_cast<const IslandRecord*>(islands)[island];
            solve = !(rec.flags & kIslandWarpWork) && activity.Active(rec);
        }
        if (solve && rec.seg_cnt <= kSplitShare) {
            if (apply_cached_impulses)
                for (uint32_t r = 0u; r < rec.seg_cnt; ++r)
                    velocity_row(rec, row_order[rec.seg_off + r], true, nullptr);
            for (uint32_t it = 0u; it < vel_iters; ++it)
                for (uint32_t r = 0u; r < rec.seg_cnt; ++r)
                    velocity_row(rec, row_order[rec.seg_off + r], false, nullptr);
            if (pos_iters != 0u) {
                begin_position(rec, 0u, 1u);
                for (uint32_t it = 0u; it < pos_iters; ++it)
                    for (uint32_t r = 0u; r < rec.seg_cnt; ++r)
                        position_row(rec, row_order[rec.seg_off + r], nullptr);
            }
        }
        uint32_t wide = __ballot_sync(~0u, solve && rec.seg_cnt > kSplitShare);
        while (wide != 0u) {
            const uint32_t src = static_cast<uint32_t>(__ffs(static_cast<int>(wide))) - 1u;
            wide &= wide - 1u;
            const IslandRecord w{__shfl_sync(~0u, rec.seg_off, src), __shfl_sync(~0u, rec.seg_cnt, src),
                                 __shfl_sync(~0u, rec.flags, src), __shfl_sync(~0u, rec.env, src)};
            const auto sweep = [&](bool cached) {
                SweepScalarIslandWarp(w, row_order, urows, body_inv_mass, point_masses,
                    body_lin_vel, body_ang_vel, error, sh, lane,
                    [&](uint32_t slot, RowSplit& split) { velocity_row(w, slot, cached, &split); });
            };
            if (apply_cached_impulses) sweep(true);
            for (uint32_t it = 0u; it < vel_iters; ++it) sweep(false);
            if (pos_iters == 0u) continue;
            begin_position(w, lane, warpSize);
            __syncwarp();
            for (uint32_t it = 0u; it < pos_iters; ++it)
                SweepScalarIslandWarp(w, row_order, urows, body_inv_mass, pseudo_points,
                    body_pseudo_lin, body_pseudo_ang, VelocityErrorView{}, sh, lane,
                    [&](uint32_t slot, RowSplit& split) { position_row(w, slot, &split); });
        }
    }
}

// The per-DOF scatter target: the global link row + spatial component the cooked
// dof maps route this articulation tile's DOF to (comp==~0u => a 1-DOF joint qdot).
__device__ __forceinline__ void ArticDofTarget(
    uint32_t env, uint32_t a, uint32_t k, uint32_t dof_stride,
    uint32_t links_per_dog, uint32_t base_link_count,
    const uint32_t* __restrict__ dof_to_link,
    const uint32_t* __restrict__ dof_to_component,
    uint32_t& comp, size_t& gl) {
    const size_t flat = static_cast<size_t>(env) * dof_stride + k;
    comp = dof_to_component[flat];
    const uint32_t link = dof_to_link[flat] + a * links_per_dog;
    gl = static_cast<size_t>(env) * base_link_count + link;
}

template <bool pair_driven>
__global__ void SolveRowsBlockIslandKernel(
    const NkRow* __restrict__ urows,
    float* __restrict__ lambda,
    const float* __restrict__ chain_jacobian,
    const float* __restrict__ row_minv_jt,
    const float* __restrict__ chain_jacobian_b,  // side-B (or null)
    const float* __restrict__ row_minv_jt_b,     // side-B (or null)
    const float* __restrict__ row_meff,
    const float* __restrict__ row_damping,
    float* __restrict__ qdot_flat,
    Spatial6* __restrict__ link_velocity,
    float* __restrict__ qdot,
    math::Vec3* __restrict__ body_lin_vel,
    math::Vec3* __restrict__ body_ang_vel,
    const float* __restrict__ body_inv_mass,
    const math::SymmetricMat3* __restrict__ body_world_inv_inertia,
    PointMassView point_masses,
    const uint32_t* __restrict__ islands,
    const uint32_t* __restrict__ segments,
    const uint32_t* __restrict__ row_order,
    const uint32_t* __restrict__ dof_to_link,
    const uint32_t* __restrict__ dof_to_component,
    uint32_t* __restrict__ pd_solve_scratch,  // PairDriven GLOBAL order+seg scratch
    float* __restrict__ row_penetration,    // split-impulse depth (or null)
    float* __restrict__ row_pseudo_lambda,        // split-impulse accumulator (or null)
    float* __restrict__ qdot_pseudo,              // per-link pseudo joint vel (or null)
    Spatial6* __restrict__ link_velocity_pseudo,  // per-link pseudo spatial vel (or null)
    float* __restrict__ qdot_pseudo_flat,         // per-artic pseudo tile (or null)
    math::Vec3* __restrict__ body_pseudo_lin_vel, // per-body pseudo lin vel (or null)
    math::Vec3* __restrict__ body_pseudo_ang_vel, // per-body pseudo ang vel (or null)
    math::Vec3* __restrict__ particle_pseudo_vel, // per-particle pseudo vel (or null)
    math::Vec3* __restrict__ grid_pseudo_vel,
    uint32_t total_islands,
    uint32_t rows_per_env,
    uint32_t dof_stride,
    uint32_t base_link_count,
    uint32_t artics_per_env,    // co-resident artic tiles per env (1 at K==1)
    uint32_t vel_iters,
    uint32_t pos_iters,
    float pos_beta, float pos_slop,
    float dt, float vel_tolerance,
    float baumgarte_max_velocity, bool apply_cached_impulses, VelocityErrorView error) {
    constexpr uint32_t with_b_arm = pair_driven ? 1u : 0u;
    for (uint64_t cursor = blockIdx.x; cursor < total_islands; cursor += gridDim.x) {
        const uint32_t island = static_cast<uint32_t>(cursor);
        // Cook-time island records index their color segments.
        const IslandRecord rec =
            reinterpret_cast<const IslandRecord*>(islands)[island];
        const uint32_t seg_off = rec.seg_off;
        const uint32_t seg_cnt = rec.seg_cnt;
        const uint32_t flags = rec.flags;
        const uint32_t env = rec.env;
        const uint32_t lane = threadIdx.x;
        const uint32_t env_row_base = env * rows_per_env;
        const uint32_t k_tiles = artics_per_env == 0u ? 1u : artics_per_env;
        const uint32_t env_artic_base = env * k_tiles;  // first global artic of this env

        // Shared storage belongs to this block and is reused after each island finishes.
        extern __shared__ unsigned char dyn_sh[];
        // Union islands cache row Jacobians; PairDriven islands keep them in global storage.
        const bool cache_jw = (with_b_arm == 0u);
        // the qdot region. Union: the legacy kMaxArticulationDof reservation (shared
        // footprint byte-for-byte the H1-golden carve). PairDriven: compact K-tile.
        const uint32_t qdot_floats =
            (with_b_arm != 0u) ? (k_tiles * dof_stride) : kMaxArticulationDof;
        float* const qdot_sh = reinterpret_cast<float*>(dyn_sh);
        float* const qdot_error_sh = qdot_sh + qdot_floats;
        error.qdot = qdot_error_sh;
        for (uint32_t i = lane; i < qdot_floats; i += blockDim.x) qdot_error_sh[i] = 0.0f;
        // Row data stays in each environment's global scratch; velocity tiles use shared memory.
        // A position solve adds an equally sized pseudo-velocity tile.
        const bool pos_pass = (pos_iters > 0u) && !cache_jw;
        float* const qdot_pseudo_sh = pos_pass ? (qdot_error_sh + qdot_floats) : nullptr;
        float* lambda_sh = nullptr;
        float* meff_sh = nullptr;
        float* damping_sh = nullptr;
        float* J_sh = nullptr;   // shared J/w cache (Union only).
        float* w_sh = nullptr;
        SlimRow* slim_sh = nullptr;
        uint32_t* order_sh = nullptr;
        uint32_t* seg_sh = nullptr;
        if (cache_jw) {
            lambda_sh = qdot_error_sh + qdot_floats;
            meff_sh = lambda_sh + rows_per_env;
            damping_sh = meff_sh + rows_per_env;
            J_sh = damping_sh + rows_per_env;
            w_sh = J_sh + static_cast<size_t>(rows_per_env) * dof_stride;
            slim_sh = reinterpret_cast<SlimRow*>(
                w_sh + static_cast<size_t>(rows_per_env) * dof_stride);
            order_sh = reinterpret_cast<uint32_t*>(slim_sh + rows_per_env);
            seg_sh = order_sh + rows_per_env;                        // 2R u32
        } else {
            // PairDriven islands build their ordered segments in per-environment global scratch.
            order_sh = pd_solve_scratch + static_cast<size_t>(3u) * env_row_base;
            seg_sh = order_sh + rows_per_env;                        // 2R u32
        }

        const bool has_artic = (flags & kIslandHasArticulation) != 0u && dof_stride > 0u;
        if (has_artic) {
            // load EVERY co-resident articulation tile of this env (K tiles). At
            // K==1 this is the single legacy tile (qdot_flat[env*dof_stride]).
            for (uint32_t i = lane; i < k_tiles * dof_stride; i += blockDim.x) {
                qdot_sh[i] = qdot_flat[static_cast<size_t>(env_artic_base) * dof_stride + i];
            }
        }
        // Union islands cache row parameters; PairDriven islands read them from global storage.
        if (cache_jw) {
            for (uint32_t i = lane; i < rows_per_env; i += blockDim.x) {
                const size_t g = static_cast<size_t>(env_row_base) + i;
                lambda_sh[i] = lambda[g];
                meff_sh[i] = row_meff[g];
                damping_sh[i] = row_damping[g];
                slim_sh[i] = MakeSlimRow(urows[g], env_row_base, env_artic_base);
            }
        }
        // Union: cache the per-row J/w into shared (the dense-sweep latency core).
        // PairDriven reads J/w/J_b/w_b from global in the sweep (no shared cache).
        if (cache_jw) {
            for (size_t i = lane; i < static_cast<size_t>(rows_per_env) * dof_stride;
                 i += blockDim.x) {
                const size_t g = static_cast<size_t>(env_row_base) * dof_stride + i;
                J_sh[i] = chain_jacobian[g];
                w_sh[i] = row_minv_jt[g];
            }
        }
        // Compact active rows once, preserving the order of all surviving segments.
        __shared__ uint32_t live_seg_cnt_sh;
        if (lane == 0u) {
            uint32_t out_rows = 0u;
            uint32_t out_segs = 0u;
            for (uint32_t s = 0u; s < seg_cnt; ++s) {
                const uint32_t off = segments[static_cast<size_t>(seg_off + s) * 2u + 0u];
                const uint32_t cnt = segments[static_cast<size_t>(seg_off + s) * 2u + 1u];
                const uint32_t seg_begin = out_rows;
                for (uint32_t i = 0u; i < cnt; ++i) {
                    const uint32_t gslot = row_order[off + i];
                    // slim_sh is not yet visible (no barrier) — read the flag from
                    // the GLOBAL record (one u32 per row, once per launch).
                    if (urows[gslot].flags & nk::nk_row_flags::kActive) {
                        order_sh[out_rows++] = gslot;
                    }
                }
                if (out_rows > seg_begin) {
                    seg_sh[2u * out_segs + 0u] = seg_begin;
                    seg_sh[2u * out_segs + 1u] = out_rows - seg_begin;
                    ++out_segs;
                }
            }
            live_seg_cnt_sh = out_segs;
        }
        __syncthreads();
        const uint32_t live_seg_cnt = live_seg_cnt_sh;

        const uint32_t warp = lane >> 5u;
        const uint32_t wlane = lane & 31u;
        const uint32_t nwarps = blockDim.x >> 5u;
        __shared__ uint32_t sweep_changed;

        if (with_b_arm != 0u && apply_cached_impulses) {
            for (uint32_t s = 0u; s < live_seg_cnt; ++s) {
                const uint32_t off = seg_sh[2u * s + 0u];
                const uint32_t cnt = seg_sh[2u * s + 1u];
                for (uint32_t idx = warp; idx < cnt; idx += nwarps) {
                    const uint32_t gslot = order_sh[off + idx];
                    SolveUnionRowWarp(gslot - env_row_base, gslot, env_row_base,
                                      env_artic_base, gslot, wlane, nullptr,
                                      nullptr, nullptr, nullptr, lambda, row_meff,
                                      row_damping,
                                      chain_jacobian, row_minv_jt,
                                      chain_jacobian_b, row_minv_jt_b,
                                      qdot_sh, urows, body_lin_vel, body_ang_vel,
                                      body_inv_mass, body_world_inv_inertia,
                                      point_masses,
                                      dof_stride, dt, true, error);
                }
                __syncthreads();
            }
        }
        for (uint32_t it = 0u; it < vel_iters; ++it) {
            bool warp_changed = false;
            if (lane == 0u) sweep_changed = 0u;
            __syncthreads();
            for (uint32_t s = 0u; s < live_seg_cnt; ++s) {
                const uint32_t off = seg_sh[2u * s + 0u];
                const uint32_t cnt = seg_sh[2u * s + 1u];
                for (uint32_t idx = warp; idx < cnt; idx += nwarps) {
                    const uint32_t gslot = order_sh[off + idx];
                    // Uncached islands read both sides' Jacobians at global slots.
                    const bool changed = SolveUnionRowWarp(
                        gslot - env_row_base, gslot, env_row_base, env_artic_base,
                        cache_jw ? gslot - env_row_base : gslot, wlane, slim_sh, lambda_sh,
                        meff_sh, damping_sh, lambda, row_meff, row_damping,
                        cache_jw ? J_sh : chain_jacobian, cache_jw ? w_sh : row_minv_jt,
                        chain_jacobian_b, row_minv_jt_b, qdot_sh, urows, body_lin_vel,
                        body_ang_vel, body_inv_mass, body_world_inv_inertia, point_masses,
                        dof_stride, dt, false, error, nullptr, vel_tolerance);
                    warp_changed |= changed;
                    __syncwarp();
                }
                __syncthreads();
            }
            if (warp_changed && wlane == 0u) atomicOr(&sweep_changed, 1u);
            __syncthreads();
            const bool converged = sweep_changed == 0u;
            __syncthreads();
            if (converged) break;
        }

        // Penetration drives a fresh pseudo-velocity field in the same fixed row order.
        if (pos_pass) {
            if (has_artic) {
                for (uint32_t i = lane; i < k_tiles * dof_stride; i += blockDim.x) {
                    qdot_pseudo_sh[i] = 0.0f;
                }
            }
            // Clear pseudo accumulators only for rows owned by this island.
            if (live_seg_cnt > 0u) {
                const uint32_t last_off = seg_sh[2u * (live_seg_cnt - 1u) + 0u];
                const uint32_t last_cnt = seg_sh[2u * (live_seg_cnt - 1u) + 1u];
                const uint32_t island_rows = last_off + last_cnt;
                for (uint32_t i = warp; i < island_rows; i += nwarps)
                    UpdateSpeculativePenetration<true>(order_sh[i], env_row_base, env_artic_base,
                        row_penetration, urows, chain_jacobian, chain_jacobian_b,
                        qdot_sh, body_lin_vel, body_ang_vel, point_masses, dof_stride, dt, wlane);
                for (uint32_t i = lane; i < island_rows; i += blockDim.x) {
                    row_pseudo_lambda[order_sh[i]] = 0.0f;
                }
            }
            __syncthreads();
            for (uint32_t it = 0u; it < pos_iters; ++it) {
                for (uint32_t s = 0u; s < live_seg_cnt; ++s) {
                    const uint32_t off = seg_sh[2u * s + 0u];
                    const uint32_t cnt = seg_sh[2u * s + 1u];
                    for (uint32_t idx = warp; idx < cnt; idx += nwarps) {
                        const uint32_t gslot = order_sh[off + idx];
                        SolvePositionRowWarp(
                            gslot, env_row_base, env_artic_base, gslot, wlane,
                            row_meff, row_penetration, row_pseudo_lambda,
                            chain_jacobian, row_minv_jt,
                            chain_jacobian_b, row_minv_jt_b,
                            qdot_pseudo_sh, urows,
                            body_pseudo_lin_vel, body_pseudo_ang_vel,
                            body_inv_mass, body_world_inv_inertia,
                            point_masses.Pseudo(particle_pseudo_vel, grid_pseudo_vel),
                            dof_stride, pos_beta, pos_slop, dt,
                            baumgarte_max_velocity);
                        __syncwarp();
                    }
                    __syncthreads();
                }
            }
        }

        // Every warp waits for ordered position updates before scattering shared velocities.
        __syncthreads();

        // Write back this island's cached lambdas; globally stored lambdas are already current.
        if (cache_jw && live_seg_cnt > 0u) {
            const uint32_t last_off = seg_sh[2u * (live_seg_cnt - 1u) + 0u];
            const uint32_t last_cnt = seg_sh[2u * (live_seg_cnt - 1u) + 1u];
            const uint32_t island_rows = last_off + last_cnt;
            for (uint32_t i = lane; i < island_rows; i += blockDim.x) {
                const uint32_t gslot = order_sh[i];
                lambda[gslot] = lambda_sh[gslot - env_row_base];
            }
        }

        // Scatter the solved velocity tiles through the cooked DOF maps.
        if (has_artic) {
            if (with_b_arm == 0u) {
                for (uint32_t k = lane; k < dof_stride; k += blockDim.x) {
                    const size_t flat = static_cast<size_t>(env) * dof_stride + k;
                    const uint32_t link = dof_to_link[flat];
                    const uint32_t comp = dof_to_component[flat];
                    const size_t gl = static_cast<size_t>(env) * base_link_count + link;
                    const float v = qdot_sh[k];
                    if (comp != ~0u) {
                        link_velocity[gl].v[comp] = v;
                    } else {
                        qdot[gl] = v;
                    }
                    qdot_flat[flat] = v;
                }
            } else {
                // Homogeneous articulation tiles occupy contiguous link ranges within an environment.
                const uint32_t links_per_dog =
                    (k_tiles > 0u) ? (base_link_count / k_tiles) : base_link_count;
                for (uint32_t i = lane; i < k_tiles * dof_stride; i += blockDim.x) {
                    const uint32_t a = i / dof_stride;       // env-local tile
                    const uint32_t k = i - a * dof_stride;   // DOF within tile
                    uint32_t comp; size_t gl;
                    ArticDofTarget(env, a, k, dof_stride, links_per_dog, base_link_count,
                                   dof_to_link, dof_to_component, comp, gl);
                    const float v = qdot_sh[static_cast<size_t>(a) * dof_stride + k];
                    if (comp != ~0u) {
                        link_velocity[gl].v[comp] = v;
                    } else {
                        qdot[gl] = v;
                    }
                    qdot_flat[static_cast<size_t>(env_artic_base + a) * dof_stride + k] = v;
                    // Scatter pseudo velocity to separate buffers consumed by position integration.
                    if (pos_pass) {
                        const float vp = qdot_pseudo_sh[static_cast<size_t>(a) * dof_stride + k];
                        if (comp != ~0u) {
                            link_velocity_pseudo[gl].v[comp] = vp;
                        } else {
                            qdot_pseudo[gl] = vp;
                        }
                        qdot_pseudo_flat[static_cast<size_t>(env_artic_base + a) *
                                             dof_stride + k] = vp;
                    }
                }
            }
        }
        __syncthreads();
    }
}

// Control words change inside a cooperative launch, so reads past a grid barrier bypass L1.
__device__ inline uint32_t LoadControl(const uint32_t* word) {
    return *reinterpret_cast<const volatile uint32_t*>(word);
}

// Words that other threads also clear inside the same barrier interval use atomic stores.
__device__ inline void ClearShared(uint32_t* word) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(*word).store(0u, cuda::memory_order_relaxed);
}

// Lane k of warp w tests position w + k * warps, so the positions take accepts spread over every
// warp of the grid and each warp visits its own in order. Visits also get the position's index m
// among its warp's positions (position = w + m * warps).
template <typename Take, typename Visit>
__device__ void ForEachLivePosition(uint32_t live, Take&& take, Visit&& visit) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t stride = gridDim.x * blockDim.x;
    const uint32_t warps = stride / 32u;
    uint32_t first = 0u;
    for (uint32_t base = (blockIdx.x * blockDim.x + threadIdx.x) / 32u; base < live;
         base += stride, first += 32u) {
        const uint32_t position = base + lane * warps;
        uint32_t mask = __ballot_sync(0xffffffffu, position < live && take(position));
        while (mask != 0u) {
            const uint32_t bit = static_cast<uint32_t>(__ffs(static_cast<int>(mask))) - 1u;
            mask &= mask - 1u;
            visit(base + bit * warps, first + bit);
        }
    }
}

__device__ inline LaneOwners LiveRowOwners(const NkRow* urows, uint32_t slot, PointMassView points,
                                           const float* body_inv_mass, const ColorScratch& s,
                                           uint32_t lane) {
    return LoadLaneOwners(LoadRowWarp(urows, slot, lane), points, body_inv_mass, s, lane);
}

// Whole-word stores clear every owner's color words; no other thread writes them meanwhile.
__device__ inline void ClearOwnerColors(const ColorScratch& s, uint32_t* words) {
    static_assert(kColorWords == 4u, "one vector store per owner");
    const uint64_t owners = uint64_t{s.bodies} + s.particles + s.grid + s.rows;
    uint4* const vec = reinterpret_cast<uint4*>(words);
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < owners;
         i += uint64_t{gridDim.x} * blockDim.x)
        vec[i] = make_uint4(0u, 0u, 0u, 0u);
}

// One pass of iterated greedy recoloring over an epoch's classes: even passes take classes
// from the highest color down, odd passes the largest first. into must start empty.
__device__ __noinline__ void ReinsertColorClasses(const NkRow* urows, PointMassView points,
                                                  const float* body_inv_mass,
                                                  const uint32_t* live_order, ColorScratch s,
                                                  uint32_t epoch, uint32_t pass, uint32_t live,
                                                  uint32_t* into) {
    const auto grid = cooperative_groups::this_grid();
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t threads = gridDim.x * blockDim.x;
    const uint32_t warp = thread / 32u;
    const uint32_t warps = threads / 32u;
    // Palette-sized scratch the final color listing rewrites later.
    uint32_t* const count = s.color_lane;
    uint32_t* const start = s.color_tail;
    uint32_t* const lane_end = s.color_pair;
    uint32_t* const warp_start = s.color_position;
    uint32_t* const order = s.color_slot;
    const LaneOwnerTable table(s);
    for (uint32_t c = thread; c < kColorPalette; c += threads) count[c] = 0u;
    grid.sync();
    for (uint32_t position = thread; position < live; position += threads) {
        const uint32_t color = s.pos_color[position];
        if (color / kColorPalette == epoch) atomicAdd(count + color % kColorPalette, 1u);
    }
    grid.sync();
    if (blockIdx.x == 0u) {
        static_assert(kColorPalette <= kColorBlockSize, "one ranking thread per class");
        const uint32_t c = threadIdx.x;
        if (c < kColorPalette) order[c] = kOwnerEmpty;
        __syncthreads();
        if (c < kColorPalette) {
            const uint32_t size = count[c];
            uint32_t rank = 0u, first = 0u;
            for (uint32_t d = 0u; d < kColorPalette; ++d) {
                const uint32_t other = count[d];
                if (d < c) first += other;
                if (other != 0u && (pass % 2u == 0u ? d > c
                                                    : other > size || (other == size && d < c)))
                    ++rank;
            }
            if (size != 0u) order[rank] = c;
            start[c] = lane_end[c] = first;
            warp_start[c] = first + size;
        }
    }
    grid.sync();
    // A class lists its lane rows from its start and its other rows down from its end.
    for (uint32_t position = thread; position < live; position += threads) {
        const uint32_t color = s.pos_color[position];
        if (color / kColorPalette != epoch) continue;
        const uint32_t c = color % kColorPalette;
        s.color_rows[table.Count(position) != 0u
            ? atomicAdd(lane_end + c, 1u) : atomicSub(warp_start + c, 1u) - 1u] = position;
    }
    grid.sync();
    for (uint32_t k = 0u; k < kColorPalette; ++k) {
        const uint32_t c = LoadControl(order + k);
        if (c == kOwnerEmpty) break;
        const uint32_t first = start[c];
        const uint32_t middle = lane_end[c];
        const uint32_t end = first + count[c];
        for (uint32_t i = first + thread; i < middle; i += threads) {
            const uint32_t position = s.color_rows[i];
            const uint32_t owners = table.Count(position);
            uint32_t taken[kColorWords] = {};
            for (uint32_t j = 0u; j < owners; ++j) {
                const uint32_t owner = table.Owner(j, position);
                #pragma unroll
                for (uint32_t w = 0u; w < kColorWords; ++w)
                    taken[w] |= into[size_t{owner} * kColorWords + w];
            }
            const uint32_t color = FirstFreeColor(taken, 0u);
            for (uint32_t j = 0u; j < owners; ++j)
                atomicOr(into + size_t{table.Owner(j, position)} * kColorWords + color / 32u,
                         1u << (color % 32u));
            s.pos_color[position] = epoch * kColorPalette + color;
        }
        // Warp rows take warps from the last one down, beside the lane rows.
        for (uint32_t i = middle + (warps - 1u - warp); i < end; i += warps) {
            const uint32_t position = s.color_rows[i];
            const LaneOwners owners =
                LiveRowOwners(urows, live_order[position], points, body_inv_mass, s, lane);
            uint32_t taken[kColorWords] = {};
            VisitLaneOwners(owners, [&](uint32_t owner) {
                #pragma unroll
                for (uint32_t w = 0u; w < kColorWords; ++w)
                    taken[w] |= into[size_t{owner} * kColorWords + w];
            });
            #pragma unroll
            for (uint32_t w = 0u; w < kColorWords; ++w)
                taken[w] = __reduce_or_sync(0xffffffffu, taken[w]);
            const uint32_t color = FirstFreeColor(taken, 0u);
            VisitLaneOwners(owners, [&](uint32_t owner) {
                atomicOr(into + size_t{owner} * kColorWords + color / 32u, 1u << (color % 32u));
            });
            if (lane == 0u) s.pos_color[position] = epoch * kColorPalette + color;
        }
        grid.sync();
    }
}

// Lane rows repack every launch, since assembly rewrites their rhs and effective mass.
__device__ __noinline__ void PackLaneRecords(const NkRow* urows, PointMassView points,
                                             ColorScratch s, const float* row_meff,
                                             const float* row_damping, float dt) {
    const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t threads = gridDim.x * blockDim.x;
    const uint32_t colors = LoadControl(s.control + kControlColors);
    for (uint32_t color = 0u; color < colors; ++color) {
        const uint32_t end = s.color_lane[color];
        for (uint32_t i = s.color_start[color] + thread; i < end; i += threads) {
            const uint32_t slot = s.color_rows[i];
            LaneRecord record;
            PackLaneRecord(urows[slot], slot, row_meff[slot], row_damping[slot], dt, points,
                           record);
            s.lane[i] = record;
        }
    }
}

struct LivePrepareArgs {
    const float* lambda = nullptr;
    const float* row_meff = nullptr;
    const float* row_damping = nullptr;
    const float* chain_jacobian = nullptr;
    const float* chain_jacobian_b = nullptr;
    const float* qdot_flat = nullptr;
    const math::Vec3* body_linear = nullptr;
    const math::Vec3* body_angular = nullptr;
    const uint32_t* row_order = nullptr;
    const uint32_t* cc_root = nullptr;
    const uint32_t* cc_artic_first = nullptr;
    const float* row_penetration = nullptr;
    float* row_pseudo_lambda = nullptr;
    uint32_t total_rows = 0u;
    uint32_t rows_per_env = 0u;
    uint32_t artics_per_env = 0u;
    uint32_t dof_stride = 0u;
    uint32_t bodies_per_env = 0u;
    uint32_t owner_cache_slots = 0u;  // per warp, in the launch's dynamic shared memory
    float dt = 0.0f;
    float pos_slop = 0.0f;
    bool reuse_schedule = false;
    bool verify_idle = false;
};

// The first owner_cache_slots positions of each warp keep their owners in shared memory, one per
// lane. Later positions, or rows with more owners than lanes, reload them from the row per visit.
struct WarpOwnerCache {
    uint32_t* owners = nullptr;
    uint32_t slots = 0u;
    uint32_t filled = 0u;  // bit m: position m holds its owners

    __device__ bool Holds(uint32_t m) const { return m < slots && ((filled >> m) & 1u) != 0u; }
    __device__ void Store(uint32_t m, const LaneOwners& lane_owners, uint32_t lane) {
        if (m >= slots || !lane_owners.packed ||
            !__all_sync(0xffffffffu, lane_owners.second == kOwnerEmpty))
            return;
        owners[m * 32u + lane] = lane_owners.first;
        filled |= 1u << m;
    }
    __device__ LaneOwners Load(uint32_t m, uint32_t lane) const {
        LaneOwners result;
        result.first = owners[m * 32u + lane];
        return result;
    }
};

// Each lane screens one of 32 consecutive rows; rows that no impulse or penetration decides
// then take the warp's velocity test one after another.
__device__ __noinline__ void MarkLiveRows(const NkRow* urows, PointMassView points,
                                          IslandActivityView activity, uint32_t* live_scan,
                                          ColorScratch s, LivePrepareArgs prep) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp = thread >> 5u;
    const uint32_t warp_stride = gridDim.x * (blockDim.x >> 5u);
    uint32_t valid_end = 0u;
    for (uint32_t base = warp * 32u; base < prep.total_rows; base += warp_stride * 32u) {
        const uint32_t cursor = base + lane;
        const uint32_t root = cursor < prep.total_rows ? activity.sorted_roots[cursor] : ~0u;
        const uint32_t valid = __ballot_sync(0xffffffffu, root != ~0u);
        if (valid == 0u) break;
        valid_end = base + 32u - static_cast<uint32_t>(__clz(static_cast<int>(valid)));
        uint32_t slot = 0u;
        bool active = false;
        bool decided = true;
        if (root != ~0u) {
            slot = prep.row_order[cursor];
            if (prep.row_pseudo_lambda != nullptr) prep.row_pseudo_lambda[slot] = 0.0f;
            const uint32_t flags = urows[slot].flags;
            if (prep.row_penetration != nullptr && !(flags & nk::nk_row_flags::kVelocityOnly))
                active = prep.row_penetration[slot] > prep.pos_slop;
            if (!active && (flags & nk::nk_row_flags::kActive)) {
                active = prep.lambda[slot] != 0.0f ||
                         (flags & (nk::nk_row_flags::kFriction | nk::nk_row_flags::kMaterialBlock |
                                   nk::nk_row_flags::kVertexBlock));
                if (!active && (flags & nk::nk_row_flags::kBlockNormal)) {
                    const uint32_t count = urows[slot].group_normal_count;
                    active = prep.lambda[slot + count] != 0.0f ||
                             prep.lambda[slot + 2u * count] != 0.0f;
                }
                decided = active;
            }
        }
        // Each row's words load while the warp tests the row before it.
        uint32_t pending = __ballot_sync(0xffffffffu, !decided);
        uint32_t owner = static_cast<uint32_t>(__ffs(static_cast<int>(pending))) - 1u;
        uint32_t row_slot = __shfl_sync(0xffffffffu, slot, owner & 31u);
        uint32_t word = pending != 0u ? RowWordWarp(urows, row_slot, lane) : 0u;
        while (pending != 0u) {
            const uint32_t next = pending & (pending - 1u);
            const uint32_t next_owner = static_cast<uint32_t>(__ffs(static_cast<int>(next))) - 1u;
            const uint32_t next_slot = __shfl_sync(0xffffffffu, slot, next_owner & 31u);
            const uint32_t next_word = next != 0u ? RowWordWarp(urows, next_slot, lane) : 0u;
            const uint32_t env = row_slot / prep.rows_per_env;
            const uint32_t env_artic_base = env * prep.artics_per_env;
            const float* qdot = prep.qdot_flat != nullptr
                ? prep.qdot_flat + static_cast<size_t>(env_artic_base) * prep.dof_stride : nullptr;
            const NkRow row = BroadcastRowWarp(word);
            const bool needed = RowVelocityNeedsSolveWarp(urows, row, row_slot,
                env * prep.rows_per_env, env_artic_base, prep.lambda, prep.row_meff,
                prep.row_damping, prep.chain_jacobian, prep.chain_jacobian_b, qdot,
                prep.body_linear, prep.body_angular, points, prep.dof_stride, prep.dt, lane);
            if (lane == owner) active = needed;
            pending = next;
            owner = next_owner;
            row_slot = next_slot;
            word = next_word;
        }
        if (root != ~0u) live_scan[cursor] = active ? 1u : 0u;
        if (root != ~0u && active) atomicOr(&activity.needs_solve[root], kIslandNeedsSolve);
    }
    if (lane == 0u && valid_end != 0u)
        atomicMax(s.control + kControlValidRows, valid_end);
}

__global__ void CountHubRowsKernel(const NkRow* __restrict__ urows, uint32_t total_rows,
                                   const float* __restrict__ body_inv_mass, ColorScratch s) {
    for (uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x; slot < total_rows;
         slot += gridDim.x * blockDim.x) {
        const uint32_t flags = urows[slot].flags;
        if (!(flags & nk::nk_row_flags::kActive) || (flags & nk::nk_row_flags::kBlockTangent))
            continue;
        for (uint32_t index = 0u; index < 2u; ++index) {
            const NkRowSide& side = index == 0u ? urows[slot].a : urows[slot].b;
            if (side.kind == kNkSideRigid && side.index < s.bodies && body_inv_mass[side.index] > 0.0f)
                atomicAdd(s.hub_count + side.index, 1u);
        }
    }
}

// Coloring and the sweeps read this inverse mass, so no row writes a hub or takes it as owner.
__global__ void MaskHubOwnersKernel(const float* __restrict__ body_inv_mass, ColorScratch s) {
    for (uint32_t body = blockIdx.x * blockDim.x + threadIdx.x; body < s.bodies;
         body += gridDim.x * blockDim.x)
        s.owner_inv_mass[body] = s.hub_count[body] > kHubRows ? 0.0f : body_inv_mass[body];
}

// Pending rows pick a color free at all their owners; among rows picking one color at a shared
// owner only the highest priority keeps it. Full palettes open a new epoch of colors.
__global__ void __launch_bounds__(kColorBlockSize) PrepareLiveColorsKernel(
    const NkRow* __restrict__ urows, PointMassView points, const float* __restrict__ body_inv_mass,
    const uint32_t* __restrict__ islands, const uint32_t* __restrict__ island_count_dev,
    IslandActivityView activity, uint32_t* __restrict__ live_scan,
    uint32_t* __restrict__ live_order, ColorScratch s, LivePrepareArgs prep) {
    using BlockScanT = cub::BlockScan<uint32_t, kColorBlockSize>;
    static_assert(kColorSlots % kColorBlockSize == 0u, "colors split evenly over the scan block");
    __shared__ typename BlockScanT::TempStorage scan_temp;
    __shared__ uint32_t histogram[kColorSlots];
    __shared__ uint32_t block_chain;
    extern __shared__ uint32_t owner_cache_shared[];
    const auto grid = cooperative_groups::this_grid();
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t threads = gridDim.x * blockDim.x;
    // A verifying pass keeps the schedule it continues unless an idle row broke.
    if (prep.verify_idle && s.control[kControlIdleViolations] == 0u) return;
    const uint32_t previous_live = prep.reuse_schedule ? s.control[kControlLive] : 0u;
    if (thread == 0u) {
        s.control[kControlValidRows] = 0u;
        s.control[kControlScheduleChanged] = prep.reuse_schedule ? 0u : 1u;
    }
    // Islands stay active through verification, so every one still advances its positions.
    if (!prep.verify_idle)
        for (uint32_t i = thread; i < prep.total_rows; i += threads)
            activity.needs_solve[i] &= ~kIslandNeedsSolve;
    grid.sync();

    MarkLiveRows(urows, points, activity, live_scan, s, prep);
    grid.sync();

    const uint32_t valid_rows = s.control[kControlValidRows];
    const uint32_t tiles = (valid_rows + kColorBlockSize - 1u) / kColorBlockSize;
    const uint32_t tiles_per_block = (tiles + gridDim.x - 1u) / gridDim.x;
    const uint32_t begin_row = blockIdx.x * tiles_per_block * kColorBlockSize;
    const uint32_t end_row = min(valid_rows,
        begin_row + tiles_per_block * kColorBlockSize);
    uint32_t running = 0u;
    for (uint32_t base = begin_row; base < end_row; base += kColorBlockSize) {
        const uint32_t idx = base + threadIdx.x;
        // Island build flags are read after marking has finished updating the same words.
        const uint32_t root = idx < valid_rows ? activity.sorted_roots[idx] : ~0u;
        const uint32_t flag = root != ~0u && live_scan[idx] != 0u &&
            (activity.needs_solve[root] & kIslandWarpWork) != 0u ? 1u : 0u;
        uint32_t inclusive = 0u, tile_total = 0u;
        BlockScanT(scan_temp).InclusiveSum(flag, inclusive, tile_total);
        if (idx < valid_rows) {
            s.pos_color[idx] = flag;
            live_scan[idx] = running + inclusive;
        }
        running += tile_total;
        __syncthreads();
    }
    if (threadIdx.x == 0u) s.block_count[blockIdx.x] = running;
    grid.sync();
    if (thread == 0u) {
        uint32_t total = 0u;
        for (uint32_t block = 0u; block < gridDim.x; ++block) {
            const uint32_t count = s.block_count[block];
            s.block_count[block] = total;
            total += count;
        }
        s.control[kControlLive] = total;
        if (total != previous_live) s.control[kControlScheduleChanged] = 1u;
    }
    grid.sync();
    const uint32_t row_base = s.block_count[blockIdx.x];
    for (uint32_t idx = begin_row + threadIdx.x; idx < end_row; idx += kColorBlockSize) {
        const uint32_t prefix = row_base + live_scan[idx];
        live_scan[idx] = prefix;
        if (s.pos_color[idx] != 0u) {
            const uint32_t slot = prep.row_order[idx];
            if (prep.reuse_schedule && prefix <= previous_live && live_order[prefix - 1u] != slot)
                atomicOr(s.control + kControlScheduleChanged, 1u);
            live_order[prefix - 1u] = slot;
        }
    }
    grid.sync();

    const uint32_t live = s.control[kControlLive];
    // Verification continues the compensated articulation velocities it inherits.
    if (!prep.verify_idle)
        for (uint32_t i = thread; i < s.artic_dofs; i += threads) s.qdot_error[i] = 0.0f;
    if (thread < 3u) s.control[kControlChanged + thread] = 0u;
    if (LoadControl(s.control + kControlScheduleChanged) == 0u) {
        PackLaneRecords(urows, points, s, prep.row_meff, prep.row_damping, prep.dt);
        return;
    }
    for (uint32_t i = thread; i < kColorSlots; i += threads)
        s.color_cursor[i] = s.color_pair_count[i] = s.color_position_count[i] = 0u;
    if (thread > kControlLive && thread < kControlScheduleChanged) s.control[thread] = 0u;
    WarpOwnerCache cache;
    cache.slots = min(prep.owner_cache_slots, 32u);
    cache.owners = owner_cache_shared + (threadIdx.x / 32u) * cache.slots * 32u;
    const auto owners_at = [&](uint32_t position, uint32_t m) {
        return cache.Holds(m) ? cache.Load(m, lane)
                              : LiveRowOwners(urows, live_order[position], points,
                                              body_inv_mass, s, lane);
    };
    // A lane row keeps its few owners by position and colors on one thread; a lane row with an
    // owner outside the dense ranges falls to the warp path, which leaves it to the chain.
    const LaneOwnerTable table(s);
    for (uint32_t position = thread; position < live; position += threads) {
        const NkRow row = urows[live_order[position]];
        uint32_t count = 0u;
        if (IsLaneRow(row, points)) {
            bool packed = true;
            VisitRowOwners(row, points, body_inv_mass, s, [&](uint32_t owner) {
                if (owner == kOwnerEmpty) packed = false;
                else table.Owner(count++, position) = owner;
            });
            if (!packed) count = 0u;
        }
        table.Count(position) = count;
        if (count != 0u) {
            s.pos_color[position] = kColorPending;
            s.pos_tent[position] = kOwnerEmpty;
        }
    }
    grid.sync();
    const auto warp_row = [&](uint32_t position) { return table.Count(position) == 0u; };
    ForEachLivePosition(live, warp_row, [&](uint32_t position, uint32_t m) {
        const LaneOwners owners =
            LiveRowOwners(urows, live_order[position], points, body_inv_mass, s, lane);
        cache.Store(m, owners, lane);
        if (lane == 0u) {
            const bool sequential = IsSequentialArticRow(urows[live_order[position]], s);
            s.pos_color[position] = owners.packed && !sequential ? kColorPending : kColorChain;
            s.pos_tent[position] = kOwnerEmpty;
        }
    });
    grid.sync();

    const auto tentative = [&](uint32_t position) {
        return table.Count(position) == 0u && s.pos_tent[position] != kOwnerEmpty;
    };
    // Lane rows run each round phase on one thread each, beside the warp rows.
    const auto for_lane_rows = [&](bool tentative_rows, auto&& visit) {
        for (uint32_t position = thread; position < live; position += threads) {
            const uint32_t owners = table.Count(position);
            if (owners == 0u) continue;
            if (tentative_rows ? s.pos_tent[position] == kOwnerEmpty
                               : s.pos_color[position] != kColorPending)
                continue;
            visit(position, owners);
        }
    };
    uint32_t round = 0u;
    for (uint32_t epoch = 0u; epoch < kColorEpochs; ++epoch) {
        bool pending = true;
        for (uint32_t step = 0u; pending && step < kColorRounds; ++step, ++round) {
            uint32_t* const pending_flag = s.control + kControlPending + round % 3u;
            if (thread == 0u) s.control[kControlPending + (round + 1u) % 3u] = 0u;
            bool overflow = false;
            for_lane_rows(false, [&](uint32_t position, uint32_t owners) {
                const uint32_t slot = live_order[position];
                uint32_t taken[kColorWords] = {};
                for (uint32_t j = 0u; j < owners; ++j) {
                    const uint32_t owner = table.Owner(j, position);
                    #pragma unroll
                    for (uint32_t w = 0u; w < kColorWords; ++w)
                        taken[w] |= s.used[size_t{owner} * kColorWords + w];
                }
                const uint32_t color = FirstFreeColor(
                    taken, MixSlot((slot % prep.rows_per_env) ^ (round * 0x9e3779b9u)) % kColorPalette);
                if (color == kColorPalette) {
                    overflow = true;
                    s.pos_color[position] = kColorOverflow;
                    return;
                }
                const uint32_t bit = 1u << (color % 32u);
                for (uint32_t j = 0u; j < owners; ++j) {
                    const size_t at = size_t{table.Owner(j, position)} * kColorWords + color / 32u;
                    if (atomicOr(s.tent + at, bit) & bit) atomicOr(s.dup + at, bit);
                }
                s.pos_tent[position] = color;
            });
            ForEachLivePosition(live,
                [&](uint32_t position) {
                    return warp_row(position) && s.pos_color[position] == kColorPending;
                },
                [&](uint32_t position, uint32_t m) {
                    const uint32_t slot = live_order[position];
                    const LaneOwners owners = owners_at(position, m);
                    uint32_t taken[kColorWords] = {};
                    VisitLaneOwners(owners, [&](uint32_t owner) {
                        #pragma unroll
                        for (uint32_t w = 0u; w < kColorWords; ++w)
                            taken[w] |= s.used[size_t{owner} * kColorWords + w];
                    });
                    #pragma unroll
                    for (uint32_t w = 0u; w < kColorWords; ++w)
                        taken[w] = __reduce_or_sync(0xffffffffu, taken[w]);
                    const uint32_t color = FirstFreeColor(
                        taken, MixSlot((slot % prep.rows_per_env) ^ (round * 0x9e3779b9u)) % kColorPalette);
                    if (color == kColorPalette) {
                        overflow = true;
                        if (lane == 0u) s.pos_color[position] = kColorOverflow;
                        return;
                    }
                    const uint32_t bit = 1u << (color % 32u);
                    VisitLaneOwners(owners, [&](uint32_t owner) {
                        const size_t at = size_t{owner} * kColorWords + color / 32u;
                        if (atomicOr(s.tent + at, bit) & bit) atomicOr(s.dup + at, bit);
                    });
                    if (lane == 0u) s.pos_tent[position] = color;
                });
            if (__syncthreads_or(overflow) && threadIdx.x == 0u)
                atomicOr(s.control + kControlOverflow + epoch % 2u, 1u);
            grid.sync();

            for_lane_rows(true, [&](uint32_t position, uint32_t owners) {
                const uint32_t color = s.pos_tent[position];
                const uint32_t priority = MixSlot(live_order[position] % prep.rows_per_env);
                for (uint32_t j = 0u; j < owners; ++j) {
                    const uint32_t owner = table.Owner(j, position);
                    if (s.dup[size_t{owner} * kColorWords + color / 32u] & (1u << (color % 32u)))
                        atomicMax(s.claim + owner, priority);
                }
            });
            ForEachLivePosition(live, tentative, [&](uint32_t position, uint32_t m) {
                const uint32_t slot = live_order[position];
                const LaneOwners owners = owners_at(position, m);
                const uint32_t color = s.pos_tent[position];
                const uint32_t priority = MixSlot(slot % prep.rows_per_env);
                VisitLaneOwners(owners, [&](uint32_t owner) {
                    if (s.dup[size_t{owner} * kColorWords + color / 32u] & (1u << (color % 32u)))
                        atomicMax(s.claim + owner, priority);
                });
            });
            grid.sync();

            bool left = false;
            for_lane_rows(true, [&](uint32_t position, uint32_t owners) {
                const uint32_t color = s.pos_tent[position];
                const uint32_t priority = MixSlot(live_order[position] % prep.rows_per_env);
                const uint32_t bit = 1u << (color % 32u);
                bool keep = true;
                for (uint32_t j = 0u; j < owners; ++j) {
                    const uint32_t owner = table.Owner(j, position);
                    if ((s.dup[size_t{owner} * kColorWords + color / 32u] & bit) &&
                        s.claim[owner] != priority)
                        keep = false;
                }
                if (!keep) {
                    left = true;
                    return;
                }
                for (uint32_t j = 0u; j < owners; ++j)
                    atomicOr(s.used + size_t{table.Owner(j, position)} * kColorWords + color / 32u,
                             bit);
                s.pos_color[position] = epoch * kColorPalette + color;
            });
            ForEachLivePosition(live, tentative, [&](uint32_t position, uint32_t m) {
                const uint32_t slot = live_order[position];
                const LaneOwners owners = owners_at(position, m);
                const uint32_t color = s.pos_tent[position];
                const uint32_t priority = MixSlot(slot % prep.rows_per_env);
                const uint32_t bit = 1u << (color % 32u);
                bool keep = true;
                VisitLaneOwners(owners, [&](uint32_t owner) {
                    if ((s.dup[size_t{owner} * kColorWords + color / 32u] & bit) &&
                        s.claim[owner] != priority)
                        keep = false;
                });
                if (__all_sync(0xffffffffu, keep)) {
                    VisitLaneOwners(owners, [&](uint32_t owner) {
                        atomicOr(s.used + size_t{owner} * kColorWords + color / 32u, bit);
                    });
                    if (lane == 0u) s.pos_color[position] = epoch * kColorPalette + color;
                } else {
                    left = true;
                }
            });
            if (__syncthreads_or(left) && threadIdx.x == 0u) atomicOr(pending_flag, 1u);
            grid.sync();

            // Every picked word held only this round's picks, so clearing whole words is exact.
            for_lane_rows(true, [&](uint32_t position, uint32_t owners) {
                const uint32_t color = s.pos_tent[position];
                for (uint32_t j = 0u; j < owners; ++j) {
                    const uint32_t owner = table.Owner(j, position);
                    const size_t at = size_t{owner} * kColorWords + color / 32u;
                    ClearShared(s.tent + at);
                    ClearShared(s.dup + at);
                    ClearShared(s.claim + owner);
                }
                s.pos_tent[position] = kOwnerEmpty;
            });
            ForEachLivePosition(live, tentative, [&](uint32_t position, uint32_t m) {
                const LaneOwners owners = owners_at(position, m);
                const uint32_t color = s.pos_tent[position];
                VisitLaneOwners(owners, [&](uint32_t owner) {
                    const size_t at = size_t{owner} * kColorWords + color / 32u;
                    ClearShared(s.tent + at);
                    ClearShared(s.dup + at);
                    ClearShared(s.claim + owner);
                });
                __syncwarp();
                if (lane == 0u) s.pos_tent[position] = kOwnerEmpty;
            });
            grid.sync();
            pending = LoadControl(pending_flag) != 0u;
        }
        // Passes reinsert whole classes first-fit into the other word array; rows of a class
        // share no owner, so a class places at once and the color count never grows.
        uint32_t* from = s.used;
        uint32_t* into = s.tent;
        for (uint32_t pass = 0u; live != 0u && pass < kRecolorPasses; ++pass) {
            ReinsertColorClasses(urows, points, body_inv_mass, live_order, s, epoch, pass, live,
                                 into);
            ClearOwnerColors(s, from);
            grid.sync();
            uint32_t* const swap = from;
            from = into;
            into = swap;
        }
        const bool overflow = LoadControl(s.control + kControlOverflow + epoch % 2u) != 0u;
        const bool last = !overflow || pending || epoch + 1u == kColorEpochs;
        ClearOwnerColors(s, from);
        for (uint32_t position = thread; position < live; position += threads) {
            const uint32_t color = s.pos_color[position];
            if (color == kColorPending || color == kColorOverflow)
                s.pos_color[position] = last ? kColorChain : kColorPending;
        }
        if (thread == 0u) s.control[kControlOverflow + (epoch + 1u) % 2u] = 0u;
        grid.sync();
        if (last) break;
    }

    // Each block counts a contiguous chunk, so chain rows keep their live order.
    const uint32_t chunk = (live + gridDim.x - 1u) / gridDim.x;
    const uint32_t begin = static_cast<uint32_t>(
        min(uint64_t{live}, uint64_t{blockIdx.x} * chunk));
    const uint32_t end = min(live, begin + chunk);
    for (uint32_t c = threadIdx.x; c < kColorSlots; c += blockDim.x) histogram[c] = 0u;
    if (threadIdx.x == 0u) block_chain = 0u;
    __syncthreads();
    uint32_t chained = 0u;
    for (uint32_t position = begin + threadIdx.x; position < end; position += blockDim.x) {
        const uint32_t color = s.pos_color[position];
        if (color >= kColorSlots) {
            ++chained;
            continue;
        }
        atomicAdd(histogram + color, 1u);
        const NkRow& row = urows[live_order[position]];
        if (IsPairRow(row)) atomicAdd(s.color_pair_count + color, 1u);
        if (IsPositionRow(row)) atomicAdd(s.color_position_count + color, 1u);
    }
    chained = __reduce_add_sync(0xffffffffu, chained);
    if (lane == 0u && chained != 0u) atomicAdd(&block_chain, chained);
    __syncthreads();
    for (uint32_t c = threadIdx.x; c < kColorSlots; c += blockDim.x)
        if (histogram[c] != 0u) atomicAdd(s.color_cursor + c, histogram[c]);
    if (threadIdx.x == 0u) s.block_count[blockIdx.x] = block_chain;
    grid.sync();

    if (blockIdx.x == 0u) {
        constexpr uint32_t per_thread = kColorSlots / kColorBlockSize;
        uint32_t counts[per_thread];
        uint32_t rows = 0u;
        uint32_t filled = 0u;
        #pragma unroll
        for (uint32_t j = 0u; j < per_thread; ++j) {
            counts[j] = s.color_cursor[threadIdx.x * per_thread + j];
            rows += counts[j];
            filled += counts[j] != 0u ? 1u : 0u;
        }
        uint32_t row_start = 0u, color_index = 0u, total_rows = 0u, total_colors = 0u;
        BlockScanT(scan_temp).ExclusiveSum(rows, row_start, total_rows);
        __syncthreads();
        BlockScanT(scan_temp).ExclusiveSum(filled, color_index, total_colors);
        #pragma unroll
        for (uint32_t j = 0u; j < per_thread; ++j) {
            if (counts[j] == 0u) continue;
            const uint32_t palette = threadIdx.x * per_thread + j;
            const uint32_t pairs = s.color_pair_count[palette];
            s.color_slot[color_index] = palette;
            s.color_position[color_index] = s.color_position_count[palette];
            s.color_pair[color_index] = row_start + counts[j] - pairs;
            s.color_start[color_index++] = row_start;
            s.color_cursor[palette] = row_start;
            row_start += counts[j];
            s.color_tail[palette] = row_start - pairs;
            s.color_pair_tail[palette] = row_start;
        }
        if (threadIdx.x == 0u) {
            s.color_start[total_colors] = total_rows;
            s.control[kControlColors] = total_colors;
        }
        __syncthreads();
    }
    uint32_t earlier = 0u;
    for (uint32_t b = threadIdx.x; b < blockIdx.x; b += blockDim.x) earlier += s.block_count[b];
    uint32_t ignored = 0u, chain_base = 0u;
    BlockScanT(scan_temp).ExclusiveSum(earlier, ignored, chain_base);
    __syncthreads();
    for (uint32_t tile = begin; tile < end; tile += blockDim.x) {
        const uint32_t position = tile + threadIdx.x;
        const uint32_t flag = position < end && s.pos_color[position] == kColorChain ? 1u : 0u;
        uint32_t offset = 0u, tile_total = 0u;
        BlockScanT(scan_temp).ExclusiveSum(flag, offset, tile_total);
        if (position < end) s.chain_excl[position] = chain_base + offset;
        if (flag != 0u) s.chain_rows[chain_base + offset] = live_order[position];
        chain_base += tile_total;
        __syncthreads();
    }
    if (blockIdx.x == gridDim.x - 1u && threadIdx.x == 0u) s.chain_excl[live] = chain_base;
    grid.sync();

    // A color lists its lane rows from its start, then its warp rows, then its pair rows.
    for (uint32_t position = thread; position < live; position += threads) {
        const uint32_t color = s.pos_color[position];
        if (color >= kColorSlots) continue;
        const uint32_t slot = live_order[position];
        const NkRow& row = urows[slot];
        if (IsPairRow(row))
            s.color_rows[atomicSub(s.color_pair_tail + color, 1u) - 1u] = slot;
        else if (IsLaneRow(row, points))
            s.color_rows[atomicAdd(s.color_cursor + color, 1u)] = slot;
        else
            s.color_rows[atomicSub(s.color_tail + color, 1u) - 1u] = slot;
    }
    const uint32_t island_count = *island_count_dev;
    for (uint32_t i = thread; i < island_count; i += threads) {
        const IslandRecord rec = reinterpret_cast<const IslandRecord*>(islands)[i];
        if (!(rec.flags & kIslandWarpWork) || rec.seg_cnt == 0u || !activity.Active(rec)) continue;
        const uint32_t live_off = rec.seg_off == 0u ? 0u : live_scan[rec.seg_off - 1u];
        const uint32_t live_end = live_scan[rec.seg_off + rec.seg_cnt - 1u];
        if (s.chain_excl[live_end] != s.chain_excl[live_off])
            s.chain_islands[atomicAdd(s.control + kControlChainIslands, 1u)] = i;
    }
    grid.sync();
    for (uint32_t color = thread; color < s.control[kControlColors]; color += threads)
        s.color_lane[color] = s.color_cursor[s.color_slot[color]];
    grid.sync();
    PackLaneRecords(urows, points, s, prep.row_meff, prep.row_damping, prep.dt);

    // Rows with an articulation side keep live order; each articulation then lists its entries
    // inside its island's span, so every sum over them runs in a fixed order.
    if (threadIdx.x == 0u) block_chain = 0u;
    __syncthreads();
    uint32_t schur_rows = 0u;
    for (uint32_t position = begin + threadIdx.x; position < end; position += blockDim.x)
        schur_rows += IsSchurRow(urows[live_order[position]], s) ? 1u : 0u;
    schur_rows = __reduce_add_sync(0xffffffffu, schur_rows);
    if (lane == 0u && schur_rows != 0u) atomicAdd(&block_chain, schur_rows);
    __syncthreads();
    if (threadIdx.x == 0u) s.block_count[blockIdx.x] = block_chain;
    for (uint32_t i = thread; i < 2u * s.articulations; i += threads) s.artic_range[i] = 0u;
    grid.sync();
    earlier = 0u;
    for (uint32_t b = threadIdx.x; b < blockIdx.x; b += blockDim.x) earlier += s.block_count[b];
    uint32_t artic_base = 0u;
    BlockScanT(scan_temp).ExclusiveSum(earlier, ignored, artic_base);
    __syncthreads();
    for (uint32_t tile = begin; tile < end; tile += blockDim.x) {
        const uint32_t position = tile + threadIdx.x;
        const uint32_t flag =
            position < end && IsSchurRow(urows[live_order[position]], s) ? 1u : 0u;
        uint32_t offset = 0u, tile_total = 0u;
        BlockScanT(scan_temp).ExclusiveSum(flag, offset, tile_total);
        if (position < end) s.artic_excl[position] = artic_base + offset;
        if (flag != 0u) s.artic_rows[artic_base + offset] = live_order[position];
        artic_base += tile_total;
        __syncthreads();
    }
    if (blockIdx.x == gridDim.x - 1u && threadIdx.x == 0u) {
        s.artic_excl[live] = artic_base;
        s.control[kControlArticRows] = artic_base;
    }
    grid.sync();

    // Rows with a hub side keep live order too; each hub then lists its entries inside its
    // island's span, so every sum over them runs in a fixed order.
    if (threadIdx.x == 0u) block_chain = 0u;
    __syncthreads();
    uint32_t hub_rows = 0u;
    for (uint32_t position = begin + threadIdx.x; position < end; position += blockDim.x)
        hub_rows += IsHubRow(urows[live_order[position]], s) ? 1u : 0u;
    hub_rows = __reduce_add_sync(0xffffffffu, hub_rows);
    if (lane == 0u && hub_rows != 0u) atomicAdd(&block_chain, hub_rows);
    __syncthreads();
    if (threadIdx.x == 0u) s.block_count[blockIdx.x] = block_chain;
    for (uint32_t i = thread; i < 2u * s.bodies; i += threads) s.hub_range[i] = 0u;
    grid.sync();
    earlier = 0u;
    for (uint32_t b = threadIdx.x; b < blockIdx.x; b += blockDim.x) earlier += s.block_count[b];
    uint32_t hub_base = 0u;
    BlockScanT(scan_temp).ExclusiveSum(earlier, ignored, hub_base);
    __syncthreads();
    for (uint32_t tile = begin; tile < end; tile += blockDim.x) {
        const uint32_t position = tile + threadIdx.x;
        const uint32_t flag = position < end && IsHubRow(urows[live_order[position]], s) ? 1u : 0u;
        uint32_t offset = 0u, tile_total = 0u;
        BlockScanT(scan_temp).ExclusiveSum(flag, offset, tile_total);
        if (position < end) s.hub_excl[position] = hub_base + offset;
        if (flag != 0u) s.hub_rows[hub_base + offset] = live_order[position];
        hub_base += tile_total;
        __syncthreads();
    }
    if (blockIdx.x == gridDim.x - 1u && threadIdx.x == 0u) {
        s.hub_excl[live] = hub_base;
        s.control[kControlHubRows] = hub_base;
    }
    grid.sync();
    // A row names at most two hubs, so an island's entries fit twice its row span.
    if (LoadControl(s.control + kControlHubRows) != 0u && prep.bodies_per_env != 0u) {
        for (uint32_t i = thread / warpSize; i < island_count; i += threads / warpSize) {
            const IslandRecord rec = reinterpret_cast<const IslandRecord*>(islands)[i];
            if (!(rec.flags & kIslandWarpWork) || rec.seg_cnt == 0u || !activity.Active(rec))
                continue;
            const uint32_t live_off = rec.seg_off == 0u ? 0u : live_scan[rec.seg_off - 1u];
            const uint32_t first = s.hub_excl[live_off];
            const uint32_t last = s.hub_excl[live_scan[rec.seg_off + rec.seg_cnt - 1u]];
            if (first == last) continue;
            uint32_t cursor = 2u * first;
            for (uint32_t local = 0u; local < prep.bodies_per_env; ++local) {
                const uint32_t body = rec.env * prep.bodies_per_env + local;
                if (body >= s.bodies || s.hub_count[body] <= kHubRows) continue;
                const uint32_t entry_begin = cursor;
                for (uint32_t base = first; base < last; base += warpSize) {
                    const uint32_t k = base + lane;
                    uint32_t sides = 0u;
                    uint32_t slot = 0u;
                    if (k < last) {
                        slot = s.hub_rows[k];
                        const NkRowSide side_a = urows[slot].a;
                        const NkRowSide side_b = urows[slot].b;
                        sides = (side_a.kind == kNkSideRigid && side_a.index == body ? 1u : 0u) |
                                (side_b.kind == kNkSideRigid && side_b.index == body ? 2u : 0u);
                    }
                    const uint32_t mask = __ballot_sync(0xffffffffu, sides != 0u);
                    if (sides != 0u)
                        s.hub_entries[cursor + __popc(mask & ((1u << lane) - 1u))] = (slot << 2u) | sides;
                    cursor += __popc(mask);
                }
                if (lane == 0u && cursor != entry_begin) {
                    s.hub_range[2u * body] = entry_begin;
                    s.hub_range[2u * body + 1u] = cursor;
                }
            }
        }
    }
    if (prep.cc_root == nullptr || prep.cc_artic_first == nullptr || prep.artics_per_env == 0u)
        return;
    // A row names at most two articulations, so an island's entries fit twice its row span.
    const uint32_t warp = thread / warpSize;
    const uint32_t warps = threads / warpSize;
    for (uint32_t i = warp; i < island_count; i += warps) {
        const IslandRecord rec = reinterpret_cast<const IslandRecord*>(islands)[i];
        if (!(rec.flags & kIslandWarpWork) || rec.seg_cnt == 0u || !activity.Active(rec)) continue;
        const uint32_t live_off = rec.seg_off == 0u ? 0u : live_scan[rec.seg_off - 1u];
        const uint32_t first = s.artic_excl[live_off];
        const uint32_t last = s.artic_excl[live_scan[rec.seg_off + rec.seg_cnt - 1u]];
        if (first == last) continue;
        const uint32_t root = activity.sorted_roots[rec.seg_off];
        uint32_t cursor = 2u * first;
        for (uint32_t tile = 0u; tile < prep.artics_per_env; ++tile) {
            const uint32_t articulation = rec.env * prep.artics_per_env + tile;
            const uint32_t claim = prep.cc_artic_first[articulation];
            if (claim == ~0u || prep.cc_root[claim] != root) continue;
            const uint32_t entry_begin = cursor;
            for (uint32_t base = first; base < last; base += warpSize) {
                const uint32_t k = base + lane;
                uint32_t sides = 0u;
                uint32_t slot = 0u;
                if (k < last) {
                    slot = s.artic_rows[k];
                    const NkRowSide side_a = urows[slot].a;
                    const NkRowSide side_b = urows[slot].b;
                    sides = (side_a.kind == kNkSideArtic && side_a.index == articulation ? 1u : 0u) |
                            (side_b.kind == kNkSideArtic && side_b.index == articulation ? 2u : 0u);
                }
                const uint32_t mask = __ballot_sync(0xffffffffu, sides != 0u);
                if (sides != 0u)
                    s.artic_entries[cursor + __popc(mask & ((1u << lane) - 1u))] = (slot << 2u) | sides;
                cursor += __popc(mask);
            }
            if (lane == 0u) {
                s.artic_range[2u * articulation] = entry_begin;
                s.artic_range[2u * articulation + 1u] = cursor;
            }
        }
    }
}

struct ColoredSolveArgs {
    const NkRow* urows = nullptr;
    float* lambda = nullptr;
    const float* chain_jacobian = nullptr;
    const float* row_minv_jt = nullptr;
    const float* chain_jacobian_b = nullptr;
    const float* row_minv_jt_b = nullptr;
    const float* row_meff = nullptr;
    const float* row_damping = nullptr;
    const float* articulation_mass = nullptr;     // per articulation, dense (M + dt C)
    const float* articulation_mass_inv = nullptr; // its inverse
    float* qdot_flat = nullptr;
    Spatial6* link_velocity = nullptr;
    float* qdot = nullptr;
    math::Vec3* body_linear = nullptr;
    math::Vec3* body_angular = nullptr;
    const float* body_inv_mass = nullptr;   // zero for a hub, which no sweep writes
    const float* rigid_inv_mass = nullptr;  // every dynamic body's own
    const math::SymmetricMat3* body_inv_inertia = nullptr;
    PointMassView points;
    const uint32_t* islands = nullptr;
    const uint32_t* live_scan = nullptr;
    const uint32_t* live_order = nullptr;
    const uint32_t* island_root_sorted = nullptr;
    const uint32_t* cc_root = nullptr;
    const uint32_t* cc_artic_first = nullptr;
    const uint32_t* dof_to_link = nullptr;
    const uint32_t* dof_to_component = nullptr;
    float* row_penetration = nullptr;
    float* row_pseudo_lambda = nullptr;
    float* qdot_pseudo = nullptr;
    Spatial6* link_velocity_pseudo = nullptr;
    float* qdot_pseudo_flat = nullptr;
    math::Vec3* body_pseudo_linear = nullptr;
    math::Vec3* body_pseudo_angular = nullptr;
    math::Vec3* particle_pseudo = nullptr;
    math::Vec3* grid_pseudo = nullptr;
    VelocityErrorView error;
    nkops::VertexBlockView vertex_blocks;
    uint32_t* vbd_velocity_sweep_count = nullptr;
    uint32_t* solver_color_counts = nullptr;
    uint64_t* solver_phase_time = nullptr;
    uint32_t env_count = 0u;
    uint32_t rows_per_env = 0u;
    uint32_t artics_per_env = 0u;
    uint32_t articulation_count = 0u;
    uint32_t dof_stride = 0u;
    uint32_t base_link_count = 0u;
    uint32_t vel_iters = 0u;
    uint32_t pos_iters = 0u;
    float pos_beta = 0.0f;
    float pos_slop = 0.0f;
    float dt = 0.0f;
    float vel_tolerance = 0.0f;
    float baumgarte_max_velocity = 0.0f;
    bool apply_cached = false;
    bool verify_idle = false;
};

enum class ChainSweep : uint32_t { WarmStart, Velocity, Position };

// Words of solver_color_counts, and the timed phases of solver_phase_time in order.
constexpr uint32_t kSolverCountWords = 6u;
enum SolverPhase : uint32_t {
    kPhaseWarmStart, kPhaseVertex, kPhaseColor, kPhaseChain, kPhaseSchur, kPhaseHub,
    kPhaseStationary, kPhasePosition, kSolverPhases
};

__device__ inline uint64_t GlobalNanoseconds() {
    uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

__host__ __device__ inline uint32_t TriangleIndex(uint32_t row, uint32_t col, uint32_t n) {
    return row * (2u * n - row + 1u) / 2u + (col - row);
}

// The impulse axes a row couples to its articulations, and the J rows holding them.
__device__ inline uint32_t SchurAxes(const NkRow& row, uint32_t slot, bool pseudo,
                                     uint32_t* j_rows) {
    const uint32_t flags = row.flags;
    if (!(flags & nk::nk_row_flags::kActive) || (flags & nk::nk_row_flags::kBlockTangent)) return 0u;
    if (pseudo && (flags & (nk::nk_row_flags::kFriction | nk::nk_row_flags::kVelocityOnly)))
        return 0u;
    const uint32_t axes = !pseudo && (flags & nk::nk_row_flags::kBlockNormal) ? 3u : 1u;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
        j_rows[axis] = slot + (axis < axes ? axis * row.group_normal_count : 0u);
    return axes;
}

// Each axis's step gain is the inverse response its own row step uses; a zero gain leaves an
// axis out of the coupling (separating, sliding, at a bound, or bounded by other rows).
struct SchurRowTerms {
    uint32_t axes = 0u;
    uint32_t j_rows[3]{};
    float gain[3]{};
    float lambda[3]{};
};

__device__ SchurRowTerms LoadSchurRowTerms(const ColoredSolveArgs& a, uint32_t slot, bool pseudo) {
    const NkRow& row = a.urows[slot];
    SchurRowTerms t;
    t.axes = SchurAxes(row, slot, pseudo, t.j_rows);
    const float* lambda = pseudo ? a.row_pseudo_lambda : a.lambda;
    for (uint32_t axis = 0u; axis < t.axes; ++axis) t.lambda[axis] = lambda[t.j_rows[axis]];
    if (t.axes == 0u) return t;
    const float meff = a.row_meff[slot];
    if (pseudo) {
        t.gain[0] = t.lambda[0] > 0.0f && meff > 0.0f ? meff : 0.0f;
        return t;
    }
    if (t.axes == 3u) {
        const math::SymmetricMat3& k = row.contact_response;
        if (!(t.lambda[0] > 0.0f) || !(k.xx > 0.0f)) return t;
        t.gain[0] = 1.0f / k.xx;
        const float mu = fmaxf(row.mu, row.friction_secondary);
        const float largest =
            constraint::CoulombTangentSpectralResponse(k, row.mu, row.friction_secondary);
        if (!(mu > 0.0f) || !(largest > 0.0f)) return t;
        const float first_scale = fmaxf(row.mu, 0.0f) / mu;
        const float second_scale = fmaxf(row.friction_secondary, 0.0f) / mu;
        const float first = first_scale > 0.0f ? t.lambda[1] / first_scale : 0.0f;
        const float second = second_scale > 0.0f ? t.lambda[2] / second_scale : 0.0f;
        if (hypotf(first, second) < t.lambda[0] * mu) {
            t.gain[1] = first_scale * first_scale / largest;
            t.gain[2] = second_scale * second_scale / largest;
        }
        return t;
    }
    if (row.flags & nk::nk_row_flags::kFriction) return t;
    const float damping_scale = 1.0f / (1.0f + a.row_damping[slot] * a.dt);
    const float implicit_mass =
        meff / (1.0f - meff * row.compliance_alpha * (1.0f - damping_scale));
    if (t.lambda[0] > row.lower && t.lambda[0] < row.upper && implicit_mass > 0.0f)
        t.gain[0] = implicit_mass;
    return t;
}

// The row's own projection after the coupling moved its velocity by w.
__device__ void ProjectSchurRow(const NkRow& row, const SchurRowTerms& t, const float* w,
                                bool pseudo, float* out) {
    for (uint32_t axis = 0u; axis < 3u; ++axis)
        out[axis] = t.lambda[axis] - (t.gain[axis] != 0.0f ? t.gain[axis] * w[axis] : 0.0f);
    if (pseudo) {
        out[0] = fmaxf(out[0], 0.0f);
    } else if (t.axes == 3u) {
        out[0] = fmaxf(out[0], 0.0f);
        const float mu = fmaxf(row.mu, row.friction_secondary);
        if (!(mu > 0.0f) || !(out[0] > 0.0f)) {
            out[1] = out[2] = 0.0f;
            return;
        }
        const float first_scale = fmaxf(row.mu, 0.0f) / mu;
        const float second_scale = fmaxf(row.friction_secondary, 0.0f) / mu;
        const float first = first_scale > 0.0f ? out[1] / first_scale : 0.0f;
        const float second = second_scale > 0.0f ? out[2] / second_scale : 0.0f;
        const float length = hypotf(first, second);
        const float scale = length > out[0] * mu ? out[0] * mu / length : 1.0f;
        out[1] = first * scale * first_scale;
        out[2] = second * scale * second_scale;
    } else if (t.gain[0] != 0.0f) {
        out[0] = fminf(fmaxf(out[0], row.lower), row.upper);
    }
}

// Each color sweeps across the grid behind one barrier; chain rows then run per island in live
// order on one block. Colors see articulations frozen, and a Schur step couples those rows.
__global__ void __launch_bounds__(kColorBlockSize) SolveColoredRowsKernel(ColoredSolveArgs a,
                                                                          ColorScratch s) {
    const auto grid = cooperative_groups::this_grid();
    extern __shared__ __align__(16) unsigned char colored_shared[];
    __shared__ __align__(16) unsigned char row_storage[3u * (kColorBlockSize / 32u) * sizeof(NkRow)];
    __shared__ uint32_t batch_needed;
    constexpr uint32_t kBatchRows = kColorBlockSize / 32u;
    __shared__ unsigned long long cache_key[kChainCacheEntries];
    __shared__ float cache_velocity[3u * kChainCacheEntries], cache_error[3u * kChainCacheEntries];
    __shared__ float cache_inverse_mass[kChainCacheEntries];
    __shared__ uint32_t cache_lane[kColorBlockSize];
    __shared__ float cache_jacobian[9u * kColorBlockSize];
    __shared__ uint32_t cache_shape[kBatchRows], batch_slot[kBatchRows];
    __shared__ float cache_scalar[kChainRowScalars * kBatchRows];
    __shared__ float hub_partial[kBatchRows * kHubItem];
    const ChainPointCache cache{cache_key, cache_velocity, cache_error, cache_inverse_mass,
                                cache_lane, cache_jacobian, cache_shape, cache_scalar};
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = lane >> 5u;
    const uint32_t wlane = lane & 31u;
    const uint32_t nwarps = blockDim.x >> 5u;
    const uint32_t dofs = a.dof_stride;
    const uint32_t k_tiles = a.artics_per_env == 0u ? 1u : a.artics_per_env;
    float* const staged_j = reinterpret_cast<float*>(colored_shared);
    const size_t staged_size = size_t{3u} * nwarps * dofs;
    // Sequential chain rows stage their articulation responses M^-1 J^T behind the Jacobians.
    float* const staged_w = staged_j + 2u * staged_size;
    NkRow* const staged_rows = reinterpret_cast<NkRow*>(row_storage);
    // A color fills one warp of every block before a second warp of any block.
    const uint32_t color_warp = warp * gridDim.x + blockIdx.x;
    const uint32_t color_warps = gridDim.x * nwarps;
    const uint32_t colors = s.control[kControlColors];
    const uint32_t chain_islands = s.control[kControlChainIslands];
    const uint32_t live = s.control[kControlLive];
    const bool pos_pass = a.pos_iters > 0u && a.row_penetration != nullptr;
    if (a.solver_color_counts != nullptr && blockIdx.x == 0u && threadIdx.x == 0u) {
        // One block runs each island's chain rows in order, so the longest island paces a sweep.
        uint32_t longest = 0u;
        for (uint32_t k = 0u; k < chain_islands; ++k) {
            const IslandRecord rec =
                reinterpret_cast<const IslandRecord*>(a.islands)[s.chain_islands[k]];
            const uint32_t live_off = rec.seg_off == 0u ? 0u : a.live_scan[rec.seg_off - 1u];
            longest = max(longest, s.chain_excl[a.live_scan[rec.seg_off + rec.seg_cnt - 1u]] -
                                   s.chain_excl[live_off]);
        }
        const uint32_t counts[kSolverCountWords] = {a.vertex_blocks.layout.colors, colors,
            s.chain_excl[live], chain_islands, longest, s.control[kControlArticRows]};
        for (uint32_t env = 0u; env < a.env_count; ++env)
            for (uint32_t k = 0u; k < kSolverCountWords; ++k)
                a.solver_color_counts[kSolverCountWords * env + k] = counts[k];
    }
    // Shared, so the timing thread adds no registers to the sweeps.
    __shared__ unsigned long long phase_time[kSolverPhases];
    const bool timed = a.solver_phase_time != nullptr && blockIdx.x == 0u && threadIdx.x == 0u;
    if (timed)
        for (uint32_t k = 0u; k < kSolverPhases; ++k) phase_time[k] = 0ull;
    uint64_t phase_mark = timed ? GlobalNanoseconds() : 0u;
    const auto lap = [&](uint32_t phase) {
        if (!timed) return;
        const uint64_t now = GlobalNanoseconds();
        phase_time[phase] += now - phase_mark;
        phase_mark = now;
    };
    const PointMassView pseudo_points = a.points.Pseudo(a.particle_pseudo, a.grid_pseudo);
    bool changed = false;

    const auto env_qdot = [&](float* velocity, uint32_t env) {
        return velocity != nullptr ? velocity + size_t{env * k_tiles} * dofs : nullptr;
    };
    const auto pending = [&](const NkRow& row, uint32_t slot) {
        return !IsSequentialArticRow(row, s) &&
               (HasArticulationSide(row) || IsHubSide(row.a, s) || IsHubSide(row.b, s))
            ? s.artic_pending + size_t{slot} * 3u : nullptr;
    };

    // A color lists its lane rows first; each takes one thread, every other row a warp.
    auto color_sweep = [&](auto&& solve_row, auto&& solve_lane, auto&& solve_pair,
                           bool position = false) {
        NkRow* const staged = staged_rows + 3u * warp;
        for (uint32_t color = 0u; color < colors;) {
            if (position && s.color_position[color] == 0u) {
                ++color;
                continue;
            }
            // Consecutive colors of at most one row per warp run in one block under block barriers.
            uint32_t run = color;
            while (run < colors && s.color_start[run + 1u] - s.color_start[run] <= nwarps) ++run;
            if (run != color) {
                for (; blockIdx.x == 0u && color < run; ++color) {
                    const uint32_t i = s.color_start[color] + warp;
                    if (i < s.color_lane[color]) {
                        const uint32_t slot = s.color_rows[i];
                        if (wlane == 0u) solve_lane(i, slot, slot / a.rows_per_env);
                    } else if (i < s.color_start[color + 1u]) {
                        const uint32_t slot = s.color_rows[i];
                        StageRowWarp(a.urows, slot, staged, wlane);
                        solve_row(slot, slot / a.rows_per_env, staged);
                    }
                    __syncthreads();
                }
                color = run;
                grid.sync();
                continue;
            }
            const uint32_t lanes = s.color_lane[color];
            for (uint32_t i = s.color_start[color] + color_warp * warpSize + wlane; i < lanes;
                 i += color_warps * warpSize) {
                const uint32_t slot = s.color_rows[i];
                solve_lane(i, slot, slot / a.rows_per_env);
            }
            // Warp and pair rows take warps from the last one down, beside the lane rows.
            const uint32_t rank = color_warps - 1u - color_warp;
            const uint32_t end = s.color_start[color + 1u];
            const uint32_t pairs = s.color_pair[color];
            for (uint32_t i = lanes + rank; i < pairs; i += color_warps) {
                const uint32_t slot = s.color_rows[i];
                StageRowWarp(a.urows, slot, staged, wlane);
                solve_row(slot, slot / a.rows_per_env, staged);
            }
            constexpr uint32_t halves = 32u / kPairRowWidth;
            const uint32_t half = wlane / kPairRowWidth;
            const uint32_t half_mask = (0xffffffffu >> (32u - kPairRowWidth))
                                       << (half * kPairRowWidth);
            const uint32_t pair_rank = (rank + color_warps - (pairs - lanes) % color_warps) %
                                       color_warps;
            for (uint32_t base = pairs + pair_rank * halves; base < end;
                 base += color_warps * halves) {
                const uint32_t i = base + half;
                if (i < end) solve_pair(s.color_rows[i], wlane % kPairRowWidth, half_mask);
                __syncwarp();
            }
            grid.sync();
            ++color;
        }
    };

    // S = M + sum J^T g J over active axes and P = J^T (lambda - start) reduce per articulation
    // in fixed-order groups; S du = P corrects every coupled row, then u takes the new impulse.
    const uint32_t artic_rows = s.control[kControlArticRows];
    const bool schur = artic_rows != 0u && dofs != 0u && s.articulations != 0u &&
                       a.articulation_mass != nullptr && a.articulation_mass_inv != nullptr;
    const uint32_t schur_groups = SchurGroups(s.articulations, gridDim.x);
    const uint32_t triangle = TriangleSize(dofs);
    const size_t item_floats = size_t{triangle} + dofs;
    float* const schur_j = reinterpret_cast<float*>(colored_shared);
    float* const schur_gain = schur_j + staged_size;
    float* const schur_delta = schur_gain + 3u * nwarps;
    uint32_t triangle_pair[kSchurTrianglePerThread];
    for (uint32_t m = 0u; m < kSchurTrianglePerThread; ++m) {
        uint32_t index = lane + m * blockDim.x;
        triangle_pair[m] = ~0u;
        if (index >= triangle) continue;
        uint32_t row = 0u;
        while (index >= dofs - row) index -= dofs - row++;
        triangle_pair[m] = (row << 16u) | (row + index);
    }
    const auto artic_empty = [&](uint32_t articulation) {
        return s.artic_range[2u * articulation] == s.artic_range[2u * articulation + 1u];
    };

    auto schur_snapshot = [&](bool pseudo, bool zero) {
        const float* lambda = pseudo ? a.row_pseudo_lambda : a.lambda;
        for (uint32_t i = blockIdx.x * blockDim.x + lane; i < artic_rows;
             i += gridDim.x * blockDim.x) {
            const uint32_t slot = s.artic_rows[i];
            uint32_t j_rows[3];
            const uint32_t axes = SchurAxes(a.urows[slot], slot, pseudo, j_rows);
            for (uint32_t axis = 0u; axis < 3u; ++axis) {
                s.artic_start[size_t{slot} * 3u + axis] =
                    axis < axes && !zero ? lambda[j_rows[axis]] : 0.0f;
                s.artic_pending[size_t{slot} * 3u + axis] = 0.0f;
            }
        }
        grid.sync();
    };

    auto schur_gather = [&](bool pseudo, bool matrix) {
        const uint32_t items = s.articulations * schur_groups;
        for (uint32_t item = blockIdx.x; item < items; item += gridDim.x) {
            const uint32_t articulation = item / schur_groups;
            const uint32_t group = item % schur_groups;
            const uint32_t first = s.artic_range[2u * articulation];
            const uint32_t last = s.artic_range[2u * articulation + 1u];
            const uint32_t chunk = (last - first + schur_groups - 1u) / schur_groups;
            const uint32_t begin = min(last, first + group * chunk);
            const uint32_t end = min(last, begin + chunk);
            float sum[kSchurTrianglePerThread]{};
            float moment = 0.0f;
            for (uint32_t base = begin; base < end; base += nwarps) {
                const uint32_t count = min(nwarps, end - base);
                if (warp < count) {
                    const uint32_t entry = s.artic_entries[base + warp];
                    const uint32_t slot = entry >> 2u;
                    const SchurRowTerms t = LoadSchurRowTerms(a, slot, pseudo);
                    for (uint32_t axis = 0u; axis < 3u; ++axis) {
                        float* const j = schur_j + size_t{3u * warp + axis} * dofs;
                        for (uint32_t k = wlane; k < dofs; k += warpSize) {
                            float value = 0.0f;
                            if (axis < t.axes) {
                                const size_t at = size_t{t.j_rows[axis]} * dofs + k;
                                if (entry & 1u) value += a.chain_jacobian[at];
                                if (entry & 2u) value += a.chain_jacobian_b[at];
                            }
                            j[k] = value;
                        }
                        if (wlane == 0u) {
                            schur_gain[3u * warp + axis] = matrix ? t.gain[axis] : 0.0f;
                            schur_delta[3u * warp + axis] = axis < t.axes
                                ? t.lambda[axis] - s.artic_start[size_t{slot} * 3u + axis] : 0.0f;
                        }
                    }
                }
                __syncthreads();
                for (uint32_t w = 0u; w < count; ++w) {
                    for (uint32_t axis = 0u; axis < 3u; ++axis) {
                        const float* j = schur_j + size_t{3u * w + axis} * dofs;
                        const float gain = schur_gain[3u * w + axis];
                        if (gain != 0.0f) {
                            #pragma unroll
                            for (uint32_t m = 0u; m < kSchurTrianglePerThread; ++m) {
                                if (triangle_pair[m] == ~0u) continue;
                                sum[m] += gain * j[triangle_pair[m] >> 16u] *
                                          j[triangle_pair[m] & 0xffffu];
                            }
                        }
                        if (lane < dofs) moment += j[lane] * schur_delta[3u * w + axis];
                    }
                }
                __syncthreads();
            }
            float* const out = s.artic_partial + item * item_floats;
            if (matrix) {
                #pragma unroll
                for (uint32_t m = 0u; m < kSchurTrianglePerThread; ++m)
                    if (triangle_pair[m] != ~0u) out[lane + m * blockDim.x] = sum[m];
            }
            if (lane < dofs) out[triangle + lane] = moment;
        }
    };

    // A warp sums one entry's group partials: lanes stride the groups and a fixed shuffle tree
    // adds them. Entries [first, first + count) of `span` articulations from `from` go to out.
    auto reduce_partials = [&](uint32_t from, uint32_t span, uint32_t first, uint32_t count,
                               float* out, size_t out_stride, uint32_t warp_index,
                               uint32_t warp_count) {
        for (uint32_t e = warp_index; e < span * count; e += warp_count) {
            const uint32_t x = e % count;
            const float* column = s.artic_partial +
                size_t{from + e / count} * schur_groups * item_floats + first + x;
            float value = 0.0f;
            for (uint32_t g = wlane; g < schur_groups; g += warpSize)
                value += column[size_t{g} * item_floats];
            value = WarpSum(value);
            if (wlane == 0u) out[size_t{e / count} * out_stride + x] = value;
        }
    };

    // Every warp of the grid sums the group partials, then one block factors each articulation's
    // S = L L^T with its triangle entries held in registers, one barrier per column.
    auto schur_solve = [&]() {
        reduce_partials(0u, s.articulations, 0u, static_cast<uint32_t>(item_floats),
                        s.artic_summed, item_floats, blockIdx.x * nwarps + warp,
                        gridDim.x * nwarps);
        grid.sync();
        // Entries reach `column` once final, the step before their column pivots.
        float* const column = reinterpret_cast<float*>(colored_shared);
        float* const factor = column + size_t{dofs} * dofs;
        float* const inverse_pivot = factor + size_t{dofs} * dofs;
        for (uint32_t articulation = blockIdx.x; articulation < s.articulations;
             articulation += gridDim.x) {
            if (artic_empty(articulation)) continue;
            const float* mass = a.articulation_mass + size_t{articulation} * dofs * dofs;
            const float* summed = s.artic_summed + size_t{articulation} * item_floats;
            float value[kSchurTrianglePerThread];
            #pragma unroll
            for (uint32_t m = 0u; m < kSchurTrianglePerThread; ++m) {
                value[m] = 0.0f;
                if (triangle_pair[m] == ~0u) continue;
                const uint32_t k = triangle_pair[m] >> 16u;
                const uint32_t i = triangle_pair[m] & 0xffffu;
                value[m] = mass[size_t{k} * dofs + i] + summed[lane + m * blockDim.x];
                if (k == 0u) column[size_t{i} * dofs] = value[m];
            }
            float rhs[kSchurLaneRows];
            #pragma unroll
            for (uint32_t r = 0u; r < kSchurLaneRows; ++r) {
                const uint32_t i = wlane + r * warpSize;
                rhs[r] = warp == 0u && i < dofs ? summed[triangle + i] : 0.0f;
            }
            __syncthreads();
            // Padding DOFs carry an empty row and column, so their unit pivot leaves them at zero.
            for (uint32_t j = 0u; j < dofs; ++j) {
                const float diagonal = column[size_t{j} * dofs + j];
                const float pivot = diagonal > 0.0f ? sqrtf(diagonal) : 1.0f;
                const float inverse = 1.0f / pivot;
                #pragma unroll
                for (uint32_t m = 0u; m < kSchurTrianglePerThread; ++m) {
                    if (triangle_pair[m] == ~0u) continue;
                    const uint32_t k = triangle_pair[m] >> 16u;
                    const uint32_t i = triangle_pair[m] & 0xffffu;
                    if (k == j) {
                        factor[size_t{i} * dofs + j] = i == j ? pivot : value[m] * inverse;
                    } else if (k > j) {
                        value[m] -= column[size_t{i} * dofs + j] * column[size_t{k} * dofs + j] *
                                    (inverse * inverse);
                        if (k == j + 1u) column[size_t{i} * dofs + k] = value[m];
                    }
                }
                if (lane == 0u) inverse_pivot[j] = inverse;
                __syncthreads();
            }
            // One warp substitutes forward then back; lane l holds rows l, l + 32, ... and the
            // owner of each step's row broadcasts it.
            if (warp == 0u) {
                const auto own = [&](uint32_t j) {
                    float held = rhs[0];
                    #pragma unroll
                    for (uint32_t r = 1u; r < kSchurLaneRows; ++r)
                        if (j >= r * warpSize) held = rhs[r];
                    return __shfl_sync(~0u, held, j % warpSize) * inverse_pivot[j];
                };
                for (uint32_t j = 0u; j < dofs; ++j) {
                    const float x = own(j);
                    #pragma unroll
                    for (uint32_t r = 0u; r < kSchurLaneRows; ++r) {
                        const uint32_t i = wlane + r * warpSize;
                        if (i == j) rhs[r] = x;
                        else if (i > j && i < dofs) rhs[r] -= factor[size_t{i} * dofs + j] * x;
                    }
                }
                for (uint32_t jj = dofs; jj > 0u; --jj) {
                    const uint32_t j = jj - 1u;
                    const float x = own(j);
                    #pragma unroll
                    for (uint32_t r = 0u; r < kSchurLaneRows; ++r) {
                        const uint32_t i = wlane + r * warpSize;
                        if (i == j) rhs[r] = x;
                        else if (i < j) rhs[r] -= factor[size_t{j} * dofs + i] * x;
                    }
                }
                #pragma unroll
                for (uint32_t r = 0u; r < kSchurLaneRows; ++r) {
                    const uint32_t i = wlane + r * warpSize;
                    if (i < dofs) s.artic_step[size_t{articulation} * dofs + i] = rhs[r];
                }
            }
            __syncthreads();
        }
    };

    // Each coupled row re-projects against the velocity du gives it; the change reaches its
    // other sides at the row's next visit.
    auto schur_correct = [&](bool pseudo) {
        float* const lambda = pseudo ? a.row_pseudo_lambda : a.lambda;
        for (uint32_t i = color_warp; i < artic_rows; i += color_warps) {
            const uint32_t slot = s.artic_rows[i];
            const SchurRowTerms t = LoadSchurRowTerms(a, slot, pseudo);
            if (t.gain[0] == 0.0f && t.gain[1] == 0.0f && t.gain[2] == 0.0f) continue;
            const NkRow& row = a.urows[slot];
            float w[3]{};
            for (uint32_t side = 0u; side < 2u; ++side) {
                const NkRowSide& endpoint = side == 0u ? row.a : row.b;
                if (endpoint.kind != kNkSideArtic || artic_empty(endpoint.index)) continue;
                const float* J = side == 0u ? a.chain_jacobian : a.chain_jacobian_b;
                const float* step = s.artic_step + size_t{endpoint.index} * dofs;
                for (uint32_t axis = 0u; axis < t.axes; ++axis) {
                    if (t.gain[axis] == 0.0f) continue;
                    for (uint32_t k = wlane; k < dofs; k += warpSize)
                        w[axis] += J[size_t{t.j_rows[axis]} * dofs + k] * step[k];
                }
            }
            for (uint32_t axis = 0u; axis < 3u; ++axis) w[axis] = WarpSum(w[axis]);
            if (wlane != 0u) continue;
            float out[3];
            ProjectSchurRow(row, t, w, pseudo, out);
            for (uint32_t axis = 0u; axis < t.axes; ++axis) {
                const float delta = out[axis] - t.lambda[axis];
                if (delta == 0.0f) continue;
                lambda[t.j_rows[axis]] = out[axis];
                s.artic_pending[size_t{slot} * 3u + axis] += delta;
            }
        }
    };

    // u += M^-1 J^T (lambda - start); the articulation then holds every row impulse.
    auto schur_commit = [&](bool pseudo) {
        schur_gather(pseudo, false);
        grid.sync();
        float* const velocity = pseudo ? a.qdot_pseudo_flat : a.qdot_flat;
        float* const moment = reinterpret_cast<float*>(colored_shared);
        for (uint32_t articulation = blockIdx.x; articulation < s.articulations;
             articulation += gridDim.x) {
            if (artic_empty(articulation)) continue;
            const float* partial = s.artic_partial + size_t{articulation} * schur_groups * item_floats;
            reduce_partials(articulation, 1u, triangle, dofs, moment, 0u, warp, nwarps);
            __syncthreads();
            const float* inverse = a.articulation_mass_inv + size_t{articulation} * dofs * dofs;
            for (uint32_t k = lane; k < dofs; k += blockDim.x) {
                float sum = 0.0f;
                for (uint32_t j = 0u; j < dofs; ++j) sum += inverse[size_t{k} * dofs + j] * moment[j];
                const size_t at = size_t{articulation} * dofs + k;
                float value = velocity[at];
                AddVelocity(value, pseudo ? nullptr : s.qdot_error + at, sum);
                velocity[at] = value;
            }
            __syncthreads();
        }
        const float* lambda = pseudo ? a.row_pseudo_lambda : a.lambda;
        for (uint32_t i = blockIdx.x * blockDim.x + lane; i < artic_rows;
             i += gridDim.x * blockDim.x) {
            const uint32_t slot = s.artic_rows[i];
            uint32_t j_rows[3];
            const uint32_t axes = SchurAxes(a.urows[slot], slot, pseudo, j_rows);
            for (uint32_t axis = 0u; axis < axes; ++axis)
                s.artic_start[size_t{slot} * 3u + axis] = lambda[j_rows[axis]];
        }
        grid.sync();
    };

    auto schur_step = [&](bool pseudo, bool correct) {
        if (!schur) return;
        if (correct) {
            schur_gather(pseudo, true);
            grid.sync();
            schur_solve();
            grid.sync();
            schur_correct(pseudo);
            grid.sync();
        }
        schur_commit(pseudo);
    };

    // A hub couples its rows through its six velocity DOFs as an articulation does through its
    // own: S = sum J^T g J and P = J^T (lambda - start), then (M + S) du = P.
    const uint32_t hub_rows = s.control[kControlHubRows];
    const bool hubs = hub_rows != 0u && a.rigid_inv_mass != nullptr;
    const auto hub_empty = [&](uint32_t body) {
        return s.hub_range[2u * body] == s.hub_range[2u * body + 1u];
    };
    // Axis rows carry their own side terms; sides bit 0 is side a, bit 1 side b.
    const auto hub_jacobian = [&](uint32_t j_row, uint32_t sides, float* j) {
        const NkRow& row = a.urows[j_row];
        math::Vec3 linear{}, angular{};
        if (sides & 1u) { linear += row.a.jlin; angular += row.a.jang; }
        if (sides & 2u) { linear += row.b.jlin; angular += row.b.jang; }
        j[0] = linear.x; j[1] = linear.y; j[2] = linear.z;
        j[3] = angular.x; j[4] = angular.y; j[5] = angular.z;
    };

    auto hub_snapshot = [&](bool pseudo, bool zero) {
        const float* lambda = pseudo ? a.row_pseudo_lambda : a.lambda;
        for (uint32_t i = blockIdx.x * blockDim.x + lane; i < hub_rows;
             i += gridDim.x * blockDim.x) {
            const uint32_t slot = s.hub_rows[i];
            uint32_t j_rows[3];
            const uint32_t axes = SchurAxes(a.urows[slot], slot, pseudo, j_rows);
            for (uint32_t axis = 0u; axis < 3u; ++axis) {
                s.hub_start[size_t{slot} * 3u + axis] =
                    axis < axes && !zero ? lambda[j_rows[axis]] : 0.0f;
                s.artic_pending[size_t{slot} * 3u + axis] = 0.0f;
            }
        }
        grid.sync();
    };

    // One block per hub: threads stride its entries, then warps and the block sum in fixed order.
    auto hub_gather = [&](bool pseudo, bool matrix) {
        for (uint32_t body = blockIdx.x; body < s.bodies; body += gridDim.x) {
            if (hub_empty(body)) continue;
            float sum[kHubItem]{};
            for (uint32_t e = s.hub_range[2u * body] + lane; e < s.hub_range[2u * body + 1u];
                 e += blockDim.x) {
                const uint32_t entry = s.hub_entries[e];
                const uint32_t slot = entry >> 2u;
                const SchurRowTerms t = LoadSchurRowTerms(a, slot, pseudo);
                for (uint32_t axis = 0u; axis < t.axes; ++axis) {
                    float j[6];
                    hub_jacobian(t.j_rows[axis], entry & 3u, j);
                    const float delta = t.lambda[axis] - s.hub_start[size_t{slot} * 3u + axis];
                    #pragma unroll
                    for (uint32_t k = 0u; k < 6u; ++k) sum[kHubTriangle + k] += j[k] * delta;
                    const float gain = matrix ? t.gain[axis] : 0.0f;
                    if (gain == 0.0f) continue;
                    uint32_t at = 0u;
                    #pragma unroll
                    for (uint32_t r = 0u; r < 6u; ++r)
                        #pragma unroll
                        for (uint32_t c = r; c < 6u; ++c) sum[at++] += gain * j[r] * j[c];
                }
            }
            #pragma unroll
            for (uint32_t k = 0u; k < kHubItem; ++k) {
                const float value = WarpSum(sum[k]);
                if (wlane == 0u) hub_partial[warp * kHubItem + k] = value;
            }
            __syncthreads();
            if (lane < kHubItem) {
                float total = 0.0f;
                for (uint32_t w = 0u; w < nwarps; ++w) total += hub_partial[w * kHubItem + lane];
                s.hub_system[size_t{body} * kHubItem + lane] = total;
            }
            __syncthreads();
        }
    };

    // (I + M^-1 S) du = M^-1 P by pivoted elimination; M^-1 needs no inverted inertia.
    auto hub_solve = [&]() {
        for (uint32_t body = blockIdx.x * blockDim.x + lane; body < s.bodies;
             body += gridDim.x * blockDim.x) {
            if (hub_empty(body)) continue;
            const float* system = s.hub_system + size_t{body} * kHubItem;
            const float im = a.rigid_inv_mass[body];
            const math::SymmetricMat3 inertia = a.body_inv_inertia[body];
            const auto inverse_mass = [&](const float* x, float* out) {
                const math::Vec3 w = inertia.Multiply({x[3], x[4], x[5]});
                out[0] = im * x[0]; out[1] = im * x[1]; out[2] = im * x[2];
                out[3] = w.x; out[4] = w.y; out[5] = w.z;
            };
            float m[6][7];
            for (uint32_t c = 0u; c < 6u; ++c) {
                float column[6];
                for (uint32_t r = 0u; r < 6u; ++r) {
                    const uint32_t lo = r < c ? r : c, hi = r < c ? c : r;
                    column[r] = system[lo * 6u - lo * (lo - 1u) / 2u + (hi - lo)];
                }
                float scaled[6];
                inverse_mass(column, scaled);
                for (uint32_t r = 0u; r < 6u; ++r) m[r][c] = scaled[r] + (r == c ? 1.0f : 0.0f);
            }
            float rhs[6];
            inverse_mass(system + kHubTriangle, rhs);
            for (uint32_t r = 0u; r < 6u; ++r) m[r][6] = rhs[r];
            for (uint32_t c = 0u; c < 6u; ++c) {
                uint32_t pivot = c;
                for (uint32_t r = c + 1u; r < 6u; ++r)
                    if (fabsf(m[r][c]) > fabsf(m[pivot][c])) pivot = r;
                if (pivot != c)
                    for (uint32_t k = c; k < 7u; ++k) {
                        const float swap = m[c][k];
                        m[c][k] = m[pivot][k];
                        m[pivot][k] = swap;
                    }
                const float inverse = 1.0f / m[c][c];
                for (uint32_t r = c + 1u; r < 6u; ++r) {
                    const float factor = m[r][c] * inverse;
                    for (uint32_t k = c; k < 7u; ++k) m[r][k] -= factor * m[c][k];
                }
            }
            for (uint32_t cc = 6u; cc > 0u; --cc) {
                const uint32_t c = cc - 1u;
                float value = m[c][6];
                for (uint32_t k = c + 1u; k < 6u; ++k) value -= m[c][k] * m[k][6];
                m[c][6] = value / m[c][c];
            }
            for (uint32_t k = 0u; k < 6u; ++k) s.hub_step[size_t{body} * 6u + k] = m[k][6];
        }
    };

    // Each coupled row re-projects against the velocity du gives it; the change reaches its
    // other sides at the row's next visit.
    auto hub_correct = [&](bool pseudo) {
        float* const lambda = pseudo ? a.row_pseudo_lambda : a.lambda;
        for (uint32_t i = blockIdx.x * blockDim.x + lane; i < hub_rows;
             i += gridDim.x * blockDim.x) {
            const uint32_t slot = s.hub_rows[i];
            const SchurRowTerms t = LoadSchurRowTerms(a, slot, pseudo);
            if (t.gain[0] == 0.0f && t.gain[1] == 0.0f && t.gain[2] == 0.0f) continue;
            const NkRow& row = a.urows[slot];
            float w[3]{};
            for (uint32_t side = 0u; side < 2u; ++side) {
                const NkRowSide& endpoint = side == 0u ? row.a : row.b;
                if (!IsHubSide(endpoint, s) || hub_empty(endpoint.index)) continue;
                const float* step = s.hub_step + size_t{endpoint.index} * 6u;
                for (uint32_t axis = 0u; axis < t.axes; ++axis) {
                    if (t.gain[axis] == 0.0f) continue;
                    float j[6];
                    hub_jacobian(t.j_rows[axis], 1u << side, j);
                    for (uint32_t k = 0u; k < 6u; ++k) w[axis] += j[k] * step[k];
                }
            }
            float out[3];
            ProjectSchurRow(row, t, w, pseudo, out);
            for (uint32_t axis = 0u; axis < t.axes; ++axis) {
                const float delta = out[axis] - t.lambda[axis];
                if (delta == 0.0f) continue;
                lambda[t.j_rows[axis]] = out[axis];
                s.artic_pending[size_t{slot} * 3u + axis] += delta;
            }
        }
    };

    // v += M^-1 J^T (lambda - start); the hub then holds every row impulse.
    auto hub_commit = [&](bool pseudo) {
        hub_gather(pseudo, false);
        grid.sync();
        math::Vec3* const linear = pseudo ? a.body_pseudo_linear : a.body_linear;
        math::Vec3* const angular = pseudo ? a.body_pseudo_angular : a.body_angular;
        for (uint32_t body = blockIdx.x * blockDim.x + lane; body < s.bodies;
             body += gridDim.x * blockDim.x) {
            if (hub_empty(body)) continue;
            const float* moment = s.hub_system + size_t{body} * kHubItem + kHubTriangle;
            const float im = a.rigid_inv_mass[body];
            AddVelocity(linear[body], pseudo ? nullptr : a.error.Linear(body),
                        math::Vec3{moment[0], moment[1], moment[2]} * im);
            AddVelocity(angular[body], pseudo ? nullptr : a.error.Angular(body),
                        a.body_inv_inertia[body].Multiply({moment[3], moment[4], moment[5]}));
        }
        const float* lambda = pseudo ? a.row_pseudo_lambda : a.lambda;
        for (uint32_t i = blockIdx.x * blockDim.x + lane; i < hub_rows;
             i += gridDim.x * blockDim.x) {
            const uint32_t slot = s.hub_rows[i];
            uint32_t j_rows[3];
            const uint32_t axes = SchurAxes(a.urows[slot], slot, pseudo, j_rows);
            for (uint32_t axis = 0u; axis < axes; ++axis)
                s.hub_start[size_t{slot} * 3u + axis] = lambda[j_rows[axis]];
        }
        grid.sync();
    };

    auto hub_step = [&](bool pseudo, bool correct) {
        if (!hubs) return;
        if (correct) {
            hub_gather(pseudo, true);
            grid.sync();
            hub_solve();
            grid.sync();
            hub_correct(pseudo);
            grid.sync();
        }
        hub_commit(pseudo);
    };

    // After its row is staged, a warp stages the block's tangent rows and, when the sweep
    // measures velocities, every articulation arm's J.
    auto stage_chain_terms = [&](uint32_t slot, bool jacobian) {
        const NkRow& normal = staged_rows[3u * warp];
        const uint32_t axes = (normal.flags & nk::nk_row_flags::kBlockNormal)
            ? 3u : ((normal.flags & nk::nk_row_flags::kBlockTangent) ? 0u : 1u);
        const bool sequential = IsSequentialArticRow(normal, s);
        for (uint32_t axis = 0u; axis < axes; ++axis) {
            const uint32_t at = slot + axis * normal.group_normal_count;
            if (axis != 0u) {
                uint32_t word;
                memcpy(&word, reinterpret_cast<const unsigned char*>(a.urows + at) +
                              wlane * sizeof(word), sizeof(word));
                memcpy(reinterpret_cast<unsigned char*>(staged_rows + 3u * warp + axis) +
                           wlane * sizeof(word), &word, sizeof(word));
            }
            if (!jacobian && !sequential) continue;
            for (uint32_t k = wlane; k < dofs; k += warpSize) {
                const size_t destination = size_t{3u * warp + axis} * dofs + k;
                const size_t source = size_t{at} * dofs + k;
                if (normal.a.kind == kNkSideArtic) {
                    if (jacobian) staged_j[destination] = a.chain_jacobian[source];
                    if (sequential) staged_w[destination] = a.row_minv_jt[source];
                }
                if (normal.b.kind == kNkSideArtic) {
                    if (jacobian) staged_j[staged_size + destination] = a.chain_jacobian_b[source];
                    if (sequential) staged_w[staged_size + destination] = a.row_minv_jt_b[source];
                }
            }
        }
        __syncwarp();
    };

    auto chain_velocity = [&](uint32_t first, uint32_t last, uint32_t env_row_base,
                              uint32_t env_artic_base, float* qdot) {
        // Sequential rows move articulation velocities under the compensation the commit uses.
        VelocityErrorView chain_error = a.error;
        chain_error.qdot = s.qdot_error + size_t{env_artic_base} * dofs;
        for (uint32_t base = first; base < last; base += nwarps) {
            const uint32_t batch_count = min(nwarps, last - base);
            for (uint32_t i = lane; i < kChainCacheEntries; i += blockDim.x)
                cache_key[i] = kChainCacheEmpty;
            if (lane == 0u) batch_needed = 0u;
            __syncthreads();
            // Each warp screens and stages its row together; bit 1 marks a batch left uncached.
            // A row owing a Schur correction always solves, through the uncached path.
            if (warp < batch_count) {
                const uint32_t slot = s.chain_rows[base + warp];
                NkRow* const staged = staged_rows + 3u * warp;
                StageRowWarp(a.urows, slot, staged, wlane);
                const bool tangent = (staged[0].flags & nk::nk_row_flags::kBlockTangent) != 0u;
                const float* owed = pending(staged[0], slot);
                const bool corrected = owed != nullptr &&
                    (owed[0] != 0.0f || owed[1] != 0.0f || owed[2] != 0.0f);
                const bool needed = !tangent && (corrected || RowNeedsVelocitySolveWarp(a.urows,
                    slot, env_row_base, env_artic_base, a.lambda, a.row_meff, a.row_damping,
                    a.chain_jacobian, a.chain_jacobian_b, qdot, a.body_linear,
                    a.body_angular, a.points, dofs, a.dt, wlane, staged));
                stage_chain_terms(slot, true);
                const bool cacheable = !corrected && StageChainPoints(staged, slot, warp, wlane,
                    cache, a.points, a.error, a.lambda, a.row_damping);
                if (wlane == 0u) {
                    batch_slot[warp] = slot;
                    if (needed || !cacheable)
                        atomicOr(&batch_needed, (needed ? 1u : 0u) | (cacheable ? 0u : 2u));
                }
            }
            __syncthreads();
            // A batch is skipped only if every row leaves its input unchanged.
            if ((batch_needed & 1u) == 0u) {
                __syncthreads();
                continue;
            }
            const bool cached = (batch_needed & 2u) == 0u;
            // One warp applies the staged batch in chain order, so its rows need no block barrier.
            for (uint32_t idx = warp == 0u ? 0u : batch_count; idx < batch_count; ++idx) {
                const uint32_t gslot = batch_slot[idx];
                NkRow* const prepared = staged_rows + 3u * idx;
                if (prepared[0].flags & nk::nk_row_flags::kBlockTangent) continue;
                const bool block = (prepared[0].flags & nk::nk_row_flags::kBlockNormal) != 0u;
                float* const owed = pending(prepared[0], gslot);
                const bool sequential = IsSequentialArticRow(prepared[0], s);
                const float* const w = sequential ? staged_w : nullptr;
                const float* const w_b = sequential ? staged_w + staged_size : nullptr;
                const bool significant = block && cached
                    ? SolveChainContactBlockWarp(gslot, idx, env_artic_base, wlane, prepared,
                          cache, a.lambda, staged_j, w, staged_j + staged_size, w_b,
                          qdot, a.body_linear, a.body_angular, a.body_inv_mass,
                          a.body_inv_inertia, a.points, dofs, a.dt, a.vel_tolerance,
                          chain_error, 3u * idx)
                    : block
                    ? SolvePreparedContactBlockWarp(gslot, env_artic_base, wlane, a.urows,
                          prepared, a.lambda, a.row_damping, staged_j, w,
                          staged_j + staged_size, w_b, qdot, a.body_linear, a.body_angular,
                          a.body_inv_mass, a.body_inv_inertia, a.points, dofs, a.dt,
                          a.vel_tolerance, chain_error, 3u * idx, 1u, true, owed)
                    : SolveUnionRowWarp(gslot - env_row_base, gslot, env_row_base,
                          env_artic_base, 3u * idx, wlane, nullptr, nullptr, nullptr, nullptr,
                          a.lambda, a.row_meff, a.row_damping, staged_j, w,
                          staged_j + staged_size, w_b, qdot, a.urows, a.body_linear,
                          a.body_angular, a.body_inv_mass, a.body_inv_inertia, a.points, dofs,
                          a.dt, false, chain_error, prepared, a.vel_tolerance, owed);
                if (significant && wlane == 0u) changed = true;
                __syncwarp();
            }
            __syncthreads();
            if (cached) CommitChainPoints(cache, a.points, a.error);
            __syncthreads();
        }
    };

    auto chain_position = [&](uint32_t first, uint32_t last, uint32_t env_row_base,
                              uint32_t env_artic_base, float* qdot) {
        for (uint32_t base = first; base < last; base += nwarps) {
            const uint32_t batch_count = min(nwarps, last - base);
            // An unchanged batch cannot alter a later row's input during its ordered sweep.
            if (lane == 0u) batch_needed = 0u;
            __syncthreads();
            if (warp < batch_count) {
                const uint32_t gslot = s.chain_rows[base + warp];
                const bool needed = SolvePositionRowWarp<false>(gslot, env_row_base,
                    env_artic_base, gslot, wlane, a.row_meff, a.row_penetration,
                    a.row_pseudo_lambda, a.chain_jacobian, nullptr, a.chain_jacobian_b,
                    nullptr, qdot, a.urows, a.body_pseudo_linear, a.body_pseudo_angular,
                    a.body_inv_mass, a.body_inv_inertia, pseudo_points, dofs, a.pos_beta,
                    a.pos_slop, a.dt, a.baumgarte_max_velocity, nullptr,
                    pending(a.urows[gslot], gslot));
                if (needed && wlane == 0u) atomicOr(&batch_needed, 1u);
            }
            __syncthreads();
            if (batch_needed == 0u) {
                __syncthreads();
                continue;
            }
            for (uint32_t idx = warp == 0u ? 0u : batch_count; idx < batch_count; ++idx) {
                const uint32_t gslot = s.chain_rows[base + idx];
                const bool sequential = IsSequentialArticRow(a.urows[gslot], s);
                SolvePositionRowWarp(gslot, env_row_base, env_artic_base, gslot, wlane,
                    a.row_meff, a.row_penetration, a.row_pseudo_lambda, a.chain_jacobian,
                    sequential ? a.row_minv_jt : nullptr, a.chain_jacobian_b,
                    sequential ? a.row_minv_jt_b : nullptr, qdot, a.urows,
                    a.body_pseudo_linear, a.body_pseudo_angular, a.body_inv_mass,
                    a.body_inv_inertia, pseudo_points, dofs, a.pos_beta, a.pos_slop, a.dt,
                    a.baumgarte_max_velocity, nullptr, pending(a.urows[gslot], gslot));
                __syncwarp();
            }
            __syncthreads();
        }
    };

    auto chain_sweep = [&](ChainSweep sweep) {
        float* const velocity = sweep == ChainSweep::Position ? a.qdot_pseudo_flat : a.qdot_flat;
        for (uint32_t k = blockIdx.x; k < chain_islands; k += gridDim.x) {
            const IslandRecord rec =
                reinterpret_cast<const IslandRecord*>(a.islands)[s.chain_islands[k]];
            const uint32_t env_row_base = rec.env * a.rows_per_env;
            const uint32_t env_artic_base = rec.env * k_tiles;
            const uint32_t live_off = rec.seg_off == 0u ? 0u : a.live_scan[rec.seg_off - 1u];
            const uint32_t first = s.chain_excl[live_off];
            const uint32_t last = s.chain_excl[a.live_scan[rec.seg_off + rec.seg_cnt - 1u]];
            float* const qdot = env_qdot(velocity, rec.env);
            if (sweep == ChainSweep::WarmStart) {
                VelocityErrorView chain_error = a.error;
                chain_error.qdot = s.qdot_error + size_t{env_artic_base} * dofs;
                // Every warp stages a row of the batch; one warp applies them in chain order.
                for (uint32_t base = first; base < last; base += nwarps) {
                    const uint32_t batch_count = min(nwarps, last - base);
                    if (warp < batch_count) {
                        const uint32_t slot = s.chain_rows[base + warp];
                        StageRowWarp(a.urows, slot, staged_rows + 3u * warp, wlane);
                        stage_chain_terms(slot, false);
                    }
                    __syncthreads();
                    for (uint32_t idx = warp == 0u ? 0u : batch_count; idx < batch_count; ++idx) {
                        const uint32_t slot = s.chain_rows[base + idx];
                        NkRow* const prepared = staged_rows + 3u * idx;
                        const bool sequential = IsSequentialArticRow(prepared[0], s);
                        SolveUnionRowWarp(slot - env_row_base, slot, env_row_base,
                            env_artic_base, 3u * idx, wlane, nullptr, nullptr, nullptr, nullptr,
                            a.lambda, a.row_meff, a.row_damping, staged_j,
                            sequential ? staged_w : nullptr, staged_j + staged_size,
                            sequential ? staged_w + staged_size : nullptr, qdot, a.urows,
                            a.body_linear, a.body_angular, a.body_inv_mass, a.body_inv_inertia,
                            a.points, dofs, a.dt, true, chain_error, prepared, 0.0f,
                            pending(prepared[0], slot));
                        __syncwarp();
                    }
                    __syncthreads();
                }
            } else if (sweep == ChainSweep::Velocity) {
                chain_velocity(first, last, env_row_base, env_artic_base, qdot);
            } else {
                chain_position(first, last, env_row_base, env_artic_base, qdot);
            }
            __syncthreads();
        }
    };

    // Each static color of vertex blocks steps its vertices in every env across the grid's warps.
    const nkops::VertexBlockView& blocks = a.vertex_blocks;
    auto vertex_sweep = [&]() {
        for (uint32_t color = 0u; color < blocks.layout.colors; ++color) {
            const uint32_t first = blocks.color_segments[2u * color];
            const uint32_t count = blocks.color_segments[2u * color + 1u];
            for (uint32_t item = color_warp; item < count * blocks.layout.env_count;
                 item += color_warps) {
                const uint32_t env = item / count;
                const uint32_t vertex = blocks.color_vertices[first + item - env * count];
                changed |= nkops::SolveVertexBlockWarp(blocks, a.points, a.error.particle, env,
                                                       vertex, wlane, a.vel_tolerance);
            }
            grid.sync();
        }
    };

    // A verifying pass sweeps velocity only when the solved velocity broke an idle row.
    const bool sweep_velocity = !a.verify_idle || s.control[kControlIdleViolations] != 0u;
    if (sweep_velocity && schur) schur_snapshot(false, a.apply_cached);
    if (sweep_velocity && hubs) hub_snapshot(false, a.apply_cached);
    if (sweep_velocity && a.apply_cached) {
        color_sweep([&](uint32_t slot, uint32_t env, NkRow* staged) {
            SolveUnionRowWarp(slot - env * a.rows_per_env, slot, env * a.rows_per_env,
                env * k_tiles, slot, wlane, nullptr, nullptr, nullptr, nullptr, a.lambda,
                a.row_meff, a.row_damping, a.chain_jacobian, nullptr, a.chain_jacobian_b,
                nullptr, env_qdot(a.qdot_flat, env), a.urows, a.body_linear, a.body_angular,
                a.body_inv_mass, a.body_inv_inertia, a.points, dofs, a.dt, true, a.error,
                nullptr, 0.0f, pending(staged[0], slot));
        }, [&](uint32_t i, uint32_t, uint32_t) {
            SolveLaneRecord(s.lane[i], a.lambda, a.points, a.error, true, 0.0f);
        }, [&](uint32_t slot, uint32_t lane, uint32_t mask) {
            SolveMaterialBlockGroup<kPairRowWidth>(slot, a.urows[slot], a.urows, a.lambda,
                a.points, a.dt, true, 0.0f, a.error, lane, mask);
        });
        chain_sweep(ChainSweep::WarmStart);
        grid.sync();
        schur_step(false, false);
        hub_step(false, false);
    }
    lap(kPhaseWarmStart);
    uint32_t completed_sweeps = 0u;
    for (uint32_t it = 0u; sweep_velocity && it < a.vel_iters; ++it) {
        completed_sweeps = it + 1u;
        if (blockIdx.x == 0u && threadIdx.x == 0u)
            s.control[kControlChanged + (it + 1u) % 3u] = 0u;
        changed = false;
        vertex_sweep();
        lap(kPhaseVertex);
        color_sweep([&](uint32_t slot, uint32_t env, NkRow* staged) {
            const uint32_t env_row_base = env * a.rows_per_env;
            float* const qdot = env_qdot(a.qdot_flat, env);
            float* const owed = pending(staged[0], slot);
            bool significant = false;
            if (staged[0].flags & nk::nk_row_flags::kBlockNormal) {
                significant = SolvePreparedContactBlockWarp(slot, env * k_tiles, wlane, a.urows,
                    staged, a.lambda, a.row_damping, a.chain_jacobian, nullptr,
                    a.chain_jacobian_b, nullptr, qdot, a.body_linear, a.body_angular,
                    a.body_inv_mass, a.body_inv_inertia, a.points, dofs, a.dt, a.vel_tolerance,
                    a.error, slot, staged[0].group_normal_count, false, owed);
            } else {
                significant = SolveUnionRowWarp(slot - env_row_base, slot, env_row_base,
                    env * k_tiles, slot, wlane, nullptr, nullptr, nullptr, nullptr, a.lambda,
                    a.row_meff, a.row_damping, a.chain_jacobian, nullptr,
                    a.chain_jacobian_b, nullptr, qdot, a.urows, a.body_linear,
                    a.body_angular, a.body_inv_mass, a.body_inv_inertia, a.points, dofs, a.dt,
                    false, a.error, staged, a.vel_tolerance, owed);
            }
            changed |= significant;
        }, [&](uint32_t i, uint32_t, uint32_t) {
            changed |= SolveLaneRecord(s.lane[i], a.lambda, a.points, a.error, false,
                                       a.vel_tolerance);
        }, [&](uint32_t slot, uint32_t lane, uint32_t mask) {
            changed |= SolveMaterialBlockGroup<kPairRowWidth>(slot, a.urows[slot], a.urows,
                a.lambda, a.points, a.dt, false, a.vel_tolerance, a.error, lane, mask);
        });
        lap(kPhaseColor);
        chain_sweep(ChainSweep::Velocity);
        uint32_t* const flag = s.control + kControlChanged + it % 3u;
        if (__syncthreads_or(changed) && threadIdx.x == 0u) atomicOr(flag, 1u);
        grid.sync();
        lap(kPhaseChain);
        bool more = LoadControl(flag) != 0u;
        schur_step(false, more && it + 1u < a.vel_iters);
        lap(kPhaseSchur);
        hub_step(false, more && it + 1u < a.vel_iters);
        lap(kPhaseHub);
        if (!more && blocks.layout.vertices != 0u) {
            bool nonstationary = false;
            for (uint32_t item = color_warp;
                 item < blocks.layout.vertices * blocks.layout.env_count; item += color_warps) {
                const uint32_t env = item / blocks.layout.vertices;
                const uint32_t vertex = item - env * blocks.layout.vertices;
                nonstationary |= !nkops::VertexBlockStationaryWarp(
                    blocks, a.points, a.error.particle, env, vertex, wlane, a.vel_tolerance);
            }
            if (__syncthreads_or(nonstationary) && threadIdx.x == 0u) atomicOr(flag, 1u);
            grid.sync();
            more = LoadControl(flag) != 0u;
        }
        lap(kPhaseStationary);
        if (!more) break;
    }
    if (a.vbd_velocity_sweep_count != nullptr) {
        for (uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
             env < a.env_count; env += gridDim.x * blockDim.x)
            a.vbd_velocity_sweep_count[env] += completed_sweeps;
    }

    // Penetration drives a fresh pseudo-velocity field over the same colors and chains.
    if (pos_pass) {
        // Lanes screen 32 rows for the speculative flag; the warp then advances each flagged row.
        for (uint32_t base = color_warp * warpSize; base < live; base += color_warps * warpSize) {
            const uint32_t position = base + wlane;
            const uint32_t row_slot = position < live ? a.live_order[position] : 0u;
            uint32_t flagged = __ballot_sync(0xffffffffu, position < live &&
                (a.urows[row_slot].flags & nk::nk_row_flags::kSpeculative));
            while (flagged != 0u) {
                const uint32_t source = static_cast<uint32_t>(__ffs(static_cast<int>(flagged))) - 1u;
                flagged &= flagged - 1u;
                const uint32_t slot = __shfl_sync(0xffffffffu, row_slot, source);
                const uint32_t env = slot / a.rows_per_env;
                UpdateSpeculativePenetration<true>(slot, env * a.rows_per_env, env * k_tiles,
                    a.row_penetration, a.urows, a.chain_jacobian, a.chain_jacobian_b,
                    env_qdot(a.qdot_flat, env), a.body_linear, a.body_angular, a.points, dofs,
                    a.dt, wlane);
            }
        }
        grid.sync();
        if (schur) schur_snapshot(true, false);
        if (hubs) hub_snapshot(true, false);
        for (uint32_t it = 0u; it < a.pos_iters; ++it) {
            color_sweep([&](uint32_t slot, uint32_t env, const NkRow* staged) {
                SolvePositionRowWarp(slot, env * a.rows_per_env, env * k_tiles, slot, wlane,
                    a.row_meff, a.row_penetration, a.row_pseudo_lambda, a.chain_jacobian,
                    nullptr, a.chain_jacobian_b, nullptr, env_qdot(a.qdot_pseudo_flat, env),
                    a.urows, a.body_pseudo_linear, a.body_pseudo_angular, a.body_inv_mass,
                    a.body_inv_inertia, pseudo_points, dofs, a.pos_beta, a.pos_slop, a.dt,
                    a.baumgarte_max_velocity, staged, pending(staged[0], slot));
            }, [&](uint32_t, uint32_t slot, uint32_t env) {
                SolvePositionRowScalar(slot, env * a.rows_per_env, env * k_tiles, a.row_meff,
                    a.row_penetration, a.row_pseudo_lambda, a.urows, a.body_pseudo_linear,
                    a.body_pseudo_angular, a.body_inv_mass, a.body_inv_inertia, pseudo_points,
                    a.pos_beta, a.pos_slop, a.dt, a.baumgarte_max_velocity);
            }, [](uint32_t, uint32_t, uint32_t) {}, true);
            chain_sweep(ChainSweep::Position);
            grid.sync();
            schur_step(true, it + 1u < a.pos_iters);
            hub_step(true, it + 1u < a.pos_iters);
        }
    }
    lap(kPhasePosition);
    if (timed) {
        for (uint32_t env = 0u; env < a.env_count; ++env)
            for (uint32_t k = 0u; k < kSolverPhases; ++k)
                a.solver_phase_time[size_t{env} * kSolverPhases + k] += phase_time[k];
    }

    // Every articulation tile scatters through the cooked DOF maps, solved or not.
    if (dofs == 0u || a.articulation_count == 0u || a.dof_to_link == nullptr ||
        a.dof_to_component == nullptr || a.qdot_flat == nullptr) return;
    const uint32_t links_per_dog = a.base_link_count / k_tiles;
    const uint32_t total = a.articulation_count * dofs;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += gridDim.x * blockDim.x) {
        const uint32_t tile = t / dofs;
        const uint32_t k = t - tile * dofs;
        const uint32_t env = tile / k_tiles;
        uint32_t comp;
        size_t gl;
        ArticDofTarget(env, tile - env * k_tiles, k, dofs, links_per_dog, a.base_link_count,
                       a.dof_to_link, a.dof_to_component, comp, gl);
        const float v = a.qdot_flat[t];
        if (comp != ~0u) a.link_velocity[gl].v[comp] = v;
        else a.qdot[gl] = v;
        if (!pos_pass || a.qdot_pseudo_flat == nullptr) continue;
        const float vp = a.qdot_pseudo_flat[t];
        if (comp != ~0u) a.link_velocity_pseudo[gl].v[comp] = vp;
        else a.qdot_pseudo[gl] = vp;
    }
}

Status SolveColoredIslands(const ModelView& model, const DataView& data,
                           const SolveRowsBlockIslandParams& p, IslandActivityView activity,
                           const ColorScratch& scratch, const uint32_t* live_scan,
                           const uint32_t* live_order, VelocityErrorView error,
                           uint32_t artics_per_env, bool pos_pass, cudaStream_t stream) {
    const auto cooperative = RequireCooperativeLaunch();
    if (cooperative == cudaErrorNotSupported) return Status::Unsupported;
    if (cooperative != cudaSuccess) return Status::Failed;
    if (live_scan == nullptr || live_order == nullptr || p.rows_per_env == 0u)
        return Status::InvalidArgument;
    const uint64_t owners = uint64_t{p.total_body_count} + p.total_particle_count +
                            p.total_grid_count + uint64_t{p.rows_per_env} * p.env_count;
    if (owners >= kOwnerEmpty) return Status::InvalidArgument;
    int device = 0;
    int sm_count = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess)
        return Status::Failed;
    const uint32_t grid_bound = static_cast<uint32_t>(std::min<uint64_t>(
        kColorGridLimit, uint64_t{static_cast<uint32_t>(sm_count)} * kColorBlocksPerSm));
    constexpr uint32_t warps = kColorBlockSize / 32u;
    // Chain J and M^-1 J^T staging, a Schur gather batch, or one articulation's matrix, factor
    // and right side.
    const size_t staged = 3u * size_t{warps} * p.max_dof;
    const size_t solve_shared = sizeof(float) * std::max({4u * staged, staged + 6u * warps,
        2u * size_t{p.max_dof} * p.max_dof + size_t{p.max_dof}});
    uint32_t solve_blocks = 0u;
    if (ResidentGridSize(SolveColoredRowsKernel, kColorBlockSize, solve_shared, grid_bound,
                         &solve_blocks) != cudaSuccess)
        return Status::Failed;
    const PointMassView points = PointMasses(data);
    const auto* urows = reinterpret_cast<const NkRow*>(data.urows);
    ColoredSolveArgs args;
    args.urows = urows;
    args.lambda = data.lambda;
    args.chain_jacobian = static_cast<const float*>(data.chain_jacobian);
    args.row_minv_jt = static_cast<const float*>(data.row_minv_jt);
    args.chain_jacobian_b = static_cast<const float*>(data.chain_jacobian_b);
    args.row_minv_jt_b = static_cast<const float*>(data.row_minv_jt_b);
    args.row_meff = static_cast<const float*>(data.row_meff);
    args.row_damping = static_cast<const float*>(data.row_damping);
    // Mimic couplings leave the reduced mass, with empty coupled rows, and the coupled inverse.
    args.articulation_mass = p.mimic_couplings != 0u ? data.m_reduced : data.m;
    args.articulation_mass_inv = p.mimic_couplings != 0u ? data.m_inv_coupled : data.m_inv;
    args.qdot_flat = data.qdot_flat;
    args.link_velocity = reinterpret_cast<Spatial6*>(data.link_velocity);
    args.qdot = data.qdot;
    args.body_linear = data.body_linear_velocity;
    args.body_angular = data.body_angular_velocity;
    args.body_inv_mass = scratch.owner_inv_mass;
    args.rigid_inv_mass = static_cast<const float*>(data.body_inv_mass);
    args.body_inv_inertia = static_cast<const math::SymmetricMat3*>(data.body_world_inv_inertia);
    args.points = points;
    args.islands = data.island_quads;
    args.live_scan = live_scan;
    args.live_order = live_order;
    args.island_root_sorted = data.island_root_sorted;
    args.cc_root = data.cc_root;
    args.cc_artic_first = p.articulation_count > 0u ? data.cc_artic_first : nullptr;
    args.dof_to_link = static_cast<const uint32_t*>(model.dof_to_link);
    args.dof_to_component = static_cast<const uint32_t*>(model.dof_to_component);
    if (pos_pass) {
        args.row_penetration = data.row_penetration;
        args.row_pseudo_lambda = static_cast<float*>(data.row_pseudo_lambda);
        args.qdot_pseudo = static_cast<float*>(data.qdot_pseudo);
        args.link_velocity_pseudo = reinterpret_cast<Spatial6*>(data.link_velocity_pseudo);
        args.qdot_pseudo_flat = static_cast<float*>(data.qdot_pseudo_flat);
        args.body_pseudo_linear = data.body_pseudo_linear_velocity;
        args.body_pseudo_angular = data.body_pseudo_angular_velocity;
        args.particle_pseudo = data.particle_pseudo_vel;
        args.grid_pseudo = data.grid_pseudo_vel;
    }
    args.error = error;
    args.vbd_velocity_sweep_count = data.vbd_velocity_sweep_count;
    args.solver_color_counts = data.solver_color_counts;
    args.solver_phase_time = data.solver_phase_time;
    args.env_count = p.env_count;
    if (p.vertex_blocks.colors != 0u) {
        if (data.particle_response == nullptr || data.particle_row_impulse == nullptr ||
            model.vbd_color_segments == nullptr ||
            data.vbd_free_rate == nullptr || data.vbd_inertia == nullptr || data.vbd_step == nullptr ||
            (p.rows_per_env != 0u && data.cc_particle_first == nullptr))
            return Status::InvalidArgument;
        nkops::VertexBlockView& blocks = args.vertex_blocks;
        blocks.elements = model.vbd_elements;
        blocks.membrane_start = data.vbd_membrane_start;
        blocks.offsets = model.vbd_incidence_offsets;
        blocks.incidence = model.vbd_incidence;
        blocks.color_vertices = model.vbd_color_vertices;
        blocks.color_segments = model.vbd_color_segments;
        blocks.start = data.particle_prev_pos;
        blocks.free_rate = data.vbd_free_rate;
        blocks.inertia = data.vbd_inertia;
        blocks.effective_step = data.vbd_step;
        blocks.solver_audit = p.measure_vertex_audit != 0u ? data.vbd_solve_audit : nullptr;
        blocks.row_first = p.rows_per_env != 0u ? data.cc_particle_first : nullptr;
        blocks.env_status = data.env_status;
        blocks.layout = p.vertex_blocks;
        blocks.dt = p.dt;
    }
    args.rows_per_env = p.rows_per_env;
    args.artics_per_env = artics_per_env;
    args.articulation_count = p.articulation_count;
    args.dof_stride = p.max_dof;
    args.base_link_count = p.base_link_count;
    args.vel_iters = static_cast<uint32_t>(p.vel_iters);
    args.pos_iters = pos_pass ? static_cast<uint32_t>(p.pos_iters) : 0u;
    args.pos_beta = p.pos_beta;
    args.pos_slop = p.pos_slop;
    args.dt = p.dt;
    args.vel_tolerance = p.vel_tolerance;
    args.baumgarte_max_velocity = p.baumgarte_max_velocity;
    args.apply_cached = p.continue_impulses == 0u;
    args.verify_idle = p.verify_idle != 0u;
    if (LaunchCooperativeCuda(SolveColoredRowsKernel, dim3(solve_blocks), dim3(kColorBlockSize),
            solve_shared, stream, args, scratch) != cudaSuccess)
        return Status::Failed;
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

// Island rows outside the position sweeps advance their speculative depth after the solve.
// The solve writes only pseudo velocities after its own update, so both read the same state.
__global__ void UpdateIdlePenetrationKernel(
    const NkRow* __restrict__ urows, const float* __restrict__ chain_jacobian,
    const float* __restrict__ chain_jacobian_b, const float* __restrict__ qdot_flat,
    const math::Vec3* __restrict__ body_lin_vel, const math::Vec3* __restrict__ body_ang_vel,
    PointMassView point_masses, const uint32_t* __restrict__ islands,
    const uint32_t* __restrict__ island_count_dev, IslandActivityView activity,
    const uint32_t* __restrict__ row_order, const uint32_t* __restrict__ live_scan,
    float* __restrict__ row_penetration, uint32_t rows_per_env, uint32_t artics_per_env,
    uint32_t dof_stride, float dt) {
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const uint32_t stride = gridDim.x * (blockDim.x / warpSize);
    const uint32_t k_tiles = artics_per_env == 0u ? 1u : artics_per_env;
    const uint32_t island_count = *island_count_dev;
    for (uint32_t island = 0u; island < island_count; ++island) {
        const IslandRecord rec = reinterpret_cast<const IslandRecord*>(islands)[island];
        const bool active = activity.Active(rec);
        if (!(rec.flags & kIslandWarpWork) || (!active && !(rec.flags & kIslandHasArticulation)))
            continue;
        const uint32_t env_artic_base = rec.env * k_tiles;
        const float* const qdot = qdot_flat != nullptr
            ? qdot_flat + static_cast<size_t>(env_artic_base) * dof_stride : nullptr;
        // Positions stride across the grid, so consecutive islands share the warps evenly.
        const uint32_t end = rec.seg_off + rec.seg_cnt;
        for (uint32_t idx = rec.seg_off + (warp + stride - rec.seg_off % stride) % stride;
             idx < end; idx += stride) {
            const bool live = live_scan == nullptr ||
                live_scan[idx] != (idx == 0u ? 0u : live_scan[idx - 1u]);
            if (active && live) continue;
            UpdateSpeculativePenetration<true>(row_order[idx], rec.env * rows_per_env,
                env_artic_base, row_penetration, urows, chain_jacobian, chain_jacobian_b, qdot,
                body_lin_vel, body_ang_vel, point_masses, dof_stride, dt, lane);
        }
    }
}

// Rows an active island left idle are tested at the solved velocity; a row whose next step
// would move it beyond the sweep tolerance counts as broken.
__global__ void CountViolatedIdleRowsKernel(
    const NkRow* __restrict__ urows, const float* __restrict__ lambda,
    const float* __restrict__ row_meff, const float* __restrict__ row_damping,
    const float* __restrict__ chain_jacobian, const float* __restrict__ chain_jacobian_b,
    const float* __restrict__ qdot_flat, const math::Vec3* __restrict__ body_lin_vel,
    const math::Vec3* __restrict__ body_ang_vel, PointMassView point_masses,
    const uint32_t* __restrict__ islands, const uint32_t* __restrict__ island_count_dev,
    IslandActivityView activity, const uint32_t* __restrict__ row_order,
    const uint32_t* __restrict__ live_scan, uint32_t* __restrict__ violations,
    uint32_t rows_per_env, uint32_t artics_per_env, uint32_t dof_stride, float dt,
    float vel_tolerance) {
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const uint32_t stride = gridDim.x * (blockDim.x / warpSize);
    const uint32_t k_tiles = artics_per_env == 0u ? 1u : artics_per_env;
    const uint32_t island_count = *island_count_dev;
    uint32_t broken = 0u;
    for (uint32_t island = 0u; island < island_count; ++island) {
        const IslandRecord rec = reinterpret_cast<const IslandRecord*>(islands)[island];
        if (!(rec.flags & kIslandWarpWork) || !activity.Active(rec)) continue;
        const uint32_t env_artic_base = rec.env * k_tiles;
        const float* const qdot = qdot_flat != nullptr
            ? qdot_flat + static_cast<size_t>(env_artic_base) * dof_stride : nullptr;
        const uint32_t end = rec.seg_off + rec.seg_cnt;
        for (uint32_t idx = rec.seg_off + (warp + stride - rec.seg_off % stride) % stride;
             idx < end; idx += stride) {
            if (live_scan[idx] != (idx == 0u ? 0u : live_scan[idx - 1u])) continue;
            const uint32_t slot = row_order[idx];
            const uint32_t flags = urows[slot].flags;
            if (!(flags & nk::nk_row_flags::kActive) || (flags & nk::nk_row_flags::kBlockTangent))
                continue;
            const NkRow row = LoadRowWarp(urows, slot, lane);
            const IdleRowStep step = IdleRowStepWarp(urows, row, slot, rec.env * rows_per_env,
                env_artic_base, lambda, row_meff, row_damping, chain_jacobian, chain_jacobian_b,
                qdot, body_lin_vel, body_ang_vel, point_masses, dof_stride, dt, lane);
            if (vel_tolerance > 0.0f ? step.velocity > vel_tolerance : step.impulse != 0.0f)
                ++broken;
        }
    }
    if (lane == 0u && broken != 0u) atomicAdd(violations, broken);
}

// Expose lower/upper impulses independently from contact and actuator telemetry.
__global__ void WriteJointLimitImpulseKernel(
    const float* __restrict__ lambda, uint32_t total_link_count,
    uint32_t env_count, uint32_t base_link_count, uint32_t rows_per_env,
    uint32_t contact_rows_per_env, float* __restrict__ limit_impulse) {
    const uint32_t link = blockIdx.x * blockDim.x + threadIdx.x;
    if (link >= total_link_count) return;
    const uint32_t env = link / base_link_count;
    if (env >= env_count) return;
    const uint32_t local_link = link - env * base_link_count;
    const uint32_t row_base = env * rows_per_env + contact_rows_per_env +
                              local_link * 2u;
    limit_impulse[static_cast<size_t>(link) * 2u] = lambda[row_base];
    limit_impulse[static_cast<size_t>(link) * 2u + 1u] = lambda[row_base + 1u];
}

// --- op entry point ---------------------------------------------------------

// Vertex stationarity is measured after all row and vertex updates, without advancing state.
__global__ void MeasureVertexResidualKernel(ModelView model, DataView data,
                                            SolveRowsBlockIslandParams p,
                                            const math::Vec3* particle_error) {
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const uint32_t stride = gridDim.x * (blockDim.x / warpSize);
    nkops::VertexBlockView b;
    b.elements = model.vbd_elements;
    b.membrane_start = data.vbd_membrane_start;
    b.offsets = model.vbd_incidence_offsets;
    b.incidence = model.vbd_incidence;
    b.start = data.particle_prev_pos;
    b.free_rate = data.vbd_free_rate;
    b.inertia = data.vbd_inertia;
    b.solver_audit = p.measure_vertex_audit != 0u ? data.vbd_solve_audit : nullptr;
    b.layout = p.vertex_blocks;
    b.dt = p.dt;
    for (uint32_t item = warp; item < b.layout.dynamic_vertices * b.layout.env_count; item += stride) {
        const uint32_t env = item / b.layout.dynamic_vertices;
        const uint32_t vertex = model.vbd_color_vertices[item % b.layout.dynamic_vertices];
        const uint32_t particle = b.Particle(env, vertex);
        if (!(data.particle_inv_mass[particle] > 0.0f)) continue;
        const uint32_t slot = b.Slot(env, vertex);
        math::Vec3 gradient;
        math::SymmetricMat3 hessian;
        nkops::GatherVertexBlock(b, data.particle_vel, env, vertex, lane, warpSize, gradient, hessian);
        gradient = {WarpSum(gradient.x), WarpSum(gradient.y), WarpSum(gradient.z)};
        hessian = {WarpSum(hessian.xx), WarpSum(hessian.yy), WarpSum(hessian.zz),
                   WarpSum(hessian.xy), WarpSum(hessian.xz), WarpSum(hessian.yz)};
        if (lane != 0u) continue;
        const math::Vec3 u = data.particle_vel[particle] -
            (particle_error != nullptr ? particle_error[particle] : math::Vec3{});
        const float inertia = b.inertia[slot];
        const math::Vec3 force = -nk::vbd::InertialGradient(u, b.free_rate[slot], inertia, b.dt) - gradient +
                                data.particle_row_impulse[particle] / b.dt;
        nk::vbd::AddIdentity(hessian, inertia);
        math::SymmetricMat3 inverse;
        const bool invertible = nk::vbd::Invert(hessian, 1.0e-6f * inertia * inertia * inertia, &inverse);
        const math::Vec3 correction = invertible ? inverse.Multiply(force) / b.dt : math::Vec3{};
        nkops::RecordVertexEquationAudit(b, env, vertex, force, correction);
        const auto magnitude = [](math::Vec3 v) {
            if (!(fabsf(v.x) <= FLT_MAX && fabsf(v.y) <= FLT_MAX && fabsf(v.z) <= FLT_MAX))
                return FLT_MAX;
            return fmaxf(fabsf(v.x), fmaxf(fabsf(v.y), fabsf(v.z)));
        };
        const float metrics[2] = {invertible ? magnitude(correction) : FLT_MAX, magnitude(force)};
        for (uint32_t metric = 0u; metric < 2u; ++metric) {
            const unsigned long long packed =
                (static_cast<unsigned long long>(__float_as_uint(metrics[metric])) << 32u) | (~particle);
            atomicMax(reinterpret_cast<unsigned long long*>(data.vbd_solve_metrics + env * 2u + metric), packed);
        }
    }
}

// Diagnostic reads occur after every island has committed its physical velocity and impulse.
__global__ void MeasureContactResidualKernel(DataView data, SolveRowsBlockIslandParams p) {
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const uint32_t stride = gridDim.x * (blockDim.x / warpSize);
    const uint32_t artics = p.articulation_count > 0u ? p.articulation_count / p.env_count : 1u;
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const PointMassView points = PointMasses(data);
    for (uint32_t slot = warp; slot < p.rows_per_env * p.env_count; slot += stride) {
        const NkRow normal = LoadRowWarp(rows, slot, lane);
        if (!(normal.flags & nk::nk_row_flags::kActive) ||
            !(normal.flags & nk::nk_row_flags::kBlockNormal)) continue;
        const uint32_t env = slot / p.rows_per_env;
        const uint32_t env_row = env * p.rows_per_env;
        const uint32_t env_artic = env * artics;
        const float* qdot = data.qdot_flat != nullptr ?
            data.qdot_flat + static_cast<size_t>(env_artic) * p.max_dof : nullptr;
        math::Vec3 velocity, impulse;
        float* velocity_components[] = {&velocity.x, &velocity.y, &velocity.z};
        float* impulse_components[] = {&impulse.x, &impulse.y, &impulse.z};
        for (uint32_t axis = 0u; axis < 3u; ++axis) {
            const uint32_t at = slot + axis * normal.group_normal_count;
            const NkRow row = LoadRowWarp(rows, at, lane);
            const SlimRow slim = MakeSlimRow(row, env_row, env_artic);
            const float jv = ComputeSlimRowVelocity<true>(slim, at, at,
                data.chain_jacobian, data.chain_jacobian_b, qdot, rows,
                data.body_linear_velocity, data.body_angular_velocity, points, p.max_dof, lane);
            const float scale = axis == 0u ? 1.0f / (1.0f + data.row_damping[at] * p.dt) : 1.0f;
            *impulse_components[axis] = data.lambda[at];
            *velocity_components[axis] = jv - row.rhs * p.dt * scale +
                row.compliance_alpha * scale * data.lambda[at];
        }
        if (lane != 0u) continue;
        const auto residual = constraint::EvaluateCoulombContactResidual(
            normal.contact_response, velocity, impulse, normal.mu, normal.friction_secondary);
        atomicAdd(&data.contact_solve_counts[env * constraint::kContactSolveCountSize], 1u);
        if (!residual.valid)
            atomicAdd(&data.contact_solve_counts[env * constraint::kContactSolveCountSize + 1u], 1u);
        const float metrics[] = {residual.normal_natural_velocity, residual.tangent_natural_velocity,
            residual.normal_velocity_violation, residual.normal_impulse_violation,
            residual.normal_complementarity, residual.friction_impulse_violation,
            residual.friction_power_violation, residual.friction_dissipation_work};
        static_assert(sizeof(metrics) / sizeof(float) == constraint::kContactSolveMetricCount);
        for (uint32_t metric = 0u; metric < constraint::kContactSolveMetricCount; ++metric) {
            const float value = residual.valid ? metrics[metric] : FLT_MAX;
            const unsigned long long packed =
                (static_cast<unsigned long long>(__float_as_uint(value)) << 32u) | (~slot);
            atomicMax(reinterpret_cast<unsigned long long*>(&data.contact_solve_metrics[
                env * constraint::kContactSolveMetricCount + metric]), packed);
        }
    }
}

Status OpSolveRowsBlockIsland(const ModelView& model, const DataView& data,
                              const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const SolveRowsBlockIslandParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    // Deferred positions and verification continue the dynamic schedule of the same step.
    if ((p->position_later != 0u || p->verify_idle != 0u) &&
        (p->family != kContactFamilyPairDriven || p->force_static_islands != 0u ||
         (p->position_later != 0u && p->pos_iters != 0u) ||
         (p->verify_idle != 0u && p->continue_impulses == 0u)))
        return Status::InvalidArgument;
    const uint64_t row_capacity = uint64_t{p->rows_per_env} * p->env_count;
    if (row_capacity > uint64_t{kOwnerEmpty} -
            uint64_t{kColorGridLimit} * kColorBlockSize)
        return Status::InvalidArgument;
    // A verifying pass adds its sweeps and phase times to those of the pass it continues.
    if (p->measure_vertex_audit != 0u && p->vertex_blocks.vertices > 0u &&
        (data.vbd_solve_audit == nullptr || cudaMemsetAsync(data.vbd_solve_audit, 0,
            size_t{p->vertex_blocks.vertices} * p->env_count * nk::kVbdSolveAuditColumnCount * sizeof(float),
            stream) != cudaSuccess)) return Status::Failed;
    if (data.vbd_velocity_sweep_count == nullptr ||
        (p->verify_idle == 0u &&
         cudaMemsetAsync(data.vbd_velocity_sweep_count, 0,
                         size_t{p->env_count} * sizeof(uint32_t), stream) != cudaSuccess))
        return Status::Failed;
    if (data.solver_color_counts == nullptr ||
        cudaMemsetAsync(data.solver_color_counts, 0,
                        size_t{p->env_count} * kSolverCountWords * sizeof(uint32_t),
                        stream) != cudaSuccess)
        return Status::Failed;
    if (data.solver_phase_time == nullptr ||
        (p->verify_idle == 0u &&
         cudaMemsetAsync(data.solver_phase_time, 0,
                         size_t{p->env_count} * kSolverPhases * sizeof(uint64_t),
                         stream) != cudaSuccess))
        return Status::Failed;
    if (p->measure_contact_residual != 0u) {
        if (data.contact_solve_metrics == nullptr || data.contact_solve_counts == nullptr)
            return Status::InvalidArgument;
        if (cudaMemsetAsync(data.contact_solve_metrics, 0,
                size_t{p->env_count} * constraint::kContactSolveMetricCount * sizeof(uint64_t), stream) != cudaSuccess ||
            cudaMemsetAsync(data.contact_solve_counts, 0,
                size_t{p->env_count} * constraint::kContactSolveCountSize * sizeof(uint32_t), stream) != cudaSuccess)
            return Status::Failed;
        if (p->vertex_blocks.dynamic_vertices > 0u &&
            (data.vbd_solve_metrics == nullptr || cudaMemsetAsync(data.vbd_solve_metrics, 0,
                size_t{p->env_count} * 2u * sizeof(uint64_t), stream) != cudaSuccess))
            return Status::Failed;
    }

    if (p->family == kContactFamilyUnionCsr ||
        p->family == kContactFamilyPairDriven) {
        if (p->total_islands == 0u) {
            return Status::Ok;
        }
        if (p->max_dof > kMaxArticulationDof) {
            return Status::Failed;  // shared qdot tile capacity (legacy cap).
        }
        const uint64_t error_bytes = SolverVelocityScratchBytes(
            p->total_body_count, p->total_particle_count, p->total_grid_count);
        if (error_bytes > p->workspace_bytes ||
            (error_bytes > 0u && data.solver_velocity_scratch == nullptr)) return Status::InvalidArgument;
        VelocityErrorView error;
        if (error_bytes > 0u) {
            // Verification keeps the compensation of the velocities it continues.
            if (p->verify_idle == 0u &&
                cudaMemsetAsync(data.solver_velocity_scratch, 0, error_bytes, stream) != cudaSuccess)
                return Status::Failed;
            error.body_linear = reinterpret_cast<math::Vec3*>(data.solver_velocity_scratch);
            error.body_angular = error.body_linear + p->total_body_count;
            error.particle = error.body_angular + p->total_body_count;
            error.grid = error.particle + p->total_particle_count;
        }
        const bool with_b_arm = (p->family == kContactFamilyPairDriven);
        const uint32_t artics_per_env =
            (p->articulation_count > 0u && p->env_count > 0u)
                ? (p->articulation_count / p->env_count) : 1u;
        // The dynamic CC pass runs for PairDriven; the validation hook forces the
        // cook-time static schedule (the byte-identity reference) instead.
        const bool run_dynamic = with_b_arm && (p->force_static_islands == 0u);
        if (!run_dynamic && p->vertex_blocks.colors != 0u) return Status::Unsupported;
        // Distinct dynamic entities bound the number of live islands.
        const uint64_t entity_bound = uint64_t{p->articulation_count} +
                                      p->total_body_count + p->total_particle_count;
        if (entity_bound > std::numeric_limits<uint32_t>::max()) return Status::InvalidArgument;
        const uint32_t max_island_bound = static_cast<uint32_t>(entity_bound);
        const uint32_t island_bound = run_dynamic ? max_island_bound : p->total_islands;
        if (island_bound == 0u) {
            return Status::Ok;
        }
        // qdot region: legacy kMaxArticulationDof reservation for UnionCsr (the
        // H1-golden footprint), the compact K-tile reservation for PairDriven .
        const uint64_t qdot_count = with_b_arm
            ? uint64_t{artics_per_env} * p->max_dof : kMaxArticulationDof;
        if (qdot_count > std::numeric_limits<uint32_t>::max()) return Status::InvalidArgument;
        const uint32_t qdot_floats = static_cast<uint32_t>(qdot_count);
        // Split-impulse position pass runs ONLY on the PairDriven path (pos_iters>0).
        const bool pos_pass = (p->pos_iters > 0u) && with_b_arm;
        // Rows a later pass will project go live now, so the velocity sweeps include them.
        const bool penetration_live = pos_pass || p->position_later != 0u;
        // Static schedules solve each island on one block with its optional row cache.
        const size_t shared_bytes = IslandSharedBytes(p->rows_per_env, p->max_dof,
                                                      qdot_floats, !with_b_arm, pos_pass);
        const uint32_t island_block_size =
            with_b_arm ? kPairDrivenIslandBlockSize : kUnionIslandBlockSize;
        const auto island_kernel = with_b_arm ? SolveRowsBlockIslandKernel<true>
                                              : SolveRowsBlockIslandKernel<false>;
        uint32_t grid_islands = 0u;
        if (!run_dynamic && ResidentGridSize(island_kernel, island_block_size, shared_bytes,
                                             island_bound, &grid_islands) != cudaSuccess) {
            return Status::Failed;
        }
        // Zero every global pseudo accumulator: a solve writes only the entries its rows reach.
        if (pos_pass) {
            if (p->total_grid_count > 0u && data.grid_pseudo_vel != nullptr) {
                const size_t bytes = static_cast<size_t>(p->total_grid_count) * sizeof(math::Vec3);
                if (cudaMemsetAsync(data.grid_pseudo_vel, 0, bytes, stream) != cudaSuccess)
                    return Status::Failed;
            }
            if (p->total_body_count > 0u && data.body_pseudo_linear_velocity != nullptr) {
                const size_t bbytes =
                    static_cast<size_t>(p->total_body_count) * sizeof(math::Vec3);
                if (cudaMemsetAsync(data.body_pseudo_linear_velocity, 0, bbytes,
                                    stream) != cudaSuccess ||
                    cudaMemsetAsync(data.body_pseudo_angular_velocity, 0, bbytes,
                                    stream) != cudaSuccess) {
                    return Status::Failed;
                }
            }
            if (p->total_particle_count > 0u && data.particle_pseudo_vel != nullptr) {
                const size_t pbytes =
                    static_cast<size_t>(p->total_particle_count) * sizeof(math::Vec3);
                if (cudaMemsetAsync(data.particle_pseudo_vel, 0, pbytes, stream) !=
                    cudaSuccess) {
                    return Status::Failed;
                }
            }
            const size_t total_link_count =
                static_cast<size_t>(p->base_link_count) * p->env_count;
            const size_t total_artic_dof =
                static_cast<size_t>(p->articulation_count) * p->max_dof;
            if (total_link_count > 0u && data.qdot_pseudo != nullptr) {
                if (cudaMemsetAsync(data.qdot_pseudo, 0,
                                    total_link_count * sizeof(float), stream) !=
                        cudaSuccess ||
                    cudaMemsetAsync(data.link_velocity_pseudo, 0,
                                    total_link_count * sizeof(Spatial6), stream) !=
                        cudaSuccess) {
                    return Status::Failed;
                }
            }
            if (total_artic_dof > 0u && data.qdot_pseudo_flat != nullptr) {
                if (cudaMemsetAsync(data.qdot_pseudo_flat, 0,
                                    total_artic_dof * sizeof(float), stream) !=
                        cudaSuccess) {
                    return Status::Failed;
                }
            }
        }
        IslandActivityView activity;
        // The scratch carries the inclusive live-row scan followed by the compacted live order.
        const uint32_t scratch_rows = p->rows_per_env * p->env_count;
        uint32_t* const live_scan = run_dynamic ? data.pd_solve_scratch : nullptr;
        uint32_t* const live_order = live_scan != nullptr ? live_scan + scratch_rows : nullptr;
        if (run_dynamic) {
            // Island roots keep their build flags in cc_parent; each solve resets the activity bit.
            const uint32_t total_rows = p->rows_per_env * p->env_count;
            activity = {data.island_root_sorted, data.cc_parent};
            ColorScratch scratch;
            (void)BindColorScratch(*p, data.solve_color_scratch, &scratch);
            if (scratch.control == nullptr) return Status::InvalidArgument;
            LivePrepareArgs prep;
            prep.lambda = data.lambda;
            prep.row_meff = static_cast<const float*>(data.row_meff);
            prep.row_damping = static_cast<const float*>(data.row_damping);
            prep.chain_jacobian = static_cast<const float*>(data.chain_jacobian);
            prep.chain_jacobian_b = static_cast<const float*>(data.chain_jacobian_b);
            prep.qdot_flat = data.qdot_flat;
            prep.body_linear = data.body_linear_velocity;
            prep.body_angular = data.body_angular_velocity;
            prep.row_order = data.island_rows;
            prep.cc_root = data.cc_root;
            prep.cc_artic_first = p->articulation_count > 0u ? data.cc_artic_first : nullptr;
            prep.row_penetration = penetration_live ? data.row_penetration : nullptr;
            prep.row_pseudo_lambda = penetration_live ? data.row_pseudo_lambda : nullptr;
            prep.total_rows = total_rows;
            prep.rows_per_env = p->rows_per_env;
            prep.artics_per_env = artics_per_env;
            prep.dof_stride = p->max_dof;
            prep.bodies_per_env = p->env_count != 0u ? p->total_body_count / p->env_count : 0u;
            prep.dt = p->dt;
            prep.pos_slop = p->pos_slop;
            prep.reuse_schedule = p->continue_impulses != 0u;
            prep.verify_idle = p->verify_idle != 0u;
            // Preparing is a chain of dependent loads per row, so it fills every resident slot.
            // Owner caches take the shared memory those resident blocks leave unused.
            size_t spare_shared = 0u;
            if (SpareSharedBytes(PrepareLiveColorsKernel, kColorBlockSize, &spare_shared) !=
                cudaSuccess)
                return Status::Failed;
            constexpr size_t kOwnerSlotBytes = size_t{kColorBlockSize} * sizeof(uint32_t);
            prep.owner_cache_slots =
                static_cast<uint32_t>(std::min<size_t>(spare_shared / kOwnerSlotBytes, 32u));
            const size_t owner_cache_bytes = prep.owner_cache_slots * kOwnerSlotBytes;
            uint32_t prepare_blocks = 0u;
            if (ResidentGridSize(PrepareLiveColorsKernel, kColorBlockSize, owner_cache_bytes,
                                 kColorGridLimit, &prepare_blocks) != cudaSuccess)
                return Status::Failed;
            if (p->verify_idle != 0u) {
                constexpr uint32_t verify_block_size = 128u;
                uint32_t verify_blocks = 0u;
                uint32_t* const violations = scratch.control + kControlIdleViolations;
                if (cudaMemsetAsync(violations, 0, sizeof(uint32_t), stream) != cudaSuccess ||
                    ResidentGridSize(CountViolatedIdleRowsKernel, verify_block_size, 0u,
                        (total_rows + verify_block_size / 32u - 1u) / (verify_block_size / 32u),
                        &verify_blocks) != cudaSuccess)
                    return Status::Failed;
                LaunchCuda(CountViolatedIdleRowsKernel, dim3(verify_blocks),
                    dim3(verify_block_size), 0u, stream,
                    reinterpret_cast<const NkRow*>(data.urows), data.lambda,
                    static_cast<const float*>(data.row_meff),
                    static_cast<const float*>(data.row_damping),
                    static_cast<const float*>(data.chain_jacobian),
                    static_cast<const float*>(data.chain_jacobian_b), data.qdot_flat,
                    data.body_linear_velocity, data.body_angular_velocity, PointMasses(data),
                    data.island_quads, data.island_count, activity, data.island_rows, live_scan,
                    violations, p->rows_per_env, artics_per_env, p->max_dof, p->dt,
                    p->vel_tolerance);
                if (cudaGetLastError() != cudaSuccess) return Status::Failed;
            }
            // Verification continues the same rows, so the hub masses it inherits still hold.
            if (scratch.bodies != 0u && p->verify_idle == 0u) {
                if (cudaMemsetAsync(scratch.hub_count, 0, sizeof(uint32_t) * scratch.bodies,
                                    stream) != cudaSuccess)
                    return Status::Failed;
                const auto blocks = [](uint64_t items) {
                    return static_cast<uint32_t>(std::min<uint64_t>((items + 255u) / 256u, 1024u));
                };
                if (total_rows != 0u)
                    LaunchCuda(CountHubRowsKernel, dim3(blocks(total_rows)), dim3(256u), 0u, stream,
                               reinterpret_cast<const NkRow*>(data.urows), total_rows,
                               static_cast<const float*>(data.body_inv_mass), scratch);
                LaunchCuda(MaskHubOwnersKernel, dim3(blocks(scratch.bodies)), dim3(256u), 0u,
                           stream, static_cast<const float*>(data.body_inv_mass), scratch);
                if (cudaGetLastError() != cudaSuccess) return Status::Failed;
            }
            if (LaunchCooperativeCuda(PrepareLiveColorsKernel, dim3(prepare_blocks),
                    dim3(kColorBlockSize), owner_cache_bytes, stream,
                    reinterpret_cast<const NkRow*>(data.urows),
                    PointMasses(data),
                    static_cast<const float*>(scratch.owner_inv_mass),
                    static_cast<const uint32_t*>(data.island_quads),
                    static_cast<const uint32_t*>(data.island_count), activity,
                    live_scan, live_order, scratch, prep) != cudaSuccess)
                return Status::Failed;
            const uint32_t scalar_blocks = max_island_bound < kScalarIslandGridBlocks
                ? max_island_bound : kScalarIslandGridBlocks;
            // Scalar islands sweep every row they hold, so verification only projects them.
            LaunchCuda(
                SolveRowsScalarIslandsKernel, dim3(scalar_blocks),
                dim3(kScalarIslandBlockSize), 0u, stream,
                reinterpret_cast<const NkRow*>(data.urows), data.lambda,
                static_cast<const float*>(data.row_meff),
                static_cast<const float*>(data.row_damping),
                data.body_linear_velocity, data.body_angular_velocity,
                static_cast<const float*>(data.body_inv_mass),
                static_cast<const math::SymmetricMat3*>(data.body_world_inv_inertia),
                PointMasses(data),
                data.island_quads, data.island_rows,
                pos_pass ? data.row_penetration : nullptr,
                pos_pass ? static_cast<float*>(data.row_pseudo_lambda) : nullptr,
                pos_pass ? data.body_pseudo_linear_velocity : nullptr,
                pos_pass ? data.body_pseudo_angular_velocity : nullptr,
                pos_pass ? data.particle_pseudo_vel : nullptr,
                pos_pass ? data.grid_pseudo_vel : nullptr,
                data.island_count, activity, p->rows_per_env, artics_per_env,
                p->verify_idle != 0u ? 0u : static_cast<uint32_t>(p->vel_iters),
                static_cast<uint32_t>(p->pos_iters),
                p->pos_beta, p->pos_slop, p->dt,
                p->baumgarte_max_velocity, p->continue_impulses == 0u, error);
            if (cudaGetLastError() != cudaSuccess) return Status::Failed;
            const Status colored = SolveColoredIslands(model, data, *p, activity, scratch,
                live_scan, live_order, error, artics_per_env, pos_pass, stream);
            if (colored != Status::Ok) return colored;
        } else {
            LaunchCuda(island_kernel, dim3(grid_islands),
                       dim3(island_block_size), static_cast<uint32_t>(shared_bytes),
                       stream,
                       reinterpret_cast<const NkRow*>(data.urows),
                       data.lambda,
                       static_cast<const float*>(data.chain_jacobian),
                       static_cast<const float*>(data.row_minv_jt),
                       with_b_arm ? static_cast<const float*>(data.chain_jacobian_b)
                                  : nullptr,
                       with_b_arm ? static_cast<const float*>(data.row_minv_jt_b)
                                  : nullptr,
                       static_cast<const float*>(data.row_meff),
                       static_cast<const float*>(data.row_damping),
                       data.qdot_flat,
                       reinterpret_cast<Spatial6*>(data.link_velocity),
                       data.qdot,
                       data.body_linear_velocity,
                       data.body_angular_velocity,
                       static_cast<const float*>(data.body_inv_mass),
                       static_cast<const math::SymmetricMat3*>(data.body_world_inv_inertia),
                       PointMasses(data),
                       static_cast<const uint32_t*>(model.island_row_offsets),
                       static_cast<const uint32_t*>(model.island_color_segments),
                       static_cast<const uint32_t*>(model.row_order),
                       static_cast<const uint32_t*>(model.dof_to_link),
                       static_cast<const uint32_t*>(model.dof_to_component),
                       data.pd_solve_scratch,
                       pos_pass ? data.row_penetration : nullptr,
                       pos_pass ? static_cast<float*>(data.row_pseudo_lambda) : nullptr,
                       pos_pass ? static_cast<float*>(data.qdot_pseudo) : nullptr,
                       pos_pass ? reinterpret_cast<Spatial6*>(data.link_velocity_pseudo) : nullptr,
                       pos_pass ? static_cast<float*>(data.qdot_pseudo_flat) : nullptr,
                       pos_pass ? data.body_pseudo_linear_velocity : nullptr,
                       pos_pass ? data.body_pseudo_angular_velocity : nullptr,
                       pos_pass ? data.particle_pseudo_vel : nullptr,
                       pos_pass ? data.grid_pseudo_vel : nullptr,
                       p->total_islands, p->rows_per_env, p->max_dof,
                       p->base_link_count, artics_per_env,
                       static_cast<uint32_t>(p->vel_iters),
                       static_cast<uint32_t>(p->pos_iters),
                       p->pos_beta, p->pos_slop, p->dt, p->vel_tolerance,
                       p->baumgarte_max_velocity, p->continue_impulses == 0u, error);
            if (cudaGetLastError() != cudaSuccess) return Status::Failed;
        }
        if (run_dynamic && pos_pass && data.row_penetration != nullptr) {
            constexpr uint32_t idle_block_size = 128u;
            const uint32_t total_rows = p->rows_per_env * p->env_count;
            uint32_t idle_blocks = 0u;
            if (ResidentGridSize(UpdateIdlePenetrationKernel, idle_block_size, 0u,
                    (total_rows + idle_block_size / 32u - 1u) / (idle_block_size / 32u),
                    &idle_blocks) != cudaSuccess) return Status::Failed;
            LaunchCuda(UpdateIdlePenetrationKernel, dim3(idle_blocks), dim3(idle_block_size), 0u,
                stream, reinterpret_cast<const NkRow*>(data.urows),
                static_cast<const float*>(data.chain_jacobian),
                static_cast<const float*>(data.chain_jacobian_b), data.qdot_flat,
                data.body_linear_velocity, data.body_angular_velocity,
                PointMasses(data),
                data.island_quads, data.island_count, activity, data.island_rows, live_scan,
                data.row_penetration, p->rows_per_env, artics_per_env, p->max_dof, p->dt);
            if (cudaGetLastError() != cudaSuccess) return Status::Failed;
        }
        if (with_b_arm && p->base_link_count > 0u &&
            p->joint_limit_rows_per_env >= p->base_link_count * 2u &&
            data.joint_limit_impulse != nullptr) {
            const uint32_t total_links = p->base_link_count * p->env_count;
            const uint32_t blocks =
                (total_links + kPairDrivenIslandBlockSize - 1u) /
                kPairDrivenIslandBlockSize;
            LaunchCuda(WriteJointLimitImpulseKernel, dim3(blocks),
                       dim3(kPairDrivenIslandBlockSize), 0u, stream,
                       data.lambda, total_links, p->env_count,
                       p->base_link_count, p->rows_per_env,
                       p->contact_rows_per_env, data.joint_limit_impulse);
        }
        if (p->measure_contact_residual != 0u && p->vertex_blocks.dynamic_vertices > 0u) {
            constexpr uint32_t block_size = 128u;
            const uint32_t bound =
                (p->vertex_blocks.dynamic_vertices * p->env_count - 1u) / (block_size / 32u) + 1u;
            uint32_t blocks = 0u;
            if (ResidentGridSize(MeasureVertexResidualKernel, block_size, 0u, bound, &blocks) != cudaSuccess)
                return Status::Failed;
            LaunchCuda(MeasureVertexResidualKernel, dim3(blocks), dim3(block_size), 0u, stream,
                       model, data, *p, error.particle);
        }
        if (p->measure_contact_residual != 0u && p->rows_per_env > 0u) {
            constexpr uint32_t block_size = 128u;
            const uint32_t bound = (p->rows_per_env * p->env_count - 1u) / (block_size / 32u) + 1u;
            uint32_t blocks = 0u;
            if (ResidentGridSize(MeasureContactResidualKernel, block_size, 0u, bound, &blocks) != cudaSuccess)
                return Status::Failed;
            LaunchCuda(MeasureContactResidualKernel, dim3(blocks), dim3(block_size), 0u, stream, data, *p);
        }
        return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
    }

    // An unconfigured contact family has no rows to solve.
    return Status::Ok;
}

} // namespace

uint64_t SolverVelocityScratchBytes(uint32_t body_count, uint32_t particle_count,
                                    uint32_t grid_count) {
    return (2ull * body_count + particle_count + grid_count) * sizeof(math::Vec3);
}

uint64_t SolveColorScratchWords(const SolveRowsBlockIslandParams& params) {
    if (params.family != kContactFamilyPairDriven || params.rows_per_env == 0u ||
        params.env_count == 0u) return 0u;
    return BindColorScratch(params, nullptr, nullptr);
}

void RegisterNkSolveRowsOps() {
    SetCudaOp(NkOp::SolveRowsBlockIsland, &OpSolveRowsBlockIsland);
}

} // namespace nuka::phi
