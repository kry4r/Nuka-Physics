// Independent islands retain ordered row updates and separate real/pseudo velocities.

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cub/block/block_scan.cuh>

#include "constraint/coulomb_contact.hpp"
#include <cstring>
#include <limits>

#include "math/cuda_vec_ops.cuh"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/launch_grid.cuh"
#include "phi/backend_cuda/ops/island_schedule.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/union_types.cuh"

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
        if (velocity != nullptr && inverse_mass != nullptr && inverse_mass[term.index] > 0.0f)
            AddVelocity(velocity[term.index], error.Point(term.kind, term.index),
                        term.jacobian * (inverse_mass[term.index] * delta));
    }
}

// Independent warps screen consecutive rows before the ordered updates.
constexpr uint32_t kUnionIslandBlockSize = 64u;
constexpr uint32_t kPairDrivenIslandBlockSize = 32u;
constexpr uint32_t kScalarIslandBlockSize = 32u;
constexpr uint32_t kScalarIslandGridBlocks = 64u;
// A colored row spreads at most two owners over each lane of its warp.
constexpr uint32_t kOwnerKeysPerRow = 64u;
constexpr uint32_t kOwnerEmpty = ~0u;
constexpr uint32_t kOwnerGroupKind = 7u;
// Live rows sharing no written state take one color and sweep together across the grid.
// Rows the palette cannot place, or with articulation owners, run in island order instead.
constexpr uint32_t kColorPalette = 128u;
constexpr uint32_t kColorWords = kColorPalette / 32u;
constexpr uint32_t kColorEpochs = 16u;
constexpr uint32_t kColorSlots = kColorPalette * kColorEpochs;
constexpr uint32_t kColorRounds = 256u;
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
constexpr uint32_t kControlWords = 16u;
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
__forceinline__ __device__ NkRow LoadRowWarp(const NkRow* rows, uint32_t slot, uint32_t lane) {
    static_assert(sizeof(NkRow) == 32u * sizeof(uint32_t));
    uint32_t word;
    memcpy(&word, reinterpret_cast<const unsigned char*>(rows + slot) + lane * sizeof(word), sizeof(word));
    NkRow result;
    #pragma unroll
    for (uint32_t i = 0u; i < sizeof(NkRow) / sizeof(word); ++i) {
        const uint32_t value = __shfl_sync(0xffffffffu, word, i);
        memcpy(reinterpret_cast<unsigned char*>(&result) + i * sizeof(word), &value, sizeof(value));
    }
    return result;
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

__device__ bool ContactFrictionActive(const NkRow& normal, float damping, float dt,
                                     float velocity, float impulse) {
    if (!(fmaxf(normal.mu, normal.friction_secondary) > 0.0f)) return false;
    const float scale = 1.0f / (1.0f + damping * dt);
    const float residual = normal.rhs * dt * scale - velocity - normal.compliance_alpha * scale * impulse;
    return constraint::ProjectedContactNormal(normal.contact_response.xx, residual, impulse) > 0.0f;
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
                    AddBlockVelocity(value, error.Point(a.kind, a.index),
                        a.jacobian * inverse_mass[a.index], b.jacobian * inverse_mass[a.index],
                        c.jacobian * inverse_mass[a.index], impulse);
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
                    AddBlockVelocity(value, error.Point(a.kind, a.index),
                        a.jacobian * inverse_mass[a.index], b.jacobian * inverse_mass[a.index],
                        c.jacobian * inverse_mass[a.index], impulse);
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

__device__ float ComputePreparedSideVelocity(
    const NkRowSide& side, uint32_t j_row, bool side_b, uint32_t env_artic_base,
    const float* J, const float* J_b, const float* qdot,
    const math::Vec3* body_linear, const math::Vec3* body_angular,
    PointMassView points, uint32_t dofs, uint32_t lane) {
    if (side.kind == kNkSideArtic) {
        const uint32_t tile = side.index - env_artic_base;
        const float* source = side_b && J_b != nullptr ? J_b : J;
        const float* jacobian = source + static_cast<size_t>(j_row) * dofs;
        const float* velocity = qdot + static_cast<size_t>(tile) * dofs;
        float result = 0.0f;
        for (uint32_t k = lane; k < dofs; k += warpSize)
            result += jacobian[k] * velocity[k];
        return WarpSum(result);
    }
    if (PointMassView::IsPointSide(side.kind))
        return points.RowVelocityWarp(side, lane);
    float result = 0.0f;
    if (side.kind == kNkSideRigid && lane == 0u)
        result = side.jlin.Dot(body_linear[side.index]) +
                 side.jang.Dot(body_angular[side.index]);
    return __shfl_sync(0xffffffffu, result, 0u);
}

// The three axis rows of a contact side share its state, so one pass reads each
// velocity once; each axis sums in the same order as ComputePreparedSideVelocity.
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

__device__ bool PreparedSidesDisjoint(
    const NkRowSide& a, const NkRowSide& b, PointMassView points) {
    if (a.kind == kNkSideStatic || b.kind == kNkSideStatic) return true;
    if (a.kind == kNkSideArtic || b.kind == kNkSideArtic ||
        a.kind == kNkSideRigid || b.kind == kNkSideRigid)
        return a.kind != b.kind || a.index != b.index;
    if (!PointMassView::IsPointSide(a.kind) || !PointMassView::IsPointSide(b.kind))
        return true;
    uint32_t i = 0u;
    uint32_t j = 0u;
    const uint32_t a_count = points.Count(a);
    const uint32_t b_count = points.Count(b);
    while (i < a_count && j < b_count) {
        const auto first = points.At(a, i);
        const auto second = points.At(b, j);
        const uint64_t first_key = (uint64_t{first.kind} << 32u) | first.index;
        const uint64_t second_key = (uint64_t{second.kind} << 32u) | second.index;
        if (first_key == second_key) return false;
        if (first_key <= second_key) ++i;
        if (second_key <= first_key) ++j;
    }
    return true;
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

// Views of solve_color_scratch. Dense owners index bodies, particles, grid nodes, then row groups.
struct ColorScratch {
    uint32_t* used = nullptr;          // kColorWords per owner: colors taken this epoch
    uint32_t* tent = nullptr;          // kColorWords per owner: this round's picks
    uint32_t* dup = nullptr;           // kColorWords per owner: picks seen twice
    uint32_t* claim = nullptr;         // per owner: highest priority among duplicate picks
    uint32_t* pos_color = nullptr;     // per live position
    uint32_t* pos_tent = nullptr;      // per live position: (round << 8) | color
    uint32_t* color_rows = nullptr;
    uint32_t* color_start = nullptr;   // bounds of the non-empty colors
    uint32_t* color_cursor = nullptr;  // kColorSlots
    uint32_t* chain_rows = nullptr;
    uint32_t* chain_excl = nullptr;    // per live position, then the total
    uint32_t* chain_islands = nullptr;
    uint32_t* block_count = nullptr;   // kColorGridLimit
    uint32_t* control = nullptr;       // kControlWords
    float* qdot_error = nullptr;       // per articulation DOF
    uint32_t bodies = 0u;
    uint32_t particles = 0u;
    uint32_t grid = 0u;
    uint32_t rows = 0u;
    uint32_t artic_dofs = 0u;
};

// Lays the scratch out from base; a null base only counts the words.
uint64_t BindColorScratch(const SolveRowsBlockIslandParams& p, uint32_t* base, ColorScratch* out) {
    const uint64_t rows = uint64_t{p.rows_per_env} * p.env_count;
    const uint64_t owners = uint64_t{p.total_body_count} + p.total_particle_count +
                            p.total_grid_count + rows;
    uint64_t words = 0u;
    auto take = [&](uint64_t count) {
        uint32_t* const at = base != nullptr ? base + words : nullptr;
        words += count;
        return at;
    };
    ColorScratch s;
    s.used = take(kColorWords * owners);
    s.tent = take(kColorWords * owners);
    s.dup = take(kColorWords * owners);
    s.claim = take(owners);
    s.pos_color = take(rows);
    s.pos_tent = take(rows);
    s.color_rows = take(rows);
    s.color_start = take(kColorSlots + 1u);
    s.color_cursor = take(kColorSlots);
    s.chain_rows = take(rows);
    s.chain_excl = take(rows + 1u);
    s.chain_islands = take(rows);
    s.block_count = take(kColorGridLimit);
    s.control = take(kControlWords);
    s.qdot_error = reinterpret_cast<float*>(take(uint64_t{p.articulation_count} * p.max_dof));
    s.bodies = p.total_body_count;
    s.particles = p.total_particle_count;
    s.grid = p.total_grid_count;
    s.rows = static_cast<uint32_t>(rows);
    s.artic_dofs = p.articulation_count * p.max_dof;
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
// an owner outside the dense ranges (an articulation tile) is reported unpacked.
struct LaneOwners {
    uint32_t first = kOwnerEmpty;
    uint32_t second = kOwnerEmpty;
    bool packed = true;
};

__device__ LaneOwners LoadLaneOwners(const NkRow& row, PointMassView points,
                                     const float* body_inv_mass, const ColorScratch& s,
                                     uint32_t lane) {
    const uint32_t a_owners = SideOwnerCount(row.a, points, body_inv_mass);
    const uint32_t owners = 1u + a_owners + SideOwnerCount(row.b, points, body_inv_mass);
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
            AddBlockVelocity(value, error.Point(a.kind, a.index),
                a.jacobian * inverse_mass[a.index], b.jacobian * inverse_mass[a.index],
                c.jacobian * inverse_mass[a.index], impulse);
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

__device__ void SolvePreparedContactBlock(
    uint32_t gslot, uint32_t env_artic_base, uint32_t j_row,
    uint32_t warp, uint32_t lane, const NkRow* rows,
    float* lambda, const float* row_damping, const float* J, const float* minv_j,
    const float* J_b, const float* minv_j_b, float* qdot,
    math::Vec3* body_linear, math::Vec3* body_angular,
    const float* body_inv_mass, const math::SymmetricMat3* body_inv_inertia,
    PointMassView points, uint32_t dofs, float dt, float tangent_response,
    float vel_tolerance, VelocityErrorView error,
    float* side_velocity, float* delta, uint32_t* friction_active,
    uint32_t* changed, uint32_t* significant) {
    const NkRow& normal = rows[0];
    const uint32_t count = normal.group_normal_count;
    const uint32_t point = (gslot - normal.group_first) % count;
    const uint32_t normal_slot = normal.group_first + point;
    const uint32_t tangent_first_slot = normal.group_first + count + point;
    const uint32_t tangent_second_slot = normal.group_first + 2u * count + point;
    const uint32_t warps = blockDim.x / warpSize;
    for (uint32_t component = warp; component < 6u; component += warps) {
        const uint32_t axis = component / 2u;
        const uint32_t side = component & 1u;
        const NkRowSide& descriptor = side == 0u ? rows[axis].a : rows[axis].b;
        const float velocity = ComputePreparedSideVelocity(descriptor, j_row + axis, side != 0u,
            env_artic_base, J, J_b, qdot, body_linear, body_angular, points, dofs, lane);
        if (lane == 0u) side_velocity[2u * axis + side] = velocity;
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        const PreparedContactStep step = ContactBlockStep<true>(rows[0], rows[1], rows[2],
            {lambda[normal_slot], lambda[tangent_first_slot], lambda[tangent_second_slot]},
            row_damping[normal_slot],
            {side_velocity[0] + side_velocity[1], side_velocity[2] + side_velocity[3],
             side_velocity[4] + side_velocity[5]}, dt, tangent_response, vel_tolerance);
        lambda[normal_slot] = step.impulse.x;
        lambda[tangent_first_slot] = step.impulse.y;
        lambda[tangent_second_slot] = step.impulse.z;
        delta[0] = step.delta.x;
        delta[1] = step.delta.y;
        delta[2] = step.delta.z;
        *changed = step.changed;
        *significant = step.significant;
        *friction_active = PreparedSidesDisjoint(rows[0].a, rows[0].b, points) ? 1u : 0u;
    }
    __syncthreads();
    if (*changed != 0u) {
        const math::Vec3 impulse{delta[0], delta[1], delta[2]};
        if (*friction_active != 0u) {
            for (uint32_t side = warp; side < 2u; side += warps)
                ApplyPreparedContactSide(rows, side, j_row, 1u, env_artic_base, lane, impulse,
                    qdot, minv_j, minv_j_b, body_linear, body_angular, body_inv_mass,
                    body_inv_inertia, points, dofs, error);
        } else if (warp == 0u) {
            ApplyPreparedContactSide(rows, 0u, j_row, 1u, env_artic_base, lane, impulse,
                qdot, minv_j, minv_j_b, body_linear, body_angular, body_inv_mass,
                body_inv_inertia, points, dofs, error);
            ApplyPreparedContactSide(rows, 1u, j_row, 1u, env_artic_base, lane, impulse,
                qdot, minv_j, minv_j_b, body_linear, body_angular, body_inv_mass,
                body_inv_inertia, points, dofs, error);
        }
    }
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
    AddVelocity(value, nullptr, term.jacobian[0] * (term.inverse_mass * delta));
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
        term.jacobian[0] * term.inverse_mass, term.jacobian[1] * term.inverse_mass,
        term.jacobian[2] * term.inverse_mass, impulse);
    velocity[term.index] = value;
    if (compensation != nullptr) *compensation = residue;
}

// One warp solves a prepared contact whose written state no concurrent row touches.
// Tangent rows, term ranges and old impulses load together; the result reports significance.
__device__ bool SolvePreparedContactBlockWarp(
    uint32_t gslot, uint32_t env_artic_base, uint32_t lane, const NkRow* urows, NkRow* rows,
    float* lambda, const float* row_damping, const float* J, const float* minv_j,
    const float* J_b, const float* minv_j_b, float* qdot,
    math::Vec3* body_linear, math::Vec3* body_angular,
    const float* body_inv_mass, const math::SymmetricMat3* body_inv_inertia,
    PointMassView points, uint32_t dofs, float dt, float vel_tolerance,
    VelocityErrorView error) {
    const uint32_t count = rows[0].group_normal_count;
    const uint32_t point = (gslot - rows[0].group_first) % count;
    const uint32_t normal_slot = rows[0].group_first + point;
    const uint32_t tangent_first_slot = normal_slot + count;
    const uint32_t tangent_second_slot = normal_slot + 2u * count;
    const NkRowSide side_a = rows[0].a;
    const bool cached = PointMassView::IsPointSide(side_a.kind);
    nk::PointEndpointRange range{0u, 1u};
    if (cached && side_a.kind == nk::kNkSidePointEndpoint) range = points.ranges[side_a.index];
    const math::Vec3 old{lambda[normal_slot], lambda[tangent_first_slot],
                         lambda[tangent_second_slot]};
    const float damping = row_damping[normal_slot];
    uint32_t first_word;
    uint32_t second_word;
    memcpy(&first_word, reinterpret_cast<const unsigned char*>(urows + tangent_first_slot) +
                        lane * sizeof(first_word), sizeof(first_word));
    memcpy(&second_word, reinterpret_cast<const unsigned char*>(urows + tangent_second_slot) +
                         lane * sizeof(second_word), sizeof(second_word));
    memcpy(reinterpret_cast<unsigned char*>(rows + 1) + lane * sizeof(first_word), &first_word,
           sizeof(first_word));
    memcpy(reinterpret_cast<unsigned char*>(rows + 2) + lane * sizeof(second_word), &second_word,
           sizeof(second_word));
    __syncwarp();
    const NkRowSide sides_a[3] = {rows[0].a, rows[1].a, rows[2].a};
    PointLaneTerm term;
    const bool fits = cached && range.count <= warpSize;
    float point_velocity[3];
    if (fits) LoadPointSideVelocities(sides_a, range, points, error, lane, term, point_velocity);
    const math::Vec3 a = fits
        ? math::Vec3{point_velocity[0], point_velocity[1], point_velocity[2]}
        : ComputePreparedSideVelocities(rows, 0u, gslot, count, env_artic_base, J, J_b, qdot,
              body_linear, body_angular, points, dofs, lane);
    const math::Vec3 b = ComputePreparedSideVelocities(rows, 1u, gslot, count, env_artic_base,
        J, J_b, qdot, body_linear, body_angular, points, dofs, lane);
    const float side_velocity[6] = {a.x, b.x, a.y, b.y, a.z, b.z};
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
        } else {
            ApplyPreparedContactSide(rows, 0u, gslot, count, env_artic_base, lane, step.delta,
                qdot, minv_j, minv_j_b, body_linear, body_angular, body_inv_mass,
                body_inv_inertia, points, dofs, error);
        }
        ApplyPreparedContactSide(rows, 1u, gslot, count, env_artic_base, lane, step.delta,
            qdot, minv_j, minv_j_b, body_linear, body_angular, body_inv_mass,
            body_inv_inertia, points, dofs, error);
    }
    return step.significant;
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
    if (lambda[slot] != 0.0f || (flags & nk::nk_row_flags::kFriction)) return true;
    const NkRow row = prepared != nullptr ? *prepared : LoadRowWarp(urows, slot, lane);
    if ((row.flags & nk::nk_row_flags::kBlockNormal) &&
        (lambda[slot + row.group_normal_count] != 0.0f ||
         lambda[slot + 2u * row.group_normal_count] != 0.0f)) return true;
    const SlimRow sr = MakeSlimRow(row, env_row_base, env_artic_base);
    const float jv = ComputeSlimRowVelocity<true>(sr, slot, slot,
        chain_jacobian, chain_jacobian_b, qdot, urows, body_lin_vel,
        body_ang_vel, point_masses, dof_stride, lane, &row);
    const float old_impulse = lambda[slot];
    const float damping_scale = 1.0f / (1.0f + row_damping[slot] * dt);
    const float residual = row.rhs * dt * damping_scale - jv -
                           row.compliance_alpha * damping_scale * old_impulse;
    if (row.flags & nk::nk_row_flags::kBlockNormal)
        return constraint::ProjectedContactNormal(row.contact_response.xx, residual, old_impulse) != 0.0f;
    const float effective_mass = row_meff[slot];
    const float implicit_mass = effective_mass /
        (1.0f - effective_mass * row.compliance_alpha * (1.0f - damping_scale));
    return fminf(fmaxf(old_impulse + implicit_mass * residual, row.lower), row.upper) != 0.0f;
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
                                  float vel_tolerance = 0.0f) {
    const NkRow row = prepared != nullptr ? prepared[0] : LoadRowWarp(urows, gslot, wlane);
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
            ApplySlimContactImpulse(sr, sr_block, sr_block_second,
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
        ApplySlimImpulse(sr, gslot, j_row, wlane, delta, env_artic_base, qdot_sh,
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
                                     const NkRow* prepared = nullptr) {
    const SlimRow sr = MakeSlimRow(prepared != nullptr ? *prepared : LoadRowWarp(urows, gslot, wlane),
                                   env_row_base, env_artic_base);
    const uint32_t flags = sr.flags;
    if (!(flags & nk::nk_row_flags::kActive) ||
        (flags & (nk::nk_row_flags::kFriction | nk::nk_row_flags::kBlockTangent |
                  nk::nk_row_flags::kVelocityOnly))) {
        return false;
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
        SlimRow rest = sr;
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

__device__ void ApplyDynamicImpulseScalar(
    uint32_t gslot, float delta, const NkRow* __restrict__ urows,
    math::Vec3* __restrict__ body_lin_vel,
    math::Vec3* __restrict__ body_ang_vel,
    const float* __restrict__ body_inv_mass,
    const math::SymmetricMat3* __restrict__ body_world_inv_inertia,
    PointMassView point_masses, VelocityErrorView error = {}) {
    if (delta == 0.0f) return;
    const NkRow row = urows[gslot];
    for (int side = 0; side < 2; ++side) {
        const NkRowSide& sd = side == 0 ? row.a : row.b;
        if (PointMassView::IsPointSide(sd.kind)) {
            ApplyPointImpulse(sd, delta, point_masses, error);
        } else if (sd.kind == kNkSideRigid) {
            const float im = body_inv_mass[sd.index];
            if (im > 0.0f) {
                const math::Vec3 angular_response = body_world_inv_inertia[sd.index].Multiply(sd.jang);
                math::Vec3& v = body_lin_vel[sd.index];
                math::Vec3& w = body_ang_vel[sd.index];
                AddVelocity(v, error.Linear(sd.index), sd.jlin * (im * delta));
                AddVelocity(w, error.Angular(sd.index), angular_response * delta);
            }
        }
    }
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
    bool apply_cached_impulse, VelocityErrorView error) {
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

    const float jv = apply_cached_impulse ? 0.0f : ComputeSlimRowVelocity(
        sr, gslot, gslot, nullptr, nullptr, nullptr, urows, body_lin_vel,
        body_ang_vel, point_masses, 0u);
    float block_jv_tangent1 = 0.0f;
    float block_jv_tangent2 = 0.0f;
    if (block_row && !apply_cached_impulse &&
        ContactFrictionActive(urows[gslot], row_damping[gslot], dt, jv, block_old_normal)) {
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

    const float effective_mass = row_meff[gslot];
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
        const auto projected = constraint::ProjectedCoulombStep(normal.contact_response,
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
                                  body_world_inv_inertia, point_masses, error);
        ApplyDynamicImpulseScalar(block_tangent1_slot, block_delta_tangent1, urows,
                                  body_lin_vel, body_ang_vel, body_inv_mass,
                                  body_world_inv_inertia, point_masses, error);
        ApplyDynamicImpulseScalar(block_tangent2_slot, block_delta_tangent2, urows,
                                  body_lin_vel, body_ang_vel, body_inv_mass,
                                  body_world_inv_inertia, point_masses, error);
    } else {
        ApplyDynamicImpulseScalar(gslot, delta, urows, body_lin_vel, body_ang_vel,
                                  body_inv_mass, body_world_inv_inertia,
                                  point_masses, error);
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
    float beta, float slop, float dt, float baumgarte_max_velocity) {
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
    const float effective_mass = row_meff[gslot];
    const float old_imp = row_pseudo_lambda[gslot];
    const float new_imp = fmaxf(old_imp + effective_mass * (bias - jv), 0.0f);
    row_pseudo_lambda[gslot] = new_imp;
    const float d = new_imp - old_imp;
    const float delta = d;
    if (delta == 0.0f) return;

    ApplyDynamicImpulseScalar(gslot, delta, urows, body_pseudo_lin, body_pseudo_ang,
                              body_inv_mass, body_world_inv_inertia,
                              point_masses);
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
    // Interleave live islands across blocks; each island retains one owner and row order.
    const uint64_t stride = uint64_t{gridDim.x} * blockDim.x;
    for (uint64_t cursor = blockIdx.x + uint64_t{threadIdx.x} * gridDim.x;
         cursor < live_islands; cursor += stride) {
        const uint32_t island = static_cast<uint32_t>(cursor);
        const IslandRecord rec =
            reinterpret_cast<const IslandRecord*>(islands)[island];
        if (rec.flags & kIslandWarpWork) continue;
        if (!activity.Active(rec)) continue;
        const uint32_t env_row_base = rec.env * rows_per_env;
        const uint32_t env_artic_base =
            rec.env * (artics_per_env == 0u ? 1u : artics_per_env);
        if (apply_cached_impulses) {
            for (uint32_t r = 0u; r < rec.seg_cnt; ++r) {
                SolveDynamicRowScalar(row_order[rec.seg_off + r], env_row_base,
                                      env_artic_base, urows, lambda, row_meff,
                                      row_damping,
                                      body_lin_vel, body_ang_vel, body_inv_mass,
                                      body_world_inv_inertia, point_masses, dt, true, error);
            }
        }
        for (uint32_t it = 0u; it < vel_iters; ++it) {
            for (uint32_t r = 0u; r < rec.seg_cnt; ++r) {
                SolveDynamicRowScalar(row_order[rec.seg_off + r], env_row_base,
                                      env_artic_base, urows, lambda, row_meff,
                                      row_damping,
                                      body_lin_vel, body_ang_vel, body_inv_mass,
                                      body_world_inv_inertia, point_masses, dt, false, error);
            }
        }
        if (pos_iters == 0u) continue;
        for (uint32_t r = 0u; r < rec.seg_cnt; ++r) {
            const uint32_t slot = row_order[rec.seg_off + r];
            row_pseudo_lambda[slot] = 0.0f;
            UpdateSpeculativePenetration<false>(slot, env_row_base, env_artic_base,
                row_penetration, urows, nullptr, nullptr, nullptr, body_lin_vel, body_ang_vel,
                point_masses, 0u, dt);
        }
        for (uint32_t it = 0u; it < pos_iters; ++it) {
            for (uint32_t r = 0u; r < rec.seg_cnt; ++r) {
                SolvePositionRowScalar(
                    row_order[rec.seg_off + r], env_row_base, env_artic_base,
                    row_meff, row_penetration, row_pseudo_lambda, urows,
                    body_pseudo_lin, body_pseudo_ang, body_inv_mass,
                    body_world_inv_inertia, point_masses.Pseudo(particle_pseudo_vel, grid_pseudo_vel),
                    pos_beta, pos_slop, dt, baumgarte_max_velocity);
            }
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
                    const bool changed = SolveUnionRowWarp(
                        gslot - env_row_base, gslot, env_row_base, env_artic_base,
                        gslot - env_row_base, wlane, slim_sh, lambda_sh, meff_sh,
                        damping_sh, lambda, row_meff, row_damping, J_sh, w_sh,
                        nullptr, nullptr, qdot_sh, urows, body_lin_vel, body_ang_vel,
                        body_inv_mass, body_world_inv_inertia, point_masses,
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

// Each warp walks 32-wide stripes of live positions and visits those take accepts in order.
template <typename Take, typename Visit>
__device__ void ForEachLivePosition(uint32_t live, Take&& take, Visit&& visit) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t base = (blockIdx.x * blockDim.x + threadIdx.x) & ~31u; base < live;
         base += stride) {
        uint32_t mask = __ballot_sync(0xffffffffu, base + lane < live && take(base + lane));
        while (mask != 0u) {
            const uint32_t bit = static_cast<uint32_t>(__ffs(static_cast<int>(mask))) - 1u;
            mask &= mask - 1u;
            visit(base + bit);
        }
    }
}

__device__ inline LaneOwners LiveRowOwners(const NkRow* urows, uint32_t slot, PointMassView points,
                                           const float* body_inv_mass, const ColorScratch& s,
                                           uint32_t lane) {
    return LoadLaneOwners(LoadRowWarp(urows, slot, lane), points, body_inv_mass, s, lane);
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
    const float* row_penetration = nullptr;
    float* row_pseudo_lambda = nullptr;
    uint32_t total_rows = 0u;
    uint32_t rows_per_env = 0u;
    uint32_t artics_per_env = 0u;
    uint32_t dof_stride = 0u;
    float dt = 0.0f;
    float pos_slop = 0.0f;
};

__device__ __noinline__ void MarkLiveRows(const NkRow* urows, PointMassView points,
                                          IslandActivityView activity, uint32_t* live_scan,
                                          ColorScratch s, LivePrepareArgs prep) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp = thread >> 5u;
    const uint32_t warp_stride = gridDim.x * (blockDim.x >> 5u);
    uint32_t valid_end = 0u;
    for (uint32_t cursor = warp; cursor < prep.total_rows; cursor += warp_stride) {
        const uint32_t root = activity.sorted_roots[cursor];
        if (root == ~0u) break;
        valid_end = cursor + 1u;
        const uint32_t slot = prep.row_order[cursor];
        if (prep.row_pseudo_lambda != nullptr && lane == 0u)
            prep.row_pseudo_lambda[slot] = 0.0f;
        bool active = false;
        if (prep.row_penetration != nullptr &&
            !(urows[slot].flags & nk::nk_row_flags::kVelocityOnly))
            active |= prep.row_penetration[slot] > prep.pos_slop;
        if (!active) {
            const uint32_t env = slot / prep.rows_per_env;
            const uint32_t env_artic_base = env * prep.artics_per_env;
            const float* qdot = prep.qdot_flat != nullptr
                ? prep.qdot_flat + static_cast<size_t>(env_artic_base) * prep.dof_stride : nullptr;
            active = RowNeedsVelocitySolveWarp(urows, slot, env * prep.rows_per_env,
                env_artic_base, prep.lambda, prep.row_meff, prep.row_damping,
                prep.chain_jacobian, prep.chain_jacobian_b, qdot, prep.body_linear,
                prep.body_angular, points, prep.dof_stride, prep.dt, lane);
        }
        const bool warp_work = (activity.needs_solve[root] & kIslandWarpWork) != 0u;
        if (lane == 0u) live_scan[cursor] = active && warp_work ? 1u : 0u;
        if (active && lane == 0u) atomicOr(&activity.needs_solve[root], kIslandNeedsSolve);
    }
    if (lane == 0u && valid_end != 0u)
        atomicMax(s.control + kControlValidRows, valid_end);
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
    const auto grid = cooperative_groups::this_grid();
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t threads = gridDim.x * blockDim.x;
    if (thread == 0u) s.control[kControlValidRows] = 0u;
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
        const uint32_t flag = idx < valid_rows &&
            activity.sorted_roots[idx] != ~0u && live_scan[idx] != 0u ? 1u : 0u;
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
    }
    grid.sync();
    const uint32_t row_base = s.block_count[blockIdx.x];
    for (uint32_t idx = begin_row + threadIdx.x; idx < end_row; idx += kColorBlockSize) {
        const uint32_t prefix = row_base + live_scan[idx];
        live_scan[idx] = prefix;
        if (s.pos_color[idx] != 0u) live_order[prefix - 1u] = prep.row_order[idx];
    }
    grid.sync();

    const uint32_t live = s.control[kControlLive];
    for (uint32_t i = thread; i < kColorSlots; i += threads) s.color_cursor[i] = 0u;
    for (uint32_t i = thread; i < s.artic_dofs; i += threads) s.qdot_error[i] = 0.0f;
    if (thread > kControlLive && thread < kControlWords) s.control[thread] = 0u;
    ForEachLivePosition(live, [](uint32_t) { return true; }, [&](uint32_t position) {
        const LaneOwners owners =
            LiveRowOwners(urows, live_order[position], points, body_inv_mass, s, lane);
        if (lane == 0u) {
            s.pos_color[position] = owners.packed ? kColorPending : kColorChain;
            s.pos_tent[position] = kOwnerEmpty;
        }
    });
    grid.sync();

    const auto tentative = [&](uint32_t position) { return s.pos_tent[position] != kOwnerEmpty; };
    uint32_t round = 0u;
    for (uint32_t epoch = 0u; epoch < kColorEpochs; ++epoch) {
        bool pending = true;
        for (uint32_t step = 0u; pending && step < kColorRounds; ++step, ++round) {
            uint32_t* const pending_flag = s.control + kControlPending + round % 3u;
            if (thread == 0u) s.control[kControlPending + (round + 1u) % 3u] = 0u;
            bool overflow = false;
            ForEachLivePosition(live,
                [&](uint32_t position) { return s.pos_color[position] == kColorPending; },
                [&](uint32_t position) {
                    const uint32_t slot = live_order[position];
                    const LaneOwners owners =
                        LiveRowOwners(urows, slot, points, body_inv_mass, s, lane);
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
                        taken, MixSlot(slot ^ (round * 0x9e3779b9u)) % kColorPalette);
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

            ForEachLivePosition(live, tentative, [&](uint32_t position) {
                const uint32_t slot = live_order[position];
                const LaneOwners owners = LiveRowOwners(urows, slot, points, body_inv_mass, s, lane);
                const uint32_t color = s.pos_tent[position];
                const uint32_t priority = MixSlot(slot);
                VisitLaneOwners(owners, [&](uint32_t owner) {
                    if (s.dup[size_t{owner} * kColorWords + color / 32u] & (1u << (color % 32u)))
                        atomicMax(s.claim + owner, priority);
                });
            });
            grid.sync();

            bool left = false;
            ForEachLivePosition(live, tentative, [&](uint32_t position) {
                const uint32_t slot = live_order[position];
                const LaneOwners owners = LiveRowOwners(urows, slot, points, body_inv_mass, s, lane);
                const uint32_t color = s.pos_tent[position];
                const uint32_t priority = MixSlot(slot);
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
            ForEachLivePosition(live, tentative, [&](uint32_t position) {
                const LaneOwners owners =
                    LiveRowOwners(urows, live_order[position], points, body_inv_mass, s, lane);
                const uint32_t color = s.pos_tent[position];
                VisitLaneOwners(owners, [&](uint32_t owner) {
                    const size_t at = size_t{owner} * kColorWords + color / 32u;
                    s.tent[at] = 0u;
                    s.dup[at] = 0u;
                    s.claim[owner] = 0u;
                });
                __syncwarp();
                if (lane == 0u) s.pos_tent[position] = kOwnerEmpty;
            });
            grid.sync();
            pending = LoadControl(pending_flag) != 0u;
        }
        const bool overflow = LoadControl(s.control + kControlOverflow + epoch % 2u) != 0u;
        const bool last = !overflow || pending || epoch + 1u == kColorEpochs;
        ForEachLivePosition(live,
            [&](uint32_t position) { return s.pos_color[position] / kColorPalette == epoch; },
            [&](uint32_t position) {
                const LaneOwners owners =
                    LiveRowOwners(urows, live_order[position], points, body_inv_mass, s, lane);
                VisitLaneOwners(owners, [&](uint32_t owner) {
                    #pragma unroll
                    for (uint32_t w = 0u; w < kColorWords; ++w)
                        s.used[size_t{owner} * kColorWords + w] = 0u;
                });
            });
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
        if (color < kColorSlots) atomicAdd(histogram + color, 1u);
        else ++chained;
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
            s.color_start[color_index++] = row_start;
            s.color_cursor[threadIdx.x * per_thread + j] = row_start;
            row_start += counts[j];
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

    for (uint32_t position = thread; position < live; position += threads) {
        const uint32_t color = s.pos_color[position];
        if (color < kColorSlots)
            s.color_rows[atomicAdd(s.color_cursor + color, 1u)] = live_order[position];
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
    float* qdot_flat = nullptr;
    Spatial6* link_velocity = nullptr;
    float* qdot = nullptr;
    math::Vec3* body_linear = nullptr;
    math::Vec3* body_angular = nullptr;
    const float* body_inv_mass = nullptr;
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
};

enum class ChainSweep : uint32_t { WarmStart, Velocity, Position };

// Each color sweeps across the grid behind one barrier. Chain rows then run per island in live
// order on one block, with the island's articulation tiles staged in shared memory.
__global__ void __launch_bounds__(kColorBlockSize) SolveColoredRowsKernel(ColoredSolveArgs a,
                                                                          ColorScratch s) {
    const auto grid = cooperative_groups::this_grid();
    extern __shared__ __align__(16) unsigned char colored_shared[];
    __shared__ __align__(16) unsigned char row_storage[3u * (kColorBlockSize / 32u) * sizeof(NkRow)];
    __shared__ uint32_t batch_needed, row_changed, row_significant, friction_active;
    __shared__ float contact_side_velocity[6], contact_delta[3];
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = lane >> 5u;
    const uint32_t wlane = lane & 31u;
    const uint32_t nwarps = blockDim.x >> 5u;
    const uint32_t dofs = a.dof_stride;
    const uint32_t k_tiles = a.artics_per_env == 0u ? 1u : a.artics_per_env;
    const uint32_t tile_floats = k_tiles * dofs;
    float* const tile_sh = reinterpret_cast<float*>(colored_shared);
    float* const tile_error_sh = tile_sh + tile_floats;
    float* const staged_j = tile_error_sh + tile_floats;
    const size_t staged_size = size_t{3u} * nwarps * dofs;
    float* const staged_tangent_response = staged_j + 4u * staged_size;
    NkRow* const staged_rows = reinterpret_cast<NkRow*>(row_storage);
    // A color fills one warp of every block before a second warp of any block.
    const uint32_t color_warp = warp * gridDim.x + blockIdx.x;
    const uint32_t color_warps = gridDim.x * nwarps;
    const uint32_t colors = s.control[kControlColors];
    const uint32_t chain_islands = s.control[kControlChainIslands];
    const uint32_t live = s.control[kControlLive];
    const bool pos_pass = a.pos_iters > 0u && a.row_penetration != nullptr;
    const PointMassView pseudo_points = a.points.Pseudo(a.particle_pseudo, a.grid_pseudo);
    VelocityErrorView chain_error = a.error;
    chain_error.qdot = tile_error_sh;
    bool changed = false;

    auto color_sweep = [&](bool stage, auto&& solve_row) {
        NkRow* const staged = staged_rows + 3u * warp;
        for (uint32_t color = 0u; color < colors; ++color) {
            const uint32_t end = s.color_start[color + 1u];
            for (uint32_t i = s.color_start[color] + color_warp; i < end; i += color_warps) {
                const uint32_t slot = s.color_rows[i];
                if (stage) StageRowWarp(a.urows, slot, staged, wlane);
                solve_row(slot, slot / a.rows_per_env, staged);
            }
            grid.sync();
        }
    };

    auto chain_velocity = [&](uint32_t first, uint32_t last, uint32_t env_row_base,
                              uint32_t env_artic_base) {
        for (uint32_t base = first; base < last; base += nwarps) {
            const uint32_t batch_count = min(nwarps, last - base);
            if (lane == 0u) batch_needed = 0u;
            __syncthreads();
            if (warp < batch_count) {
                const uint32_t slot = s.chain_rows[base + warp];
                StageRowWarp(a.urows, slot, staged_rows + 3u * warp, wlane);
                const bool tangent =
                    (staged_rows[3u * warp].flags & nk::nk_row_flags::kBlockTangent) != 0u;
                const bool needed = !tangent && RowNeedsVelocitySolveWarp(a.urows, slot,
                    env_row_base, env_artic_base, a.lambda, a.row_meff, a.row_damping,
                    a.chain_jacobian, a.chain_jacobian_b, tile_sh, a.body_linear,
                    a.body_angular, a.points, dofs, a.dt, wlane, staged_rows + 3u * warp);
                if (needed && wlane == 0u) atomicOr(&batch_needed, 1u);
            }
            __syncthreads();
            // A batch is skipped only if every row leaves its input unchanged.
            if (batch_needed == 0u) {
                __syncthreads();
                continue;
            }
            if (warp < batch_count) {
                const uint32_t slot = s.chain_rows[base + warp];
                const NkRow& normal = staged_rows[3u * warp];
                if (wlane == 0u && (normal.flags & nk::nk_row_flags::kBlockNormal))
                    staged_tangent_response[warp] = constraint::CoulombTangentSpectralResponse(
                        normal.contact_response, normal.mu, normal.friction_secondary);
                const uint32_t axes = (normal.flags & nk::nk_row_flags::kBlockNormal)
                    ? 3u : ((normal.flags & nk::nk_row_flags::kBlockTangent) ? 0u : 1u);
                for (uint32_t axis = 0u; axis < axes; ++axis) {
                    const uint32_t at = slot + axis * normal.group_normal_count;
                    if (axis != 0u) {
                        uint32_t word;
                        memcpy(&word, reinterpret_cast<const unsigned char*>(a.urows + at) +
                                      wlane * sizeof(word), sizeof(word));
                        memcpy(reinterpret_cast<unsigned char*>(staged_rows + 3u * warp + axis) +
                                   wlane * sizeof(word), &word, sizeof(word));
                    }
                    for (uint32_t k = wlane; k < dofs; k += warpSize) {
                        const size_t destination = size_t{3u * warp + axis} * dofs + k;
                        const size_t source = size_t{at} * dofs + k;
                        if (normal.a.kind == kNkSideArtic) {
                            staged_j[destination] = a.chain_jacobian[source];
                            staged_j[staged_size + destination] = a.row_minv_jt[source];
                        }
                        if (normal.b.kind == kNkSideArtic) {
                            staged_j[2u * staged_size + destination] = a.chain_jacobian_b[source];
                            staged_j[3u * staged_size + destination] = a.row_minv_jt_b[source];
                        }
                    }
                }
            }
            __syncthreads();
            for (uint32_t idx = 0u; idx < batch_count; ++idx) {
                const uint32_t gslot = s.chain_rows[base + idx];
                const NkRow* prepared = staged_rows + 3u * idx;
                if (prepared[0].flags & nk::nk_row_flags::kBlockTangent) {
                    continue;
                } else if (prepared[0].flags & nk::nk_row_flags::kBlockNormal) {
                    SolvePreparedContactBlock(gslot, env_artic_base, 3u * idx, warp, wlane,
                        prepared, a.lambda, a.row_damping, staged_j, staged_j + staged_size,
                        staged_j + 2u * staged_size, staged_j + 3u * staged_size, tile_sh,
                        a.body_linear, a.body_angular, a.body_inv_mass, a.body_inv_inertia,
                        a.points, dofs, a.dt, staged_tangent_response[idx], a.vel_tolerance,
                        chain_error, contact_side_velocity, contact_delta, &friction_active,
                        &row_changed, &row_significant);
                } else {
                    if (lane == 0u) row_significant = 0u;
                    __syncthreads();
                    if (warp == 0u) {
                        const bool significant = SolveUnionRowWarp(
                            gslot - env_row_base, gslot, env_row_base, env_artic_base, 3u * idx,
                            wlane, nullptr, nullptr, nullptr, nullptr, a.lambda, a.row_meff,
                            a.row_damping, staged_j, staged_j + staged_size,
                            staged_j + 2u * staged_size, staged_j + 3u * staged_size, tile_sh,
                            a.urows, a.body_linear, a.body_angular, a.body_inv_mass,
                            a.body_inv_inertia, a.points, dofs, a.dt, false, chain_error,
                            prepared, a.vel_tolerance);
                        if (wlane == 0u) row_significant = significant ? 1u : 0u;
                    }
                }
                __syncthreads();
                if (lane == 0u && row_significant != 0u) changed = true;
                __syncthreads();
            }
        }
    };

    auto chain_position = [&](uint32_t first, uint32_t last, uint32_t env_row_base,
                              uint32_t env_artic_base) {
        for (uint32_t base = first; base < last; base += nwarps) {
            const uint32_t batch_count = min(nwarps, last - base);
            // An unchanged batch cannot alter a later row's input during its ordered sweep.
            if (lane == 0u) batch_needed = 0u;
            __syncthreads();
            if (warp < batch_count) {
                const uint32_t gslot = s.chain_rows[base + warp];
                const bool needed = SolvePositionRowWarp<false>(gslot, env_row_base,
                    env_artic_base, gslot, wlane, a.row_meff, a.row_penetration,
                    a.row_pseudo_lambda, a.chain_jacobian, a.row_minv_jt, a.chain_jacobian_b,
                    a.row_minv_jt_b, tile_sh, a.urows, a.body_pseudo_linear,
                    a.body_pseudo_angular, a.body_inv_mass, a.body_inv_inertia, pseudo_points,
                    dofs, a.pos_beta, a.pos_slop, a.dt, a.baumgarte_max_velocity);
                if (needed && wlane == 0u) atomicOr(&batch_needed, 1u);
            }
            __syncthreads();
            if (batch_needed == 0u) {
                __syncthreads();
                continue;
            }
            for (uint32_t idx = warp == 0u ? 0u : batch_count; idx < batch_count; ++idx) {
                const uint32_t gslot = s.chain_rows[base + idx];
                SolvePositionRowWarp(gslot, env_row_base, env_artic_base, gslot, wlane,
                    a.row_meff, a.row_penetration, a.row_pseudo_lambda, a.chain_jacobian,
                    a.row_minv_jt, a.chain_jacobian_b, a.row_minv_jt_b, tile_sh, a.urows,
                    a.body_pseudo_linear, a.body_pseudo_angular, a.body_inv_mass,
                    a.body_inv_inertia, pseudo_points, dofs, a.pos_beta, a.pos_slop, a.dt,
                    a.baumgarte_max_velocity);
                __syncwarp();
            }
            __syncthreads();
        }
    };

    // An island stages every articulation tile of its environment and writes back its own.
    auto chain_sweep = [&](ChainSweep sweep) {
        float* const velocity = sweep == ChainSweep::Position ? a.qdot_pseudo_flat : a.qdot_flat;
        const bool compensated = sweep != ChainSweep::Position;
        const bool tiles = velocity != nullptr && dofs > 0u && a.cc_artic_first != nullptr;
        for (uint32_t k = blockIdx.x; k < chain_islands; k += gridDim.x) {
            const IslandRecord rec =
                reinterpret_cast<const IslandRecord*>(a.islands)[s.chain_islands[k]];
            const uint32_t root = a.island_root_sorted[rec.seg_off];
            const uint32_t env_row_base = rec.env * a.rows_per_env;
            const uint32_t env_artic_base = rec.env * k_tiles;
            const uint32_t live_off = rec.seg_off == 0u ? 0u : a.live_scan[rec.seg_off - 1u];
            const uint32_t first = s.chain_excl[live_off];
            const uint32_t last = s.chain_excl[a.live_scan[rec.seg_off + rec.seg_cnt - 1u]];
            float* const env_velocity = tiles ? velocity + size_t{env_artic_base} * dofs : nullptr;
            float* const env_error = s.qdot_error + size_t{env_artic_base} * dofs;
            for (uint32_t i = lane; tiles && i < tile_floats; i += blockDim.x) {
                tile_sh[i] = env_velocity[i];
                if (compensated) tile_error_sh[i] = env_error[i];
            }
            __syncthreads();
            if (sweep == ChainSweep::WarmStart) {
                for (uint32_t i = warp == 0u ? first : last; i < last; ++i) {
                    const uint32_t slot = s.chain_rows[i];
                    SolveUnionRowWarp(slot - env_row_base, slot, env_row_base, env_artic_base,
                        slot, wlane, nullptr, nullptr, nullptr, nullptr, a.lambda, a.row_meff,
                        a.row_damping, a.chain_jacobian, a.row_minv_jt, a.chain_jacobian_b,
                        a.row_minv_jt_b, tile_sh, a.urows, a.body_linear, a.body_angular,
                        a.body_inv_mass, a.body_inv_inertia, a.points, dofs, a.dt, true,
                        chain_error);
                    __syncwarp();
                }
            } else if (sweep == ChainSweep::Velocity) {
                chain_velocity(first, last, env_row_base, env_artic_base);
            } else {
                chain_position(first, last, env_row_base, env_artic_base);
            }
            __syncthreads();
            for (uint32_t i = lane; tiles && i < tile_floats; i += blockDim.x) {
                const uint32_t claim = a.cc_artic_first[env_artic_base + i / dofs];
                if (claim == ~0u || a.cc_root[claim] != root) continue;
                env_velocity[i] = tile_sh[i];
                if (compensated) env_error[i] = tile_error_sh[i];
            }
            __syncthreads();
        }
    };

    if (a.apply_cached) {
        color_sweep(false, [&](uint32_t slot, uint32_t env, NkRow*) {
            SolveUnionRowWarp(slot - env * a.rows_per_env, slot, env * a.rows_per_env,
                env * k_tiles, slot, wlane, nullptr, nullptr, nullptr, nullptr, a.lambda,
                a.row_meff, a.row_damping, a.chain_jacobian, a.row_minv_jt, a.chain_jacobian_b,
                a.row_minv_jt_b, nullptr, a.urows, a.body_linear, a.body_angular,
                a.body_inv_mass, a.body_inv_inertia, a.points, dofs, a.dt, true, a.error);
        });
        chain_sweep(ChainSweep::WarmStart);
        grid.sync();
    }
    for (uint32_t it = 0u; it < a.vel_iters; ++it) {
        if (blockIdx.x == 0u && threadIdx.x == 0u)
            s.control[kControlChanged + (it + 1u) % 3u] = 0u;
        changed = false;
        color_sweep(true, [&](uint32_t slot, uint32_t env, NkRow* staged) {
            const uint32_t env_row_base = env * a.rows_per_env;
            bool significant = false;
            if (staged[0].flags & nk::nk_row_flags::kBlockNormal) {
                significant = SolvePreparedContactBlockWarp(slot, env * k_tiles, wlane, a.urows,
                    staged, a.lambda, a.row_damping, a.chain_jacobian, a.row_minv_jt,
                    a.chain_jacobian_b, a.row_minv_jt_b, nullptr, a.body_linear, a.body_angular,
                    a.body_inv_mass, a.body_inv_inertia, a.points, dofs, a.dt, a.vel_tolerance,
                    a.error);
            } else {
                significant = SolveUnionRowWarp(slot - env_row_base, slot, env_row_base,
                    env * k_tiles, slot, wlane, nullptr, nullptr, nullptr, nullptr, a.lambda,
                    a.row_meff, a.row_damping, a.chain_jacobian, a.row_minv_jt,
                    a.chain_jacobian_b, a.row_minv_jt_b, nullptr, a.urows, a.body_linear,
                    a.body_angular, a.body_inv_mass, a.body_inv_inertia, a.points, dofs, a.dt,
                    false, a.error, staged, a.vel_tolerance);
            }
            changed |= significant;
        });
        chain_sweep(ChainSweep::Velocity);
        uint32_t* const flag = s.control + kControlChanged + it % 3u;
        if (__syncthreads_or(changed) && threadIdx.x == 0u) atomicOr(flag, 1u);
        grid.sync();
        if (LoadControl(flag) == 0u) break;
    }

    // Penetration drives a fresh pseudo-velocity field over the same colors and chains.
    if (pos_pass) {
        for (uint32_t position = color_warp; position < live; position += color_warps) {
            const uint32_t slot = a.live_order[position];
            const uint32_t env = slot / a.rows_per_env;
            const float* qdot = a.qdot_flat != nullptr
                ? a.qdot_flat + size_t{env * k_tiles} * dofs : nullptr;
            UpdateSpeculativePenetration<true>(slot, env * a.rows_per_env, env * k_tiles,
                a.row_penetration, a.urows, a.chain_jacobian, a.chain_jacobian_b, qdot,
                a.body_linear, a.body_angular, a.points, dofs, a.dt, wlane);
        }
        grid.sync();
        for (uint32_t it = 0u; it < a.pos_iters; ++it) {
            color_sweep(true, [&](uint32_t slot, uint32_t env, const NkRow* staged) {
                SolvePositionRowWarp(slot, env * a.rows_per_env, env * k_tiles, slot, wlane,
                    a.row_meff, a.row_penetration, a.row_pseudo_lambda, a.chain_jacobian,
                    a.row_minv_jt, a.chain_jacobian_b, a.row_minv_jt_b, nullptr, a.urows,
                    a.body_pseudo_linear, a.body_pseudo_angular, a.body_inv_mass,
                    a.body_inv_inertia, pseudo_points, dofs, a.pos_beta, a.pos_slop, a.dt,
                    a.baumgarte_max_velocity, staged);
            });
            chain_sweep(ChainSweep::Position);
            grid.sync();
        }
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
    const uint32_t k_tiles = artics_per_env == 0u ? 1u : artics_per_env;
    constexpr uint32_t warps = kColorBlockSize / 32u;
    const size_t solve_shared = sizeof(float) *
        (2u * size_t{k_tiles} * p.max_dof + 4u * 3u * size_t{warps} * p.max_dof + warps);
    uint32_t solve_blocks = 0u;
    if (ResidentGridSize(SolveColoredRowsKernel, kColorBlockSize, solve_shared, grid_bound,
                         &solve_blocks) != cudaSuccess)
        return Status::Failed;
    const PointMassView points{data.particle_inv_mass, data.particle_vel, data.grid_inv_mass,
                               data.grid_velocity, data.point_endpoint_ranges,
                               data.point_endpoint_terms};
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
    args.qdot_flat = data.qdot_flat;
    args.link_velocity = reinterpret_cast<Spatial6*>(data.link_velocity);
    args.qdot = data.qdot;
    args.body_linear = data.body_linear_velocity;
    args.body_angular = data.body_angular_velocity;
    args.body_inv_mass = static_cast<const float*>(data.body_inv_mass);
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

// Diagnostic reads occur after every island has committed its physical velocity and impulse.
__global__ void MeasureContactResidualKernel(DataView data, SolveRowsBlockIslandParams p) {
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const uint32_t stride = gridDim.x * (blockDim.x / warpSize);
    const uint32_t artics = p.articulation_count > 0u ? p.articulation_count / p.env_count : 1u;
    const auto* rows = reinterpret_cast<const NkRow*>(data.urows);
    const PointMassView points{data.particle_inv_mass, data.particle_vel,
        data.grid_inv_mass, data.grid_velocity, data.point_endpoint_ranges, data.point_endpoint_terms};
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
    const uint64_t row_capacity = uint64_t{p->rows_per_env} * p->env_count;
    if (row_capacity > uint64_t{kOwnerEmpty} -
            uint64_t{kColorGridLimit} * kColorBlockSize)
        return Status::InvalidArgument;
    if (p->measure_contact_residual != 0u) {
        if (data.contact_solve_metrics == nullptr || data.contact_solve_counts == nullptr)
            return Status::InvalidArgument;
        if (cudaMemsetAsync(data.contact_solve_metrics, 0,
                size_t{p->env_count} * constraint::kContactSolveMetricCount * sizeof(uint64_t), stream) != cudaSuccess ||
            cudaMemsetAsync(data.contact_solve_counts, 0,
                size_t{p->env_count} * constraint::kContactSolveCountSize * sizeof(uint32_t), stream) != cudaSuccess)
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
            if (cudaMemsetAsync(data.solver_velocity_scratch, 0, error_bytes, stream) != cudaSuccess)
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
            prep.row_penetration = pos_pass ? data.row_penetration : nullptr;
            prep.row_pseudo_lambda = pos_pass ? data.row_pseudo_lambda : nullptr;
            prep.total_rows = total_rows;
            prep.rows_per_env = p->rows_per_env;
            prep.artics_per_env = artics_per_env;
            prep.dof_stride = p->max_dof;
            prep.dt = p->dt;
            prep.pos_slop = p->pos_slop;
            int device = 0, sm_count = 0;
            if (cudaGetDevice(&device) != cudaSuccess ||
                cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                       device) != cudaSuccess)
                return Status::Failed;
            const uint32_t prepare_bound = static_cast<uint32_t>(std::min<uint64_t>(
                kColorGridLimit, uint64_t{static_cast<uint32_t>(sm_count)} * kColorBlocksPerSm));
            uint32_t prepare_blocks = 0u;
            if (ResidentGridSize(PrepareLiveColorsKernel, kColorBlockSize, 0u,
                                 prepare_bound, &prepare_blocks) != cudaSuccess)
                return Status::Failed;
            if (LaunchCooperativeCuda(PrepareLiveColorsKernel, dim3(prepare_blocks),
                    dim3(kColorBlockSize), 0u, stream,
                    reinterpret_cast<const NkRow*>(data.urows),
                    PointMassView{data.particle_inv_mass, data.particle_vel,
                        data.grid_inv_mass, data.grid_velocity,
                        data.point_endpoint_ranges, data.point_endpoint_terms},
                    static_cast<const float*>(data.body_inv_mass),
                    static_cast<const uint32_t*>(data.island_quads),
                    static_cast<const uint32_t*>(data.island_count), activity,
                    live_scan, live_order, scratch, prep) != cudaSuccess)
                return Status::Failed;
            const uint32_t scalar_blocks = max_island_bound < kScalarIslandGridBlocks
                ? max_island_bound : kScalarIslandGridBlocks;
            LaunchCuda(
                SolveRowsScalarIslandsKernel, dim3(scalar_blocks),
                dim3(kScalarIslandBlockSize), 0u, stream,
                reinterpret_cast<const NkRow*>(data.urows), data.lambda,
                static_cast<const float*>(data.row_meff),
                static_cast<const float*>(data.row_damping),
                data.body_linear_velocity, data.body_angular_velocity,
                static_cast<const float*>(data.body_inv_mass),
                static_cast<const math::SymmetricMat3*>(data.body_world_inv_inertia),
                PointMassView{data.particle_inv_mass, data.particle_vel,
                              data.grid_inv_mass, data.grid_velocity,
                              data.point_endpoint_ranges, data.point_endpoint_terms},
                data.island_quads, data.island_rows,
                pos_pass ? data.row_penetration : nullptr,
                pos_pass ? static_cast<float*>(data.row_pseudo_lambda) : nullptr,
                pos_pass ? data.body_pseudo_linear_velocity : nullptr,
                pos_pass ? data.body_pseudo_angular_velocity : nullptr,
                pos_pass ? data.particle_pseudo_vel : nullptr,
                pos_pass ? data.grid_pseudo_vel : nullptr,
                data.island_count, activity, p->rows_per_env, artics_per_env,
                static_cast<uint32_t>(p->vel_iters),
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
                       PointMassView{data.particle_inv_mass, data.particle_vel,
                                  data.grid_inv_mass, data.grid_velocity,
                                  data.point_endpoint_ranges, data.point_endpoint_terms},
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
                PointMassView{data.particle_inv_mass, data.particle_vel,
                              data.grid_inv_mass, data.grid_velocity,
                              data.point_endpoint_ranges, data.point_endpoint_terms},
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
