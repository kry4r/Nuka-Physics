// Independent islands retain ordered row updates and separate real/pseudo velocities.

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
constexpr uint32_t kDynamicIslandBlockSize = 256u;
constexpr uint32_t kScalarIslandBlockSize = 32u;
// One block carries the live-row scan so the compacted order stays deterministic.
constexpr uint32_t kCompactBlockSize = 1024u;
constexpr uint32_t kScalarIslandGridBlocks = 64u;
// A batch hashes the state keys its rows write into a double-buffered shared table.
constexpr uint32_t kOwnerTableSlots = 1024u;
constexpr uint32_t kOwnerKeysPerRow = 64u;
constexpr uint32_t kOwnerEmpty = ~0u;
constexpr uint32_t kOwnerGroupKind = 7u;
// A live row either runs ordered, runs concurrently with its batch, or is solved by its normal.
constexpr uint32_t kRowOrdered = 0u;
constexpr uint32_t kRowFree = 1u;
constexpr uint32_t kRowSkipped = 2u;

struct IslandActivityView {
    const uint32_t* sorted_roots = nullptr;
    uint32_t* needs_solve = nullptr;

    __device__ bool Active(const IslandRecord& island) const {
        return needs_solve == nullptr || needs_solve[sorted_roots[island.seg_off]] != 0u;
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

// Static schedules cache row data and Jacobians; dynamic schedules use global rows.
// Both allocate velocity tiles, with separate pseudo tiles for the position solve.
inline size_t IslandSharedBytes(uint32_t rows_per_env, uint32_t dof_stride,
                                uint32_t qdot_floats, bool cache_jw,
                                bool pos_pass, uint32_t k_tiles, uint32_t staged_warps = 0u) {
    if (!cache_jw) {
        // Global rows need shared velocity tiles and a per-island articulation index.
        return sizeof(float) * qdot_floats * (pos_pass ? 3u : 2u) +
               sizeof(uint32_t) * 2ull * k_tiles +
               sizeof(float) * staged_warps * (12ull * dof_stride + 1ull);
    }
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

template <bool cooperative = false>
__device__ float ComputeSlimRowVelocity(
    const SlimRow& sr, uint32_t gslot, uint32_t j_row,
    const float* J, const float* J_b, const float* qdot,
    const NkRow* __restrict__ urows,
    const math::Vec3* __restrict__ body_lin_vel,
    const math::Vec3* __restrict__ body_ang_vel,
    PointMassView point_masses, uint32_t dof_stride, uint32_t lane = 0u,
    const NkRow* prepared = nullptr) {
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
            if constexpr (cooperative) dyn_jv = point_masses.RowVelocityWarp(SlimPointSide(sr), lane);
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

// An owner key packs a state kind with its index; an unpackable index yields no key.
__device__ inline uint32_t OwnerKey(uint32_t kind, uint32_t index) {
    return index < (1u << 29u) ? (index << 3u) | kind : kOwnerEmpty;
}

__device__ inline uint32_t SideOwnerCount(const NkRowSide& side, PointMassView points,
                                          const float* body_inv_mass) {
    if (!PreparedSideWritable(side, body_inv_mass)) return 0u;
    return PointMassView::IsPointSide(side.kind) ? points.Count(side) : 1u;
}

__device__ inline uint32_t SideOwnerKey(const NkRowSide& side, uint32_t term, PointMassView points) {
    if (!PointMassView::IsPointSide(side.kind)) return OwnerKey(side.kind, side.index);
    const auto contribution = points.At(side, term);
    return OwnerKey(contribution.kind, contribution.index);
}

// Linear probing gives each key one slot whose mask collects every batch row holding it.
__device__ uint32_t InsertOwner(uint32_t* keys, uint32_t* masks, uint32_t key, uint32_t bit) {
    static_assert(kOwnerTableSlots == 1024u, "the hash keeps the top ten bits");
    uint32_t slot = (key * 2654435761u) >> 22u;
    for (;;) {
        const uint32_t previous = atomicCAS(keys + slot, kOwnerEmpty, key);
        if (previous == kOwnerEmpty || previous == key) {
            atomicOr(masks + slot, bit);
            return slot;
        }
        slot = (slot + 1u) & (kOwnerTableSlots - 1u);
    }
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

// One projected Coulomb step from the six side velocities of a prepared contact.
__device__ PreparedContactStep ProjectPreparedContact(
    const NkRow* rows, math::Vec3 old, float damping, const float* side_velocity, float dt,
    float tangent_response, float vel_tolerance) {
    const NkRow& normal = rows[0];
    const float old_normal = old.x;
    const float old_first = old.y;
    const float old_second = old.z;
    const float damping_scale = 1.0f / (1.0f + damping * dt);
    const float normal_velocity = side_velocity[0] + side_velocity[1];
    const float first_velocity = side_velocity[2] + side_velocity[3];
    const float second_velocity = side_velocity[4] + side_velocity[5];
    const math::Vec3 residual{
        normal.rhs * dt * damping_scale - normal_velocity -
            normal.compliance_alpha * damping_scale * old_normal,
        rows[1].rhs * dt - first_velocity - rows[1].compliance_alpha * old_first,
        rows[2].rhs * dt - second_velocity - rows[2].compliance_alpha * old_second};
    PreparedContactStep step;
    step.impulse = constraint::ProjectedCoulombStepWithResponse(
        normal.contact_response, residual, {old_normal, old_first, old_second},
        normal.mu, normal.friction_secondary, tangent_response);
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
        const PreparedContactStep step = ProjectPreparedContact(rows,
            {lambda[normal_slot], lambda[tangent_first_slot], lambda[tangent_second_slot]},
            row_damping[normal_slot], side_velocity, dt, tangent_response, vel_tolerance);
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
__device__ math::Vec3 LoadPointSideVelocities(
    const NkRowSide (&sides)[3], nk::PointEndpointRange range, PointMassView points,
    VelocityErrorView error, uint32_t lane, PointLaneTerm& term) {
    float result[3] = {0.0f, 0.0f, 0.0f};
    const uint32_t i = range.count == 1u ? 0u : lane;
    if (i < range.count) {
        if (sides[0].kind == nk::kNkSidePointEndpoint) {
            const nk::PointEndpointTerm& entry = points.terms[range.first + i];
            term.kind = entry.kind;
            term.index = entry.index;
            #pragma unroll
            for (uint32_t axis = 0u; axis < 3u; ++axis)
                term.jacobian[axis] = entry.TransposeMultiply(sides[axis].jlin);
        } else {
            term.kind = sides[0].kind;
            term.index = sides[0].index;
            #pragma unroll
            for (uint32_t axis = 0u; axis < 3u; ++axis) term.jacobian[axis] = sides[axis].jlin;
        }
        const math::Vec3* velocity = points.Velocity(term.kind);
        const float* inverse_mass = points.InverseMass(term.kind);
        const math::Vec3* compensation = error.Point(term.kind, term.index);
        if (velocity != nullptr) {
            term.velocity = velocity[term.index];
            #pragma unroll
            for (uint32_t axis = 0u; axis < 3u; ++axis)
                result[axis] += term.jacobian[axis].Dot(term.velocity);
        }
        if (inverse_mass != nullptr) term.inverse_mass = inverse_mass[term.index];
        if (compensation != nullptr) term.error = *compensation;
    }
    if (range.count == 1u) return {result[0], result[1], result[2]};
    return {WarpSum(result[0]), WarpSum(result[1]), WarpSum(result[2])};
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
    const math::Vec3 a = fits
        ? LoadPointSideVelocities(sides_a, range, points, error, lane, term)
        : ComputePreparedSideVelocities(rows, 0u, gslot, count, env_artic_base, J, J_b, qdot,
              body_linear, body_angular, points, dofs, lane);
    const math::Vec3 b = ComputePreparedSideVelocities(rows, 1u, gslot, count, env_artic_base,
        J, J_b, qdot, body_linear, body_angular, points, dofs, lane);
    const float side_velocity[6] = {a.x, b.x, a.y, b.y, a.z, b.z};
    const float tangent_response = constraint::CoulombTangentSpectralResponse(
        rows[0].contact_response, rows[0].mu, rows[0].friction_secondary);
    const PreparedContactStep step = ProjectPreparedContact(rows, old, damping, side_velocity,
        dt, tangent_response, vel_tolerance);
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

// Zero-update islands retain both impulse fields at zero throughout every ordered sweep.
__global__ void MarkIslandActivityKernel(
    const NkRow* __restrict__ urows, const float* __restrict__ lambda,
    const float* __restrict__ row_meff, const float* __restrict__ row_damping,
    const float* __restrict__ chain_jacobian, const float* __restrict__ chain_jacobian_b,
    const float* __restrict__ qdot_flat, const math::Vec3* __restrict__ body_lin_vel,
    const math::Vec3* __restrict__ body_ang_vel, PointMassView point_masses,
    const uint32_t* __restrict__ row_order, IslandActivityView activity,
    const float* __restrict__ row_penetration, float* __restrict__ row_pseudo_lambda,
    uint32_t* __restrict__ live_flags,
    uint32_t total_rows, uint32_t rows_per_env, uint32_t artics_per_env,
    uint32_t dof_stride, float dt, float pos_slop) {
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const uint32_t stride = gridDim.x * (blockDim.x / warpSize);
    for (uint32_t cursor = warp; cursor < total_rows; cursor += stride) {
        const uint32_t root = activity.sorted_roots[cursor];
        if (root == ~0u) return;
        const uint32_t slot = row_order[cursor];
        if (row_pseudo_lambda != nullptr && lane == 0u) row_pseudo_lambda[slot] = 0.0f;
        bool active = false;
        if (row_penetration != nullptr && !(urows[slot].flags & nk::nk_row_flags::kVelocityOnly))
            active |= row_penetration[slot] > pos_slop;
        if (!active) {
            const uint32_t env = slot / rows_per_env;
            const uint32_t env_artic_base = env * artics_per_env;
            const float* qdot = qdot_flat != nullptr
                ? qdot_flat + static_cast<size_t>(env_artic_base) * dof_stride : nullptr;
            active = RowNeedsVelocitySolveWarp(urows, slot, env * rows_per_env, env_artic_base,
                lambda, row_meff, row_damping, chain_jacobian, chain_jacobian_b, qdot,
                body_lin_vel, body_ang_vel, point_masses, dof_stride, dt, lane);
        }
        if (lane == 0u && live_flags != nullptr) live_flags[cursor] = active ? 1u : 0u;
        if (active && lane == 0u) atomicOr(&activity.needs_solve[root], 1u);
    }
}

// Gather each island's live rows into one ordered list. The sweep then walks only
// rows that can still change, and live_scan carries each island's slice bounds.
__global__ void CompactLiveRowsKernel(const uint32_t* __restrict__ sorted_roots,
                                      const uint32_t* __restrict__ row_order,
                                      uint32_t* __restrict__ live_scan,
                                      uint32_t* __restrict__ live_order,
                                      uint32_t total_rows) {
    using BlockScanT = cub::BlockScan<uint32_t, kCompactBlockSize>;
    __shared__ typename BlockScanT::TempStorage temp;
    __shared__ uint32_t running, tail_seen;
    if (threadIdx.x == 0u) { running = 0u; tail_seen = 0u; }
    __syncthreads();
    for (uint32_t base = 0u; base < total_rows; base += kCompactBlockSize) {
        const uint32_t idx = base + threadIdx.x;
        const bool in_prefix = idx < total_rows && sorted_roots[idx] != ~0u;
        if (idx < total_rows && !in_prefix) tail_seen = 1u;
        const uint32_t flag = in_prefix ? live_scan[idx] : 0u;
        uint32_t inclusive = 0u;
        BlockScanT(temp).InclusiveSum(flag, inclusive);
        const uint32_t start = running;
        if (flag != 0u) live_order[start + inclusive - 1u] = row_order[idx];
        __syncthreads();
        if (idx < total_rows) live_scan[idx] = start + inclusive;
        if (threadIdx.x == kCompactBlockSize - 1u) running = start + inclusive;
        __syncthreads();
        if (tail_seen != 0u) break;
    }
}

// Batch b of an island holds its live rows w * batches + b, one per solve warp. A row sharing
// no written state with the rest of its batch commutes with it and is marked free.
__global__ void ClassifyLiveRowsKernel(const NkRow* __restrict__ urows, PointMassView point_masses,
                                       const float* __restrict__ body_inv_mass,
                                       const uint32_t* __restrict__ islands,
                                       const uint32_t* __restrict__ island_count_dev,
                                       IslandActivityView activity,
                                       const uint32_t* __restrict__ live_scan,
                                       const uint32_t* __restrict__ live_order,
                                       uint32_t* __restrict__ row_modes) {
    __shared__ uint32_t owner_table[4u * kOwnerTableSlots];
    __shared__ __align__(16) unsigned char row_storage[kDynamicIslandBlockSize / 32u * sizeof(NkRow)];
    __shared__ uint32_t island_list[kDynamicIslandBlockSize];
    __shared__ uint32_t island_list_cnt;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t wlane = threadIdx.x & 31u;
    const uint32_t nwarps = blockDim.x >> 5u;
    NkRow* const staged = reinterpret_cast<NkRow*>(row_storage) + warp;
    for (uint32_t i = threadIdx.x; i < 4u * kOwnerTableSlots; i += blockDim.x)
        owner_table[i] = (i / kOwnerTableSlots) % 2u == 0u ? kOwnerEmpty : 0u;
    uint32_t held_low = kOwnerEmpty;
    uint32_t held_high = kOwnerEmpty;
    uint32_t round = 0u;
    const uint32_t island_count = *island_count_dev;
    for (uint32_t base = 0u; base < island_count; base += blockDim.x) {
        if (threadIdx.x == 0u) island_list_cnt = 0u;
        __syncthreads();
        if (base + threadIdx.x < island_count) {
            const IslandRecord rec = reinterpret_cast<const IslandRecord*>(islands)[base + threadIdx.x];
            if ((rec.flags & kIslandWarpWork) && rec.seg_cnt > 0u && activity.Active(rec))
                island_list[atomicAdd(&island_list_cnt, 1u)] = base + threadIdx.x;
        }
        __syncthreads();
        const uint32_t listed = island_list_cnt;
        for (uint32_t k = 0u; k < listed; ++k) {
            const IslandRecord rec = reinterpret_cast<const IslandRecord*>(islands)[island_list[k]];
            const uint32_t live_off = rec.seg_off == 0u ? 0u : live_scan[rec.seg_off - 1u];
            const uint32_t live_rows = live_scan[rec.seg_off + rec.seg_cnt - 1u] - live_off;
            const uint32_t batches = (live_rows + nwarps - 1u) / nwarps;
            for (uint32_t b = blockIdx.x; b < batches; b += gridDim.x, ++round) {
                // The previous round's table is released while this round fills the other.
                uint32_t* const keys = owner_table + (round & 1u) * 2u * kOwnerTableSlots;
                uint32_t* const masks = keys + kOwnerTableSlots;
                uint32_t* const released = owner_table + (~round & 1u) * 2u * kOwnerTableSlots;
                if (held_low != kOwnerEmpty) {
                    released[held_low] = kOwnerEmpty;
                    released[kOwnerTableSlots + held_low] = 0u;
                }
                if (held_high != kOwnerEmpty) {
                    released[held_high] = kOwnerEmpty;
                    released[kOwnerTableSlots + held_high] = 0u;
                }
                held_low = kOwnerEmpty;
                held_high = kOwnerEmpty;
                const uint32_t position = warp * batches + b;
                const bool valid = position < live_rows;
                uint32_t mode = kRowOrdered;
                bool participates = false;
                if (valid) {
                    StageRowWarp(urows, live_order[live_off + position], staged, wlane);
                    const NkRow& row = staged[0];
                    const uint32_t a_owners = SideOwnerCount(row.a, point_masses, body_inv_mass);
                    const uint32_t owners =
                        1u + a_owners + SideOwnerCount(row.b, point_masses, body_inv_mass);
                    if (row.flags & nk::nk_row_flags::kBlockTangent) {
                        mode = kRowSkipped;
                    } else if (owners <= kOwnerKeysPerRow) {
                        bool packed = true;
                        for (uint32_t j = wlane; j < owners; j += warpSize) {
                            const uint32_t key = j == 0u
                                ? OwnerKey(kOwnerGroupKind, row.group_first)
                                : (j - 1u < a_owners
                                    ? SideOwnerKey(row.a, j - 1u, point_masses)
                                    : SideOwnerKey(row.b, j - 1u - a_owners, point_masses));
                            if (key == kOwnerEmpty) {
                                packed = false;
                                continue;
                            }
                            const uint32_t held = InsertOwner(keys, masks, key, 1u << warp);
                            if (j < warpSize) held_low = held;
                            else held_high = held;
                        }
                        participates = __all_sync(0xffffffffu, packed);
                    }
                }
                __syncthreads();
                if (participates) {
                    const uint32_t others = ~(1u << warp);
                    const bool shared_owner =
                        (held_low != kOwnerEmpty && (masks[held_low] & others) != 0u) ||
                        (held_high != kOwnerEmpty && (masks[held_high] & others) != 0u);
                    if (!__any_sync(0xffffffffu, shared_owner)) mode = kRowFree;
                }
                if (valid && wlane == 0u) row_modes[live_off + position] = mode;
                __syncthreads();
            }
        }
        __syncthreads();
    }
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
        // Static schedules cache lambda and effective mass; dynamic schedules read global rows.
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
            const float damping_scale = 1.0f / (1.0f + damping * dt);
            // row_meff stores 1/(A+R), also used by the position pass. Convert
            // locally to 1/(A+R/(1+b*dt)) to match the normalized residual.
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
                    const float lg = lambda_sh != nullptr
                                         ? lambda_sh[sr.group_local + g]
                                         : lambda[env_row_base + sr.group_local + g];
                    total += fmaxf(lg, 0.0f);
                }
                lower = 0.0f;
                upper = fmaxf(sr.mu, 0.0f) * total;
            }
            const float new_impulse =
                fminf(fmaxf(old_impulse + lambda_inc, lower), upper);
            if (lambda_sh != nullptr) lambda_sh[ls] = new_impulse;
            else lambda[gslot] = new_impulse;
            const float d = new_impulse - old_impulse;
            delta = d;
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
                                     float baumgarte_max_velocity) {
    const SlimRow sr = MakeSlimRow(LoadRowWarp(urows, gslot, wlane), env_row_base, env_artic_base);
    const uint32_t flags = sr.flags;
    if (!(flags & nk::nk_row_flags::kActive) ||
        (flags & (nk::nk_row_flags::kFriction | nk::nk_row_flags::kBlockTangent |
                  nk::nk_row_flags::kVelocityOnly))) {
        return false;
    }

    // Both reaction sides participate in geometric push-out, preserving center of mass.
    const float jv = ComputeSlimRowVelocity<true>(
        sr, gslot, j_row, J_sh, J_b_sh, qdot_pseudo_sh, urows,
        body_pseudo_lin, body_pseudo_ang, point_masses, dof_stride, wlane);

    float delta = 0.0f;
    if (wlane == 0u) {
        // Pseudo separating velocity from the penetration, capped at
        // baumgarte_max_velocity (+inf default => byte-identical) for bounded push-out.
        const float depth = row_penetration[gslot];
        const float bias =
            fminf(beta * fmaxf(depth - slop, 0.0f) / dt, baumgarte_max_velocity);
        const float effective_mass = row_meff[gslot];
        const float old_imp = row_pseudo_lambda[gslot];
        // GEOMETRIC projection: no -R*lambda compliance term (this is position,
        // not a compliant force). One-sided (pseudo impulse >= 0).
        const float new_imp = fmaxf(old_imp + effective_mass * (bias - jv), 0.0f);
        if constexpr (apply) row_pseudo_lambda[gslot] = new_imp;
        const float d = new_imp - old_imp;
        delta = d;
    }
    delta = __shfl_sync(0xffffffffu, delta, 0);
    if (apply && delta != 0.0f) {
        ApplySlimImpulse(sr, gslot, j_row, wlane, delta, env_artic_base, qdot_pseudo_sh,
                         urows, J_sh, w_sh, J_b_sh, w_b_sh,
                         body_pseudo_lin, body_pseudo_ang, body_inv_mass,
                         body_world_inv_inertia, point_masses,
                         dof_stride);
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

template <bool pair_driven, bool dynamic>
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
    const uint32_t* __restrict__ live_scan,   // inclusive live count per island slot
    const uint32_t* __restrict__ live_order,  // compacted live rows (or null)
    float* __restrict__ row_penetration,    // split-impulse depth (or null)
    float* __restrict__ row_pseudo_lambda,        // split-impulse accumulator (or null)
    float* __restrict__ qdot_pseudo,              // per-link pseudo joint vel (or null)
    Spatial6* __restrict__ link_velocity_pseudo,  // per-link pseudo spatial vel (or null)
    float* __restrict__ qdot_pseudo_flat,         // per-artic pseudo tile (or null)
    math::Vec3* __restrict__ body_pseudo_lin_vel, // per-body pseudo lin vel (or null)
    math::Vec3* __restrict__ body_pseudo_ang_vel, // per-body pseudo ang vel (or null)
    math::Vec3* __restrict__ particle_pseudo_vel, // per-particle pseudo vel (or null)
    math::Vec3* __restrict__ grid_pseudo_vel,
    const uint32_t* __restrict__ island_count_dev, // dynamic island count (or null)
    IslandActivityView activity,
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
    const uint32_t live_islands = dynamic ? *island_count_dev : total_islands;
    for (uint64_t cursor = blockIdx.x; cursor < live_islands; cursor += gridDim.x) {
        const uint32_t island = static_cast<uint32_t>(cursor);
        // Dynamic records index contiguous active rows; static records index color segments.
        const IslandRecord rec =
            reinterpret_cast<const IslandRecord*>(islands)[island];
        const uint32_t seg_off = rec.seg_off;
        const uint32_t seg_cnt = rec.seg_cnt;
        const uint32_t flags = rec.flags;
        const uint32_t env = rec.env;
        if (dynamic && !(flags & kIslandWarpWork)) continue;
        // Live rows form a contiguous slice of the compacted list, bounded by the scan.
        const bool compacted = dynamic && live_scan != nullptr && live_order != nullptr;
        const uint32_t live_off =
            compacted ? (seg_off == 0u ? 0u : live_scan[seg_off - 1u]) : seg_off;
        const uint32_t live_rows =
            compacted ? live_scan[seg_off + seg_cnt - 1u] - live_off : seg_cnt;
        const uint32_t* const walk_order = compacted ? live_order : row_order;
        const bool active = activity.Active(rec);
        if (!active && !(flags & kIslandHasArticulation)) continue;
        const uint32_t lane = threadIdx.x;
        const uint32_t env_row_base = env * rows_per_env;
        const uint32_t k_tiles = artics_per_env == 0u ? 1u : artics_per_env;
        const uint32_t env_artic_base = env * k_tiles;  // first global artic of this env

        // Shared storage belongs to this block and is reused after each island finishes.
        extern __shared__ unsigned char dyn_sh[];
        // Static schedules cache row Jacobians; dynamic schedules keep them in global storage.
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
        // A present bitmap and compact tile list restrict shared velocity writes to this island.
        uint32_t* tile_present = nullptr;
        uint32_t* tile_list = nullptr;
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
            float* const tile_base =
                qdot_sh + static_cast<size_t>(qdot_floats) * (pos_pass ? 3u : 2u);
            tile_present = reinterpret_cast<uint32_t*>(tile_base);
            tile_list = tile_present + k_tiles;
            // Static schedules build their ordered segments in per-environment global scratch.
            order_sh = pd_solve_scratch + static_cast<size_t>(3u) * env_row_base;
            seg_sh = order_sh + rows_per_env;                        // 2R u32
        }
        __shared__ uint32_t tile_cnt_sh;

        const bool has_artic = (flags & kIslandHasArticulation) != 0u && dof_stride > 0u;
        // Union-find closes each island's articulation tile set; other islands never write it.
        if (dynamic && has_artic) {
            for (uint32_t i = lane; i < k_tiles; i += blockDim.x) tile_present[i] = 0u;
            if (lane == 0u) tile_cnt_sh = 0u;
            __syncthreads();
            for (uint32_t r = lane; r < seg_cnt; r += blockDim.x) {
                const NkRow& row = urows[row_order[seg_off + r]];
                if (row.a.kind == kNkSideArtic && row.a.index >= env_artic_base) {
                    atomicOr(&tile_present[row.a.index - env_artic_base], 1u);
                }
                if (row.b.kind == kNkSideArtic && row.b.index >= env_artic_base) {
                    atomicOr(&tile_present[row.b.index - env_artic_base], 1u);
                }
            }
            __syncthreads();
            for (uint32_t t = lane; t < k_tiles; t += blockDim.x) {
                if (tile_present[t] != 0u) tile_list[atomicAdd(&tile_cnt_sh, 1u)] = t;
            }
            __syncthreads();
        }
        if (has_artic) {
            if (dynamic) {
                // load ONLY this component's tiles (into their env-local qdot_sh slots);
                // the unloaded slots are never read (no component row touches them).
                const uint32_t tc = tile_cnt_sh;
                for (uint32_t u = 0u; u < tc; ++u) {
                    const uint32_t tile = tile_list[u];
                    for (uint32_t k = lane; k < dof_stride; k += blockDim.x) {
                        qdot_sh[tile * dof_stride + k] = qdot_flat[
                            static_cast<size_t>(env_artic_base + tile) * dof_stride + k];
                    }
                }
            } else {
                // load EVERY co-resident articulation tile of this env (K tiles). At
                // K==1 this is the single legacy tile (qdot_flat[env*dof_stride]).
                for (uint32_t i = lane; i < k_tiles * dof_stride; i += blockDim.x) {
                    qdot_sh[i] = qdot_flat[static_cast<size_t>(env_artic_base) * dof_stride + i];
                }
            }
        }
        // Static schedules cache row parameters; dynamic schedules read them from global storage.
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
        // Compact active static rows once, preserving the order of all surviving segments.
        __shared__ uint32_t live_seg_cnt_sh;
        if (dynamic) {
            // Dynamic schedules already group active rows into deterministic single-row segments.
            if (lane == 0u) live_seg_cnt_sh = active ? live_rows : 0u;
        } else if (lane == 0u) {
            uint32_t out_rows = 0u;
            uint32_t out_segs = 0u;
            const uint32_t span_start =
                seg_cnt > 0u ? segments[static_cast<size_t>(seg_off) * 2u + 0u] : 0u;
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
            (void)span_start;
        }
        __syncthreads();
        const uint32_t live_seg_cnt = live_seg_cnt_sh;
        // Live-row modes come from ClassifyLiveRowsKernel, indexed by compacted position.
        const uint32_t* const row_modes = compacted ? pd_solve_scratch : nullptr;

        const uint32_t warp = lane >> 5u;
        const uint32_t wlane = lane & 31u;
        const uint32_t nwarps = blockDim.x >> 5u;
        __shared__ __align__(16) unsigned char row_storage[
            3u * kDynamicIslandBlockSize / 32u * sizeof(NkRow)];
        auto* staged_rows = reinterpret_cast<NkRow*>(row_storage);
        float* const staged_j = dynamic ? reinterpret_cast<float*>(tile_list + k_tiles) : nullptr;
        const size_t staged_size = size_t{3u} * nwarps * dof_stride;
        float* const staged_tangent_response = dynamic ? staged_j + 4u * staged_size : nullptr;
        __shared__ uint32_t batch_needed, sweep_changed, free_rows_sh, ordered_rows_sh;
        __shared__ uint32_t row_changed, row_significant, friction_active;
        __shared__ float contact_side_velocity[6], contact_delta[3];
        __shared__ uint32_t ordered_list[kDynamicIslandBlockSize];
        __shared__ uint32_t ordered_warp_rows[kDynamicIslandBlockSize / 32u];
        // Batch b hands warp w the live row w * batches + b, so one batch spans the island.
        const uint32_t batches = dynamic ? (live_seg_cnt + nwarps - 1u) / nwarps : 0u;
        if (row_modes != nullptr) {
            if (lane == 0u) {
                free_rows_sh = 0u;
                ordered_rows_sh = 0u;
            }
            __syncthreads();
            uint32_t free_rows = 0u;
            uint32_t ordered_rows = 0u;
            for (uint32_t position = lane; position < live_seg_cnt; position += blockDim.x) {
                const uint32_t mode = row_modes[live_off + position];
                free_rows += mode == kRowFree ? 1u : 0u;
                ordered_rows += mode == kRowOrdered ? 1u : 0u;
            }
            free_rows = __reduce_add_sync(0xffffffffu, free_rows);
            ordered_rows = __reduce_add_sync(0xffffffffu, ordered_rows);
            if (wlane == 0u) {
                atomicAdd(&free_rows_sh, free_rows);
                atomicAdd(&ordered_rows_sh, ordered_rows);
            }
            __syncthreads();
        }

        // A row sharing no written state with the rest of its batch commutes with it, so its
        // warp solves it concurrently with the other warps.
        auto free_sweep = [&](auto&& solve_row) {
            if (row_modes == nullptr || free_rows_sh == 0u) return;
            NkRow* const staged = staged_rows + 3u * warp;
            for (uint32_t b = 0u; b < batches; ++b) {
                const uint32_t position = warp * batches + b;
                const bool valid = position < live_seg_cnt;
                const uint32_t slot = valid ? walk_order[live_off + position] : 0u;
                const uint32_t mode = valid ? row_modes[live_off + position] : kRowOrdered;
                if (mode == kRowFree) {
                    StageRowWarp(urows, slot, staged, wlane);
                    solve_row(slot, staged);
                }
                __syncthreads();
            }
        };
        // Rows that share state keep their relative live order, gathered one chunk at a time.
        auto ordered_sweep = [&](auto&& solve_list) {
            if (row_modes != nullptr && ordered_rows_sh == 0u) return;
            for (uint32_t chunk = 0u; chunk < live_seg_cnt; chunk += blockDim.x) {
                const uint32_t position = chunk + lane;
                uint32_t slot = 0u;
                bool take = false;
                if (position < live_seg_cnt) {
                    slot = walk_order[live_off + position];
                    take = row_modes == nullptr || row_modes[live_off + position] == kRowOrdered;
                }
                const uint32_t ballot = __ballot_sync(0xffffffffu, take);
                if (wlane == 0u) ordered_warp_rows[warp] = __popc(ballot);
                __syncthreads();
                uint32_t before = 0u;
                uint32_t count = 0u;
                for (uint32_t w = 0u; w < nwarps; ++w) {
                    if (w < warp) before += ordered_warp_rows[w];
                    count += ordered_warp_rows[w];
                }
                if (take) ordered_list[before + __popc(ballot & ((1u << wlane) - 1u))] = slot;
                __syncthreads();
                if (count > 0u) solve_list(count);
                __syncthreads();
            }
        };

        if (with_b_arm != 0u && apply_cached_impulses) {
            if constexpr (dynamic) {
                auto apply_cached = [&](uint32_t gslot) {
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
                };
                free_sweep([&](uint32_t gslot, const NkRow*) { apply_cached(gslot); });
                ordered_sweep([&](uint32_t count) {
                    for (uint32_t idx = warp == 0u ? 0u : count; idx < count; ++idx) {
                        apply_cached(ordered_list[idx]);
                        __syncwarp();
                    }
                });
            } else {
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
        }
        for (uint32_t it = 0u; it < vel_iters; ++it) {
            bool warp_changed = false;
            if (lane == 0u) sweep_changed = 0u;
            __syncthreads();
            if constexpr (dynamic) {
                // Free rows read their Jacobians from global slots with tangent axes a group apart.
                free_sweep([&](uint32_t gslot, NkRow* staged) {
                    bool significant = false;
                    if (staged[0].flags & nk::nk_row_flags::kBlockNormal) {
                        significant = SolvePreparedContactBlockWarp(gslot, env_artic_base,
                            wlane, urows, staged, lambda, row_damping, chain_jacobian,
                            row_minv_jt, chain_jacobian_b, row_minv_jt_b, qdot_sh,
                            body_lin_vel, body_ang_vel, body_inv_mass, body_world_inv_inertia,
                            point_masses, dof_stride, dt, vel_tolerance, error);
                    } else {
                        significant = SolveUnionRowWarp(
                            gslot - env_row_base, gslot, env_row_base, env_artic_base,
                            gslot, wlane, nullptr, nullptr, nullptr, nullptr,
                            lambda, row_meff, row_damping, chain_jacobian,
                            row_minv_jt, chain_jacobian_b, row_minv_jt_b,
                            qdot_sh, urows,
                            body_lin_vel, body_ang_vel, body_inv_mass,
                            body_world_inv_inertia, point_masses, dof_stride, dt,
                            false, error, staged, vel_tolerance);
                    }
                    if (significant && wlane == 0u) atomicOr(&sweep_changed, 1u);
                });
                ordered_sweep([&](uint32_t count) {
                    for (uint32_t s = 0u; s < count; s += nwarps) {
                        const uint32_t batch_count = min(nwarps, count - s);
                        if (lane == 0u) batch_needed = 0u;
                        __syncthreads();
                        if (warp < batch_count) {
                            const uint32_t slot = ordered_list[s + warp];
                            StageRowWarp(urows, slot, staged_rows + 3u * warp, wlane);
                            const bool tangent = (staged_rows[3u * warp].flags &
                                nk::nk_row_flags::kBlockTangent) != 0u;
                            const bool needed = !tangent && RowNeedsVelocitySolveWarp(urows, slot,
                                env_row_base, env_artic_base, lambda, row_meff, row_damping,
                                chain_jacobian, chain_jacobian_b, qdot_sh, body_lin_vel,
                                body_ang_vel, point_masses, dof_stride, dt, wlane,
                                staged_rows + 3u * warp);
                            if (needed && wlane == 0u) atomicOr(&batch_needed, 1u);
                        }
                        __syncthreads();
                        // A batch is skipped only if every row leaves its input unchanged.
                        if (batch_needed == 0u) {
                            __syncthreads();
                            continue;
                        }
                        if (warp < batch_count) {
                            const uint32_t slot = ordered_list[s + warp];
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
                                    memcpy(&word, reinterpret_cast<const unsigned char*>(urows + at) +
                                                  wlane * sizeof(word), sizeof(word));
                                    memcpy(reinterpret_cast<unsigned char*>(staged_rows + 3u * warp + axis) +
                                               wlane * sizeof(word), &word, sizeof(word));
                                }
                                for (uint32_t k = wlane; k < dof_stride; k += warpSize) {
                                    const size_t destination =
                                        size_t{3u * warp + axis} * dof_stride + k;
                                    const size_t source = size_t{at} * dof_stride + k;
                                    if (normal.a.kind == kNkSideArtic) {
                                        staged_j[destination] = chain_jacobian[source];
                                        staged_j[staged_size + destination] = row_minv_jt[source];
                                    }
                                    if (normal.b.kind == kNkSideArtic) {
                                        staged_j[2u * staged_size + destination] = chain_jacobian_b[source];
                                        staged_j[3u * staged_size + destination] = row_minv_jt_b[source];
                                    }
                                }
                            }
                        }
                        __syncthreads();
                        for (uint32_t idx = 0u; idx < batch_count; ++idx) {
                            const uint32_t gslot = ordered_list[s + idx];
                            const NkRow* prepared = staged_rows + 3u * idx;
                            if (prepared[0].flags & nk::nk_row_flags::kBlockTangent) {
                                continue;
                            } else if (prepared[0].flags & nk::nk_row_flags::kBlockNormal) {
                                SolvePreparedContactBlock(gslot, env_artic_base, 3u * idx,
                                    warp, wlane, prepared, lambda, row_damping,
                                    staged_j, staged_j + staged_size,
                                    staged_j + 2u * staged_size, staged_j + 3u * staged_size,
                                    qdot_sh, body_lin_vel, body_ang_vel, body_inv_mass,
                                    body_world_inv_inertia, point_masses, dof_stride, dt,
                                    staged_tangent_response[idx], vel_tolerance, error,
                                    contact_side_velocity, contact_delta, &friction_active,
                                    &row_changed, &row_significant);
                            } else {
                                if (lane == 0u) row_significant = 0u;
                                __syncthreads();
                                if (warp == 0u) {
                                    const bool changed = SolveUnionRowWarp(
                                        gslot - env_row_base, gslot, env_row_base, env_artic_base,
                                        3u * idx, wlane, nullptr, nullptr, nullptr, nullptr,
                                        lambda, row_meff, row_damping, staged_j,
                                        staged_j + staged_size, staged_j + 2u * staged_size,
                                        staged_j + 3u * staged_size, qdot_sh, urows,
                                        body_lin_vel, body_ang_vel, body_inv_mass,
                                        body_world_inv_inertia, point_masses, dof_stride, dt,
                                        false, error, prepared, vel_tolerance);
                                    if (wlane == 0u) row_significant = changed ? 1u : 0u;
                                }
                            }
                            __syncthreads();
                            if (lane == 0u && row_significant != 0u) atomicOr(&sweep_changed, 1u);
                            __syncthreads();
                        }
                    }
                });
            } else {
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
            }
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
            if (dynamic) {
                for (uint32_t i = warp; i < seg_cnt; i += nwarps)
                    UpdateSpeculativePenetration<true>(row_order[seg_off + i], env_row_base,
                        env_artic_base, row_penetration, urows, chain_jacobian, chain_jacobian_b,
                        qdot_sh, body_lin_vel, body_ang_vel, point_masses, dof_stride, dt, wlane);
                for (uint32_t i = lane; i < seg_cnt; i += blockDim.x) {
                    row_pseudo_lambda[row_order[seg_off + i]] = 0.0f;
                }
            } else if (live_seg_cnt > 0u) {
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
                if constexpr (dynamic) {
                    const PointMassView pseudo_points =
                        point_masses.Pseudo(particle_pseudo_vel, grid_pseudo_vel);
                    free_sweep([&](uint32_t gslot, const NkRow*) {
                        SolvePositionRowWarp(gslot, env_row_base, env_artic_base, gslot, wlane,
                            row_meff, row_penetration, row_pseudo_lambda, chain_jacobian,
                            row_minv_jt, chain_jacobian_b, row_minv_jt_b, qdot_pseudo_sh, urows,
                            body_pseudo_lin_vel, body_pseudo_ang_vel, body_inv_mass,
                            body_world_inv_inertia, pseudo_points, dof_stride, pos_beta,
                            pos_slop, dt, baumgarte_max_velocity);
                    });
                    ordered_sweep([&](uint32_t count) {
                        for (uint32_t s = 0u; s < count; s += nwarps) {
                            const uint32_t cnt = min(nwarps, count - s);
                            // An unchanged batch cannot alter a later row's input during its ordered sweep.
                            if (lane == 0u) batch_needed = 0u;
                            __syncthreads();
                            if (warp < cnt) {
                                const uint32_t gslot = ordered_list[s + warp];
                                const bool needed = SolvePositionRowWarp<false>(
                                    gslot, env_row_base, env_artic_base, gslot, wlane,
                                    row_meff, row_penetration, row_pseudo_lambda,
                                    chain_jacobian, row_minv_jt, chain_jacobian_b, row_minv_jt_b,
                                    qdot_pseudo_sh, urows, body_pseudo_lin_vel, body_pseudo_ang_vel,
                                    body_inv_mass, body_world_inv_inertia, pseudo_points,
                                    dof_stride, pos_beta, pos_slop, dt, baumgarte_max_velocity);
                                if (needed && wlane == 0u) atomicOr(&batch_needed, 1u);
                            }
                            __syncthreads();
                            if (batch_needed == 0u) {
                                __syncthreads();
                                continue;
                            }
                            for (uint32_t idx = warp == 0u ? 0u : cnt; idx < cnt; ++idx) {
                                const uint32_t gslot = ordered_list[s + idx];
                                SolvePositionRowWarp(gslot, env_row_base, env_artic_base, gslot,
                                    wlane, row_meff, row_penetration, row_pseudo_lambda,
                                    chain_jacobian, row_minv_jt, chain_jacobian_b, row_minv_jt_b,
                                    qdot_pseudo_sh, urows, body_pseudo_lin_vel, body_pseudo_ang_vel,
                                    body_inv_mass, body_world_inv_inertia, pseudo_points,
                                    dof_stride, pos_beta, pos_slop, dt, baumgarte_max_velocity);
                                __syncwarp();
                            }
                            __syncthreads();
                        }
                    });
                } else {
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
                // Dynamic schedules scatter only owned tiles; static schedules own every environment tile.
                const uint32_t scatter_tiles = dynamic ? tile_cnt_sh : k_tiles;
                for (uint32_t i = lane; i < scatter_tiles * dof_stride; i += blockDim.x) {
                    const uint32_t u = i / dof_stride;       // tile slot in the iteration
                    const uint32_t k = i - u * dof_stride;   // DOF within tile
                    const uint32_t a = dynamic ? tile_list[u] : u;  // env-local tile
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

// Flush qdot_flat -> link_velocity for an articulation no active row claimed
// (cc_artic_first==sentinel): the static all-tiles scatter did this every step.
__global__ void FlushOrphanArticKernel(
    const uint32_t* __restrict__ cc_artic_first, uint32_t artic_count,
    uint32_t artics_per_env, uint32_t dof_stride, uint32_t base_link_count,
    const float* __restrict__ qdot_flat, Spatial6* __restrict__ link_velocity,
    float* __restrict__ qdot, const uint32_t* __restrict__ dof_to_link,
    const uint32_t* __restrict__ dof_to_component) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= artic_count * dof_stride) return;
    const uint32_t ag = t / dof_stride;
    if (cc_artic_first[ag] != ~0u) return;  // a contact row already solved this tile.
    const uint32_t k = t - ag * dof_stride;
    const uint32_t ape = (artics_per_env == 0u) ? 1u : artics_per_env;
    const uint32_t env = ag / ape;
    const uint32_t a = ag - env * ape;       // env-local tile (env_artic_base + a == ag).
    const uint32_t links_per_dog = (ape > 0u) ? (base_link_count / ape) : base_link_count;
    uint32_t comp; size_t gl;
    ArticDofTarget(env, a, k, dof_stride, links_per_dog, base_link_count,
                   dof_to_link, dof_to_component, comp, gl);
    const float v = qdot_flat[static_cast<size_t>(ag) * dof_stride + k];
    if (comp != ~0u) {
        link_velocity[gl].v[comp] = v;
    } else {
        qdot[gl] = v;
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
        // Shared storage includes velocity tiles and the schedule's optional row cache.
        size_t shared_bytes = IslandSharedBytes(p->rows_per_env, p->max_dof,
                                                qdot_floats, !with_b_arm,
                                                pos_pass, artics_per_env);
        uint32_t island_block_size = run_dynamic ? kDynamicIslandBlockSize :
            (with_b_arm ? kPairDrivenIslandBlockSize : kUnionIslandBlockSize);
        const auto island_kernel = run_dynamic ? SolveRowsBlockIslandKernel<true, true> :
            (with_b_arm ? SolveRowsBlockIslandKernel<true, false> : SolveRowsBlockIslandKernel<false, false>);
        uint32_t grid_islands = 0u;
        if (run_dynamic) {
            for (;;) {
                shared_bytes = IslandSharedBytes(p->rows_per_env, p->max_dof, qdot_floats,
                    false, pos_pass, artics_per_env, island_block_size / 32u);
                const auto status = ResidentGridSize(island_kernel, island_block_size,
                    shared_bytes, island_bound, &grid_islands);
                if (status == cudaSuccess) break;
                if (status != cudaErrorInvalidConfiguration || island_block_size == 32u)
                    return Status::Failed;
                island_block_size /= 2u;
            }
        } else if (ResidentGridSize(island_kernel, island_block_size, shared_bytes,
                                    island_bound, &grid_islands) != cudaSuccess) {
            return Status::Failed;
        }
        // Zero every GLOBAL pseudo accumulator (rigid/particle/articulation): the
        // dynamic schedule rewrites only solved tiles, so a dropped tile reads 0 here.
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
        // Dynamic schedule: the per-step CC pass (island_quads/rows + island_count).
        // Static schedule: the cook-time arrays (model.island_row_offsets/row_order).
        const uint32_t* const islands_in = run_dynamic
            ? data.island_quads : static_cast<const uint32_t*>(model.island_row_offsets);
        const uint32_t* const row_order_in = run_dynamic
            ? data.island_rows : static_cast<const uint32_t*>(model.row_order);
        const uint32_t* const island_count_dev = run_dynamic ? data.island_count : nullptr;
        IslandActivityView activity;
        // The scratch tail past the per-env batch flags carries the live-row list.
        const uint32_t scratch_rows = p->rows_per_env * p->env_count;
        uint32_t* const live_scan = run_dynamic && data.pd_solve_scratch != nullptr
            ? data.pd_solve_scratch + scratch_rows : nullptr;
        uint32_t* const live_order = live_scan != nullptr ? live_scan + scratch_rows : nullptr;
        if (run_dynamic) {
            // Island emission releases cc_parent; each solve reuses it for root activity.
            const uint32_t total_rows = p->rows_per_env * p->env_count;
            activity = {data.island_root_sorted, data.cc_parent};
            if (cudaMemsetAsync(activity.needs_solve, 0,
                                static_cast<size_t>(total_rows) * sizeof(uint32_t),
                                stream) != cudaSuccess) return Status::Failed;
            constexpr uint32_t activity_block_size = 128u;
            const uint32_t activity_bound = (total_rows + activity_block_size / 32u - 1u) /
                                            (activity_block_size / 32u);
            uint32_t activity_blocks = 0u;
            if (ResidentGridSize(MarkIslandActivityKernel, activity_block_size, 0u,
                                 activity_bound, &activity_blocks) != cudaSuccess) return Status::Failed;
            LaunchCuda(MarkIslandActivityKernel, dim3(activity_blocks),
                dim3(activity_block_size), 0u, stream,
                reinterpret_cast<const NkRow*>(data.urows), data.lambda,
                static_cast<const float*>(data.row_meff), static_cast<const float*>(data.row_damping),
                static_cast<const float*>(data.chain_jacobian),
                static_cast<const float*>(data.chain_jacobian_b), data.qdot_flat,
                data.body_linear_velocity, data.body_angular_velocity,
                PointMassView{data.particle_inv_mass, data.particle_vel,
                    data.grid_inv_mass, data.grid_velocity,
                    data.point_endpoint_ranges, data.point_endpoint_terms},
                data.island_rows, activity,
                pos_pass ? data.row_penetration : nullptr,
                pos_pass ? data.row_pseudo_lambda : nullptr,
                live_scan,
                total_rows, p->rows_per_env, artics_per_env, p->max_dof, p->dt, p->pos_slop);
            if (cudaGetLastError() != cudaSuccess) return Status::Failed;
            LaunchCuda(CompactLiveRowsKernel, dim3(1u), dim3(kCompactBlockSize), 0u, stream,
                       data.island_root_sorted, data.island_rows, live_scan, live_order,
                       total_rows);
            if (cudaGetLastError() != cudaSuccess) return Status::Failed;
            // Classification needs the island solve's warp count, so it shares its block size.
            const uint32_t classify_warps = island_block_size / 32u;
            uint32_t classify_blocks = 0u;
            if (ResidentGridSize(ClassifyLiveRowsKernel, island_block_size, 0u,
                    (total_rows + classify_warps - 1u) / classify_warps,
                    &classify_blocks) != cudaSuccess) return Status::Failed;
            LaunchCuda(ClassifyLiveRowsKernel, dim3(classify_blocks), dim3(island_block_size), 0u,
                       stream, reinterpret_cast<const NkRow*>(data.urows),
                       PointMassView{data.particle_inv_mass, data.particle_vel,
                                     data.grid_inv_mass, data.grid_velocity,
                                     data.point_endpoint_ranges, data.point_endpoint_terms},
                       static_cast<const float*>(data.body_inv_mass), islands_in,
                       island_count_dev, activity, live_scan, live_order, data.pd_solve_scratch);
            if (cudaGetLastError() != cudaSuccess) return Status::Failed;
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
        }
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
                   islands_in,
                   static_cast<const uint32_t*>(model.island_color_segments),
                   row_order_in,
                   static_cast<const uint32_t*>(model.dof_to_link),
                   static_cast<const uint32_t*>(model.dof_to_component),
                   data.pd_solve_scratch, live_scan, live_order,
                   pos_pass ? data.row_penetration : nullptr,
                   pos_pass ? static_cast<float*>(data.row_pseudo_lambda) : nullptr,
                   pos_pass ? static_cast<float*>(data.qdot_pseudo) : nullptr,
                   pos_pass ? reinterpret_cast<Spatial6*>(data.link_velocity_pseudo) : nullptr,
                   pos_pass ? static_cast<float*>(data.qdot_pseudo_flat) : nullptr,
                   pos_pass ? data.body_pseudo_linear_velocity : nullptr,
                   pos_pass ? data.body_pseudo_angular_velocity : nullptr,
                   pos_pass ? data.particle_pseudo_vel : nullptr,
                   pos_pass ? data.grid_pseudo_vel : nullptr,
                   island_count_dev, activity,
                   p->total_islands, p->rows_per_env, p->max_dof,
                   p->base_link_count, artics_per_env,
                   static_cast<uint32_t>(p->vel_iters),
                   static_cast<uint32_t>(p->pos_iters),
                   p->pos_beta, p->pos_slop, p->dt, p->vel_tolerance,
                   p->baumgarte_max_velocity, p->continue_impulses == 0u, error);
        if (cudaGetLastError() != cudaSuccess) return Status::Failed;
        // Flush the articulation tiles the dynamic schedule dropped (the static path
        // scatters all tiles in-kernel). cc_artic_first is BuildSolveIslands' claim table.
        if (run_dynamic && p->articulation_count > 0u && p->max_dof > 0u &&
            data.cc_artic_first != nullptr) {
            const uint32_t flush_threads = p->articulation_count * p->max_dof;
            const uint32_t flush_blocks =
                (flush_threads + kPairDrivenIslandBlockSize - 1u) /
                kPairDrivenIslandBlockSize;
            LaunchCuda(FlushOrphanArticKernel, dim3(flush_blocks),
                       dim3(kPairDrivenIslandBlockSize), 0u, stream,
                       data.cc_artic_first, p->articulation_count, artics_per_env,
                       p->max_dof, p->base_link_count, data.qdot_flat,
                       reinterpret_cast<Spatial6*>(data.link_velocity), data.qdot,
                       static_cast<const uint32_t*>(model.dof_to_link),
                       static_cast<const uint32_t*>(model.dof_to_component));
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

void RegisterNkSolveRowsOps() {
    SetCudaOp(NkOp::SolveRowsBlockIsland, &OpSolveRowsBlockIsland);
}

} // namespace nuka::phi
