// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- world-pose FK + foot-vs-ground contacts
// ---------------------------------------------------------------------------

#include "runtime/articulation/articulation_contacts.hpp"

#include "math/quat.hpp"
#include "math/transform.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::runtime::articulation {

namespace {

constexpr uint32_t kInvalidLink = ~0u;

// -- device vec/quat helpers ------------------------------------------------
// math::Quat::operator*, FromAxisAngle and Rotate are host-only (constexpr /
// inline, not __device__), so the device path hand-rolls them here, mirroring
// the w-first Hamilton-product and half-angle conventions in math/quat.hpp and
// the RotateByQuat short form already used in articulation_jacobian.cu.

__device__ math::Vec3 AddVec(math::Vec3 a, math::Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ math::Vec3 ScaleVec(math::Vec3 v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

__device__ float Dot3(math::Vec3 a, math::Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ math::Vec3 Cross3(math::Vec3 a, math::Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

// Normalizes a contact normal, falling back to +Z for a degenerate input.
// Mirrors articulation_jacobian.cu's NormalizeOrUp (local copy to keep edits
// scoped to this TU).
__device__ math::Vec3 NormalizeOrUpLocal(math::Vec3 normal) {
    const float length_sq = Dot3(normal, normal);
    if (length_sq > 1.0e-10f) {
        const float inv_length = rsqrtf(length_sq);
        return {normal.x * inv_length, normal.y * inv_length, normal.z * inv_length};
    }
    return {0.0f, 0.0f, 1.0f};
}

// math::Quat's constructors are host-only constexpr, so the device path builds
// quaternions by mutating a default-constructed value (field assignment is
// device-legal) rather than via brace/Identity()/operator* construction.
__device__ math::Quat MakeQuat(float w, float x, float y, float z) {
    math::Quat q;
    q.w = w;
    q.x = x;
    q.y = y;
    q.z = z;
    return q;
}

__device__ math::Quat QuatIdentity() {
    return MakeQuat(1.0f, 0.0f, 0.0f, 0.0f);
}

// Hamilton product this = a * b (w-first), mirroring math::Quat::operator*.
__device__ math::Quat QuatMul(math::Quat a, math::Quat b) {
    return MakeQuat(
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w);
}

__device__ math::Quat QuatNormalize(math::Quat q) {
    const float norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (norm_sq <= 1.0e-12f) {
        return QuatIdentity();
    }
    const float inv_norm = rsqrtf(norm_sq);
    return MakeQuat(q.w * inv_norm, q.x * inv_norm, q.y * inv_norm, q.z * inv_norm);
}

// Unit quaternion for a rotation of `angle` about `axis`, mirroring
// math::Quat::FromAxisAngle (half-angle, normalized axis).
__device__ math::Quat QuatFromAxisAngle(math::Vec3 axis, float angle) {
    const float length_sq = Dot3(axis, axis);
    if (length_sq <= 1.0e-12f) {
        return QuatIdentity();
    }
    const math::Vec3 a = ScaleVec(axis, rsqrtf(length_sq));
    const float half = angle * 0.5f;
    const float s = sinf(half);
    const float c = cosf(half);
    return MakeQuat(c, a.x * s, a.y * s, a.z * s);
}

// Rotates `v` by (near-)unit quaternion `q` (q * v * q^-1 short form). Mirrors
// the device RotateByQuat in articulation_jacobian.cu / math::Quat::Rotate.
__device__ math::Vec3 RotateByQuat(math::Quat q, math::Vec3 v) {
    const float norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (norm_sq > 1.0e-12f) {
        const float inv_norm = rsqrtf(norm_sq);
        q.w *= inv_norm;
        q.x *= inv_norm;
        q.y *= inv_norm;
        q.z *= inv_norm;
    }
    const math::Vec3 qv{q.x, q.y, q.z};
    const math::Vec3 t = ScaleVec(Cross3(qv, v), 2.0f);
    const math::Vec3 wt = ScaleVec(t, q.w);
    const math::Vec3 qvt = Cross3(qv, t);
    return AddVec(AddVec(v, wt), qvt);
}

// Composes lhs o rhs (apply rhs first), mirroring math::Transform::operator*.
__device__ math::Transform ComposeTransform(const math::Transform& lhs,
                                            const math::Transform& rhs) {
    math::Transform out;
    out.position = AddVec(lhs.position, RotateByQuat(lhs.rotation, rhs.position));
    out.rotation = QuatNormalize(QuatMul(lhs.rotation, rhs.rotation));
    return out;
}

// SE(3) parent->child relative transform for one link. Mirrors JointTransform
// in featherstone_aba.cu (lines 349-365), but in position+quaternion form
// rather than the spatial 6x6 motion form (which transposes the rotation):
//   translation = local_pose.position + parent_offset (+ axis * q for prismatic)
//   rotation    = local_pose.rotation * jointRotation(axis, q)
// Revolute rotates about joint_axis by q; prismatic/fixed have identity rotation.
__device__ math::Transform RelativeTransform(const ArticulationDeviceState& state,
                                             uint32_t link) {
    const ArticulationJointType type = state.joint_type[link];
    const math::Vec3 axis = state.joint_axis[link];
    const math::Transform local_pose = state.link_local_pose[link];
    const math::Vec3 parent_offset = state.parent_offset[link];

    math::Transform relative;
    relative.position = AddVec(local_pose.position, parent_offset);
    relative.rotation = local_pose.rotation;
    if (type == ArticulationJointType::Revolute) {
        relative.rotation =
            QuatNormalize(QuatMul(local_pose.rotation,
                                  QuatFromAxisAngle(axis, state.q[link])));
    } else if (type == ArticulationJointType::Prismatic) {
        relative.position =
            AddVec(relative.position, ScaleVec(axis, state.q[link]));
    }
    return relative;
}

// (A) One block per articulation, single lane. Forward pass over the links in
// local order; parent-local-index < child is guaranteed by the cooker, so each
// parent's world pose is already final when its child is processed.
__global__ void UpdateWorldLinkPosesKernel(ArticulationDeviceState state,
                                           math::Transform* out_world_pose) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        const math::Transform relative = RelativeTransform(state, link);
        const uint32_t parent_local = state.parent_link[link];
        if (parent_local == kInvalidLink) {
            // Root: world = identity o relative = relative.
            out_world_pose[link] = relative;
        } else {
            out_world_pose[link] =
                ComposeTransform(out_world_pose[offset + parent_local], relative);
        }
    }
}

// (B) One thread per environment. Walks the env's feet in fixed order, compacts
// penetrating feet into the env's own [env*stride, env*stride+count) slots, and
// writes the env's contact count. No atomics; each env owns its slot block.
__global__ void DetectFootGroundContactsKernel(const math::Transform* world_pose,
                                               const FootShape* feet,
                                               uint32_t foot_count,
                                               uint32_t env_count,
                                               uint32_t base_link_count,
                                               float ground_height,
                                               uint32_t* out_contact_link,
                                               math::Vec3* out_contact_point,
                                               math::Vec3* out_contact_normal,
                                               float* out_contact_depth,
                                               uint32_t* out_contact_count) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= env_count) {
        return;
    }

    const uint32_t base_slot = env * kMaxFootContactsPerEnv;
    uint32_t count = 0u;
    for (uint32_t foot = 0u; foot < foot_count; ++foot) {
        const FootShape shape = feet[foot];
        const uint32_t link = env * base_link_count + shape.calf_local_link;
        const math::Transform calf = world_pose[link];
        // Foot sphere center in world = calf_world o local_offset.
        const math::Vec3 center =
            AddVec(calf.position, RotateByQuat(calf.rotation, shape.local_offset));
        const float depth = (ground_height + shape.radius) - center.z;
        if (depth > 0.0f) {
            const uint32_t slot = base_slot + count;
            out_contact_link[slot] = link;
            // Contact point: foot center projected onto the ground surface.
            out_contact_point[slot] = {center.x, center.y, ground_height};
            out_contact_normal[slot] = {0.0f, 0.0f, 1.0f};
            out_contact_depth[slot] = depth;
            ++count;
        }
    }
    // Zero-fill the env's unused trailing slots so stale data never leaks.
    for (uint32_t slot = count; slot < kMaxFootContactsPerEnv; ++slot) {
        const uint32_t index = base_slot + slot;
        out_contact_link[index] = kInvalidLink;
        out_contact_point[index] = {0.0f, 0.0f, 0.0f};
        out_contact_normal[index] = {0.0f, 0.0f, 0.0f};
        out_contact_depth[index] = 0.0f;
    }
    out_contact_count[env] = count;
}

// ---------------------------------------------------------------------------
// p01-F T4 -- CRBA joint-space inertia M, LDL^T inverse, contact m_eff
// ---------------------------------------------------------------------------
//
// The spatial-algebra helpers below are minimal re-implementations of the
// (anonymous-namespace, hence unreachable) device helpers in
// featherstone_aba.cu, kept byte-for-byte equivalent in op order so this path
// reproduces ABA Pass-2's proven convention:
//   * Dot6                   <- featherstone_aba.cu:61
//   * Copy36                 <- featherstone_aba.cu:258
//   * Mat66MulVec6Local      <- featherstone_aba.cu:248 (Mat66MulVec6)
//   * TransformInertiaToParentLocal  <- featherstone_aba.cu:224 (X^T Ic X)
//   * TransformForceTransposeLocal   <- featherstone_aba.cu:212 (X^T f)
// They are local copies (not a shared header) because this task scopes edits to
// articulation_contacts.{cu,hpp}.

constexpr float kMinDiagonal = 1.0e-6f;

__device__ float Dot6Local(const float* a, const float* b) {
    float sum = 0.0f;
    for (uint32_t i = 0u; i < 6u; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

__device__ void Copy36Local(const float* src, float* dst) {
    for (uint32_t i = 0u; i < 36u; ++i) {
        dst[i] = src[i];
    }
}

__device__ void Mat66MulVec6Local(const float* matrix, const float* vector, float* out) {
    for (uint32_t row = 0u; row < 6u; ++row) {
        float value = 0.0f;
        for (uint32_t col = 0u; col < 6u; ++col) {
            value += matrix[row * 6u + col] * vector[col];
        }
        out[row] = value;
    }
}

// parent_delta = X^T * child_inertia * X (composite-inertia push to parent).
__device__ void TransformInertiaToParentLocal(const LinkSpatialTransform& transform,
                                              const float* child_inertia,
                                              float* parent_delta) {
    float temp[36];
    for (uint32_t row = 0u; row < 6u; ++row) {
        for (uint32_t col = 0u; col < 6u; ++col) {
            float value = 0.0f;
            for (uint32_t k = 0u; k < 6u; ++k) {
                value += child_inertia[row * 6u + k] * transform.X[k * 6u + col];
            }
            temp[row * 6u + col] = value;
        }
    }
    for (uint32_t row = 0u; row < 6u; ++row) {
        for (uint32_t col = 0u; col < 6u; ++col) {
            float value = 0.0f;
            for (uint32_t k = 0u; k < 6u; ++k) {
                value += transform.X[k * 6u + row] * temp[k * 6u + col];
            }
            parent_delta[row * 6u + col] = value;
        }
    }
}

// out = X^T * in (push a spatial force from this link's frame to its parent).
__device__ void TransformForceTransposeLocal(const LinkSpatialTransform& transform,
                                            const float* in,
                                            float* out) {
    for (uint32_t row = 0u; row < 6u; ++row) {
        float value = 0.0f;
        for (uint32_t col = 0u; col < 6u; ++col) {
            value += transform.X[col * 6u + row] * in[col];
        }
        out[row] = value;
    }
}

__device__ uint32_t JointDofCountDevice(ArticulationJointType type) {
    switch (type) {
        case ArticulationJointType::Revolute:
        case ArticulationJointType::Prismatic:
            return 1u;
        case ArticulationJointType::Fixed:
            return 0u;
    }
    return 0u;
}

// Local dof_index of a link: base-inclusive prefix sum of per-joint DOF counts
// over [offset, link). Mirrors the chain Jacobian's dof_index exactly, so M's
// rows/cols line up with J's columns.
__device__ uint32_t LocalDofIndex(const ArticulationDeviceState& state,
                                  uint32_t offset,
                                  uint32_t link) {
    uint32_t index = 0u;
    for (uint32_t k = offset; k < link; ++k) {
        index += JointDofCountDevice(state.joint_type[k]);
    }
    return index;
}

// (1) Dense symmetric M per articulation via CRBA. One block, single lane.
__global__ void ComputeArticulationInertiaMKernel(ArticulationDeviceState state,
                                                  uint32_t max_dof,
                                                  LinkSpatialInertia* composite,
                                                  float* out_inertia_M) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    const size_t tile_stride = static_cast<size_t>(max_dof) * max_dof;
    float* const M = out_inertia_M + static_cast<size_t>(articulation) * tile_stride;

    // Zero the whole tile (leading dof_count block filled below; padding stays 0).
    for (size_t i = 0u; i < tile_stride; ++i) {
        M[i] = 0.0f;
    }

    // Seed composite inertia from the rigid-body spatial inertia. We must NOT
    // read link_articulated_I (ABA Pass-2 clobbers it); link_inertia is the
    // pristine rigid-body inertia in each link's own frame.
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        Copy36Local(state.link_inertia[link].I, composite[link].I);
    }

    // Composite leaf->root: Ic[parent] += X^T Ic[link] X (ABA Pass-2 without the
    // articulated-inertia reduction). Fixed-DOF links still propagate their mass.
    for (uint32_t reverse = count; reverse > 0u; --reverse) {
        const uint32_t link = offset + reverse - 1u;
        const uint32_t parent_local = state.parent_link[link];
        if (parent_local == kInvalidLink) {
            continue;
        }
        float parent_delta[36];
        TransformInertiaToParentLocal(state.link_xup[link], composite[link].I, parent_delta);
        const uint32_t parent_link = offset + parent_local;
        for (uint32_t i = 0u; i < 36u; ++i) {
            composite[parent_link].I[i] += parent_delta[i];
        }
    }

    // For each non-fixed joint i: F = Ic_i S_i (force in i's frame). M[i][i] =
    // S_i^T F. Then walk to the root pushing F up by X^T at each step; at every
    // non-fixed ancestor j, M[i][j] = M[j][i] = S_j^T F (F now in j's frame).
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        if (JointDofCountDevice(state.joint_type[link]) == 0u) {
            continue;
        }
        const uint32_t dof_i = LocalDofIndex(state, offset, link);
        if (dof_i >= max_dof) {
            continue;
        }

        float force[6];
        Mat66MulVec6Local(composite[link].I, state.joint_motion_subspace[link].s, force);

        float diagonal = Dot6Local(state.joint_motion_subspace[link].s, force);
        // Reflected rotor inertia + floor (matches ABA Pass-2 diagonal guard).
        diagonal += state.joint_armature[link];
        diagonal += kInertiaDiagonalEpsilon;
        M[static_cast<size_t>(dof_i) * max_dof + dof_i] = diagonal;

        uint32_t walk = link;
        while (true) {
            const uint32_t parent_local = state.parent_link[walk];
            if (parent_local == kInvalidLink) {
                break;
            }
            // Push F from `walk`'s frame to its parent's frame.
            float pushed[6];
            TransformForceTransposeLocal(state.link_xup[walk], force, pushed);
            for (uint32_t i = 0u; i < 6u; ++i) {
                force[i] = pushed[i];
            }
            walk = offset + parent_local;
            if (JointDofCountDevice(state.joint_type[walk]) == 0u) {
                continue;
            }
            const uint32_t dof_j = LocalDofIndex(state, offset, walk);
            if (dof_j >= max_dof) {
                continue;
            }
            const float entry = Dot6Local(state.joint_motion_subspace[walk].s, force);
            M[static_cast<size_t>(dof_i) * max_dof + dof_j] = entry;
            M[static_cast<size_t>(dof_j) * max_dof + dof_i] = entry;
        }
    }
}

// (2) Per-articulation unpivoted LDL^T of the leading dof_count block, then the
// explicit symmetric inverse. One block, single lane. Local scratch is sized for
// the Go2's 12 DOF (and any articulation up to kMaxFactorDof).
constexpr uint32_t kMaxFactorDof = 18u;  // 6-DOF floating base + 12 revolute.

__global__ void FactorArticulationInertiaMKernel(ArticulationDeviceState state,
                                                 uint32_t max_dof,
                                                 const float* inertia_M,
                                                 float* out_inertia_M_inv) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    uint32_t dof = 0u;
    for (uint32_t local = 0u; local < count; ++local) {
        dof += JointDofCountDevice(state.joint_type[offset + local]);
    }
    if (dof > kMaxFactorDof) {
        dof = kMaxFactorDof;
    }

    const size_t tile_stride = static_cast<size_t>(max_dof) * max_dof;
    const float* const M = inertia_M + static_cast<size_t>(articulation) * tile_stride;
    float* const Minv = out_inertia_M_inv + static_cast<size_t>(articulation) * tile_stride;
    for (size_t i = 0u; i < tile_stride; ++i) {
        Minv[i] = 0.0f;
    }
    if (dof == 0u) {
        return;
    }

    // Copy the leading dof x dof block into dense local scratch.
    float a[kMaxFactorDof * kMaxFactorDof];
    for (uint32_t r = 0u; r < dof; ++r) {
        for (uint32_t c = 0u; c < dof; ++c) {
            a[r * kMaxFactorDof + c] = M[static_cast<size_t>(r) * max_dof + c];
        }
    }

    // Unpivoted LDL^T: A = L D L^T, L unit-lower-triangular, D diagonal.
    // L stored in the strict lower triangle of `a`, D on its diagonal.
    float d[kMaxFactorDof];
    for (uint32_t j = 0u; j < dof; ++j) {
        float djj = a[j * kMaxFactorDof + j];
        for (uint32_t k = 0u; k < j; ++k) {
            djj -= a[j * kMaxFactorDof + k] * a[j * kMaxFactorDof + k] * d[k];
        }
        if (djj < kMinDiagonal) {
            djj = kMinDiagonal;  // SPD floor; guards a degenerate config.
        }
        d[j] = djj;
        for (uint32_t i = j + 1u; i < dof; ++i) {
            float lij = a[i * kMaxFactorDof + j];
            for (uint32_t k = 0u; k < j; ++k) {
                lij -= a[i * kMaxFactorDof + k] * a[j * kMaxFactorDof + k] * d[k];
            }
            a[i * kMaxFactorDof + j] = lij / djj;
        }
    }

    // Solve A x = e_col for each identity column to form M^-1 (symmetric).
    for (uint32_t col = 0u; col < dof; ++col) {
        float y[kMaxFactorDof];
        // Forward solve L y = e_col.
        for (uint32_t i = 0u; i < dof; ++i) {
            float value = (i == col) ? 1.0f : 0.0f;
            for (uint32_t k = 0u; k < i; ++k) {
                value -= a[i * kMaxFactorDof + k] * y[k];
            }
            y[i] = value;
        }
        // Diagonal solve D z = y (in place).
        for (uint32_t i = 0u; i < dof; ++i) {
            y[i] /= d[i];
        }
        // Backward solve L^T x = z.
        float x[kMaxFactorDof];
        for (uint32_t ii = dof; ii > 0u; --ii) {
            const uint32_t i = ii - 1u;
            float value = y[i];
            for (uint32_t k = i + 1u; k < dof; ++k) {
                value -= a[k * kMaxFactorDof + i] * x[k];
            }
            x[i] = value;
        }
        for (uint32_t r = 0u; r < dof; ++r) {
            Minv[static_cast<size_t>(r) * max_dof + col] = x[r];
        }
    }
}

// (3) Per-contact effective mass m_eff = 1 / (J M^-1 J^T). One thread / contact.
__global__ void ComputeContactEffectiveMassKernel(ArticulationDeviceState state,
                                                  const uint32_t* contact_link_indices,
                                                  const float* chain_jacobian,
                                                  const float* inertia_M_inv,
                                                  uint32_t contact_count,
                                                  uint32_t dof_stride,
                                                  float* out_effective_mass) {
    const uint32_t contact = blockIdx.x * blockDim.x + threadIdx.x;
    if (contact >= contact_count) {
        return;
    }

    const uint32_t link = contact_link_indices[contact];
    if (link >= state.total_link_count) {
        out_effective_mass[contact] = 0.0f;
        return;
    }

    const uint32_t articulation = state.link_to_articulation[link];
    const size_t tile_stride = static_cast<size_t>(dof_stride) * dof_stride;
    const float* const Minv =
        inertia_M_inv + static_cast<size_t>(articulation) * tile_stride;
    const float* const J =
        chain_jacobian + static_cast<size_t>(contact) * dof_stride;

    // denom = J M^-1 J^T. Padding columns of J are zero, so iterating the full
    // stride is safe even though M^-1's padding rows/cols are zero.
    float denom = 0.0f;
    for (uint32_t r = 0u; r < dof_stride; ++r) {
        float row = 0.0f;
        for (uint32_t c = 0u; c < dof_stride; ++c) {
            row += Minv[static_cast<size_t>(r) * dof_stride + c] * J[c];
        }
        denom += J[r] * row;
    }

    out_effective_mass[contact] = 1.0f / fmaxf(denom, kEffectiveMassDenomEpsilon);
}

// ---------------------------------------------------------------------------
// p01-F T5 -- articulated contact rows + block-PGS solve (contact<->ABA coupling)
// ---------------------------------------------------------------------------

// Branch-stable orthonormal tangent basis (t1,t2) for a unit normal n. Picks the
// world axis least aligned with n as the reference, so the same n ALWAYS yields
// the same basis (friction-row determinism). t1 = normalize(ref - (ref.n) n),
// t2 = n x t1.
__device__ void TangentBasis(math::Vec3 n, math::Vec3* t1, math::Vec3* t2) {
    // Fixed reference selection: prefer +X unless n is nearly parallel to it,
    // then +Y. The comparison is on |n.x| vs a fixed threshold => no data-
    // dependent tie that could flip between identical inputs.
    math::Vec3 reference = {1.0f, 0.0f, 0.0f};
    if (fabsf(n.x) > 0.9f) {
        reference = {0.0f, 1.0f, 0.0f};
    }
    const float proj = Dot3(reference, n);
    math::Vec3 tangent_a = {reference.x - proj * n.x,
                            reference.y - proj * n.y,
                            reference.z - proj * n.z};
    const float len_sq = Dot3(tangent_a, tangent_a);
    if (len_sq > 1.0e-12f) {
        tangent_a = ScaleVec(tangent_a, rsqrtf(len_sq));
    } else {
        tangent_a = {1.0f, 0.0f, 0.0f};
    }
    const math::Vec3 tangent_b = Cross3(n, tangent_a);
    *t1 = tangent_a;
    *t2 = tangent_b;
}

// One thread per contact slot (env-major, fixed stride). Builds (t1,t2) for each
// active contact's normal; inactive slots get zero vectors.
__global__ void ComputeContactTangentBasisKernel(const uint32_t* contact_link,
                                                 const math::Vec3* contact_normal,
                                                 uint32_t slot_count,
                                                 math::Vec3* out_tangent1,
                                                 math::Vec3* out_tangent2) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slot_count) {
        return;
    }
    if (contact_link[slot] == kInvalidLink) {
        out_tangent1[slot] = {0.0f, 0.0f, 0.0f};
        out_tangent2[slot] = {0.0f, 0.0f, 0.0f};
        return;
    }
    const math::Vec3 normal = NormalizeOrUpLocal(contact_normal[slot]);
    math::Vec3 t1;
    math::Vec3 t2;
    TangentBasis(normal, &t1, &t2);
    out_tangent1[slot] = t1;
    out_tangent2[slot] = t2;
}

// One thread per contact slot. Assembles the per-contact row set (normal + the
// two friction tangents) from the detection + m_eff inputs. Inactive slots get a
// kRowInactive row the solver skips.
__global__ void AssembleArticulatedContactRowsKernel(
    ArticulationDeviceState state,
    const uint32_t* contact_link,
    const math::Vec3* contact_normal,
    const float* contact_depth,
    const float* normal_effective_mass,
    const float* tangent1_effective_mass,
    const float* tangent2_effective_mass,
    uint32_t slot_count,
    ArticulatedContactRow* out_rows) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slot_count) {
        return;
    }

    ArticulatedContactRow row;
    const uint32_t link = contact_link[slot];
    if (link == kInvalidLink || link >= state.total_link_count) {
        row.row_type = ContactRowType::kRowInactive;
        row.articulation = kInvalidLink;
        row.contact_link = kInvalidLink;
        out_rows[slot] = row;
        return;
    }

    const math::Vec3 normal = NormalizeOrUpLocal(contact_normal[slot]);
    math::Vec3 t1;
    math::Vec3 t2;
    TangentBasis(normal, &t1, &t2);

    row.row_type = ContactRowType::kRowNormal;
    row.articulation = state.link_to_articulation[link];
    row.contact_link = link;
    row.effective_mass = normal_effective_mass[slot];
    row.effective_mass_t1 = tangent1_effective_mass[slot];
    row.effective_mass_t2 = tangent2_effective_mass[slot];
    row.depth = contact_depth[slot];
    row.normal = normal;
    row.tangent1 = t1;
    row.tangent2 = t2;
    out_rows[slot] = row;
}

// Fused block-per-articulation PGS solve + apply. One block, single lane. Reads
// the env's fixed-stride row slots, seeds qdot_work from state.qdot (= qdot_free),
// sweeps the rows in fixed order, and writes the corrected qdot_work back. The
// dof column -> global qdot index map is the base-inclusive prefix sum over the
// articulation's links (== T3 dof_index == T4 LocalDofIndex), realized as the
// dof_to_link table built once per block.
__global__ void SolveArticulatedContactRowsKernel(ArticulationDeviceState state,
                                                  const ArticulatedContactRow* rows,
                                                  const float* normal_jacobian,
                                                  const float* tangent1_jacobian,
                                                  const float* tangent2_jacobian,
                                                  const float* inertia_M_inv,
                                                  uint32_t dof_stride,
                                                  float dt,
                                                  float friction_coefficient,
                                                  float baumgarte_max_velocity,
                                                  float* inout_lambda) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];

    // dof_to_link[k] = global link of the k-th DOF (base-inclusive prefix sum).
    uint32_t dof_to_link[kMaxContactSolverDof];
    uint32_t dof = 0u;
    for (uint32_t local = 0u; local < count && dof < kMaxContactSolverDof; ++local) {
        const uint32_t link = offset + local;
        if (JointDofCountDevice(state.joint_type[link]) != 0u) {
            dof_to_link[dof] = link;
            ++dof;
        }
    }
    if (dof == 0u || dof_stride == 0u) {
        return;
    }

    // Working joint-velocity vector, seeded from qdot_free (current state.qdot).
    float qdot_work[kMaxContactSolverDof];
    for (uint32_t k = 0u; k < dof; ++k) {
        qdot_work[k] = state.qdot[dof_to_link[k]];
    }

    const size_t tile_stride = static_cast<size_t>(dof_stride) * dof_stride;
    const float* const Minv =
        inertia_M_inv + static_cast<size_t>(articulation) * tile_stride;

    // This articulation's contact slots (env-major; env == articulation in the
    // 1-articulation-per-env design). lambda_n[slot] accumulates the normal
    // impulse used by the friction cone.
    const uint32_t slot_base = articulation * kMaxFootContactsPerEnv;
    float lambda_n[kMaxFootContactsPerEnv];
    float lambda_t1[kMaxFootContactsPerEnv];
    float lambda_t2[kMaxFootContactsPerEnv];
    for (uint32_t s = 0u; s < kMaxFootContactsPerEnv; ++s) {
        // Warm-start from the persisted buffer when present AND the same link
        // still occupies the slot (else cold start at 0). Warm-start affects
        // convergence only -- never determinism (the determinism gate runs cold).
        float ln = 0.0f;
        float lt1 = 0.0f;
        float lt2 = 0.0f;
        if (inout_lambda != nullptr) {
            const size_t base = static_cast<size_t>(slot_base + s) * 3u;
            ln = inout_lambda[base + 0u];
            lt1 = inout_lambda[base + 1u];
            lt2 = inout_lambda[base + 2u];
        }
        lambda_n[s] = ln;
        lambda_t1[s] = lt1;
        lambda_t2[s] = lt2;
    }

    const float inv_dt = (dt > 0.0f) ? (1.0f / dt) : 0.0f;
    const float mu = fmaxf(friction_coefficient, 0.0f);

    // Apply any warm-started impulse so qdot_work reflects the seed lambda before
    // the first sweep (qdot_work += M^-1 J^T * actual). Inlined (no device lambda)
    // so the local qdot_work array is updated in place unambiguously.
    for (uint32_t s = 0u; s < kMaxFootContactsPerEnv; ++s) {
        const ArticulatedContactRow row = rows[slot_base + s];
        if (row.row_type != ContactRowType::kRowNormal ||
            row.articulation != articulation) {
            continue;
        }
        const size_t row_base = static_cast<size_t>(slot_base + s) * dof_stride;
        const float* const jrows[3] = {normal_jacobian + row_base,
                                       tangent1_jacobian + row_base,
                                       tangent2_jacobian + row_base};
        const float seeds[3] = {lambda_n[s], lambda_t1[s], lambda_t2[s]};
        for (uint32_t which = 0u; which < 3u; ++which) {
            const float actual = seeds[which];
            if (actual == 0.0f) {
                continue;
            }
            const float* const jac_row = jrows[which];
            for (uint32_t r = 0u; r < dof; ++r) {
                float acc = 0.0f;
                const float* minv_row = Minv + static_cast<size_t>(r) * dof_stride;
                for (uint32_t c = 0u; c < dof; ++c) {
                    acc += minv_row[c] * jac_row[c];
                }
                qdot_work[r] += acc * actual;
            }
        }
    }

    for (uint32_t iter = 0u; iter < kContactSolverIterations; ++iter) {
        // -- Normal rows first, fixed slot order. ---------------------------
        for (uint32_t s = 0u; s < kMaxFootContactsPerEnv; ++s) {
            const ArticulatedContactRow row = rows[slot_base + s];
            if (row.row_type != ContactRowType::kRowNormal ||
                row.articulation != articulation) {
                continue;
            }
            const float* const jn =
                normal_jacobian + static_cast<size_t>(slot_base + s) * dof_stride;
            const float bias = fminf(
                kBaumgarteBeta * inv_dt * fmaxf(row.depth - kPenetrationSlop, 0.0f),
                baumgarte_max_velocity);
            float jv = 0.0f;
            for (uint32_t k = 0u; k < dof; ++k) {
                jv += jn[k] * qdot_work[k];
            }
            const float delta = row.effective_mass * (bias - jv);
            const float old_lambda = lambda_n[s];
            // Normal impulse: lower = 0, upper = +inf.
            float new_lambda = old_lambda + delta;
            if (new_lambda < 0.0f) {
                new_lambda = 0.0f;
            }
            const float actual = new_lambda - old_lambda;
            lambda_n[s] = new_lambda;
            if (actual != 0.0f) {
                for (uint32_t r = 0u; r < dof; ++r) {
                    float acc = 0.0f;
                    const float* minv_row = Minv + static_cast<size_t>(r) * dof_stride;
                    for (uint32_t c = 0u; c < dof; ++c) {
                        acc += minv_row[c] * jn[c];
                    }
                    qdot_work[r] += acc * actual;
                }
            }
        }
        // -- Friction tangent rows, fixed slot order. -----------------------
        for (uint32_t s = 0u; s < kMaxFootContactsPerEnv; ++s) {
            const ArticulatedContactRow row = rows[slot_base + s];
            if (row.row_type != ContactRowType::kRowNormal ||
                row.articulation != articulation) {
                continue;
            }
            const float limit = mu * fmaxf(lambda_n[s], 0.0f);
            const size_t row_base = static_cast<size_t>(slot_base + s) * dof_stride;
            // Tangent 1 then tangent 2, fixed order.
            for (uint32_t which = 0u; which < 2u; ++which) {
                const float* const jt =
                    (which == 0u ? tangent1_jacobian : tangent2_jacobian) + row_base;
                const float meff = (which == 0u) ? row.effective_mass_t1
                                                 : row.effective_mass_t2;
                float jv = 0.0f;
                for (uint32_t k = 0u; k < dof; ++k) {
                    jv += jt[k] * qdot_work[k];
                }
                const float delta = meff * (0.0f - jv);
                const float old_lambda = (which == 0u) ? lambda_t1[s] : lambda_t2[s];
                float new_lambda = old_lambda + delta;
                new_lambda = fminf(fmaxf(new_lambda, -limit), limit);
                const float actual = new_lambda - old_lambda;
                if (which == 0u) {
                    lambda_t1[s] = new_lambda;
                } else {
                    lambda_t2[s] = new_lambda;
                }
                if (actual != 0.0f) {
                    for (uint32_t r = 0u; r < dof; ++r) {
                        float acc = 0.0f;
                        const float* minv_row = Minv + static_cast<size_t>(r) * dof_stride;
                        for (uint32_t c = 0u; c < dof; ++c) {
                            acc += minv_row[c] * jt[c];
                        }
                        qdot_work[r] += acc * actual;
                    }
                }
            }
        }
    }

    // Final normal-row sweep. The interleaved normal+friction PGS converges to the
    // correct boxed-friction LCP fixed point, but the last update in the loop is a
    // FRICTION row, so qdot_work leaves the loop with the normal velocities slightly
    // perturbed by friction's final application (the system is strongly coupled:
    // J_n M^-1 J_t^T is O(1) here for the bent Go2 stance). The normal constraint is
    // a single row, so one more normal-only sweep re-imposes jv_n = bias EXACTLY
    // (delta = m_eff*(bias - jv) drives the unclamped row to the target in one step)
    // without disturbing the converged friction impulses. This makes the returned
    // qdot satisfy the non-penetration constraint to machine precision; the friction
    // cone (|lambda_t| <= mu*lambda_n) is preserved because lambda_n only grows here.
    for (uint32_t s = 0u; s < kMaxFootContactsPerEnv; ++s) {
        const ArticulatedContactRow row = rows[slot_base + s];
        if (row.row_type != ContactRowType::kRowNormal ||
            row.articulation != articulation) {
            continue;
        }
        const float* const jn =
            normal_jacobian + static_cast<size_t>(slot_base + s) * dof_stride;
        const float bias = fminf(
            kBaumgarteBeta * inv_dt * fmaxf(row.depth - kPenetrationSlop, 0.0f),
            baumgarte_max_velocity);
        float jv = 0.0f;
        for (uint32_t k = 0u; k < dof; ++k) {
            jv += jn[k] * qdot_work[k];
        }
        const float delta = row.effective_mass * (bias - jv);
        const float old_lambda = lambda_n[s];
        float new_lambda = old_lambda + delta;
        if (new_lambda < 0.0f) {
            new_lambda = 0.0f;
        }
        const float actual = new_lambda - old_lambda;
        lambda_n[s] = new_lambda;
        if (actual != 0.0f) {
            for (uint32_t r = 0u; r < dof; ++r) {
                float acc = 0.0f;
                const float* minv_row = Minv + static_cast<size_t>(r) * dof_stride;
                for (uint32_t c = 0u; c < dof; ++c) {
                    acc += minv_row[c] * jn[c];
                }
                qdot_work[r] += acc * actual;
            }
        }
    }

#ifdef NUKA_T5_KERNEL_DEBUG
    if (articulation == 0u) {
        printf("[kdbg] dof=%u Minv00=%.4f Minv11=%.4f qdot_work[0..2]=%.6g %.6g %.6g "
               "lambda_n0=%.4f\n",
               dof, Minv[0], Minv[static_cast<size_t>(dof_stride) + 1u],
               qdot_work[0], qdot_work[1], qdot_work[2], lambda_n[0]);
    }
#endif

    // Write the corrected joint velocity back for every DOF of the articulation.
    for (uint32_t k = 0u; k < dof; ++k) {
        state.qdot[dof_to_link[k]] = qdot_work[k];
    }

    // Persist this step's impulses for the next step's warm start.
    if (inout_lambda != nullptr) {
        for (uint32_t s = 0u; s < kMaxFootContactsPerEnv; ++s) {
            const size_t base = static_cast<size_t>(slot_base + s) * 3u;
            inout_lambda[base + 0u] = lambda_n[s];
            inout_lambda[base + 1u] = lambda_t1[s];
            inout_lambda[base + 2u] = lambda_t2[s];
        }
    }
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

} // namespace

void UpdateWorldLinkPoses(const phi::DeviceContext& context,
                          ArticulationDeviceState state,
                          math::Transform* out_world_pose) {
    if (state.articulation_count == 0u || state.total_link_count == 0u) {
        return;
    }
    if (out_world_pose == nullptr) {
        throw std::runtime_error("UpdateWorldLinkPoses requires an output buffer");
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    // One block per articulation; the forward pass runs on a single lane (the
    // parent->child dependency is sequential within an articulation).
    UpdateWorldLinkPosesKernel<<<state.articulation_count, 32u, 0u, stream>>>(
        state, out_world_pose);
    CheckCuda(cudaGetLastError(), "UpdateWorldLinkPosesKernel launch");
}

void DetectFootGroundContacts(const phi::DeviceContext& context,
                              const math::Transform* world_pose,
                              const FootShape* feet,
                              uint32_t foot_count,
                              uint32_t env_count,
                              uint32_t base_link_count,
                              float ground_height,
                              uint32_t* out_contact_link,
                              math::Vec3* out_contact_point,
                              math::Vec3* out_contact_normal,
                              float* out_contact_depth,
                              uint32_t* out_contact_count) {
    if (env_count == 0u) {
        return;
    }
    if (foot_count > kMaxFootContactsPerEnv) {
        throw std::runtime_error(
            "DetectFootGroundContacts: foot_count exceeds kMaxFootContactsPerEnv");
    }
    if (world_pose == nullptr || feet == nullptr || out_contact_link == nullptr ||
        out_contact_point == nullptr || out_contact_normal == nullptr ||
        out_contact_depth == nullptr || out_contact_count == nullptr) {
        throw std::runtime_error(
            "DetectFootGroundContacts requires device input and output buffers");
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    constexpr uint32_t kBlockSize = 128u;
    const uint32_t block_count = (env_count + kBlockSize - 1u) / kBlockSize;
    DetectFootGroundContactsKernel<<<block_count, kBlockSize, 0u, stream>>>(
        world_pose,
        feet,
        foot_count,
        env_count,
        base_link_count,
        ground_height,
        out_contact_link,
        out_contact_point,
        out_contact_normal,
        out_contact_depth,
        out_contact_count);
    CheckCuda(cudaGetLastError(), "DetectFootGroundContactsKernel launch");
}

uint32_t ArticulationJointDofCount(ArticulationJointType type) {
    switch (type) {
        case ArticulationJointType::Revolute:
        case ArticulationJointType::Prismatic:
            return 1u;
        case ArticulationJointType::Fixed:
            return 0u;
    }
    return 0u;
}

uint32_t ArticulationDofCount(const ArticulationHostState& host, uint32_t articulation) {
    if (articulation >= host.ArticulationCount()) {
        return 0u;
    }
    const uint32_t offset = host.articulation_link_offset[articulation];
    const uint32_t count = host.articulation_link_count[articulation];
    uint32_t dof = 0u;
    for (uint32_t local = 0u; local < count; ++local) {
        dof += ArticulationJointDofCount(host.joint_type[offset + local]);
    }
    return dof;
}

void ComputeArticulationInertiaM(const phi::DeviceContext& context,
                                 ArticulationDeviceState state,
                                 uint32_t max_dof,
                                 LinkSpatialInertia* composite_inertia_scratch,
                                 float* out_inertia_M) {
    if (state.articulation_count == 0u || state.total_link_count == 0u || max_dof == 0u) {
        return;
    }
    if (composite_inertia_scratch == nullptr || out_inertia_M == nullptr) {
        throw std::runtime_error(
            "ComputeArticulationInertiaM requires device scratch and output buffers");
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    ComputeArticulationInertiaMKernel<<<state.articulation_count, 32u, 0u, stream>>>(
        state, max_dof, composite_inertia_scratch, out_inertia_M);
    CheckCuda(cudaGetLastError(), "ComputeArticulationInertiaMKernel launch");
}

void FactorArticulationInertiaM(const phi::DeviceContext& context,
                                ArticulationDeviceState state,
                                uint32_t max_dof,
                                const float* inertia_M,
                                float* out_inertia_M_inv) {
    if (state.articulation_count == 0u || max_dof == 0u) {
        return;
    }
    if (inertia_M == nullptr || out_inertia_M_inv == nullptr) {
        throw std::runtime_error(
            "FactorArticulationInertiaM requires device input and output buffers");
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    FactorArticulationInertiaMKernel<<<state.articulation_count, 32u, 0u, stream>>>(
        state, max_dof, inertia_M, out_inertia_M_inv);
    CheckCuda(cudaGetLastError(), "FactorArticulationInertiaMKernel launch");
}

void ComputeContactEffectiveMass(const phi::DeviceContext& context,
                                 ArticulationDeviceState state,
                                 const uint32_t* contact_link_indices,
                                 const float* chain_jacobian,
                                 const float* inertia_M_inv,
                                 uint32_t contact_count,
                                 uint32_t dof_stride,
                                 float* out_effective_mass) {
    if (contact_count == 0u || dof_stride == 0u) {
        return;
    }
    if (contact_link_indices == nullptr || chain_jacobian == nullptr ||
        inertia_M_inv == nullptr || out_effective_mass == nullptr) {
        throw std::runtime_error(
            "ComputeContactEffectiveMass requires device input and output buffers");
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    constexpr uint32_t kBlockSize = 128u;
    const uint32_t block_count = (contact_count + kBlockSize - 1u) / kBlockSize;
    ComputeContactEffectiveMassKernel<<<block_count, kBlockSize, 0u, stream>>>(
        state,
        contact_link_indices,
        chain_jacobian,
        inertia_M_inv,
        contact_count,
        dof_stride,
        out_effective_mass);
    CheckCuda(cudaGetLastError(), "ComputeContactEffectiveMassKernel launch");
}

void ComputeContactTangentBasis(const phi::DeviceContext& context,
                                const uint32_t* contact_link,
                                const math::Vec3* contact_normal,
                                uint32_t env_count,
                                math::Vec3* out_tangent1,
                                math::Vec3* out_tangent2) {
    if (env_count == 0u) {
        return;
    }
    if (contact_link == nullptr || contact_normal == nullptr ||
        out_tangent1 == nullptr || out_tangent2 == nullptr) {
        throw std::runtime_error(
            "ComputeContactTangentBasis requires device input and output buffers");
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    const uint32_t slot_count = env_count * kMaxFootContactsPerEnv;
    constexpr uint32_t kBlockSize = 128u;
    const uint32_t block_count = (slot_count + kBlockSize - 1u) / kBlockSize;
    ComputeContactTangentBasisKernel<<<block_count, kBlockSize, 0u, stream>>>(
        contact_link, contact_normal, slot_count, out_tangent1, out_tangent2);
    CheckCuda(cudaGetLastError(), "ComputeContactTangentBasisKernel launch");
}

void AssembleArticulatedContactRows(const phi::DeviceContext& context,
                                    ArticulationDeviceState state,
                                    const uint32_t* contact_link,
                                    const math::Vec3* contact_normal,
                                    const float* contact_depth,
                                    const float* normal_effective_mass,
                                    const float* tangent1_effective_mass,
                                    const float* tangent2_effective_mass,
                                    uint32_t env_count,
                                    uint32_t dof_stride,
                                    ArticulatedContactRow* out_rows) {
    if (env_count == 0u || dof_stride == 0u) {
        return;
    }
    if (contact_link == nullptr || contact_normal == nullptr ||
        contact_depth == nullptr || normal_effective_mass == nullptr ||
        tangent1_effective_mass == nullptr || tangent2_effective_mass == nullptr ||
        out_rows == nullptr) {
        throw std::runtime_error(
            "AssembleArticulatedContactRows requires device input and output buffers");
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    const uint32_t slot_count = env_count * kMaxFootContactsPerEnv;
    constexpr uint32_t kBlockSize = 128u;
    const uint32_t block_count = (slot_count + kBlockSize - 1u) / kBlockSize;
    AssembleArticulatedContactRowsKernel<<<block_count, kBlockSize, 0u, stream>>>(
        state,
        contact_link,
        contact_normal,
        contact_depth,
        normal_effective_mass,
        tangent1_effective_mass,
        tangent2_effective_mass,
        slot_count,
        out_rows);
    CheckCuda(cudaGetLastError(), "AssembleArticulatedContactRowsKernel launch");
}

void SolveArticulatedContactRows(const phi::DeviceContext& context,
                                 ArticulationDeviceState state,
                                 const ArticulatedContactRow* rows,
                                 const float* normal_jacobian,
                                 const float* tangent1_jacobian,
                                 const float* tangent2_jacobian,
                                 const float* inertia_M_inv,
                                 uint32_t env_count,
                                 uint32_t dof_stride,
                                 float dt,
                                 float* inout_lambda,
                                 float friction_coefficient,
                                 float baumgarte_max_velocity) {
    if (state.articulation_count == 0u || env_count == 0u || dof_stride == 0u) {
        return;
    }
    if (dof_stride > kMaxContactSolverDof) {
        throw std::runtime_error(
            "SolveArticulatedContactRows: dof_stride exceeds kMaxContactSolverDof");
    }
    if (rows == nullptr || normal_jacobian == nullptr ||
        tangent1_jacobian == nullptr || tangent2_jacobian == nullptr ||
        inertia_M_inv == nullptr) {
        throw std::runtime_error(
            "SolveArticulatedContactRows requires device input buffers");
    }

    // env_count is documented for the caller's buffer sizing; the kernel derives
    // each articulation's slot block from its own index (env == articulation in
    // the 1-articulation-per-env design).
    (void)env_count;

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    SolveArticulatedContactRowsKernel<<<state.articulation_count, 32u, 0u, stream>>>(
        state,
        rows,
        normal_jacobian,
        tangent1_jacobian,
        tangent2_jacobian,
        inertia_M_inv,
        dof_stride,
        dt,
        friction_coefficient,
        baumgarte_max_velocity,
        inout_lambda);
    CheckCuda(cudaGetLastError(), "SolveArticulatedContactRowsKernel launch");
}

} // namespace nuka::runtime::articulation
