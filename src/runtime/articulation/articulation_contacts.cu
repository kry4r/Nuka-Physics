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

} // namespace nuka::runtime::articulation
