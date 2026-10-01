#pragma once

#include "math/cuda_vec_ops.cuh"
#include "phi/backend_cuda/ops/kinematics.cuh"

namespace nuka::phi {
namespace {

struct DatPointRef {
    uint32_t owner = 0u;
    uint32_t vertex = 0u;
    bool particle = false;
};

__device__ float DatQuatArc(math::Quat a, math::Quat b) {
    const float dot = fabsf(a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z);
    return 2.0f * acosf(fminf(1.0f, fmaxf(0.0f, dot)));
}

__device__ math::Quat DatSlerp(math::Quat a, math::Quat b, float s) {
    float dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
    if (dot < 0.0f) {
        b = math::gpu::MakeQuat(-b.w, -b.x, -b.y, -b.z);
        dot = -dot;
    }
    float wa = 1.0f - s, wb = s;
    if (dot < 0.9995f) {
        const float angle = acosf(fminf(1.0f, dot));
        const float inverse = 1.0f / sinf(angle);
        wa = sinf((1.0f - s) * angle) * inverse;
        wb = sinf(s * angle) * inverse;
    }
    return math::gpu::QuatNormalizeRsqrt(math::gpu::MakeQuat(
        wa * a.w + wb * b.w, wa * a.x + wb * b.x,
        wa * a.y + wb * b.y, wa * a.z + wb * b.z), 1.0e-12f);
}

__device__ math::Transform DatInterpolatePose(math::Transform a,
                                               math::Transform b, float s) {
    math::Transform result;
    result.position = a.position + (b.position - a.position) * s;
    result.rotation = DatSlerp(a.rotation, b.rotation, s);
    return result;
}

__device__ math::Vec3 DatBodyLocalPoint(const ModelView& model, uint32_t body,
                                        uint32_t vertex) {
    const auto info = model.mesh_surface_info[body];
    const size_t at = (size_t{info.vertex_offset} + vertex) * 3u;
    return {model.hull_verts[at], model.hull_verts[at + 1u], model.hull_verts[at + 2u]};
}

__device__ bool DatBodyLink(const DatTruncateParams& p, const ModelView& model,
                            uint32_t env, uint32_t body, uint32_t* link,
                            math::Transform* local) {
    const uint32_t global = env * p.bodies_per_env + body;
    const uint32_t proxy = model.body_collidable_link
        ? model.body_collidable_link[global] : ~0u;
    if (proxy != ~0u) {
        *link = env * p.links_per_env + proxy;
        *local = model.body_collidable_local[global];
        return true;
    }
    const uint32_t primary = model.body_to_link
        ? model.body_to_link[body] : ~0u;
    if (primary == ~0u) return false;
    *link = env * p.links_per_env + primary;
    *local = model.link_geom_kind[*link] != 0u
        ? model.link_geom_local[*link] : math::Transform{};
    return true;
}

__device__ math::Transform DatBodyPoseAt(const DatTruncateParams& p,
                                         const ModelView& model,
                                         const DataView& data, uint32_t env,
                                         uint32_t body, float s) {
    const uint32_t global = env * p.bodies_per_env + body;
    uint32_t link = ~0u;
    math::Transform local{};
    if (DatBodyLink(p, model, env, body, &link, &local)) {
        const uint32_t articulation = env * p.articulations_per_env +
            model.link_to_articulation[link];
        const uint32_t offset = model.articulation_link_offset[articulation];
        const uint32_t count = model.articulation_link_count[articulation];
        math::Transform result = local;
        for (uint32_t traversed = 0u; traversed < count; ++traversed) {
            const uint32_t parent = model.parent_link[link];
            const auto type = static_cast<ArticulationJointType>(model.joint_type[link]);
            if (parent == ~0u && type == ArticulationJointType::FloatingBase) {
                const auto base = DatInterpolatePose(data.dat_prev_base_pose[articulation],
                                                     data.base_pose[articulation], s);
                return nkops::ComposeFrame(base, result);
            }
            const float q = data.dat_prev_q[link] +
                (data.q[link] - data.dat_prev_q[link]) * s;
            const auto relative = nkops::JointRelativeFrame(type,
                model.joint_axis[link], model.link_local_pose[link],
                model.parent_offset[link], q);
            result = nkops::ComposeFrame(relative, result);
            if (parent == ~0u) return result;
            if (parent >= count) break;
            link = offset + parent;
        }
        return result;
    }
    const uint32_t owner_local = model.body_collidable_body
        ? model.body_collidable_body[global] : ~0u;
    if (owner_local != ~0u) {
        const uint32_t owner = env * p.bodies_per_env + owner_local;
        return nkops::ComposeFrame(DatInterpolatePose(
            data.dat_prev_body_pose[owner], data.body_pose[owner], s),
            model.body_collidable_local[global]);
    }
    return DatInterpolatePose(data.dat_prev_body_pose[global], data.body_pose[global], s);
}

__device__ math::Vec3 DatPointAt(const DatTruncateParams& p, const ModelView& model,
                                 const DataView& data, uint32_t env,
                                 DatPointRef ref, float s) {
    if (ref.particle) {
        const auto start = data.particle_prev_pos[ref.owner];
        return start + (data.particle_pos[ref.owner] - start) * s;
    }
    const auto pose = DatBodyPoseAt(p, model, data, env, ref.owner, s);
    return pose.position + math::gpu::RotateByQuatNormalized(
        pose.rotation, DatBodyLocalPoint(model, ref.owner, ref.vertex));
}

__device__ float DatPointLipschitz(const DatTruncateParams& p, const ModelView& model,
                                   const DataView& data, uint32_t env, DatPointRef ref) {
    if (ref.particle)
        return sqrtf((data.particle_pos[ref.owner] -
                      data.particle_prev_pos[ref.owner]).LengthSq());
    const uint32_t global = env * p.bodies_per_env + ref.owner;
    const auto vertex = DatBodyLocalPoint(model, ref.owner, ref.vertex);
    uint32_t link = ~0u;
    math::Transform local{};
    if (DatBodyLink(p, model, env, ref.owner, &link, &local)) {
        const uint32_t articulation = env * p.articulations_per_env +
            model.link_to_articulation[link];
        const uint32_t offset = model.articulation_link_offset[articulation];
        const uint32_t count = model.articulation_link_count[articulation];
        float radius = sqrtf((local.position + math::gpu::RotateByQuatNormalized(
            local.rotation, vertex)).LengthSq());
        float speed = 0.0f;
        for (uint32_t traversed = 0u; traversed < count; ++traversed) {
            const uint32_t parent = model.parent_link[link];
            const auto type = static_cast<ArticulationJointType>(model.joint_type[link]);
            if (parent == ~0u && type == ArticulationJointType::FloatingBase) {
                const auto a = data.dat_prev_base_pose[articulation];
                const auto b = data.base_pose[articulation];
                return speed + sqrtf((b.position - a.position).LengthSq()) +
                    DatQuatArc(a.rotation, b.rotation) * radius;
            }
            const float delta = fabsf(data.q[link] - data.dat_prev_q[link]);
            if (type == ArticulationJointType::Revolute) speed += delta * radius;
            if (type == ArticulationJointType::Prismatic) speed += delta;
            radius += sqrtf((model.link_local_pose[link].position +
                             model.parent_offset[link]).LengthSq());
            if (type == ArticulationJointType::Prismatic)
                radius += fmaxf(fabsf(data.dat_prev_q[link]), fabsf(data.q[link]));
            if (parent == ~0u) return speed;
            if (parent >= count) break;
            link = offset + parent;
        }
        return INFINITY;
    }
    const uint32_t owner_local = model.body_collidable_body
        ? model.body_collidable_body[global] : ~0u;
    uint32_t owner = global;
    math::Vec3 local_point = vertex;
    if (owner_local != ~0u) {
        owner = env * p.bodies_per_env + owner_local;
        const auto pose = model.body_collidable_local[global];
        local_point = pose.position + math::gpu::RotateByQuatNormalized(
            pose.rotation, vertex);
    }
    const auto a = data.dat_prev_body_pose[owner];
    const auto b = data.body_pose[owner];
    return sqrtf((b.position - a.position).LengthSq()) +
        DatQuatArc(a.rotation, b.rotation) * sqrtf(local_point.LengthSq());
}

__device__ void DatMinOwnerBeta(const DatTruncateParams& p, const ModelView& model,
                                const DataView& data, uint32_t env,
                                DatPointRef ref, float beta) {
    beta = fminf(1.0f, fmaxf(0.0f, beta));
    if (ref.particle) {
        atomicMin(reinterpret_cast<uint32_t*>(data.dat_particle_beta + ref.owner), __float_as_uint(beta));
        return;
    }
    const uint32_t global = env * p.bodies_per_env + ref.owner;
    uint32_t link = ~0u;
    math::Transform local{};
    if (DatBodyLink(p, model, env, ref.owner, &link, &local)) {
        const uint32_t articulation = env * p.articulations_per_env +
            model.link_to_articulation[link];
        atomicMin(data.dat_artic_beta + articulation, __float_as_uint(beta));
        return;
    }
    const uint32_t owner_local = model.body_collidable_body
        ? model.body_collidable_body[global] : ~0u;
    const uint32_t owner = owner_local == ~0u
        ? global : env * p.bodies_per_env + owner_local;
    if (data.body_inv_mass[owner] > 0.0f)
        atomicMin(data.dat_body_beta + owner, __float_as_uint(beta));
}

struct DatInterval {
    float start, end, value_start, value_end;
    uint32_t depth;
};

__device__ float DatPlaneFraction(const DatTruncateParams& p, const ModelView& model,
                                  const DataView& data, uint32_t env, DatPointRef ref,
                                  math::Vec3 origin, math::Vec3 normal,
                                  float offset, float speed) {
    if (!(speed < FLT_MAX)) return 0.0f;
    if (speed == 0.0f) return 1.0f;
    float values[5];
    for (uint32_t i = 0u; i <= 4u; ++i)
        values[i] = (DatPointAt(p, model, data, env, ref, 0.25f * i) - origin).Dot(normal) - offset;
    if (!(values[0] > 0.0f)) return 0.0f;
    DatInterval stack[20];
    uint32_t top = 0u;
    for (uint32_t i = 4u; i > 0u; --i)
        stack[top++] = {0.25f * (i - 1u), 0.25f * i,
                        values[i - 1u], values[i], 0u};
    while (top > 0u) {
        const auto interval = stack[--top];
        const float width = interval.end - interval.start;
        if (fminf(interval.value_start, interval.value_end) > 0.5f * speed * width)
            continue;
        if (interval.depth == 12u) return interval.start;
        const float mid = 0.5f * (interval.start + interval.end);
        const float value_mid = (DatPointAt(p, model, data, env, ref, mid) - origin).Dot(normal) - offset;
        stack[top++] = {mid, interval.end, value_mid,
                        interval.value_end, interval.depth + 1u};
        stack[top++] = {interval.start, mid, interval.value_start,
                        value_mid, interval.depth + 1u};
    }
    return 1.0f;
}

}  // namespace
}  // namespace nuka::phi
