#pragma once

#include "math/cuda_vec_ops.cuh"
#include "phi/backend_cuda/ops/dat_chain.cuh"
#include "phi/backend_cuda/ops/kinematics.cuh"

namespace nuka::phi {
namespace {

struct DatPointRef {
    uint32_t owner = 0u;
    uint32_t vertex = 0u;
    bool particle = false;
};

struct DatBodyMotion {
    math::Transform local;
    float translation = 0.0f, turn = 0.0f, rounding = 0.0f;
};

struct DatPairPoses {
    math::Transform a_start, a_end, b_start, b_end;
    math::Transform a_quarters[3], b_quarters[3];
    DatBodyMotion a_motion, b_motion;
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

// Pose of a body at a step fraction, in the world or, given a frame link, relative to that link.
__device__ math::Transform DatBodyPoseAt(const DatTruncateParams& p,
                                         const ModelView& model,
                                         const DataView& data, uint32_t env,
                                         uint32_t body, float s, uint32_t frame = ~0u) {
    const uint32_t global = env * p.bodies_per_env + body;
    uint32_t link = ~0u;
    math::Transform local{};
    if (DatBodyLink(p, model, env, body, &link, &local)) {
        const uint32_t articulation = model.link_to_articulation[link];
        const uint32_t offset = model.articulation_link_offset[articulation];
        const uint32_t count = model.articulation_link_count[articulation];
        math::Transform result = local;
        for (uint32_t traversed = 0u; traversed < count; ++traversed) {
            if (link == frame) return result;
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
                                 DatPointRef ref, float s, uint32_t frame = ~0u) {
    if (ref.particle) {
        const auto start = data.particle_prev_pos[ref.owner];
        return start + (data.particle_pos[ref.owner] - start) * s;
    }
    const auto pose = DatBodyPoseAt(p, model, data, env, ref.owner, s, frame);
    return pose.position + math::gpu::RotateByQuatNormalized(
        pose.rotation, DatBodyLocalPoint(model, ref.owner, ref.vertex));
}

// A rigid primitive shares one pose evaluation across all of its vertices.
__device__ collision::DatPrimitive DatPrimitiveAt(const DatTruncateParams& p,
    const ModelView& model, const DataView& data, uint32_t env, const DatPointRef* refs,
    uint32_t count, float fraction, uint32_t frame = ~0u,
    const math::Transform* body_pose = nullptr) {
    collision::DatPrimitive result;
    result.count = count;
    const auto pose = refs[0].particle ? math::Transform{} : body_pose ? *body_pose
        : DatBodyPoseAt(p, model, data, env, refs[0].owner, fraction, frame);
    for (uint32_t i = 0u; i < count; ++i)
        result.vertex[i] = refs[i].particle
            ? DatPointAt(p, model, data, env, refs[i], fraction, frame)
            : pose.position + math::gpu::RotateByQuatNormalized(pose.rotation,
                DatBodyLocalPoint(model, refs[i].owner, refs[i].vertex));
    return result;
}

__device__ bool DatMixedRefs(DatTruncateParams p, ModelView model, DataView data,
                              uint32_t env, uint32_t slot, DatPointRef* a,
                              uint32_t* count_a, DatPointRef* b, uint32_t* count_b) {
    const uint32_t kind = data.dat_pair_kind[slot];
    const bool particle_a = (kind & 4u) != 0u;
    const bool particle_b = (kind & 8u) != 0u;
    const uint32_t owner_a = data.dat_pair_owner_a[slot];
    const uint32_t owner_b = data.dat_pair_owner_b[slot];
    const uint32_t feature_a = data.dat_pair_feature_a[slot];
    const uint32_t feature_b = data.dat_pair_feature_b[slot];
    if ((kind & 3u) == 1u) {
        *count_a = 1u;
        *count_b = 3u;
        if (particle_a) {
            if (owner_a < env * p.particles_per_env ||
                owner_a >= (env + 1u) * p.particles_per_env) return false;
            a[0] = {owner_a, 0u, true};
        } else {
            if (owner_a >= p.bodies_per_env ||
                feature_a >= model.mesh_surface_info[owner_a].vertex_count) return false;
            a[0] = {owner_a, feature_a, false};
        }
        if (particle_b) {
            if (owner_b >= p.surfaces_per_env) return false;
            const auto info = model.particle_surface_info[owner_b];
            if (feature_b >= info.triangle_count) return false;
            const size_t at = (size_t{info.triangle_offset} + feature_b) * 3u;
            for (uint32_t j = 0u; j < 3u; ++j) {
                const uint32_t vertex = model.particle_surface_triangles[at + j];
                if (vertex >= info.vertex_count) return false;
                b[j] = {env * p.particles_per_env + info.vertex_offset + vertex, 0u, true};
            }
        } else {
            if (owner_b >= p.bodies_per_env) return false;
            const auto info = model.mesh_surface_info[owner_b];
            if (feature_b >= info.triangle_count) return false;
            const size_t at = (size_t{info.triangle_offset} + feature_b) * 3u;
            for (uint32_t j = 0u; j < 3u; ++j) {
                const uint32_t vertex = model.mesh_triangles[at + j];
                if (vertex >= info.vertex_count) return false;
                b[j] = {owner_b, vertex, false};
            }
        }
        return true;
    }
    if ((kind & 3u) != 2u || particle_b) return false;
    *count_a = *count_b = 2u;
    if (particle_a) {
        if (owner_a >= p.surfaces_per_env) return false;
        const auto info = model.particle_surface_info[owner_a];
        const auto edges = model.particle_surface_edge_info[owner_a];
        if (feature_a >= edges.edge_count) return false;
        const auto edge = model.particle_surface_edges[edges.edge_offset + feature_a];
        if (edge.vertex0 >= info.vertex_count || edge.vertex1 >= info.vertex_count) return false;
        a[0] = {env * p.particles_per_env + info.vertex_offset + edge.vertex0, 0u, true};
        a[1] = {env * p.particles_per_env + info.vertex_offset + edge.vertex1, 0u, true};
    } else {
        if (owner_a >= p.bodies_per_env) return false;
        const auto info = model.mesh_surface_info[owner_a];
        const auto edges = model.mesh_edge_info[owner_a];
        if (feature_a >= edges.edge_count) return false;
        const auto edge = model.mesh_edges[edges.edge_offset + feature_a];
        if (edge.vertex0 >= info.vertex_count || edge.vertex1 >= info.vertex_count) return false;
        a[0] = {owner_a, edge.vertex0, false};
        a[1] = {owner_a, edge.vertex1, false};
    }
    if (owner_b >= p.bodies_per_env) return false;
    const auto info = model.mesh_surface_info[owner_b];
    const auto edges = model.mesh_edge_info[owner_b];
    if (feature_b >= edges.edge_count) return false;
    const auto edge = model.mesh_edges[edges.edge_offset + feature_b];
    if (edge.vertex0 >= info.vertex_count || edge.vertex1 >= info.vertex_count) return false;
    b[0] = {owner_b, edge.vertex0, false};
    b[1] = {owner_b, edge.vertex1, false};
    return true;
}

// Interpolated joints bound the path curvature by 3 * turn * speed, with turn the rotation they
// sum to, so the path speed is at most the chord plus half that, with a slack for FK rounding.
__device__ float DatTightSpeed(const DatTruncateParams& p, const ModelView& model,
                               const DataView& data, uint32_t env, DatPointRef ref,
                               uint32_t frame, float speed, float turn, float* curvature = nullptr,
                               const math::Vec3* start = nullptr, const math::Vec3* end = nullptr) {
    if (curvature) *curvature = collision::DatChordDeviation(speed, turn);
    if (!(speed > 0.0f && speed <= FLT_MAX) || !(1.5f * turn < 1.0f)) return speed;
    const auto a = start ? *start : DatPointAt(p, model, data, env, ref, 0.0f, frame);
    const auto b = end ? *end : DatPointAt(p, model, data, env, ref, 1.0f, frame);
    const float rounding = 64.0f * FLT_EPSILON *
        fmaxf(sqrtf(a.LengthSq()), sqrtf(b.LengthSq()));
    return fminf(speed, sqrtf((b - a).LengthSq()) + 1.5f * turn * speed + rounding);
}

// A chain's speed bound is affine in the vertex radius in its owning link frame.
__device__ DatBodyMotion DatBodyMotionBounds(const DatTruncateParams& p, const ModelView& model,
    const DataView& data, uint32_t env, uint32_t body, uint32_t frame) {
    DatBodyMotion result;
    const uint32_t global = env * p.bodies_per_env + body;
    uint32_t link = ~0u;
    if (DatBodyLink(p, model, env, body, &link, &result.local)) {
        const uint32_t articulation = model.link_to_articulation[link];
        const uint32_t offset = model.articulation_link_offset[articulation];
        const uint32_t count = model.articulation_link_count[articulation];
        result.rounding = 8.0f * FLT_EPSILON * (count + 1u);
        float radius = 0.0f;
        for (uint32_t traversed = 0u; traversed < count; ++traversed) {
            if (link == frame) return result;
            const uint32_t parent = model.parent_link[link];
            const auto type = static_cast<ArticulationJointType>(model.joint_type[link]);
            if (parent == ~0u && type == ArticulationJointType::FloatingBase) {
                const auto a = data.dat_prev_base_pose[articulation], b = data.base_pose[articulation];
                const float arc = DatQuatArc(a.rotation, b.rotation);
                result.translation += sqrtf((b.position - a.position).LengthSq()) + arc * radius;
                result.turn += arc;
                return result;
            }
            const float delta = fabsf(data.q[link] - data.dat_prev_q[link]);
            if (type == ArticulationJointType::Revolute) {
                result.translation += delta * radius;
                result.turn += delta;
            }
            if (type == ArticulationJointType::Prismatic) result.translation += delta;
            radius += sqrtf((model.link_local_pose[link].position + model.parent_offset[link]).LengthSq());
            if (type == ArticulationJointType::Prismatic)
                radius += fmaxf(fabsf(data.dat_prev_q[link]), fabsf(data.q[link]));
            if (parent == ~0u) return result;
            if (parent >= count) break;
            link = offset + parent;
        }
        result.translation = INFINITY;
        return result;
    }
    const uint32_t owner_local = model.body_collidable_body ? model.body_collidable_body[global] : ~0u;
    const uint32_t owner = owner_local == ~0u ? global : env * p.bodies_per_env + owner_local;
    if (owner_local != ~0u) result.local = model.body_collidable_local[global];
    const auto a = data.dat_prev_body_pose[owner], b = data.body_pose[owner];
    result.translation = sqrtf((b.position - a.position).LengthSq());
    result.turn = DatQuatArc(a.rotation, b.rotation);
    result.rounding = 8.0f * FLT_EPSILON;
    return result;
}

__device__ float DatRadiusSpeed(const DatBodyMotion& motion, float radius) {
    return (motion.translation + motion.turn * radius) * (1.0f + motion.rounding);
}

__device__ float DatPointLipschitz(const DatTruncateParams& p, const ModelView& model,
    const DataView& data, uint32_t env, DatPointRef ref, uint32_t frame = ~0u,
    float* curvature = nullptr, const math::Vec3* start = nullptr, const math::Vec3* end = nullptr,
    const DatBodyMotion* cached = nullptr) {
    if (curvature) *curvature = 0.0f;
    if (ref.particle)
        return sqrtf((data.particle_pos[ref.owner] - data.particle_prev_pos[ref.owner]).LengthSq());
    const auto motion = cached ? *cached : DatBodyMotionBounds(p, model, data, env, ref.owner, frame);
    const auto vertex = DatBodyLocalPoint(model, ref.owner, ref.vertex);
    const float radius = sqrtf((motion.local.position + math::gpu::RotateByQuatNormalized(
        motion.local.rotation, vertex)).LengthSq());
    return DatTightSpeed(p, model, data, env, ref, frame, DatRadiusSpeed(motion, radius),
                         motion.turn, curvature, start, end);
}

__device__ float DatBodyCurvature(const ModelView& model, uint32_t body, const DatBodyMotion& motion) {
    const float radius = sqrtf(motion.local.position.LengthSq()) + model.mesh_vertex_reach[body];
    return collision::DatChordDeviation(DatRadiusSpeed(motion, radius), motion.turn);
}

// Bound kinds of the articulation witness: vertex motion, particle pair, body pair, chain travel.
constexpr uint32_t kDatBoundMotion = 0u, kDatBoundParticlePair = 1u, kDatBoundBodyPair = 2u,
                   kDatBoundChain = 3u, kDatBoundKinds = 4u, kDatNoCounterpart = 0xFFFFu;

// Env-local index of a witness counterpart: a particle or a body.
__device__ uint32_t DatCounterpart(const DatTruncateParams& p, const DatPointRef& ref) {
    return (ref.particle && p.particles_per_env > 0u ? ref.owner % p.particles_per_env
                                                     : ref.owner) & 0xFFFFu;
}

__device__ void DatMinOwnerBeta(const DatTruncateParams& p, const ModelView& model,
                                const DataView& data, uint32_t env,
                                DatPointRef ref, float beta, uint32_t bound,
                                uint32_t counterpart) {
    beta = fminf(1.0f, fmaxf(0.0f, beta));
    if (ref.particle) {
        atomicMin(reinterpret_cast<uint32_t*>(data.dat_particle_beta + ref.owner), __float_as_uint(beta));
        return;
    }
    const uint32_t global = env * p.bodies_per_env + ref.owner;
    uint32_t link = ~0u;
    math::Transform local{};
    if (DatBodyLink(p, model, env, ref.owner, &link, &local)) {
        const uint32_t articulation = model.link_to_articulation[link];
        atomicMin(data.dat_artic_beta + articulation, __float_as_uint(beta));
        if (data.dat_artic_witness != nullptr && bound < kDatBoundKinds) {
            const unsigned long long key =
                (static_cast<unsigned long long>(__float_as_uint(beta)) << 32u) |
                ((ref.owner & 0xFFFFu) << 16u) | (counterpart & 0xFFFFu);
            atomicMin(reinterpret_cast<unsigned long long*>(data.dat_artic_witness) +
                      size_t{articulation} * kDatBoundKinds + bound, key);
        }
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
                                  float offset, float speed, uint32_t frame = ~0u,
                                  float curvature = INFINITY, const math::Vec3* start = nullptr,
                                  const math::Vec3* finish = nullptr,
                                  const math::Transform* quarters = nullptr) {
    if (!(speed < FLT_MAX)) return 0.0f;
    if (speed == 0.0f) return 1.0f;
    float values[5];
    float scale = sqrtf(origin.LengthSq()) + fabsf(offset) + speed;
    const auto first = start ? *start : DatPointAt(p, model, data, env, ref, 0.0f, frame);
    const auto last = finish ? *finish : DatPointAt(p, model, data, env, ref, 1.0f, frame);
    const float endpoint_scale = fmaxf(sqrtf(first.LengthSq()), sqrtf(last.LengthSq()));
    values[0] = (first - origin).Dot(normal) - offset;
    values[4] = (last - origin).Dot(normal) - offset;
    if (!(values[0] > 0.0f)) return 0.0f;
    const float reserve = 64.0f * FLT_EPSILON * fmaxf(scale, endpoint_scale + speed);
    if (fminf(values[0], values[4]) > fminf(0.5f * speed, curvature + reserve)) return 1.0f;
    scale = fmaxf(scale, endpoint_scale);
    for (uint32_t i = 1u; i < 4u; ++i) {
        const auto point = quarters && !ref.particle ? quarters[i - 1u].position +
            math::gpu::RotateByQuatNormalized(quarters[i - 1u].rotation,
                DatBodyLocalPoint(model, ref.owner, ref.vertex)) :
            DatPointAt(p, model, data, env, ref, 0.25f * i, frame);
        scale = fmaxf(scale, sqrtf(point.LengthSq()));
        values[i] = (point - origin).Dot(normal) - offset;
    }
    const float rounding = 64.0f * FLT_EPSILON * scale;
    DatInterval stack[20];
    uint32_t top = 0u;
    for (uint32_t i = 4u; i > 0u; --i)
        stack[top++] = {0.25f * (i - 1u), 0.25f * i,
                        values[i - 1u], values[i], 0u};
    while (top > 0u) {
        const auto interval = stack[--top];
        const float width = interval.end - interval.start;
        if (fminf(interval.value_start, interval.value_end) >
            fminf(0.5f * speed * width, curvature * width * width + rounding))
            continue;
        if (interval.depth == 12u) return interval.start;
        const float mid = 0.5f * (interval.start + interval.end);
        const float value_mid = (DatPointAt(p, model, data, env, ref, mid, frame) - origin)
            .Dot(normal) - offset;
        stack[top++] = {mid, interval.end, value_mid,
                        interval.value_end, interval.depth + 1u};
        stack[top++] = {interval.start, mid, interval.value_start,
                        value_mid, interval.depth + 1u};
    }
    return 1.0f;
}

}  // namespace
}  // namespace nuka::phi
