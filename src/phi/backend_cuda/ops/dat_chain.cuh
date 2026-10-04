#pragma once

#include <cmath>
#include <cstdint>

#include "collision/dat_geometry.hpp"
#include "math/cuda_vec_ops.cuh"
#include "nk/model/generated/views.hpp"
#include "phi/articulation_contract.hpp"

namespace nuka::phi {
namespace {

template <class Params>
__device__ bool DatBodyLink(const Params& p, const ModelView& model,
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

// Lowest link whose subtree holds both bodies when they belong to one articulation, else ~0u.
template <class Params>
__device__ uint32_t DatCommonFrame(const Params& p, const ModelView& model, uint32_t env,
                                   uint32_t body_a, uint32_t body_b) {
    uint32_t a = ~0u, b = ~0u;
    math::Transform local{};
    if (p.articulations_per_env == 0u || !DatBodyLink(p, model, env, body_a, &a, &local) ||
        !DatBodyLink(p, model, env, body_b, &b, &local) ||
        model.link_to_articulation[a] != model.link_to_articulation[b]) return ~0u;
    const uint32_t articulation = env * p.articulations_per_env +
        model.link_to_articulation[a];
    const uint32_t offset = model.articulation_link_offset[articulation];
    const uint32_t count = model.articulation_link_count[articulation];
    uint32_t depth_a = 0u, depth_b = 0u;
    for (uint32_t link = a; model.parent_link[link] < count && depth_a < count; ++depth_a)
        link = offset + model.parent_link[link];
    for (uint32_t link = b; model.parent_link[link] < count && depth_b < count; ++depth_b)
        link = offset + model.parent_link[link];
    for (; depth_a > depth_b; --depth_a) a = offset + model.parent_link[a];
    for (; depth_b > depth_a; --depth_b) b = offset + model.parent_link[b];
    for (uint32_t traversed = 0u; a != b && traversed < count; ++traversed) {
        if (!(model.parent_link[a] < count && model.parent_link[b] < count)) return ~0u;
        a = offset + model.parent_link[a];
        b = offset + model.parent_link[b];
    }
    return a == b ? a : ~0u;
}

// Speed bound of every point of a body relative to the frame link: predicted from joint rates
// when given, else the joint motion of the step from start.
template <class Params>
__device__ float DatChainSpeed(const Params& p, const ModelView& model, const DataView& data,
                               uint32_t env, uint32_t body, uint32_t frame,
                               const float* rates, const float* start) {
    uint32_t link = ~0u;
    math::Transform local{};
    if (!DatBodyLink(p, model, env, body, &link, &local)) return INFINITY;
    const uint32_t articulation = env * p.articulations_per_env +
        model.link_to_articulation[link];
    const uint32_t offset = model.articulation_link_offset[articulation];
    const uint32_t count = model.articulation_link_count[articulation];
    float radius = sqrtf(local.position.LengthSq()) + model.mesh_vertex_reach[body];
    float speed = 0.0f;
    for (uint32_t traversed = 0u; traversed < count; ++traversed) {
        if (link == frame) return speed;
        const uint32_t parent = model.parent_link[link];
        if (parent >= count) break;
        const auto type = static_cast<ArticulationJointType>(model.joint_type[link]);
        const float rate = rates ? rates[link] : fabsf(data.q[link] - start[link]) / p.dt;
        if (type == ArticulationJointType::Revolute) speed += rate * radius;
        if (type == ArticulationJointType::Prismatic) speed += rate;
        radius += sqrtf((model.link_local_pose[link].position +
                         model.parent_offset[link]).LengthSq());
        if (type == ArticulationJointType::Prismatic)
            radius += rates ? fabsf(start[link]) + rate * p.dt
                            : fmaxf(fabsf(start[link]), fabsf(data.q[link]));
        link = offset + parent;
    }
    return INFINITY;
}

// Uncapped query radius of two links of one articulation, from the joint rates predicted at
// detection and the start joint positions.
template <class Params>
__device__ float DatArticMotionRadius(const Params& p, const ModelView& model,
                                      const DataView& data, uint32_t env, uint32_t a,
                                      uint32_t b, uint32_t frame, const float* start) {
    return collision::DatMotionRadius(0.0005f + p.margin, p.dt,
        DatChainSpeed(p, model, data, env, a, frame, data.dat_joint_rate, start),
        DatChainSpeed(p, model, data, env, b, frame, data.dat_joint_rate, start),
        p.relaxation);
}

}  // namespace
}  // namespace nuka::phi
