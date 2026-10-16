#pragma once
// CUDA-free inputs of the sensor transform scatter: per-instance pose bindings and the
// device FK fields they read.

#include <cstdint>

namespace nuka::phi {

// Opaque phi backend handle (phi/backend.hpp); the scatter orders after its main stream.
struct Backend;

// One render instance's pose binding. kind: 0=Static, 1=Link, 2=Body, 3=Base; row indexes
// the env's link or body rows; cached_visual_local is pos(3) + quat(w,x,y,z).
struct InstanceScatterRow {
    uint32_t kind = 0;
    uint32_t row  = 0;
    float    cached_visual_local[7] = {0, 0, 0, 1, 0, 0, 0};
};

// Device FK field bases (math::Transform rows, env-major) and the env layout; a null field
// keeps that kind's bind pose.
struct ScatterFkSource {
    const void* link_pose = nullptr;
    const void* body_pose = nullptr;
    const void* base_pose = nullptr;
    uint32_t    env_index       = 0;
    uint32_t    links_per_env   = 0;
    uint32_t    bodies_per_env  = 0;
    Backend*    world_backend   = nullptr;
};

}  // namespace nuka::phi
