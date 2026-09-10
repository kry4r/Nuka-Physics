#pragma once

#include <cstdint>

#include "nk/solve/nk_row.hpp"

#if defined(__CUDACC__)
#define NUKA_OWNER_HD __host__ __device__
#else
#define NUKA_OWNER_HD
#endif

namespace nuka::nk {

struct CollidableOwner {
    uint32_t kind = ~0u;
    uint32_t body = ~0u;
    uint32_t link = ~0u;
    uint32_t articulation = ~0u;
};

// Tables are environment-major with local values; returned indices are global.
// Link collidables retain a reaction slot while link/articulation identify their owner.
NUKA_OWNER_HD inline CollidableOwner ResolveCollidableOwner(
    int32_t body_id, uint32_t env, uint32_t collidable, uint32_t bodies_per_env,
    uint32_t links_per_env, uint32_t artics_per_env,
    const uint32_t* body_to_link, const uint32_t* body_to_articulation,
    const uint32_t* body_collidable_body) {
    CollidableOwner result;
    if (collidable >= bodies_per_env) return result;
    const uint32_t base = env * bodies_per_env;
    uint32_t body = base + collidable;
    if (body_id < 0) {
        result.kind = kNkSideStatic;
        result.body = body;
        return result;
    }
    if (body_collidable_body != nullptr) {
        const uint32_t proxy_owner = body_collidable_body[body];
        if (proxy_owner != ~0u) {
            if (proxy_owner >= bodies_per_env) return result;
            body = base + proxy_owner;
            if (body_collidable_body[body] != ~0u) return result;
        }
    }
    const uint32_t link = body_to_link != nullptr ? body_to_link[body] : ~0u;
    if (link == ~0u) {
        result.kind = kNkSideRigid;
        result.body = body;
        return result;
    }
    if (link >= links_per_env || body_to_articulation == nullptr) return result;
    const uint32_t articulation = body_to_articulation[body];
    if (articulation >= artics_per_env) return result;
    result.kind = kNkSideArtic;
    result.body = body;
    result.link = env * links_per_env + link;
    result.articulation = env * artics_per_env + articulation;
    return result;
}

}  // namespace nuka::nk

#undef NUKA_OWNER_HD
