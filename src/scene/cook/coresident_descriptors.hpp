#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::coresident -- the union-cook CONTACT DESCRIPTOR plain-data
// structs (CoResidentFingertip / CoResidentCup / CoResidentFootSphere /
// CoResidentGround).
//
// RELOCATED here (M9 T11-core-b1) out of the DOOMED
// src/runtime/coresident/unified_coresident_stepper.hpp (which T11-core-b2
// DELETES with the rest of the coresident directory). These four structs are
// the PLAIN-DATA contact descriptors the surviving union cook carries:
// UnionSceneTemplate holds the fingertips / cup / feet / ground; BuildNkUnionModel
// reads them to emit the UnionCsr slots. They are PURE DATA -- no kernels, no
// device handles, no phi::Buffer, no CUDA types -- which is exactly why they can
// live under src/scene/cook (the host-only cook directory) while their old home
// in the coresident dir is deleted.
//
// NAMESPACE NOTE. Kept in nuka::runtime::coresident (NOT moved to
// nuka::scene::cook) ON PURPOSE: every consumer already qualifies these types as
// `coresident::CoResident*`, so keeping the same namespace makes the relocation
// pure #include churn (no qualifier edits at any consumer), preserving byte-exact
// cook fidelity with the least risk. The namespace name is not the M9 gate symbol;
// it is a transitional artifact that disappears when the native CookToModel->
// UnionCsr cook replaces these structs.
//
// The transitional unified_coresident_stepper.hpp (alive until T11-core-b2)
// #includes THIS header so its still-present class + GraspConfig / StandConfig
// keep compiling -- it no longer DEFINES these four structs.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime::coresident {

// A STATIC ground plane the box rests on -- routed THROUGH the unified spine, NOT a
// hand-coded clamp. The box<->ground contact is detected (box AABB vs the plane),
// narrowphased (C3b analytical box x plane), emitted (EmitCompliantContactRows, box
// side RigidInvMass / ground side StaticNull), and solved (UnifiedSolve) in the SAME
// pipeline as the foot<->box pair. This is the support the grasp-hold scene needs: a
// movable box on a static ground, with the falling foot pressing DOWN onto it -> the
// contact force builds + HOLDS instead of both bodies free-falling together (two
// unsupported free-falling bodies separate -- physically correct, but not a "hold").
// The box reacts two-way to BOTH the foot above and the ground below; the ground is
// immovable (StaticNull: invM=0, no-op apply). The plane normal is world +Z.
struct CoResidentGround {
    float    height = 0.0f;          // plane z (the box bottom rests on this).
    uint32_t broadphase_id = 8000u;  // distinct from box + every go2 link body id.
};

// One fingertip contact: a sphere collidable on an articulation link.
struct CoResidentFingertip {
    uint32_t   link = ~0u;          // the ArticulationLink handle (broadphase id).
    math::Vec3 local_offset{};      // sphere center in the link frame.
    float      radius = 0.0f;
    uint32_t   broadphase_handle = ~0u;  // same as `link` (the cross-pair handle).
};

// A movable rigid CUP whose collision shape is a CONVEX HULL (the C7a cup). The
// hull vertices are MESH-LOCAL (cook-time constant); each step the stepper rebuilds
// a world-space ConvexHullView from these + the cup's live BodyState pose.
struct CoResidentCup {
    std::vector<float> hull_verts;          // flat x,y,z triples, MESH-LOCAL, constant.
    uint32_t           broadphase_body_id = 7000u;  // distinct from every link id.
    uint32_t           VertexCount() const {
        return static_cast<uint32_t>(hull_verts.size() / 3u);
    }
};

// One foot collision sphere on an articulation ankle link (mirrors CoResidentFingertip
// but contacts the STATIC ground, not a movable cup). A foot polygon is N of these
// (e.g. 2 spheres per ankle -- toe + heel -- on a 2-foot biped == the 4-corner polygon).
struct CoResidentFootSphere {
    uint32_t   link = ~0u;          // the ArticulationLink handle (ankle link).
    math::Vec3 local_offset{};      // sphere center in the ankle-link frame.
    float      radius = 0.0f;
    uint32_t   broadphase_handle = ~0u;  // UNIQUE per sphere (resolver picks geometry);
                                         // the chain-J uses `link` (the real ankle link).
};

}  // namespace nuka::runtime::coresident
