// ---------------------------------------------------------------------------
// v0.8 C5c-1 -- floating-base go2 foot<->ground contact through the UNIFIED spine
// ---------------------------------------------------------------------------
// The highest-risk WIRING task of the v0.8 collision/contact spine: route a
// floating-base go2's feet<->static-ground contact through the already-built
// C2->C3->C4->C5 pieces end-to-end and SELF-VALIDATE it (no MJX, no integrator):
//
//   (foot, ground) CandidatePairs
//     -> contact_stream_driver::BuildContactManifolds  (C5a, resolver-decoupled)
//       -> narrowphase_dispatch sphere x plane          (C3b analytical handler)
//         -> EmitCompliantContactRows                   (C4b compliant rows + R/aref)
//           -> UnifiedSolve w/ the C5b-core ArticulationChainJ reaction arm (C5b)
//
// The CRUX this test PROVES: the foot contact force recoils into the FLOATING
// BASE through an 18-wide foot chain Jacobian (6 base + 12 leg). A 12-wide
// (leg-only) J would silently drop the base reaction and the base would fall --
// a real bug, not an ABA artifact (the C4c IMPORTANT-1 flat-qdot warning made
// flesh). We discover that the production chain-J builder
// (articulation_jacobian.cu ComputeContactChainJacobianKernel, the "T8b" path)
// ALREADY emits 18 columns for a FloatingBase root, so we REUSE it (plus the CRBA
// ComputeArticulationInertiaM / FactorArticulationInertiaM for the 18x18 M^-1) --
// we do NOT hand-roll a second Jacobian or inertia.
//
// THE FLAT-QDOT(18) PACKING (why the C4c warning is addressed by construction).
// The production SolveArticulatedContactRowsKernel packs an 18-wide working
// vector as [base spatial v (omega-first) from link_velocity[root].v[0..5],
// then the 12 leg qdot] -- the SAME column order the chain-J and the M^-1 tile
// use (a base-inclusive prefix sum). The C5b-core UnifiedSolve articulation arm
// operates on a flat dof_stride-wide qdot slice; we seed slots [0..5] from the
// base spatial velocity and [6..17] from the leg qdot, run the solve, and read
// the base recoil back from slots [0..5]. So the base reaction does NOT vanish:
// it lands in the contiguous base columns of the flat qdot, exactly as the
// production kernel scatters it to link_velocity[root].
//
// SCOPE (C5c-1, VALIDATED-NOT-WIRED). No production path is flipped (world_stepper
// / the production AppendContactGroup/BuildContactRows/
// SolveArticulatedContactRows are untouched). No golden / generated / memory edit.
// Self-validated WITHOUT MJX. Explicitly DEFERRED:
//   - C5c-2: MJX single-step base-6 qacc parity for this unified path.
//   - C5c-3: the flat-qdot(18) integrator + multi-step stability rollout.
// ---------------------------------------------------------------------------

#include "collision/analytical_manifold.hpp"   // amf::PrimParams / PrimFrame
#include "collision/candidate_pair.hpp"        // CandidatePair, CollidableRef
#include "collision/contact_stream_driver.hpp" // BuildContactManifolds, ResolvedShape
#include "constraint/contact_manifold.hpp"     // ContactManifold
#include "constraint/row_articulation_refs.hpp"
#include "constraint/row_buffers.hpp"
#include "constraint/row_builder.hpp"          // EmitCompliantContactRows
#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer_legacy.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_jacobian.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/articulation/featherstone_aba.hpp"
#include "runtime/world_builder.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cooker.hpp"
#include "solver/solver_config.hpp"  // SolverConfig (iters/dt knobs; legacy UnifiedSolve dropped M9 T11)

#include "foot_chain_jacobian.hpp"  // shared ComputeFootChainJ18 (FK-refresh-correct)
#include "nk_solve_harness.hpp"     // M9 T11: the nk SolveRowsBlockIsland re-point seam

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <vector>

namespace {

namespace articulation = nuka::runtime::articulation;
namespace amf = nuka::collision::amf;
using nuka::collision::BuildContactManifolds;
using nuka::collision::CandidatePair;
using nuka::collision::ResolvedShape;
using nuka::collision::ShapeResolver;
using nuka::constraint::CollidableRef;
using nuka::constraint::CollidableType;
using nuka::constraint::ContactManifold;
using nuka::constraint::ContactRowSides;
using nuka::constraint::kInvalidArtIndex;
using nuka::constraint::ReactionProviderKind;
using nuka::constraint::RowArticulationRefs;
using nuka::constraint::RowArticulationSide;
using nuka::constraint::RowBuffers;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::scene::ShapeType;

constexpr uint32_t kInvalidLink = ~0u;
constexpr uint32_t kDof = 18u;  // 6-DOF floating base + 12 revolute legs.

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

// Cooked go2_float plus its base-relative foot shapes + per-link masses. Mirrors
// CookGo2Float() in tests/runtime/test_floating_base_contact.cpp (the production-
// pipeline analogue): this test routes the SAME physics through the NEW spine.
struct CookedFloat {
    articulation::ArticulationHostState host;
    std::vector<articulation::FootShape> feet;
    std::vector<float> link_mass;  // per global link (cooked).
};

CookedFloat CookGo2Float() {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    const auto scene = nuka::import::LoadUsd(scene_path.string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);

    CookedFloat result;
    result.host = articulation::BuildArticulationHostState(
        world.template_view.articulations, world.template_view.body_table);
    const uint32_t link_count = result.host.TotalLinkCount();

    // Cooked per-link mass = the [3][3] entry of the 6x6 link spatial inertia.
    result.link_mass.assign(link_count, 0.0f);
    for (uint32_t link = 0u; link < link_count; ++link) {
        result.link_mass[link] = result.host.link_inertia[link].I[3u * 6u + 3u];
    }

    // Foot spheres: every Sphere shape -> its owning calf link + local offset.
    const auto& shapes = world.template_view.shape_table;
    for (uint32_t shape = 0u; shape < shapes.types.size(); ++shape) {
        if (shapes.types[shape] != nuka::scene::ShapeType::Sphere) {
            continue;
        }
        const uint32_t body = shapes.body_ids[shape];
        uint32_t calf_link = kInvalidLink;
        for (uint32_t link = 0u; link < link_count; ++link) {
            if (result.host.link_body[link] == body) {
                calf_link = link;
                break;
            }
        }
        if (calf_link == kInvalidLink) {
            continue;
        }
        articulation::FootShape foot;
        foot.calf_local_link = calf_link;
        foot.local_offset = shape < shapes.local_transforms.size()
                                ? shapes.local_transforms[shape].position
                                : Vec3::Zero();
        foot.radius = shape < shapes.radii.size() ? shapes.radii[shape] : 0.0f;
        result.feet.push_back(foot);
    }
    return result;
}

// Run FK at the current host state and return the world pose of every link.
std::vector<Transform> ForwardKinematics(const nuka::phi::DeviceContext& context,
                                         const articulation::ArticulationHostState& host) {
    const uint32_t link_count = host.TotalLinkCount();
    auto device = articulation::UploadArticulationState(context, host);
    nuka::phi::Buffer pose_buf(static_cast<size_t>(link_count) * sizeof(Transform),
                               nuka::phi::MemoryKind::Device);
    articulation::UpdateWorldLinkPoses(context, device.View(),
                                       static_cast<Transform*>(pose_buf.Data()));
    context.stream.Synchronize();
    std::vector<Transform> poses(link_count);
    pose_buf.CopyToHost(poses.data(), poses.size() * sizeof(Transform));
    return poses;
}

// The world contact point of a foot (its sphere center projected to the FK link
// pose). This is the SAME material point the manifold + the chain-J consume.
Vec3 FootWorldCenter(const std::vector<Transform>& poses,
                     const articulation::FootShape& foot) {
    const Transform calf = poses[foot.calf_local_link];
    return calf.position + calf.rotation.Rotate(foot.local_offset);
}

// Reusable on-device CRBA -> 18x18 M^-1 for ONE articulation (a standalone
// composite-inertia + factorization pass). REUSES the production CRBA +
// factorization -- no hand-rolled second inertia. Requires ABA pass-1 to have run
// (link_xup / motion subspace current), so we run ComputeAccelerations first.
std::vector<float> ComputeInverseInertia18(const nuka::phi::DeviceContext& context,
                                           const articulation::ArticulationHostState& host,
                                           float gravity_z) {
    auto device = articulation::UploadArticulationState(context, host);
    auto view = device.View();
    const uint32_t link_count = host.TotalLinkCount();
    // ABA pass-1 populates link_xup + the spatial motion subspace CRBA needs.
    articulation::FeatherstoneAba::ComputeAccelerations(context, view, gravity_z);

    nuka::phi::Buffer composite(
        static_cast<size_t>(link_count) * sizeof(articulation::LinkSpatialInertia),
        nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer m(static_cast<size_t>(kDof) * kDof * sizeof(float),
                        nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer m_inv(static_cast<size_t>(kDof) * kDof * sizeof(float),
                            nuka::phi::MemoryKind::Device);
    articulation::ComputeArticulationInertiaM(
        context, view, kDof,
        static_cast<articulation::LinkSpatialInertia*>(composite.Data()),
        static_cast<float*>(m.Data()));
    articulation::FactorArticulationInertiaM(
        context, view, kDof, static_cast<const float*>(m.Data()),
        static_cast<float*>(m_inv.Data()));
    context.stream.Synchronize();
    std::vector<float> out(static_cast<size_t>(kDof) * kDof);
    m_inv.CopyToHost(out.data(), out.size() * sizeof(float));
    return out;
}

// 18-wide foot chain Jacobian via the shared FK-refresh-correct helper (see
// tests/solver/foot_chain_jacobian.hpp). The shared helper refreshes link_pose
// from the passed FK world poses (the subsume copy formerly omitted this; for the
// +Z-normal coplanar-feet scenario here the refresh is invisible to the base-column
// assertions, but it is the correct version -- the named-debt fold).
std::vector<float> ComputeFootChainJ18(const nuka::phi::DeviceContext& context,
                                       const articulation::ArticulationHostState& host,
                                       const std::vector<Transform>& fk_world_poses,
                                       uint32_t contact_link,
                                       const Vec3& contact_point,
                                       const Vec3& contact_normal) {
    return nuka::test::ComputeFootChainJ18(context, host, fk_world_poses,
                                           contact_link, contact_point, contact_normal,
                                           kDof);
}

// A ground PrimParams whose plane normal is world +Z (SpherePlane reads the
// normal from plane.frame.cy, so we put +Z there with an orthonormal basis).
amf::PrimParams MakeGroundPrim(float ground_height) {
    amf::PrimParams p;
    p.frame.cx = Vec3{1.0f, 0.0f, 0.0f};
    p.frame.cy = Vec3{0.0f, 0.0f, 1.0f};   // plane normal = world +Z
    p.frame.cz = Vec3{0.0f, -1.0f, 0.0f};  // right-handed: cx x cy = cz
    p.frame.t = Vec3{0.0f, 0.0f, ground_height};
    return p;
}

amf::PrimParams MakeFootPrim(const Vec3& center, float radius) {
    amf::PrimParams p;
    p.radius = radius;
    p.frame.t = center;  // identity rotation; sphere is rotation-invariant.
    return p;
}

// Handles: foot sides use the global calf link index; the single ground side uses
// a fixed handle. The resolver maps each to its ResolvedShape.
constexpr uint32_t kGroundHandle = 4096u;  // distinct from any link index.

CollidableRef MakeFootRef(uint32_t calf_link) {
    CollidableRef ref;
    ref.type = CollidableType::ArticulationLink;
    ref.react = ReactionProviderKind::ArticulationChainJ;
    ref.handle = calf_link;
    return ref;
}

CollidableRef MakeGroundRef() {
    CollidableRef ref;
    ref.type = CollidableType::StaticWorld;
    ref.react = ReactionProviderKind::StaticNull;
    ref.handle = kGroundHandle;
    return ref;
}

// The whole foot-ground UNIFIED-spine assembly for ONE settled go2_float: it
// returns the solved flat qdot(18) plus the per-foot solved normal lambda and the
// per-foot 18-wide chain-J. base_vz0 is the (downward, body-z) base velocity the
// flat qdot is seeded with; the contact must recoil it toward 0.
struct UnifiedFootGroundResult {
    std::vector<float> qdot;             // 18-wide, post-solve (base[0..5]+legs).
    std::vector<float> qdot0;            // 18-wide, pre-solve seed.
    std::vector<std::vector<float>> foot_J;  // per foot, 18-wide chain-J (normal).
    std::vector<float> foot_lambda;      // per foot, solved normal impulse.
    std::vector<float> contact_depth;    // per foot manifold penetration.
    std::vector<float> jv_before;        // per foot, J . qdot0 (pre-solve normal vel).
    std::vector<float> jv_after;         // per foot, J . qdot  (post-solve normal vel).
    uint32_t row_count = 0u;
    double total_mass = 0.0;
    Vec3 base_origin{};                  // base pose origin at the settled state.
};

// J . qdot over a flat dof_stride-wide chain-J row and qdot slice (the contact
// normal velocity the compliant row drives toward its target).
float ChainJv(const std::vector<float>& j, const std::vector<float>& qdot) {
    float jv = 0.0f;
    for (uint32_t r = 0u; r < kDof; ++r) jv += j[r] * qdot[r];
    return jv;
}

UnifiedFootGroundResult RunUnifiedFootGround(const nuka::phi::DeviceContext& context,
                                             const CookedFloat& cooked,
                                             float ground_height,
                                             float base_vz0,
                                             const nuka::solver::SolverConfig& config) {
    UnifiedFootGroundResult result;
    const auto& host = cooked.host;
    const uint32_t root = host.articulation_link_offset[0];
    const uint32_t link_count = host.TotalLinkCount();
    for (uint32_t l = 0u; l < link_count; ++l) result.total_mass += cooked.link_mass[l];
    result.base_origin = host.base_pose[0].position;

    // --- FK -> foot world centers (the material contact points) --------------
    const auto poses = ForwardKinematics(context, host);

    // --- 18x18 M^-1 via production CRBA (reuse, not hand-rolled) --------------
    const float kGravityZ = -9.81f;
    const std::vector<float> minv = ComputeInverseInertia18(context, host, kGravityZ);

    // --- (foot, ground) CandidatePairs + ShapeResolvers ----------------------
    const Vec3 kUp{0.0f, 0.0f, 1.0f};
    const amf::PrimParams ground_prim = MakeGroundPrim(ground_height);

    std::vector<CandidatePair> pairs;
    std::vector<uint32_t> foot_calf_link;     // per pair: the contact link.
    std::vector<Vec3> foot_point;             // per pair: the world contact center.
    for (const auto& foot : cooked.feet) {
        const Vec3 center = FootWorldCenter(poses, foot);
        CandidatePair pair;
        pair.a = MakeFootRef(foot.calf_local_link);
        pair.b = MakeGroundRef();
        pairs.push_back(pair);
        foot_calf_link.push_back(foot.calf_local_link);
        foot_point.push_back(center);
    }

    // Resolver: foot handle -> a sphere at its FK center; ground handle -> plane.
    // (The decoupling seam: the driver never reads the cooked table -- the
    // resolver supplies the prim + world frame, sourced here from the FK pose.)
    ShapeResolver resolve = [&](const CollidableRef& ref, ResolvedShape* out) -> bool {
        if (ref.type == CollidableType::StaticWorld) {
            out->type = ShapeType::Plane;
            out->prim = ground_prim;
            return true;
        }
        if (ref.type == CollidableType::ArticulationLink) {
            for (size_t f = 0u; f < cooked.feet.size(); ++f) {
                if (cooked.feet[f].calf_local_link == ref.handle) {
                    out->type = ShapeType::Sphere;
                    out->prim = MakeFootPrim(foot_point[f], cooked.feet[f].radius);
                    return true;
                }
            }
        }
        return false;
    };

    // --- C5a driver: CandidatePairs -> sphere x plane ContactManifolds -------
    std::vector<ContactManifold> manifolds;
    BuildContactManifolds(pairs, resolve, &manifolds);

    // --- C4b: compliant rows + sides -----------------------------------------
    nuka::constraint::ContactRowComplianceInputs inputs;
    inputs.vel = 0.0f;          // qvel=0 at the contact (the base falls, legs held).
    inputs.invweight = 1.0f;
    inputs.dt = config.velocity_iterations > 0u ? 1.0f / 240.0f : 0.0f;
    inputs.condim = 1u;         // FRICTIONLESS: 4 clean normal rows, no tangent J.
    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    EmitCompliantContactRows(manifolds, inputs, &rows, &sides);
    result.row_count = rows.RowCount();

    // --- Pack the per-foot 18-wide chain-J (normal=+Z), one slot per manifold -
    // The manifolds are appended in pairs order (sequential walk preserves it), so
    // manifold m corresponds to foot_calf_link[m] / foot_point[m]. (The
    // ASSERT_EQ(row_count, feet.size()) in the tests guards this alignment -- a
    // dropped/extra manifold would shift it and fail loudly, not silently.)
    // CONTACT POINT: we feed the foot CENTER (foot_point[m]) to the chain-J, while
    // the manifold's geometric point is the sphere BOTTOM (center - r*n). For a +Z
    // normal this is harmless: the base columns (roll=lever_y, pitch=-lever_x,
    // lin_z=1) all drop the z component, so the lever_z difference is invisible.
    // This mirrors the production test_floating_base_contact precedent (it feeds the
    // material foot center, not the ground-projected point). NOTE: this z-invariance
    // is SPECIFIC to a vertical normal + coplanar feet; non-coplanar / tilted-normal
    // contacts would need the true manifold point (out of C5c-1 scope).
    std::vector<float> chain_jacobians;  // [manifold_count * kDof]
    result.foot_J.resize(manifolds.size());
    result.contact_depth.resize(manifolds.size());
    for (size_t m = 0u; m < manifolds.size(); ++m) {
        const std::vector<float> j =
            ComputeFootChainJ18(context, host, poses, foot_calf_link[m], foot_point[m], kUp);
        chain_jacobians.insert(chain_jacobians.end(), j.begin(), j.end());
        result.foot_J[m] = j;
        result.contact_depth[m] =
            manifolds[m].point_count > 0u ? manifolds[m].points[0].penetration : 0.0f;
    }

    // --- Build art_refs + fix the row coloring key. EmitCompliantContactRows set
    // each row's body index = manifold.handle (4 DISTINCT calf links) and did NOT
    // build RowArticulationRefs. For the C5b articulation arm we (1) set the
    // articulation side's row body index to the coloring key body_count+art_index
    // (= 0+0 = 0, SHARED across all 4 feet so they SERIALIZE -- no race on the
    // shared base DOFs [0..5]), and (2) emit one RowArticulationRefs per row with
    // the foot's art_index/chain_jac_slot. There is ONE articulation (art_index 0)
    // and chain_jac_slot == the manifold/foot index. body_count is 0 (no rigid
    // bodies), so the coloring key is exactly art_index = 0.
    const uint32_t body_count = 0u;
    const uint32_t kArtIndex = 0u;
    std::vector<RowArticulationRefs> art_refs(rows.RowCount());
    for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
        // condim=1 -> rows are all normal rows, one per manifold, in manifold order.
        const uint32_t slot = r;  // chain_jac_slot == foot/manifold index.
        // Side A is the foot (ArticulationChainJ); side B is the ground (Static).
        rows.body_indices[2u * r + 0u] = body_count + kArtIndex;  // coloring key.
        rows.body_indices[2u * r + 1u] = nuka::constraint::kInvalidBodyIndex;
        art_refs[r].a = RowArticulationSide{kArtIndex, slot};
        art_refs[r].b = RowArticulationSide{};  // ground: not an articulation.
    }

    // --- Seed the flat qdot(18): base spatial v (omega-first) in [0..5], legs in
    // [6..17]. We give the base a downward body-z velocity (slot 5 == v_lin body z)
    // so the base "falls"; legs start at rest. Identity base orientation => body z
    // == world z, so a negative slot-5 is a downward base velocity.
    std::vector<float> qdot(kDof, 0.0f);
    qdot[5] = base_vz0;  // downward base vertical velocity (body frame == world).
    result.qdot0 = qdot;

    // --- C5b-core compliant contact solve: M9 T11 nk-ONLY -- the SAME assembled
    // rows / chain-J / M^-1 / seeded flat qdot run through the nk SolveRows-
    // BlockIsland op (nk_solve_harness.hpp). The legacy solver::UnifiedSolve arm
    // (deleted in M9 T11-core) is gone; the SAME 18-wide-J / base-recoil / foot-
    // normal-velocity / D1 gates judge the result. Articulation-only -> no bodies.
    std::vector<nuka::runtime::rigid::BodyState> empty_bodies;
    {
        const auto nk_res = nk_harness::NkSolveRows(
            rows, sides, art_refs, chain_jacobians, minv, qdot, &empty_bodies,
            kDof, static_cast<uint16_t>(config.velocity_iterations), inputs.dt);
        EXPECT_TRUE(nk_res.ok) << "nk solve harness failed";
        if (nk_res.ok) {
            qdot = nk_res.qdot;
            for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
                rows.rows[r].lambda = nk_res.lambda[r];
            }
        }
    }

    result.qdot = qdot;
    result.foot_lambda.resize(rows.RowCount());
    result.jv_before.resize(rows.RowCount());
    result.jv_after.resize(rows.RowCount());
    for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
        result.foot_lambda[r] = rows.rows[r].lambda;
        // condim=1: row r is the normal row of foot/manifold r -> chain-J slot r.
        result.jv_before[r] = ChainJv(result.foot_J[r], result.qdot0);
        result.jv_after[r] = ChainJv(result.foot_J[r], result.qdot);
    }
    return result;
}

// Many-iteration converged PGS config (4 coupled feet need a converged solve to
// pin the feet -> drive base-z qdot toward 0). Compliant rows fold R into the
// denominator only (no -R*lambda residual), so more iterations drift toward the
// HARD-contact limit -- which is exactly what we want here (feet pinned).
nuka::solver::SolverConfig ConvergedConfig() {
    nuka::solver::SolverConfig cfg;
    cfg.velocity_iterations = 64u;
    cfg.position_iterations = 0u;
    cfg.slop = 0.0f;
    cfg.baumgarte = 0.0f;
    return cfg;
}

// Seat the ground a hair above the rest foot bottom so the feet start just
// penetrating (a small, solvable depth). Mirrors test_floating_base_contact.
float SeatGround(const nuka::phi::DeviceContext& context, const CookedFloat& cooked) {
    const auto poses = ForwardKinematics(context, cooked.host);
    float min_bottom = std::numeric_limits<float>::infinity();
    for (const auto& foot : cooked.feet) {
        const Vec3 center = FootWorldCenter(poses, foot);
        min_bottom = std::min(min_bottom, center.z - foot.radius);
    }
    constexpr float kRestPenetration = 0.002f;
    return min_bottom + kRestPenetration;
}

}  // namespace

// ===========================================================================
// (B) THE 18-WIDE J PROOF (the key assertion -- structural base-column check).
//   Compute the per-foot 18-wide chain-J for normal=+Z at identity base
//   orientation, and assert the 6 base columns have the EXACT structure a +Z
//   foot-point Jacobian must have:
//     col 0 (base roll)  = +lever_y   (= foot_y - base_y)
//     col 1 (base pitch) = -lever_x   (= -(foot_x - base_x))
//     col 2 (base yaw)   = 0
//     col 3,4 (base lin x,y) = 0
//     col 5 (base lin z)     = 1
//   This proves the base reaction arm is PHYSICALLY right (not merely nonzero):
//   front/rear feet get opposite-sign pitch, left/right get opposite-sign roll,
//   and the upward push lands in col 5. A 12-wide (leg-only) J fails (all base
//   columns zero); a transposed / wrong-frame J fails the structure.
// ===========================================================================
TEST(FootGroundSubsume, FootChainJacobianHas18WideBaseColumns) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    auto& host = cooked.host;
    ASSERT_EQ(host.ArticulationCount(), 1u);
    const uint32_t root = host.articulation_link_offset[0];
    ASSERT_EQ(host.joint_type[root], articulation::ArticulationJointType::FloatingBase);
    ASSERT_EQ(articulation::ArticulationDofCount(host, 0u), kDof)
        << "go2_float base-inclusive DOF must be 18 (6 base + 12 legs)";
    ASSERT_FALSE(cooked.feet.empty());

    // Identity base orientation so body==world and the base-column structure is
    // exactly the analytic +Z foot-point Jacobian.
    host.base_pose[0].rotation = Quat::Identity();
    const Vec3 base_origin = host.base_pose[0].position;
    const auto poses = ForwardKinematics(context, host);

    const Vec3 kUp{0.0f, 0.0f, 1.0f};
    double max_base_col_err = 0.0;
    bool any_nonzero_pitch = false;
    bool any_nonzero_roll = false;
    for (const auto& foot : cooked.feet) {
        const Vec3 center = FootWorldCenter(poses, foot);
        const std::vector<float> j =
            ComputeFootChainJ18(context, host, poses, foot.calf_local_link, center, kUp);
        ASSERT_EQ(j.size(), static_cast<size_t>(kDof));

        const float lever_x = center.x - base_origin.x;
        const float lever_y = center.y - base_origin.y;
        // Expected base columns for normal=+Z at identity orientation.
        const float exp[6] = {+lever_y, -lever_x, 0.0f, 0.0f, 0.0f, 1.0f};
        std::printf("[diag] (B) foot@link %u center=(%+.4f,%+.4f,%+.4f) "
                    "J_base=(%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f) "
                    "exp=(%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f)\n",
                    foot.calf_local_link, center.x, center.y, center.z,
                    j[0], j[1], j[2], j[3], j[4], j[5],
                    exp[0], exp[1], exp[2], exp[3], exp[4], exp[5]);
        for (uint32_t c = 0u; c < 6u; ++c) {
            max_base_col_err = std::max<double>(max_base_col_err, std::fabs(j[c] - exp[c]));
        }
        // The base columns MUST be nonzero (not a 12-wide leg-only J): col 5 == 1.
        EXPECT_NEAR(j[5], 1.0f, 1.0e-4f)
            << "base linear-z column != 1 -- the foot J dropped the base (12-wide?)";
        any_nonzero_pitch |= std::fabs(j[1]) > 1.0e-3f;
        any_nonzero_roll |= std::fabs(j[0]) > 1.0e-3f;
    }
    std::printf("[diag] (B) max base-column structural error = %.3e\n", max_base_col_err);

    // Structural correctness (the load-bearing assertion): the base columns match
    // the analytic +Z foot-point Jacobian to tight tolerance.
    EXPECT_LT(max_base_col_err, 5.0e-3)
        << "foot chain-J base columns do not match the analytic +Z Jacobian "
           "(roll=+lever_y, pitch=-lever_x, lin_z=1) -- base reaction arm is WRONG";
    // Non-trivial moment arms (the feet are offset from the base in x and y), so a
    // base force genuinely produces base roll+pitch coupling (not a degenerate J).
    EXPECT_TRUE(any_nonzero_pitch) << "no foot produced a base-pitch column";
    EXPECT_TRUE(any_nonzero_roll) << "no foot produced a base-roll column";
}

// ===========================================================================
// (A)+(B) BASE RECOIL through the UNIFIED spine: seed the base with a downward
//   vertical velocity, run the foot-ground contact through C2->C3->C4->C5, and
//   assert the contact pins the feet so the base-z velocity recoils UPWARD toward
//   zero (the 18-wide J carries the foot reaction into the base). A 12-wide J
//   would leave base-z untouched (the base keeps falling).
// ===========================================================================
TEST(FootGroundSubsume, ContactRecoilsFloatingBaseThroughUnifiedSpine) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    cooked.host.base_pose[0].rotation = Quat::Identity();
    for (auto& v : cooked.host.link_velocity) for (float& c : v.v) c = 0.0f;
    for (float& qd : cooked.host.qdot) qd = 0.0f;

    const float ground_height = SeatGround(context, cooked);
    const float kBaseVz0 = -1.0f;  // downward (body z == world z at identity).

    const auto out =
        RunUnifiedFootGround(context, cooked, ground_height, kBaseVz0, ConvergedConfig());

    // Driver produced one normal row per foot (frictionless condim=1).
    ASSERT_EQ(out.row_count, static_cast<uint32_t>(cooked.feet.size()))
        << "expected one compliant normal row per foot";
    ASSERT_GE(out.row_count, 4u) << "go2 has 4 feet -> 4 contacts";

    // Each foot manifold penetrated (the seated ground) and produced a POSITIVE
    // upward normal impulse (the feet push up on the base).
    double sum_lambda = 0.0;
    for (uint32_t r = 0u; r < out.row_count; ++r) {
        EXPECT_GT(out.contact_depth[r], 0.0f) << "foot " << r << " did not penetrate";
        EXPECT_GT(out.foot_lambda[r], 0.0f) << "foot " << r << " gave no upward impulse";
        sum_lambda += out.foot_lambda[r];
    }

    // CORROBORATING (not the discriminator): the flat qdot base columns [0..5]
    // MOVED, i.e. the contact reached the base. NOTE: a 12-wide (leg-only) J would
    // ALSO move the base here, because M^-1 is DENSE -- leg<->base off-diagonal
    // coupling carries leg impulses into the base. So base movement + upward-z are
    // physical sanity, NOT proof of an 18-wide J. The DISCRIMINATING proof is the
    // structural base-column check in FootChainJacobianHas18WideBaseColumns (j[5]==1
    // and the base columns == the analytic +Z Jacobian).
    //
    // The base delta is (roll=col0, pitch=col1, yaw=col2, lin x/y/z=col3/4/5):
    //   roll (col0) ~ 0  -- the 4 feet are LEFT/RIGHT symmetric (y=+/-0.142), so
    //                       equal upward lambda gives cancelling roll torques;
    //   pitch (col1) != 0 -- FRONT/REAR are asymmetric in BOTH lever (x=+0.178 vs
    //                       -0.209) AND solved lambda, so pitch does not cancel.
    // That zero-roll / nonzero-pitch split independently confirms the full 6-DOF
    // base block is live and physical.
    double base_delta_norm = 0.0;
    for (uint32_t c = 0u; c < 6u; ++c) {
        base_delta_norm += std::fabs(out.qdot[c] - out.qdot0[c]);
    }
    std::printf("[diag] (B) base qdot pre  = (%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f)\n",
                out.qdot0[0], out.qdot0[1], out.qdot0[2],
                out.qdot0[3], out.qdot0[4], out.qdot0[5]);
    std::printf("[diag] (B) base qdot post = (%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f)"
                "  (roll,pitch,yaw,lx,ly,lz)\n",
                out.qdot[0], out.qdot[1], out.qdot[2],
                out.qdot[3], out.qdot[4], out.qdot[5]);
    std::printf("[diag] (A) base v_z: %+.5f -> %+.5f   Sigma lambda_n = %.5f\n",
                out.qdot0[5], out.qdot[5], sum_lambda);
    EXPECT_GT(base_delta_norm, 1.0e-3)
        << "the contact solve did not move the base DOFs at all (sanity check)";

    // (B) The recoil OPPOSES gravity: base-z velocity corrected UPWARD (less
    // negative). qdot[5] is the body-z linear velocity (== world z at identity).
    EXPECT_GT(out.qdot[5], out.qdot0[5])
        << "base vertical velocity was not corrected upward by the contact";
    EXPECT_GT(out.qdot[5] - out.qdot0[5], 1.0e-3)
        << "base vertical recoil is negligible -- contact did not push the base up";

    // (A) The feet PIN the base: a converged solve drives the base-z velocity
    // toward 0 (the contact arrests the downward fall). It need not reach exactly
    // 0 (the 4-foot load shares into roll/pitch + leg DOFs), but it must be
    // substantially arrested from the -1.0 m/s seed.
    EXPECT_LT(std::fabs(out.qdot[5]), std::fabs(out.qdot0[5]))
        << "base-z velocity not arrested by the contact";
    EXPECT_LT(out.qdot[5], 0.0f + 1.0e-3f)
        << "base over-shot upward (unexpected for a pinned resting contact)";
}

// ===========================================================================
// (A) CONSTRAINT SATISFACTION (the in-scope single-solve ground-reaction proof).
//   A single UnifiedSolve is VELOCITY-ONLY (the caller owns gravity + integration):
//   it can only REDISTRIBUTE the existing momentum, driving each foot's normal
//   velocity toward the compliant target -- it does NOT reach the steady-state
//   Sigma(lambda_n) ~ m_total*g*dt force balance, which is an EQUILIBRIUM property
//   that emerges only with held legs over many steps (the production 600-step
//   settle in test_floating_base_contact) or a multi-step integrator. That
//   equilibrium magnitude balance is C5c-3 (flat-qdot(18) integrator + rollout),
//   explicitly DEFERRED. What a single solve DOES prove in-scope: every foot's
//   post-solve normal velocity |J . qdot| is driven near the compliant target
//   (>> reduced from its pre-solve value) -- the feet stop penetrating -- AND
//   (test B) the reaction reaches the base. We report Sigma(lambda_n) vs m*g*dt
//   as NUMBERS (not an assertion) and explain why they differ: the free legs
//   absorb the response (tiny foot effective mass through the free hip->calf
//   chain), so a single solve's impulse tracks foot-velocity cancellation, not Mg.
// ===========================================================================
TEST(FootGroundSubsume, FootNormalVelocitiesDrivenToTargetAndBaseFeelsIt) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    cooked.host.base_pose[0].rotation = Quat::Identity();
    for (auto& v : cooked.host.link_velocity) for (float& c : v.v) c = 0.0f;
    for (float& qd : cooked.host.qdot) qd = 0.0f;

    const float ground_height = SeatGround(context, cooked);
    const float kGravityZ = -9.81f;
    const float dt = 1.0f / 240.0f;

    double total_mass = 0.0;
    for (uint32_t l = 0u; l < cooked.host.TotalLinkCount(); ++l) {
        total_mass += cooked.link_mass[l];
    }
    const double mg_dt = total_mass * static_cast<double>(-kGravityZ) * dt;

    // Seed the base with a clear downward velocity so the feet are penetrating AND
    // approaching (Jv_before < 0 == closing): the compliant solve must arrest it.
    const float base_vz0 = -1.0f;
    const auto out =
        RunUnifiedFootGround(context, cooked, ground_height, base_vz0, ConvergedConfig());
    ASSERT_GE(out.row_count, 4u);

    // (A) Per-foot constraint satisfaction: the post-solve normal velocity is
    // driven near the compliant target (~0 / slightly separating) -- |Jv_after| is
    // a small fraction of |Jv_before|. This is the velocity-solve's actual job.
    double max_jv_after = 0.0;
    double max_jv_before = 0.0;
    double sum_lambda = 0.0;
    for (uint32_t r = 0u; r < out.row_count; ++r) {
        std::printf("[diag] (A) foot %u: Jv_before=%+.5f Jv_after=%+.5f lambda=%.6f\n",
                    r, out.jv_before[r], out.jv_after[r], out.foot_lambda[r]);
        max_jv_after = std::max(max_jv_after, std::fabs((double)out.jv_after[r]));
        max_jv_before = std::max(max_jv_before, std::fabs((double)out.jv_before[r]));
        sum_lambda += out.foot_lambda[r];
        // Each foot was CLOSING before (penetrating + approaching) and is no longer
        // closing after (target >= 0): the compliant non-penetration row did its job.
        EXPECT_LT(out.jv_before[r], 0.0f) << "foot " << r << " was not closing pre-solve";
        EXPECT_GT(out.jv_after[r], out.jv_before[r])
            << "foot " << r << " normal velocity not corrected toward separation";
    }
    EXPECT_LT(max_jv_after, 0.2 * max_jv_before)
        << "post-solve foot normal velocities not driven near the compliant target";

    // The response went ~entirely into the FREE legs (tiny foot effective mass),
    // which is WHY Sigma(lambda_n) is small -- not a missing base reaction.
    double base_delta = 0.0, leg_delta = 0.0;
    for (uint32_t c = 0u; c < 6u; ++c) base_delta += std::fabs(out.qdot[c] - out.qdot0[c]);
    for (uint32_t c = 6u; c < kDof; ++c) leg_delta += std::fabs(out.qdot[c] - out.qdot0[c]);
    std::printf("[diag] (A) |base qdot delta|=%.5f  |leg qdot delta|=%.5f\n",
                base_delta, leg_delta);
    std::printf("[diag] (A) base[0..5] delta = (%+.5f,%+.5f,%+.5f,%+.5f,%+.5f,%+.5f)\n",
                out.qdot[0] - out.qdot0[0], out.qdot[1] - out.qdot0[1],
                out.qdot[2] - out.qdot0[2], out.qdot[3] - out.qdot0[3],
                out.qdot[4] - out.qdot0[4], out.qdot[5] - out.qdot0[5]);

    // REPORTED, NOT ASSERTED: the equilibrium force balance Sigma(lambda_n)~m*g*dt
    // is a steady-state property deferred to C5c-3 (integrator + rollout). A single
    // free-leg velocity solve does not reach it.
    std::printf("[diag] (A) REPORT-ONLY (deferred to C5c-3): total mass=%.4f kg  "
                "m*g*dt=%.6f  Sigma lambda_n=%.6f  ratio=%.4f\n",
                total_mass, mg_dt, sum_lambda, sum_lambda / mg_dt);

    // The base DID feel the reaction (the 18-wide J coupling), and BOTH the base
    // AND the legs moved (the load shared through the full reduced-coordinate tree).
    EXPECT_GT(base_delta, 1.0e-3) << "base did not feel the contact (18-wide J?)";
    EXPECT_GT(leg_delta, 1.0e-3) << "legs did not absorb the contact response";
    EXPECT_GT(out.qdot[5], out.qdot0[5]) << "base-z not corrected upward";
}

// ===========================================================================
// (C) DETERMINISM (D1): two identical unified-spine runs produce byte-identical
//   solved qdot AND lambda. The C5b articulation arm + the new articulation-only
//   solve path must stay deterministic.
// ===========================================================================
TEST(FootGroundSubsume, UnifiedSpineDeterministicTwoRun) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    cooked.host.base_pose[0].rotation = Quat::Identity();
    for (auto& v : cooked.host.link_velocity) for (float& c : v.v) c = 0.0f;
    for (float& qd : cooked.host.qdot) qd = 0.0f;
    const float ground_height = SeatGround(context, cooked);

    const auto a = RunUnifiedFootGround(context, cooked, ground_height, -1.0f, ConvergedConfig());
    const auto b = RunUnifiedFootGround(context, cooked, ground_height, -1.0f, ConvergedConfig());

    ASSERT_EQ(a.qdot.size(), b.qdot.size());
    ASSERT_EQ(a.foot_lambda.size(), b.foot_lambda.size());
    EXPECT_EQ(std::memcmp(a.qdot.data(), b.qdot.data(), a.qdot.size() * sizeof(float)), 0)
        << "two identical unified-spine runs produced different qdot (nondeterministic)";
    EXPECT_EQ(std::memcmp(a.foot_lambda.data(), b.foot_lambda.data(),
                          a.foot_lambda.size() * sizeof(float)),
              0)
        << "two identical unified-spine runs produced different lambda (nondeterministic)";
    std::printf("[diag] (C) determinism: qdot+lambda byte-identical across 2 runs\n");
}
