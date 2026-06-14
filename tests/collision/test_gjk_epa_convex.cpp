// ---------------------------------------------------------------------------
// v0.8 C3c -- convex narrowphase (GJK / EPA / face-clip) tests.
// ---------------------------------------------------------------------------
// M9 T11 re-point: the legacy host dispatch (narrowphase_dispatch.hpp,
// ResolveNarrowphase / NarrowphaseConvex) was deleted; the KEPT geometry math is
// the HD `cvx::` header (collision/convex_narrowphase.hpp). These tests now call
// the cvx:: handlers DIRECTLY (cvx::ConvexNarrowphase / cvx::SphereHull /
// cvx::HullPlane), mirroring what NarrowphaseConvex did (plane special-case,
// sphere special-case, general GJK/EPA path). The dispatch-routing-only test
// (RoutesToRealConvexHandler) is gone (it tested deleted infra); EVERY geometry
// assertion below survives unchanged:
//   - two penetrating tetrahedra (as convex hulls) -> EPA depth + normal vs a
//     hand oracle.
//   - two penetrating boxes (as convex hulls) -> depth/normal vs analytic, AND a
//     TIGHT cross-check of depth + normal against the C3b box-box SAT result
//     (point sets/keys intentionally NOT compared -- different feature schemes).
//   - a clearly-separated pair -> GJK reports no overlap (empty manifold).
//   - convex (hull) x box / convex (hull) x sphere -> correctness.
//   - convex (hull) x plane (both orderings) -> the plane special path + sign.
//   - sphere x hull shallow-pen MONOTONICITY (the UNIQUE C3-hardening regression;
//     cvx::SphereHull directly).
//   - 2-run bit-identity (memset+memcmp on the whole manifold) + N>=32 cross-
//     replica (>=32 identical pairs -> byte-identical manifolds) on a SYMMETRIC,
//     tie-prone box-box-as-hull config (the EPA-determinism gate).
// ---------------------------------------------------------------------------

#include "collision/convex_narrowphase.hpp"
#include "collision/analytical_manifold.hpp"
#include "constraint/contact_manifold.hpp"
#include "import/usd_importer.hpp"        // cup hull reproducer (C3 hardening regression)
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cooker.hpp"

// M5 (plan): the SDF precision oracle — drive the NarrowphaseSdf op (SAMP point
// set x cooked sparse SDF grid) against ANALYTIC truth (sphere x sphere /
// sphere x box at known separations / penetrations), tolerance <= the bake cell
// size. The GJK/EPA tests above survive to M9; this is the ADDED SDF-path oracle.
// NOTE (M9 T11): device_context.hpp (which pulls <cuda_runtime.h> -> __forceinline__
// / cudaStream_t) MUST precede sdf_types.cuh; the deleted narrowphase_dispatch.hpp
// formerly pulled it in first (via candidate_pair.hpp) — order it explicitly now.
#include "phi/device_context.hpp"
#include "phi/backend.hpp"
#include "phi/buffer.hpp"
#include "phi/buffer_transfer_v2.hpp"
#include "phi/backend_cuda/ops/sdf_types.cuh"  // SdfPairDev + LaunchNarrowphaseSdf
#include "runtime/sdf/sparse_sdf_query.cuh"     // PackSdfCellKey

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using nuka::collision::amf::BuildPrimFrame;
using nuka::collision::amf::PrimParams;
using nuka::collision::cvx::ConvexHullView;
using nuka::collision::cvx::SupportKind;
using nuka::collision::cvx::SupportProxy;
using nuka::constraint::ContactManifold;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::scene::ShapeType;

namespace {

constexpr float kTol = 2.0e-3f;  // EPA converges to kEpaTol=1e-4; slack for accum.

// --- Hull backing storage (must outlive the ConvexHullView pointing at it). ---
// A unit AXIS-ALIGNED box's 8 corners in mesh-local space (half-extent `he`),
// in the C3b corner order (bit0=x,bit1=y,bit2=z) -- order does not matter to the
// support scan, but keep it fixed for determinism clarity.
std::vector<float> BoxVerts(Vec3 he) {
    std::vector<float> v;
    for (uint32_t c = 0; c < 8u; ++c) {
        const float sx = (c & 1u) ? he.x : -he.x;
        const float sy = (c & 2u) ? he.y : -he.y;
        const float sz = (c & 4u) ? he.z : -he.z;
        v.push_back(sx); v.push_back(sy); v.push_back(sz);
    }
    return v;
}

// A regular-ish tetrahedron's 4 verts in mesh-local space (around the origin).
std::vector<float> TetraVerts(float s) {
    // 4 verts of a tetra inscribed in a cube (the standard alternating corners).
    return {
         s,  s,  s,
         s, -s, -s,
        -s,  s, -s,
        -s, -s,  s,
    };
}

ConvexHullView MakeHull(const std::vector<float>& verts, Vec3 pos,
                        Quat rot = Quat::Identity()) {
    ConvexHullView h;
    h.verts = verts.data();
    h.vcount = static_cast<uint32_t>(verts.size() / 3u);
    h.frame = BuildPrimFrame(Transform{pos, rot});
    return h;
}

PrimParams MakeBoxPrim(Vec3 he, Vec3 pos, Quat rot = Quat::Identity()) {
    PrimParams p;
    p.half_extents = he;
    p.frame = BuildPrimFrame(Transform{pos, rot});
    return p;
}
PrimParams MakeSpherePrim(float r, Vec3 pos) {
    PrimParams p;
    p.radius = r;
    p.frame = BuildPrimFrame(Transform{pos, Quat::Identity()});
    return p;
}
PrimParams MakePlanePrim(Vec3 pos, Quat rot = Quat::Identity()) {
    PrimParams p;
    p.frame = BuildPrimFrame(Transform{pos, rot});
    return p;
}
PrimParams MakeCapsulePrim(float r, float hh, Vec3 pos, Quat rot = Quat::Identity()) {
    PrimParams p;
    p.radius = r;
    p.half_height = hh;
    p.frame = BuildPrimFrame(Transform{pos, rot});
    return p;
}

// --- Independent penetration-depth oracle (the GENERAL definition, no closed
// form needed). For two convex sets and a unit direction `n`, the overlap along
// n is support_A(-n) + support_B(n) where support_X(d) = max over X's WORLD verts
// of v.Dot(d). The TRUE penetration depth is the overlap along the MTV axis (the
// emitted normal), and for ANY axis the overlap is an UPPER bound on the true
// depth (no axis separates tighter than the MTV) -> so depth_along(any u) >= true
// depth. This is the brute-force oracle EPA's face-plane distance must match. */
float SupportMaxWorld(const std::vector<float>& verts, Vec3 pos, Vec3 dir) {
    float best = -3.4e38f;
    const uint32_t n = static_cast<uint32_t>(verts.size() / 3u);
    for (uint32_t i = 0; i < n; ++i) {
        const Vec3 w{verts[i * 3 + 0] + pos.x, verts[i * 3 + 1] + pos.y,
                     verts[i * 3 + 2] + pos.z};
        const float d = w.Dot(dir);
        if (d > best) best = d;
    }
    return best;
}
// Overlap of two AXIS-ALIGNED-placed hulls (verts in local + translation pos)
// along direction `n`: support_A(-n) + support_B(+n).
float OverlapAlong(const std::vector<float>& va, Vec3 pa,
                   const std::vector<float>& vb, Vec3 pb, Vec3 n) {
    return SupportMaxWorld(va, pa, -n) + SupportMaxWorld(vb, pb, n);
}

// Build a cvx::SupportProxy for one side from its shape type + geometry payload
// (hull view OR inline primitive params), mirroring the dispatch's MakeConvexProxy.
SupportProxy MakeProxy(ShapeType t, const ConvexHullView* hull,
                       const PrimParams& prim) {
    SupportProxy p;
    switch (t) {
        case ShapeType::ConvexHull:
        case ShapeType::TriMesh:
        case ShapeType::HeightField:
            p.kind = SupportKind::Hull;   p.hull = hull; break;
        case ShapeType::Box:
            p.kind = SupportKind::Box;    p.prim = &prim; break;
        case ShapeType::Sphere:
            p.kind = SupportKind::Sphere; p.prim = &prim; break;
        case ShapeType::Capsule:
            p.kind = SupportKind::Capsule; p.prim = &prim; break;
        case ShapeType::Plane:
            p.kind = SupportKind::Box;    p.prim = &prim; break;  // unreached
    }
    return p;
}

// Route a pair through the KEPT cvx:: handlers DIRECTLY, faithfully reproducing
// NarrowphaseConvex's three branches (plane special-case, sphere special-case,
// general GJK/EPA). The manifold is memset to 0 first so a 2-run memcmp covers
// padding + unused point slots (the EPA D1 gate). hull_a/hull_b are the hull-side
// geometry (nullptr when that side is a primitive); prim_a/prim_b the inline prim
// params (default for the hull side).
ContactManifold Route(ShapeType ta, const ConvexHullView* hull_a, PrimParams prim_a,
                      ShapeType tb, const ConvexHullView* hull_b, PrimParams prim_b) {
    ContactManifold out;
    std::memset(&out, 0, sizeof(out));
    // --- Plane special case (either side). ---
    if (ta == ShapeType::Plane || tb == ShapeType::Plane) {
        const bool plane_is_a = (ta == ShapeType::Plane);
        const ConvexHullView* hull = plane_is_a ? hull_b : hull_a;
        const PrimParams& plane = plane_is_a ? prim_a : prim_b;
        if (hull == nullptr) { out.Clear(); return out; }
        const Vec3 plane_n =
            nuka::collision::amf::Norm(plane.frame.cy, Vec3::UnitY());
        const Vec3 normal_for_hull = plane_is_a ? Vec3{-plane_n.x, -plane_n.y, -plane_n.z}
                                                : plane_n;
        nuka::collision::cvx::HullPlane(*hull, plane, normal_for_hull, &out);
        return out;
    }
    // --- Sphere special case (either side) vs a hull. ---
    if (ta == ShapeType::Sphere || tb == ShapeType::Sphere) {
        const bool sphere_is_a = (ta == ShapeType::Sphere);
        const ShapeType hull_t = sphere_is_a ? tb : ta;
        const ConvexHullView* hull = sphere_is_a ? hull_b : hull_a;
        if (hull_t == ShapeType::ConvexHull || hull_t == ShapeType::TriMesh ||
            hull_t == ShapeType::HeightField) {
            if (hull == nullptr) { out.Clear(); return out; }
            const PrimParams& sphere = sphere_is_a ? prim_a : prim_b;
            nuka::collision::cvx::SphereHull(sphere, *hull, sphere_is_a, &out);
            return out;
        }
    }
    // --- General GJK/EPA/face-clip path. ---
    SupportProxy A = MakeProxy(ta, hull_a, prim_a);
    SupportProxy B = MakeProxy(tb, hull_b, prim_b);
    if ((A.kind == SupportKind::Hull && A.hull == nullptr) ||
        (B.kind == SupportKind::Hull && B.hull == nullptr)) {
        out.Clear(); return out;
    }
    nuka::collision::cvx::ConvexNarrowphase(A, B, &out);
    return out;
}

// Route a hull-vs-hull pair through the general cvx path (memset out first).
ContactManifold RouteHullHull(const ConvexHullView& a, const ConvexHullView& b) {
    return Route(ShapeType::ConvexHull, &a, PrimParams{},
                 ShapeType::ConvexHull, &b, PrimParams{});
}

bool Vec3Near(Vec3 a, Vec3 b, float tol = kTol) {
    return std::fabs(a.x - b.x) < tol && std::fabs(a.y - b.y) < tol &&
           std::fabs(a.z - b.z) < tol;
}

// --- Cup hull loader (C3 hardening regression reproducer; mirrors the grasp
// spike's LoadCupHull). The cup is a fetch-per-env asset (.nuka-assets gitignored);
// the regression SKIPs when absent. ----------------------------------------------
const std::string kCupUsda =
    ".nuka-assets/newton_assets/manipulation_objects/cup/model.usda";

bool CupHullAvailable() { return std::filesystem::exists(kCupUsda); }

// Load the C7a cup -> a SINGLE cooked ConvexHull, recenter its verts about the
// bbox center (the grasp poses the cup with its center at BodyState.position), and
// report the +X support-x (the wall the fingertip pinches). Out verts are flat
// x,y,z triples, mesh-local, centered.
void LoadCupHullCenteredAndFaceX(std::vector<float>* verts, float* face_x) {
    auto scene = nuka::import::LoadUsd(kCupUsda);
    for (size_t i = 0; i < scene.ShapeCount(); ++i) {
        auto& s = scene.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        if (!s.mesh_vertices.empty())
            s.decompose_mode = nuka::scene::DecomposeMode::Skip;  // single hull.
    }
    const auto blob = nuka::scene::CookScene(scene);
    const auto& cg = blob.convex_geometry;
    verts->clear();
    *face_x = 0.0f;
    if (cg.vertex_counts.empty()) return;
    const uint32_t voff = cg.vertex_offsets[0];
    const uint32_t vcnt = cg.vertex_counts[0];
    verts->assign(cg.vertices.begin() + voff * 3u,
                  cg.vertices.begin() + (voff + vcnt) * 3u);
    Vec3 lo{(*verts)[0], (*verts)[1], (*verts)[2]}, hi = lo;
    for (uint32_t i = 0u; i < vcnt; ++i) {
        const Vec3 v{(*verts)[i * 3u], (*verts)[i * 3u + 1u], (*verts)[i * 3u + 2u]};
        lo = Vec3{std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
        hi = Vec3{std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
    }
    const Vec3 ctr = (lo + hi) * 0.5f;
    float fx = -3.4e38f;
    for (uint32_t i = 0u; i < vcnt; ++i) {
        (*verts)[i * 3u + 0u] -= ctr.x;
        (*verts)[i * 3u + 1u] -= ctr.y;
        (*verts)[i * 3u + 2u] -= ctr.z;
        if ((*verts)[i * 3u + 0u] > fx) fx = (*verts)[i * 3u + 0u];
    }
    *face_x = fx;
}

}  // namespace

// ===========================================================================
// Two penetrating tetrahedra -> EPA depth/normal vs hand oracle.
// ===========================================================================
// Two tetra (vert reach 0.5 along each axis from BoxVerts-style corners). Place
// them overlapping along +X: A at origin, B shifted -dx so they interpenetrate.
TEST(GjkEpaConvex, TetraTetraPenetration) {
    const Vec3 pa{0.0f, 0.0f, 0.0f};
    const Vec3 pb{0.6f, 0.0f, 0.0f};
    const auto va = TetraVerts(0.5f);
    const auto vb = TetraVerts(0.5f);
    // A centered at origin, B centered at (0.6,0,0): the tetras' x-extent is
    // [-0.5,0.5]; B's leftmost vert reaches 0.6-0.5=0.1 < A's 0.5 -> overlap.
    const ConvexHullView A = MakeHull(va, pa);
    const ConvexHullView B = MakeHull(vb, pb);
    const ContactManifold m = RouteHullHull(A, B);
    ASSERT_GT(m.point_count, 0u) << "expected a contact";
    const Vec3 n = m.points[0].normal;
    // (1) normal is ~unit.
    EXPECT_NEAR(std::sqrt(n.LengthSq()), 1.0f, 1.0e-2f);
    // (2) DEPTH ORACLE: the emitted penetration must equal the brute-force overlap
    //     of the two vertex sets along the emitted normal (the general definition;
    //     this proves EPA actually CONVERGED on a triangular face, not just that
    //     depth>0). OverlapAlong(.,n) == support_A(-n)+support_B(n).
    const float d_oracle = OverlapAlong(va, pa, vb, pb, n);
    EXPECT_NEAR(m.points[0].penetration, d_oracle, kTol)
        << "EPA depth must match the support-difference oracle along the normal";
    // (3) NORMAL is the MINIMUM-translation axis: no sampled unit axis separates
    //     tighter than the emitted depth (overlap along ANY axis >= true depth).
    const Vec3 axes[7] = {Vec3{1,0,0}, Vec3{0,1,0}, Vec3{0,0,1},
                          Vec3{0.577f,0.577f,0.577f}, Vec3{-0.577f,0.577f,0.577f},
                          Vec3{0.707f,0.707f,0.0f}, Vec3{0.0f,0.707f,0.707f}};
    for (const Vec3& u : axes) {
        const float ov = OverlapAlong(va, pa, vb, pb, u);
        EXPECT_GE(ov, m.points[0].penetration - kTol)
            << "found an axis separating tighter than the EPA normal (n not MTV)";
    }
    EXPECT_GT(m.points[0].penetration, 0.0f);
}

// ===========================================================================
// Two penetrating boxes (as hulls) -> depth/normal vs analytic + C3b SAT.
// ===========================================================================
TEST(GjkEpaConvex, BoxBoxAsHull_DepthNormalVsAnalytic) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const auto vb = BoxVerts(he);
    // A at origin, B at (0.8,0,0): overlap on X = (0.5+0.5) - 0.8 = 0.2. Faces are
    // axis-aligned -> reference/incident on the +/-X faces, normal ~ +/-X.
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
    const ConvexHullView B = MakeHull(vb, Vec3{0.8f, 0.0f, 0.0f});
    const ContactManifold m = RouteHullHull(A, B);
    ASSERT_GT(m.point_count, 0u);
    // Analytic: depth = 0.2, normal = sep dir for A = -X (A must move -X off B).
    const float kExpectDepth = 0.2f;
    const Vec3  kExpectNormalA{-1.0f, 0.0f, 0.0f};
    for (uint32_t i = 0; i < m.point_count; ++i) {
        EXPECT_NEAR(m.points[i].penetration, kExpectDepth, kTol);
        EXPECT_TRUE(Vec3Near(m.points[i].normal, kExpectNormalA, 1.0e-2f))
            << "n=(" << m.points[i].normal.x << "," << m.points[i].normal.y
            << "," << m.points[i].normal.z << ")";
    }
    EXPECT_EQ(m.point_count, 4u) << "axis-aligned face-face -> 4-pt patch";

    // --- TIGHT cross-check vs C3b box-box SAT (depth + normal only). ---
    using nuka::collision::amf::BoxBox;
    PrimParams pa = MakeBoxPrim(he, Vec3{0.0f, 0.0f, 0.0f});
    PrimParams pb = MakeBoxPrim(he, Vec3{0.8f, 0.0f, 0.0f});
    ContactManifold sat;
    BoxBox(pa, pb, &sat);
    ASSERT_GT(sat.point_count, 0u);
    EXPECT_NEAR(sat.points[0].penetration, kExpectDepth, kTol);
    // Both paths agree on depth + normal direction (NOT point ids/sets).
    EXPECT_NEAR(m.points[0].penetration, sat.points[0].penetration, kTol);
    EXPECT_TRUE(Vec3Near(m.points[0].normal, sat.points[0].normal, 1.0e-2f));
}

// ===========================================================================
// Separated pair -> GJK reports no overlap (empty manifold).
// ===========================================================================
TEST(GjkEpaConvex, SeparatedPairNoContact) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const auto vb = BoxVerts(he);
    // A at origin, B at (3,0,0): gap = 3 - 1 = 2.0 (clearly separated, far from
    // the iteration cap's near-touch false-positive zone).
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
    const ConvexHullView B = MakeHull(vb, Vec3{3.0f, 0.0f, 0.0f});
    const ContactManifold m = RouteHullHull(A, B);
    EXPECT_EQ(m.point_count, 0u) << "separated -> no contact";
}

// ===========================================================================
// Convex (hull) x box (primitive) -> correctness.
// ===========================================================================
TEST(GjkEpaConvex, HullVsBoxPrimitive) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
    const PrimParams B = MakeBoxPrim(he, Vec3{0.0f, 0.8f, 0.0f});  // overlap on Y=0.2
    const ContactManifold m = Route(ShapeType::ConvexHull, &A, PrimParams{},
                                    ShapeType::Box, nullptr, B);
    ASSERT_GT(m.point_count, 0u);
    for (uint32_t i = 0; i < m.point_count; ++i) {
        EXPECT_NEAR(m.points[i].penetration, 0.2f, kTol);
        // sep dir for A (hull) = -Y (hull pushes down off the box above it).
        EXPECT_TRUE(Vec3Near(m.points[i].normal, Vec3{0.0f, -1.0f, 0.0f}, 1.0e-2f));
    }
}

// ===========================================================================
// Convex (hull) x sphere (primitive) -> 1-pt witness (sphere has no face).
// ===========================================================================
TEST(GjkEpaConvex, HullVsSpherePrimitive) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
    // Sphere radius 0.5 centered at (0.8,0,0): nearest box face is +X at x=0.5,
    // sphere left edge at 0.3 -> overlap 0.2 along +X.
    const PrimParams B = MakeSpherePrim(0.5f, Vec3{0.8f, 0.0f, 0.0f});
    const ContactManifold m = Route(ShapeType::ConvexHull, &A, PrimParams{},
                                    ShapeType::Sphere, nullptr, B);
    ASSERT_EQ(m.point_count, 1u) << "sphere side has no face -> 1-pt witness";
    EXPECT_NEAR(m.points[0].penetration, 0.2f, kTol);
    // sep dir for A (hull) = -X (hull pushes away from sphere on +X side).
    EXPECT_TRUE(Vec3Near(m.points[0].normal, Vec3{-1.0f, 0.0f, 0.0f}, 1.0e-2f));
}

// ===========================================================================
// Sphere CENTER INSIDE the hull (deep overlap) -> the SphereHull inside-fallback
// must return a SANE bounded manifold (NOT garbage / NOT a crash). The grasp never
// reaches this (the fingertip center stays outside the cup wall), so this is a
// smoke/robustness bound on the spec-required center-inside branch, not the
// validated shallow path. A unit-cube hull (he=0.5) with a small sphere (r=0.05)
// at the hull CENTER: the center is deep inside, the shallowest face is 0.5 away,
// so penetration = r + inside_depth ~ 0.55 (> radius), normal ~unit and axis-aligned.
// ===========================================================================
TEST(GjkEpaConvex, SphereCenterInsideHull_SaneBoundedManifold) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto vh = BoxVerts(he);
    const ConvexHullView H = MakeHull(vh, Vec3{0.0f, 0.0f, 0.0f});
    const PrimParams S = MakeSpherePrim(0.05f, Vec3{0.0f, 0.0f, 0.0f});  // center inside
    const ContactManifold m = Route(ShapeType::Sphere, nullptr, S,
                                    ShapeType::ConvexHull, &H, PrimParams{});
    ASSERT_EQ(m.point_count, 1u) << "center-inside must still emit a contact";
    const float pen = m.points[0].penetration;
    EXPECT_TRUE(std::isfinite(pen)) << "center-inside penetration is non-finite (garbage)";
    EXPECT_GT(pen, 0.05f) << "penetration must exceed the radius (radius + inside depth)";
    EXPECT_NEAR(pen, 0.55f, 1.0e-2f) << "pen = r(0.05) + shallowest face depth(0.5)";
    const Vec3 n = m.points[0].normal;
    EXPECT_NEAR(std::sqrt(n.LengthSq()), 1.0f, 1.0e-2f) << "normal must be ~unit";
    // The shallowest exit from the cube center is some +/- axis face (all 0.5 away;
    // the deterministic scan picks one) -> the normal is an axis direction.
    const float ax = std::fabs(n.x), ay = std::fabs(n.y), az = std::fabs(n.z);
    EXPECT_GT(std::max(ax, std::max(ay, az)), 0.98f)
        << "center-inside exit normal must be ~axis-aligned for a cube hull";
}

// ===========================================================================
// C3 HARDENING REGRESSION: sphere x convex-hull detection is MONOTONE in depth.
// ===========================================================================
// THE BUG (C7b-1 grasp dropout): a fingertip SPHERE (r=0.008) pinching the cup
// CONVEX HULL fed the sphere through the general boolean-GJK->EPA path. EPA has a
// SHALLOW-PENETRATION non-monotonicity on the sphere's CURVED support: a sphere
// whose CENTER is OUTSIDE the hull (only the radius penetrating) was detected at
// pen~0.07mm, DROPPED OUT across pen~1.6-2.8mm, then re-detected from ~2.87mm --
// detect->drop->detect (probe-measured on the ACTUAL cup hull). The cup's tiny
// equator tilt walked a contact into that dead band periodically -> a per-21-step
// single-finger contact dropout. The fix special-cases sphere x hull via the EXACT
// point-to-convex distance (closest point on the hull to the sphere center),
// bypassing EPA -> MONOTONE at all depths.
//
// THIS TEST locks the fix on the EXACT reproducer (the real cup hull -- a synthetic
// box/cylinder hull does NOT reproduce the dead band: probed, the old EPA path is
// monotone on a clean axis-aligned/symmetric hull; only the irregular V-HACD cup
// hull triggers it on a STRAIGHT march). MARCH the fingertip sphere (r=0.008)
// STRAIGHT along -X into the cup's +X wall from SEPARATED, through the OLD dead band
// (0->~6mm penetration, center-OUTSIDE only so the analytic depth is monotone), in
// sub-mm steps. ASSERT detection is MONOTONE: once detected it STAYS detected, and
// the reported penetration is monotonically NON-DECREASING (no detect->drop->detect).
//
// BITE: against the OLD path (sphere routed back through general GJK->EPA) the march
// goes detect -> point_count==0 for 7 consecutive samples (pen 1.60->2.80mm) ->
// detect again -> this test FAILS (a detected contact dropped out). Through the
// special-case path it is GREEN.
//
// CI-TEETH DEBT (named): this regression is ASSET-GATED (mirrors the grasp spike) --
// it SKIPs when the cup is absent. So in a CLEAN CI CLONE (no .nuka-assets fetch) a
// revert of the SphereHull fix would show GREEN here: NO always-on test catches a
// SphereHull monotonicity revert. (HullVsSpherePrimitive does NOT cover this: it
// checks ONE depth on a clean box, not monotonicity across the dead band, and uses
// the special-case path -- it cannot bite a revert.) A time-boxed hardening attempt
// (advisor #2, v0.8 review batch) tried to reproduce the dead band on a SMALL
// SYNTHETIC IRREGULAR hull (an asymmetrically-perturbed slanted-wall prism) marched
// 0->6mm in 0.2mm steps, comparing the direct EPA path (sphere SupportProxy ->
// cvx::ConvexNarrowphase) vs cvx::SphereHull: the OLD/EPA path detected at contact
// onset (~0.2mm) and NEVER dropped across the 0->6mm march (no detect->drop->detect),
// so a synthetic hull did not reproduce the V-HACD cup's dead band -- consistent with
// "only the irregular cup hull triggers it" (the head-on march likely hit a locally
// near-symmetric edge at y=0). Per the one-attempt
// time-box this was NOT iterated; the regression stays asset-gated = the CI-teeth debt
// remains open (closes when the cup is in CI, or a future irregular reproducer lands).
//
// Runs BOTH sphere-as-A and sphere-as-B (the kSph][kHull + kHull][kSph slots + sign).
TEST(GjkEpaConvex, SphereHullShallowPenetrationIsMonotone) {
    if (!CupHullAvailable())
        GTEST_SKIP() << "newton-assets cup not present (fetch-per-env)";
    // Load the actual cooked cup hull (centered about its bbox center) -- the only
    // geometry that reproduces the EPA shallow-penetration dead band.
    std::vector<float> verts;
    float face_x = 0.0f;
    LoadCupHullCenteredAndFaceX(&verts, &face_x);
    ASSERT_GT(verts.size(), 0u);
    ConvexHullView H;
    H.verts = verts.data();
    H.vcount = static_cast<uint32_t>(verts.size() / 3u);
    H.frame = BuildPrimFrame(Transform{Vec3{0.0f, 0.0f, 0.0f}, Quat::Identity()});

    constexpr float kR = 0.008f;            // fingertip sphere radius.
    // March the sphere CENTER toward the +X cup wall along -X. Contact begins when
    // (cx - r) < face_x i.e. cx < face_x + r. We sweep cx from face_x+r+5mm
    // (separated) down to face_x+r-6mm (penetration 6mm), in 0.0002 m (0.2mm) steps
    // -- sub-mm resolution straight through the OLD 1.6-2.8mm dead band. The center
    // stays OUTSIDE (max pen 6mm < radius 8mm) so the analytic depth is monotone.
    const float cx_start = face_x + kR + 0.005f;     // 5mm separated
    auto run = [&](bool sphere_is_a, float cx) -> ContactManifold {
        const PrimParams S = MakeSpherePrim(kR, Vec3{cx, 0.0f, 0.0f});
        if (sphere_is_a) {
            return Route(ShapeType::Sphere, nullptr, S,
                         ShapeType::ConvexHull, &H, PrimParams{});
        }
        return Route(ShapeType::ConvexHull, &H, PrimParams{},
                     ShapeType::Sphere, nullptr, S);
    };

    for (int side = 0; side < 2; ++side) {
        const bool sphere_is_a = (side == 0);
        bool   detected = false;
        float  last_pen = -1.0f;
        int    steps = 0, contact_steps = 0;
        // cx from cx_start down to (face_x + kR - 0.006), 0.2mm steps (55 samples).
        for (int i = 0; i <= 55; ++i) {
            const float cx = cx_start - 0.0002f * static_cast<float>(i);
            const ContactManifold m = run(sphere_is_a, cx);
            ++steps;
            const float true_pen = face_x + kR - cx;   // analytic penetration (m).
            if (m.point_count > 0u) {
                ++contact_steps;
                const float pen = m.points[0].penetration;
                detected = true;
                // (b) penetration POSITIVE and ~matches the analytic depth (the
                //     closest-point query is exact: pen = r - |center - wall|).
                EXPECT_GT(pen, 0.0f) << "side " << side << " cx=" << cx;
                EXPECT_NEAR(pen, true_pen, 1.5e-3f)
                    << "side " << side << " cx=" << cx << " pen!=analytic depth";
                // (c) MONOTONE NON-DECREASING in depth as the sphere advances.
                if (last_pen >= 0.0f) {
                    EXPECT_GE(pen, last_pen - 1.0e-4f)
                        << "side " << side << " cx=" << cx
                        << " penetration DECREASED as the sphere went deeper "
                           "(non-monotone EPA dead band: " << last_pen << "->" << pen << ")";
                }
                last_pen = pen;
                // (d) NORMAL is +/-X (sep dir for side A); ~unit.
                const Vec3 n = m.points[0].normal;
                EXPECT_NEAR(std::sqrt(n.LengthSq()), 1.0f, 1.0e-2f);
                const Vec3 expect = sphere_is_a ? Vec3{1.0f, 0.0f, 0.0f}
                                                : Vec3{-1.0f, 0.0f, 0.0f};
                EXPECT_TRUE(Vec3Near(n, expect, 5.0e-2f))
                    << "side " << side << " cx=" << cx << " wrong normal";
            } else {
                // (a) THE BITE: a gap AFTER a detection == the dead-band dropout.
                EXPECT_FALSE(detected)
                    << "side " << side << " cx=" << cx << " (true_pen=" << true_pen
                    << " m): contact DROPPED OUT after being detected -- the "
                       "sphere x hull shallow-penetration dead band regressed";
            }
        }
        // Once contact begins it must persist for the rest of the (deepening) march.
        EXPECT_GT(contact_steps, steps / 2)
            << "side " << side << " too few contact samples -- detection never "
               "established or dropped out across the march";
    }
}

// ===========================================================================
// Convex (hull) x capsule (primitive) -> exercises SupportCapsule + 1-pt witness.
// ===========================================================================
TEST(GjkEpaConvex, HullVsCapsulePrimitive) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
    // Capsule radius 0.25, half-height 0.5 (axis = local Y), centered at
    // (0.65,0,0): its nearest surface point is the radius-0.25 sphere swept along
    // Y; closest approach to the box +X face (x=0.5) is at x = 0.65-0.25 = 0.40 <
    // 0.5 -> overlap 0.10 along +X. (Capsule has no flat face -> 1-pt witness.)
    const PrimParams B = MakeCapsulePrim(0.25f, 0.5f, Vec3{0.65f, 0.0f, 0.0f});
    const ContactManifold m = Route(ShapeType::ConvexHull, &A, PrimParams{},
                                    ShapeType::Capsule, nullptr, B);
    ASSERT_EQ(m.point_count, 1u) << "capsule side has no face -> 1-pt witness";
    EXPECT_NEAR(m.points[0].penetration, 0.10f, kTol);
    // sep dir for A (hull) = -X (hull pushes away from the capsule on +X).
    EXPECT_TRUE(Vec3Near(m.points[0].normal, Vec3{-1.0f, 0.0f, 0.0f}, 1.0e-2f));
}

// ===========================================================================
// C3-batch review fix: capsule x box / capsule x capsule have NO closed-form, so
// SelectTier routes them to the CONVEX tier (two primitive SupportProxies through
// the SAME GJK/EPA). These pairs USED to fall to the Analytical stub (silent empty
// manifold) -- prove they now produce a real contact.
// ===========================================================================
TEST(GjkEpaConvex, CapsuleBoxPenetration) {
    // Both sides are PRIMITIVES (no hull): capsule(A) at origin, box(B) at +X.
    // capsule radius 0.25 (right surface x=0.25); box he 0.5 at x=0.6 (left face
    // x=0.10) -> overlap 0.15 along X. sep dir for A(capsule) = -X. capsule x box
    // has no closed form -> the general GJK/EPA (cvx) path (NOT the analytical stub).
    const PrimParams A = MakeCapsulePrim(0.25f, 0.5f, Vec3{0.0f, 0.0f, 0.0f});
    const PrimParams B = MakeBoxPrim(Vec3{0.5f, 0.5f, 0.5f}, Vec3{0.6f, 0.0f, 0.0f});
    const ContactManifold m = Route(ShapeType::Capsule, nullptr, A,
                                    ShapeType::Box, nullptr, B);
    ASSERT_GT(m.point_count, 0u) << "capsule x box must produce a contact, not the stub";
    EXPECT_NEAR(m.points[0].penetration, 0.15f, 1.0e-2f);
    EXPECT_TRUE(Vec3Near(m.points[0].normal, Vec3{-1.0f, 0.0f, 0.0f}, 1.0e-2f));
}

TEST(GjkEpaConvex, CapsuleCapsulePenetration) {
    // Two parallel (axis-Y) capsules, radius 0.3, centers 0.5 apart in X ->
    // combined radius 0.6 > 0.5 -> overlap 0.1 along X. sep dir for A = -X. No
    // closed form -> the general GJK/EPA (cvx) path (NOT the analytical stub).
    const PrimParams A = MakeCapsulePrim(0.3f, 0.5f, Vec3{0.0f, 0.0f, 0.0f});
    const PrimParams B = MakeCapsulePrim(0.3f, 0.5f, Vec3{0.5f, 0.0f, 0.0f});
    const ContactManifold m = Route(ShapeType::Capsule, nullptr, A,
                                    ShapeType::Capsule, nullptr, B);
    ASSERT_GT(m.point_count, 0u) << "capsule x capsule must produce a contact, not the stub";
    EXPECT_NEAR(m.points[0].penetration, 0.10f, 1.0e-2f);
    // Parallel-axis capsules => the CSO is degenerate (a line of equally-close
    // points along the shared Y axis), the EPA-hard case. The depth is exact and
    // the normal is -X to ~1% (a ~0.01 off-axis wobble on the degenerate CSO),
    // so widen the normal tol slightly -- this is EVIDENCE the EPA stays robust
    // on the parallel-degenerate config, not a sign error.
    EXPECT_TRUE(Vec3Near(m.points[0].normal, Vec3{-1.0f, 0.0f, 0.0f}, 2.0e-2f));
}

// ===========================================================================
// Convex (hull) x plane -> the plane special path, BOTH orderings (sign bug).
// ===========================================================================
TEST(GjkEpaConvex, HullVsPlane_BothOrders) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto vh = BoxVerts(he);
    // Box resting so its bottom corners dip below the plane at y=0 by 0.1:
    // box center at y = 0.4 -> bottom corners at y = -0.1 (penetration 0.1).
    const ConvexHullView H = MakeHull(vh, Vec3{0.0f, 0.4f, 0.0f});
    const PrimParams P = MakePlanePrim(Vec3{0.0f, 0.0f, 0.0f});  // plane normal +Y

    // Order 1: hull = A, plane = B. sep dir for A (hull) = +Y (push hull up).
    {
        const ContactManifold m = Route(ShapeType::ConvexHull, &H, PrimParams{},
                                        ShapeType::Plane, nullptr, P);
        ASSERT_GT(m.point_count, 0u);
        EXPECT_EQ(m.point_count, 4u) << "4 bottom corners below plane";
        for (uint32_t i = 0; i < m.point_count; ++i) {
            EXPECT_NEAR(m.points[i].penetration, 0.1f, kTol);
            EXPECT_TRUE(Vec3Near(m.points[i].normal, Vec3{0.0f, 1.0f, 0.0f}, 1.0e-2f));
        }
    }
    // Order 2: plane = A, hull = B. sep dir for A (plane) = -Y (plane pushes the
    // OTHER way; the swapped-slot sign test).
    {
        const ContactManifold m = Route(ShapeType::Plane, nullptr, P,
                                        ShapeType::ConvexHull, &H, PrimParams{});
        ASSERT_GT(m.point_count, 0u);
        for (uint32_t i = 0; i < m.point_count; ++i) {
            EXPECT_NEAR(m.points[i].penetration, 0.1f, kTol);
            EXPECT_TRUE(Vec3Near(m.points[i].normal, Vec3{0.0f, -1.0f, 0.0f}, 1.0e-2f))
                << "plane-as-A normal must be -Y (swapped sign)";
        }
    }
}

// ===========================================================================
// D1: 2-run bit-identity + N>=32 cross-replica on a SYMMETRIC tie-prone config.
// ===========================================================================
// Symmetric face-face box-box-as-hull stack: ALL clip points tie on penetration,
// so the EPA face scan + the FPS reducer are the sole determinants. A heap-based
// EPA would diverge here; the array-scan EPA must be byte-identical run-to-run.
TEST(GjkEpaConvex, Determinism_TwoRunAndCrossReplica) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const auto vb = BoxVerts(he);
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
    const ConvexHullView B = MakeHull(vb, Vec3{0.0f, 0.9f, 0.0f});  // overlap Y=0.1

    // 2-run bit identity (RouteHullHull memsets the manifold before the cvx call,
    // so the memcmp covers padding + unused point slots).
    const ContactManifold m1 = RouteHullHull(A, B);
    const ContactManifold m2 = RouteHullHull(A, B);
    ASSERT_GT(m1.point_count, 0u);
    EXPECT_EQ(std::memcmp(&m1, &m2, sizeof(ContactManifold)), 0)
        << "2-run manifold must be byte-identical (EPA determinism)";

    // N>=32 cross-replica: 40 identical pairs in one loop -> all byte-identical.
    constexpr int kReplicas = 40;
    const ContactManifold ref = RouteHullHull(A, B);
    for (int r = 0; r < kReplicas; ++r) {
        const ContactManifold mr = RouteHullHull(A, B);
        EXPECT_EQ(std::memcmp(&ref, &mr, sizeof(ContactManifold)), 0)
            << "replica " << r << " diverged (EPA determinism gate)";
    }
}

// ===========================================================================
// REGRESSION (C3c adversarial review): ORIGIN-ON-BOUNDARY EPA false-negative.
// ===========================================================================
// The MOST COMMON sim contact (flat-faced resting/stacking, origin exactly on the
// CSO boundary of the GJK seed tetra) used to return an EMPTY manifold from a
// DETERMINISTIC EPA churn (the seed tetra did not strictly enclose the origin ->
// dist==0 seed faces with inward normals -> non-monotone polytope -> face overflow
// -> stale depth 0 -> dropped). These tests pin the EXACT failing config + a SWEEP
// that exposes the data-dependence a single hand-picked config hides.

// (1) THE EXACT REPRODUCER: two unit boxes-as-hulls, A at origin, B at z=0.90
// (overlap 0.10 along +Z). Pre-fix: point_count==0. Post-fix: pen≈0.10, normal≈±Z.
TEST(GjkEpaConvex, OriginOnBoundary_ExactRepro_Z090) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const auto vb = BoxVerts(he);
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
    const ConvexHullView B = MakeHull(vb, Vec3{0.0f, 0.0f, 0.90f});  // overlap Z=0.10
    const ContactManifold m = RouteHullHull(A, B);
    ASSERT_GT(m.point_count, 0u)
        << "origin-on-boundary face-flush contact must NOT be an empty manifold";
    for (uint32_t i = 0; i < m.point_count; ++i) {
        EXPECT_NEAR(m.points[i].penetration, 0.10f, kTol);
        // sep dir for A = -Z (A must move -Z to escape B above it on +Z).
        EXPECT_TRUE(Vec3Near(m.points[i].normal, Vec3{0.0f, 0.0f, -1.0f}, 1.0e-2f))
            << "n=(" << m.points[i].normal.x << "," << m.points[i].normal.y << ","
            << m.points[i].normal.z << ")";
    }
    EXPECT_EQ(m.point_count, 4u) << "axis-aligned face-face -> 4-pt patch";
}

// (2) THE SWEEP (the real anti-regression): boxes-as-hulls overlapping by
// {0.05,0.10,0.15,0.20,0.30} along EACH of +X,+Y,+Z, PLUS a diagonal offset.
// Each -> correct depth (the brute-force support-difference oracle along the
// emitted normal) + a unit normal aligned with the contact axis. A single config
// hides the GJK-seed data-dependence; the whole grid is what catches it.
TEST(GjkEpaConvex, OriginOnBoundary_OverlapSweep_AllAxes) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const auto vb = BoxVerts(he);
    const float overlaps[5] = {0.05f, 0.10f, 0.15f, 0.20f, 0.30f};
    // Axis-aligned: the +X,+Y,+Z separations (center offset = 1.0 - overlap).
    const Vec3 axis_dir[3] = {Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}};
    const Vec3 expect_n[3] = {Vec3{-1, 0, 0}, Vec3{0, -1, 0}, Vec3{0, 0, -1}};
    int ok = 0;
    for (int ax = 0; ax < 3; ++ax) {
        for (int oi = 0; oi < 5; ++oi) {
            const float ov = overlaps[oi];
            const float sep = 1.0f - ov;                  // center-to-center
            const Vec3 pos = axis_dir[ax] * sep;
            const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
            const ConvexHullView B = MakeHull(vb, pos);
            const ContactManifold m = RouteHullHull(A, B);
            ASSERT_GT(m.point_count, 0u)
                << "axis " << ax << " overlap " << ov << " -> EMPTY (regression!)";
            // Depth oracle along the emitted normal (the general support-difference).
            const Vec3 n = m.points[0].normal;
            const float d_oracle = OverlapAlong(va, Vec3{0, 0, 0}, vb, pos, n);
            EXPECT_NEAR(m.points[0].penetration, d_oracle, kTol)
                << "axis " << ax << " overlap " << ov << " depth != oracle";
            EXPECT_NEAR(m.points[0].penetration, ov, kTol)
                << "axis " << ax << " overlap " << ov << " wrong depth";
            EXPECT_TRUE(Vec3Near(n, expect_n[ax], 2.0e-2f))
                << "axis " << ax << " overlap " << ov << " wrong normal";
            ++ok;
        }
    }
    // (2b) A DIAGONAL offset (the on-EDGE/on-vertex degeneracy is orientation-
    // dependent; an axis-only sweep can miss the diagonal variant). Two unit boxes
    // overlapping along the body diagonal: each axis overlaps 0.20 (sep 0.80 on all
    // three axes). The MTV is still an axis face (depth 0.20), but the GJK seed
    // straddles the origin differently than the pure-axis cases.
    {
        const Vec3 pos{0.80f, 0.80f, 0.80f};
        const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
        const ConvexHullView B = MakeHull(vb, pos);
        const ContactManifold m = RouteHullHull(A, B);
        ASSERT_GT(m.point_count, 0u) << "diagonal offset -> EMPTY (regression!)";
        const Vec3 n = m.points[0].normal;
        const float d_oracle = OverlapAlong(va, Vec3{0, 0, 0}, vb, pos, n);
        EXPECT_NEAR(m.points[0].penetration, d_oracle, kTol)
            << "diagonal depth != support-difference oracle along the normal";
        // The MTV depth equals the min per-axis overlap (0.20 on each axis here).
        EXPECT_NEAR(m.points[0].penetration, 0.20f, kTol);
        // No sampled axis separates tighter than the emitted depth (n is the MTV).
        const Vec3 probe[6] = {Vec3{1,0,0}, Vec3{0,1,0}, Vec3{0,0,1},
                               Vec3{-1,0,0}, Vec3{0,-1,0}, Vec3{0,0,-1}};
        for (const Vec3& u : probe) {
            const float ovx = OverlapAlong(va, Vec3{0, 0, 0}, vb, pos, u);
            EXPECT_GE(ovx, m.points[0].penetration - kTol)
                << "diagonal: an axis separates tighter than the EPA normal";
        }
        ++ok;
    }
    EXPECT_EQ(ok, 16) << "expected 15 axis configs + 1 diagonal";
}

// (3) FLUSH-FACE convex x convex: two NON-box convex hulls (hexagonal prisms)
// with a fully coplanar contact face overlapping slightly along +Z. This is the
// general (not-a-box) origin-on-boundary class -> must be non-empty with a +Z/-Z
// normal and the right depth. Proves the fix is geometry-general, not box-special.
TEST(GjkEpaConvex, OriginOnBoundary_FlushFaceConvexConvex) {
    // A hexagonal prism (6 side verts top + 6 bottom), half-height 0.5 along Z,
    // circumradius 0.5 in XY. Two coplanar hex caps -> a flush +Z/-Z contact.
    std::vector<float> hex;
    for (int z = -1; z <= 1; z += 2) {
        const float hz = 0.5f * static_cast<float>(z);
        for (int k = 0; k < 6; ++k) {
            const float ang = static_cast<float>(k) * (3.14159265f / 3.0f);
            hex.push_back(0.5f * std::cos(ang));
            hex.push_back(0.5f * std::sin(ang));
            hex.push_back(hz);
        }
    }
    // A at origin (top cap z=0.5), B at z=0.90 (bottom cap z=0.40) -> overlap 0.10
    // along Z, the two hex caps flush/coplanar in the overlap band.
    const ConvexHullView A = MakeHull(hex, Vec3{0.0f, 0.0f, 0.0f});
    const ConvexHullView B = MakeHull(hex, Vec3{0.0f, 0.0f, 0.90f});
    const ContactManifold m = RouteHullHull(A, B);
    ASSERT_GT(m.point_count, 0u)
        << "flush-face convex x convex must NOT be an empty manifold";
    for (uint32_t i = 0; i < m.point_count; ++i) {
        EXPECT_NEAR(m.points[i].penetration, 0.10f, kTol);
        EXPECT_TRUE(Vec3Near(m.points[i].normal, Vec3{0.0f, 0.0f, -1.0f}, 2.0e-2f))
            << "n=(" << m.points[i].normal.x << "," << m.points[i].normal.y << ","
            << m.points[i].normal.z << ")";
    }
}

// (4) D1 on the FAILING class: 2-run byte-identity + N>=32 cross-replica on the
// z=0.90 origin-on-boundary config (the exact bug class, which now exercises the
// new enclosure pre-loop + sliver-skip path -- code the OLD determinism gate at
// y=0.9 also exercises, but pin it on the SPECIFIC failing seed too).
TEST(GjkEpaConvex, OriginOnBoundary_DeterminismOnFailingClass) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const auto vb = BoxVerts(he);
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
    const ConvexHullView B = MakeHull(vb, Vec3{0.0f, 0.0f, 0.90f});  // the repro seed

    ContactManifold m1, m2;
    std::memset(&m1, 0, sizeof(m1));
    std::memset(&m2, 0, sizeof(m2));
    m1 = RouteHullHull(A, B);
    m2 = RouteHullHull(A, B);
    ASSERT_GT(m1.point_count, 0u);
    EXPECT_EQ(std::memcmp(&m1, &m2, sizeof(ContactManifold)), 0)
        << "2-run on the failing origin-on-boundary class must be byte-identical";

    constexpr int kReplicas = 40;  // N >= 32 cross-replica
    ContactManifold ref = RouteHullHull(A, B);
    for (int r = 0; r < kReplicas; ++r) {
        const ContactManifold mr = RouteHullHull(A, B);
        EXPECT_EQ(std::memcmp(&ref, &mr, sizeof(ContactManifold)), 0)
            << "replica " << r << " diverged (origin-on-boundary D1 gate)";
    }
}

// (5) GENUINE TOUCH (depth 0) MUST NOT become a spurious deep contact. Two unit
// boxes EXACTLY face-flush (B at z=1.0, overlap 0): the origin sits ON the CSO
// surface. The robust-EPA fix must report NO contact here -- NOT a far +/-axis
// face of the (flat, ~2-deep-on-the-other-side) box CSO. A tiny POSITIVE overlap
// just inside the boundary must still produce a real contact (the fix discriminates
// depth-0 touching from a real overlap, it does not blanket-drop near-zero depth).
TEST(GjkEpaConvex, GenuineTouchAndNearBoundary_NoSpuriousContact) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const auto vb = BoxVerts(he);
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});

    // Exact touch (overlap 0) -> NO contact (depth 0).
    {
        const ConvexHullView B = MakeHull(vb, Vec3{0.0f, 0.0f, 1.0f});
        const ContactManifold m = RouteHullHull(A, B);
        EXPECT_EQ(m.point_count, 0u)
            << "exact face-flush touch (overlap 0) must be NO contact, "
               "not a spurious deep manifold from the flat CSO's far side";
    }
    // Just SEPARATED (gap 0.001 and 0.05) -> NO contact.
    for (float gap : {0.001f, 0.05f}) {
        const ConvexHullView B = MakeHull(vb, Vec3{0.0f, 0.0f, 1.0f + gap});
        const ContactManifold m = RouteHullHull(A, B);
        EXPECT_EQ(m.point_count, 0u) << "separated (gap=" << gap << ") -> no contact";
    }
    // Just OVERLAPPING (overlap 0.001) -> a real small contact, normal -Z.
    {
        const ConvexHullView B = MakeHull(vb, Vec3{0.0f, 0.0f, 0.999f});
        const ContactManifold m = RouteHullHull(A, B);
        ASSERT_GT(m.point_count, 0u) << "tiny real overlap must still be a contact";
        EXPECT_NEAR(m.points[0].penetration, 0.001f, 1.0e-3f);
        EXPECT_TRUE(Vec3Near(m.points[0].normal, Vec3{0.0f, 0.0f, -1.0f}, 2.0e-2f));
    }
}

// (6) ROTATED / OBLIQUE origin-on-boundary: the bug is GJK-SEED-ORIENTATION-
// dependent, and (1)-(5) are all axis-aligned. A box B ROTATED off-axis and
// overlapping A produces a DIFFERENT seed straddle of the origin. The centroid-
// oriented fix is geometry-general -> each must give the right depth (the
// brute-force support-difference oracle along the emitted normal). Locks
// orientation coverage so an axis-only sweep can't hide an oblique regression.
TEST(GjkEpaConvex, OriginOnBoundary_RotatedOblique) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const auto vb = BoxVerts(he);
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
    // (axis, degrees, center-z) chosen to give a clear positive overlap.
    struct Cfg { Vec3 axis; float deg; float z; };
    const Cfg cfgs[5] = {
        {Vec3{1, 1, 0}, 30.0f, 1.00f}, {Vec3{0, 1, 1}, 30.0f, 0.95f},
        {Vec3{1, 1, 1}, 20.0f, 1.05f}, {Vec3{1, 2, 0}, 35.0f, 0.90f},
        {Vec3{2, 1, 1}, 15.0f, 1.00f},
    };
    for (const Cfg& c : cfgs) {
        const float r = c.deg * 3.14159265f / 180.0f;
        const float s = std::sin(r * 0.5f);
        const float l = std::sqrt(c.axis.x * c.axis.x + c.axis.y * c.axis.y +
                                  c.axis.z * c.axis.z);
        const Quat rot{c.axis.x / l * s, c.axis.y / l * s, c.axis.z / l * s,
                       std::cos(r * 0.5f)};
        const Vec3 pos{0.0f, 0.0f, c.z};
        const ConvexHullView B = MakeHull(vb, pos, rot);
        const ContactManifold m = RouteHullHull(A, B);
        ASSERT_GT(m.point_count, 0u)
            << "rotated " << c.deg << "deg @z=" << c.z << " -> EMPTY (regression!)";
        // Depth oracle along the emitted normal, using the WORLD-TRANSFORMED verts
        // of both hulls (B is rotated, so support must use the view's frame).
        const Vec3 n = m.points[0].normal;
        float ma = -3.4e38f, mb = -3.4e38f;
        for (uint32_t i = 0; i < A.vcount; ++i) {
            const float d = A.Vertex(i).Dot(Vec3{-n.x, -n.y, -n.z});
            if (d > ma) ma = d;
        }
        for (uint32_t i = 0; i < B.vcount; ++i) {
            const float d = B.Vertex(i).Dot(n);
            if (d > mb) mb = d;
        }
        const float oracle = ma + mb;
        EXPECT_NEAR(m.points[0].penetration, oracle, kTol)
            << "rotated " << c.deg << "deg @z=" << c.z
            << " depth != support-difference oracle along the normal";
        EXPECT_GT(m.points[0].penetration, 0.0f);
    }
}

// ===========================================================================
// M5 — SDF PRECISION ORACLE (plan M5: test_gjk_epa_convex -> SDF precision
// oracle). Bakes a DENSE analytic narrow-band SDF for a target primitive
// (sphere / box) at a known cell size into the SparseSdfDevice layout, builds
// the OTHER shape's sample-point set, drives the NarrowphaseSdf op, and asserts
// the sampled contact DEPTH + NORMAL match the ANALYTIC truth within a
// tolerance <= the bake cell size (plan §3.5 precision contract). Also a 2-run
// byte-identity (D1) check — the same posture the GJK/EPA Determinism case had,
// re-pointed at the SDF path.
// ===========================================================================
namespace {

namespace nsdf = nuka::runtime::sdf;
namespace nkops = nuka::phi::nkops;

// Test-only RAII over the phi v2 opaque Buffer* (stream-0 DeviceBufferType;
// NULL-stream transfers are byte+ordering identical to the legacy synchronous
// memcpy). Mirrors the legacy phi::Buffer surface (default-ctor + sized-ctor +
// Data/CopyFromHost/CopyToHost, move-only) the SDF oracle harness uses so it
// stays byte-identical after the buffer sweep.
class OwnedDeviceBuffer {
public:
    OwnedDeviceBuffer() = default;
    explicit OwnedDeviceBuffer(size_t bytes) {
        buf_ = nuka::phi::BufferAlloc(
            nuka::phi::DeviceBufferType(nuka::phi::InitBestDevice()), bytes);
    }
    explicit OwnedDeviceBuffer(nuka::phi::Buffer* buf) : buf_(buf) {}
    ~OwnedDeviceBuffer() { if (buf_ != nullptr) nuka::phi::BufferFree(buf_); }
    OwnedDeviceBuffer(OwnedDeviceBuffer&& o) noexcept : buf_(o.buf_) { o.buf_ = nullptr; }
    OwnedDeviceBuffer& operator=(OwnedDeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (buf_ != nullptr) nuka::phi::BufferFree(buf_);
            buf_ = o.buf_; o.buf_ = nullptr;
        }
        return *this;
    }
    OwnedDeviceBuffer(const OwnedDeviceBuffer&) = delete;
    OwnedDeviceBuffer& operator=(const OwnedDeviceBuffer&) = delete;
    void* Data() const { return buf_ != nullptr ? nuka::phi::BufferBase(buf_) : nullptr; }
    void CopyFromHost(const void* src, size_t bytes) {
        if (buf_ != nullptr && bytes > 0u) nuka::phi::BufferUpload(buf_, src, 0, bytes);
    }
    void CopyToHost(void* dst, size_t bytes) const {
        if (buf_ != nullptr && bytes > 0u) nuka::phi::BufferDownload(buf_, dst, 0, bytes);
    }
private:
    nuka::phi::Buffer* buf_ = nullptr;
};

template <typename T>
OwnedDeviceBuffer UploadOwned(const std::vector<T>& values) {
    return OwnedDeviceBuffer(nuka::phi::UploadVectorV2(
        nuka::phi::DeviceBufferType(nuka::phi::InitBestDevice()), values));
}

// One cooked SDF grid + its sample-point set + a pair list, all device-resident.
struct SdfOracleScene {
    // Model device tables.
    OwnedDeviceBuffer samp_points, samp_ranges, shape_table;
    OwnedDeviceBuffer sdf_headers, sdf_cell_count, sdf_keys, sdf_values, sdf_grads;
    OwnedDeviceBuffer body_pose, pairs;
    // Data device outputs.
    OwnedDeviceBuffer ucount, upoint, unormal, udepth, contact_count;
    uint32_t pair_count = 0, slot_stride = 0;
    float cell = 0.0f;
};

// Bake a DENSE SDF grid covering [lo,hi]^3 at `cell` spacing for an analytic
// signed-distance functor f(local) -> phi (negative inside). origin at lo.
template <typename F>
void BakeAnalyticSdf(Vec3 lo, Vec3 hi, float cell, F&& f,
                     std::vector<uint64_t>& keys, std::vector<float>& values,
                     std::vector<Vec3>& grads, uint32_t dims[3], Vec3& origin) {
    origin = lo;
    auto dim = [&](float a, float b) {
        return static_cast<uint32_t>(std::floor((b - a) / cell)) + 2u;
    };
    dims[0] = dim(lo.x, hi.x); dims[1] = dim(lo.y, hi.y); dims[2] = dim(lo.z, hi.z);
    for (uint32_t i = 0; i < dims[0]; ++i)
        for (uint32_t j = 0; j < dims[1]; ++j)
            for (uint32_t k = 0; k < dims[2]; ++k) {
                const Vec3 p{lo.x + i * cell, lo.y + j * cell, lo.z + k * cell};
                const float phi = f(p);
                // Central-difference gradient (the cooker's gradient field).
                const float h = 0.5f * cell;
                const Vec3 g{(f(Vec3{p.x + h, p.y, p.z}) - f(Vec3{p.x - h, p.y, p.z})) / cell,
                             (f(Vec3{p.x, p.y + h, p.z}) - f(Vec3{p.x, p.y - h, p.z})) / cell,
                             (f(Vec3{p.x, p.y, p.z + h}) - f(Vec3{p.x, p.y, p.z - h})) / cell};
                keys.push_back(nsdf::PackSdfCellKey(i, j, k));
                values.push_back(phi);
                grads.push_back(g);
            }
    // keys are emitted in ascending (i,j,k) order -> already sorted (D1 binary
    // search contract). Stable-sort defensively in case the loop order changes.
}

// Build the device scene: target body 1 holds the SDF (sphere/box) at origin;
// sampling body 0 holds a single sample point we place at a known world spot.
SdfOracleScene BuildSdfScene(const std::vector<float>& sample_local,
                             Transform sample_pose, Transform target_pose,
                             const std::vector<uint64_t>& keys,
                             const std::vector<float>& values,
                             const std::vector<Vec3>& grads,
                             const uint32_t dims[3], Vec3 origin, float voxel,
                             float cell) {
    SdfOracleScene s;
    s.cell = cell;
    s.slot_stride = 4u;  // 4 contact slots / env (>= the 1 pair we drive).
    // SAMP pool + per-body ranges (body 0 owns [0, count); body 1 empty).
    s.samp_points = UploadOwned(sample_local);
    std::vector<uint32_t> ranges = {0u, static_cast<uint32_t>(sample_local.size() / 3u),
                                    0u, 0u};
    s.samp_ranges = UploadOwned(ranges);
    // shape_table: 2 bodies x 8 f32 (body 1 carries sdf_grid = 0).
    std::vector<float> shp(2u * 8u, 0.0f);
    auto put_u = [&](size_t at, uint32_t v) { std::memcpy(&shp[at], &v, 4); };
    put_u(8u + 7u, 0u);  // body 1 sdf_grid = 0
    put_u(0u + 7u, ~0u); // body 0 analytic (no sdf)
    s.shape_table = UploadOwned(shp);
    // SDF header: origin.xyz, voxel, dims.xyz(u32 bits), cell_offset(u32 bits).
    std::vector<float> hdr(8u, 0.0f);
    hdr[0] = origin.x; hdr[1] = origin.y; hdr[2] = origin.z; hdr[3] = voxel;
    std::memcpy(&hdr[4], &dims[0], 4); std::memcpy(&hdr[5], &dims[1], 4);
    std::memcpy(&hdr[6], &dims[2], 4);
    const uint32_t cell_offset = 0u; std::memcpy(&hdr[7], &cell_offset, 4);
    s.sdf_headers = UploadOwned(hdr);
    std::vector<uint32_t> cc = {static_cast<uint32_t>(keys.size())};
    s.sdf_cell_count = UploadOwned(cc);
    s.sdf_keys = UploadOwned(keys);
    s.sdf_values = UploadOwned(values);
    s.sdf_grads = UploadOwned(grads);
    // body_pose (env-major, bodies_per_env = 2).
    std::vector<Transform> poses = {sample_pose, target_pose};
    s.body_pose = UploadOwned(poses);
    // one pair: env 0, slot 0, sample body 0 vs target body 1, grid 0.
    std::vector<nkops::SdfPairDev> prs = {{0u, 0u, 0u, 1u, 0u, 2u}};
    s.pairs = UploadOwned(prs);
    s.pair_count = 1u;
    // outputs (slot_stride slots x {count, 4-pt point/normal/depth}).
    s.ucount = OwnedDeviceBuffer(s.slot_stride * sizeof(uint32_t));
    s.upoint = OwnedDeviceBuffer(s.slot_stride * 4u * sizeof(Vec3));
    s.unormal = OwnedDeviceBuffer(s.slot_stride * 4u * sizeof(Vec3));
    s.udepth = OwnedDeviceBuffer(s.slot_stride * 4u * sizeof(float));
    s.contact_count = OwnedDeviceBuffer(sizeof(uint32_t));
    std::vector<uint32_t> zero = {0u};
    s.contact_count.CopyFromHost(zero.data(), sizeof(uint32_t));
    return s;
}

// Run the op + download slot 0's first manifold point.
struct SdfHit { uint32_t count; Vec3 point; Vec3 normal; float depth; };
SdfHit RunSdfOp(SdfOracleScene& s, float margin = 0.0f) {
    const auto status = nuka::phi::LaunchNarrowphaseSdf(
        static_cast<const float*>(s.samp_points.Data()),
        static_cast<const uint32_t*>(s.samp_ranges.Data()),
        static_cast<const float*>(s.shape_table.Data()),
        static_cast<const float*>(s.sdf_headers.Data()),
        static_cast<const uint32_t*>(s.sdf_cell_count.Data()),
        static_cast<const uint64_t*>(s.sdf_keys.Data()),
        static_cast<const float*>(s.sdf_values.Data()),
        static_cast<const Vec3*>(s.sdf_grads.Data()),
        static_cast<const Transform*>(s.body_pose.Data()),
        static_cast<const nkops::SdfPairDev*>(s.pairs.Data()),
        s.pair_count, /*k=*/4u, margin, s.slot_stride,
        static_cast<uint32_t*>(s.ucount.Data()),
        static_cast<Vec3*>(s.upoint.Data()),
        static_cast<Vec3*>(s.unormal.Data()),
        static_cast<float*>(s.udepth.Data()),
        static_cast<uint32_t*>(s.contact_count.Data()),
        /*stream=*/nullptr);
    EXPECT_EQ(static_cast<int>(status), static_cast<int>(nuka::phi::Status::Ok));
    cudaDeviceSynchronize();
    SdfHit h{};
    s.ucount.CopyToHost(&h.count, sizeof(uint32_t));
    s.upoint.CopyToHost(&h.point, sizeof(Vec3));
    s.unormal.CopyToHost(&h.normal, sizeof(Vec3));
    s.udepth.CopyToHost(&h.depth, sizeof(float));
    return h;
}

}  // namespace

// SPHERE x SPHERE: target = unit sphere (r=0.3) at origin; the SAMPLING shape's
// single sample point is placed at a known penetration depth along +X. The SDF
// reports depth = r - |q| at that point; analytic truth = the same.
TEST(SdfPrecisionOracle, SphereSampleVsSphereSdf_DepthMatchesAnalytic) {
    const float cell = 0.01f;          // bake cell size = the tolerance bound.
    const float R = 0.30f;
    std::vector<uint64_t> keys; std::vector<float> vals; std::vector<Vec3> grads;
    uint32_t dims[3]; Vec3 origin;
    BakeAnalyticSdf(Vec3{-0.5f, -0.5f, -0.5f}, Vec3{0.5f, 0.5f, 0.5f}, cell,
                    [&](Vec3 p) { return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z) - R; },
                    keys, vals, grads, dims, origin);

    // Place the sample point at world (0.25, 0, 0): inside the sphere by
    // R - 0.25 = 0.05. sample body at identity; sample local == world.
    std::vector<float> sample = {0.25f, 0.0f, 0.0f};
    SdfOracleScene s = BuildSdfScene(sample, Transform::Identity(),
                                     Transform::Identity(), keys, vals, grads,
                                     dims, origin, cell, cell);
    const SdfHit h = RunSdfOp(s);
    ASSERT_EQ(h.count, 1u) << "sample inside the sphere -> exactly one contact";
    EXPECT_NEAR(h.depth, R - 0.25f, cell) << "SDF depth != analytic (R - |q|)";
    // Normal = +X (gradient points outward along +X at (0.25,0,0)).
    EXPECT_NEAR(h.normal.x, 1.0f, 0.05f);
    EXPECT_NEAR(h.normal.y, 0.0f, 0.05f);
    EXPECT_NEAR(h.normal.z, 0.0f, 0.05f);
}

// SPHERE-SAMPLE x BOX SDF: target = box he=(0.2,0.2,0.2); sample inside near the
// +Z face. SDF depth at a point inside an axis box = min face distance.
TEST(SdfPrecisionOracle, SampleVsBoxSdf_DepthMatchesAnalytic) {
    const float cell = 0.01f;
    const Vec3 he{0.20f, 0.20f, 0.20f};
    auto box_sdf = [&](Vec3 p) {
        // Inside (negative): -min over faces; this oracle only samples interior.
        const float dx = he.x - std::fabs(p.x);
        const float dy = he.y - std::fabs(p.y);
        const float dz = he.z - std::fabs(p.z);
        const float inside = std::min(dx, std::min(dy, dz));
        return -inside;  // negative inside, ~0 at surface.
    };
    std::vector<uint64_t> keys; std::vector<float> vals; std::vector<Vec3> grads;
    uint32_t dims[3]; Vec3 origin;
    BakeAnalyticSdf(Vec3{-0.4f, -0.4f, -0.4f}, Vec3{0.4f, 0.4f, 0.4f}, cell,
                    box_sdf, keys, vals, grads, dims, origin);

    // Sample at world (0,0,0.16): inside, depth to +Z face = 0.2 - 0.16 = 0.04.
    std::vector<float> sample = {0.0f, 0.0f, 0.16f};
    SdfOracleScene s = BuildSdfScene(sample, Transform::Identity(),
                                     Transform::Identity(), keys, vals, grads,
                                     dims, origin, cell, cell);
    const SdfHit h = RunSdfOp(s);
    ASSERT_EQ(h.count, 1u);
    EXPECT_NEAR(h.depth, he.z - 0.16f, cell) << "SDF box depth != min-face analytic";
    // Normal points out the nearest (+Z) face.
    EXPECT_NEAR(h.normal.z, 1.0f, 0.1f);
}

// SEPARATED: sample OUTSIDE the SDF band -> no contact (count 0).
TEST(SdfPrecisionOracle, SeparatedSampleEmitsNoContact) {
    const float cell = 0.01f;
    const float R = 0.20f;
    std::vector<uint64_t> keys; std::vector<float> vals; std::vector<Vec3> grads;
    uint32_t dims[3]; Vec3 origin;
    BakeAnalyticSdf(Vec3{-0.3f, -0.3f, -0.3f}, Vec3{0.3f, 0.3f, 0.3f}, cell,
                    [&](Vec3 p) { return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z) - R; },
                    keys, vals, grads, dims, origin);
    // Sample at (0.27,0,0): OUTSIDE the sphere (phi = 0.07 > 0) -> no contact.
    std::vector<float> sample = {0.27f, 0.0f, 0.0f};
    SdfOracleScene s = BuildSdfScene(sample, Transform::Identity(),
                                     Transform::Identity(), keys, vals, grads,
                                     dims, origin, cell, cell);
    const SdfHit h = RunSdfOp(s);
    EXPECT_EQ(h.count, 0u) << "a sample outside the surface must NOT emit a contact";
}

// D1: two runs of the SAME scene -> byte-identical manifold (fixed deepest-K
// order + the D1 trilinear sampler). Re-points the GjkEpaConvex determinism
// posture at the SDF path.
TEST(SdfPrecisionOracle, TwoRunByteIdentity) {
    const float cell = 0.01f;
    const float R = 0.30f;
    std::vector<uint64_t> keys; std::vector<float> vals; std::vector<Vec3> grads;
    uint32_t dims[3]; Vec3 origin;
    BakeAnalyticSdf(Vec3{-0.5f, -0.5f, -0.5f}, Vec3{0.5f, 0.5f, 0.5f}, cell,
                    [&](Vec3 p) { return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z) - R; },
                    keys, vals, grads, dims, origin);
    std::vector<float> sample = {0.10f, 0.10f, 0.05f};  // multi-axis interior.
    SdfOracleScene s1 = BuildSdfScene(sample, Transform::Identity(),
                                      Transform::Identity(), keys, vals, grads,
                                      dims, origin, cell, cell);
    const SdfHit a = RunSdfOp(s1);
    SdfOracleScene s2 = BuildSdfScene(sample, Transform::Identity(),
                                      Transform::Identity(), keys, vals, grads,
                                      dims, origin, cell, cell);
    const SdfHit b = RunSdfOp(s2);
    EXPECT_EQ(a.count, b.count);
    EXPECT_EQ(std::memcmp(&a.point, &b.point, sizeof(Vec3)), 0);
    EXPECT_EQ(std::memcmp(&a.normal, &b.normal, sizeof(Vec3)), 0);
    EXPECT_EQ(std::memcmp(&a.depth, &b.depth, sizeof(float)), 0);
}
