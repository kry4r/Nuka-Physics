// ---------------------------------------------------------------------------
// v0.7 p09-D: XPBD soft-body COOKER tests (exit-crit 3 -- cookers operational).
// ---------------------------------------------------------------------------
// The cooker is a THIN orchestration layer: a programmatic XpbdSoftBodySpec ->
// XpbdConstraintSet, by dispatching to the existing p09-B/C topology builders and
// mapping the authored material params. These tests assert, on the RETURNED host
// constraint set (pure CPU -- no device upload / step), that:
//
//   1. CLOTH: a hand-verifiable 2-triangle quad cooks to the known edge + bend
//      counts, and a 2x2 grid scales to the combinatorially-derived counts.
//   2. SOFTBODY: a single tet cooks to 6 distance + 1 volume; emit_tet_edges=false
//      isolates the 1 volume constraint.
//   3. SHAPE-MATCH: a spec cooks to ONE cluster over all particles with the rest
//      positions preserved.
//   4. PARAM MAPPING: the TWO distinct mappings are exercised directly --
//      cloth/tet stiffness 1e4 -> compliance_alpha 1e-4 (on every emitted row),
//      and shape_match stiffness 0.5 -> cluster.stiffness 0.5 (NOT 1/0.5). Edge
//      cases: stiffness<=0 -> alpha 0 (rigid); shape_match clamps to [0,1].
//   5. DETERMINISM: cooking the same spec twice yields a byte-identical
//      constraint list (CPU cook-time, but asserted).
// ---------------------------------------------------------------------------

#include "import/cooker/xpbd_cooker.hpp"

#include "math/vec3.hpp"
#include "runtime/soft/cloth_topology.hpp"
#include "runtime/soft/tetmesh_topology.hpp"
#include "runtime/soft/xpbd_world.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

namespace {

using nuka::import::cooker::ComplianceAlphaFromStiffness;
using nuka::import::cooker::CookXpbdSoftBody;
using nuka::import::cooker::ShapeMatchFractionFromStiffness;
using nuka::import::cooker::SoftBodyType;
using nuka::import::cooker::XpbdSoftBodySpec;
using nuka::math::Vec3;
using nuka::runtime::soft::ClothTriangle;
using nuka::runtime::soft::TetMeshTet;
using nuka::runtime::soft::XpbdConstraintSet;

// --- Specs ----------------------------------------------------------------

// A flat unit quad split into two triangles by the (v0,v2) diagonal. Vertices:
//   2 --- 3        (0,0) (1,0) at y=0 row; (0,1) (1,1) at y=1 row.
//   |  \  |        Triangles: {0,1,2} and {1,3,2}. Shared (interior) edge: 1-2.
//   0 --- 1        5 unique edges: 0-1, 0-2, 1-2(diag/shared), 1-3, 2-3.
XpbdSoftBodySpec QuadClothSpec(float stiffness) {
    XpbdSoftBodySpec s;
    s.type = SoftBodyType::Cloth;
    s.rest_positions = {
        Vec3{0.0f, 0.0f, 0.0f},  // 0
        Vec3{1.0f, 0.0f, 0.0f},  // 1
        Vec3{0.0f, 1.0f, 0.0f},  // 2
        Vec3{1.0f, 1.0f, 0.0f},  // 3
    };
    s.triangles = {ClothTriangle{{0u, 1u, 2u}}, ClothTriangle{{1u, 3u, 2u}}};
    s.stiffness = stiffness;
    return s;
}

// A 2x2 cell grid (3x3 = 9 vertices), each cell split by its (lo-left, up-right)
// diagonal into 2 triangles (8 triangles total). Combinatorial unique-edge count:
//   horizontal edges : (rows=3) * (cells-per-row=2) = 6
//   vertical   edges : (cols=3) * (cells-per-col=2) = 6
//   diagonal   edges : one per cell = 2*2            = 4
//   total distance   = 16.
// Interior edges (shared by exactly two triangles) -> bend constraints:
//   every diagonal is interior (the two tris of its own cell): 4
//   shared horizontal edges (between vertically-adjacent cells): the middle row's
//     2 horizontal edges = 2
//   shared vertical edges (between horizontally-adjacent cells): the middle col's
//     2 vertical edges = 2
//   total bend = 8.
XpbdSoftBodySpec GridClothSpec() {
    XpbdSoftBodySpec s;
    s.type = SoftBodyType::Cloth;
    constexpr int kN = 2;  // cells per side
    auto vid = [](int ix, int iy) { return static_cast<uint32_t>(iy * (kN + 1) + ix); };
    for (int iy = 0; iy <= kN; ++iy)
        for (int ix = 0; ix <= kN; ++ix)
            s.rest_positions.push_back(Vec3{static_cast<float>(ix), static_cast<float>(iy), 0.0f});
    for (int iy = 0; iy < kN; ++iy)
        for (int ix = 0; ix < kN; ++ix) {
            const uint32_t a = vid(ix, iy), b = vid(ix + 1, iy);
            const uint32_t c = vid(ix, iy + 1), d = vid(ix + 1, iy + 1);
            // Split by the a-d diagonal: triangles {a,b,d} and {a,d,c}.
            s.triangles.push_back(ClothTriangle{{a, b, d}});
            s.triangles.push_back(ClothTriangle{{a, d, c}});
        }
    s.stiffness = 1.0e4f;
    return s;
}

// A single unit tet (the 4 corners). 6 edges, 1 volume.
XpbdSoftBodySpec TetSpec(bool emit_edges, float stiffness) {
    XpbdSoftBodySpec s;
    s.type = SoftBodyType::SoftBody;
    s.rest_positions = {
        Vec3{0.0f, 0.0f, 0.0f},
        Vec3{1.0f, 0.0f, 0.0f},
        Vec3{0.0f, 1.0f, 0.0f},
        Vec3{0.0f, 0.0f, 1.0f},
    };
    s.tets = {TetMeshTet{{0u, 1u, 2u, 3u}}};
    s.emit_tet_edges = emit_edges;
    s.stiffness = stiffness;
    return s;
}

XpbdSoftBodySpec ShapeMatchSpec(float stiffness) {
    XpbdSoftBodySpec s;
    s.type = SoftBodyType::ShapeMatch;
    s.rest_positions = {
        Vec3{0.0f, 0.0f, 0.0f},
        Vec3{1.0f, 0.0f, 0.0f},
        Vec3{0.0f, 1.0f, 0.0f},
        Vec3{0.0f, 0.0f, 1.0f},
        Vec3{1.0f, 1.0f, 1.0f},
    };
    s.stiffness = stiffness;
    return s;
}

// Serialize a constraint set to bytes for exact two-cook determinism comparison.
std::string ConstraintBytes(const XpbdConstraintSet& cs) {
    std::string out;
    auto put = [&out](const void* p, size_t n) {
        out.append(static_cast<const char*>(p), n);
    };
    for (const auto& d : cs.distance) put(&d, sizeof(d));
    for (const auto& b : cs.bend) put(&b, sizeof(b));
    for (const auto& v : cs.volume) put(&v, sizeof(v));
    for (const auto& c : cs.shape_match) {
        put(c.particle.data(), c.particle.size() * sizeof(uint32_t));
        put(c.rest_positions.data(), c.rest_positions.size() * sizeof(Vec3));
        put(c.cluster_mass.data(), c.cluster_mass.size() * sizeof(float));
        put(&c.stiffness, sizeof(c.stiffness));
    }
    return out;
}

// --- Param mapping (the two distinct mappings) -----------------------------

TEST(XpbdCookerMapping, ElasticStiffnessToComplianceAlpha) {
    EXPECT_FLOAT_EQ(ComplianceAlphaFromStiffness(1.0e4f), 1.0e-4f);
    EXPECT_FLOAT_EQ(ComplianceAlphaFromStiffness(100.0f), 0.01f);
    // stiffness <= 0 => rigid (alpha 0), not a div-by-zero.
    EXPECT_FLOAT_EQ(ComplianceAlphaFromStiffness(0.0f), 0.0f);
    EXPECT_FLOAT_EQ(ComplianceAlphaFromStiffness(-5.0f), 0.0f);
}

TEST(XpbdCookerMapping, ShapeMatchStiffnessIsAGoalPullFractionNotACompliance) {
    // The discriminating check: 0.5 must map to 0.5, NOT 1/0.5 == 2.
    EXPECT_FLOAT_EQ(ShapeMatchFractionFromStiffness(0.5f), 0.5f);
    EXPECT_NE(ShapeMatchFractionFromStiffness(0.5f), 1.0f / 0.5f);
    // Clamp to [0,1].
    EXPECT_FLOAT_EQ(ShapeMatchFractionFromStiffness(2.0f), 1.0f);
    EXPECT_FLOAT_EQ(ShapeMatchFractionFromStiffness(1.0e4f), 1.0f);
    EXPECT_FLOAT_EQ(ShapeMatchFractionFromStiffness(0.0f), 0.0f);
    EXPECT_FLOAT_EQ(ShapeMatchFractionFromStiffness(-1.0f), 0.0f);
}

// --- Cloth -----------------------------------------------------------------

TEST(XpbdCookerCloth, QuadCooksKnownEdgeAndBendCounts) {
    const auto cs = CookXpbdSoftBody(QuadClothSpec(1.0e4f));
    // 5 unique edges; 1 interior (shared) edge -> 1 bend; no volume / cluster.
    EXPECT_EQ(cs.distance.size(), 5u);
    EXPECT_EQ(cs.bend.size(), 1u);
    EXPECT_EQ(cs.volume.size(), 0u);
    EXPECT_EQ(cs.shape_match.size(), 0u);

    // Mapping applied on every emitted elastic row: 1e4 -> 1e-4.
    for (const auto& d : cs.distance) EXPECT_FLOAT_EQ(d.compliance_alpha, 1.0e-4f);
    for (const auto& b : cs.bend) EXPECT_FLOAT_EQ(b.compliance_alpha, 1.0e-4f);
}

TEST(XpbdCookerCloth, GridScalesToCombinatorialCounts) {
    const auto cs = CookXpbdSoftBody(GridClothSpec());
    EXPECT_EQ(cs.distance.size(), 16u);  // 6 horiz + 6 vert + 4 diag
    EXPECT_EQ(cs.bend.size(), 8u);       // 4 diag + 2 shared-horiz + 2 shared-vert
    EXPECT_EQ(cs.volume.size(), 0u);
}

TEST(XpbdCookerCloth, EmitBendFalseDropsBendKeepsDistance) {
    auto spec = QuadClothSpec(1.0e4f);
    spec.emit_bend = false;
    const auto cs = CookXpbdSoftBody(spec);
    EXPECT_EQ(cs.distance.size(), 5u);
    EXPECT_EQ(cs.bend.size(), 0u);
}

TEST(XpbdCookerCloth, RigidStiffnessGivesZeroAlpha) {
    const auto cs = CookXpbdSoftBody(QuadClothSpec(0.0f));
    ASSERT_FALSE(cs.distance.empty());
    for (const auto& d : cs.distance) EXPECT_FLOAT_EQ(d.compliance_alpha, 0.0f);
    for (const auto& b : cs.bend) EXPECT_FLOAT_EQ(b.compliance_alpha, 0.0f);
}

// --- SoftBody (tet) --------------------------------------------------------

TEST(XpbdCookerSoftBody, SingleTetCooks6EdgesAnd1Volume) {
    const auto cs = CookXpbdSoftBody(TetSpec(/*emit_edges=*/true, 1.0e4f));
    EXPECT_EQ(cs.distance.size(), 6u);
    EXPECT_EQ(cs.volume.size(), 1u);
    EXPECT_EQ(cs.bend.size(), 0u);
    EXPECT_EQ(cs.shape_match.size(), 0u);
    for (const auto& d : cs.distance) EXPECT_FLOAT_EQ(d.compliance_alpha, 1.0e-4f);
    EXPECT_FLOAT_EQ(cs.volume[0].compliance_alpha, 1.0e-4f);
}

TEST(XpbdCookerSoftBody, EmitEdgesFalseIsolatesVolume) {
    const auto cs = CookXpbdSoftBody(TetSpec(/*emit_edges=*/false, 1.0e4f));
    EXPECT_EQ(cs.distance.size(), 0u);
    EXPECT_EQ(cs.volume.size(), 1u);
}

// --- Shape-match -----------------------------------------------------------

TEST(XpbdCookerShapeMatch, CooksOneClusterOverAllParticles) {
    const auto spec = ShapeMatchSpec(0.5f);
    const auto cs = CookXpbdSoftBody(spec);
    EXPECT_EQ(cs.distance.size(), 0u);
    EXPECT_EQ(cs.bend.size(), 0u);
    EXPECT_EQ(cs.volume.size(), 0u);
    ASSERT_EQ(cs.shape_match.size(), 1u);

    const auto& cl = cs.shape_match[0];
    ASSERT_EQ(cl.particle.size(), spec.rest_positions.size());
    ASSERT_EQ(cl.rest_positions.size(), spec.rest_positions.size());
    ASSERT_EQ(cl.cluster_mass.size(), spec.rest_positions.size());
    for (size_t i = 0; i < spec.rest_positions.size(); ++i) {
        EXPECT_EQ(cl.particle[i], static_cast<uint32_t>(i));
        EXPECT_EQ(cl.rest_positions[i], spec.rest_positions[i]);
        EXPECT_FLOAT_EQ(cl.cluster_mass[i], 1.0f);  // uniform default weight
    }
    // The shape-match mapping (0.5 -> 0.5), proven on the cooked cluster too.
    EXPECT_FLOAT_EQ(cl.stiffness, 0.5f);
}

TEST(XpbdCookerShapeMatch, EmptySpecCooksNoCluster) {
    XpbdSoftBodySpec s;
    s.type = SoftBodyType::ShapeMatch;  // no rest positions
    const auto cs = CookXpbdSoftBody(s);
    EXPECT_EQ(cs.shape_match.size(), 0u);
}

// --- Determinism -----------------------------------------------------------

TEST(XpbdCookerDeterminism, RepeatedCooksAreByteIdentical) {
    const std::vector<XpbdSoftBodySpec> specs = {
        QuadClothSpec(1.0e4f), GridClothSpec(), TetSpec(true, 250.0f),
        ShapeMatchSpec(0.7f)};
    for (const auto& spec : specs) {
        const std::string golden = ConstraintBytes(CookXpbdSoftBody(spec));
        for (int run = 0; run < 4; ++run) {
            EXPECT_EQ(ConstraintBytes(CookXpbdSoftBody(spec)), golden)
                << "cook run " << run << " differs byte-wise";
        }
    }
}

}  // namespace
