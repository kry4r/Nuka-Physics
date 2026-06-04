// ---------------------------------------------------------------------------
// v0.8 C4b: test of the NEW compliant-normal + pyramid-friction row emitter
// (src/constraint/row_builder.cpp EmitCompliantContactRows). VALIDATED-NOT-WIRED.
// ---------------------------------------------------------------------------
// This emitter is DELIBERATELY NOT wired into the production stepper (C5 wires
// it + re-baselines goldens). These tests prove:
//   (1) row counts: a penetrating 4-point condim=3 box -> 4 normal + 16 friction
//       rows (4 spokes/point).
//   (2) each normal row's compliance_alpha==R and rhs==aref from an INDEPENDENT
//       ComputeCompliantRow call on the same inputs (cross-check, not echo).
//   (3) the 4 friction spokes per point sum ~= 0 and span the tangent plane
//       symmetrically (balanced polygonal approximation of the isotropic disk),
//       and friction rows carry NO compliance bias.
//   (4) the ContactRowSides side-output: exactly one per appended row, in order,
//       carrying the manifold's per-side CollidableRef (type/react/handle).
//   (5) 2-run byte-identity of the emitted rows (D1, host-vs-host).
//   (6) INERTNESS: the old path (BuildContactRows -> AppendContactGroup) on the
//       same manifold still produces its unchanged 2-tangent output.
// ---------------------------------------------------------------------------

#include "constraint/row_builder.hpp"
#include "constraint/solref_solimp.hpp"

#include <gtest/gtest.h>

#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

using namespace nuka::constraint;
namespace math = nuka::math;

namespace {

// A penetrating box-on-plane style manifold: 4 corner points, +Y normal,
// non-default solref/solimp/friction so the cross-check is meaningful.
ContactManifold MakeBoxManifold() {
    ContactManifold manifold;
    manifold.a.type = CollidableType::RigidBody;
    manifold.a.react = ReactionProviderKind::RigidInvMass;
    manifold.a.handle = 7u;
    manifold.b.type = CollidableType::StaticWorld;
    manifold.b.react = ReactionProviderKind::StaticNull;
    manifold.b.handle = 99u;
    manifold.friction = 0.8f;
    manifold.restitution = 0.1f;
    // non-default solimp (dmin,dmax,width,mid,power)
    manifold.solimp[0] = 0.85f;
    manifold.solimp[1] = 0.97f;
    manifold.solimp[2] = 0.002f;
    manifold.solimp[3] = 0.4f;
    manifold.solimp[4] = 2.0f;

    const math::Vec3 normal{0.0f, 1.0f, 0.0f};
    const float xs[4] = {-0.5f, 0.5f, 0.5f, -0.5f};
    const float zs[4] = {-0.5f, -0.5f, 0.5f, 0.5f};
    for (int i = 0; i < 4; ++i) {
        ContactPoint p;
        p.position = {xs[i], 0.0f, zs[i]};
        p.normal = normal;
        p.penetration = 0.01f + 0.002f * static_cast<float>(i);  // distinct depths
        p.normal_impulse = 0.0f;
        p.solref_timeconst = 0.02f;
        p.solref_dampratio = 1.0f;
        manifold.AddPoint(p);
    }
    return manifold;
}

ContactRowComplianceInputs MakeInputs() {
    ContactRowComplianceInputs in;
    in.vel = -0.3f;       // approaching (signed normal velocity)
    in.invweight = 1.5f;  // reaction inv-weight sum
    in.dt = 0.005f;
    in.condim = 3u;       // -> 4 pyramid spokes/point
    in.refsafe = true;
    return in;
}

}  // namespace

// (1) row counts: 4 normal + 16 friction = 20 rows.
TEST(CompliantRows, BoxEmitsFourNormalSixteenFriction) {
    const ContactManifold manifold = MakeBoxManifold();
    const ContactRowComplianceInputs inputs = MakeInputs();

    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    EmitCompliantContactRows(
        std::span<const ContactManifold>(&manifold, 1), inputs, &rows, &sides);

    ASSERT_EQ(rows.RowCount(), 20u);
    EXPECT_EQ(sides.size(), 20u);

    // First 4 are normal rows (Unilateral), next 16 are friction rows.
    uint32_t normal_count = 0;
    uint32_t friction_count = 0;
    for (uint32_t i = 0; i < rows.RowCount(); ++i) {
        const Row& r = rows.rows[i];
        EXPECT_EQ(r.row_class_id, kMaximalContactRowClassId);
        if (r.flags & row_flags::Unilateral) {
            ++normal_count;
            EXPECT_LT(i, 4u);  // normals emitted first
            EXPECT_FLOAT_EQ(r.lower, 0.0f);
            EXPECT_FLOAT_EQ(r.upper, FLT_MAX);
        } else if (r.flags & row_flags::Friction) {
            ++friction_count;
            EXPECT_GE(i, 4u);
        }
    }
    EXPECT_EQ(normal_count, 4u);
    EXPECT_EQ(friction_count, 16u);

    // Material bookkeeping reflects the 4+16 layout.
    EXPECT_EQ(rows.materials[0].normal_row_count, 4u);
    EXPECT_EQ(rows.materials[0].first_friction_row, 4u);
    EXPECT_EQ(rows.materials[0].friction_row_count, 16u);
    EXPECT_FLOAT_EQ(rows.materials[0].friction, 0.8f);
}

// (2) cross-check: each normal row's R/aref == an INDEPENDENT ComputeCompliantRow.
TEST(CompliantRows, NormalRowsCarryComplianceFromC4a) {
    const ContactManifold manifold = MakeBoxManifold();
    const ContactRowComplianceInputs inputs = MakeInputs();

    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    EmitCompliantContactRows(
        std::span<const ContactManifold>(&manifold, 1), inputs, &rows, &sides);

    for (uint32_t i = 0; i < manifold.point_count; ++i) {
        const ContactPoint& p = manifold.points[i];
        const float solref[2] = {p.solref_timeconst, p.solref_dampratio};
        const float pos = -p.penetration;  // signed: penetrating is negative
        const CompliantContactRow expect = ComputeCompliantRow(
            solref, manifold.solimp, pos, pos,
            inputs.vel, inputs.invweight, inputs.dt, inputs.refsafe);

        // BYTE-exact: same inputs, same NUKA_SRSI_HD function -> identical bits.
        EXPECT_EQ(std::memcmp(&rows.rows[i].compliance_alpha, &expect.R,
                              sizeof(float)), 0)
            << "normal row " << i << " R mismatch";
        EXPECT_EQ(std::memcmp(&rows.rows[i].rhs, &expect.aref_bias,
                              sizeof(float)), 0)
            << "normal row " << i << " aref mismatch";
    }

    // Friction rows carry NO compliance bias.
    for (uint32_t i = 4u; i < rows.RowCount(); ++i) {
        EXPECT_FLOAT_EQ(rows.rows[i].compliance_alpha, 0.0f);
        EXPECT_FLOAT_EQ(rows.rows[i].rhs, 0.0f);
    }
}

// (3) friction pyramid edges sum to the isotropic disk: spokes per point sum ~= 0
//     and span the tangent plane symmetrically (orthonormal +-t0,+-t1).
TEST(CompliantRows, FrictionPyramidApproximatesIsotropicDisk) {
    const ContactManifold manifold = MakeBoxManifold();
    const ContactRowComplianceInputs inputs = MakeInputs();

    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    EmitCompliantContactRows(
        std::span<const ContactManifold>(&manifold, 1), inputs, &rows, &sides);

    const math::Vec3 normal{0.0f, 1.0f, 0.0f};
    constexpr float kTol = 1.0e-6f;

    // 4 friction rows per point, starting at index 4. Each spoke's side-A
    // Jacobian linear part is the spoke direction.
    for (uint32_t pt = 0; pt < manifold.point_count; ++pt) {
        const uint32_t base = 4u + pt * 4u;
        const math::Vec3 d0 = rows.JacobianForRowBody(base + 0u, 0u).linear;
        const math::Vec3 d1 = rows.JacobianForRowBody(base + 1u, 0u).linear;
        const math::Vec3 d2 = rows.JacobianForRowBody(base + 2u, 0u).linear;
        const math::Vec3 d3 = rows.JacobianForRowBody(base + 3u, 0u).linear;

        // Balanced: the 4 spokes sum to ~0 (symmetric polygon).
        const math::Vec3 sum = d0 + d1 + d2 + d3;
        EXPECT_NEAR(sum.x, 0.0f, kTol);
        EXPECT_NEAR(sum.y, 0.0f, kTol);
        EXPECT_NEAR(sum.z, 0.0f, kTol);

        // Opposite spokes are antiparallel (the +-t pairs).
        EXPECT_NEAR((d0 + d1).Length(), 0.0f, kTol);
        EXPECT_NEAR((d2 + d3).Length(), 0.0f, kTol);

        // Each spoke is a UNIT direction lying in the tangent plane (perp to n).
        for (const auto& d : {d0, d1, d2, d3}) {
            EXPECT_NEAR(d.Length(), 1.0f, kTol);
            EXPECT_NEAR(d.Dot(normal), 0.0f, kTol);
        }

        // The two axes span the plane: t0 perp t1 (orthonormal basis).
        EXPECT_NEAR(d0.Dot(d2), 0.0f, kTol);
    }
}

// (4) ContactRowSides: one per appended row, in order, carrying the manifold's
//     per-side CollidableRef (type/react/handle).
TEST(CompliantRows, SideOutputMatchesManifoldCollidableRefs) {
    const ContactManifold manifold = MakeBoxManifold();
    const ContactRowComplianceInputs inputs = MakeInputs();

    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    EmitCompliantContactRows(
        std::span<const ContactManifold>(&manifold, 1), inputs, &rows, &sides);

    ASSERT_EQ(sides.size(), rows.RowCount());
    for (const auto& s : sides) {
        EXPECT_EQ(s.a.type, CollidableType::RigidBody);
        EXPECT_EQ(s.a.react, ReactionProviderKind::RigidInvMass);
        EXPECT_EQ(s.a.handle, 7u);
        EXPECT_EQ(s.b.type, CollidableType::StaticWorld);
        EXPECT_EQ(s.b.react, ReactionProviderKind::StaticNull);
        EXPECT_EQ(s.b.handle, 99u);
    }
}

// (4b) base-offset correctness: pre-existing rows in the buffer are NOT claimed;
//      side-output covers ONLY the rows this emitter appends, aligned by base.
TEST(CompliantRows, SideOutputSnapshotsBaseRowCount) {
    RowBuffers rows;
    std::vector<ContactRowSides> sides;

    // Pre-seed the buffer with an unrelated row via the OLD path.
    BuildContactRows(2u, &rows);
    const uint32_t base = rows.RowCount();
    ASSERT_GT(base, 0u);

    const ContactManifold manifold = MakeBoxManifold();
    EmitCompliantContactRows(
        std::span<const ContactManifold>(&manifold, 1), MakeInputs(),
        &rows, &sides);

    EXPECT_EQ(rows.RowCount(), base + 20u);
    EXPECT_EQ(sides.size(), 20u);  // only the C4b rows, not the pre-seeded ones
    // The emitter's first material group_id points at the base (its own first
    // appended row), not at 0.
    EXPECT_EQ(rows.materials[base].group_id, base);
    EXPECT_EQ(rows.materials[base].first_friction_row, base + 4u);
}

// (5) 2-run byte-identity (D1, host-vs-host).
TEST(CompliantRows, TwoRunByteIdentity) {
    const ContactManifold manifold = MakeBoxManifold();
    const ContactRowComplianceInputs inputs = MakeInputs();

    RowBuffers a;
    std::vector<ContactRowSides> sa;
    EmitCompliantContactRows(
        std::span<const ContactManifold>(&manifold, 1), inputs, &a, &sa);

    RowBuffers b;
    std::vector<ContactRowSides> sb;
    EmitCompliantContactRows(
        std::span<const ContactManifold>(&manifold, 1), inputs, &b, &sb);

    ASSERT_EQ(a.RowCount(), b.RowCount());
    ASSERT_EQ(a.rows.size(), b.rows.size());
    EXPECT_EQ(std::memcmp(a.rows.data(), b.rows.data(),
                          a.rows.size() * sizeof(Row)), 0);
    ASSERT_EQ(a.jacobian_data.size(), b.jacobian_data.size());
    EXPECT_EQ(std::memcmp(a.jacobian_data.data(), b.jacobian_data.data(),
                          a.jacobian_data.size() * sizeof(RowJacobian6)), 0);
    ASSERT_EQ(a.materials.size(), b.materials.size());
    EXPECT_EQ(std::memcmp(a.materials.data(), b.materials.data(),
                          a.materials.size() * sizeof(RowMaterial)), 0);
    ASSERT_EQ(sa.size(), sb.size());
    EXPECT_EQ(std::memcmp(sa.data(), sb.data(),
                          sa.size() * sizeof(ContactRowSides)), 0);
}

// (6) INERTNESS: the OLD path on the same manifold is UNCHANGED (2-tangent, 4+2).
TEST(CompliantRows, OldPathUnchangedByNewEmitter) {
    const ContactManifold manifold = MakeBoxManifold();

    RowBuffers old_rows;
    BuildContactRows(std::vector<ContactManifold>{manifold}, &old_rows);

    // 4 normal + 2 tangent = 6 rows (the legacy AppendContactGroup output).
    ASSERT_EQ(old_rows.RowCount(), 6u);
    EXPECT_EQ(old_rows.materials[0].normal_row_count, 4u);
    EXPECT_EQ(old_rows.materials[0].first_friction_row, 4u);
    EXPECT_EQ(old_rows.materials[0].friction_row_count, 2u);
    // Legacy normal rows have NO compliance (rhs/alpha == 0): unchanged.
    EXPECT_FLOAT_EQ(old_rows.rows[0].rhs, 0.0f);
    EXPECT_FLOAT_EQ(old_rows.rows[0].compliance_alpha, 0.0f);
}
