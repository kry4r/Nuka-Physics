// CouplingProvider op-list guard: the row provider emits a deterministic,
// canonically-ordered OpCall list (build-twice byte-identical incl. Params PODs).

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "math/vec3.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/pipeline.hpp"
#include "nk/solve/nk_row.hpp"   // kPairDrivenRowsPerSlot
#include "phi/backend.hpp"
#include "phi/op_schema.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
using nuka::math::Vec3;

constexpr uint32_t kKindBox = 2u;

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    cfg.vel_iters = 32u;
    cfg.pos_iters = 4u;
    cfg.contact_margin = 0.0f;
    return cfg;
}

// A static-box collidable so has_collidables / has_contacts fire (the rigid
// contact ops + the body<->particle slot reserve the coupling rides).
void AddStaticBox(nk::Model& m, const Vec3& pos, float half, int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = nuka::math::Transform::Identity();
    bi.pose.position = pos;
    bi.inv_mass = 0.0f;
    bi.inv_inertia = Vec3{0, 0, 0};
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindBox;
    sh.params[0] = half; sh.params[1] = half; sh.params[2] = half;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

// Rigid-only: two static boxes, no particles, no articulation.
nk::Model BuildRigidOnly() {
    nk::Model m;
    AddStaticBox(m, Vec3{0, 0, 0}, 0.10f, 0);
    AddStaticBox(m, Vec3{1, 0, 0}, 0.05f, 1);
    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.env_count = 1u;
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_contacts_per_env = 8u;
    cap.max_rows_per_env = 8u * nk::kPairDrivenRowsPerSlot;
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;
    return m;
}

// Articulated/go2-shaped: links + dofs present (no real chain data needed — the
// op-list membership is a capacity property, not a topology property).
nk::Model BuildArticulated() {
    nk::Model m;
    AddStaticBox(m, Vec3{0, 0, 0}, 0.10f, 0);
    nk::ModelCapacities& cap = m.capacities;
    cap.env_count = 1u;
    cap.bodies_per_env = 1u;
    cap.max_bodies_total = 1u;
    cap.links_per_env = 13u;
    cap.dofs_per_env = 12u;
    cap.articulations_per_env = 1u;
    cap.max_contacts_per_env = 8u;
    cap.max_rows_per_env = 8u * nk::kPairDrivenRowsPerSlot;
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;
    return m;
}

// Add a particle system of the given mode onto a collidable base.
void AddParticles(nk::Model& m, nk::Model::ParticleMode mode, uint32_t count,
                  uint32_t n_soft, nk::Model::CoupledInternal ci) {
    nk::Model::ModelParticles& mp = m.particles;
    mp.mode = mode;
    mp.coupled_internal = ci;
    mp.n_soft_particles = n_soft;
    for (uint32_t i = 0; i < count; ++i) {
        mp.initial_pos.push_back(Vec3{0.0f, 0.0f, 0.2f + 0.01f * i});
        mp.initial_vel.push_back(Vec3{0.0f, 0.0f, -1.0f});
        mp.inv_mass.push_back(1.0f / 0.01f);
    }
    mp.xpbd_iters = 1u;
    mp.pp_contact_d_min = 0.04f;
    mp.pp_contact_compliance = 0.0f;
    mp.pp_contact_iters = 1u;
    // PBF coefficients so the fluid arm fills non-trivial Params (still byte-exact
    // build-to-build; the point is to exercise the moved field assignments).
    mp.pbf_rest_density = 1000.0f;
    mp.pbf_support_radius = 0.05f;
    mp.pbf_particle_mass = 0.01f;
    mp.pbf_iters = 2u;
    mp.pbf_xsph_viscosity = 0.01f;
    mp.pbf_surface_tension = 0.001f;
    mp.cell_size = 0.05f;
    mp.query_radius = 0.05f;
    mp.grid_dims[0] = mp.grid_dims[1] = mp.grid_dims[2] = 8u;
    nk::ModelCapacities& cap = m.capacities;
    cap.particles_per_env = count;
    // The cook grows the contact budget for the particle reserve; mirror that so
    // the rigid + body<->particle slot sub-ranges are disjoint (rigid_cap > 0).
    cap.max_contacts_per_env += count * 4u;
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    cap.max_grid_cells = 8u * 8u * 8u;
}

nk::Model BuildXpbdSoft() {
    nk::Model m = BuildRigidOnly();
    AddParticles(m, nk::Model::ParticleMode::Xpbd, 4u, 0u,
                 nk::Model::CoupledInternal::None);
    return m;
}

nk::Model BuildPbfFluid() {
    nk::Model m = BuildRigidOnly();
    AddParticles(m, nk::Model::ParticleMode::Pbf, 4u, 0u,
                 nk::Model::CoupledInternal::None);
    return m;
}

nk::Model BuildCoupled() {
    nk::Model m = BuildRigidOnly();
    AddParticles(m, nk::Model::ParticleMode::Coupled, 4u, 0u,
                 nk::Model::CoupledInternal::Pbf);
    return m;
}

nk::Model BuildSoftFluid() {
    nk::Model m = BuildRigidOnly();
    AddParticles(m, nk::Model::ParticleMode::SoftFluid, 6u, 3u,
                 nk::Model::CoupledInternal::None);
    return m;
}

// Robot + cloth + fluid: articulated base + a SoftFluid particle mix.
nk::Model BuildRobotClothFluid() {
    nk::Model m = BuildArticulated();
    AddParticles(m, nk::Model::ParticleMode::SoftFluid, 8u, 4u,
                 nk::Model::CoupledInternal::None);
    return m;
}

// The canonical fixed op order helper predicates: capture an op-enum sequence.
std::vector<nphi::NkOp> OpSequence(const nk::Pipeline& p) {
    std::vector<nphi::NkOp> seq;
    seq.reserve(p.Calls().size());
    for (const nphi::OpCall& c : p.Calls()) seq.push_back(c.op);
    return seq;
}

// Find the first OpCall of a given op and copy its Params bytes out (the POD is
// trivially-copyable -> memcmp is exact). Returns false if the op is absent.
template <class ParamsT>
bool CopyParams(const nk::Pipeline& p, nphi::NkOp op, ParamsT* out) {
    for (const nphi::OpCall& c : p.Calls()) {
        if (c.op == op) {
            std::memcpy(out, c.params, sizeof(ParamsT));
            return true;
        }
    }
    return false;
}

// Build twice and assert the op sequence + the three moved PODs are identical.
void AssertBuildTwiceIdentical(nk::Model (*build)(), const char* name) {
    nk::Model m1 = build();
    nk::Model m2 = build();
    nk::Pipeline p1, p2;
    p1.Build(m1, Cfg(), nullptr);
    p2.Build(m2, Cfg(), nullptr);

    const std::vector<nphi::NkOp> s1 = OpSequence(p1);
    const std::vector<nphi::NkOp> s2 = OpSequence(p2);
    ASSERT_EQ(s1.size(), s2.size()) << name << ": op count differs build-to-build";
    for (size_t i = 0; i < s1.size(); ++i) {
        EXPECT_EQ(static_cast<int>(s1[i]), static_cast<int>(s2[i]))
            << name << ": op[" << i << "] differs build-to-build";
    }

    // FULL byte memcmp of each moved POD (present iff has_particles).
    const bool has_particles = m1.capacities.particles_per_env > 0u;
    nphi::NarrowphaseBodyParticleParams np1{}, np2{};
    nphi::ParticleFinalizeParams pf1{}, pf2{};
    nphi::ParticleParticleContactParams pp1{}, pp2{};
    const bool np_a = CopyParams(p1, nphi::NkOp::NarrowphaseBodyParticle, &np1);
    const bool np_b = CopyParams(p2, nphi::NkOp::NarrowphaseBodyParticle, &np2);
    const bool pf_a = CopyParams(p1, nphi::NkOp::ParticleFinalize, &pf1);
    const bool pf_b = CopyParams(p2, nphi::NkOp::ParticleFinalize, &pf2);
    const bool pp_a = CopyParams(p1, nphi::NkOp::ParticleParticleContact, &pp1);
    const bool pp_b = CopyParams(p2, nphi::NkOp::ParticleParticleContact, &pp2);
    EXPECT_EQ(np_a, has_particles) << name << ": NarrowphaseBodyParticle presence";
    EXPECT_EQ(pf_a, has_particles) << name << ": ParticleFinalize presence";
    EXPECT_EQ(pp_a, has_particles) << name << ": ParticleParticleContact presence";
    EXPECT_EQ(np_a, np_b);
    EXPECT_EQ(pf_a, pf_b);
    EXPECT_EQ(pp_a, pp_b);
    if (has_particles) {
        EXPECT_EQ(std::memcmp(&np1, &np2, sizeof(np1)), 0)
            << name << ": NarrowphaseBodyParticleParams bytes differ";
        EXPECT_EQ(std::memcmp(&pf1, &pf2, sizeof(pf1)), 0)
            << name << ": ParticleFinalizeParams bytes differ";
        EXPECT_EQ(std::memcmp(&pp1, &pp2, sizeof(pp1)), 0)
            << name << ": ParticleParticleContactParams bytes differ";
    }
}

// Index of the first occurrence of an op in the sequence (-1 if absent), used to
// assert the relative emission order the coupling extraction must preserve.
int IndexOf(const std::vector<nphi::NkOp>& seq, nphi::NkOp op) {
    for (size_t i = 0; i < seq.size(); ++i)
        if (seq[i] == op) return static_cast<int>(i);
    return -1;
}

}  // namespace

TEST(CouplingProviderOpList, RigidOnlyBuildTwiceIdentical) {
    AssertBuildTwiceIdentical(&BuildRigidOnly, "rigid-only");
}
TEST(CouplingProviderOpList, ArticulatedBuildTwiceIdentical) {
    AssertBuildTwiceIdentical(&BuildArticulated, "articulated");
}
TEST(CouplingProviderOpList, XpbdSoftBuildTwiceIdentical) {
    AssertBuildTwiceIdentical(&BuildXpbdSoft, "xpbd-soft");
}
TEST(CouplingProviderOpList, PbfFluidBuildTwiceIdentical) {
    AssertBuildTwiceIdentical(&BuildPbfFluid, "pbf-fluid");
}
TEST(CouplingProviderOpList, CoupledBuildTwiceIdentical) {
    AssertBuildTwiceIdentical(&BuildCoupled, "coupled");
}
TEST(CouplingProviderOpList, SoftFluidBuildTwiceIdentical) {
    AssertBuildTwiceIdentical(&BuildSoftFluid, "soft-fluid");
}
TEST(CouplingProviderOpList, RobotClothFluidBuildTwiceIdentical) {
    AssertBuildTwiceIdentical(&BuildRobotClothFluid, "robot-cloth-fluid");
}

// Rigid-only registers NO coupling provider op: none of the three body<->particle
// ops appear, the op list is literally today's rigid path.
TEST(CouplingProviderOpList, RigidOnlyEmitsNoCouplingOps) {
    nk::Model m = BuildRigidOnly();
    nk::Pipeline p;
    p.Build(m, Cfg(), nullptr);
    const std::vector<nphi::NkOp> seq = OpSequence(p);
    EXPECT_EQ(IndexOf(seq, nphi::NkOp::NarrowphaseBodyParticle), -1);
    EXPECT_EQ(IndexOf(seq, nphi::NkOp::ParticleFinalize), -1);
    EXPECT_EQ(IndexOf(seq, nphi::NkOp::ParticleParticleContact), -1);
}

// The row provider's emission points: NarrowphaseBodyParticle between Sdf and the
// tangent basis; ParticleFinalize then ParticleParticleContact after IntegratePos.
TEST(CouplingProviderOpList, RowProviderEmitsAtCanonicalPositions) {
    nk::Model m = BuildSoftFluid();
    nk::Pipeline p;
    p.Build(m, Cfg(), nullptr);
    const std::vector<nphi::NkOp> seq = OpSequence(p);

    const int sdf = IndexOf(seq, nphi::NkOp::NarrowphaseSdf);
    const int bp  = IndexOf(seq, nphi::NkOp::NarrowphaseBodyParticle);
    const int tan = IndexOf(seq, nphi::NkOp::ContactTangentBasis);
    ASSERT_GE(sdf, 0); ASSERT_GE(bp, 0); ASSERT_GE(tan, 0);
    EXPECT_LT(sdf, bp) << "NarrowphaseBodyParticle must follow NarrowphaseSdf";
    EXPECT_LT(bp, tan) << "NarrowphaseBodyParticle must precede ContactTangentBasis";

    const int ipos = IndexOf(seq, nphi::NkOp::IntegratePosition);
    const int fin  = IndexOf(seq, nphi::NkOp::ParticleFinalize);
    const int ppc  = IndexOf(seq, nphi::NkOp::ParticleParticleContact);
    const int rd   = IndexOf(seq, nphi::NkOp::ReadoutContactWrench);
    ASSERT_GE(ipos, 0); ASSERT_GE(fin, 0); ASSERT_GE(ppc, 0); ASSERT_GE(rd, 0);
    EXPECT_LT(ipos, fin) << "ParticleFinalize must follow IntegratePosition";
    EXPECT_LT(fin, ppc) << "ParticleParticleContact must follow ParticleFinalize";
    EXPECT_LT(ppc, rd) << "ParticleParticleContact must precede ReadoutContactWrench";
}
