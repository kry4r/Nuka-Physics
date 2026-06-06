// ---------------------------------------------------------------------------
// nuka::runtime::coupling -- the PBD<->rigid unified-contact CO-STEP bridge
// IMPLEMENTATION (v0.8 C6a). See unified_costep.hpp for the design + contract.
// ---------------------------------------------------------------------------
// The body wires the REAL v0.8 front-end together for the "couple" phase. Every
// stage REUSES an existing, validated piece -- nothing here re-derives broadphase
// / narrowphase / row emission / solve. The only NEW logic is the host glue:
//   (a) the rigid box AABBs + a body-id == box-index map,
//   (b) the ShapeResolver (Particle -> Sphere prim @ particle pos; RigidBody ->
//       Box prim @ box pose), indexing the host SoAs by ref.handle,
//   (c) building the 1:1 CouplingPairs from each EMITTED manifold's own {a,b}
//       (the driver skips 0-point pairs, so the manifolds are NOT 1:1 with the
//       broadphase candidates -- the manifolds' refs are the source of truth).
// ---------------------------------------------------------------------------

#include "runtime/coupling/unified_costep.hpp"

#include "collision/aabb.hpp"                     // AABB
#include "collision/analytical_manifold.hpp"     // amf::PrimParams
#include "collision/broadphase_lbvh.hpp"         // BuildLbvhForQuery
#include "collision/candidate_pair.hpp"          // CandidatePair, CollidableRef
#include "collision/contact_stream_driver.hpp"   // BuildContactManifolds, ResolvedShape
#include "collision/particle_candidate_pairs.hpp"// BuildParticleRigidCandidatePairs
#include "constraint/collidable.hpp"             // CollidableType
#include "constraint/contact_manifold.hpp"       // ContactManifold
#include "constraint/contact_row_sides.hpp"      // ContactRowSides
#include "constraint/row_buffers.hpp"            // RowBuffers
#include "phi/buffer.hpp"                         // phi::Buffer
#include "phi/buffer_transfer.hpp"               // phi::UploadVector
#include "runtime/coupling/coupling_row_framework.hpp"  // EmitCouplingRows, CouplingPair
#include "solver/unified_solve.hpp"              // UnifiedSolve, SolveContext

#include <cstdint>
#include <vector>

namespace nuka::runtime::coupling {

namespace {

namespace amf = collision::amf;
using collision::AABB;
using collision::CandidatePair;
using collision::ResolvedShape;
using collision::ShapeResolver;
using constraint::CollidableRef;
using constraint::CollidableType;
using constraint::ContactManifold;
using constraint::ContactRowSides;
using constraint::RowBuffers;
using math::Vec3;
using scene::SystemPairMatrix;

}  // namespace

UnifiedCoStepReport StepUnifiedParticleRigidCoupling(
    const phi::DeviceContext& context,
    const Vec3* particle_positions,
    std::vector<Vec3>* particle_velocities,
    const std::vector<float>& particle_inv_mass,
    uint32_t particle_count,
    float particle_radius,
    scene::SystemKind system,
    const std::vector<CoupleBox>& boxes,
    std::vector<runtime::rigid::BodyState>* bodies,
    const constraint::ContactRowComplianceInputs& compliance,
    float dt,
    const solver::SolverConfig& config) {
    UnifiedCoStepReport report;
    if (particle_positions == nullptr || particle_velocities == nullptr ||
        bodies == nullptr || particle_count == 0u || boxes.empty()) {
        return report;
    }

    // --- (a) the rigid box world AABBs + the body-id == box-index map ----------
    // Body id == box index 0..box_count-1 so the rigid-side handle the emitter
    // reads off the manifold (manifold.b.handle) directly indexes `*bodies`.
    std::vector<AABB> rigid_aabbs;
    std::vector<uint32_t> rigid_body_ids;
    rigid_aabbs.reserve(boxes.size());
    rigid_body_ids.reserve(boxes.size());
    for (uint32_t i = 0u; i < boxes.size(); ++i) {
        const CoupleBox& b = boxes[i];
        AABB box;
        box.min = b.center - Vec3{b.half_extent, b.half_extent, b.half_extent};
        box.max = b.center + Vec3{b.half_extent, b.half_extent, b.half_extent};
        rigid_aabbs.push_back(box);
        rigid_body_ids.push_back(i);
    }

    // --- the rigid LBVH (retain nodes for the cross-system query) -------------
    phi::Buffer d_rigid = phi::UploadVector(rigid_aabbs);
    auto tree = collision::gpu::BuildLbvhForQuery(
        context, static_cast<const AABB*>(d_rigid.Data()),
        static_cast<uint32_t>(rigid_aabbs.size()));
    if (!tree.HasNodes()) {
        return report;
    }

    // --- (b1) the broadphase: upload PREDICTED particle positions + query -----
    phi::Buffer d_part = phi::UploadVector(
        std::vector<Vec3>(particle_positions, particle_positions + particle_count));
    const auto* d_part_ptr = static_cast<const Vec3*>(d_part.Data());

    SystemPairMatrix matrix;  // default all-enabled; gate is meaningful per system.
    auto stream = collision::BuildParticleRigidCandidatePairs(
        context, d_part_ptr, particle_count, system, tree, rigid_body_ids.data(),
        particle_radius, matrix);
    const std::vector<CandidatePair> pairs = stream.DownloadPairs();
    report.candidate_pairs = static_cast<uint32_t>(pairs.size());
    if (pairs.empty()) {
        return report;
    }

    // --- (b2) the ShapeResolver (the C5 driver decoupling seam) ---------------
    // Particle side -> a Sphere prim at the particle's PREDICTED world position
    // (indexed by ref.handle == particle SoA index). RigidBody side -> an
    // axis-aligned Box prim at the box pose (indexed by ref.handle == box index).
    ShapeResolver resolve = [&](const CollidableRef& ref, ResolvedShape* out) -> bool {
        if (ref.type == CollidableType::Particle) {
            if (ref.handle >= particle_count) {
                return false;
            }
            out->type = scene::ShapeType::Sphere;
            amf::PrimParams p;
            p.radius = particle_radius;
            p.frame.t = particle_positions[ref.handle];  // identity rotation.
            out->prim = p;
            return true;
        }
        if (ref.type == CollidableType::RigidBody) {
            if (ref.handle >= boxes.size()) {
                return false;
            }
            const CoupleBox& b = boxes[ref.handle];
            out->type = scene::ShapeType::Box;
            amf::PrimParams p;
            p.half_extents = Vec3{b.half_extent, b.half_extent, b.half_extent};
            p.frame.t = b.center;  // identity frame -> axis-aligned box.
            out->prim = p;
            return true;
        }
        return false;
    };

    // --- (c) drive the manifolds (sphere x box -> analytical tier) ------------
    std::vector<ContactManifold> manifolds;
    collision::BuildContactManifolds(pairs, resolve, &manifolds);
    report.manifolds = static_cast<uint32_t>(manifolds.size());
    if (manifolds.empty()) {
        return report;
    }

    // --- build 1:1 CouplingPairs from each EMITTED manifold's own {a,b} --------
    // The driver SKIPS 0-point pairs, so the manifolds are NOT 1:1 with `pairs`;
    // each manifold carries the two CollidableRefs the narrowphase stamped, so
    // the cleanest 1:1 coupling-pair stream is built directly from them.
    std::vector<CouplingPair> coupling_pairs;
    coupling_pairs.reserve(manifolds.size());
    for (const ContactManifold& m : manifolds) {
        coupling_pairs.push_back(CouplingPair{m.a, m.b, m.manifold_key});
    }

    // --- emit the compliant coupling rows (re-classed to id13) ----------------
    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    EmitCouplingRows(
        std::span<const CouplingPair>(coupling_pairs.data(), coupling_pairs.size()),
        std::span<const ContactManifold>(manifolds.data(), manifolds.size()),
        compliance, &rows, &sides);
    report.rows = rows.RowCount();
    if (rows.RowCount() == 0u) {
        return report;
    }
    // Count the rows whose front-end-tagged sides carry a ParticleInvMass arm (the
    // builder/registry fills the particle side's react; StampSides copies it onto
    // the manifold; the emitter copies it onto the ContactRowSides). A nonzero count
    // proves UnifiedSolve dispatches the particle reaction arm.
    for (const ContactRowSides& s : sides) {
        if (s.a.react == constraint::ReactionProviderKind::ParticleInvMass ||
            s.b.react == constraint::ReactionProviderKind::ParticleInvMass) {
            ++report.particle_arm_rows;
        }
    }

    // --- the unified two-way solve (mutates particle velocities + bodies) ------
    solver::SolveContext ctx;
    ctx.rows = &rows;
    ctx.state = bodies;
    ctx.sides = &sides;
    ctx.dt = dt;
    ctx.particles.inv_mass = &particle_inv_mass;
    ctx.particles.velocities = particle_velocities;
    solver::UnifiedSolve(ctx, config);

    return report;
}

}  // namespace nuka::runtime::coupling
