// ---------------------------------------------------------------------------
// nuka::import::cooker -- XPBD soft-body cooker (v0.7 p09-D). See xpbd_cooker.hpp
// for the param->engine mapping and scope. THIN: no constraint math here.
// ---------------------------------------------------------------------------

#include "import/cooker/xpbd_cooker.hpp"

#include <cstdint>

namespace nuka::import::cooker {

namespace {
using runtime::soft::BuildClothConstraints;
using runtime::soft::BuildTetMeshConstraints;
using runtime::soft::ClothTopologyOptions;
using runtime::soft::TetMeshTopologyOptions;
using runtime::soft::XpbdConstraintSet;
using runtime::soft::XpbdShapeMatchCluster;
}  // namespace

float ComplianceAlphaFromStiffness(float stiffness) {
    // 1/stiffness; stiffness <= 0 => rigid/inextensible (alpha 0). Guards a
    // div-by-zero and the "rigid" sentinel in one branch.
    return (stiffness > 0.0f) ? (1.0f / stiffness) : 0.0f;
}

float ShapeMatchFractionFromStiffness(float stiffness) {
    // The cluster `stiffness` is a per-step goal-pull FRACTION in [0,1], NOT a
    // compliance: clamp the authored value into [0,1] (no inversion).
    if (stiffness <= 0.0f) return 0.0f;
    if (stiffness >= 1.0f) return 1.0f;
    return stiffness;
}

XpbdConstraintSet CookXpbdSoftBody(const XpbdSoftBodySpec& spec) {
    XpbdConstraintSet out;

    switch (spec.type) {
        case SoftBodyType::Cloth: {
            ClothTopologyOptions opts;
            const float alpha = ComplianceAlphaFromStiffness(spec.stiffness);
            opts.distance_compliance_alpha = alpha;
            opts.bend_compliance_alpha = alpha;
            opts.emit_distance_constraints = true;
            opts.emit_bend_constraints = spec.emit_bend;
            BuildClothConstraints(spec.rest_positions, spec.triangles, opts, out);
            break;
        }
        case SoftBodyType::SoftBody: {
            TetMeshTopologyOptions opts;
            const float alpha = ComplianceAlphaFromStiffness(spec.stiffness);
            opts.distance_compliance_alpha = alpha;
            opts.volume_compliance_alpha = alpha;
            opts.emit_distance_constraints = spec.emit_tet_edges;
            BuildTetMeshConstraints(spec.rest_positions, spec.tets, opts, out);
            break;
        }
        case SoftBodyType::ShapeMatch: {
            // No topology builder exists for the meshless cluster (id 9): cooking
            // it is just rest-position + weight assembly over the whole particle
            // set, so do it inline (still THIN -- the polar-decomposition / goal
            // MATH lives in the GPU id9 row, not here). One cluster per spec.
            // The nk soft cook derives the rest centroid c0 and the q_i =
            // x_i^0 - c0 from rest_positions + cluster_mass; we only fill the raw
            // rest data.
            if (!spec.rest_positions.empty()) {
                XpbdShapeMatchCluster cluster;
                const uint32_t n = static_cast<uint32_t>(spec.rest_positions.size());
                cluster.particle.reserve(n);
                cluster.rest_positions.reserve(n);
                cluster.cluster_mass.reserve(n);
                for (uint32_t i = 0; i < n; ++i) {
                    cluster.particle.push_back(i);
                    cluster.rest_positions.push_back(spec.rest_positions[i]);
                    // Uniform shape-match weight. The spec carries no per-particle
                    // mass (rest geometry only), so every particle contributes
                    // equally to the centroid / covariance sums -- the standard
                    // shape-match default (Mueller et al. 2005). A future importer
                    // that knows per-particle masses can override this.
                    cluster.cluster_mass.push_back(1.0f);
                }
                cluster.stiffness = ShapeMatchFractionFromStiffness(spec.stiffness);
                out.shape_match.push_back(std::move(cluster));
            }
            break;
        }
    }

    return out;
}

} // namespace nuka::import::cooker
