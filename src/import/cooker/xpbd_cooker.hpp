#pragma once
// ---------------------------------------------------------------------------
// nuka::import::cooker -- XPBD soft-body COOKER (v0.7 p09-D, exit-crit 3)
// ---------------------------------------------------------------------------
//
// A THIN, deterministic, cook-time CPU orchestration layer that turns a
// programmatic soft-body SPEC into an XPBD constraint set (the host
// runtime::soft::XpbdConstraintSet the GPU solve iterates). It owns NO
// constraint MATH: it maps the authored material params (stiffness / damping /
// type) onto the engine's per-row fields and DISPATCHES to the existing p09-B/C
// cook-time topology builders:
//
//   type == Cloth       -> runtime::soft::BuildClothConstraints   (stretch + bend)
//   type == SoftBody     -> runtime::soft::BuildTetMeshConstraints (edge   + volume)
//   type == ShapeMatch   -> a single XpbdShapeMatchCluster (id 9; built here --
//                          there is no topology builder for the meshless cluster;
//                          cooking it is just rest-position + weight assembly).
//
// PARAM -> ENGINE MAPPING (the USD MaterialX schema, documented in full in
// docs/architecture/xpbd-soft-schema.md). The mapping is TYPE-DEPENDENT because
// the elastic rows and the shape-match cluster take DIFFERENT physical fields:
//
//   Cloth / SoftBody (elastic rows -- distance, bend, volume):
//       compliance_alpha = (stiffness > 0) ? 1/stiffness : 0   (0 == rigid)
//     The row field IS a compliance (1/stiffness), so the inverse is the mapping.
//
//   ShapeMatch (id 9 cluster):
//       cluster.stiffness = clamp(stiffness, 0, 1)              (goal-pull fraction)
//     The cluster's `stiffness` is the per-step goal-pull FRACTION in [0,1]
//     (p_i += w*stiffness*(goal_i - p_i)), NOT a compliance. Applying 1/k here
//     would INVERT the physics (high authored stiffness -> near-zero pull ->
//     floppy). So for shape_match the authored nuka:soft:stiffness is interpreted
//     DIRECTLY as the [0,1] fraction and clamped.
//
//   nuka:soft:damping -> RESERVED. No XPBD row struct (nor any topology Options)
//     carries a damping_beta field today, so damping is parsed/stored on the spec
//     but maps to NOTHING wired in the engine. It is reserved for a future XPBD
//     velocity-damping row; the cooker neither drops it silently nor invents a
//     field for it. See the schema doc.
//
// DETERMINISM: this cooker is a pure function of (spec). It only reorders/maps;
// the builders it calls emit constraints in a fixed sorted order, so two cooks of
// the same spec yield an IDENTICAL XpbdConstraintSet (asserted in the cook test).
//
// SCOPE: this is the PROGRAMMATIC cook target. USD/MJCF FILE PARSING for soft
// bodies (populating XpbdSoftBodySpec from an authored asset) is DEFERRED to the
// p16 demo asset pipeline (folded into the mesh-loader orphan, task #19); the
// future importer fills this spec and calls CookXpbdSoftBody. This file does NOT
// parse files and does NOT touch usd_importer.cpp / mjcf_importer.cpp.
// ---------------------------------------------------------------------------

#include "import/cooker/xpbd_cooker_types.hpp"  // XpbdConstraintSet/XpbdShapeMatchCluster (CUDA-free)
#include "math/vec3.hpp"
#include "runtime/soft/cloth_topology.hpp"
#include "runtime/soft/tetmesh_topology.hpp"

#include <cstdint>
#include <vector>

namespace nuka::import::cooker {

// The authored soft-body kind. Matches the `nuka:soft:type` USD token:
//   Cloth     <- "cloth"        (triangle mesh -> distance + bend)
//   SoftBody  <- "softbody"     (tet mesh      -> distance + volume)
//   ShapeMatch<- "shape_match"  (one meshless cluster over all particles)
enum class SoftBodyType : uint8_t { Cloth, SoftBody, ShapeMatch };

// Programmatic description of one soft body to cook. The future USD/MJCF importer
// (p16/#19) populates this; today the tests + any in-tree caller build it
// directly. Exactly one of `triangles` / `tets` is used, per `type`:
//   Cloth      -> triangles (each ClothTriangle = 3 particle indices)
//   SoftBody   -> tets      (each TetMeshTet    = 4 particle indices)
//   ShapeMatch -> neither   (the whole particle set is one cluster)
struct XpbdSoftBodySpec {
    SoftBodyType type = SoftBodyType::Cloth;

    // Rest geometry: one Vec3 per particle, indexed by the triangle/tet indices.
    std::vector<math::Vec3> rest_positions;

    // Cloth topology (used iff type == Cloth).
    std::vector<runtime::soft::ClothTriangle> triangles;

    // Tet topology (used iff type == SoftBody).
    std::vector<runtime::soft::TetMeshTet> tets;

    // nuka:soft:stiffness. Cloth/SoftBody: 1/stiffness -> compliance_alpha
    // (<=0 => alpha 0 == rigid). ShapeMatch: clamped into [0,1] as the goal-pull
    // fraction. Default 1e4 (a fairly stiff elastic body).
    float stiffness = 1.0e4f;

    // nuka:soft:damping. RESERVED -- stored but not wired to any engine field
    // today (no row struct carries damping_beta). Default 0.01 to match the
    // schema's documented default.
    float damping = 0.01f;

    // Cloth-only: emit the bend constraints (interior-edge Bergou stencil). When
    // false, only the stretch (distance) constraints are emitted. Mirrors
    // ClothTopologyOptions::emit_bend_constraints. Ignored for non-Cloth types.
    bool emit_bend = true;

    // SoftBody-only: emit the per-edge distance constraints alongside the volume
    // constraints. Mirrors TetMeshTopologyOptions::emit_distance_constraints.
    // Ignored for non-SoftBody types.
    bool emit_tet_edges = true;
};

// Map an authored elastic stiffness to an XPBD compliance_alpha (= 1/stiffness;
// stiffness <= 0 => 0 == rigid/inextensible). The single source of truth for the
// elastic-row mapping; exposed so the cook test asserts it directly.
float ComplianceAlphaFromStiffness(float stiffness);

// Map an authored stiffness to a shape-match goal-pull fraction (clamp to [0,1]).
// Exposed so the cook test asserts the (distinct) shape-match mapping directly.
float ShapeMatchFractionFromStiffness(float stiffness);

// Cook one soft-body spec into an XpbdConstraintSet. PURE + deterministic:
// dispatches on spec.type, maps the material params, and appends the constraints
// (via the existing topology builders for Cloth/SoftBody, or a directly-assembled
// single cluster for ShapeMatch). The constraint MATH lives in the builders / the
// id9 row; this only orchestrates. The result feeds the nk soft world cook
// (nk::Model::ModelParticles, mode == Xpbd) by the caller.
runtime::soft::XpbdConstraintSet CookXpbdSoftBody(const XpbdSoftBodySpec& spec);

} // namespace nuka::import::cooker
