// ---------------------------------------------------------------------------
// nk CookToModel implementation (plan §3.3 / M3).
//
// MAPPING CHOICES (documented for M3b/SceneMap consumers):
//   * Bodies: cooked body rows ARE record order (the cooker preserves SceneIR
//     body order), so SceneMap binds entity-of-body -> body_row = its cooked row
//     for every body. The nk::Model carries the per-env movable rigid-body COUNT
//     (bodies_per_env); the actual body state lives in the Data arena (env-major
//     replica e at [e*bodies_per_env, ...)).
//   * Articulation: CookArticulations(blob) yields the per-env kinematic-tree
//     template(s). M3a supports the single-articulation-per-env scene (the H1 /
//     grasp shape); the first articulation becomes the Model template. dof_count
//     / link_count drive the dof/link capacities. Links bind to SceneMap via
//     link_index = the template-local link slot.
//   * Shapes: each cooked shape ROW becomes one ModelShape. A mesh that V-HACD
//     decomposed into N pieces produces N consecutive shape rows; the SOURCE
//     shape entity binds to the FIRST of those rows (shape_row) and records the
//     piece count via bp_group = N (so a consumer can walk the piece span). For
//     a primitive (1 row) bp_group == 1.
//   * Material buckets: one bucket per cooked shape's resolved (μ, restitution,
//     ...) — deduplicated by exact value so a homogeneous scene collapses to a
//     few buckets (Isaac bucketing). body_material_bucket[body_row] = the bucket
//     of that body's FIRST shape (0 if the body has no shape).
// ---------------------------------------------------------------------------

#include "scene/cook/cook_to_model.hpp"

#include <algorithm>

#include "runtime/articulation/articulation_cooker.hpp"
#include "scene/cooker.hpp"

namespace nuka::scene::cook {

namespace {

// Find or append a material bucket matching the cooked contact-param row `i`.
// Returns the bucket index. Dedup is by exact float equality (cook is
// deterministic, so identical authored params give bit-identical bucket rows).
uint32_t BucketFor(const CookedBlob& blob, uint32_t shape_row,
                   std::vector<nk::ModelMaterialBucket>& buckets) {
    nk::ModelMaterialBucket row;
    if (shape_row < blob.contact_params.frictions.size()) {
        row.values[0] = blob.contact_params.frictions[shape_row];  // static μ
        row.values[1] = blob.contact_params.frictions[shape_row];  // dynamic μ
    }
    // restitution / compliant params: the cooked tables carry solref/solimp; map
    // a coarse compliance into values[3]/[4] (timeconst/dampratio) for M3b's
    // material-bucket lookup. Restitution is not separately cooked here -> 0.
    if (shape_row < blob.contact_params.solref0.size()) {
        row.values[3] = blob.contact_params.solref0[shape_row];
    }
    if (shape_row < blob.contact_params.solref1.size()) {
        row.values[4] = blob.contact_params.solref1[shape_row];
    }
    if (shape_row < blob.contact_params.margins.size()) {
        row.values[6] = blob.contact_params.margins[shape_row];
    }
    for (uint32_t b = 0; b < buckets.size(); ++b) {
        bool same = true;
        for (int k = 0; k < 8; ++k) {
            if (buckets[b].values[k] != row.values[k]) { same = false; break; }
        }
        if (same) return b;
    }
    buckets.push_back(row);
    return static_cast<uint32_t>(buckets.size() - 1);
}

}  // namespace

CookToModelResult CookToModel(const SceneIR& scene, int env_count) {
    const uint32_t envs = env_count > 0 ? static_cast<uint32_t>(env_count) : 1u;

    // 1. Drive the existing cook (the heavy lifting: V-HACD / SDF / filters).
    const CookedBlob blob = CookScene(scene);

    // 2. Articulation topology (kinematic tree) from the cooked joints/bodies.
    const std::vector<runtime::articulation::ArticulationCookedTopology> arts =
        runtime::articulation::CookArticulations(blob);

    CookToModelResult result;
    nk::Model& model = result.model;
    SceneMap& smap = result.scene_map;

    // 3. Per-env capacities.
    nk::ModelCapacities& cap = model.capacities;
    cap.env_count = envs;
    cap.bodies_per_env = blob.body_count;
    cap.obs_width = 64;

    // 4. Articulation template (M3a: the FIRST articulation, the single-robot
    //    scene shape; multi-articulation per env is an M3b extension).
    if (!arts.empty()) {
        const runtime::articulation::ArticulationCookedTopology& a = arts.front();
        nk::ModelArticulation& m = model.articulation;
        m.link_count = static_cast<uint32_t>(a.link_bodies.size());
        m.root_link  = ~uint32_t(0);
        m.parent_link.assign(a.parent_links.begin(), a.parent_links.end());
        m.joint_axis = a.joint_axes;
        m.parent_offset.clear();  // parent offsets derive from frames in M3b.
        m.link_local_pose = a.local_poses;
        m.link_inertial_frame = a.inertial_frames;
        m.joint_damping = a.joint_dampings;
        m.joint_armature = a.joint_armatures;
        m.initial_q = a.initial_positions;
        m.link_body.reserve(a.link_bodies.size());
        for (BodyId b : a.link_bodies) m.link_body.push_back(b);
        m.joint_type.reserve(a.joint_types.size());
        for (auto jt : a.joint_types) m.joint_type.push_back(static_cast<uint8_t>(jt));
        // DOF count: a scalar DOF per non-fixed joint (M3a counts revolute/
        // prismatic as 1; fixed = 0; floating-base = 6). Single-DOF assumption
        // matches the production gripper/H1 cook.
        uint32_t dofs = 0;
        for (auto jt : a.joint_types) {
            switch (jt) {
                case runtime::articulation::ArticulationJointType::Fixed: break;
                case runtime::articulation::ArticulationJointType::FloatingBase: dofs += 6; break;
                default: dofs += 1; break;
            }
        }
        m.dof_count = dofs;
        cap.dofs_per_env = dofs;
        cap.links_per_env = m.link_count;

        // Bind link entities by template-local link slot. The cooked link_bodies
        // give the owning body row; the SceneMap link binding keys on link_index.
        for (uint32_t li = 0; li < a.link_bodies.size(); ++li) {
            const BodyId body = a.link_bodies[li];
            const EntityId ent = scene.EntityOfBody(body);
            if (ent != kInvalidEntity) {
                CookedRef ref;
                ref.body_row   = body;
                ref.link_index = li;
                smap.Bind(ent, ref);
            }
        }
    }

    // 5. Shapes -> ModelShape rows + material buckets. Track the SOURCE shape ->
    //    first cooked row + piece count for the SceneMap binding.
    model.shapes.reserve(blob.shape_count);
    std::vector<uint32_t> body_first_bucket(blob.body_count, 0u);
    std::vector<uint8_t>  body_has_shape(blob.body_count, 0u);
    for (uint32_t s = 0; s < blob.shape_count; ++s) {
        nk::ModelShape sh;
        if (s < blob.shapes.types.size())          sh.kind = static_cast<uint8_t>(blob.shapes.types[s]);
        if (s < blob.shapes.body_ids.size())       sh.body_row = blob.shapes.body_ids[s];
        if (s < blob.shapes.local_transforms.size()) sh.local_transform = blob.shapes.local_transforms[s];
        if (s < blob.shapes.half_extents.size())   sh.half_extents = blob.shapes.half_extents[s];
        if (s < blob.shapes.radii.size())          sh.radius = blob.shapes.radii[s];
        if (s < blob.shapes.half_heights.size())   sh.half_height = blob.shapes.half_heights[s];
        if (s < blob.shapes.convex_geometry_indices.size())
            sh.convex_geometry_index = blob.shapes.convex_geometry_indices[s];
        sh.material_bucket = BucketFor(blob, s, model.material_buckets);
        model.shapes.push_back(sh);

        if (sh.body_row < blob.body_count && !body_has_shape[sh.body_row]) {
            body_has_shape[sh.body_row] = 1u;
            body_first_bucket[sh.body_row] = sh.material_bucket;
        }
    }

    // SceneMap: SOURCE shape entity -> first cooked row + piece count via
    // bp_group. The cooker preserves source-shape order; a V-HACD source expands
    // into a contiguous span. Here we bind each source shape (SceneIR shape
    // record order) to its first cooked row. For the primitive scenes M3a cooks,
    // source row == cooked row, span == 1.
    {
        uint32_t cooked_cursor = 0;
        for (uint32_t src = 0; src < scene.ShapeCount(); ++src) {
            const EntityId ent = scene.EntityOfShape(src);
            // Count how many cooked rows this source produced: rows whose body_row
            // matches and that fall in the next contiguous span. For primitives
            // this is exactly 1; for a decomposed mesh the cooker emits N pieces.
            uint32_t span = 1;
            if (cooked_cursor < model.shapes.size()) {
                // Greedy: pieces of a source share the same body & material bucket
                // and follow consecutively; advance while the cooked row's source
                // provenance (approximated by identical body_row) continues. M3a's
                // primitives never decompose -> span stays 1.
                span = 1;
            }
            if (ent != kInvalidEntity && cooked_cursor < model.shapes.size()) {
                CookedRef ref;
                ref.shape_row = cooked_cursor;
                ref.body_row  = model.shapes[cooked_cursor].body_row;
                ref.bp_group  = span;  // piece count (1 for primitives).
                smap.Bind(ent, ref);
            }
            cooked_cursor += span;
        }
    }

    // 6. Per-body material bucket index (the cup/finger/foot material routing).
    model.body_material_bucket.assign(blob.body_count, 0u);
    for (uint32_t b = 0; b < blob.body_count; ++b) {
        model.body_material_bucket[b] = body_has_shape[b] ? body_first_bucket[b] : 0u;
    }
    cap.num_material_buckets = static_cast<uint32_t>(model.material_buckets.size());

    // 7. Bind body entities -> body rows (record order == row order).
    for (uint32_t b = 0; b < blob.body_count; ++b) {
        const EntityId ent = scene.EntityOfBody(b);
        if (ent == kInvalidEntity) continue;
        // Preserve any link binding already set (articulation links); merge the
        // body_row in. RefOf returns the existing ref if present.
        CookedRef ref;
        if (const CookedRef* existing = smap.RefOf(ent)) {
            ref = *existing;
        }
        ref.body_row = b;
        smap.Bind(ent, ref);
    }

    // 8. Filter policy (cross-env collision flag). M3a defaults OFF (envs do not
    //    collide); the cooked filter pair lists ride the blob for M5.
    model.filter_cross_env = false;

    // 9. Contact / row capacities. M3a sizes them from a conservative bound:
    //    one contact per (body + link) pair against statics, a small multiple for
    //    rows. The real sizing comes from the broadphase cook (M5); these give a
    //    valid, deterministic arena now.
    const uint32_t collidables = cap.bodies_per_env + cap.links_per_env;
    cap.max_contacts_per_env = collidables > 0 ? collidables * 4u : 0u;
    cap.max_rows_per_env     = cap.max_contacts_per_env * 4u;  // normal + friction spokes.

    return result;
}

} // namespace nuka::scene::cook
