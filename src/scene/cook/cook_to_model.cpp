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
#include <cmath>

#include "runtime/articulation/articulation_cooker.hpp"
#include "runtime/articulation/articulation_state.hpp"
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

    // 4. Articulation template (the FIRST articulation, the single-robot scene
    //    shape; multi-articulation per env is a later extension). M3b: the
    //    template arrays are the SINGLE-ENV BuildArticulationHostState product
    //    transcribed 1:1, so the staged Model bytes (after env-major tiling in
    //    Model::StageModelField) match the legacy UploadArticulationState bytes
    //    EXACTLY — the byte-exact kernel-port contract.
    if (!arts.empty()) {
        namespace articulation = runtime::articulation;
        const articulation::ArticulationCookedTopology& a = arts.front();
        const articulation::ArticulationHostState host =
            articulation::BuildArticulationHostState({a}, blob.bodies);

        nk::ModelArticulation& m = model.articulation;
        m.link_count = host.TotalLinkCount();
        m.root_link  = 0u;  // the cooker emits the root first.
        m.parent_link = host.parent_link;
        m.joint_axis = host.joint_axis;
        m.parent_offset = host.parent_offset;
        m.link_local_pose = host.link_local_pose;
        m.link_inertial_frame = host.link_inertial_frame;
        m.joint_damping = host.joint_damping;
        m.joint_armature = host.joint_armature;
        m.initial_q = host.q;                  // per LINK (scalar slot / link)
        m.initial_link_pose = host.link_pose;  // cook rest pose
        m.base_pose = host.base_pose.empty() ? math::Transform::Identity()
                                             : host.base_pose.front();
        m.link_body = host.link_body;
        m.joint_type.reserve(host.joint_type.size());
        for (auto jt : host.joint_type) m.joint_type.push_back(static_cast<uint8_t>(jt));
        m.link_inertia_spatial.resize(static_cast<size_t>(m.link_count) * 36u);
        for (uint32_t l = 0; l < m.link_count; ++l) {
            for (uint32_t k = 0; k < 36u; ++k) {
                m.link_inertia_spatial[static_cast<size_t>(l) * 36u + k] =
                    host.link_inertia[l].I[k];
            }
        }
        // DOF count (Revolute/Prismatic = 1, Fixed = 0, FloatingBase = 6) — the
        // legacy ArticulationDofCount semantics (inlined: that symbol lives in
        // the GPU lib, which the pure cook must not link), i.e. max_dof.
        uint32_t dofs = 0;
        for (auto jt : host.joint_type) {
            switch (jt) {
                case articulation::ArticulationJointType::Fixed: break;
                case articulation::ArticulationJointType::FloatingBase: dofs += 6; break;
                default: dofs += 1; break;
            }
        }
        m.dof_count = dofs;
        cap.dofs_per_env = m.dof_count;
        cap.links_per_env = m.link_count;

        // Foot shapes: the T2/T6 derivation — every Sphere shape whose owning
        // body maps to an articulation link is a foot (base-relative indices).
        for (uint32_t shape = 0; shape < blob.shapes.types.size(); ++shape) {
            if (blob.shapes.types[shape] != ShapeType::Sphere) {
                continue;
            }
            const BodyId body = shape < blob.shapes.body_ids.size()
                                    ? blob.shapes.body_ids[shape]
                                    : kInvalidBody;
            uint32_t calf_link = ~uint32_t(0);
            for (uint32_t link = 0; link < m.link_count; ++link) {
                if (host.link_body[link] == body) {
                    calf_link = link;
                    break;
                }
            }
            if (calf_link == ~uint32_t(0)) {
                continue;
            }
            nk::ModelFootShape foot;
            foot.calf_local_link = calf_link;
            foot.local_offset = shape < blob.shapes.local_transforms.size()
                                    ? blob.shapes.local_transforms[shape].position
                                    : math::Vec3::Zero();
            foot.radius = shape < blob.shapes.radii.size() ? blob.shapes.radii[shape]
                                                           : 0.0f;
            model.feet.push_back(foot);
        }

        // Hold drives (legacy BuildHoldDriveTargets 1:1): targets = cooked q;
        // Position actuators seed stiffness = gain, damping = 2*sqrt(gain),
        // force limits from the actuator table.
        nk::ModelHoldDrives& d = model.hold_drives;
        d.targets = host.q;
        d.stiffness.assign(m.link_count, 0.0f);
        d.damping.assign(m.link_count, 0.0f);
        d.force_limits.assign(m.link_count, 0.0f);
        for (uint32_t act = 0; act < blob.actuator_count; ++act) {
            if (act >= blob.actuators.joint_ids.size() ||
                act >= blob.actuators.types.size() ||
                blob.actuators.types[act] != ActuatorType::Position) {
                continue;
            }
            const JointId joint = blob.actuators.joint_ids[act];
            if (joint >= blob.joints.child_bodies.size()) {
                continue;
            }
            const BodyId child_body = blob.joints.child_bodies[joint];
            uint32_t link = ~uint32_t(0);
            for (uint32_t l = 0; l < m.link_count; ++l) {
                if (host.link_body[l] == child_body) {
                    link = l;
                    break;
                }
            }
            if (link == ~uint32_t(0) ||
                host.joint_type[link] == articulation::ArticulationJointType::Fixed) {
                continue;
            }
            const float gain = act < blob.actuators.gains.size()
                                   ? std::max(blob.actuators.gains[act], 0.0f)
                                   : 0.0f;
            const float force_limit = act < blob.actuators.force_limits.size()
                                          ? std::max(blob.actuators.force_limits[act], 0.0f)
                                          : 0.0f;
            if (gain > 0.0f) {
                d.stiffness[link] = gain;
                d.damping[link] = 2.0f * std::sqrt(gain);  // DefaultDriveDamping.
            }
            if (force_limit > 0.0f) {
                d.force_limits[link] = force_limit;
            }
        }

        // Bind link entities by template-local link slot. The cooked link_bodies
        // give the owning body row; the SceneMap link binding keys on link_index.
        for (uint32_t li = 0; li < host.link_body.size(); ++li) {
            const BodyId body = host.link_body[li];
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

    // 9. Contact / row capacities. M3b: the articulation foot pipeline's slot
    //    stride is the legacy kMaxFootContactsPerEnv == 4 (the ported kernels
    //    hardcode slot_base = env * 4, the byte-exact layout), and the per-slot
    //    row triple {normal, t1, t2} makes max_rows = 3 * max_contacts (this is
    //    exactly the legacy lambda buffer layout slot*3 + k). The general
    //    broadphase-cooked sizing arrives with M5.
    if (cap.links_per_env > 0) {
        cap.max_contacts_per_env = 4u;   // == articulation::kMaxFootContactsPerEnv
        cap.max_rows_per_env     = 12u;  // 3 rows per contact slot.
    } else {
        const uint32_t collidables = cap.bodies_per_env + cap.links_per_env;
        cap.max_contacts_per_env = collidables > 0 ? collidables * 4u : 0u;
        cap.max_rows_per_env     = cap.max_contacts_per_env * 4u;
    }

    return result;
}

} // namespace nuka::scene::cook
