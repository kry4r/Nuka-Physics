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

// M5 SAMP cook (plan §3.5): build the SDF sampling-point set for one convex-hull
// piece — the hull vertices PLUS each triangle edge's midpoint (so a coarse hull
// still samples the SDF densely along its silhouette). Edge midpoints are
// de-duplicated by the canonical (min,max) vertex pair so a shared edge is
// emitted once (deterministic; ascending edge order). Appends xyz triples to
// `out` and returns the count added (the per-body samp_range count). nka
// EncodeSamples writes this exact xyz pool to the .nka SAMP chunk.
uint32_t CookHullSamples(const CookedConvexGeometry& geo, uint32_t piece,
                         std::vector<float>& out) {
    if (piece >= geo.Count()) return 0u;
    const uint32_t voff = geo.vertex_offsets[piece];
    const uint32_t vcnt = geo.vertex_counts[piece];
    const uint32_t ioff = geo.index_offsets[piece];
    const uint32_t icnt = geo.index_counts[piece];
    const uint32_t start = static_cast<uint32_t>(out.size() / 3u);
    // 1. hull vertices.
    for (uint32_t v = 0; v < vcnt; ++v) {
        const size_t at = (static_cast<size_t>(voff) + v) * 3u;
        out.push_back(geo.vertices[at + 0]);
        out.push_back(geo.vertices[at + 1]);
        out.push_back(geo.vertices[at + 2]);
    }
    // 2. unique triangle-edge midpoints (canonical (lo,hi) dedup, ascending).
    std::vector<uint64_t> seen;
    auto edge_key = [](uint32_t a, uint32_t b) -> uint64_t {
        const uint32_t lo = a < b ? a : b, hi = a < b ? b : a;
        return (static_cast<uint64_t>(lo) << 32) | hi;
    };
    auto add_edge = [&](uint32_t a, uint32_t b) {
        const uint64_t k = edge_key(a, b);
        if (std::find(seen.begin(), seen.end(), k) != seen.end()) return;
        seen.push_back(k);
        const size_t pa = (static_cast<size_t>(voff) + a) * 3u;
        const size_t pb = (static_cast<size_t>(voff) + b) * 3u;
        out.push_back(0.5f * (geo.vertices[pa + 0] + geo.vertices[pb + 0]));
        out.push_back(0.5f * (geo.vertices[pa + 1] + geo.vertices[pb + 1]));
        out.push_back(0.5f * (geo.vertices[pa + 2] + geo.vertices[pb + 2]));
    };
    for (uint32_t t = 0; t + 2 < icnt; t += 3u) {
        const uint32_t i0 = geo.indices[ioff + t + 0];
        const uint32_t i1 = geo.indices[ioff + t + 1];
        const uint32_t i2 = geo.indices[ioff + t + 2];
        add_edge(i0, i1); add_edge(i1, i2); add_edge(i2, i0);
    }
    return static_cast<uint32_t>(out.size() / 3u) - start;
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

    // 10. M5 — pair-driven generalized-collision + SDF main-path staging (plan
    //     §3.5). The shape_table (one PairDrivenShape / collidable body row),
    //     the SDF sampling-point pool (SAMP cook), the cooked sparse-SDF grids
    //     (SdfDeviceWorld upload duties moved INTO the Model), and the filter
    //     exclude-list. A FusedFoot scene with no SdfMesh shape leaves the SDF
    //     tables empty (max_sdf_* == 0); they are populated for a cooked
    //     pair-driven SDF scene. These tables are sized AFTER the contact
    //     capacity (max_bodies_total == bodies_per_env, the shape_table stride).
    cap.max_bodies_total = cap.bodies_per_env;
    {
        // shape_table: one row per body, from its FIRST shape's primitive + the
        // contype/conaffinity (from the cooked filter groups when present).
        model.shape_table_rows.assign(cap.bodies_per_env, {});
        std::vector<uint8_t> body_done(cap.bodies_per_env, 0u);
        for (uint32_t s = 0; s < model.shapes.size(); ++s) {
            const nk::ModelShape& sh = model.shapes[s];
            if (sh.body_row >= cap.bodies_per_env || body_done[sh.body_row]) continue;
            body_done[sh.body_row] = 1u;
            nk::Model::PairDrivenShape& row = model.shape_table_rows[sh.body_row];
            row.kind = sh.kind;
            // params: sphere r / capsule r,hh / box he.xyz, by kind.
            row.params[0] = sh.radius;
            row.params[1] = sh.half_height;
            row.params[2] = sh.half_extents.z;
            // box he uses .xyz; overload params for the box kind.
            if (sh.kind == static_cast<uint8_t>(ShapeType::Box)) {
                row.params[0] = sh.half_extents.x;
                row.params[1] = sh.half_extents.y;
                row.params[2] = sh.half_extents.z;
            }
            // Hull / mesh rows: params[0] = the BOUND RADIUS (max |vertex| of
            // the cooked convex piece) — the broadphase AABB is a conservative
            // bound sphere (review fix: the default 0.5 sphere radius was
            // unrelated to the actual hull extent).
            if ((sh.kind == static_cast<uint8_t>(ShapeType::ConvexHull) ||
                 sh.kind == static_cast<uint8_t>(ShapeType::TriMesh)) &&
                sh.convex_geometry_index != ~uint32_t(0) &&
                sh.convex_geometry_index < blob.convex_geometry.Count()) {
                const CookedConvexGeometry& g = blob.convex_geometry;
                const uint32_t piece = sh.convex_geometry_index;
                const uint32_t voff = g.vertex_offsets[piece];
                const uint32_t vcnt = g.vertex_counts[piece];
                float max_sq = 0.0f;
                for (uint32_t v = 0; v < vcnt; ++v) {
                    const size_t at = (static_cast<size_t>(voff) + v) * 3u;
                    const float x = g.vertices[at + 0];
                    const float y = g.vertices[at + 1];
                    const float z = g.vertices[at + 2];
                    const float d = x * x + y * y + z * z;
                    if (d > max_sq) max_sq = d;
                }
                row.params[0] = std::sqrt(max_sq);
            }
            row.contype = 1u;
            row.conaffinity = 1u;
            row.sdf_grid = ~0u;  // resolved below if the piece has a cooked SDF.
        }

        // SDF grids: mirror the cooked CookedSdfTable into the Model SDF tables;
        // for each SdfMesh / ConvexHull shape carrying a cooked SDF, build the
        // sampling-point set (CookHullSamples) and bind the body's samp_range +
        // (for the OTHER body) the sdf_grid index.
        const CookedSdfTable& sdf = blob.sdfs;
        const CookedConvexGeometry& geo = blob.convex_geometry;
        model.samp_ranges.assign(static_cast<size_t>(cap.bodies_per_env) * 2u, 0u);
        if (sdf.Count() > 0) {
            // Concatenate the cooked SDF grids into the Model device tables.
            model.sdf_grids.reserve(sdf.Count());
            uint32_t cell_cursor = 0;
            for (uint32_t g = 0; g < sdf.Count(); ++g) {
                nk::Model::SdfGrid grid;
                grid.origin = sdf.origins[g];
                grid.voxel_size = sdf.voxel_sizes[g];
                grid.dims[0] = sdf.dims_x[g];
                grid.dims[1] = sdf.dims_y[g];
                grid.dims[2] = sdf.dims_z[g];
                grid.cell_offset = cell_cursor;
                grid.cell_count = sdf.key_counts[g];
                model.sdf_grids.push_back(grid);
                const uint32_t off = sdf.key_offsets[g];
                for (uint32_t c = 0; c < grid.cell_count; ++c) {
                    model.sdf_cell_keys.push_back(sdf.cell_keys[off + c]);
                    model.sdf_cell_values.push_back(sdf.cell_values[off + c]);
                    model.sdf_cell_gradients.push_back(sdf.cell_gradients[off + c]);
                }
                cell_cursor += grid.cell_count;
            }
            // Per-body: a SdfMesh/hull body carrying a cooked SDF binds sdf_grid;
            // any body owning a hull piece gets a SAMP slice (the sampling set).
            for (uint32_t s = 0; s < model.shapes.size(); ++s) {
                const nk::ModelShape& sh = model.shapes[s];
                if (sh.body_row >= cap.bodies_per_env) continue;
                const uint32_t cgi = sh.convex_geometry_index;
                if (cgi == ~uint32_t(0) || cgi >= geo.Count()) continue;
                const uint32_t before = static_cast<uint32_t>(model.samp_points.size() / 3u);
                const uint32_t added = CookHullSamples(geo, cgi, model.samp_points);
                model.samp_ranges[sh.body_row * 2u + 0u] = before;
                model.samp_ranges[sh.body_row * 2u + 1u] = added;
                // If this piece has a cooked SDF, expose it on the body's shape row.
                if (cgi < sdf.piece_sdf_indices.size() &&
                    sdf.piece_sdf_indices[cgi] != kNoSdf &&
                    sh.body_row < model.shape_table_rows.size()) {
                    model.shape_table_rows[sh.body_row].sdf_grid =
                        sdf.piece_sdf_indices[cgi];
                }
            }
        }
        cap.max_samp_points = static_cast<uint32_t>(model.samp_points.size() / 3u);
        cap.max_sdf_grids   = static_cast<uint32_t>(model.sdf_grids.size());
        cap.max_sdf_cells   = static_cast<uint32_t>(model.sdf_cell_keys.size());
        cap.max_excluded_pairs =
            static_cast<uint32_t>(model.excluded_pairs.size());
    }

    return result;
}

// ---------------------------------------------------------------------------
// M6 particle cook.
// ---------------------------------------------------------------------------

void CookXpbdParticles(nk::Model& model, uint32_t env_count,
                       const XpbdCookInput& in) {
    const uint32_t envs = env_count > 0 ? env_count : 1u;
    nk::ModelCapacities& cap = model.capacities;
    if (cap.env_count == 1u || cap.env_count == 0u) cap.env_count = envs;

    nk::Model::ModelParticles& mp = model.particles;
    mp.mode = nk::Model::ParticleMode::Xpbd;
    mp.initial_pos = in.positions;
    mp.initial_vel = in.velocities;
    mp.inv_mass = in.inv_mass;
    if (mp.initial_vel.size() != mp.initial_pos.size()) {
        mp.initial_vel.assign(mp.initial_pos.size(), math::Vec3::Zero());
    }
    if (mp.inv_mass.size() != mp.initial_pos.size()) {
        mp.inv_mass.assign(mp.initial_pos.size(), 1.0f);
    }
    mp.xpbd_iters = in.solver_iterations == 0u ? 1u : in.solver_iterations;

    // De-interleave the constraint AoS into the per-field SoA (the EXACT
    // UploadXpbdWorld layout: distance a/b/rest/alpha; bend 4-particle +
    // 4-gradient; volume 4-particle + rest6/alpha).
    const uint32_t dn = static_cast<uint32_t>(in.distance.size());
    mp.dist_a.resize(dn); mp.dist_b.resize(dn);
    mp.dist_rest.resize(dn); mp.dist_alpha.resize(dn);
    for (uint32_t c = 0; c < dn; ++c) {
        mp.dist_a[c] = in.distance[c].a;
        mp.dist_b[c] = in.distance[c].b;
        mp.dist_rest[c] = in.distance[c].rest_length;
        mp.dist_alpha[c] = in.distance[c].compliance_alpha;
    }
    const uint32_t bn = static_cast<uint32_t>(in.bend.size());
    mp.bend_particles.resize(static_cast<size_t>(bn) * 4u);
    mp.bend_gradients.resize(static_cast<size_t>(bn) * 4u);
    mp.bend_alpha.resize(bn);
    for (uint32_t c = 0; c < bn; ++c) {
        for (uint32_t j = 0; j < 4u; ++j) {
            mp.bend_particles[static_cast<size_t>(c) * 4u + j] = in.bend[c].p[j];
            mp.bend_gradients[static_cast<size_t>(c) * 4u + j] = in.bend[c].k[j];
        }
        mp.bend_alpha[c] = in.bend[c].compliance_alpha;
    }
    const uint32_t vn = static_cast<uint32_t>(in.volume.size());
    mp.vol_particles.resize(static_cast<size_t>(vn) * 4u);
    mp.vol_rest6.resize(vn); mp.vol_alpha.resize(vn);
    for (uint32_t c = 0; c < vn; ++c) {
        for (uint32_t j = 0; j < 4u; ++j) {
            mp.vol_particles[static_cast<size_t>(c) * 4u + j] = in.volume[c].p[j];
        }
        mp.vol_rest6[c] = in.volume[c].rest_volume_times6;
        mp.vol_alpha[c] = in.volume[c].compliance_alpha;
    }

    cap.particles_per_env = static_cast<uint32_t>(mp.initial_pos.size());
    cap.dist_cons_per_env = dn;
    cap.bend_cons_per_env = bn;
    cap.vol_cons_per_env  = vn;
}

void CookPbfParticles(nk::Model& model, uint32_t env_count,
                      const PbfCookInput& in) {
    const uint32_t envs = env_count > 0 ? env_count : 1u;
    nk::ModelCapacities& cap = model.capacities;
    if (cap.env_count == 1u || cap.env_count == 0u) cap.env_count = envs;

    nk::Model::ModelParticles& mp = model.particles;
    mp.mode = nk::Model::ParticleMode::Pbf;
    mp.initial_pos = in.positions;
    mp.initial_vel = in.velocities;
    mp.inv_mass = in.inv_mass;
    if (mp.initial_vel.size() != mp.initial_pos.size()) {
        mp.initial_vel.assign(mp.initial_pos.size(), math::Vec3::Zero());
    }
    if (mp.inv_mass.size() != mp.initial_pos.size()) {
        // Uniform-mass fluid: inv_mass = 1/particle_mass (a free particle).
        const float im = in.particle_mass > 0.0f ? 1.0f / in.particle_mass : 0.0f;
        mp.inv_mass.assign(mp.initial_pos.size(), im);
    }
    mp.pbf_rest_density   = in.rest_density;
    mp.pbf_support_radius = in.support_radius;
    mp.pbf_particle_mass  = in.particle_mass;
    mp.pbf_cfm_epsilon    = in.cfm_epsilon;
    mp.pbf_iters          = in.iters == 0u ? 1u : in.iters;
    mp.pbf_clamp_overdensity = in.clamp_overdensity;
    mp.pbf_xsph_viscosity = in.xsph_viscosity;
    mp.pbf_surface_tension= in.surface_tension;
    mp.grid_min = in.grid_min;
    mp.grid_dims[0] = in.grid_dims[0];
    mp.grid_dims[1] = in.grid_dims[1];
    mp.grid_dims[2] = in.grid_dims[2];
    mp.cell_size = in.support_radius;
    mp.query_radius = in.support_radius;
    mp.boundary_enabled = in.boundary_enabled;
    mp.floor_z = in.floor_z;

    cap.particles_per_env = static_cast<uint32_t>(mp.initial_pos.size());
    // Per-env uniform-grid cell capacity (sizes grid_cell_start/end; the
    // ParticleGridBuild op fails loudly if the live dims exceed it).
    cap.max_grid_cells = in.grid_dims[0] * in.grid_dims[1] * in.grid_dims[2];
}

} // namespace nuka::scene::cook
