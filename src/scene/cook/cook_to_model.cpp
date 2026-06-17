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

#include "collision/shape_kind.hpp"  // nuka::collision::ShapeKind (R2: one enum)
#include "sensor/terrain/terrain_field.hpp"  // SampleTerrainHeight (H2 generator)
#include "nk/solve/nk_row.hpp"       // kPairDrivenRowsPerSlot (B1 row budget)
#include "runtime/articulation/articulation_contacts.hpp"  // contact-slot strides
#include "runtime/articulation/articulation_cooker.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "scene/cooker.hpp"
#include "scene/ecs/components.hpp"
#include "scene/graph/scene_graph.hpp"

namespace nuka::scene::cook {

namespace {

// M7 T5b — resolve the AUTHORED initial-condition for the cooked articulation
// (the settle product, controller ruling R4: BAKED). Two sources, in priority:
//   (1) the InitialStateComponent set on the articulation root entity by
//       ApplySettleToRegistry (the ECS/settle writeback path), and
//   (2) the SceneIR's `initial_state` map keyed by a tree node PATH (the .nks
//       persisted IC): match the articulation root's node path, then any ancestor
//       PREFIX of it (the import attach prefix, e.g. "h1"), and finally — when the
//       map carries exactly one entry whose qpos length equals the cooked link
//       count — that sole entry (a single-articulation scene).
// `qpos` is per-LINK (length == link_count) and `root` is the settled base pose,
// matching model.articulation.initial_q / base_pose. Returns false (a strict
// no-op: the cook-rest pose is kept) when no IC of the right length is authored —
// the case for EVERY existing golden/scene that does not author an IC.
bool ResolveAuthoredIC(const SceneIR& scene, uint32_t root_body, uint32_t link_count,
                       std::vector<float>& out_qpos, math::Transform& out_root) {
    const EntityId root_ent = root_body != ~uint32_t(0)
                                  ? scene.EntityOfBody(root_body)
                                  : kInvalidEntity;

    // (1) InitialStateComponent on the root entity (the settle writeback).
    if (root_ent != kInvalidEntity) {
        if (const InitialStateComponent* ic =
                scene.Ecs().Get<InitialStateComponent>(root_ent)) {
            if (ic->qpos.size() == link_count) {
                out_qpos = ic->qpos;
                out_root = ic->root;
                return true;
            }
        }
    }

    // (2) the SceneIR's `initial_state` map (the persisted .nks IC).
    const SceneInitialState& map = scene.InitialState();
    if (map.empty()) return false;

    // Derive the root link's node path (so we can match by exact path or prefix).
    std::string root_path;
    if (root_ent != kInvalidEntity) {
        if (const auto node = scene.Ecs().NodeOf(root_ent)) {
            root_path = scene.Tree().PathOf(node);
        }
    }
    // Exact-path match, then ancestor-prefix match (e.g. "h1" for "h1/pelvis").
    if (!root_path.empty()) {
        if (const auto it = map.find(root_path); it != map.end() &&
                                                 it->second.qpos.size() == link_count) {
            out_qpos = it->second.qpos;
            out_root = it->second.root;
            return true;
        }
        for (const auto& [key, art_ic] : map) {
            if (art_ic.qpos.size() != link_count) continue;
            // `key` is an ancestor prefix of root_path iff root_path starts with
            // key followed by the path separator '/'.
            if (root_path.size() > key.size() &&
                root_path.compare(0, key.size(), key) == 0 &&
                root_path[key.size()] == '/') {
                out_qpos = art_ic.qpos;
                out_root = art_ic.root;
                return true;
            }
        }
    }
    // Sole-entry fallback (a single-articulation scene): the only IC whose qpos
    // length matches the cooked link count.
    if (map.size() == 1u && map.begin()->second.qpos.size() == link_count) {
        out_qpos = map.begin()->second.qpos;
        out_root = map.begin()->second.root;
        return true;
    }
    return false;
}

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

namespace {

// L-RECON-B: the shared cook implementation. The cook options gate the two HEAVY
// per-shape stages (V-HACD vs single-hull, SDF bake). The public 2-arg overload
// passes the DEFAULT options (legacy, byte-identical); the PairDriven 3-arg passes
// the general-path options (single-hull + no SDF) so the whole-body union scene
// cooks tractably. No entity/scene-name branch — the options are the only seam.
CookToModelResult CookToModelImpl(const SceneIR& scene, int env_count,
                                  const CookSceneOptions& cook_options) {
    const uint32_t envs = env_count > 0 ? static_cast<uint32_t>(env_count) : 1u;

    // 1. Drive the existing cook (the heavy lifting: V-HACD / SDF / filters).
    const CookedBlob blob = CookScene(scene, cook_options);

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

    // M7 T5b — which body rows are owned by the articulation (driven by FK, NOT
    // free rigid bodies). The movable-body body_init pass below MUST keep these
    // rows at inv_mass 0 so the rigid-body gravity-kick / IntegrateBodyPosition
    // arms (which run over bodies_per_env, gated on body_inv_mass > 0) stay
    // no-ops for them — exactly today's behavior (an articulation-only world has
    // an EMPTY body_init, so SeedInitialState skips the body block entirely).
    std::vector<uint8_t> body_is_articulation_link(blob.body_count, 0u);

    // 4. Articulation template(s). M3b transcribed the SINGLE-ENV
    //    BuildArticulationHostState product 1:1; the WP1 multi-articulation
    //    foundation transcribes the FULL set of co-resident topologies (K Go2 in
    //    ONE env). BuildArticulationHostState(arts, ...) ALREADY loops over every
    //    topology, building the flat per-link arrays + link_to_articulation[] +
    //    articulation_link_count/offset[] with a running global offset, so the
    //    only change here is to pass the WHOLE `arts` vector (not just the front)
    //    and carry the per-articulation bookkeeping into nk::Model. At
    //    articulations_per_env == 1 (the legacy single-robot scene) the host state
    //    is identical to the BuildArticulationHostState({arts.front()}) product,
    //    so the staged Model bytes are byte-for-byte unchanged (the K==1 D1
    //    invariant). dofs_per_env stays the MAX single-dog DOF (NEVER summed): the
    //    CRBA M-tile is per-articulation (one max_dof^2 block per dog).
    if (!arts.empty()) {
        namespace articulation = runtime::articulation;
        const articulation::ArticulationHostState host =
            articulation::BuildArticulationHostState(arts, blob.bodies);

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
        // WP1 multi-articulation co-residence bookkeeping (host already built it).
        m.link_to_articulation = host.link_to_articulation;
        m.articulation_link_count = host.articulation_link_count;
        m.articulation_link_offset = host.articulation_link_offset;
        m.base_poses = host.base_pose;  // K root world poses (one per co-resident dog).
        m.articulation_count =
            static_cast<uint32_t>(host.articulation_link_count.size());

        // Mark every body row owned by an articulation link (so the movable
        // body_init pass skips them — they are integrated by FK, not as free
        // rigid bodies).
        for (uint32_t lb : host.link_body) {
            if (lb < body_is_articulation_link.size()) {
                body_is_articulation_link[lb] = 1u;
            }
        }

        // R3 (general contact pipeline Phase 0): the shape->body INVERSE tables.
        // link_body is link->body; the general PairDriven row-emission (S5) needs
        // body->owner. For each TEMPLATE link l with owning body row b, record
        // body_to_link[b] = l and body_to_articulation[b] = the LOCAL articulation
        // of l. A body row owned by NO link (free rigid / static) keeps ~0u (so
        // its side resolves free-rigid; the shape_table body_id == -1 resolves
        // static). Both stage template-local + tile env-major. INERT in Phase 0.
        model.body_to_link.assign(cap.bodies_per_env, ~uint32_t(0));
        model.body_to_articulation.assign(cap.bodies_per_env, ~uint32_t(0));
        for (uint32_t l = 0; l < host.link_body.size(); ++l) {
            const uint32_t b = host.link_body[l];
            if (b >= cap.bodies_per_env) continue;
            model.body_to_link[b] = l;
            model.body_to_articulation[b] =
                l < host.link_to_articulation.size() ? host.link_to_articulation[l]
                                                     : 0u;
        }

        // M7 T5b — seed the AUTHORED settled IC (qpos -> initial_q per-link, root
        // -> base pose), OVERRIDING the cook-rest pose just set above. A strict
        // no-op when no IC is authored (ResolveAuthoredIC returns false): every
        // existing golden/scene keeps the cook-rest pose byte-for-byte. The root
        // link's owning body row is host.link_body[m.root_link] (root_link == 0).
        // `has_authored_ic` is reused below so the PD hold drive targets pin the
        // SETTLED stance (not the cook-rest q) when an IC is authored.
        bool has_authored_ic = false;
        {
            const uint32_t root_body =
                m.root_link < host.link_body.size() ? host.link_body[m.root_link]
                                                    : ~uint32_t(0);
            std::vector<float> ic_qpos;
            math::Transform ic_root = math::Transform::Identity();
            if (ResolveAuthoredIC(scene, root_body, m.link_count, ic_qpos, ic_root)) {
                m.initial_q = ic_qpos;
                m.base_pose = ic_root;
                has_authored_ic = true;
            }
        }
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
        //
        // WP1 multi-articulation: dofs_per_env is the MAX single-DOG DOF (the
        // per-articulation CRBA M-tile is max_dof^2, ONE block per co-resident dog
        // -- NEVER the SUM, which would (a) be a G0-dishonest monolithic DOF and
        // (b) blow the per-artic 64-DOF cap at K>=4). Compute each articulation's
        // own DOF over its link slice [offset, offset+count) and keep the max.
        // For the single-articulation scene this is exactly the prior whole-host
        // sum (one articulation == the whole host), so dofs_per_env is unchanged.
        auto joint_dof = [](articulation::ArticulationJointType jt) -> uint32_t {
            switch (jt) {
                case articulation::ArticulationJointType::Fixed: return 0u;
                case articulation::ArticulationJointType::FloatingBase: return 6u;
                default: return 1u;
            }
        };
        uint32_t max_dof = 0u;
        for (uint32_t ai = 0; ai < m.articulation_count; ++ai) {
            const uint32_t off = ai < host.articulation_link_offset.size()
                                     ? host.articulation_link_offset[ai] : 0u;
            const uint32_t cnt = ai < host.articulation_link_count.size()
                                     ? host.articulation_link_count[ai] : 0u;
            uint32_t dofs = 0u;
            for (uint32_t l = off; l < off + cnt && l < host.joint_type.size(); ++l) {
                dofs += joint_dof(host.joint_type[l]);
            }
            if (dofs > max_dof) max_dof = dofs;
        }
        m.dof_count = max_dof;          // PER-ARTICULATION (max single-dog DOF).
        cap.dofs_per_env = m.dof_count;
        cap.links_per_env = m.link_count;        // SUM across co-resident dogs.
        cap.articulations_per_env = m.articulation_count;  // K (1 for legacy scenes).

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

        // Per-link collision geometry: per template-link, record the FIRST
        // primitive (Sphere/Box/Capsule) collision shape whose owning body maps to
        // that link, in the link's LOCAL frame. This is the source the GENERAL
        // contact path poses into world space (SyncLinkBodyPose: link_pose o
        // link_geom_local -> the link's collidable body row in the LBVH). Cooked
        // for EVERY scene; the FusedFoot/UnionCsr graphs never read it (it only
        // feeds the PairDriven SyncLinkBodyPose op) -> K==1 FUSED byte-identical.
        //   kind sentinel: 0 == none; a primitive stores (ShapeType + 1) so the
        //   default-zero (no-shape) link is unambiguously inactive.
        m.link_geom_kind.assign(m.link_count, 0u);
        m.link_geom_params.assign(static_cast<size_t>(m.link_count) * 4u, 0.0f);
        m.link_geom_local.assign(m.link_count, math::Transform::Identity());
        for (uint32_t shape = 0; shape < blob.shapes.types.size(); ++shape) {
            const ShapeType st = blob.shapes.types[shape];
            if (st != ShapeType::Sphere && st != ShapeType::Box &&
                st != ShapeType::Capsule) {
                continue;  // only bounded primitives are collidable here.
            }
            const BodyId body = shape < blob.shapes.body_ids.size()
                                    ? blob.shapes.body_ids[shape]
                                    : kInvalidBody;
            uint32_t owner_link = ~uint32_t(0);
            for (uint32_t link = 0; link < m.link_count; ++link) {
                if (host.link_body[link] == body) { owner_link = link; break; }
            }
            if (owner_link == ~uint32_t(0)) {
                continue;  // body is not an articulation link.
            }
            if (m.link_geom_kind[owner_link] != 0u) {
                continue;  // keep the FIRST primitive for this link (deterministic).
            }
            m.link_geom_kind[owner_link] = static_cast<uint32_t>(st) + 1u;
            const float radius = shape < blob.shapes.radii.size()
                                     ? blob.shapes.radii[shape] : 0.0f;
            const float half_height = shape < blob.shapes.half_heights.size()
                                          ? blob.shapes.half_heights[shape] : 0.0f;
            const math::Vec3 he = shape < blob.shapes.half_extents.size()
                                      ? blob.shapes.half_extents[shape]
                                      : math::Vec3::Zero();
            const size_t b = static_cast<size_t>(owner_link) * 4u;
            if (st == ShapeType::Sphere) {
                m.link_geom_params[b + 0] = radius;
            } else if (st == ShapeType::Capsule) {
                m.link_geom_params[b + 0] = radius;
                m.link_geom_params[b + 1] = half_height;
            } else {  // Box
                m.link_geom_params[b + 0] = he.x;
                m.link_geom_params[b + 1] = he.y;
                m.link_geom_params[b + 2] = he.z;
            }
            m.link_geom_local[owner_link] =
                shape < blob.shapes.local_transforms.size()
                    ? blob.shapes.local_transforms[shape]
                    : math::Transform::Identity();
        }

        // Hold drives (legacy BuildHoldDriveTargets 1:1): targets = cooked q;
        // Position actuators seed stiffness = gain, damping = 2*sqrt(gain),
        // force limits from the actuator table.
        nk::ModelHoldDrives& d = model.hold_drives;
        // Hold the SETTLED stance (m.initial_q) when an IC was authored, else the
        // cook-rest q (== host.q): m.initial_q is host.q verbatim when no IC,
        // keeping this byte-identical for every existing scene.
        d.targets = has_authored_ic ? m.initial_q : host.q;
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

    // 7b. M7 T5b — movable rigid-body body_init (pose + inv_mass + inv_inertia),
    //     so a generic .nks-cooked MOVABLE cup falls/rests under gravity instead
    //     of being frozen at the arena's zero-init Identity (inv_mass 0). Matches
    //     BuildNkUnionModel's cup body_init fill (pose/inv_mass/inv_inertia from
    //     the body record) so the generic path agrees with the union path.
    //
    //     GUARD (the no-regression contract): body_init stays EMPTY unless the
    //     scene has >=1 genuinely-MOVABLE FREE rigid body (a body row that is not
    //     an articulation link, not static, with mass > 0 / inv_mass > 0). For
    //     every existing golden/scene (articulation-only feet, particle, the
    //     coupled_grasp_soft static box wall) no such body exists -> body_init
    //     stays empty -> SeedInitialState skips the body block -> BYTE-IDENTICAL.
    //
    //     When a movable free body IS present, body_init is sized to ALL body
    //     rows (env-major layout: SeedInitialState indexes body_init[b]); every
    //     row gets its cooked world pose, but STATIC and ARTICULATION-LINK rows
    //     keep inv_mass / inv_inertia 0 (the body integrate arms remain no-ops
    //     for them, so the articulation is untouched). Only the free movable
    //     bodies carry a non-zero inv_mass and so respond to gravity + contacts.
    {
        bool any_movable_free = false;
        for (uint32_t b = 0; b < blob.body_count; ++b) {
            const bool is_link = b < body_is_articulation_link.size() &&
                                 body_is_articulation_link[b] != 0u;
            const bool is_static = b < blob.bodies.is_static.size() &&
                                   blob.bodies.is_static[b] != 0u;
            const float inv_mass = b < blob.bodies.inv_masses.size()
                                       ? blob.bodies.inv_masses[b]
                                       : 0.0f;
            if (!is_link && !is_static && inv_mass > 0.0f) {
                any_movable_free = true;
                break;
            }
        }
        if (any_movable_free) {
            model.body_init.assign(blob.body_count, nk::Model::BodyInit{});
            for (uint32_t b = 0; b < blob.body_count; ++b) {
                nk::Model::BodyInit& bi = model.body_init[b];
                // Pose: the body's cooked WORLD pose (ResolveWorldTransform). For
                // a free movable body this is its local->world seat; the union
                // path uses the same record-derived pose.
                if (b < blob.bodies.poses.size()) bi.pose = blob.bodies.poses[b];
                // Velocities start at rest (the cook seeds no initial body vel).
                bi.linear_velocity = math::Vec3::Zero();
                bi.angular_velocity = math::Vec3::Zero();
                // inv_mass / inv_inertia: ONLY for genuinely-movable free bodies;
                // static + articulation-link rows stay frozen (inv_mass 0).
                const bool is_link = b < body_is_articulation_link.size() &&
                                     body_is_articulation_link[b] != 0u;
                const bool is_static = b < blob.bodies.is_static.size() &&
                                       blob.bodies.is_static[b] != 0u;
                if (!is_link && !is_static) {
                    if (b < blob.bodies.inv_masses.size())
                        bi.inv_mass = blob.bodies.inv_masses[b];
                    if (b < blob.bodies.inv_inertias.size())
                        bi.inv_inertia = blob.bodies.inv_inertias[b];
                }
            }
        }
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
    //
    // WP1 NOTE (multi-articulation foot contact is the LATER contact crux, NOT
    // this foundation task): kMaxFootContactsPerEnv == 4 is a COMPILE-TIME constant
    // and the FUSED foot-detection kernel keys slots by ENV (base_slot = env*4),
    // NOT by articulation. With K co-resident dogs (4 feet each) one env's K*4 feet
    // would overflow the 4 env slots / mis-slot across dogs. Growing this to K*4
    // here WITHOUT re-keying contacts_foot.cu's slot math (env -> articulation) and
    // the `rows` field stride would be incorrect AND would shift the K==1 `rows`
    // byte layout (D1 break). So the cap is left at 4 (byte-identical at K==1); the
    // multi-dog foot/leg contact pipeline (K*4 slots + env->artic slot re-key +
    // capsule-capsule narrowphase + two-articulation reaction scatter) is WP5-8,
    // the genuinely-new contact crux. The WP1 co-step spike free-falls the dogs so
    // no ground contact is exercised -- it validates the multi-artic FORWARD
    // dynamics co-residence, which is what this foundation delivers.
    if (cap.links_per_env > 0) {
        // WP5-8: with K co-resident articulations (dogs), the FUSED solver is
        // block-per-articulation and reads slot block [artic*kMaxFootContactsPerEnv,
        // +stride); articulation index runs [0, K*env_count). So the per-env
        // contact-slot budget must cover K articulation blocks: K * 4 slots, with
        // 3 rows/slot. At K==1 (the legacy single-robot scene) this is EXACTLY
        // 4 / 12 -- byte-identical (the K==1 D1 invariant; a multi-dog scene is a
        // DISTINCT validation surface). This grows the fused-foot slot stream to
        // hold every dog's per-articulation foot-contact block (general multi-body
        // collision rides the PairDriven path's own candidate-slot budget instead).
        const uint32_t k = cap.articulations_per_env == 0u ? 1u
                                                           : cap.articulations_per_env;
        // Per-dog FUSED-foot slot stride. At K==1 (the legacy single-robot scene)
        // this is EXACTLY kMaxFootContactsPerEnv (4) -> max_contacts 4 / rows 12,
        // byte-identical. At K>1 it grows to kMultiDogContactsPerArtic (12) so each
        // dog's per-articulation foot block has headroom. 3 rows per contact slot
        // {normal, t1, t2}. (PairDriven overrides this budget with its own
        // candidate-slot layout; this stride only governs the FusedFoot path.)
        const uint32_t stride = (k > 1u)
            ? ::nuka::runtime::articulation::kMultiDogContactsPerArtic
            : ::nuka::runtime::articulation::kMaxFootContactsPerEnv;
        cap.max_contacts_per_env = stride * k;     // K * per-dog stride
        cap.max_rows_per_env     = 3u * stride * k;  // 3 rows per contact slot.
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
            // unrelated to the actual hull extent). L-RECON-D: ALSO pack this
            // piece's MESH-LOCAL verts into the concatenated model.hull_verts pool
            // and record the row's (hull_vert_offset, hull_vert_count) slice so the
            // cvx narrowphase collides THIS shape's hull (not one global hull).
            if ((sh.kind == static_cast<uint8_t>(ShapeType::ConvexHull) ||
                 sh.kind == static_cast<uint8_t>(ShapeType::TriMesh)) &&
                sh.convex_geometry_index != ~uint32_t(0) &&
                sh.convex_geometry_index < blob.convex_geometry.Count()) {
                const CookedConvexGeometry& g = blob.convex_geometry;
                const uint32_t piece = sh.convex_geometry_index;
                const uint32_t voff = g.vertex_offsets[piece];
                const uint32_t vcnt = g.vertex_counts[piece];
                const uint32_t hull_base =
                    static_cast<uint32_t>(model.hull_verts.size() / 3u);
                float max_sq = 0.0f;
                for (uint32_t v = 0; v < vcnt; ++v) {
                    const size_t at = (static_cast<size_t>(voff) + v) * 3u;
                    const float x = g.vertices[at + 0];
                    const float y = g.vertices[at + 1];
                    const float z = g.vertices[at + 2];
                    const float d = x * x + y * y + z * z;
                    if (d > max_sq) max_sq = d;
                    model.hull_verts.push_back(x);
                    model.hull_verts.push_back(y);
                    model.hull_verts.push_back(z);
                }
                row.params[0] = std::sqrt(max_sq);
                row.hull_vert_offset = hull_base;
                row.hull_vert_count = vcnt;
            }
            row.contype = 1u;
            row.conaffinity = 1u;
            row.sdf_grid = ~0u;  // resolved below if the piece has a cooked SDF.
            // R1 (general contact pipeline Phase 0): the shape->body indirection.
            // A cooked collidable row maps to its OWNING body row (never static
            // here — static collidables are emitted separately, R5). group 0 ==
            // the default collide-all group. INERT in Phase 0 (no routing reads
            // these lanes; R4/S5 wire them in Phase 1).
            row.body_id = static_cast<int32_t>(sh.body_row);
            row.group = 0u;
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
        // L-RECON-D: the convex-hull vertex pool capacity = the concatenated
        // per-shape hull verts packed above (xyz triples). 0 for a hull-free
        // scene (go2) -> the hull_verts segment stays zero bytes (byte-inert).
        cap.max_hull_verts =
            static_cast<uint32_t>(model.hull_verts.size() / 3u);

        // R5 (general contact pipeline Phase 0): emit a STATIC ground collidable
        // row into shape_table as a first-class collidable with body_id == -1 (no
        // reaction side). Appended AFTER the per-body rows so the per-body rows
        // [0, bodies_per_env) keep their exact indices + bytes; the static row
        // lives at index bodies_per_env. contype/conaffinity == 1 (collide-all).
        // INERT in Phase 0: the broadphase iterates only bodies_per_env body rows
        // per env (it never indexes the static row), and shape_table is a model
        // param pinned by no golden, so growing max_bodies_total by one row is
        // byte-safe for the gated FusedFoot/UnionCsr families. The static row's
        // WORLD pose / LBVH entry is wired in Phase 1 (R5 downstream + B3); here we
        // only register the collidable. A Plane kind anchors the flat-ground case
        // (the general heightfield collidable, kShapeHeightfield, is added by H2/H3
        // in Phase 2). params[0] carries the ground plane height for the future
        // static-pose stamp.
        {
            nk::Model::PairDrivenShape ground;
            ground.kind = ::nuka::collision::kShapePlane;
            ground.params[0] = model.ground_height;
            ground.params[1] = 0.0f;
            ground.params[2] = 0.0f;
            ground.params[3] = 0.0f;
            ground.contype = 1u;
            ground.conaffinity = 1u;
            ground.sdf_grid = ~0u;
            ground.body_id = -1;   // STATIC: no owning body, no reaction side.
            ground.group = 0u;
            model.shape_table_rows.push_back(ground);
        }
        // Grow the shape_table capacity to cover the appended static row(s). Only
        // shape_table (count max_bodies_total*10) + samp_ranges (count
        // max_bodies_total*2) scale with this; both are model params pinned by no
        // golden, and the extra samp_ranges entries default to 0 (no samples).
        cap.max_bodies_total =
            static_cast<uint32_t>(model.shape_table_rows.size());
    }

    return result;
}

}  // namespace

CookToModelResult CookToModel(const SceneIR& scene, int env_count) {
    // DEFAULT (legacy) cook options -> byte-identical to the pre-L-RECON-B cook.
    return CookToModelImpl(scene, env_count, CookSceneOptions{});
}

// B1 (general contact pipeline Phase 1B): the PairDriven cook overload. It runs
// the SAME cook (so the registry / shape_table / body_to_link / static ground row
// / multi-artic foundation are all reused verbatim) then flips the model to the
// GENERAL family: contact_family = PairDriven (the pipeline then routes the ONE
// LBVH -> cvx -> mixed-island contact path), enables cross-env
// filtering so candidate pairs stay env-local, and RE-SIZES the per-env row budget
// to the general per-candidate-slot layout (max_contacts_per_env candidate slots x
// kPairDrivenRowsPerSlot rows/slot). The FusedFoot overload is byte-untouched.
CookToModelResult CookToModel(const SceneIR& scene, int env_count,
                              const CookToModelOptions& options) {
    // L-RECON-B: the GENERAL (PairDriven) cvx narrowphase consumes NEITHER the
    // sparse SDF (OpNarrowphaseSdf is a no-op for every family) NOR V-HACD's
    // N-piece decomposition (it wants ONE convex hull per mesh). Skipping both
    // makes the whole-body H1 union scene cook tractable (>178s -> seconds), and
    // is a no-op for primitive-only scenes (go2: no mesh, no SDF). The FusedFoot
    // overload keeps the DEFAULT cook options (byte-identical to the 2-arg cook).
    CookSceneOptions cook_options;
    if (options.contact_family == CookContactFamily::PairDriven) {
        cook_options.bake_sdf = false;
        cook_options.general_single_hull = true;
    }
    CookToModelResult result = CookToModelImpl(scene, env_count, cook_options);
    if (options.contact_family != CookContactFamily::PairDriven) {
        return result;  // FusedFoot: DEFAULT cook options == byte-identical 2-arg.
    }
    nk::Model& model = result.model;
    model.contact_family = nk::ContactFamily::PairDriven;
    // Env-local candidate pairs: keep cross-env filtering ON so co-resident
    // collidables of ONE env collide (multi-dog-in-one-env / box-on-dog), not
    // across envs (the batch replicas stay independent).
    model.filter_cross_env = true;

    // Build the per-DOG flat-DOF -> (template-local link, base component) maps
    // (DofIndexOf^-1) the general island solver's qdot pack/scatter needs. The
    // 2-arg cook does NOT cook these (only union_nk_model did, for the H1 path), so
    // for a PairDriven articulated world they were EMPTY -> the floating-base
    // reaction never reached link_velocity. The map is PER-DOG (the first
    // articulation's link slice [0, links_per_dog)); the multi-tile solver applies
    // it per co-resident articulation with that artic's link offset. Mirrors
    // union_nk_model.cpp's DofIndexOf^-1 (FloatingBase root -> 6 components, scalar
    // joint -> component ~0u). Only the FIRST dog's links since every dog is the
    // same template (dofs_per_env is the per-dog DOF).
    {
        namespace articulation = runtime::articulation;
        const nk::ModelArticulation& m = model.articulation;
        const uint32_t k = model.capacities.articulations_per_env == 0u
                               ? 1u : model.capacities.articulations_per_env;
        const uint32_t links_per_dog =
            (k > 0u && m.link_count >= k) ? (m.link_count / k) : m.link_count;
        model.dof_to_link.clear();
        model.dof_to_component.clear();
        for (uint32_t l = 0; l < links_per_dog && l < m.joint_type.size(); ++l) {
            const articulation::ArticulationJointType jt =
                static_cast<articulation::ArticulationJointType>(m.joint_type[l]);
            const bool is_root_floating =
                (l == 0u) && (l < m.parent_link.size()) &&
                (m.parent_link[l] == ~uint32_t(0)) &&
                (jt == articulation::ArticulationJointType::FloatingBase);
            if (is_root_floating) {
                for (uint32_t b = 0; b < 6u; ++b) {
                    model.dof_to_link.push_back(l);
                    model.dof_to_component.push_back(b);
                }
                continue;
            }
            switch (jt) {
                case articulation::ArticulationJointType::Revolute:
                case articulation::ArticulationJointType::Prismatic:
                    model.dof_to_link.push_back(l);
                    model.dof_to_component.push_back(~uint32_t(0));
                    break;
                default:
                    break;  // Fixed (0 DOF) / non-root FloatingBase: no entry.
            }
        }
    }
    // The general per-candidate-slot row budget. The broadphase emits up to
    // max_contacts_per_env candidate slots per env; the assembly expands each into
    // kPairDrivenRowsPerSlot NkRows. The schedule + the assembly both read this.
    nk::ModelCapacities& cap = model.capacities;
    if (cap.max_contacts_per_env > 0u) {
        cap.max_rows_per_env =
            cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    }
    return result;
}

// ---------------------------------------------------------------------------
// H2 (general contact pipeline Phase 2): the ONE cook-time heightfield grid
// generator. See cook_to_model.hpp for the contract. SampleTerrainHeight is
// called ONLY here (cook); the contact kernel reads the cooked grid (H3).
// ---------------------------------------------------------------------------
uint32_t CookHeightfieldGrid(nk::Model& model,
                             const ::nuka::terrain::TerrainParams& terrain,
                             uint32_t terrain_type,
                             uint32_t nrow, uint32_t ncol, float cell_size,
                             const math::Vec3& center) {
    if (nrow < 2u || ncol < 2u || !(cell_size > 0.0f)) return ~0u;

    const float half_x = 0.5f * static_cast<float>(ncol - 1u) * cell_size;
    const float half_y = 0.5f * static_cast<float>(nrow - 1u) * cell_size;

    // 1. Sample the absolute surface height at every grid corner (the SAME pure
    //    SampleTerrainHeight the feet/obs/render call). Grid corner (r,c) sits at
    //    LOCAL (x,y) = (-half_x + c*cell, -half_y + r*cell); the WORLD column the
    //    generator samples is the local column + center.xy (the field is centred
    //    at `center`). Track the absolute z-range for the normalize + the AABB.
    const uint32_t cells = nrow * ncol;
    const uint32_t base = static_cast<uint32_t>(model.heightfield_heights.size());
    std::vector<float> abs_z(cells, 0.0f);
    float min_z = 3.402823466e+38f, max_z = -3.402823466e+38f;
    for (uint32_t r = 0; r < nrow; ++r) {
        const float ly = -half_y + static_cast<float>(r) * cell_size;
        for (uint32_t c = 0; c < ncol; ++c) {
            const float lx = -half_x + static_cast<float>(c) * cell_size;
            const float z = ::nuka::terrain::SampleTerrainHeight(
                terrain_type, lx + center.x, ly + center.y, terrain);
            abs_z[r * ncol + c] = z;
            if (z < min_z) min_z = z;
            if (z > max_z) max_z = z;
        }
    }
    if (!(max_z > min_z)) max_z = min_z + 1.0e-4f;  // flat grid: tiny finite range.

    // 2. Store NORMALIZED [0,1] heights (Newton HeightfieldData convention: the
    //    grid stores h in [0,1] and the prism extractor maps z = min_z + h*range).
    //    The descriptor's min_z/max_z are LOCAL (relative to the body pose at
    //    `center`), so a corner's local z = (abs_z - center.z); the absolute world
    //    z is then recovered as body_pose.z (== center.z) + local z.
    const float local_min = min_z - center.z;
    const float local_max = max_z - center.z;
    const float range = local_max - local_min;
    model.heightfield_heights.reserve(base + cells);
    for (uint32_t i = 0; i < cells; ++i) {
        const float local_z = abs_z[i] - center.z;
        const float h = (range > 0.0f) ? (local_z - local_min) / range : 0.0f;
        model.heightfield_heights.push_back(h);
    }

    // 3. Push the HeightfieldData descriptor.
    nk::HeightfieldData hfd;
    hfd.origin = math::Vec3{-half_x, -half_y, local_min};  // local (0,0) corner.
    hfd.cell_size = cell_size;
    hfd.nrow = nrow;
    hfd.ncol = ncol;
    hfd.min_z = local_min;
    hfd.max_z = local_max;
    hfd.data_offset = base;
    model.heightfields.push_back(hfd);
    model.capacities.max_heightfield_cells =
        static_cast<uint32_t>(model.heightfield_heights.size());

    // 4. Register the STATIC heightfield collidable: a body_init row (immovable,
    //    inv_mass==0 -> the integrator holds its pose) at `center` + a shape_table
    //    row kind=Heightfield, body_id==-1 (static, no reaction side). The AABB
    //    case (B3) reads p[0]=half_x, p[1]=half_y, p[2]=local_min, p[3]=local_max.
    const uint32_t body_row = static_cast<uint32_t>(model.body_init.size());
    nk::Model::BodyInit bi;
    bi.pose = math::Transform::Identity();
    bi.pose.position = center;
    bi.inv_mass = 0.0f;            // static / immovable.
    bi.inv_inertia = math::Vec3{0, 0, 0};
    model.body_init.push_back(bi);

    nk::Model::PairDrivenShape sh;
    sh.kind = ::nuka::collision::kShapeHeightfield;
    sh.params[0] = half_x;
    sh.params[1] = half_y;
    sh.params[2] = local_min;
    sh.params[3] = local_max;
    sh.contype = 1u;
    sh.conaffinity = 1u;
    sh.sdf_grid = ~0u;
    sh.body_id = -1;               // STATIC: no owning body, no reaction side.
    sh.group = 0u;
    if (model.shape_table_rows.size() <= body_row) {
        model.shape_table_rows.resize(body_row + 1u);
    }
    model.shape_table_rows[body_row] = sh;
    return body_row;
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
    // legacy soft-upload layout: distance a/b/rest/alpha; bend 4-particle +
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

    // M9 T11 SHAPE-MATCH (id 9): flatten the variable-size clusters into the CSR
    // (offset, size) layout over flat particle / rest-offset / weight pools. Cook
    // the rest centroid c0 = (sum_i m_i x_i^0)/sum_i m_i and the per-member rest
    // OFFSET q_i = x_i^0 - c0 ONCE -- BYTE-FAITHFUL to the legacy soft-upload
    // shape-match flatten (same fixed-order ascending accumulation).
    const uint32_t scn = static_cast<uint32_t>(in.shape_match.size());
    mp.sm_cluster_offset.clear(); mp.sm_cluster_size.clear();
    mp.sm_stiffness.clear(); mp.sm_rest_centroid.clear();
    mp.sm_particles.clear(); mp.sm_rest_q.clear(); mp.sm_mass.clear();
    mp.sm_cluster_offset.resize(scn); mp.sm_cluster_size.resize(scn);
    mp.sm_stiffness.resize(scn); mp.sm_rest_centroid.resize(scn);
    for (uint32_t cl = 0; cl < scn; ++cl) {
        const CookShapeMatchCluster& smc = in.shape_match[cl];
        const size_t n = smc.particle.size();
        // Cook c0 = (sum_i m_i x_i^0)/sum_i m_i (fixed-order ascending sum).
        float mass_sum = 0.0f;
        math::Vec3 c0{0.0f, 0.0f, 0.0f};
        for (size_t j = 0; j < n; ++j) {
            const float mi = j < smc.cluster_mass.size() ? smc.cluster_mass[j] : 1.0f;
            mass_sum += mi;
            if (j < smc.rest_positions.size()) c0 = c0 + smc.rest_positions[j] * mi;
        }
        if (mass_sum > 0.0f) c0 = c0 * (1.0f / mass_sum);
        mp.sm_cluster_offset[cl] = static_cast<uint32_t>(mp.sm_particles.size());
        mp.sm_cluster_size[cl] = static_cast<uint32_t>(n);
        mp.sm_stiffness[cl] = smc.stiffness;
        mp.sm_rest_centroid[cl] = c0;
        for (size_t j = 0; j < n; ++j) {
            mp.sm_particles.push_back(smc.particle[j]);
            const math::Vec3 x0 = j < smc.rest_positions.size()
                                      ? smc.rest_positions[j] : math::Vec3::Zero();
            mp.sm_rest_q.push_back(x0 - c0);  // q_i = x_i^0 - c0
            mp.sm_mass.push_back(j < smc.cluster_mass.size() ? smc.cluster_mass[j]
                                                             : 1.0f);
        }
    }

    cap.particles_per_env = static_cast<uint32_t>(mp.initial_pos.size());
    cap.dist_cons_per_env = dn;
    cap.bend_cons_per_env = bn;
    cap.vol_cons_per_env  = vn;
    cap.shape_match_slots_per_env   = scn;
    cap.shape_match_members_per_env =
        static_cast<uint32_t>(mp.sm_particles.size());
    // n_soft_particles is left at its default (0); it is only consulted for the
    // SoftFluid mode (CookSoftFluidParticles sets it). The single-system Xpbd
    // ops ignore it (mode-gated), so the device-staged bytes are unaffected.
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

// ---------------------------------------------------------------------------
// M9 T11 two-system cook: soft (XPBD) + fluid (PBF) co-resident in ONE Model.
// ---------------------------------------------------------------------------

void CookSoftFluidParticles(nk::Model& model, uint32_t env_count,
                            const XpbdCookInput& soft, const PbfCookInput& fluid,
                            const SoftFluidContactInput& contact) {
    const uint32_t envs = env_count > 0 ? env_count : 1u;
    nk::ModelCapacities& cap = model.capacities;
    if (cap.env_count == 1u || cap.env_count == 0u) cap.env_count = envs;

    // STRICT-SUPERSET FAST PATHS: a soft-only / fluid-only co-residence cook is
    // byte-identical to the single-system cook (so the existing single-system
    // gates and goldens are unaffected). Only when BOTH sides are present do we
    // build the [soft | fluid] composite + set the SoftFluid mode. The cross-system
    // contact is a SoftFluid-only feature (it needs both slices), so a single-
    // system fast path never carries it.
    const uint32_t n_soft = static_cast<uint32_t>(soft.positions.size());
    const uint32_t n_fluid = static_cast<uint32_t>(fluid.positions.size());
    if (n_fluid == 0u) {
        // Soft-only: identical to the canonical XPBD cook (incl. shape-match).
        CookXpbdParticles(model, env_count, soft);
        return;
    }
    if (n_soft == 0u) {
        // Fluid-only: identical to the canonical PBF cook.
        CookPbfParticles(model, env_count, fluid);
        return;
    }

    // 1) Cook the SOFT set first (fills the XPBD + shape-match templates, sets
    //    particles_per_env = n_soft, mode = Xpbd). The soft constraint indices
    //    already point into [0, n_soft) -- exactly where the soft particles land.
    CookXpbdParticles(model, env_count, soft);

    nk::Model::ModelParticles& mp = model.particles;

    // 2) APPEND the fluid particles AFTER the soft set ([soft | fluid] layout).
    //    The fluid particles are NOT referenced by any soft constraint, so no
    //    index remap is needed; the SoftFluid PBF ops scope the density solve to
    //    the fluid slice [n_soft, n_soft+n_fluid) by the n_soft split.
    const float fluid_im =
        fluid.particle_mass > 0.0f ? 1.0f / fluid.particle_mass : 0.0f;
    for (uint32_t i = 0; i < n_fluid; ++i) {
        mp.initial_pos.push_back(fluid.positions[i]);
        mp.initial_vel.push_back(
            i < fluid.velocities.size() ? fluid.velocities[i] : math::Vec3::Zero());
        mp.inv_mass.push_back(
            i < fluid.inv_mass.size() ? fluid.inv_mass[i] : fluid_im);
    }

    // 3) Carry the PBF fluid params + uniform-grid domain (the fluid slice solve).
    mp.pbf_rest_density   = fluid.rest_density;
    mp.pbf_support_radius = fluid.support_radius;
    mp.pbf_particle_mass  = fluid.particle_mass;
    mp.pbf_cfm_epsilon    = fluid.cfm_epsilon;
    mp.pbf_iters          = fluid.iters == 0u ? 1u : fluid.iters;
    mp.pbf_clamp_overdensity = fluid.clamp_overdensity;
    mp.pbf_xsph_viscosity = fluid.xsph_viscosity;
    mp.pbf_surface_tension= fluid.surface_tension;
    mp.grid_min = fluid.grid_min;
    mp.grid_dims[0] = fluid.grid_dims[0];
    mp.grid_dims[1] = fluid.grid_dims[1];
    mp.grid_dims[2] = fluid.grid_dims[2];
    mp.cell_size = fluid.support_radius;
    mp.query_radius = fluid.support_radius;
    mp.boundary_enabled = fluid.boundary_enabled;
    mp.floor_z = fluid.floor_z;

    // 4) id-10 cross-system contact (M9 T11 Phase 2): the class-blind unilateral
    //    non-penetration co-step over the FULL union. The neighbor grid is built
    //    over query_radius, so it must cover d_min for the contact pass to see
    //    every penetrating pair; widen the grid cell/query radius to >= d_min when
    //    contact is enabled (a no-op when d_min <= the fluid support radius).
    mp.pp_contact_d_min      = contact.contact_d_min;
    mp.pp_contact_compliance = contact.compliance_alpha;
    mp.pp_contact_iters = contact.solver_iterations == 0u ? 1u
                                                          : contact.solver_iterations;
    if (mp.pp_contact_d_min > 0.0f && mp.pp_contact_d_min > mp.query_radius) {
        mp.cell_size    = mp.pp_contact_d_min;
        mp.query_radius = mp.pp_contact_d_min;
    }

    // 5) The co-residence schema: mode + split index + the new total particle
    //    count. The grid is sized over the FULL union per-env (the soft particles
    //    occupy grid cells too, but the fluid density solve skips them via n_soft).
    mp.mode = nk::Model::ParticleMode::SoftFluid;
    mp.n_soft_particles = n_soft;
    cap.particles_per_env = n_soft + n_fluid;
    cap.max_grid_cells =
        fluid.grid_dims[0] * fluid.grid_dims[1] * fluid.grid_dims[2];
}

} // namespace nuka::scene::cook
