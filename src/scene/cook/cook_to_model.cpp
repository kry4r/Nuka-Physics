// ---------------------------------------------------------------------------
// nk CookToModel implementation (the design /).
//
// MAPPING CHOICES (documented for /SceneMap consumers):
// * Bodies: cooked body rows ARE record order (the cooker preserves SceneIR
// body order), so SceneMap binds entity-of-body -> body_row = its cooked row
// for every body. The nk::Model carries the per-env movable rigid-body COUNT
// (bodies_per_env); the actual body state lives in the Data arena (env-major
// replica e at [e*bodies_per_env, ...)).
// * Articulation: CookArticulations(blob) yields the per-env kinematic-tree
// template(s). the cook supports the single-articulation-per-env scene (the H1 /
// grasp shape); the first articulation becomes the Model template. dof_count
// / link_count drive the dof/link capacities. Links bind to SceneMap via
// link_index = the template-local link slot.
// * Shapes: each cooked shape ROW becomes one ModelShape. A mesh that V-HACD
// decomposed into N pieces produces N consecutive shape rows; the SOURCE
// shape entity binds to the FIRST of those rows (shape_row) and records the
// piece count via bp_group = N (so a consumer can walk the piece span). For
// a primitive (1 row) bp_group == 1.
// * Material buckets: one bucket per cooked shape's resolved (μ, restitution,
// ...) — deduplicated by exact value so a homogeneous scene collapses to a
// few buckets (Isaac bucketing). body_material_bucket[body_row] = the bucket
// of that body's FIRST shape (0 if the body has no shape).
// ---------------------------------------------------------------------------

#include "scene/cook/cook_to_model.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "collision/cross_system_query.hpp"  // kBodyParticleContactSlotsPerParticle
#include "collision/shape_kind.hpp"  // nuka::collision::ShapeKind (R2: one enum)
#include "scene/terrain/heightfield.hpp"  // HeightField (the cooked grid source)
#include "scene/terrain/heightfield_loaders.hpp"  // parametric/image grid fill
#include "nk/solve/nk_row.hpp"       // kPairDrivenRowsPerSlot (B1 row budget)
#include "runtime/articulation/articulation_cooker.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "scene/cooker.hpp"
#include "scene/ecs/components.hpp"
#include "scene/graph/scene_graph.hpp"
// The cloth-topology + fluid-box cookers (CUDA-free POD products); the media cook
// reuses them to build the particle inputs instead of reimplementing the math.
#include "runtime/soft/cloth_topology.hpp"     // BuildClothConstraints
#include "runtime/soft/tetmesh_topology.hpp"   // BuildSphereTetLattice / BuildTetMeshConstraints
#include "import/cooker/fluid_cooker.hpp"       // CookFluidBox

namespace nuka::scene::cook {

namespace {

// ExportObs per-env observation layout (readout.cu ExportObsKernel): the base
// pose (pos.xyz + quat.wxyz) followed by per-link q then per-link qdot. Used to
// DERIVE obs_width from the cooked link count instead of a baked buffer width.
constexpr uint32_t kObsBasePoseFloats = 7u;        // pos.xyz + quat.wxyz
constexpr uint32_t kObsChannelsPerLink = 2u;       // q + qdot per link
// Per-collidable candidate-pair budget for the general LBVH broadphase: the
// per-env candidate-slot capacity == collidable_count * this. Named (not a baked
// FusedFoot foot count) so the contact buffer grows with the cooked geometry.
constexpr uint32_t kCandidatePairsPerCollidable = 4u;

// One EXTRA collision geom of an articulation link (beyond the first, which folds
// into the link's own body row via link_geom). Materialized into an appended
// collidable body row so a link owns multiple collidables (multi-geom feet).
struct ProxyCollidableSpec {
    uint32_t owner_link  = ~0u;  // template-local link (pose source).
    uint32_t owner_body  = ~0u;  // owner body row (exclusion inheritance).
    uint32_t owner_artic = ~0u;  // template-local articulation.
    uint32_t shape_row   = ~0u;  // cooked ModelShape / blob shape index.
};

// resolve the AUTHORED initial-condition for the cooked articulation
// (the settle product, controller ruling R4: BAKED). Two sources, in priority:
// (1) the InitialStateComponent set on the articulation root entity by
// ApplySettleToRegistry (the ECS/settle writeback path), and
// (2) the SceneIR's `initial_state` map keyed by a tree node PATH (the .nks
// persisted IC): match the articulation root's node path, then any ancestor
// PREFIX of it (the import attach prefix, e.g. "h1"), and finally — when the
// map carries exactly one entry whose qpos length equals the cooked link
// count — that sole entry (a single-articulation scene).
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

// Resolve the cooked per-shape restitution (producer side of the per-material
// contact contract). FLAG: CookedContactParamTable (another agent's file) has no
// restitution lane yet, so this returns the spec default 0 (default-material
// scenes stay numerically unchanged). Replace the body with
// blob.contact_params.restitutions[shape_row] once that upstream lane lands.
float ResolveShapeRestitution(const CookedBlob& /*blob*/, uint32_t /*shape_row*/) {
    return 0.0f;
}

// Find or append a material bucket matching the cooked contact-param row `i`.
// Returns the bucket index. Dedup is by exact float equality (cook is
// deterministic, so identical authored params give bit-identical bucket rows).
uint32_t BucketFor(const CookedBlob& blob, uint32_t shape_row,
                   std::vector<nk::ModelMaterialBucket>& buckets) {
    nk::ModelMaterialBucket row;
    if (shape_row < blob.contact_params.frictions.size()) {
        row.values[nk::kBucketStaticMu]  = blob.contact_params.frictions[shape_row];
        row.values[nk::kBucketDynamicMu] = blob.contact_params.frictions[shape_row];
    }
    // Per-shape restitution into the named lane (producer side of the
    // per-material contact contract). The cooked blob carries no restitution lane
    // yet (CookedContactParamTable, another agent's file), so this resolves to the
    // bucket default 0 until that upstream lane lands -- see the FLAG in the cook.
    row.values[nk::kBucketRestitution] = ResolveShapeRestitution(blob, shape_row);
    // Compliant params: solref timeconst/dampratio into the named lanes.
    if (shape_row < blob.contact_params.solref0.size()) {
        row.values[nk::kBucketTimeconst] = blob.contact_params.solref0[shape_row];
    }
    if (shape_row < blob.contact_params.solref1.size()) {
        row.values[nk::kBucketDampratio] = blob.contact_params.solref1[shape_row];
    }
    if (shape_row < blob.contact_params.margins.size()) {
        row.values[nk::kBucketMargin] = blob.contact_params.margins[shape_row];
    }
    for (uint32_t b = 0; b < buckets.size(); ++b) {
        bool same = true;
        for (uint32_t k = 0; k < nk::ModelMaterialBucket::kValueCount; ++k) {
            if (buckets[b].values[k] != row.values[k]) { same = false; break; }
        }
        if (same) return b;
    }
    buckets.push_back(row);
    return static_cast<uint32_t>(buckets.size() - 1);
}

// SAMP cook : build the SDF sampling-point set for one convex-hull
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

// Grow the body-contact budget by a DISJOINT reserve above `rigid_base` (the cooked
// body<->body budget) for body<->particle rows; idempotent, byte-identical when 0.
void GrowContactBudgetForParticles(nk::ModelCapacities& cap, uint32_t rigid_base) {
    if (rigid_base == 0u) return;  // no body contacts -> no body<->particle rows.
    const uint64_t reserve =
        static_cast<uint64_t>(cap.particles_per_env) *
        collision::gpu::kBodyParticleContactSlotsPerParticle;
    const uint64_t total = static_cast<uint64_t>(rigid_base) + reserve;
    if (total > 0xFFFFFFFFull ||
        total * nk::kPairDrivenRowsPerSlot > 0xFFFFFFFFull) {
        throw std::runtime_error(
            "CookToModel: body<->particle contact budget overflows u32 "
            "(too many particles for the per-env contact slot block)");
    }
    cap.max_contacts_per_env = static_cast<uint32_t>(total);
    cap.max_rows_per_env =
        cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
}

namespace {

// the shared cook implementation. The cook options gate the two HEAVY
// per-shape stages (V-HACD vs single-hull, SDF bake). The public 2-arg overload
// passes the DEFAULT options (legacy, byte-identical); the PairDriven 3-arg passes
// the general-path options (single-hull + no SDF) so the whole-body union scene
// cooks tractably. No entity/scene-name branch — the options are the only seam.
CookToModelResult CookToModelImpl(const SceneIR& scene, int env_count,
                                  const CookSceneOptions& cook_options,
                                  bool enable_contacts = true) {
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
    // obs_width is derived from the cooked observable dimensionality below, once
    // links_per_env is known (ExportObs layout: base pose + q + qdot per link).

    // which body rows are owned by the articulation (driven by FK, NOT
    // free rigid bodies). The movable-body body_init pass below MUST keep these
    // rows at inv_mass 0 so the rigid-body gravity-kick / IntegrateBodyPosition
    // arms (which run over bodies_per_env, gated on body_inv_mass > 0) stay
    // no-ops for them — exactly today's behavior (an articulation-only world has
    // an EMPTY body_init, so SeedInitialState skips the body block entirely).
    std::vector<uint8_t> body_is_articulation_link(blob.body_count, 0u);

    // EXTRA link collision geoms (2nd+ primitive per link) -> appended collidable
    // body rows (materialized after the model is built). Empty for single-geom links.
    std::vector<ProxyCollidableSpec> link_proxies;

    // 4. Articulation template(s). transcribed the SINGLE-ENV
    // BuildArticulationHostState product 1:1; the multi-articulation
    // foundation transcribes the FULL set of co-resident topologies (K Go2 in
    // ONE env). BuildArticulationHostState(arts, ...) ALREADY loops over every
    // topology, building the flat per-link arrays + link_to_articulation +
    // articulation_link_count/offset with a running global offset, so the
    // only change here is to pass the WHOLE `arts` vector (not just the front)
    // and carry the per-articulation bookkeeping into nk::Model. At
    // articulations_per_env == 1 (the legacy single-robot scene) the host state
    // is identical to the BuildArticulationHostState({arts.front}) product,
    // so the staged Model bytes are byte-for-byte unchanged (the K==1 D1
    // invariant). dofs_per_env stays the MAX single-dog DOF (NEVER summed): the
    // CRBA M-tile is per-articulation (one max_dof^2 block per dog).
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
        m.joint_frictionloss = host.joint_frictionloss;
        m.initial_q = host.q;                  // per LINK (scalar slot / link)
        m.initial_link_pose = host.link_pose;  // cook rest pose
        m.base_pose = host.base_pose.empty() ? math::Transform::Identity()
                                             : host.base_pose.front();
        m.link_body = host.link_body;
        // multi-articulation co-residence bookkeeping (host already built it).
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

        // R3 (general contact pipeline): the shape->body INVERSE tables.
        // link_body is link->body; the general PairDriven row-emission needs
        // body->owner. For each TEMPLATE link l with owning body row b, record
        // body_to_link[b] = l and body_to_articulation[b] = the LOCAL articulation
        // of l. A body row owned by NO link (free rigid / static) keeps ~0u (so
        // its side resolves free-rigid; the shape_table body_id == -1 resolves
        // static). Both stage template-local + tile env-major. INERT before the general path is wired.
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

        // seed the AUTHORED settled IC (qpos -> initial_q per-link, root
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
        // Spatial inertia float count derived from the source 6x6 record (no bare 36).
        constexpr size_t kSpatialInertiaFloats =
            sizeof(host.link_inertia[0].I) / sizeof(float);
        m.link_inertia_spatial.resize(static_cast<size_t>(m.link_count) *
                                      kSpatialInertiaFloats);
        for (uint32_t l = 0; l < m.link_count; ++l) {
            for (size_t k = 0; k < kSpatialInertiaFloats; ++k) {
                m.link_inertia_spatial[static_cast<size_t>(l) * kSpatialInertiaFloats + k] =
                    host.link_inertia[l].I[k];
            }
        }
        // DOF count (Revolute/Prismatic = 1, Fixed = 0, FloatingBase = 6) — the
        // legacy ArticulationDofCount semantics (inlined: that symbol lives in
        // the GPU lib, which the pure cook must not link), i.e. max_dof.
        //
        // multi-articulation: dofs_per_env is the MAX single-DOG DOF (the
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

        // Foot shapes: every Sphere shape whose owning body maps to an
        // articulation link is a foot (base-relative indices).
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
        // for EVERY scene; the UnionCsr graph never reads it (it only feeds the
        // PairDriven SyncLinkBodyPose op).
        // kind sentinel: 0 == none; a primitive stores (ShapeType + 1) so the
        // default-zero (no-shape) link is unambiguously inactive.
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
                // The link already carries its FIRST primitive (folded into its own
                // body row). This 2nd+ primitive becomes an appended collidable
                // PROXY row so the link owns multiple collidables (multi-geom feet).
                ProxyCollidableSpec px;
                px.owner_link = owner_link;
                px.owner_body = body;
                px.owner_artic = body < model.body_to_articulation.size()
                                     ? model.body_to_articulation[body]
                                     : ~uint32_t(0);
                px.shape_row = shape;
                link_proxies.push_back(px);
                continue;
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

    // Observation export width = the cooked observable dimensionality (base pose +
    // q + qdot per link), so a >64-channel policy is not silently truncated. A
    // link-free scene exports just the base pose.
    cap.obs_width = kObsBasePoseFloats +
                    kObsChannelsPerLink * cap.links_per_env;

    // 5. Shapes -> ModelShape rows + material buckets. Track the SOURCE shape ->
    // first cooked row + piece count for the SceneMap binding.
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
    // record order) to its first cooked row. For the primitive scenes the cook produces,
    // source row == cooked row, span == 1.
    {
        uint32_t cooked_cursor = 0;
        for (uint32_t src = 0; src < scene.ShapeCount(); ++src) {
            const EntityId ent = scene.EntityOfShape(src);
            // Count how many cooked rows this source produced: a V-HACD source
            // (DecomposeMode::Force) expands into a contiguous span of pieces that
            // share its body_row, so advance while consecutive cooked rows keep the
            // same body_row. A primitive / single-hull source -> span 1.
            // FLAG: body_row is only an APPROXIMATION of source provenance -- a body
            // carrying several distinct source shapes would merge their rows into
            // one span. An exact fix needs per-cooked-row source ids from the cooker
            // (another agent's CookedShapeTable). NO-OP for the single-hull path
            // (every source -> 1 row), which is how every scene currently cooks.
            uint32_t span = 1;
            if (cooked_cursor < model.shapes.size()) {
                const uint32_t body = model.shapes[cooked_cursor].body_row;
                while (cooked_cursor + span < model.shapes.size() &&
                       model.shapes[cooked_cursor + span].body_row == body) {
                    ++span;
                }
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

    // 7b. Seed body_init for every non-link body row: only link rows are re-posed
    // each step, so an unseeded static/free collidable would sit at the zero pose.
    {
        bool any_free_body = false;
        for (uint32_t b = 0; b < blob.body_count; ++b) {
            const bool is_link = b < body_is_articulation_link.size() &&
                                 body_is_articulation_link[b] != 0u;
            if (!is_link) {
                any_free_body = true;
                break;
            }
        }
        if (any_free_body) {
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

    // 8. Filter policy (cross-env collision flag). the cook defaults OFF (envs do not
    // collide); the cooked filter pair lists ride the blob for .
    model.filter_cross_env = false;

    // 9. Contact / row capacities. ONE general per-env candidate-slot budget for
    // the LBVH broadphase, driven by the actual cooked collidable count (the
    // body rows the broadphase iterates) -- NO scene-type branch and NO baked
    // foot-count constants. The broadphase builds over bodies_per_env collidable
    // body rows and emits up to max_contacts_per_env candidate pairs (it SILENTLY
    // drops overflow, lbvh_traversal.cuh), so the budget scales with the
    // collidables: collidable_count * kCandidatePairsPerCollidable. The static
    // ground collidable (appended in section 10) is included via +1. The
    // PairDriven overload re-sizes max_rows_per_env to its per-candidate-slot
    // row layout; max_rows here is the broadphase-agnostic initial bound.
    {
        const uint32_t static_collidables = 1u;  // the ground plane (section 10).
        const uint32_t collidables = cap.bodies_per_env + static_collidables;
        cap.max_contacts_per_env =
            cap.bodies_per_env > 0u ? collidables * kCandidatePairsPerCollidable
                                    : 0u;
        cap.max_rows_per_env =
            cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    }
    // Contacts-OFF dynamics world (the general-path equivalent of the legacy
    // single-env enable_contacts==false): zero the per-env contact/candidate
    // budget so the pipeline emits NO broadphase/narrowphase/solve ops (has_-
    // collidables gates on max_contacts_per_env). The cooked shape_table / link_-
    // geom rows below stay sized consistently but are never read -- the world runs
    // pure articulation+rigid dynamics (FK -> CRBA -> implicit damping -> integrate),
    // and StepPlanned captures cleanly (no thrust LBVH). Also keeps the PairDriven
    // overload's row-budget resize a no-op (it guards on max_contacts_per_env > 0).
    if (!enable_contacts) {
        cap.max_contacts_per_env = 0u;
        cap.max_rows_per_env     = 0u;
    }

    // 10. — pair-driven generalized-collision + SDF main-path staging (plan
    // the spec). The shape_table (one PairDrivenShape / collidable body row),
    // the SDF sampling-point pool (SAMP cook), the cooked sparse-SDF grids
    // (SdfDeviceWorld upload duties moved INTO the Model), and the filter
    // exclude-list. A scene with no SdfMesh shape leaves the SDF
    // tables empty (max_sdf_* == 0); they are populated for a cooked
    // pair-driven SDF scene. These tables are sized AFTER the contact
    // capacity (max_bodies_total == bodies_per_env, the shape_table stride).
    cap.max_bodies_total = cap.bodies_per_env;
    {
        // shape_table: one row per body, from its FIRST shape's primitive + the
        // contype/conaffinity (from the cooked filter groups when present).
        model.shape_table_rows.assign(cap.bodies_per_env, {});
        // Per body, pick its first COLLIDING shape (contype/conaffinity != 0);
        // a visual-only body falls back to its first shape. Keeps a real collision
        // primitive as the body row instead of a non-colliding visual mesh.
        auto shape_collides = [&](uint32_t i) {
            const uint32_t ct = i < blob.contact_params.contypes.size()
                                   ? blob.contact_params.contypes[i] : 1u;
            const uint32_t ca = i < blob.contact_params.conaffinities.size()
                                   ? blob.contact_params.conaffinities[i] : 1u;
            return (ct != 0u) || (ca != 0u);
        };
        std::vector<uint32_t> body_shape(cap.bodies_per_env, ~0u);
        for (uint32_t i = 0; i < model.shapes.size(); ++i) {
            const uint32_t b = model.shapes[i].body_row;
            if (b >= cap.bodies_per_env) continue;
            if (body_shape[b] == ~uint32_t(0)) body_shape[b] = i;
            else if (shape_collides(i) && !shape_collides(body_shape[b])) body_shape[b] = i;
        }
        for (uint32_t b = 0; b < cap.bodies_per_env; ++b) {
            const uint32_t s = body_shape[b];
            if (s == ~uint32_t(0)) continue;
            const nk::ModelShape& sh = model.shapes[s];
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
            // unrelated to the actual hull extent). : ALSO pack this
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
            // contype/conaffinity from the cooked per-shape values (the MuJoCo
            // bitmask the broadphase tests), NOT a blanket collide-all -- a
            // visual-only geom (contype 0) must stay non-colliding.
            row.contype = s < blob.contact_params.contypes.size()
                              ? blob.contact_params.contypes[s] : 1u;
            row.conaffinity = s < blob.contact_params.conaffinities.size()
                                  ? blob.contact_params.conaffinities[s] : 1u;
            row.sdf_grid = ~0u;  // resolved below if the piece has a cooked SDF.
            // R1 (general contact pipeline): the shape->body indirection.
            // A cooked collidable row maps to its OWNING body row (never static
            // here — static collidables are emitted separately, R5). group 0 ==
            // the default collide-all group. INERT before the general path is wired (no routing reads
            // these lanes; a later pass wire them in the general path).
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

            // Bind each visual-mesh silhouette SDF to the collidable's sdf_grid (lane 7:
            // MLS-MPM grid BC + particle query); the rigid kind/params stay the primitive.
            for (uint32_t b = 0; b < cap.bodies_per_env; ++b) {
                if (b >= sdf.body_sdf_indices.size()) break;
                const uint32_t gi = sdf.body_sdf_indices[b];
                if (gi == kNoSdf || b >= model.shape_table_rows.size()) continue;
                model.shape_table_rows[b].sdf_grid = gi;
                // Populate the orphaned ModelShape.sdf_index for this body's rows.
                for (nk::ModelShape& msh : model.shapes)
                    if (msh.body_row == b) msh.sdf_index = gi;
            }
        }
        cap.max_samp_points = static_cast<uint32_t>(model.samp_points.size() / 3u);
        cap.max_sdf_grids   = static_cast<uint32_t>(model.sdf_grids.size());
        cap.max_sdf_cells   = static_cast<uint32_t>(model.sdf_cell_keys.size());
        // Transcribe the cooked filter exclude-list (authored <exclude> UNION
        // joint parent-child, the MuJoCo filterparent rule) into the model's
        // body-pair key list the broadphase binary-searches. Keys are body rows
        // (cooked body row == SceneIR body id, == the broadphase leaf body), so
        // no remap. The list is already (min,max)-canonical sorted+deduped.
        model.excluded_pairs.clear();
        model.excluded_pairs.reserve(blob.filter_policy.excluded_body_pairs.size());
        for (const auto& p : blob.filter_policy.excluded_body_pairs) {
            const uint32_t lo = p.first < p.second ? p.first : p.second;
            const uint32_t hi = p.first < p.second ? p.second : p.first;
            if (lo >= cap.bodies_per_env || hi >= cap.bodies_per_env) {
                throw std::runtime_error(
                    "CookToModel: excluded body pair references a body row beyond "
                    "bodies_per_env (cook topology bug)");
            }
            model.excluded_pairs.push_back(
                (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi));
        }
        // Two immovable bodies can never produce a reaction row; exclude their
        // pairs so static set dressing does not consume the candidate budget.
        for (uint32_t i = 0; i + 1u < cap.bodies_per_env; ++i) {
            if (i >= blob.bodies.is_static.size() || !blob.bodies.is_static[i])
                continue;
            for (uint32_t j = i + 1u; j < cap.bodies_per_env; ++j) {
                if (j >= blob.bodies.is_static.size() || !blob.bodies.is_static[j])
                    continue;
                model.excluded_pairs.push_back(
                    (static_cast<uint64_t>(i) << 32) | static_cast<uint64_t>(j));
            }
        }
        std::sort(model.excluded_pairs.begin(), model.excluded_pairs.end());
        model.excluded_pairs.erase(
            std::unique(model.excluded_pairs.begin(), model.excluded_pairs.end()),
            model.excluded_pairs.end());
        cap.max_excluded_pairs =
            static_cast<uint32_t>(model.excluded_pairs.size());
        // the convex-hull vertex pool capacity = the concatenated
        // per-shape hull verts packed above (xyz triples). 0 for a hull-free
        // scene (go2) -> the hull_verts segment stays zero bytes (byte-inert).
        cap.max_hull_verts =
            static_cast<uint32_t>(model.hull_verts.size() / 3u);

        // R5 (general contact pipeline): emit a STATIC ground collidable
        // row into shape_table as a first-class collidable with body_id == -1 (no
        // reaction side). Appended AFTER the per-body rows so the per-body rows
        // [0, bodies_per_env) keep their exact indices + bytes; the static row
        // lives at index bodies_per_env. contype/conaffinity == 1 (collide-all).
        // INERT before the general path is wired: the broadphase iterates only bodies_per_env body rows
        // per env (it never indexes the static row), and shape_table is a model
        // param pinned by no golden, so growing max_bodies_total by one row is
        // byte-safe for the gated UnionCsr family. The static row's
        // WORLD pose / LBVH entry is wired in the general path (R5 downstream + B3); here we
        // only register the collidable. A Plane kind anchors the flat-ground case
        // (the general heightfield collidable, kShapeHeightfield, is added by H2/H3
        // in the general path). params[0] carries the ground plane height for the future
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

    // Terrain bake (gated on a TerrainRecord -> terrain-free scenes stay byte-
    // identical): build the HeightField, bake via the shared helper, retain it.
    if (!scene.Terrain().empty()) {
        const TerrainRecord& tr = scene.Terrain().front();
        ::nuka::terrain::HeightField hf;
        if (!tr.image_path.empty()) {
            std::string err;
            if (!::nuka::terrain::LoadHeightFieldFromImage(
                    tr.image_path, tr.image_radius_x, tr.image_radius_y,
                    tr.image_elevation_z, tr.base_z, tr.origin, hf, err)) {
                throw std::runtime_error("CookToModel: terrain image load failed: " + err);
            }
        } else {
            ::nuka::terrain::TerrainGenConfig cfg;
            cfg.nrow = tr.nrow;            cfg.ncol = tr.ncol;
            cfg.cell_x = tr.cell;         cfg.cell_y = tr.cell;
            cfg.origin = tr.origin;       cfg.base_z = tr.base_z;
            cfg.grade_x = tr.grade_x;     cfg.grade_y = tr.grade_y;
            cfg.ring_rise = tr.ring_rise; cfg.ring_width = tr.ring_width;
            cfg.ring_platform = tr.ring_platform; cfg.ring_count = tr.ring_count;
            cfg.bump_height = tr.bump_height;     cfg.bump_cell = tr.bump_cell;
            cfg.feature_cell = tr.feature_cell;   cfg.feature_margin = tr.feature_margin;
            cfg.feature_seed = tr.feature_seed;
            cfg.curric_levels = tr.curric_levels; cfg.curric_types = tr.curric_types;
            if (!::nuka::terrain::GenerateHeightField(cfg, hf)) {
                throw std::runtime_error("CookToModel: degenerate terrain config");
            }
        }
        CookTerrainIntoModel(model, hf, cap.bodies_per_env);
        result.terrain = std::move(hf);
    }

    // Multi-geom collidable proxies: each EXTRA link collision primitive becomes
    // its own appended collidable body row. Reuses the whole broadphase ->
    // narrowphase -> assembly (a proxy resolves to its owner link via body_to_link)
    // + SyncLinkBodyPose's proxy-pose pass. Empty for single-geom links -> the
    // block is skipped and every existing scene is byte-identical.
    if (!link_proxies.empty()) {
        const uint32_t base = cap.bodies_per_env;   // rows before proxies.
        const uint32_t proxy_count = static_cast<uint32_t>(link_proxies.size());
        const uint32_t final_bodies = base + proxy_count;

        // Grow every per-body table to the final row count. body_collidable_link/
        // local default to ~0u/identity for the pre-proxy rows (not proxies).
        model.body_collidable_link.assign(final_bodies, ~uint32_t(0));
        model.body_collidable_local.assign(final_bodies, math::Transform::Identity());
        model.body_to_link.resize(final_bodies, ~uint32_t(0));
        model.body_to_articulation.resize(final_bodies, ~uint32_t(0));
        // Preserve any TRAILING static shape rows (a non-terrain scene appends the
        // ground plane at index bodies_per_env, one past the leaves); the proxies
        // take the new leaf slots [base, base+proxy_count) and the static rows
        // shift after them (terrain scenes have none here -> a no-op).
        std::vector<nk::Model::PairDrivenShape> trailing_static;
        if (model.shape_table_rows.size() > base) {
            trailing_static.assign(model.shape_table_rows.begin() + base,
                                   model.shape_table_rows.end());
        }
        model.shape_table_rows.resize(final_bodies);
        if (!model.body_init.empty()) {
            model.body_init.resize(final_bodies, nk::Model::BodyInit{});
        }
        if (!model.body_material_bucket.empty()) {
            model.body_material_bucket.resize(final_bodies, 0u);
        }

        // Snapshot the pre-proxy exclusion partners of a body row (inheritance).
        const std::vector<uint64_t> base_excludes = model.excluded_pairs;
        auto pair_key = [](uint32_t a, uint32_t b) {
            const uint32_t lo = a < b ? a : b, hi = a < b ? b : a;
            return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
        };
        auto add_exclude = [&](uint32_t a, uint32_t b) {
            model.excluded_pairs.push_back(pair_key(a, b));
        };
        std::vector<uint64_t> base_sorted = base_excludes;
        std::sort(base_sorted.begin(), base_sorted.end());

        for (uint32_t i = 0; i < proxy_count; ++i) {
            const ProxyCollidableSpec& px = link_proxies[i];
            const uint32_t row = base + i;
            const nk::ModelShape& sh = model.shapes[px.shape_row];

            // shape_table row for the proxy collidable (primitive from the cook).
            nk::Model::PairDrivenShape prow;
            prow.kind = sh.kind;
            prow.params[0] = sh.radius;
            prow.params[1] = sh.half_height;
            prow.params[2] = sh.half_extents.z;
            if (sh.kind == static_cast<uint8_t>(ShapeType::Box)) {
                prow.params[0] = sh.half_extents.x;
                prow.params[1] = sh.half_extents.y;
                prow.params[2] = sh.half_extents.z;
            }
            prow.contype = px.shape_row < blob.contact_params.contypes.size()
                               ? blob.contact_params.contypes[px.shape_row] : 1u;
            prow.conaffinity = px.shape_row < blob.contact_params.conaffinities.size()
                                   ? blob.contact_params.conaffinities[px.shape_row] : 1u;
            prow.sdf_grid = ~0u;
            prow.body_id = static_cast<int32_t>(row);
            prow.group = 0u;
            model.shape_table_rows[row] = prow;

            // Inverse maps (reaction -> owner link) + pose binding + material + freeze.
            model.body_to_link[row] = px.owner_link;
            model.body_to_articulation[row] = px.owner_artic;
            model.body_collidable_link[row] = px.owner_link;
            model.body_collidable_local[row] = sh.local_transform;
            if (row < model.body_material_bucket.size()) {
                model.body_material_bucket[row] = sh.material_bucket;
            }
            if (row < model.body_init.size()) {
                model.body_init[row].inv_mass = 0.0f;
                model.body_init[row].inv_inertia = math::Vec3::Zero();
            }

            // Exclusions: never contact the owner body, its excluded partners, a
            // same-link sibling, or a proxy whose owner bodies are excluded partners.
            add_exclude(row, px.owner_body);
            for (uint64_t key : base_excludes) {
                const uint32_t lo = static_cast<uint32_t>(key >> 32);
                const uint32_t hi = static_cast<uint32_t>(key & 0xFFFFFFFFu);
                if (lo == px.owner_body) add_exclude(row, hi);
                else if (hi == px.owner_body) add_exclude(row, lo);
            }
            for (uint32_t j = 0; j < i; ++j) {
                const ProxyCollidableSpec& pj = link_proxies[j];
                if (pj.owner_link == px.owner_link ||
                    std::binary_search(base_sorted.begin(), base_sorted.end(),
                                       pair_key(px.owner_body, pj.owner_body))) {
                    add_exclude(row, base + j);
                }
            }
        }

        std::sort(model.excluded_pairs.begin(), model.excluded_pairs.end());
        model.excluded_pairs.erase(
            std::unique(model.excluded_pairs.begin(), model.excluded_pairs.end()),
            model.excluded_pairs.end());

        // Re-append the preserved trailing static rows AFTER the proxy leaves, so a
        // non-terrain scene's ground plane keeps its one-past-the-leaves position.
        for (const auto& t : trailing_static) model.shape_table_rows.push_back(t);

        // Grow the capacities: the new leaves join the LBVH + candidate budget.
        // Body<->body only; a coupled world's disjoint body<->particle reserve is
        // sized separately (multi-geom feet do not co-occur with particles today).
        cap.bodies_per_env = final_bodies;
        cap.max_bodies_total = static_cast<uint32_t>(model.shape_table_rows.size());
        cap.max_excluded_pairs = static_cast<uint32_t>(model.excluded_pairs.size());
        cap.max_contacts_per_env += proxy_count * kCandidatePairsPerCollidable;
        cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    }

    return result;
}

}  // namespace

CookToModelResult CookToModel(const SceneIR& scene, int env_count) {
    // the FUSED runtime is deleted, so the 2-arg cook now produces a fully
    // general (PairDriven) model — it delegates to the general overload so the
    // contact_family / dof maps / per-candidate-slot row budget are all consistent
    // (a bare CookSceneOptions{} cook would leave a PairDriven-family model in a
    // half-configured state). PairDriven is the default CookToModelOptions.
    return CookToModel(scene, env_count, CookToModelOptions{});
}

// B1 (general contact pipeline (PairDriven)): the PairDriven (general) cook overload.
// It runs the SAME cook (so the registry / shape_table / body_to_link / static
// ground row / multi-artic foundation are all reused verbatim) then flips the model
// to the GENERAL family: contact_family = PairDriven (the pipeline then routes the
// ONE LBVH -> cvx -> mixed-island contact path), enables cross-env filtering so
// candidate pairs stay env-local, and RE-SIZES the per-env row budget to the
// general per-candidate-slot layout (max_contacts_per_env candidate slots x
// kPairDrivenRowsPerSlot rows/slot). : the FUSED runtime is deleted, so this is
// the ONLY general cook path — the options.contact_family no longer selects FUSED.
CookToModelResult CookToModel(const SceneIR& scene, int env_count,
                              const CookToModelOptions& options) {
    // options.contact_family no longer selects FUSED; the cook is always
    // general (PairDriven). options.enable_contacts is threaded to CookToModelImpl
    // (a contacts-OFF cook zeroes the contact budget for dynamics-only worlds).
    // the GENERAL (PairDriven) cvx narrowphase consumes NEITHER the
    // sparse SDF (OpNarrowphaseSdf is a no-op for every family) NOR V-HACD's
    // N-piece decomposition (it wants ONE convex hull per mesh). Skipping both
    // makes the whole-body H1 union scene cook tractable (>178s -> seconds), and
    // is a no-op for primitive-only scenes (go2: no mesh, no SDF).
    CookSceneOptions cook_options;
    cook_options.bake_sdf = false;
    cook_options.general_single_hull = true;
    // Per-body visual-mesh SDF (tight silhouette contact); a no-op for primitive-
    // only scenes (no visual trimesh to bake), so it leaves their cook unchanged.
    cook_options.bake_link_sdf = options.bake_link_sdf;
    CookToModelResult result =
        CookToModelImpl(scene, env_count, cook_options, options.enable_contacts);
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
    // articulation's link slice [0, links_per_articulation)); the multi-tile solver
    // applies it per co-resident articulation with that artic's link offset. Mirrors
    // union_nk_model.cpp's DofIndexOf^-1 (FloatingBase root -> 6 components, scalar
    // joint -> component ~0u). Only the FIRST articulation's links since every
    // co-resident articulation is the same template (dofs_per_env is per-artic DOF).
    {
        namespace articulation = runtime::articulation;
        const nk::ModelArticulation& m = model.articulation;
        const uint32_t k = model.capacities.articulations_per_env == 0u
                               ? 1u : model.capacities.articulations_per_env;
        const uint32_t links_per_articulation =
            (k > 0u && m.link_count >= k) ? (m.link_count / k) : m.link_count;
        model.dof_to_link.clear();
        model.dof_to_component.clear();
        for (uint32_t l = 0; l < links_per_articulation && l < m.joint_type.size(); ++l) {
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
// The ONE cook-time heightfield bake. See cook_to_model.hpp for the contract.
// The HeightField is the single source: obs/spawn/render sample its grid via the
// bilinear sampler; here the same grid is staged for the per-cell prism kernel.
// ---------------------------------------------------------------------------
uint32_t CookHeightfieldGrid(nk::Model& model,
                             const ::nuka::terrain::HeightField& hf) {
    if (!::nuka::terrain::IsValid(hf)) return ~0u;
    if (hf.cell_x != hf.cell_y) return ~0u;  // device locator is single-celled.

    const uint32_t nrow = hf.nrow, ncol = hf.ncol;
    const float cell = hf.cell_x;
    const float half_x = 0.5f * static_cast<float>(ncol - 1u) * cell;
    const float half_y = 0.5f * static_cast<float>(nrow - 1u) * cell;
    // The collidable body sits at the grid centre + base_z; the grid is laid out
    // in its LOCAL frame ([-half,+half] in XY, [0,scale_z] in Z). World z of a
    // corner = base_z + value*scale_z (the HeightField definition).
    const math::Vec3 center{hf.origin.x + half_x, hf.origin.y + half_y, hf.base_z};

    const uint32_t cells = nrow * ncol;
    const uint32_t base = static_cast<uint32_t>(model.heightfield_heights.size());
    model.heightfield_heights.reserve(base + cells);
    for (uint32_t i = 0; i < cells; ++i) {
        model.heightfield_heights.push_back(hf.values[i]);
    }

    nk::HeightfieldData hfd;
    hfd.origin = math::Vec3{-half_x, -half_y, 0.0f};  // local (0,0) corner.
    hfd.cell_size = cell;
    hfd.nrow = nrow;
    hfd.ncol = ncol;
    hfd.min_z = 0.0f;
    hfd.max_z = hf.scale_z;       // local z-range == the value-1 elevation.
    hfd.data_offset = base;
    model.heightfields.push_back(hfd);
    model.capacities.max_heightfield_cells =
        static_cast<uint32_t>(model.heightfield_heights.size());

    // Register the STATIC heightfield collidable: a body_init row (immovable) at
    // the grid centre + a shape_table row (kind=Heightfield, body_id==-1 static,
    // no reaction side). The AABB reads p[0..3] = half_x, half_y, min_z, max_z.
    const uint32_t body_row = static_cast<uint32_t>(model.body_init.size());
    nk::Model::BodyInit bi;
    bi.pose = math::Transform::Identity();
    bi.pose.position = center;
    bi.inv_mass = 0.0f;
    bi.inv_inertia = math::Vec3{0, 0, 0};
    model.body_init.push_back(bi);

    nk::Model::PairDrivenShape sh;
    sh.kind = ::nuka::collision::kShapeHeightfield;
    sh.params[0] = half_x;
    sh.params[1] = half_y;
    sh.params[2] = 0.0f;
    sh.params[3] = hf.scale_z;
    sh.contype = 1u;
    sh.conaffinity = 1u;
    sh.sdf_grid = ~0u;
    sh.body_id = -1;
    sh.group = 0u;
    if (model.shape_table_rows.size() <= body_row) {
        model.shape_table_rows.resize(body_row + 1u);
    }
    model.shape_table_rows[body_row] = sh;
    return body_row;
}

// The ONE terrain bake (see cook_to_model.hpp). Stages the heightfield collidable
// over the static ground-plane row at `orig_bodies` and recomputes the contact budget.
uint32_t CookTerrainIntoModel(nk::Model& model,
                              const ::nuka::terrain::HeightField& hf,
                              uint32_t orig_bodies) {
    // The heightfield seeds at orig_bodies (it overwrites the static ground-plane
    // shape row appended there, keeping the collidable count unchanged).
    model.body_init.resize(orig_bodies);
    const uint32_t body_row = CookHeightfieldGrid(model, hf);

    nk::ModelCapacities& cap = model.capacities;
    // bodies_per_env (the LBVH leaf count) now includes the static terrain collidable;
    // the CONTACT budget is the dynamic collidables (+1 static) at the shared
    // per-collidable slot rate -- the SAME rule the body cook uses.
    cap.bodies_per_env = static_cast<uint32_t>(model.body_init.size());
    cap.max_bodies_total = static_cast<uint32_t>(model.shape_table_rows.size());
    constexpr uint32_t kStaticCollidables = 1u;
    const uint32_t rigid_base =
        (orig_bodies + kStaticCollidables) * kCandidatePairsPerCollidable;
    cap.max_contacts_per_env = rigid_base;
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    // A coupled world (terrain AND particles) keeps the body<->particle slot reserve
    // disjoint above the rigid base -- no-op when particles_per_env == 0.
    GrowContactBudgetForParticles(cap, rigid_base);
    return body_row;
}

// ---------------------------------------------------------------------------
// particle cook.
// ---------------------------------------------------------------------------

void CookXpbdParticles(nk::Model& model, uint32_t env_count,
                       const XpbdCookInput& in) {
    const uint32_t envs = env_count > 0 ? env_count : 1u;
    nk::ModelCapacities& cap = model.capacities;
    if (cap.env_count == 1u || cap.env_count == 0u) cap.env_count = envs;

    nk::Model::ModelParticles& mp = model.particles;
    mp.mode = nk::Model::ParticleMode::Xpbd;
    mp.soft_friction = in.friction;  // body<->soft contact mu (solmix=max).
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

    // SHAPE-MATCH (id 9): flatten the variable-size clusters into the CSR
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

    // Cloth AERODYNAMIC-DRAG surface triangles: store the 3 indices + the cooked
    // rest area (||(b-a)x(c-a)||/2 from the rest positions). The coeffs default 0
    // (drag off); only a caller that opts in makes the drag op live.
    mp.aero_tri_verts.clear();
    mp.aero_tri_area.clear();
    const uint32_t an = static_cast<uint32_t>(in.aero_triangles.size());
    mp.aero_tri_verts.reserve(static_cast<size_t>(an) * 3u);
    mp.aero_tri_area.reserve(an);
    for (uint32_t t = 0; t < an; ++t) {
        const std::array<uint32_t, 3>& tri = in.aero_triangles[t];
        for (uint32_t j = 0; j < 3u; ++j) mp.aero_tri_verts.push_back(tri[j]);
        float area = 0.0f;
        if (tri[0] < in.positions.size() && tri[1] < in.positions.size() &&
            tri[2] < in.positions.size()) {
            const math::Vec3 e1 = in.positions[tri[1]] - in.positions[tri[0]];
            const math::Vec3 e2 = in.positions[tri[2]] - in.positions[tri[0]];
            area = 0.5f * e1.Cross(e2).Length();
        }
        mp.aero_tri_area.push_back(area);
    }
    mp.aero_drag_normal  = in.aero_drag_normal;
    mp.aero_drag_tangent = in.aero_drag_tangent;
    mp.aero_drag_max_dv  = in.aero_drag_max_dv;

    const uint32_t rigid_base = cap.max_contacts_per_env;
    cap.particles_per_env = static_cast<uint32_t>(mp.initial_pos.size());
    cap.dist_cons_per_env = dn;
    cap.bend_cons_per_env = bn;
    cap.vol_cons_per_env  = vn;
    cap.shape_match_slots_per_env   = scn;
    cap.shape_match_members_per_env =
        static_cast<uint32_t>(mp.sm_particles.size());
    cap.aero_tris_per_env = an;
    // Reserve a disjoint body<->particle slot sub-range above the rigid budget
    // (no-op when there are no body contacts -> particle-only cooks byte-identical).
    GrowContactBudgetForParticles(cap, rigid_base);
    // n_soft_particles is left at its default (0); it is only consulted for the
    // SoftFluid mode (CookSoftFluidParticles sets it). The single-system Xpbd
    // ops ignore it (mode-gated), so the device-staged bytes are unaffected.
}

void CookMpmParticles(nk::Model& model, uint32_t env_count,
                      const MpmCookInput& in) {
    const uint32_t envs = env_count > 0 ? env_count : 1u;
    nk::ModelCapacities& cap = model.capacities;
    if (cap.env_count == 1u || cap.env_count == 0u) cap.env_count = envs;
    if (in.dx <= 0.0f || in.grid_dims[0] == 0u || in.grid_dims[1] == 0u ||
        in.grid_dims[2] == 0u) {
        throw std::runtime_error(
            "CookMpmParticles: an MLS-MPM medium needs a positive cell size dx and "
            "non-zero grid_dims (the env-private background grid)");
    }

    nk::Model::ModelParticles& mp = model.particles;
    mp.mode = nk::Model::ParticleMode::Mpm;
    mp.initial_pos = in.positions;
    mp.initial_vel = in.velocities;
    mp.inv_mass = in.inv_mass;
    if (mp.initial_vel.size() != mp.initial_pos.size()) {
        mp.initial_vel.assign(mp.initial_pos.size(), math::Vec3::Zero());
    }
    if (mp.inv_mass.size() != mp.initial_pos.size()) {
        mp.inv_mass.assign(mp.initial_pos.size(), 1.0f);
    }
    // F seeded to identity per particle; C is the arena zero default.
    const size_t n = mp.initial_pos.size();
    mp.initial_F.assign(n * 9u, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        mp.initial_F[i * 9u + 0u] = 1.0f;
        mp.initial_F[i * 9u + 4u] = 1.0f;
        mp.initial_F[i * 9u + 8u] = 1.0f;
    }
    mp.initial_vol0 = in.vol0;
    if (mp.initial_vol0.size() != n) mp.initial_vol0.assign(n, in.dx * in.dx * in.dx);

    // Convert one cook material row -> the model POD; reject kind 1 loudly (granular
    // Drucker-Prager would silently run as elastic). The F-update branch reads it later.
    auto to_material = [](const MpmMaterialInput& mi) {
        if (mi.model_kind > 0.5f && mi.model_kind < 1.5f) {
            throw std::runtime_error(
                "CookMpmParticles: granular Drucker-Prager MPM (model_kind == 1) is "
                "not yet implemented");
        }
        nk::MpmMaterial m;
        m.youngs = mi.youngs; m.poisson = mi.poisson; m.density = mi.density;
        m.dp_friction = mi.dp_friction; m.dp_cohesion = mi.dp_cohesion;
        m.model_kind = mi.model_kind; m.bulk_modulus = mi.bulk_modulus;
        m.tait_gamma = mi.tait_gamma; m.viscosity = mi.viscosity;
        return m;
    };
    if (in.materials.empty()) {
        // Homogeneous cook: one material, id 0 for every particle.
        mp.initial_material_id.assign(n, 0u);
        model.mpm_materials = {to_material(in.material)};
        cap.mpm_material_count = 1u;
    } else {
        // Heterogeneous cook: N material rows + the per-particle index into them.
        model.mpm_materials.clear();
        model.mpm_materials.reserve(in.materials.size());
        for (const MpmMaterialInput& mi : in.materials) {
            model.mpm_materials.push_back(to_material(mi));
        }
        cap.mpm_material_count = static_cast<uint32_t>(in.materials.size());
        if (in.material_id.size() == n) mp.initial_material_id = in.material_id;
        else mp.initial_material_id.assign(n, 0u);
    }

    // Env-private dense grid sizing (the node product, loud u32 overflow guard).
    const uint64_t nodes64 = static_cast<uint64_t>(in.grid_dims[0]) *
                             in.grid_dims[1] * in.grid_dims[2];
    if (nodes64 > 0xFFFFFFFFull) {
        throw std::runtime_error(
            "CookMpmParticles: the MPM grid node count (dims product) overflows u32");
    }
    cap.mpm_grid_nodes_per_env = static_cast<uint32_t>(nodes64);
    mp.mpm_grid_min = in.grid_origin;
    mp.mpm_grid_dims[0] = in.grid_dims[0];
    mp.mpm_grid_dims[1] = in.grid_dims[1];
    mp.mpm_grid_dims[2] = in.grid_dims[2];
    mp.mpm_cell_size = in.dx;
    mp.mpm_substeps = in.substeps == 0u ? 1u : in.substeps;
    // Store a unit floor normal (the grid BC reads it as unit) + the plane offset.
    const math::Vec3 fn = in.floor_normal;
    const float nlen = fn.Length();
    mp.mpm_floor_normal = nlen > 1e-6f ? fn * (1.0f / nlen) : math::Vec3{0.0f, 0.0f, 1.0f};
    mp.mpm_floor_d = in.floor_d;
    mp.mpm_floor_friction = in.floor_friction;

    const uint32_t rigid_base = cap.max_contacts_per_env;
    cap.particles_per_env = static_cast<uint32_t>(n);
    // Reuse the SAME disjoint body<->particle slot reserve as the XPBD cook so
    // coupling stays one-path (no-op when there are no body contacts).
    GrowContactBudgetForParticles(cap, rigid_base);
}

void CookSoftBodyParticles(nk::Model& model, uint32_t env_count,
                           const XpbdCookInput& in, const MpmCookInput& mpm) {
    if (in.solver == nk::Model::ParticleMode::Xpbd) {
        CookXpbdParticles(model, env_count, in);
        return;
    }
    if (in.solver == nk::Model::ParticleMode::Mpm) {
        CookMpmParticles(model, env_count, mpm);
        return;
    }
    throw std::runtime_error(
        "CookSoftBodyParticles: a bulk-soft body solver must be Xpbd or Mpm");
}

void ValidateMedia(const std::vector<MediaRecord>& media) {
    using Kind = MediaRecord::Kind;
    using Method = MediaRecord::Method;
    uint32_t n_mpm = 0u, n_non_mpm = 0u, n_pbf_fluid = 0u;
    for (const MediaRecord& m : media) {
        bool legal = false;
        switch (m.kind) {
            case Kind::Cloth:   legal = m.method == Method::Xpbd; break;
            case Kind::SoftTet: legal = m.method == Method::Xpbd ||
                                        m.method == Method::MlsMpm; break;
            case Kind::Fluid:   legal = m.method == Method::Pbf ||
                                        m.method == Method::MlsMpm; break;
            case Kind::Granular: legal = m.method == Method::MlsMpm; break;
            case Kind::Cable:   legal = m.method == Method::Xpbd; break;
        }
        if (!legal) {
            throw std::runtime_error(
                "ValidateMedia: illegal medium (kind x method) -- cloth must solve "
                "with XPBD, a tet-soft body with XPBD or MLS-MPM, a fluid with PBF or "
                "MLS-MPM, a granular bed with MLS-MPM only, a cable with XPBD only");
        }
        // model_kind 4 (Drucker-Prager) and Kind::Granular imply each other: no
        // sand-tagged fluid/soft body and no granular bed cooking as another model.
        const float mk = m.mpm.model_kind;
        const bool dp_kind = mk > 3.5f && mk < 4.5f;
        if (dp_kind && m.kind != Kind::Granular) {
            throw std::runtime_error(
                "ValidateMedia: mpm model_kind 4 (Drucker-Prager) requires a "
                "Granular medium");
        }
        if (m.kind == Kind::Granular && mk != 0.0f && !dp_kind) {
            throw std::runtime_error(
                "ValidateMedia: a Granular medium is Drucker-Prager (mpm model_kind "
                "4 or unset); it cannot declare another constitutive");
        }
        // Heterogeneous fills append box sub-regions with per-fill materials; they are
        // meaningful only for a box-sampled MLS-MPM medium (fluid / granular).
        if (!m.mpm_fills.empty() &&
            !(m.method == Method::MlsMpm &&
              (m.kind == Kind::Fluid || m.kind == Kind::Granular))) {
            throw std::runtime_error(
                "ValidateMedia: mpm_fills require a box MLS-MPM medium (fluid or granular)");
        }
        if (m.method == Method::MlsMpm) ++n_mpm; else ++n_non_mpm;
        if (m.kind == Kind::Fluid && m.method == Method::Pbf) ++n_pbf_fluid;
    }
    // An MLS-MPM medium may co-reside with XPBD (cloth/tet) media in the MpmXpbd
    // [mpm | xpbd] layout, but NOT with a PBF fluid: MPM owns the env-private grid
    // and PBF its own neighbor grid, so the two grids cannot share one Model.
    if (n_mpm > 0u && n_pbf_fluid > 0u) {
        throw std::runtime_error(
            "ValidateMedia: an MLS-MPM medium may not co-reside with a PBF fluid in "
            "one Model (each owns a background grid); pair MPM with XPBD media only");
    }
    if (n_pbf_fluid > 1u) {
        throw std::runtime_error(
            "ValidateMedia: a Model carries one PBF fluid slice; a second PBF fluid "
            "medium is not representable");
    }
    if (n_mpm > 1u) {
        throw std::runtime_error(
            "ValidateMedia: a Model carries one MLS-MPM medium; a second is not "
            "representable");
    }
}

void CookPbfParticles(nk::Model& model, uint32_t env_count,
                      const PbfCookInput& in) {
    const uint32_t envs = env_count > 0 ? env_count : 1u;
    nk::ModelCapacities& cap = model.capacities;
    if (cap.env_count == 1u || cap.env_count == 0u) cap.env_count = envs;

    nk::Model::ModelParticles& mp = model.particles;
    mp.mode = nk::Model::ParticleMode::Pbf;
    mp.fluid_friction = in.friction;  // body<->fluid contact mu (solmix=max).
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

    const uint32_t rigid_base = cap.max_contacts_per_env;
    cap.particles_per_env = static_cast<uint32_t>(mp.initial_pos.size());
    // Per-env uniform-grid cell capacity (sizes grid_cell_start/end; the
    // ParticleGridBuild op fails loudly if the live dims exceed it).
    cap.max_grid_cells = in.grid_dims[0] * in.grid_dims[1] * in.grid_dims[2];
    // Reserve a disjoint body<->particle slot sub-range above the rigid budget
    // (no-op when there are no body contacts -> particle-only cooks byte-identical).
    GrowContactBudgetForParticles(cap, rigid_base);
}

// Wire the body<->particle contact radius + cross-system non-penetration co-step
// onto a cooked particle set. The body<->particle narrowphase derives its sphere
// radius from pp_contact_d_min, so this must run for a single-medium coupled world
// too or the robot tunnels through the medium. The neighbor grid is built over
// query_radius, so widen cell/query to >= d_min so it covers every penetrating pair
// (a no-op when d_min <= the cooked grid radius). The cross-system co-step is mode-
// gated to SoftFluid downstream, so a single-medium cook carries it inert.
static void ApplyParticleBodyContact(nk::Model::ModelParticles& mp,
                                     const SoftFluidContactInput& contact) {
    mp.pp_contact_d_min      = contact.contact_d_min;
    mp.pp_contact_compliance = contact.compliance_alpha;
    mp.pp_contact_iters = contact.solver_iterations == 0u ? 1u
                                                          : contact.solver_iterations;
    if (mp.pp_contact_d_min > 0.0f && mp.pp_contact_d_min > mp.query_radius) {
        mp.cell_size    = mp.pp_contact_d_min;
        mp.query_radius = mp.pp_contact_d_min;
    }
}

// ---------------------------------------------------------------------------
// Two-system cook: soft (XPBD) + fluid (PBF) co-resident in ONE Model.
// ---------------------------------------------------------------------------

void CookSoftFluidParticles(nk::Model& model, uint32_t env_count,
                            const XpbdCookInput& soft, const PbfCookInput& fluid,
                            const SoftFluidContactInput& contact) {
    const uint32_t envs = env_count > 0 ? env_count : 1u;
    nk::ModelCapacities& cap = model.capacities;
    if (cap.env_count == 1u || cap.env_count == 0u) cap.env_count = envs;
    // The body<->body budget before any particle growth; the final reserve recomputes
    // from this for the full [soft|fluid] count so the soft slice is not double-counted.
    const uint32_t rigid_base = cap.max_contacts_per_env;

    // STRICT-SUPERSET FAST PATHS: a soft-only / fluid-only co-residence cook is
    // byte-identical to the single-system cook PLUS the same body<->particle contact
    // setup the composite path applies (so a single-medium coupled world couples).
    // Only when BOTH sides are present do we build the [soft | fluid] composite + set
    // the SoftFluid mode; the cross-system particle-particle co-step is mode-gated to
    // SoftFluid downstream, so a single-system fast path carries it inert.
    const uint32_t n_soft = static_cast<uint32_t>(soft.positions.size());
    const uint32_t n_fluid = static_cast<uint32_t>(fluid.positions.size());
    if (n_fluid == 0u) {
        // Soft-only: the canonical XPBD cook + the body<->particle contact setup so a
        // robot+cloth-only world couples (incl. shape-match).
        CookXpbdParticles(model, env_count, soft);
        ApplyParticleBodyContact(model.particles, contact);
        return;
    }
    if (n_soft == 0u) {
        // Fluid-only: the canonical PBF cook + the body<->particle contact setup so a
        // robot+fluid-only world couples.
        CookPbfParticles(model, env_count, fluid);
        ApplyParticleBodyContact(model.particles, contact);
        return;
    }

    // 1) Cook the SOFT set first (fills the XPBD + shape-match templates, sets
    // particles_per_env = n_soft, mode = Xpbd). The soft constraint indices
    // already point into [0, n_soft) -- exactly where the soft particles land.
    CookXpbdParticles(model, env_count, soft);

    nk::Model::ModelParticles& mp = model.particles;

    // 2) APPEND the fluid particles AFTER the soft set ([soft | fluid] layout).
    // The fluid particles are NOT referenced by any soft constraint, so no
    // index remap is needed; the SoftFluid PBF ops scope the density solve to
    // the fluid slice [n_soft, n_soft+n_fluid) by the n_soft split.
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
    // The soft slice mu was set by the CookXpbdParticles call above; the fluid
    // slice carries its own (body<->particle contact, solmix=max with the body).
    mp.fluid_friction     = fluid.friction;
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

    // 4) The body<->particle contact radius + the cross-system non-penetration
    // co-step over the FULL union (the same setup the single-medium fast paths
    // apply, so the body<->particle radius and grid coverage are uniform).
    ApplyParticleBodyContact(mp, contact);

    // 5) The co-residence schema: mode + split index + the new total particle
    // count. The grid is sized over the FULL union per-env (the soft particles
    // occupy grid cells too, but the fluid density solve skips them via n_soft).
    mp.mode = nk::Model::ParticleMode::SoftFluid;
    mp.n_soft_particles = n_soft;
    cap.particles_per_env = n_soft + n_fluid;
    cap.max_grid_cells =
        fluid.grid_dims[0] * fluid.grid_dims[1] * fluid.grid_dims[2];
    // Reserve a disjoint body<->particle slot sub-range above the rigid budget for
    // the FULL union (recomputed from rigid_base, overriding the inner soft growth).
    GrowContactBudgetForParticles(cap, rigid_base);
}

// ---------------------------------------------------------------------------
// Two-system cook: MLS-MPM (bulk/granular/fluid) + XPBD (cloth/tet) co-resident in
// ONE Model with a contiguous [mpm | xpbd] layout. MPM occupies [0, n_mpm), the
// XPBD set [n_mpm, particles_per_env). MpmStep scopes to the MPM slice; the XPBD
// predict/project/finalize + the body<->particle rows to the XPBD slice.
// ---------------------------------------------------------------------------

void CookMpmXpbd(nk::Model& model, uint32_t env_count, const MpmCookInput& mpm,
                 const XpbdCookInput& soft, const SoftFluidContactInput& contact) {
    const uint32_t envs = env_count > 0 ? env_count : 1u;
    nk::ModelCapacities& cap = model.capacities;
    if (cap.env_count == 1u || cap.env_count == 0u) cap.env_count = envs;
    // The body<->body budget before any particle growth; the final reserve recomputes
    // from this for the full [mpm|xpbd] count so no slice is double-counted.
    const uint32_t rigid_base = cap.max_contacts_per_env;

    const uint32_t n_mpm = static_cast<uint32_t>(mpm.positions.size());
    const uint32_t n_xpbd = static_cast<uint32_t>(soft.positions.size());
    // STRICT-SUPERSET fast paths: a one-medium co-residence cook is byte-identical to
    // the single-medium cook plus the same body<->particle contact setup.
    if (n_xpbd == 0u) {
        CookMpmParticles(model, env_count, mpm);
        ApplyParticleBodyContact(model.particles, contact);
        return;
    }
    if (n_mpm == 0u) {
        CookXpbdParticles(model, env_count, soft);
        ApplyParticleBodyContact(model.particles, contact);
        return;
    }

    // 1) Cook the MPM set: it lands in [0, n_mpm) with F/vol0/material_id, the
    // env-private grid, and the material table (mode = Mpm, particles_per_env = n_mpm).
    CookMpmParticles(model, env_count, mpm);
    nk::Model::ModelParticles& mp = model.particles;

    // 2) Cook the XPBD set into a scratch Model (reuses the de-interleave / shape-
    // match flatten / aero-area cookers verbatim), then APPEND its particles and its
    // constraint SoA with every particle index REBASED by n_mpm (the cloth verts land
    // at [n_mpm, P)). The MPM continuum fields stay sized to the MPM slice; the seed's
    // identity/0 fallback fills the XPBD slice (never read by the transfer loop).
    nk::Model xtmp;
    CookXpbdParticles(xtmp, env_count, soft);
    const nk::Model::ModelParticles& xp = xtmp.particles;
    mp.initial_pos.insert(mp.initial_pos.end(), xp.initial_pos.begin(),
                          xp.initial_pos.end());
    mp.initial_vel.insert(mp.initial_vel.end(), xp.initial_vel.begin(),
                          xp.initial_vel.end());
    mp.inv_mass.insert(mp.inv_mass.end(), xp.inv_mass.begin(), xp.inv_mass.end());
    for (uint32_t v : xp.dist_a) mp.dist_a.push_back(v + n_mpm);
    for (uint32_t v : xp.dist_b) mp.dist_b.push_back(v + n_mpm);
    mp.dist_rest = xp.dist_rest;
    mp.dist_alpha = xp.dist_alpha;
    for (uint32_t v : xp.bend_particles) mp.bend_particles.push_back(v + n_mpm);
    mp.bend_gradients = xp.bend_gradients;
    mp.bend_alpha = xp.bend_alpha;
    for (uint32_t v : xp.vol_particles) mp.vol_particles.push_back(v + n_mpm);
    mp.vol_rest6 = xp.vol_rest6;
    mp.vol_alpha = xp.vol_alpha;
    // Shape-match: the CSR offsets/sizes/centroids are pool-local (the MPM slice has
    // none, so the appended pool starts at 0); only the member particle indices rebase.
    mp.sm_cluster_offset = xp.sm_cluster_offset;
    mp.sm_cluster_size = xp.sm_cluster_size;
    mp.sm_stiffness = xp.sm_stiffness;
    mp.sm_rest_centroid = xp.sm_rest_centroid;
    for (uint32_t v : xp.sm_particles) mp.sm_particles.push_back(v + n_mpm);
    mp.sm_rest_q = xp.sm_rest_q;
    mp.sm_mass = xp.sm_mass;
    for (uint32_t v : xp.aero_tri_verts) mp.aero_tri_verts.push_back(v + n_mpm);
    mp.aero_tri_area = xp.aero_tri_area;
    mp.aero_drag_normal = xp.aero_drag_normal;
    mp.aero_drag_tangent = xp.aero_drag_tangent;
    mp.aero_drag_max_dv = xp.aero_drag_max_dv;
    mp.xpbd_iters = xp.xpbd_iters;
    mp.soft_friction = xp.soft_friction;

    // 3) The body<->particle cloth contact radius + the co-residence schema.
    ApplyParticleBodyContact(mp, contact);
    mp.mode = nk::Model::ParticleMode::MpmXpbd;
    mp.n_mpm_particles = n_mpm;
    cap.particles_per_env = n_mpm + n_xpbd;
    cap.dist_cons_per_env = xtmp.capacities.dist_cons_per_env;
    cap.bend_cons_per_env = xtmp.capacities.bend_cons_per_env;
    cap.vol_cons_per_env  = xtmp.capacities.vol_cons_per_env;
    cap.shape_match_slots_per_env   = xtmp.capacities.shape_match_slots_per_env;
    cap.shape_match_members_per_env = xtmp.capacities.shape_match_members_per_env;
    cap.aero_tris_per_env = xtmp.capacities.aero_tris_per_env;
    // Reserve the disjoint body<->particle slot block over the FULL [mpm|xpbd] count
    // (recomputed from rigid_base, overriding the MPM cook's inner growth).
    GrowContactBudgetForParticles(cap, rigid_base);
}

// ---------------------------------------------------------------------------
// Media records -> particle cook (the per-medium builders + the list dispatch).
// ---------------------------------------------------------------------------

namespace {

// The particle layout of a cable medium: `chain` = segments+1 rope beads; `slab` =
// the 8 rigid-cluster box corners (0 when the slab extents are absent). {0,0} when
// the geometry is absent (segments < 1 or radius <= 0) -- no particles cook then.
struct CableLayout { uint32_t chain = 0u; uint32_t slab = 0u; };

CableLayout CableParticleLayout(const MediaRecord& media) {
    const MediaRecord::CableLine& c = media.cable_line;
    CableLayout out;
    if (c.segments < 1u || c.radius <= 0.0f) return out;
    out.chain = c.segments + 1u;
    const math::Vec3& he = c.slab.half_extents;
    if (he.x > 0.0f && he.y > 0.0f && he.z > 0.0f) out.slab = 8u;
    return out;
}

// The 8 slab box corners (index k = ix | iy<<1 | iz<<2 over {0,1}^3), placed so the
// top face center sits at the cable's loaded end (the box hangs downward from it).
std::array<math::Vec3, 8> CableSlabCorners(const MediaRecord::CableLine& c) {
    const math::Vec3& he = c.slab.half_extents;
    const math::Vec3 center = c.end - math::Vec3{0.0f, 0.0f, he.z};  // top face at end.
    std::array<math::Vec3, 8> corners{};
    for (uint32_t k = 0u; k < 8u; ++k) {
        const float sx = (k & 1u) ? 1.0f : -1.0f;
        const float sy = (k & 2u) ? 1.0f : -1.0f;
        const float sz = (k & 4u) ? 1.0f : -1.0f;
        corners[k] = center + math::Vec3{sx * he.x, sy * he.y, sz * he.z};
    }
    return corners;
}

// The 12 triangles (2 per face, outward CCW winding) of the slab box over its 8
// corners; `base` offsets the corner indices into the global particle pool.
std::vector<uint32_t> CableSlabBoxTriangles(uint32_t base) {
    static const uint32_t faces[6][4] = {
        {0u, 4u, 6u, 2u}, {1u, 3u, 7u, 5u},   // -X, +X
        {0u, 1u, 5u, 4u}, {2u, 6u, 7u, 3u},   // -Y, +Y
        {0u, 2u, 3u, 1u}, {4u, 5u, 7u, 6u},   // -Z, +Z
    };
    std::vector<uint32_t> tris;
    tris.reserve(36u);
    for (const auto& f : faces) {
        tris.insert(tris.end(), {base + f[0], base + f[1], base + f[2]});
        tris.insert(tris.end(), {base + f[0], base + f[2], base + f[3]});
    }
    return tris;
}

}  // namespace

XpbdCookInput BuildClothXpbdInput(const MediaRecord& media) {
    XpbdCookInput in;
    const MediaRecord::ClothGrid& g = media.cloth_grid;
    if (g.nx < 2u || g.ny < 2u || g.spacing <= 0.0f) {
        return in;  // no cloth (an empty soft set is treated as none).
    }
    const uint32_t nx = g.nx, ny = g.ny;
    const float s = g.spacing;
    const float x0 = g.origin.x - 0.5f * static_cast<float>(nx - 1u) * s;
    const float y0 = g.origin.y - 0.5f * static_cast<float>(ny - 1u) * s;
    std::vector<math::Vec3> rest;
    rest.reserve(static_cast<size_t>(nx) * ny);
    for (uint32_t j = 0u; j < ny; ++j) {
        for (uint32_t i = 0u; i < nx; ++i) {
            rest.push_back(math::Vec3{x0 + static_cast<float>(i) * s,
                                      y0 + static_cast<float>(j) * s, g.origin.z});
        }
    }
    auto idx = [nx](uint32_t i, uint32_t j) { return j * nx + i; };
    std::vector<runtime::soft::ClothTriangle> tris;
    for (uint32_t j = 0u; j + 1u < ny; ++j) {
        for (uint32_t i = 0u; i + 1u < nx; ++i) {
            tris.push_back({{idx(i, j), idx(i + 1u, j), idx(i + 1u, j + 1u)}});
            tris.push_back({{idx(i, j), idx(i + 1u, j + 1u), idx(i, j + 1u)}});
        }
    }
    runtime::soft::ClothTopologyOptions opts;
    opts.distance_compliance_alpha = media.xpbd.distance_alpha;
    opts.bend_compliance_alpha = media.xpbd.bend_alpha;
    runtime::soft::XpbdConstraintSet cs;
    runtime::soft::BuildClothConstraints(rest, tris, opts, cs);

    in.positions = rest;
    in.velocities.assign(rest.size(), math::Vec3::Zero());
    const float mass =
        media.xpbd.particle_mass > 0.0f ? media.xpbd.particle_mass : 0.01f;
    in.inv_mass.assign(rest.size(), 1.0f / mass);
    // Pin set: None (a free drape), Perimeter (a taut membrane), or one grid
    // edge (a hung curtain); the FromFree default defers to the legacy flag.
    using Pin = MediaRecord::ClothPin;
    Pin pin = g.pin;
    if (pin == Pin::FromFree) pin = g.free ? Pin::None : Pin::Perimeter;
    const uint32_t last_i = nx - 1u, last_j = ny - 1u;
    if (pin == Pin::Perimeter) {
        for (uint32_t k = 0u; k < nx; ++k) {
            in.inv_mass[idx(k, 0u)] = 0.0f;
            in.inv_mass[idx(k, last_j)] = 0.0f;
        }
        for (uint32_t k = 0u; k < ny; ++k) {
            in.inv_mass[idx(0u, k)] = 0.0f;
            in.inv_mass[idx(last_i, k)] = 0.0f;
        }
    } else if (pin == Pin::EdgeX0 || pin == Pin::EdgeX1) {
        const uint32_t i = pin == Pin::EdgeX0 ? 0u : last_i;
        for (uint32_t k = 0u; k < ny; ++k) in.inv_mass[idx(i, k)] = 0.0f;
    } else if (pin == Pin::EdgeY0 || pin == Pin::EdgeY1) {
        const uint32_t j = pin == Pin::EdgeY0 ? 0u : last_j;
        for (uint32_t k = 0u; k < nx; ++k) in.inv_mass[idx(k, j)] = 0.0f;
    }
    for (const auto& dc : cs.distance) {
        in.distance.push_back(
            {dc.particle_a, dc.particle_b, dc.rest_length, dc.compliance_alpha});
    }
    for (const auto& bc : cs.bend) {
        CookBendCon c;
        for (uint32_t k = 0u; k < 4u; ++k) { c.p[k] = bc.particle[k]; c.k[k] = bc.k[k]; }
        c.compliance_alpha = bc.compliance_alpha;
        in.bend.push_back(c);
    }
    in.solver_iterations =
        static_cast<uint16_t>(media.xpbd.iters != 0u ? media.xpbd.iters : 1u);
    in.friction = media.xpbd.friction;
    // Anisotropic aero drag: pass the coeffs through; seed the op with the cloth
    // triangles only when active (all-zero coeffs => no op => byte-identical cook).
    in.aero_drag_normal = media.xpbd.aero_drag_normal;
    in.aero_drag_tangent = media.xpbd.aero_drag_tangent;
    in.aero_drag_max_dv = media.xpbd.aero_drag_max_dv;
    if (media.xpbd.aero_drag_normal > 0.0f || media.xpbd.aero_drag_tangent > 0.0f) {
        in.aero_triangles.reserve(tris.size());
        for (const auto& t : tris) {
            in.aero_triangles.push_back({t.v[0], t.v[1], t.v[2]});
        }
    }
    return in;
}

std::vector<uint32_t> BuildClothSurfaceTriangles(const MediaRecord& media) {
    std::vector<uint32_t> tris;
    const MediaRecord::ClothGrid& g = media.cloth_grid;
    if (g.nx < 2u || g.ny < 2u || g.spacing <= 0.0f) return tris;
    const uint32_t nx = g.nx, ny = g.ny;
    auto idx = [nx](uint32_t i, uint32_t j) { return j * nx + i; };
    for (uint32_t j = 0u; j + 1u < ny; ++j) {
        for (uint32_t i = 0u; i + 1u < nx; ++i) {
            tris.insert(tris.end(), {idx(i, j), idx(i + 1u, j), idx(i + 1u, j + 1u),
                                     idx(i, j), idx(i + 1u, j + 1u), idx(i, j + 1u)});
        }
    }
    return tris;
}

namespace {
// The ordered box-fill list of a heterogeneous MLS-MPM Fluid/Granular medium: the
// base bed (fluid_box + the medium's material/render id) then each mpm_fill (its own
// material; render id inherits the medium when unset). Both the cook sampler and the
// render-surface builder walk THIS so their per-fill particle layouts agree.
struct MpmBoxFill {
    MediaRecord::FluidBox box;
    MediaMpmMaterial      material;
    uint32_t              render_material_id;
};
std::vector<MpmBoxFill> MpmBoxFills(const MediaRecord& media) {
    std::vector<MpmBoxFill> fills;
    fills.reserve(1u + media.mpm_fills.size());
    fills.push_back({media.fluid_box, media.mpm, media.render_material_id});
    for (const MediaRecord::MpmFill& f : media.mpm_fills) {
        fills.push_back({f.box, f.material,
                         f.render_material_id != kInvalidMaterial
                             ? f.render_material_id : media.render_material_id});
    }
    return fills;
}

// Copy a medium's authored grain-scatter render-skin params onto an instanced-sphere
// skin (all default => uniform octahedra, byte-identical).
void ApplyGrainSkin(MediaRenderSurface& s, const MediaRenderSkin& rs) {
    s.grain_round = rs.grain_round;
    s.grain_radius_jitter = rs.grain_radius_jitter;
    s.grain_tint_jitter = rs.grain_tint_jitter;
}

// One instanced-sphere skin PER fill (base bed then each mpm_fill), each with its own
// radius + render material, over the fill's slice of the MPM particle block. Returns
// the total MPM particle count so the caller advances the running base. Heterogeneous
// media only; a homogeneous medium keeps its single-skin path.
uint32_t AppendMpmFillSurfaces(std::vector<MediaRenderSurface>& out,
                               const MediaRecord& media, uint32_t base) {
    uint32_t first = base;
    for (const MpmBoxFill& f : MpmBoxFills(media)) {
        import::cooker::FluidBoxSpec spec;
        spec.min_corner = f.box.min;
        spec.max_corner = f.box.max;
        spec.spacing = f.box.spacing;
        const uint32_t cnt = import::cooker::FluidBoxLatticeCounts(spec).total;
        if (cnt == 0u) continue;  // empty sub-box: no particles, no skin (cook skips it too).
        MediaRenderSurface s;
        s.particle_radius = 0.5f * f.box.spacing;
        s.particle_first = first;
        s.particle_count = cnt;
        s.render_material_id = f.render_material_id;
        ApplyGrainSkin(s, media.render_skin);
        out.push_back(std::move(s));
        first += cnt;
    }
    return first - base;
}
}  // namespace

std::vector<MediaRenderSurface> BuildSceneMediaRenderSurfaces(
    const std::vector<MediaRecord>& media) {
    std::vector<MediaRenderSurface> surfaces;
    if (media.empty()) return surfaces;
    // A lone MLS-MPM medium is its own ParticleMode (a dense sample, not the lattice
    // vertices), so no boundary-triangle surface indexes its cooked particle set; it
    // renders as instanced particle spheres over the whole field instead.
    if (media.size() == 1u && media.front().method == MediaRecord::Method::MlsMpm) {
        const MediaRecord& m = media.front();
        if (!m.mpm_fills.empty()) {  // heterogeneous bed: one skin per fill.
            AppendMpmFillSurfaces(surfaces, m, 0u);
            return surfaces;
        }
        float sp = 0.0f;
        if (m.kind == MediaRecord::Kind::SoftTet) sp = m.tet_sphere.cell_len;
        else sp = m.fluid_box.spacing;  // Fluid + Granular share the box lattice.
        if (sp > 0.0f) {
            MediaRenderSurface s;
            s.particle_radius = 0.5f * sp;  // half the sampling lattice spacing.
            s.render_material_id = m.render_material_id;
            ApplyGrainSkin(s, m.render_skin);
            surfaces.push_back(std::move(s));
        }
        return surfaces;
    }
    // MpmXpbd: the MLS-MPM medium renders as instanced spheres over its low slice
    // [0, n_mpm); the XPBD (cloth/tet) media then triangulate [n_mpm, P). A lone MPM
    // medium already returned above, so an MPM here means the co-resident layout.
    const MediaRecord* mpm_medium = nullptr;
    for (const MediaRecord& m : media) {
        if (m.method == MediaRecord::Method::MlsMpm) { mpm_medium = &m; break; }
    }
    uint32_t base = 0u;
    if (mpm_medium != nullptr) {
        if (!mpm_medium->mpm_fills.empty()) {  // heterogeneous bed: one skin per fill.
            base = AppendMpmFillSurfaces(surfaces, *mpm_medium, 0u);
        } else {
            const uint32_t n_mpm =
                static_cast<uint32_t>(BuildMpmInput(*mpm_medium).positions.size());
            const float sp = mpm_medium->kind == MediaRecord::Kind::SoftTet
                                 ? mpm_medium->tet_sphere.cell_len
                                 : mpm_medium->fluid_box.spacing;
            if (sp > 0.0f && n_mpm > 0u) {
                MediaRenderSurface s;
                s.particle_radius = 0.5f * sp;
                s.particle_first = 0u;
                s.particle_count = n_mpm;
                s.render_material_id = mpm_medium->render_material_id;
                ApplyGrainSkin(s, mpm_medium->render_skin);
                surfaces.push_back(std::move(s));
            }
            base = n_mpm;  // the XPBD slice starts above the MPM slice.
        }
    }
    // Cloth + soft-tet concatenate into the XPBD slice in media order; track the
    // running particle base exactly as AppendSoftMedium does (fluid/MPM skipped).
    for (const MediaRecord& m : media) {
        if (m.method == MediaRecord::Method::MlsMpm) continue;  // its sphere surface is above.
        // Cable: the rope renders as an instanced bead tube over its chain range; the
        // optional welded slab renders as a triangulated rigid box over its 8 corners.
        if (m.kind == MediaRecord::Kind::Cable) {
            const CableLayout cl = CableParticleLayout(m);
            if (cl.chain > 0u) {
                MediaRenderSurface beads;
                beads.particle_radius = m.cable_line.radius;
                beads.particle_first = base;
                beads.particle_count = cl.chain;
                beads.render_material_id = m.render_material_id;
                ApplyGrainSkin(beads, m.render_skin);
                surfaces.push_back(std::move(beads));
            }
            if (cl.slab == 8u) {
                MediaRenderSurface box;
                box.triangles = CableSlabBoxTriangles(base + cl.chain);
                box.render_material_id =
                    m.cable_line.slab.render_material_id != kInvalidMaterial
                        ? m.cable_line.slab.render_material_id
                        : m.render_material_id;
                surfaces.push_back(std::move(box));
            }
            base += cl.chain + cl.slab;
            continue;
        }
        std::vector<uint32_t> tris;
        uint32_t verts = 0u;
        if (m.kind == MediaRecord::Kind::Cloth) {
            const MediaRecord::ClothGrid& g = m.cloth_grid;
            if (g.nx >= 2u && g.ny >= 2u && g.spacing > 0.0f) {
                tris = BuildClothSurfaceTriangles(m);
                verts = g.nx * g.ny;
            }
        } else if (m.kind == MediaRecord::Kind::SoftTet) {
            const MediaRecord::TetSphere& ts = m.tet_sphere;
            if (ts.radius > 0.0f && ts.cells >= 2u && ts.cell_len > 0.0f) {
                const runtime::soft::TetLattice lat =
                    runtime::soft::BuildSphereTetLattice(
                        math::Vec3{0.0f, 0.0f, 0.0f}, ts.radius, ts.cells, ts.cell_len);
                tris = runtime::soft::ExtractBoundaryTriangles(lat.rest, lat.tets);
                verts = static_cast<uint32_t>(lat.rest.size());
            }
        } else {
            continue;  // Fluid: no triangulated surface; it does not enter the soft slice.
        }
        if (!tris.empty()) {
            for (uint32_t& t : tris) t += base;
            MediaRenderSurface s;
            s.triangles = std::move(tris);
            s.normal_offset = m.render_skin.normal_offset;
            s.smooth_iters = m.render_skin.smooth_iters;
            s.smooth_lambda = m.render_skin.smooth_lambda;
            s.render_material_id = m.render_material_id;
            surfaces.push_back(std::move(s));
        }
        base += verts;
    }
    return surfaces;
}

XpbdCookInput BuildSoftTetXpbdInput(const MediaRecord& media) {
    XpbdCookInput in;
    const MediaRecord::TetSphere& ts = media.tet_sphere;
    if (ts.radius <= 0.0f || ts.cells < 2u || ts.cell_len <= 0.0f) {
        return in;  // no tet body (an empty soft set is treated as none).
    }
    // The lattice is built at the origin and translated to the authored center, so the
    // rest constraints (translation-invariant lengths/volumes) come from the origin set.
    const runtime::soft::TetLattice lat = runtime::soft::BuildSphereTetLattice(
        math::Vec3{0.0f, 0.0f, 0.0f}, ts.radius, ts.cells, ts.cell_len);
    std::vector<math::Vec3> init = lat.rest;
    for (math::Vec3& p : init) p = p + ts.center;

    runtime::soft::TetMeshTopologyOptions opts;
    opts.distance_compliance_alpha = media.xpbd.distance_alpha;
    opts.volume_compliance_alpha = media.xpbd.volume_alpha;
    opts.emit_distance_constraints = true;
    runtime::soft::XpbdConstraintSet cs;
    runtime::soft::BuildTetMeshConstraints(lat.rest, lat.tets, opts, cs);

    in.positions = init;
    in.velocities.assign(init.size(), math::Vec3::Zero());
    const float mass =
        media.xpbd.particle_mass > 0.0f ? media.xpbd.particle_mass : 0.01f;
    in.inv_mass.assign(init.size(), 1.0f / mass);  // free (unpinned) solid.
    for (const auto& dc : cs.distance) {
        in.distance.push_back(
            {dc.particle_a, dc.particle_b, dc.rest_length, dc.compliance_alpha});
    }
    for (const auto& vc : cs.volume) {
        CookVolumeCon c;
        for (uint32_t j = 0u; j < 4u; ++j) c.p[j] = vc.particle[j];
        c.rest_volume_times6 = vc.rest_volume_times6;
        c.compliance_alpha = vc.compliance_alpha;
        in.volume.push_back(c);
    }
    in.solver_iterations =
        static_cast<uint16_t>(media.xpbd.iters != 0u ? media.xpbd.iters : 1u);
    in.friction = media.xpbd.friction;
    return in;
}

XpbdCookInput BuildCableXpbdInput(const MediaRecord& media) {
    XpbdCookInput in;
    const MediaRecord::CableLine& c = media.cable_line;
    const CableLayout layout = CableParticleLayout(media);
    if (layout.chain == 0u) return in;  // no cable (absent geometry).

    // Chain particles: segments+1 samples linearly interpolated start -> end.
    const uint32_t np = layout.chain;
    const float seg = static_cast<float>(c.segments);
    for (uint32_t i = 0u; i < np; ++i) {
        const float t = static_cast<float>(i) / seg;
        in.positions.push_back(c.start * (1.0f - t) + c.end * t);
    }
    const float mass =
        media.xpbd.particle_mass > 0.0f ? media.xpbd.particle_mass : 0.01f;
    in.inv_mass.assign(np, 1.0f / mass);
    // Pin the authored endpoint(s) kinematic (inv_mass 0); the default hangs the
    // anchor and leaves the loaded end free to carry a weight.
    using Pin = MediaRecord::CableLine::Pin;
    if (c.pin == Pin::Start || c.pin == Pin::Both) in.inv_mass[0] = 0.0f;
    if (c.pin == Pin::End || c.pin == Pin::Both) in.inv_mass[np - 1u] = 0.0f;

    // One distance row per link (alpha 0 == inextensible unless a stretch compliance
    // is authored); optional skip-one rows add a little bending stiffness.
    const float d_alpha = media.xpbd.distance_alpha;
    for (uint32_t i = 0u; i + 1u < np; ++i) {
        in.distance.push_back(
            {i, i + 1u, (in.positions[i + 1u] - in.positions[i]).Length(), d_alpha});
    }
    if (c.bend) {
        const float b_alpha = media.xpbd.bend_alpha;
        for (uint32_t i = 0u; i + 2u < np; ++i) {
            in.distance.push_back({i, i + 2u,
                (in.positions[i + 2u] - in.positions[i]).Length(), b_alpha});
        }
    }

    // Optional rigid slab welded to the loaded end: 8 box corners as ONE shape-match
    // cluster (id 9), the 4 top corners distance-welded (alpha 0) to the end particle.
    if (layout.slab == 8u) {
        const std::array<math::Vec3, 8> corners = CableSlabCorners(c);
        const float slab_mass = c.slab.mass > 0.0f ? c.slab.mass : mass;
        const uint32_t base = np;  // slab corners follow the chain in the pool.
        CookShapeMatchCluster cluster;
        // Goal-pull fraction in [0,1]; 0 (an unset desc) defaults to a rigid slab.
        cluster.stiffness = c.slab.stiffness > 0.0f ? c.slab.stiffness : 1.0f;
        for (uint32_t k = 0u; k < 8u; ++k) {
            in.positions.push_back(corners[k]);
            in.inv_mass.push_back(1.0f / slab_mass);
            cluster.particle.push_back(base + k);
            cluster.rest_positions.push_back(corners[k]);
            cluster.cluster_mass.push_back(slab_mass);
        }
        in.shape_match.push_back(std::move(cluster));
        const uint32_t end_p = np - 1u;
        for (uint32_t k = 4u; k < 8u; ++k) {  // the 4 top corners (iz == 1).
            in.distance.push_back({end_p, base + k,
                (corners[k] - in.positions[end_p]).Length(), 0.0f});
        }
    }

    in.velocities.assign(in.positions.size(), math::Vec3::Zero());
    in.solver_iterations =
        static_cast<uint16_t>(media.xpbd.iters != 0u ? media.xpbd.iters : 1u);
    in.friction = media.xpbd.friction;
    return in;
}

// Append the pinned-boundary container (floor slab + side-wall ring) for a confined PBF
// pool, then fill in.inv_mass (free fluid = 1/mass, every boundary particle pinned 0).
static void AppendFluidConfinement(PbfCookInput& in, const MediaRecord& media) {
    const MediaRecord::FluidBox& fb = media.fluid_box;
    const MediaPbfMaterial& pbf = media.pbf;
    const uint32_t n_free = static_cast<uint32_t>(in.positions.size());
    const float s = fb.spacing;
    const int L = static_cast<int>(pbf.boundary_layers);

    const float wc_x = 0.5f * (pbf.walls_min.x + pbf.walls_max.x);
    const float wc_y = 0.5f * (pbf.walls_min.y + pbf.walls_max.y);
    const int nbx = static_cast<int>(std::ceil(0.5f * (pbf.walls_max.x - pbf.walls_min.x) / s));
    const int nby = static_cast<int>(std::ceil(0.5f * (pbf.walls_max.y - pbf.walls_min.y) / s));
    // CookFluidBox samples cell-centered, so the first fluid layer sits half a cell
    // above the box floor; the slab stacks downward from there.
    const float pool_bottom = fb.min.z + 0.5f * s;

    // Floor slab: `boundary_layers` filled pinned layers below the fluid bottom.
    for (int layer = 1; layer <= L; ++layer) {
        const float bz = pool_bottom - static_cast<float>(layer) * s;
        for (int iy = -nby; iy <= nby; ++iy)
            for (int ix = -nbx; ix <= nbx; ++ix)
                in.positions.push_back(
                    math::Vec3{wc_x + static_cast<float>(ix) * s,
                               wc_y + static_cast<float>(iy) * s, bz});
    }
    // Side walls: a pinned ring `boundary_layers` thick just outside the box footprint,
    // from the fluid bottom up to the box top, confining the column laterally.
    for (float z = pool_bottom; z <= pbf.walls_max.z + 1.0e-4f; z += s)
        for (int iy = -(nby + L); iy <= nby + L; ++iy)
            for (int ix = -(nbx + L); ix <= nbx + L; ++ix) {
                if (std::abs(ix) <= nbx && std::abs(iy) <= nby) continue;  // box interior.
                in.positions.push_back(
                    math::Vec3{wc_x + static_cast<float>(ix) * s,
                               wc_y + static_cast<float>(iy) * s, z});
            }

    const float im = in.particle_mass > 0.0f ? 1.0f / in.particle_mass : 0.0f;
    in.inv_mass.assign(in.positions.size(), 0.0f);  // boundary pinned.
    for (uint32_t i = 0; i < n_free; ++i) in.inv_mass[i] = im;  // free fluid.
}

PbfCookInput BuildFluidPbfInput(const MediaRecord& media) {
    PbfCookInput in;
    const MediaRecord::FluidBox& b = media.fluid_box;
    if (b.spacing <= 0.0f || b.max.x <= b.min.x || b.max.y <= b.min.y ||
        b.max.z <= b.min.z) {
        return in;  // no fluid.
    }
    import::cooker::FluidBoxSpec spec;
    spec.min_corner = b.min;
    spec.max_corner = b.max;
    spec.spacing = b.spacing;
    spec.rest_density = media.pbf.rest_density > 0.0f ? media.pbf.rest_density : 1000.0f;
    spec.position_jitter = b.position_jitter;
    const runtime::fluid::PbfParticleSet box = import::cooker::CookFluidBox(spec);

    in.positions = box.positions;
    in.particle_mass = box.particle_mass;
    in.rest_density = spec.rest_density;
    const float support_scale =
        media.pbf.support_scale > 0.0f ? media.pbf.support_scale : 1.5f;
    const float h = support_scale * b.spacing;  // SPH support over the lattice spacing.
    in.support_radius = h;
    in.cfm_epsilon = 1.0e-6f;
    in.iters = static_cast<uint16_t>(media.pbf.iters != 0u ? media.pbf.iters : 4u);
    in.clamp_overdensity = media.pbf.clamp_overdensity;
    in.boundary_enabled = true;
    in.floor_z = media.pbf.floor_z;
    in.friction = media.pbf.friction;
    auto cells = [h](float extent) {
        return static_cast<uint32_t>(std::ceil(extent / h)) + 1u;
    };
    const math::Vec3 lo = spec.min_corner, hi = spec.max_corner;
    if (!media.pbf.walls_enabled) {
        // Uniform-grid domain: enclose the box + lateral/vertical headroom so a
        // particle pushed out of the box still finds neighbours (rebuilt per step).
        in.velocities.assign(in.positions.size(), math::Vec3::Zero());
        in.grid_min = math::Vec3{lo.x - 3.0f * h, lo.y - 3.0f * h,
                                 std::min(lo.z, media.pbf.floor_z) - h};
        in.grid_dims[0] = cells((hi.x - lo.x) + 6.0f * h);
        in.grid_dims[1] = cells((hi.y - lo.y) + 6.0f * h);
        in.grid_dims[2] = cells((hi.z - in.grid_min.z) + 4.0f * h);
        return in;
    }
    // Confined pool: append the pinned container, then size the grid over the union
    // of the fluid + boundary particles + a crown headroom above the box top.
    in.walls_enabled = true;
    in.walls_min = media.pbf.walls_min;
    in.walls_max = media.pbf.walls_max;
    in.boundary_layers = media.pbf.boundary_layers;
    AppendFluidConfinement(in, media);
    in.velocities.assign(in.positions.size(), math::Vec3::Zero());
    math::Vec3 pmin{1.0e30f, 1.0e30f, 1.0e30f}, pmax{-1.0e30f, -1.0e30f, -1.0e30f};
    for (const math::Vec3& p : in.positions) {
        pmin = math::Vec3{std::min(pmin.x, p.x), std::min(pmin.y, p.y), std::min(pmin.z, p.z)};
        pmax = math::Vec3{std::max(pmax.x, p.x), std::max(pmax.y, p.y), std::max(pmax.z, p.z)};
    }
    const float headroom = (hi.z - lo.z) + 4.0f * h;  // crown room above the box top.
    in.grid_min = math::Vec3{pmin.x - 2.0f * h, pmin.y - 2.0f * h, pmin.z - 2.0f * h};
    in.grid_dims[0] = cells((pmax.x - pmin.x) + 4.0f * h);
    in.grid_dims[1] = cells((pmax.y - pmin.y) + 4.0f * h);
    in.grid_dims[2] = cells((pmax.z - in.grid_min.z) + headroom);
    return in;
}

// Heterogeneous MLS-MPM cook: sample each box fill (base bed + mpm_fills) with its
// own constitutive, tag the particles with a per-fill material id, and size the grid
// to the UNION of the sub-boxes. The medium's mpm block supplies the shared grid
// scalars (dx / substeps / floor / loft). Fluid/Granular (box) media only.
static MpmCookInput BuildMpmInputFills(const MediaRecord& media) {
    MpmCookInput in;
    const MediaMpmMaterial& grid = media.mpm;
    bool have = false;
    math::Vec3 lo{0.0f, 0.0f, 0.0f}, hi{0.0f, 0.0f, 0.0f};  // union AABB over sub-boxes.
    for (const MpmBoxFill& f : MpmBoxFills(media)) {
        const MediaRecord::FluidBox& b = f.box;
        if (!(b.spacing > 0.0f && b.max.x > b.min.x && b.max.y > b.min.y &&
              b.max.z > b.min.z)) continue;  // empty/invalid sub-box: contributes nothing.
        const float density = f.material.density > 0.0f ? f.material.density : 1000.0f;
        import::cooker::FluidBoxSpec spec;
        spec.min_corner = b.min;
        spec.max_corner = b.max;
        spec.spacing = b.spacing;
        spec.rest_density = density;
        spec.position_jitter = b.position_jitter;
        const std::vector<math::Vec3> pos = import::cooker::CookFluidBox(spec).positions;
        if (pos.empty()) continue;  // sub-box smaller than one cell: no particles.
        const uint32_t mid = static_cast<uint32_t>(in.materials.size());
        const float vol0 = b.spacing * b.spacing * b.spacing;
        const float inv_m = vol0 > 0.0f ? 1.0f / (density * vol0) : 0.0f;
        for (const math::Vec3& p : pos) {
            in.positions.push_back(p);
            in.velocities.push_back(math::Vec3::Zero());
            in.inv_mass.push_back(inv_m);
            in.vol0.push_back(vol0);
            in.material_id.push_back(mid);
        }
        MpmMaterialInput mi;
        mi.youngs = f.material.youngs; mi.poisson = f.material.poisson;
        mi.density = density; mi.dp_friction = f.material.dp_friction;
        mi.dp_cohesion = f.material.dp_cohesion;
        // A Granular medium IS Drucker-Prager; pin so an unset field cannot cook elastic.
        mi.model_kind = media.kind == MediaRecord::Kind::Granular
                            ? 4.0f : f.material.model_kind;
        mi.bulk_modulus = f.material.bulk_modulus; mi.tait_gamma = f.material.tait_gamma;
        mi.viscosity = f.material.viscosity;
        in.materials.push_back(mi);
        if (!have) { lo = b.min; hi = b.max; have = true; }
        else {
            lo = math::Vec3{std::min(lo.x, b.min.x), std::min(lo.y, b.min.y),
                            std::min(lo.z, b.min.z)};
            hi = math::Vec3{std::max(hi.x, b.max.x), std::max(hi.y, b.max.y),
                            std::max(hi.z, b.max.z)};
        }
    }
    in.dx = grid.dx;
    in.substeps = grid.substeps;
    in.floor_normal = grid.floor_normal;
    in.floor_d = grid.floor_d;
    in.floor_friction = grid.floor_friction;
    if (grid.dx > 0.0f && have) {
        const float margin = 4.0f * grid.dx;
        const float base_z = std::min(lo.z, grid.floor_d) - margin;
        const float top_z =
            hi.z + (hi.z - lo.z) + margin + std::max(0.0f, grid.loft_headroom);
        in.grid_origin = math::Vec3{lo.x - margin, lo.y - margin, base_z};
        auto cells = [&](float extent) {
            return static_cast<uint32_t>(std::ceil(extent / grid.dx)) + 1u;
        };
        in.grid_dims[0] = cells((hi.x - lo.x) + 2.0f * margin);
        in.grid_dims[1] = cells((hi.y - lo.y) + 2.0f * margin);
        in.grid_dims[2] = cells(top_z - base_z);
    }
    return in;
}

MpmCookInput BuildMpmInput(const MediaRecord& media) {
    // A heterogeneous bed (>= one mpm_fill) samples each sub-box with its own material.
    if (!media.mpm_fills.empty()) return BuildMpmInputFills(media);
    MpmCookInput in;
    const MediaMpmMaterial& mp = media.mpm;
    const float density = mp.density > 0.0f ? mp.density : 1000.0f;
    math::Vec3 lo{0.0f, 0.0f, 0.0f}, hi{0.0f, 0.0f, 0.0f};  // geometry AABB.
    float vol0 = 0.0f;                                       // per-particle sampling volume.

    if (media.kind == MediaRecord::Kind::Fluid ||
        media.kind == MediaRecord::Kind::Granular) {
        // A granular bed samples the SAME box lattice as a fluid; only the
        // constitutive (model_kind 4, Drucker-Prager) differs.
        const MediaRecord::FluidBox& b = media.fluid_box;
        if (b.spacing > 0.0f && b.max.x > b.min.x && b.max.y > b.min.y &&
            b.max.z > b.min.z) {
            import::cooker::FluidBoxSpec spec;
            spec.min_corner = b.min;
            spec.max_corner = b.max;
            spec.spacing = b.spacing;
            spec.rest_density = density;
            spec.position_jitter = b.position_jitter;
            in.positions = import::cooker::CookFluidBox(spec).positions;
            vol0 = b.spacing * b.spacing * b.spacing;
            lo = b.min;
            hi = b.max;
        }
    } else {  // SoftTet (ValidateMedia rejects a cloth -> MLS-MPM medium).
        const MediaRecord::TetSphere& ts = media.tet_sphere;
        if (ts.radius > 0.0f && ts.cells >= 2u && ts.cell_len > 0.0f) {
            const runtime::soft::TetLattice lat = runtime::soft::BuildSphereTetLattice(
                math::Vec3{0.0f, 0.0f, 0.0f}, ts.radius, ts.cells, ts.cell_len);
            in.positions = lat.rest;
            for (math::Vec3& p : in.positions) p = p + ts.center;
            vol0 = ts.cell_len * ts.cell_len * ts.cell_len;
            lo = ts.center - math::Vec3{ts.radius, ts.radius, ts.radius};
            hi = ts.center + math::Vec3{ts.radius, ts.radius, ts.radius};
        }
    }
    const size_t n = in.positions.size();
    in.velocities.assign(n, math::Vec3::Zero());
    in.inv_mass.assign(n, vol0 > 0.0f ? 1.0f / (density * vol0) : 0.0f);
    in.vol0.assign(n, vol0);
    in.material.youngs = mp.youngs;
    in.material.poisson = mp.poisson;
    in.material.density = density;
    in.material.dp_friction = mp.dp_friction;
    in.material.dp_cohesion = mp.dp_cohesion;
    // A Granular medium IS the Drucker-Prager constitutive; pin model_kind so an
    // unset/default material field cannot silently cook a granular bed as elastic.
    in.material.model_kind =
        media.kind == MediaRecord::Kind::Granular ? 4.0f : mp.model_kind;
    in.material.bulk_modulus = mp.bulk_modulus;
    in.material.tait_gamma = mp.tait_gamma;
    in.material.viscosity = mp.viscosity;
    in.dx = mp.dx;
    in.substeps = mp.substeps;
    in.floor_normal = mp.floor_normal;
    in.floor_d = mp.floor_d;
    in.floor_friction = mp.floor_friction;
    // Env-private background grid: enclose the geometry + its floor + motion headroom
    // (a body's own height above, so a bounce/splash stays inside the grid).
    if (mp.dx > 0.0f) {
        const float margin = 4.0f * mp.dx;
        const float base_z = std::min(lo.z, mp.floor_d) - margin;
        // loft_headroom (0 by default) raises the +z ceiling for kicked/lofted debris.
        const float top_z =
            hi.z + (hi.z - lo.z) + margin + std::max(0.0f, mp.loft_headroom);
        in.grid_origin = math::Vec3{lo.x - margin, lo.y - margin, base_z};
        auto cells = [&](float extent) {
            return static_cast<uint32_t>(std::ceil(extent / mp.dx)) + 1u;
        };
        in.grid_dims[0] = cells((hi.x - lo.x) + 2.0f * margin);
        in.grid_dims[1] = cells((hi.y - lo.y) + 2.0f * margin);
        in.grid_dims[2] = cells(top_z - base_z);
    }
    return in;
}

namespace {

// The body<->particle / cross-system contact diameter: 2*radius when authored, else
// the smaller present medium lattice spacing (a particle is then ~ half a cell).
float CrossContactDMin(const std::vector<MediaRecord>& media,
                       float particle_contact_radius) {
    float d_min = 2.0f * particle_contact_radius;
    if (d_min > 0.0f) return d_min;
    // An MLS-MPM medium couples to bodies via the grid and emits NO body<->particle
    // rows, so it must not shrink the row diameter of a co-resident row-making set.
    bool any_row_media = false;
    for (const MediaRecord& m : media) {
        if (m.method != MediaRecord::Method::MlsMpm) { any_row_media = true; break; }
    }
    for (const MediaRecord& m : media) {
        if (any_row_media && m.method == MediaRecord::Method::MlsMpm) continue;
        float sp = 0.0f;
        if (m.kind == MediaRecord::Kind::Cloth) sp = m.cloth_grid.spacing;
        else if (m.kind == MediaRecord::Kind::SoftTet) sp = m.tet_sphere.cell_len;
        else if (m.kind == MediaRecord::Kind::Cable) sp = 2.0f * m.cable_line.radius;
        else sp = m.fluid_box.spacing;  // Fluid + Granular share the box lattice.
        if (sp > 0.0f) d_min = (d_min > 0.0f) ? std::min(d_min, sp) : sp;
    }
    return d_min;
}

// Concatenate one XPBD medium into the soft slice, rebasing constraint particle
// indices by the running particle count (base 0 for a single medium => a copy).
void AppendSoftMedium(XpbdCookInput& dst, const XpbdCookInput& src) {
    const uint32_t base = static_cast<uint32_t>(dst.positions.size());
    dst.positions.insert(dst.positions.end(), src.positions.begin(),
                         src.positions.end());
    dst.velocities.insert(dst.velocities.end(), src.velocities.begin(),
                          src.velocities.end());
    dst.inv_mass.insert(dst.inv_mass.end(), src.inv_mass.begin(), src.inv_mass.end());
    for (CookDistanceCon dc : src.distance) {
        dc.a += base; dc.b += base; dst.distance.push_back(dc);
    }
    for (CookBendCon bc : src.bend) {
        for (uint32_t k = 0u; k < 4u; ++k) bc.p[k] += base;
        dst.bend.push_back(bc);
    }
    for (CookVolumeCon vc : src.volume) {
        for (uint32_t k = 0u; k < 4u; ++k) vc.p[k] += base;
        dst.volume.push_back(vc);
    }
    for (CookShapeMatchCluster sm : src.shape_match) {
        for (uint32_t& pi : sm.particle) pi += base;
        dst.shape_match.push_back(std::move(sm));
    }
    for (std::array<uint32_t, 3> t : src.aero_triangles) {
        t[0] += base; t[1] += base; t[2] += base;
        dst.aero_triangles.push_back(t);
    }
    // The soft slice carries one solver/friction/aero set (per-medium override is the
    // medium's own); a single soft medium sets them verbatim.
    dst.solver_iterations = src.solver_iterations;
    dst.friction = src.friction;
    dst.aero_drag_normal = src.aero_drag_normal;
    dst.aero_drag_tangent = src.aero_drag_tangent;
    dst.aero_drag_max_dv = src.aero_drag_max_dv;
    dst.solver = src.solver;
}

}  // namespace

void CookSceneMedia(nk::Model& model, uint32_t env_count,
                    const std::vector<MediaRecord>& media,
                    float particle_contact_radius) {
    if (media.empty()) return;  // no media -> ParticleMode::None (byte-identical).
    ValidateMedia(media);

    // The single MLS-MPM medium, if any (ValidateMedia guarantees at most one and no
    // PBF fluid co-resident with it).
    const MediaRecord* mpm_medium = nullptr;
    for (const MediaRecord& m : media) {
        if (m.method == MediaRecord::Method::MlsMpm) { mpm_medium = &m; break; }
    }

    // XPBD soft (cloth + tet-soft, concatenated) + the one PBF fluid. The MLS-MPM
    // medium is handled by its own composer below (skipped here).
    XpbdCookInput soft;
    PbfCookInput fluid;
    bool have_soft = false, have_fluid = false;
    for (const MediaRecord& m : media) {
        if (m.method == MediaRecord::Method::MlsMpm) continue;
        if (m.kind == MediaRecord::Kind::Cloth) {
            AppendSoftMedium(soft, BuildClothXpbdInput(m));
            have_soft = true;
        } else if (m.kind == MediaRecord::Kind::SoftTet) {
            AppendSoftMedium(soft, BuildSoftTetXpbdInput(m));
            have_soft = true;
        } else if (m.kind == MediaRecord::Kind::Cable) {
            AppendSoftMedium(soft, BuildCableXpbdInput(m));
            have_soft = true;
        } else {  // Fluid (PBF; ValidateMedia guarantees at most one).
            fluid = BuildFluidPbfInput(m);
            have_fluid = true;
        }
    }

    SoftFluidContactInput contact;
    contact.contact_d_min = CrossContactDMin(media, particle_contact_radius);

    if (mpm_medium != nullptr) {
        // An MLS-MPM medium co-resides with XPBD media in the [mpm | xpbd] layout, or
        // stands alone as its own ParticleMode. No PBF fluid co-resides (ValidateMedia).
        if (have_soft) {
            CookMpmXpbd(model, env_count, BuildMpmInput(*mpm_medium), soft, contact);
        } else {
            XpbdCookInput soft_mpm;
            soft_mpm.solver = nk::Model::ParticleMode::Mpm;  // the cook dispatch selector.
            CookSoftBodyParticles(model, env_count, soft_mpm, BuildMpmInput(*mpm_medium));
        }
        return;
    }

    if (!have_soft && !have_fluid) return;
    CookSoftFluidParticles(model, env_count, soft, fluid, contact);
}

CookToModelResult CookSceneToModel(const SceneIR& scene, int env_count,
                                   const CookToModelOptions& options) {
    CookToModelResult result = CookToModel(scene, env_count, options);
    const uint32_t envs = env_count > 0 ? static_cast<uint32_t>(env_count) : 1u;
    CookSceneMedia(result.model, envs, scene.Media());
    return result;
}

} // namespace nuka::scene::cook
