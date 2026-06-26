// ---------------------------------------------------------------------------
// nuka::scene::CookScene implementation
// ---------------------------------------------------------------------------

#include "scene/cooker.hpp"

#include "import/cooker/convex_decomposition.hpp"
#include "import/cooker/sdf_bake_backend.hpp"
#include "import/cooker/sparse_sdf_cooker.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"  // PackSdfCellKey codec (shared)
#include "scene/asset/asset_cache.hpp"       // M2c: content-addressed cook cache

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nuka::scene {

namespace {

// V-HACD content-hash cache dir (M2c). The AssetCache root: DecomposeMeshCached
// persists "<sha256>.nukacvx" files here, byte-identical to the AssetCache hull
// layout, so the disk cache and AssetCache share one content-addressed store.
// The key is over INPUTS (mesh bytes + params), so a hit is bit-identical to a
// cold cook (the V-HACD determinism test + the cooker goldens prove it). The
// on-disk store is regenerable and gitignored; a decomposition-logic change is
// handled by clearing the dir (no input change => same key by design).
constexpr const char* kDecomposeCacheDir = ".nuka_cache";

math::Transform ResolveWorldTransform(const SceneIR& scene, BodyId body_id) {
    const auto& body = scene.GetBody(body_id);
    if (body.parent_id == kInvalidBody) {
        return body.local_transform;
    }
    return ResolveWorldTransform(scene, body.parent_id) * body.local_transform;
}

import::cooker::DecomposeMode ToCookerMode(DecomposeMode mode) {
    switch (mode) {
        case DecomposeMode::Force: return import::cooker::DecomposeMode::Force;
        case DecomposeMode::Skip:  return import::cooker::DecomposeMode::Skip;
        case DecomposeMode::Auto:  break;
    }
    return import::cooker::DecomposeMode::Auto;
}

// Append one convex hull (vertices/indices/volume) into the cooked geometry
// table and return its index.
uint32_t AppendConvexGeometry(CookedConvexGeometry& geom,
                              const std::vector<float>& vertices,
                              const std::vector<uint32_t>& indices,
                              float volume) {
    const uint32_t index = geom.Count();
    geom.vertex_offsets.push_back(static_cast<uint32_t>(geom.vertices.size() / 3));
    geom.vertex_counts.push_back(static_cast<uint32_t>(vertices.size() / 3));
    geom.index_offsets.push_back(static_cast<uint32_t>(geom.indices.size()));
    geom.index_counts.push_back(static_cast<uint32_t>(indices.size()));
    geom.volumes.push_back(volume);
    geom.vertices.insert(geom.vertices.end(), vertices.begin(), vertices.end());
    geom.indices.insert(geom.indices.end(), indices.begin(), indices.end());
    return index;
}

// Append one cooked SparseSdfData into the blob's CookedSdfTable and return its
// SDF index. Cells are stored flat/concatenated; the per-SDF slice is
// [key_offsets[idx], +key_counts[idx]).
uint32_t AppendSdf(CookedSdfTable& table,
                   const import::cooker::SparseSdfData& sdf) {
    const uint32_t index = table.Count();
    table.origins.push_back({sdf.origin[0], sdf.origin[1], sdf.origin[2]});
    table.voxel_sizes.push_back(sdf.voxel_size);
    table.dims_x.push_back(sdf.dims[0]);
    table.dims_y.push_back(sdf.dims[1]);
    table.dims_z.push_back(sdf.dims[2]);
    table.key_offsets.push_back(static_cast<uint32_t>(table.cell_keys.size()));
    table.key_counts.push_back(sdf.CellCount());
    table.cell_keys.insert(table.cell_keys.end(), sdf.cell_keys.begin(), sdf.cell_keys.end());
    table.cell_values.insert(table.cell_values.end(), sdf.cell_values.begin(),
                             sdf.cell_values.end());
    for (uint32_t n = 0; n < sdf.CellCount(); ++n) {
        table.cell_gradients.push_back({sdf.cell_gradients[3 * n + 0],
                                        sdf.cell_gradients[3 * n + 1],
                                        sdf.cell_gradients[3 * n + 2]});
    }
    return index;
}

// Cook a narrow-band SDF per UNIQUE convex-geometry piece, deduplicated by
// content hash. Two identical pieces share ONE stored SDF (exit-crit 7).
// piece_sdf_indices is parallel to the convex-geometry pieces.
//
// `unique_out` / `total_out` report the dedup stats; `bytes_out` the total
// narrow-band bytes (R-C ~80MB cap surfaced by the caller).
void CookSdfsForGeometry(const CookedConvexGeometry& geom,
                         CookedSdfTable& table,
                         uint32_t* unique_out,
                         uint32_t* total_out,
                         uint64_t* bytes_out) {
    const import::cooker::SparseSdfParams params;  // defaults (auto voxel size)
    // v0.7 p07 seam: route bake through the swappable backend (CPU now; the
    // v1.0 GPU backend drops in via DefaultSdfBakeBackend with no caller change).
    const import::cooker::SdfBakeBackend& backend =
        import::cooker::DefaultSdfBakeBackend();
    // M2c: wrap the cold bake in the content-addressed AssetCache so the cooked
    // SDF persists to disk (keyed by ComputeSdfCacheKey, the same key the backend
    // dedups on within a cook). A cache HIT is bit-identical to the cold bake
    // (the SDF0 (de)serialization is exact), so the ContactMetadata D1 + cooker
    // goldens stay green. The in-process by_hash map below still dedups within a
    // single cook; the AssetCache adds cross-cook / cross-process persistence.
    AssetCache asset_cache(kDecomposeCacheDir);
    std::unordered_map<std::string, uint32_t> by_hash;  // hash -> sdf index
    table.piece_sdf_indices.assign(geom.Count(), kNoSdf);

    uint32_t total = 0;
    uint64_t bytes = 0;
    for (uint32_t piece = 0; piece < geom.Count(); ++piece) {
        const uint32_t vbase = geom.vertex_offsets[piece];
        const uint32_t vcount = geom.vertex_counts[piece];
        const uint32_t ibase = geom.index_offsets[piece];
        const uint32_t icount = geom.index_counts[piece];
        if (vcount == 0u || icount == 0u) {
            continue;  // leave kNoSdf
        }
        const float* vptr = geom.vertices.data() + static_cast<size_t>(vbase) * 3u;
        const uint32_t* iptr = geom.indices.data() + ibase;
        const uint32_t tri_count = icount / 3u;

        ++total;
        const std::string key =
            backend.CacheKey(vptr, vcount, iptr, tri_count, params);
        const auto it = by_hash.find(key);
        if (it != by_hash.end()) {
            table.piece_sdf_indices[piece] = it->second;  // dedup: reuse
            continue;
        }
        const auto sdf = asset_cache.GetOrCookSdf(
            vptr, vcount, iptr, tri_count, params,
            [&] { return backend.Bake(vptr, vcount, iptr, tri_count, params); });
        if (sdf.CellCount() == 0u) {
            continue;  // degenerate; leave kNoSdf
        }
        const uint32_t sdf_index = AppendSdf(table, sdf);
        by_hash.emplace(key, sdf_index);
        table.piece_sdf_indices[piece] = sdf_index;
        bytes += sdf.NarrowBandBytes();
    }

    if (unique_out) *unique_out = table.Count();
    if (total_out)  *total_out = total;
    if (bytes_out)  *bytes_out = bytes;
}

// Bake a narrow-band SDF per body FROM that body's VISUAL trimesh and bind it to
// that body's collision row, so particle/MPM/rigid contact rides the true
// silhouette (kills clip-through + gives a rounded grippable surface) instead of
// the inset collision primitive. Keys off a DATA property — a colliding body
// whose sibling visual shape carries triangles — never a scene name. The visual
// triangles are expressed in the body's COLLISION-primitive frame (the frame the
// runtime poses the body row in: link_pose o link_geom_local) so the cooked grid
// samples correctly under FK. Reuses the shared SDF bake backend + content-hash
// cache (deterministic, dedup'd against the convex-piece SDFs).
void CookLinkVisualSdfs(const SceneIR& scene, CookedBlob& blob) {
    blob.sdfs.body_sdf_indices.assign(blob.body_count, kNoSdf);
    const auto& shapes = scene.Shapes();

    // Per body: the FIRST colliding shape (its local frame anchors the bake). A body
    // may carry SEVERAL visual geoms (the Go2 trunk has 5 shell pieces); merge them
    // ALL so the SDF hugs the WHOLE silhouette, not one detail piece.
    auto collides = [](const CollisionShapeRecord& s) {
        return s.contype != 0u || s.conaffinity != 0u;
    };
    std::vector<uint32_t> body_collide_shape(blob.body_count, ~0u);
    for (uint32_t i = 0; i < shapes.size(); ++i) {
        const CollisionShapeRecord& s = shapes[i];
        if (s.body_id >= blob.body_count) continue;
        if (collides(s) && body_collide_shape[s.body_id] == ~uint32_t(0))
            body_collide_shape[s.body_id] = i;
    }

    // Solid fill (scoped here) so a deep interior point has a signed value+gradient
    // and a penetrating particle recovers instead of sinking through the unbanded core.
    import::cooker::SparseSdfParams params;
    params.voxel_size = 0.004f;
    params.band_voxels = 12u;
    params.solid = true;
    const import::cooker::SdfBakeBackend& backend =
        import::cooker::DefaultSdfBakeBackend();
    AssetCache asset_cache(kDecomposeCacheDir);
    std::unordered_map<std::string, uint32_t> by_hash;  // hash -> sdf index

    for (uint32_t b = 0; b < blob.body_count; ++b) {
        const uint32_t cs = body_collide_shape[b];
        if (cs == ~uint32_t(0)) continue;
        // Express every visual geom on this body in the collision primitive's local
        // frame (the frame the runtime poses the body row in) and concatenate. The
        // index buffer is rebased per geom into the merged vertex pool.
        const math::Transform to_collision = shapes[cs].local_transform.Inverse();
        std::vector<float> verts;
        std::vector<uint32_t> idx;
        for (uint32_t i = 0; i < shapes.size(); ++i) {
            const CollisionShapeRecord& vis = shapes[i];
            if (vis.body_id != b || collides(vis)) continue;
            if (vis.mesh_vertices.empty() || vis.mesh_indices.empty()) continue;
            const math::Transform xform = to_collision * vis.local_transform;
            const uint32_t base = static_cast<uint32_t>(verts.size() / 3u);
            const uint32_t vn = static_cast<uint32_t>(vis.mesh_vertices.size() / 3u);
            for (uint32_t v = 0; v < vn; ++v) {
                const math::Vec3 p = xform.TransformPoint(
                    {vis.mesh_vertices[v * 3u], vis.mesh_vertices[v * 3u + 1u],
                     vis.mesh_vertices[v * 3u + 2u]});
                verts.push_back(p.x); verts.push_back(p.y); verts.push_back(p.z);
            }
            for (uint32_t e : vis.mesh_indices) idx.push_back(base + e);
        }
        const uint32_t vcount = static_cast<uint32_t>(verts.size() / 3u);
        const uint32_t tcount = static_cast<uint32_t>(idx.size() / 3u);
        if (vcount == 0u || tcount == 0u) continue;

        const std::string key =
            backend.CacheKey(verts.data(), vcount, idx.data(), tcount, params);
        const auto it = by_hash.find(key);
        if (it != by_hash.end()) { blob.sdfs.body_sdf_indices[b] = it->second; continue; }
        const bool diag = std::getenv("NK_SDF_COOK_DIAG") != nullptr;
        const auto t0 = std::chrono::steady_clock::now();
        const auto sdf = asset_cache.GetOrCookSdf(
            verts.data(), vcount, idx.data(), tcount, params,
            [&] { return backend.Bake(verts.data(), vcount, idx.data(), tcount, params); });
        if (diag) {
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "[sdf_cook] body %u verts=%u tris=%u cells=%u %.0f ms\n",
                         b, vcount, tcount, sdf.CellCount(), ms);
        }
        if (sdf.CellCount() == 0u) continue;  // degenerate; leave kNoSdf.
        const uint32_t sdf_idx = AppendSdf(blob.sdfs, sdf);
        by_hash.emplace(key, sdf_idx);
        blob.sdfs.body_sdf_indices[b] = sdf_idx;
    }
}

// Resolve the cooked per-shape friction μ from the per-shape override and the
// (optional) per-material default. Friction resolution precedence (v0.8 C1a):
//   1. shape.friction_mu  if >= 0   (explicit per-shape override)
//   2. else material.friction_mu    if the shape has a valid material_id
//   3. else 1.0f                    (MuJoCo default)
float ResolveShapeFriction(const SceneIR& scene, const CollisionShapeRecord& src) {
    if (src.friction_mu >= 0.0f) {
        return src.friction_mu;                              // (1) per-shape override
    }
    if (src.material_id < scene.Materials().size()) {        // (2) per-material default
        return scene.Materials()[src.material_id].friction_mu;
    }
    return 1.0f;                                             // (3) MuJoCo default
}

// Push one cooked shape row AND its parallel contact-param row in lockstep, so
// the two tables can never desync. A single source CollisionShapeRecord may
// emit MULTIPLE cooked rows (one per V-HACD piece) — each call here appends one
// row to BOTH tables, and all pieces of a mesh inherit the parent's contact
// metadata (resolved_friction is computed once per source shape). (v0.8 C1a)
void PushShapeRow(CookedShapeTable& shapes,
                  CookedContactParamTable& contact_params,
                  const CollisionShapeRecord& src,
                  ShapeType type,
                  uint32_t convex_geometry_index,
                  float resolved_friction) {
    shapes.types.push_back(type);
    shapes.body_ids.push_back(src.body_id);
    shapes.material_ids.push_back(src.material_id);
    shapes.local_transforms.push_back(src.local_transform);
    shapes.half_extents.push_back(src.half_extents);
    shapes.radii.push_back(src.radius);
    shapes.half_heights.push_back(src.half_height);
    shapes.convex_geometry_indices.push_back(convex_geometry_index);

    // Parallel contact-param row (copied verbatim except the resolved friction).
    contact_params.contypes.push_back(src.contype);
    contact_params.conaffinities.push_back(src.conaffinity);
    contact_params.groups.push_back(src.collision_group);
    contact_params.solref0.push_back(src.solref[0]);
    contact_params.solref1.push_back(src.solref[1]);
    for (int k = 0; k < 5; ++k) {
        contact_params.solimp.push_back(src.solimp[k]);  // flattened, row-major
    }
    contact_params.frictions.push_back(resolved_friction);
    contact_params.condims.push_back(src.condim);
    contact_params.priorities.push_back(src.priority);
    contact_params.solmix.push_back(src.solmix);
    contact_params.margins.push_back(src.margin);
    contact_params.gaps.push_back(src.gap);
}

// Bake the cook-time filtered-pair policy (v0.8 C1c). Called AFTER bodies /
// joints / shapes are cooked. Implements three of MuJoCo's four filtering
// precedence levels (research doc §8); the fourth (contype/conaffinity bitmask)
// stays a per-pair runtime check from CookedContactParamTable.
//
//   excluded_body_pairs = union of:
//     (a) scene.ExcludePairs()  -- authored <contact><exclude>, already (min,max)
//     (b) parent-child auto-exclude -- each joint connects parent->child; that
//         body pair is excluded (MuJoCo auto-excludes parent-child unless
//         re-enabled). Joints with a kInvalidBody parent or child (e.g. a
//         floating-base Free joint) are SKIPPED (no real body pair to exclude).
//     ...canonicalized (min,max), deduplicated, ascending-sorted (binary search).
//   LIMITATION: bodies welded together WITHOUT a joint (a Fixed weld expressed
//   purely structurally, not as a JointRecord) are NOT auto-excluded in v0.8.
//   Only joint-connected parent-child pairs auto-exclude.
//
//   explicit_pairs = scene.ContactPairs() copied as-authored (NOT geom-merged):
//     each ContactPairOverride's params are stored verbatim into the
//     MergedContactParams (C1b pre-filled MuJoCo defaults for omitted attrs).
//     v0.8 does NOT merge omitted <pair> attributes with the per-geom params.
//     geom1/geom2 canonicalized (min,max) and ascending-sorted (binary search).
//     These are in SOURCE-shape space; the source->cooked-row expansion for a
//     decomposed mesh is a C2 concern (see CookedExplicitPair doc).
//
//   system_pairs = default all-enabled (forward-looking seam for C2).
void BuildFilteredPairPolicy(const SceneIR& scene, CookedBlob& blob) {
    CookedFilterPolicy& policy = blob.filter_policy;

    // -- excluded_body_pairs ------------------------------------------------
    std::vector<std::pair<BodyId, BodyId>> excludes;
    excludes.reserve(scene.ExcludePairs().size() + scene.Joints().size());

    for (const auto& p : scene.ExcludePairs()) {
        BodyId a = p.first, b = p.second;
        if (b < a) std::swap(a, b);  // re-canonicalize defensively
        excludes.emplace_back(a, b);
    }
    for (const auto& j : scene.Joints()) {
        if (j.parent_body == kInvalidBody || j.child_body == kInvalidBody) {
            continue;  // floating-base / parentless joint: no body pair
        }
        BodyId a = j.parent_body, b = j.child_body;
        if (a == b) continue;  // degenerate self-joint: nothing to exclude
        if (b < a) std::swap(a, b);
        excludes.emplace_back(a, b);
    }
    std::sort(excludes.begin(), excludes.end());
    excludes.erase(std::unique(excludes.begin(), excludes.end()), excludes.end());
    policy.excluded_body_pairs = std::move(excludes);

    // -- explicit_pairs (as-authored, NOT merged) ---------------------------
    std::vector<CookedExplicitPair> explicit_pairs;
    explicit_pairs.reserve(scene.ContactPairs().size());
    for (const auto& ov : scene.ContactPairs()) {
        CookedExplicitPair ep;
        ep.geom1 = ov.geom1;
        ep.geom2 = ov.geom2;
        if (ep.geom2 < ep.geom1) std::swap(ep.geom1, ep.geom2);  // canonical
        ep.params.condim      = ov.condim;
        ep.params.friction_mu = ov.friction_mu;
        ep.params.solref[0]   = ov.solref[0];
        ep.params.solref[1]   = ov.solref[1];
        for (int k = 0; k < 5; ++k) ep.params.solimp[k] = ov.solimp[k];
        ep.params.margin      = ov.margin;
        ep.params.gap         = ov.gap;
        explicit_pairs.push_back(ep);
    }
    std::sort(explicit_pairs.begin(), explicit_pairs.end(),
              [](const CookedExplicitPair& x, const CookedExplicitPair& y) {
                  if (x.geom1 != y.geom1) return x.geom1 < y.geom1;
                  return x.geom2 < y.geom2;
              });
    policy.explicit_pairs = std::move(explicit_pairs);

    // -- system_pairs: default all-enabled (SystemPairMatrix ctor). ----------
}

} // namespace

CookedBlob CookScene(const SceneIR& scene) {
    return CookScene(scene, CookSceneOptions{});
}

CookedBlob CookScene(const SceneIR& scene, const CookSceneOptions& options) {
    CookedBlob blob;

    const auto& bodies = scene.Bodies();
    blob.body_count = static_cast<uint32_t>(bodies.size());
    blob.bodies.poses.reserve(bodies.size());
    blob.bodies.local_poses.reserve(bodies.size());
    blob.bodies.inertial_frames.reserve(bodies.size());
    blob.bodies.masses.reserve(bodies.size());
    blob.bodies.inertias.reserve(bodies.size());
    blob.bodies.inv_masses.reserve(bodies.size());
    blob.bodies.inv_inertias.reserve(bodies.size());
    blob.bodies.is_static.reserve(bodies.size());

    for (const auto& b : bodies) {
        blob.bodies.poses.push_back(ResolveWorldTransform(scene, b.id));
        blob.bodies.local_poses.push_back(b.local_transform);
        blob.bodies.inertial_frames.push_back(b.inertial_transform);
        blob.bodies.masses.push_back(b.is_static ? 0.0f : b.mass);
        blob.bodies.inertias.push_back(b.is_static ? math::Vec3::Zero() : b.inertia);

        if (b.is_static) {
            blob.bodies.inv_masses.push_back(0.0f);
            blob.bodies.inv_inertias.push_back(math::Vec3::Zero());
        } else {
            blob.bodies.inv_masses.push_back((b.mass > 0.0f) ? (1.0f / b.mass) : 0.0f);
            blob.bodies.inv_inertias.push_back({
                (b.inertia.x > 0.0f) ? (1.0f / b.inertia.x) : 0.0f,
                (b.inertia.y > 0.0f) ? (1.0f / b.inertia.y) : 0.0f,
                (b.inertia.z > 0.0f) ? (1.0f / b.inertia.z) : 0.0f
            });
        }

        blob.bodies.is_static.push_back(b.is_static ? uint8_t(1) : uint8_t(0));
    }

    const auto& joints = scene.Joints();
    blob.joint_count = static_cast<uint32_t>(joints.size());
    blob.joints.types.reserve(joints.size());
    blob.joints.parent_bodies.reserve(joints.size());
    blob.joints.child_bodies.reserve(joints.size());
    blob.joints.axes.reserve(joints.size());
    blob.joints.parent_frames.reserve(joints.size());
    blob.joints.child_frames.reserve(joints.size());
    blob.joints.lower_limits.reserve(joints.size());
    blob.joints.upper_limits.reserve(joints.size());
    blob.joints.dampings.reserve(joints.size());
    blob.joints.armatures.reserve(joints.size());
    blob.joints.initial_positions.reserve(joints.size());

    for (const auto& j : joints) {
        blob.joints.types.push_back(j.type);
        blob.joints.parent_bodies.push_back(j.parent_body);
        blob.joints.child_bodies.push_back(j.child_body);
        blob.joints.axes.push_back(j.axis);
        blob.joints.parent_frames.push_back(j.parent_frame);
        blob.joints.child_frames.push_back(j.child_frame);
        blob.joints.lower_limits.push_back(j.lower_limit);
        blob.joints.upper_limits.push_back(j.upper_limit);
        blob.joints.dampings.push_back(j.damping);
        blob.joints.armatures.push_back(j.armature);
        blob.joints.initial_positions.push_back(j.initial_position);
    }

    const auto& shapes = scene.Shapes();
    blob.shapes.types.reserve(shapes.size());
    blob.shapes.body_ids.reserve(shapes.size());
    blob.shapes.material_ids.reserve(shapes.size());
    blob.shapes.local_transforms.reserve(shapes.size());
    blob.shapes.half_extents.reserve(shapes.size());
    blob.shapes.radii.reserve(shapes.size());
    blob.shapes.half_heights.reserve(shapes.size());
    blob.shapes.convex_geometry_indices.reserve(shapes.size());
    // Contact-param table is parallel to the shape rows (>= shapes.size() once
    // meshes decompose); reserve the lower bound. v0.8 C1a.
    blob.contact_params.contypes.reserve(shapes.size());
    blob.contact_params.conaffinities.reserve(shapes.size());
    blob.contact_params.groups.reserve(shapes.size());
    blob.contact_params.solref0.reserve(shapes.size());
    blob.contact_params.solref1.reserve(shapes.size());
    blob.contact_params.solimp.reserve(shapes.size() * 5);
    blob.contact_params.frictions.reserve(shapes.size());
    blob.contact_params.condims.reserve(shapes.size());
    blob.contact_params.priorities.reserve(shapes.size());
    blob.contact_params.solmix.reserve(shapes.size());
    blob.contact_params.margins.reserve(shapes.size());
    blob.contact_params.gaps.reserve(shapes.size());

    for (const auto& s : shapes) {
        // Resolve friction ONCE per source shape; all cooked rows this shape
        // emits (incl. every V-HACD piece) inherit the same contact metadata.
        const float resolved_friction = ResolveShapeFriction(scene, s);

        const bool is_mesh =
            (s.type == ShapeType::TriMesh || s.type == ShapeType::ConvexHull);
        const bool has_geometry = !s.mesh_vertices.empty() && !s.mesh_indices.empty();

        // Non-mesh shapes (and mesh shapes lacking geometry — e.g. importers
        // that do not yet load mesh files) pass through unchanged, one row.
        if (!is_mesh || !has_geometry) {
            PushShapeRow(blob.shapes, blob.contact_params, s, s.type,
                         kNoConvexGeometry, resolved_friction);
            continue;
        }

        import::cooker::DecomposeMode mode = ToCookerMode(s.decompose_mode);

        // L-RECON-B: the GENERAL cvx narrowphase wants ONE convex hull per mesh,
        // not V-HACD's N concave pieces. When single-hull is requested, collapse
        // every mesh to its own hull (the Skip path) UNLESS the author explicitly
        // forced decomposition (a genuinely-concave shape opting INTO V-HACD).
        // This is a STAGE gate driven by the cook option + the per-shape mode —
        // no entity/scene-name branch.
        if (options.general_single_hull &&
            mode != import::cooker::DecomposeMode::Force) {
            mode = import::cooker::DecomposeMode::Skip;
        }

        if (mode == import::cooker::DecomposeMode::Skip) {
            // Treat the source mesh as a single convex piece (store its own
            // geometry as one ConvexHull; no V-HACD run).
            const uint32_t geom_index = AppendConvexGeometry(
                blob.convex_geometry, s.mesh_vertices, s.mesh_indices, 0.0f);
            PushShapeRow(blob.shapes, blob.contact_params, s,
                         ShapeType::ConvexHull, geom_index, resolved_friction);
            continue;
        }

        // Auto / Force => run V-HACD. (A convex input naturally yields 1 piece,
        // so Auto ~= Force at this phase.) Served through the content-hash cache
        // (p06): identical (mesh, params) reuse a prior decomposition instead of
        // re-running the 0.5-5s V-HACD.
        import::cooker::ConvexDecompositionParams params;
        params.max_pieces = s.decompose_max_pieces;
        bool decompose_hit = false;
        const auto result = import::cooker::DecomposeMeshCached(
            s.mesh_vertices.data(),
            static_cast<uint32_t>(s.mesh_vertices.size() / 3),
            s.mesh_indices.data(),
            static_cast<uint32_t>(s.mesh_indices.size() / 3),
            params,
            kDecomposeCacheDir,
            &decompose_hit);
        (void)decompose_hit;

        if (!result.succeeded || result.pieces.empty()) {
            // Decomposition failed: fall back to passing the mesh through as a
            // single ConvexHull carrying its own geometry (never drop the shape).
            const uint32_t geom_index = AppendConvexGeometry(
                blob.convex_geometry, s.mesh_vertices, s.mesh_indices, 0.0f);
            PushShapeRow(blob.shapes, blob.contact_params, s,
                         ShapeType::ConvexHull, geom_index, resolved_friction);
            continue;
        }

        for (const auto& piece : result.pieces) {
            const uint32_t geom_index = AppendConvexGeometry(
                blob.convex_geometry, piece.vertices, piece.indices, piece.volume);
            PushShapeRow(blob.shapes, blob.contact_params, s,
                         ShapeType::ConvexHull, geom_index, resolved_friction);
        }
    }

    blob.shape_count = static_cast<uint32_t>(blob.shapes.types.size());

    // v0.8 C1c: bake the filtered-pair policy now that bodies / joints / shapes
    // are cooked (excludes derive from joints; explicit pairs from shapes).
    BuildFilteredPairPolicy(scene, blob);

    // v0.7 p07: cook a narrow-band SDF per unique convex-geometry piece,
    // deduplicated by content hash. The leaf cooker stays dependency-free, so
    // the dedup/byte stats are returned here; we SURFACE an over-budget scene on
    // stderr (R-C ~80MB cap). The detailed stats remain reachable via
    // CookedSdfTable (Count() vs piece count; sum of per-cell bytes) for callers
    // / tests that want them.
    if (options.bake_sdf) {
        uint32_t unique = 0, total = 0;
        uint64_t bytes = 0;
        CookSdfsForGeometry(blob.convex_geometry, blob.sdfs, &unique, &total, &bytes);
        constexpr uint64_t kSdfMemoryCapBytes = 80ull * 1024ull * 1024ull;  // R-C
        if (bytes > kSdfMemoryCapBytes) {
            std::fprintf(stderr,
                         "[nuka cooker] WARNING: cooked SDF memory %llu bytes "
                         "(%u unique of %u pieces) exceeds the ~80MB cap (R-C); "
                         "consider coarser voxel_size or more aggressive sharing.\n",
                         static_cast<unsigned long long>(bytes), unique, total);
        }
    }

    // Per-body visual-mesh SDF (the tight silhouette for particle/MPM/rigid
    // contact). A no-op for primitive-only scenes (no visual trimesh to bake).
    if (options.bake_link_sdf) {
        CookLinkVisualSdfs(scene, blob);
    }

    const auto& sensors = scene.Sensors();
    blob.sensor_count = static_cast<uint32_t>(sensors.size());
    blob.sensors.types.reserve(sensors.size());
    blob.sensors.attached_bodies.reserve(sensors.size());
    blob.sensors.local_transforms.reserve(sensors.size());
    blob.sensors.sample_rates_hz.reserve(sensors.size());

    for (const auto& s : sensors) {
        blob.sensors.types.push_back(s.type);
        blob.sensors.attached_bodies.push_back(s.mount_index);
        blob.sensors.local_transforms.push_back(s.local_offset);
        blob.sensors.sample_rates_hz.push_back(s.sample_rate_hz);
    }

    const auto& materials = scene.Materials();
    blob.material_count = static_cast<uint32_t>(materials.size());
    blob.materials.base_colors.reserve(materials.size());
    blob.materials.alphas.reserve(materials.size());
    blob.materials.roughnesses.reserve(materials.size());
    blob.materials.metallics.reserve(materials.size());

    for (const auto& m : materials) {
        blob.materials.base_colors.push_back(m.base_color);
        blob.materials.alphas.push_back(m.alpha);
        blob.materials.roughnesses.push_back(m.roughness);
        blob.materials.metallics.push_back(m.metallic);
    }

    const auto& cameras = scene.Cameras();
    blob.camera_count = static_cast<uint32_t>(cameras.size());
    blob.cameras.attached_bodies.reserve(cameras.size());
    blob.cameras.local_transforms.reserve(cameras.size());
    blob.cameras.vertical_fovs_degrees.reserve(cameras.size());
    blob.cameras.near_clips.reserve(cameras.size());
    blob.cameras.far_clips.reserve(cameras.size());

    for (const auto& c : cameras) {
        blob.cameras.attached_bodies.push_back(c.attached_body);
        blob.cameras.local_transforms.push_back(c.local_transform);
        blob.cameras.vertical_fovs_degrees.push_back(c.vertical_fov_degrees);
        blob.cameras.near_clips.push_back(c.near_clip);
        blob.cameras.far_clips.push_back(c.far_clip);
    }

    const auto& lights = scene.Lights();
    blob.light_count = static_cast<uint32_t>(lights.size());
    blob.lights.types.reserve(lights.size());
    blob.lights.attached_bodies.reserve(lights.size());
    blob.lights.local_transforms.reserve(lights.size());
    blob.lights.colors.reserve(lights.size());
    blob.lights.intensities.reserve(lights.size());

    for (const auto& l : lights) {
        blob.lights.types.push_back(l.type);
        blob.lights.attached_bodies.push_back(l.attached_body);
        blob.lights.local_transforms.push_back(l.local_transform);
        blob.lights.colors.push_back(l.color);
        blob.lights.intensities.push_back(l.intensity);
    }

    const auto& actuators = scene.Actuators();
    blob.actuator_count = static_cast<uint32_t>(actuators.size());
    blob.actuators.types.reserve(actuators.size());
    blob.actuators.joint_ids.reserve(actuators.size());
    blob.actuators.gains.reserve(actuators.size());
    blob.actuators.force_limits.reserve(actuators.size());

    for (const auto& a : actuators) {
        blob.actuators.types.push_back(a.type);
        blob.actuators.joint_ids.push_back(a.joint_id);
        blob.actuators.gains.push_back(a.gain);
        blob.actuators.force_limits.push_back(a.force_limit);
    }

    return blob;
}

} // namespace nuka::scene
