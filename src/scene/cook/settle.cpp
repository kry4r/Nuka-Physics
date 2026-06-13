// ---------------------------------------------------------------------------
// nuka::scene::cook — Settle implementation (M7 T2).
// ---------------------------------------------------------------------------

#include "scene/cook/settle.hpp"

#include "nk/model/generated/field_ids.hpp"
#include "phi/backend.hpp"       // phi::NkOp, phi::SnapshotStateParams
#include "scene/cook/placement.hpp"  // BodyWorldTransform (child->parent convert)
#include "scene/graph/scene_graph.hpp"

namespace nuka::scene::cook {

namespace {

namespace nkm = nuka::nk;

// Glob match supporting '*' (any run), '?' (one char), and the documented
// regex-flavored ".*" (treated as a single wildcard run — a '.' immediately
// followed by '*' is collapsed to '*', so "h1/.*" and "h1/*" both match
// "h1/<link>"). Iterative two-pointer with backtracking; O(n*m) worst case,
// deterministic, no <regex>.
bool GlobMatch(const char* str, const char* pat) {
    const char* star_pat = nullptr;
    const char* star_str = nullptr;
    while (*str) {
        if (*pat == '.' && *(pat + 1) == '*') {
            // ".*" -> wildcard run (regex-flavored authoring, e.g. "h1/.*").
            star_pat = pat;
            star_str = str;
            pat += 2;
            continue;
        }
        if (*pat == '*') {
            star_pat = pat++;
            star_str = str;
            continue;
        }
        if (*pat == '?' || *pat == *str) {
            ++pat;
            ++str;
            continue;
        }
        if (star_pat) {
            pat = (*star_pat == '.') ? star_pat + 2 : star_pat + 1;
            str = ++star_str;
            continue;
        }
        return false;
    }
    while (*pat == '*' || (*pat == '.' && *(pat + 1) == '*')) {
        pat += (*pat == '.') ? 2 : 1;
    }
    return *pat == '\0';
}

}  // namespace

bool PathMatchesPattern(const std::string& path, const std::string& pattern) {
    return GlobMatch(path.c_str(), pattern.c_str());
}

SettleResult Settle(nk::World& world, const SceneIR& scene, const SceneMap& map,
                    const SettleSpec& spec) {
    SettleResult result;
    if (!world.Ready()) {
        return result;  // ok == false (honest: nothing settled).
    }

    const nkm::ModelCapacities& cap = world.GetModel().capacities;
    const uint32_t L = cap.links_per_env;
    const uint32_t E = cap.env_count;
    const uint32_t B = cap.bodies_per_env;

    // -- 1. Resolve every Hold pattern -> the matched LINK indices + the gains to
    // apply. Iterate link slots in ascending order (deterministic): each slot ->
    // its entity (SceneMap) -> its derived tree path (SceneIR) -> the FIRST
    // matching Hold (spec order) supplies the gains. (Per-link kp/kd captured
    // once so step 2 does no path lookups.)
    struct HoldGain { uint32_t link; float kp; float kd; };
    std::vector<HoldGain> held_links;
    if (L > 0 && !spec.holds.empty()) {
        const SceneGraph& tree = scene.Tree();
        const Registry& ecs = scene.Ecs();
        for (uint32_t link = 0; link < L; ++link) {
            const EntityId ent = map.EntityOfLink(link);
            if (ent == kInvalidEntity) continue;
            const auto node = ecs.NodeOf(ent);
            if (!node) continue;
            const std::string path = tree.PathOf(node);
            for (const SettleSpec::Hold& h : spec.holds) {
                if (PathMatchesPattern(path, h.dof_pattern)) {
                    held_links.push_back({link, h.kp, h.kd});
                    break;
                }
            }
        }
    }

    // -- 2. Seed the PD hold: pin held links' drive target to their CURRENT q
    // and set firm gains. Read the env-major drive/q arrays, patch the matched
    // links across EVERY env identically (env-major layout [e*L + l]), re-upload.
    if (L > 0 && !held_links.empty() && E > 0) {
        const size_t n = static_cast<size_t>(L) * E;
        std::vector<float> q(n, 0.0f), target(n, 0.0f), stiff(n, 0.0f),
            damp(n, 0.0f);
        const bool dl =
            world.GetData().DownloadField(nkm::FieldId::Q, q.data(),
                                          n * sizeof(float)) &&
            world.GetData().DownloadField(nkm::FieldId::DriveTarget, target.data(),
                                          n * sizeof(float)) &&
            world.GetData().DownloadField(nkm::FieldId::DriveStiffness,
                                          stiff.data(), n * sizeof(float)) &&
            world.GetData().DownloadField(nkm::FieldId::DriveDamping, damp.data(),
                                          n * sizeof(float));
        if (!dl) return result;  // ok == false.
        for (uint32_t e = 0; e < E; ++e) {
            for (const HoldGain& g : held_links) {
                const size_t at = static_cast<size_t>(e) * L + g.link;
                // Use env 0's seeded q as the canonical hold target for every env
                // (the template is env-replicated; this keeps all envs identical
                // and the settle byte-deterministic regardless of e).
                target[at] = q[static_cast<size_t>(g.link)];
                stiff[at] = g.kp;
                damp[at] = g.kd;
            }
        }
        const bool ul =
            world.GetData().UploadField(nkm::FieldId::DriveTarget, target.data(),
                                        n * sizeof(float)) &&
            world.GetData().UploadField(nkm::FieldId::DriveStiffness, stiff.data(),
                                        n * sizeof(float)) &&
            world.GetData().UploadField(nkm::FieldId::DriveDamping, damp.data(),
                                        n * sizeof(float));
        if (!ul) return result;
    }

    // -- 3. Deterministic device settle. Step() (the byte-deterministic per-op
    // path) `spec.steps` times. (StepPlanned == Step for the rigid/articulated
    // settle scenes; Step is the portable choice that also covers any thrust-
    // bearing scene — determinism is identical either way for these scenes.)
    for (int s = 0; s < spec.steps; ++s) {
        world.Step();
    }

    // -- 4. Snapshot the settled state (so the settled q/base/body pose becomes
    // the Reset-restore source: a later SnapshotState was already taken at ctor;
    // this OVERWRITES it with the settled state — relying on T1's snapshot_body_*
    // + snapshot_q/base coverage).
    phi::SnapshotStateParams snap{};
    snap.total_link_count = L * E;
    snap.env_count = E;
    snap.total_body_count = B * E;
    if (world.DispatchOp(phi::NkOp::SnapshotState, &snap) != phi::Status::Ok) {
        return result;  // ok == false.
    }

    // -- 5. Download the settled IC (env 0) in a FIXED field order: q (per-link),
    // base pose (per-env), then movable-body poses (per-body). All env 0 slices.
    if (L > 0) {
        result.initial_state.qpos.assign(L, 0.0f);
        if (!world.GetData().DownloadField(nkm::FieldId::Q,
                                           result.initial_state.qpos.data(),
                                           static_cast<uint64_t>(L) * sizeof(float))) {
            return result;
        }
        math::Transform base = math::Transform::Identity();
        if (!world.GetData().DownloadField(nkm::FieldId::BasePose, &base,
                                           sizeof(math::Transform))) {
            return result;
        }
        result.initial_state.root = base;
    }

    if (B > 0) {
        std::vector<math::Transform> poses(B);
        if (!world.GetData().DownloadField(nkm::FieldId::BodyPose, poses.data(),
                                           static_cast<uint64_t>(B) *
                                               sizeof(math::Transform))) {
            return result;
        }
        // Key each settled pose by the body's EntityId (ascending body row order
        // -> deterministic). A body with no entity binding is still emitted under
        // kInvalidEntity so the gate can compare the full pose vector.
        result.body_poses.reserve(B);
        for (uint32_t b = 0; b < B; ++b) {
            result.body_poses.emplace_back(map.EntityOfBody(b), poses[b]);
        }
    }

    result.ok = true;
    return result;
}

void ApplySettleToRegistry(Registry& registry, EntityId articulation_entity,
                           const SettleResult& result) {
    if (articulation_entity != kInvalidEntity &&
        registry.Alive(articulation_entity)) {
        registry.Add(articulation_entity, result.initial_state);
    }
    for (const auto& [entity, pose] : result.body_poses) {
        if (entity == kInvalidEntity || !registry.Alive(entity)) continue;
        TransformComponent tc;
        if (const TransformComponent* existing =
                registry.Get<TransformComponent>(entity)) {
            tc = *existing;  // preserve scale.
        }
        tc.local = pose;     // settled world pose (root bodies live at world).
        registry.Add(entity, tc);
    }
}

void ApplySettleToSceneIR(SceneIR& scene, const SettleResult& result,
                          const SceneMap& map) {
    for (const auto& [entity, pose] : result.body_poses) {
        if (entity == kInvalidEntity) continue;
        const CookedRef* ref = map.RefOf(entity);
        if (!ref || ref->body_row == SceneMap::kNoRow ||
            ref->body_row >= scene.RigidBodyCount()) {
            continue;
        }
        const BodyId body = static_cast<BodyId>(ref->body_row);
        RigidBodyRecord& rec = scene.GetBodyMut(body);
        if (rec.parent_id == kInvalidBody) {
            // Root body: the settled WORLD pose IS the record local_transform.
            rec.local_transform = pose;
        } else {
            // Child body: convert the settled world pose into the parent frame.
            const math::Transform parent_world =
                BodyWorldTransform(scene, rec.parent_id);
            rec.local_transform = parent_world.Inverse() * pose;
        }
    }
}

}  // namespace nuka::scene::cook
