// ---------------------------------------------------------------------------
// nuka::runtime::app::HostDownloadPublisher (M8 manifest #7).
//
// Host-download the SELECTED env's pose fields out of an nk::World and compose
// them into a RenderWorld (Decision D1 / D2):
//
//     instance.world_xform = downloaded_pose * cached_visual_local
//
// `downloaded_pose` is the FK WORLD pose of the link/body frame the instance is
// bound to (RenderInstance::pose_source), read for the requested env. The pose
// fields are env-major: row (env e, local row r) sits at flat index
// e * per_env_count + r. We batch ONE DownloadField per pose field per frame
// (the whole env's LinkPose rows, the whole env's BodyPose rows, the env's
// single BasePose) into persistent host staging buffers, then index those for
// every instance -- one D2H copy per populated field, never one per instance.
//
// HOST-ONLY / zero-CUDA-token: device pose access is ONLY through
// Data::DownloadField (host). No kernel launch, no cuda-runtime call, no
// cuda_runtime / phi backend_cuda include.
// ---------------------------------------------------------------------------

#include "runtime/app/pose_publisher.hpp"

#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"

#include <cstdint>

namespace nuka::runtime::app {

// PoseSource::Kind maps to the nk Data pose field as: Link->LinkPose,
// Body->BodyPose, Base->BasePose (Static carries no field -- the instance keeps
// its bind pose). The `resolve` lambda below applies that map directly off the
// per-field staging buffers downloaded once at the top of Publish.

void HostDownloadPublisher::Publish(const nk::World& world, uint32_t env_index,
                                    render::RenderWorld& render_world) {
    const nk::ModelCapacities& caps = world.GetModel().capacities;
    const nk::Data& data = const_cast<nk::World&>(world).GetData();

    const uint32_t env =
        (caps.env_count > 0u) ? (env_index % caps.env_count) : 0u;
    const uint64_t kTf = sizeof(math::Transform);

    // -- batch the per-field env downloads ONCE -----------------------------
    // LinkPose: links_per_env rows for this env, contiguous from the env's base.
    have_link_ = false;
    if (caps.links_per_env > 0u) {
        link_poses_.resize(caps.links_per_env);
        const uint64_t off =
            static_cast<uint64_t>(env) * caps.links_per_env * kTf;
        have_link_ = data.DownloadField(nk::FieldId::LinkPose, link_poses_.data(),
                                        static_cast<uint64_t>(caps.links_per_env) * kTf,
                                        off);
    }
    // BodyPose: bodies_per_env rows for this env.
    have_body_ = false;
    if (caps.bodies_per_env > 0u) {
        body_poses_.resize(caps.bodies_per_env);
        const uint64_t off =
            static_cast<uint64_t>(env) * caps.bodies_per_env * kTf;
        have_body_ = data.DownloadField(nk::FieldId::BodyPose, body_poses_.data(),
                                        static_cast<uint64_t>(caps.bodies_per_env) * kTf,
                                        off);
    }
    // BasePose: per:env -> one Transform at the env's slot.
    base_pose_ = math::Transform::Identity();
    have_base_ = data.DownloadField(nk::FieldId::BasePose, &base_pose_, kTf,
                                    static_cast<uint64_t>(env) * kTf);

    // Resolve one instance's live FK pose from the staged buffers. Returns false
    // (caller keeps the bind pose) for Static or an out-of-range / unavailable row.
    auto resolve = [&](const render::PoseSource& src,
                       math::Transform* out) -> bool {
        switch (src.kind) {
            case render::PoseSource::Kind::Link:
                if (!have_link_ || src.row >= link_poses_.size()) return false;
                *out = link_poses_[src.row];
                return true;
            case render::PoseSource::Kind::Body:
                if (!have_body_ || src.row >= body_poses_.size()) return false;
                *out = body_poses_[src.row];
                return true;
            case render::PoseSource::Kind::Base:
                if (!have_base_) return false;
                *out = base_pose_;
                return true;
            case render::PoseSource::Kind::Static:
            default:
                return false;
        }
    };

    // -- compose: world_xform = downloaded_pose * cached_visual_local --------
    for (render::RenderInstance& inst : render_world.instances) {
        math::Transform fk;
        if (resolve(inst.pose_source, &fk)) {
            inst.world_xform = fk * inst.cached_visual_local;  // Decision D1.
        }
        // Static / unresolved: keep the build-time bind-pose world_xform.
    }
    for (render::RenderCamera& cam : render_world.cameras) {
        math::Transform fk;
        if (resolve(cam.pose_source, &fk)) {
            cam.world_xform = fk * cam.cached_visual_local;
        }
    }
    for (render::RenderLight& light : render_world.lights) {
        math::Transform fk;
        if (resolve(light.pose_source, &fk)) {
            light.world_xform = fk * light.cached_visual_local;
        }
    }
}

}  // namespace nuka::runtime::app
