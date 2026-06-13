#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::app::PosePublisher - the physics->render pose bridge (M8 #6).
//
// Each frame the TransformSyncSystem asks a PosePublisher to refresh every
// RenderInstance's `world_xform` from the live physics state. A PosePublisher
// reads the SELECTED env's FK poses out of an nk::World and composes them with
// each instance's build-time `cached_visual_local` (Decision D1):
//
//     instance.world_xform = downloaded_pose * cached_visual_local
//
// where `downloaded_pose` is the FK WORLD pose of the link/body frame the
// instance is bound to (RenderInstance::pose_source), read for env `env_index`.
// `math::Transform::operator*` is parent*child (apply rhs first), so this places
// the visual node at downloaded_pose then offsets it by the physics-frame->
// visual-frame chain render_world.cpp baked into cached_visual_local -- the exact
// 1:1 contract the M8 parity gate (G1) asserts.
//
// PosePublisher is an ABSTRACT interface so the publish SOURCE can swap without
// any caller change. M8 ships HostDownloadPublisher (host-download path via
// nk::World::GetData().DownloadField -- the only host-readback seam, data.hpp:58).
// M11's CudaVulkanInteropPublisher (device-scatter, zero D2H copy) implements the
// SAME interface and drops in behind this same abstraction with no edit to
// TransformSyncSystem / Simulation (recon Risk R5). Callers therefore depend on
// PosePublisher&, never on the concrete type.
//
// HOST-ONLY / zero-CUDA-token (this dir is under the src/runtime/app/** lint
// red-line): device pose access goes ONLY through Data::DownloadField (host).
// No triple-chevron kernel launch, no cuda-runtime call, no cuda_runtime include,
// no phi backend_cuda include anywhere in runtime/app.
// ---------------------------------------------------------------------------

#include "render/render_world.hpp"

#include <cstdint>
#include <vector>

namespace nuka::nk {
class World;
}  // namespace nuka::nk

namespace nuka::runtime::app {

// ---------------------------------------------------------------------------
// PosePublisher - abstract: refresh a RenderWorld's live transforms from the
// physics World for one env.
// ---------------------------------------------------------------------------
class PosePublisher {
public:
    virtual ~PosePublisher() = default;

    // Overwrite every render instance's (and camera's / light's) `world_xform`
    // with `downloaded_pose * cached_visual_local`, reading the FK pose of each
    // instance's pose_source row for `env_index`. Static instances keep their
    // bind-pose world_xform untouched.
    //
    // ENV-SELECTION (WRAP) CONTRACT: `env_index` is interpreted MODULO the World's
    // env_count -- the published env is `(env_count > 0) ? env_index % env_count
    // : 0`. So any out-of-range index wraps rather than reading out of bounds, and
    // env_count==0 publishes nothing meaningful. Implementations MUST apply this
    // same modulo, and any verifier that independently re-downloads a pose row for
    // comparison MUST select the SAME wrapped env (verifier and implementation
    // cannot diverge). The parity gate applies the identical modulo for this
    // reason. (Benign today: every shipped world is env_count==1 so the modulo is a
    // no-op; the contract guards an M11 device-scatter publisher.)
    virtual void Publish(const nk::World& world, uint32_t env_index,
                         render::RenderWorld& render_world) = 0;
};

// ---------------------------------------------------------------------------
// HostDownloadPublisher - the M8 publisher: host-download the SELECTED env's
// pose fields and compose into the RenderWorld (Decision D2).
//
// Implementation (pose_publisher.cpp): per Publish, it batches ONE
// Data::DownloadField per pose field (the whole LinkPose array for the env, the
// whole BodyPose array, the env's BasePose) into reusable host staging buffers,
// then for each instance maps PoseSource::Kind -> the right staging buffer + row
// and composes. This is one D2H copy per populated field per frame, NOT one tiny
// copy per instance (the manifest's batch-download optimization).
//
// The staging buffers persist across frames (hot-path allocation-free after the
// first frame), so a long rollout does not churn the heap.
// ---------------------------------------------------------------------------
class HostDownloadPublisher : public PosePublisher {
public:
    void Publish(const nk::World& world, uint32_t env_index,
                 render::RenderWorld& render_world) override;

private:
    // Per-field host staging mirrors of the selected env's pose rows. Sized lazily
    // from the World's capacities on the first Publish and reused thereafter.
    std::vector<math::Transform> link_poses_;  // env's LinkPose rows (per link)
    std::vector<math::Transform> body_poses_;  // env's BodyPose rows (per body)
    math::Transform              base_pose_ = math::Transform::Identity();  // env BasePose
    bool                         have_link_ = false;
    bool                         have_body_ = false;
    bool                         have_base_ = false;
};

}  // namespace nuka::runtime::app
