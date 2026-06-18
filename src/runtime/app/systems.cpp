// ---------------------------------------------------------------------------
// nuka::runtime::app -- the frame-loop systems (M8 manifest #9).
//
// SimSystem (nk::World step) and RenderSystem (offscreen raster). InputSystem +
// TransformSyncSystem are header-only (trivial bodies in systems.hpp).
//
// RenderSystem's draw call is the ONLY place this library touches nuka_render's
// VulkanRasterRenderer::Render symbol; it is gated behind NK_BUILD_VULKAN_VALIDATION
// (Decision D5) so a Vulkan-less build still links -- Run is then a no-op and the
// caller (Simulation) never attaches a renderer in that configuration anyway.
//
// HOST-ONLY / zero-CUDA-token (src/runtime/app/** lint red-line): the physics
// step goes through nk::World (whose device work is behind the phi v2 vtable);
// no kernel launch, no cuda-runtime call, no cuda_runtime / phi backend_cuda
// include appears here.
// ---------------------------------------------------------------------------

#include "runtime/app/systems.hpp"

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "render/render_world.hpp"

#include <cstdint>

namespace nuka::runtime::app {

namespace {

// VIEW-F2: the u8 value of ArticulationJointType::FloatingBase. The canonical enum
// lives in runtime/articulation/articulation_state.hpp (==3), but that header pulls
// in <cuda_runtime.h> -- and src/runtime/app/** is on the zero-CUDA-token red-line,
// so we cannot include it here. The Model's joint_type field is documented u8-backed
// (model.hpp:84) with FloatingBase==3 (articulation_state.hpp:32 / its CUDA-side
// VERBATIM copy articulation_types.cuh). If that enum ever renumbers, this constant
// must follow (the value is part of the cooked Model byte contract, so it is stable).
constexpr uint8_t kFloatingBaseJointType = 3u;

// Find the RenderInstance bound to `entity` (the same kind/row the publisher
// reads, plus its cached_visual_local for the VIEW-F3 inverse). Returns nullptr if
// no instance owns the entity.
const render::RenderInstance* InstanceOf(const render::RenderWorld& rw,
                                         scene::EntityId entity) {
    for (const render::RenderInstance& inst : rw.instances) {
        if (inst.entity == entity) return &inst;
    }
    return nullptr;
}

}  // namespace

bool ApplyMoveEntity(nk::World& world, const render::RenderWorld& render_world,
                     uint32_t env_index, const MoveEntity& cmd) {
    if (cmd.entity == scene::kInvalidEntity) return false;
    const render::RenderInstance* inst = InstanceOf(render_world, cmd.entity);
    if (inst == nullptr) return false;
    const render::PoseSource* src = &inst->pose_source;

    const nk::ModelCapacities& caps = world.GetModel().capacities;
    const uint32_t env =
        (caps.env_count > 0u) ? (env_index % caps.env_count) : 0u;
    nk::Data& data = world.GetData();

    const uint64_t kTf = sizeof(math::Transform);
    const uint64_t kV3 = sizeof(math::Vec3);
    const math::Vec3 zero3 = math::Vec3::Zero();

    // VIEW-F3: cmd.world_xform is a VISUAL-frame transform (the drag site uses
    // inst.world_xform == fk * cached_visual_local). The pose FIELDS (BodyPose/
    // BasePose) hold the PHYSICS-frame transform the publisher re-multiplies cvl
    // onto next frame. So strip the cached_visual_local first:
    //   physics_pose = cmd.world_xform * Inverse(cached_visual_local)
    // For an identity cvl (the cup) this is a no-op (byte-identical to before); for
    // an offset visual node it prevents the double-applied / corrupted pose.
    const math::Transform physics_pose =
        cmd.world_xform * inst->cached_visual_local.Inverse();

    switch (src->kind) {
        case render::PoseSource::Kind::Body: {
            // Movable free rigid body (e.g. the cup): the pose IS its state, so the
            // teleport is a genuine live edit. Env-major row e*B + r.
            const uint32_t B = caps.bodies_per_env;
            if (B == 0u || src->row >= B) return false;
            const uint64_t at = static_cast<uint64_t>(env) * B + src->row;
            if (!data.UploadField(nk::FieldId::BodyPose, &physics_pose, kTf,
                                  at * kTf)) {
                return false;
            }
            // OD-6: zero linear + angular velocity (place-here, no throw).
            data.UploadField(nk::FieldId::BodyLinearVelocity, &zero3, kV3, at * kV3);
            data.UploadField(nk::FieldId::BodyAngularVelocity, &zero3, kV3, at * kV3);
            return true;
        }
        case render::PoseSource::Kind::Base: {
            // Articulation FLOATING base: BasePose drives the root's 6 FK columns
            // (assemble_rows base_pose). Per:env -> one Transform at the env slot.
            if (!data.UploadField(nk::FieldId::BasePose, &physics_pose, kTf,
                                  static_cast<uint64_t>(env) * kTf)) {
                return false;
            }
            // OD-6: zero the root link's 6-float spatial velocity (env-major,
            // per:link, 6 floats/link). The root link is articulation root_link.
            const nk::ModelArticulation& art = world.GetModel().articulation;
            const uint32_t L = caps.links_per_env;
            const uint32_t root = (art.root_link != ~uint32_t(0)) ? art.root_link : 0u;
            if (L > 0u && root < L) {
                // LinkVelocity is typed Spatial6 -> size the upload by the field's
                // own type so any future padding is encoded correctly (not by 6).
                const nk::Spatial6 zero6 = {};
                const uint64_t row = static_cast<uint64_t>(env) * L + root;
                data.UploadField(nk::FieldId::LinkVelocity, &zero6, sizeof(zero6),
                                 row * sizeof(zero6));
            }
            return true;
        }
        case render::PoseSource::Kind::Link: {
            // An articulation link whose pose is an FK OUTPUT (not an independent
            // Data row). Only the floating ROOT link is a meaningful live move; it
            // is driven via BasePose, so re-route a root-link instance to Base.
            const nk::ModelArticulation& art = world.GetModel().articulation;
            const uint32_t root = (art.root_link != ~uint32_t(0)) ? art.root_link : 0u;
            // VIEW-F2: the REAL discriminator is the root link's joint type, not a
            // dof_count>=6 heuristic (a fixed-base 6+-DOF arm would falsely pass).
            const bool floating =
                root < art.joint_type.size() &&
                art.joint_type[root] == kFloatingBaseJointType;
            if (src->row == root && floating) {
                if (!data.UploadField(nk::FieldId::BasePose, &physics_pose, kTf,
                                      static_cast<uint64_t>(env) * kTf)) {
                    return false;
                }
                const uint32_t L = caps.links_per_env;
                if (L > 0u && root < L) {
                    // Size by the LinkVelocity field's own Spatial6 type (not 6).
                    const nk::Spatial6 zero6 = {};
                    const uint64_t row = static_cast<uint64_t>(env) * L + root;
                    data.UploadField(nk::FieldId::LinkVelocity, &zero6, sizeof(zero6),
                                     row * sizeof(zero6));
                }
                return true;
            }
            // Interior link: no-op (FK-derived, can't be teleported live).
            return false;
        }
        case render::PoseSource::Kind::Static:
        default:
            return false;
    }
}

bool SimSystem::Run(nk::World& world, bool planned) {
    if (planned) {
        return world.StepPlanned() == phi::Status::Ok;
    }
    return world.Step().AllOk();
}

bool RenderSystem::Run(render::VulkanRasterRenderer* renderer,
                       const render::RenderWorld& world,
                       const render::RasterOptions& options,
                       render::VulkanOffscreenReport* out_report) {
    if (renderer == nullptr || out_report == nullptr) {
        return false;
    }
#ifdef NK_BUILD_VULKAN_VALIDATION
    *out_report = renderer->Render(world, options);
    return true;
#else
    // Built without the Vulkan validation toolchain: the renderer symbol is not
    // linked into this library. RenderSystem is inert; the render-off frame loop
    // (Simulation with render disabled) never reaches here, and a render-on build
    // turns this on by defining NK_BUILD_VULKAN_VALIDATION.
    (void)renderer;
    (void)world;
    (void)options;
    return false;
#endif
}

}  // namespace nuka::runtime::app
