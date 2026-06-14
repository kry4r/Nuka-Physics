// ---------------------------------------------------------------------------
// nuka::runtime::app::CudaVulkanInteropPublisher (M11 §3.E INT-6, OD-2=(A)).
//
// Device-scatter the SELECTED env's composed object->world transforms into a
// Vulkan-imported SSBO with NO device->host copy (the zero-copy render<->physics
// shared-memory path). Implements the abstract PosePublisher; drops behind the M8
// seam with no caller change.
//
// HOST-ONLY / ZERO CUDA TOKEN (src/runtime/app/** red-line, m11-recon §R11): all
// device work routes through the CUDA-FREE phi::InteropScatterI seam. The FK device
// pointers are read via nk::Data::Ptr(FieldId) (a plain void* device address -- the
// SAME buffer DownloadField copies from, so the scattered transform is bit-exact to
// the host path, R14). No kernel launch / cuda call / backend_cuda include here.
// ---------------------------------------------------------------------------

#include "runtime/app/cuda_vulkan_interop.hpp"

#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"

#include <cstdint>

namespace nuka::runtime::app {

CudaVulkanInteropPublisher::CudaVulkanInteropPublisher() = default;
CudaVulkanInteropPublisher::~CudaVulkanInteropPublisher() = default;

// Map render::PoseSource::Kind -> the scatter kind int (0=Static,1=Link,2=Body,
// 3=Base), matching cuda_vk_scatter.cu's switch. Keep in sync with that kernel.
namespace {
uint32_t KindToInt(render::PoseSource::Kind k) {
    switch (k) {
        case render::PoseSource::Kind::Link: return 1u;
        case render::PoseSource::Kind::Body: return 2u;
        case render::PoseSource::Kind::Base: return 3u;
        case render::PoseSource::Kind::Static:
        default: return 0u;
    }
}
}  // namespace

void CudaVulkanInteropPublisher::BuildRows(const render::RenderWorld& render_world) {
    rows_.clear();
    rows_.reserve(render_world.instances.size());
    for (const render::RenderInstance& inst : render_world.instances) {
        phi::InstanceScatterRow row{};
        row.kind = KindToInt(inst.pose_source.kind);
        row.row  = inst.pose_source.row;
        // cached_visual_local laid out exactly as math::Transform's bytes are read
        // back on device: pos(x,y,z) then quat(w,x,y,z).
        const math::Transform& cvl = inst.cached_visual_local;
        row.cached_visual_local[0] = cvl.position.x;
        row.cached_visual_local[1] = cvl.position.y;
        row.cached_visual_local[2] = cvl.position.z;
        row.cached_visual_local[3] = cvl.rotation.w;
        row.cached_visual_local[4] = cvl.rotation.x;
        row.cached_visual_local[5] = cvl.rotation.y;
        row.cached_visual_local[6] = cvl.rotation.z;
        rows_.push_back(row);
    }
    instance_count_ = static_cast<uint32_t>(rows_.size());
}

bool CudaVulkanInteropPublisher::TryEnable(const nk::World& world,
                                           const render::RenderWorld& render_world,
                                           const phi::ExternalMemoryDesc& ssbo_mem,
                                           uint32_t instance_capacity) {
    enabled_ = false;

    // 1. A live CUDA scatter (nullptr when no CUDA device OR the interop CUDA lib is
    //    not linked -> the weak fallback returns nullptr). Degrade on either.
    scatter_ = phi::CreateCudaVkScatter();
    if (!scatter_) return false;

    // 2. The World must be built (FK buffers exist) -- a defensive guard; the
    //    publisher reads its Data device pointers each Publish.
    if (!world.Ready()) { scatter_.reset(); return false; }

    // 3. Import the exported Vulkan SSBO. On lavapipe the renderer never exported a
    //    valid fd (external memory unsupported) -> ssbo_mem.fd < 0 -> false.
    if (!scatter_->ImportMemory(ssbo_mem, instance_capacity)) {
        scatter_.reset();
        return false;
    }

    // 4. Bindings are build-time constant; lower them once.
    BuildRows(render_world);

    enabled_ = true;
    return true;
}

bool CudaVulkanInteropPublisher::ImportSemaphores(const phi::ExternalSemaphoreDesc& sem) {
    if (!scatter_) return false;
    return scatter_->ImportSemaphores(sem);
}

void CudaVulkanInteropPublisher::Publish(const nk::World& world, uint32_t env_index,
                                         render::RenderWorld& render_world) {
    if (!enabled_ || !scatter_) return;
    (void)render_world;  // host world_xform is NOT touched on the device path; the
                         // instanced pipeline reads the SSBO the scatter writes.

    const nk::ModelCapacities& caps = world.GetModel().capacities;
    const uint32_t env =
        (caps.env_count > 0u) ? (env_index % caps.env_count) : 0u;

    // Read the SAME live FK device field buffers DownloadField reads -- a plain
    // void* device address per field (host-side accessor, no CUDA token). A field
    // the cook didn't populate returns nullptr; the kernel keeps those rows at the
    // pre-seeded bind pose (mirrors HostDownloadPublisher's have_*_ guards).
    nk::Data& data = const_cast<nk::World&>(world).GetData();
    phi::ScatterFkSource fk{};
    fk.link_pose      = data.Ptr(nk::FieldId::LinkPose);
    fk.body_pose      = data.Ptr(nk::FieldId::BodyPose);
    fk.base_pose      = data.Ptr(nk::FieldId::BasePose);
    fk.env_index      = env;
    fk.links_per_env  = caps.links_per_env;
    fk.bodies_per_env = caps.bodies_per_env;
    // INT-F2: the backend the World ran its FK ops on (opaque, CUDA-free). The CUDA
    // scatter orders itself AFTER the FK write on this backend's main stream via an
    // event, so it never reads torn/stale transforms (cross-stream RAW hazard).
    fk.world_backend  = world.Backend();

    scatter_->Scatter(fk, rows_.data(), instance_count_);
}

}  // namespace nuka::runtime::app
