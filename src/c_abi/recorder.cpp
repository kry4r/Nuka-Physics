// ---------------------------------------------------------------------------
// nuka::c_abi -- C ABI for the offscreen RECORDER (M8 T5).
//
// A RecorderRecord owns the whole M8 host frame-loop stack bound to a device:
// the cooked nk::World (Decision D2 -- the recorder reads/steps nk::World, NOT
// the legacy WorldRecord), the RenderWorld built from the SAME cook's Registry +
// SceneMap, a HostDownloadPublisher, a VulkanRasterRenderer, and the Simulation
// that drives them. nuka_recorder_capture runs N Simulation::Frame()s and dumps
// each frame's raw RGBA framebuffer to a binary PPM P6; nuka_recorder_to_video
// shells ffmpeg over the captured sequence.
//
// VULKAN GATING (Decision D5). EVERYTHING that touches the render stack lives
// behind NK_BUILD_VULKAN_VALIDATION. With the flag OFF this TU still compiles
// (it includes NO Vulkan/render headers) and every entry point returns
// NUKA_RESULT_NOT_SUPPORTED -- libnuka stays buildable with zero Vulkan deps and
// the python binding still imports. With the flag ON the real recorder + the
// nuka_app / nuka_render link come in (see src/CMakeLists.txt).
//
// Mirrors the diffsim.cpp / union_world.cpp per-module pattern: a file-local
// HandleTable, null-guard at every entry, try/catch -> MapExceptionToResult.
// ---------------------------------------------------------------------------

#include "nuka/nuka_recorder.h"

#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"

#include <exception>
#include <memory>
#include <new>
#include <stdexcept>

#ifdef NK_BUILD_VULKAN_VALIDATION

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "render/raster/vulkan_raster_renderer.hpp"
#include "render/render_world.hpp"
#include "render/vulkan_offscreen_types.hpp"
#include "runtime/app/pose_publisher.hpp"
#include "runtime/app/simulation.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#endif  // NK_BUILD_VULKAN_VALIDATION

namespace nuka::c_abi {

#ifdef NK_BUILD_VULKAN_VALIDATION

namespace app = nuka::runtime::app;
namespace render = nuka::render;
namespace cook = nuka::scene::cook;
namespace nk = nuka::nk;
namespace nphi = nuka::phi;

// One recorder context. Owns the whole frame-loop stack: the cooked nk::World,
// the RenderWorld (moved into the Simulation), the publisher, the renderer, and
// the Simulation. Order of declaration matters for destruction -- the Simulation
// references the world + publisher + renderer, so it must be destroyed FIRST;
// declare it LAST (members destruct in reverse declaration order).
//
// LIFETIME: the phi v2 Backend created in nuka_recorder_create is OWNED here
// (nk::World only BORROWS it -- World::~World calls BackendPlanFree(backend_) but
// never frees the backend itself). The backend must therefore OUTLIVE the World.
// A C++ destructor BODY runs BEFORE the members are destroyed, so we cannot just
// BackendFree in the body and rely on member-order; we explicitly tear down the
// borrowing members (sim -> world, in dependency order) in the body FIRST, then
// free the backend while it is still valid. The remaining members (publisher /
// renderer / raster_options / scalars) own no backend reference and destruct
// normally afterwards. The phi Device is registry-OWNED (RegistryEntryGetDevice)
// and is NOT freed here.
struct RecorderRecord {
    DeviceRecord* device = nullptr;          // borrowed; must outlive the record
    nphi::Device* phi_device = nullptr;      // registry-owned; do NOT free
    nphi::Backend* backend = nullptr;        // OWNED; freed in dtor AFTER world
    std::unique_ptr<nk::World> world;        // the nk-backed physics world (D2)
    std::unique_ptr<app::HostDownloadPublisher> publisher;
    std::unique_ptr<render::VulkanRasterRenderer> renderer;
    render::RasterOptions raster_options;
    std::unique_ptr<app::Simulation> sim;    // destruct first (declared last)
    uint32_t frame_index = 0u;               // continues across capture() calls

    RecorderRecord() = default;

    ~RecorderRecord() {
        // Tear the backend-borrowing stack down in dependency order BEFORE freeing
        // the backend: the Simulation references the world, and the world borrows
        // the backend (World::~World -> BackendPlanFree(backend_)).
        sim.reset();
        renderer.reset();
        publisher.reset();
        world.reset();
        if (backend != nullptr) {
            nphi::BackendFree(backend);
            backend = nullptr;
        }
        // phi_device is registry-owned; not freed.
    }
};

#else  // !NK_BUILD_VULKAN_VALIDATION

// Vulkan-less stub: a trivial record so the handle table is well-formed; no
// entry point ever does real work (all return NUKA_RESULT_NOT_SUPPORTED).
struct RecorderRecord {
    int unused = 0;
};

#endif  // NK_BUILD_VULKAN_VALIDATION

}  // namespace nuka::c_abi

namespace {

using nuka::c_abi::RecorderRecord;

nuka::c_abi::HandleTable<nuka_recorder_t, RecorderRecord>& RecorderTable() {
    static nuka::c_abi::HandleTable<nuka_recorder_t, RecorderRecord> table;
    return table;
}

#ifdef NK_BUILD_VULKAN_VALIDATION

// Aliases used by the extern-C bodies below (global scope -> they resolve these
// unnamed-namespace aliases; the matching aliases inside nuka::c_abi only serve
// the RecorderRecord definition).
namespace app = nuka::runtime::app;
namespace render = nuka::render;
namespace cook = nuka::scene::cook;
namespace nk = nuka::nk;
namespace nphi = nuka::phi;

// Write one RGBA8 framebuffer as a binary PPM P6 ("P6\n<W> <H>\n255\n" + RGB).
// Returns false on any I/O failure (the caller maps it to NUKA_RESULT_INTERNAL).
bool WritePpmP6(const std::string& path,
                const render::VulkanOffscreenReport& report) {
    if (report.pixels.empty() || report.width == 0u || report.height == 0u) {
        return false;
    }
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return false;
    bool ok = true;
    char header[64];
    const int hn = std::snprintf(header, sizeof(header), "P6\n%u %u\n255\n",
                                 report.width, report.height);
    if (hn <= 0 ||
        std::fwrite(header, 1u, static_cast<size_t>(hn), f) !=
            static_cast<size_t>(hn)) {
        ok = false;
    }
    if (ok) {
        // Drop alpha: PPM is RGB-only. Stream one row's worth at a time into a
        // reused buffer so a 1080p frame never allocates more than 3*W bytes.
        std::vector<unsigned char> rgb(static_cast<size_t>(report.width) * 3u);
        for (uint32_t y = 0u; ok && y < report.height; ++y) {
            for (uint32_t x = 0u; x < report.width; ++x) {
                const render::VulkanRgba8& p =
                    report.pixels[static_cast<size_t>(y) * report.width + x];
                rgb[static_cast<size_t>(x) * 3u + 0u] = p.r;
                rgb[static_cast<size_t>(x) * 3u + 1u] = p.g;
                rgb[static_cast<size_t>(x) * 3u + 2u] = p.b;
            }
            if (std::fwrite(rgb.data(), 1u, rgb.size(), f) != rgb.size()) {
                ok = false;
            }
        }
    }
    if (std::fclose(f) != 0) ok = false;
    return ok;
}

// Select the phi v2 Device for the requested CUDA ordinal. Walks the registry's
// devices counting across backends until `ordinal` is reached; falls back to
// InitBestDevice() (the first available device) when the ordinal is out of range
// or negative. This replaces an unconditional InitBestDevice() (which silently
// routed every recorder to device 0) so the validated device_record ordinal is
// actually honored.
nphi::Device* SelectDeviceByOrdinal(int ordinal) {
    if (ordinal >= 0) {
        int seen = 0;
        const size_t backend_count = nphi::BackendCount();
        for (size_t b = 0u; b < backend_count; ++b) {
            nphi::RegistryEntry* entry = nphi::GetBackend(b);
            if (entry == nullptr) continue;
            const size_t dev_count = nphi::RegistryEntryDeviceCount(entry);
            for (size_t d = 0u; d < dev_count; ++d) {
                if (seen == ordinal) {
                    return nphi::RegistryEntryGetDevice(entry, d);
                }
                ++seen;
            }
        }
    }
    return nphi::InitBestDevice();  // fallback: first available device.
}

#endif  // NK_BUILD_VULKAN_VALIDATION

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// create / destroy
// ---------------------------------------------------------------------------
nuka_result_t nuka_recorder_create(nuka_device_handle device,
                                   const nuka_recorder_desc_t* desc,
                                   nuka_recorder_handle* out) {
    if (out == nullptr) return NUKA_RESULT_INVALID_ARG;
    *out = nullptr;
    if (desc == nullptr) return NUKA_RESULT_INVALID_ARG;

#ifndef NK_BUILD_VULKAN_VALIDATION
    (void)device;
    (void)desc;
    return NUKA_RESULT_NOT_SUPPORTED;  // D5: Vulkan-less build, no recorder.
#else
    if (desc->scene_path == nullptr || desc->scene_path[0] == '\0') {
        return NUKA_RESULT_INVALID_ARG;
    }
    nuka::c_abi::DeviceRecord* device_record =
        nuka::c_abi::DeviceTable().Get(device);
    if (device_record == nullptr) return NUKA_RESULT_NULL_HANDLE;

    const std::string scene_path = desc->scene_path;
    if (!std::filesystem::exists(scene_path)) return NUKA_RESULT_FILE_NOT_FOUND;

    try {
        auto record = std::make_unique<nuka::c_abi::RecorderRecord>();
        record->device = device_record;

        // --- D2: cook the authored .nks to an nk::World + RenderWorld, exactly
        // as the M8 frame-loop smoke / parity gate do (Load -> CookToModel ->
        // nk::World; BuildRenderWorld from the SAME cook's Registry + SceneMap).
        const nuka::scene::SceneIR scene =
            nuka::scene::nks::Load(scene_path);
        cook::CookToModelResult cooked = cook::CookToModel(scene, 1);
        render::RenderWorld render_world =
            render::BuildRenderWorld(scene.Ecs(), cooked.scene_map);

        // The device's phi v2 backend (the same seam the frame-loop smoke uses).
        // Honor the validated device_record's CUDA ordinal instead of an
        // unconditional InitBestDevice()->device-0 routing.
        nphi::Device* dev = SelectDeviceByOrdinal(device_record->device_id);
        if (dev == nullptr) return NUKA_RESULT_NOT_SUPPORTED;
        nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
        if (backend == nullptr) return NUKA_RESULT_NOT_SUPPORTED;
        // Take ownership immediately so any early return below still frees it via
        // the RecorderRecord destructor (BackendFree ordered after world teardown).
        record->phi_device = dev;
        record->backend = backend;

        nk::Pipeline::SolverConfig cfg;
        cfg.dt = (desc->dt > 0.0f) ? desc->dt : (1.0f / 240.0f);
        cfg.gravity[0] = 0.0f;
        cfg.gravity[1] = 0.0f;
        cfg.gravity[2] = -9.81f;

        record->world = std::make_unique<nk::World>(
            std::move(cooked.model), 1u, dev, backend, cfg);
        if (!record->world->Ready()) return NUKA_RESULT_INTERNAL;

        // The offscreen raster renderer (throws if no Vulkan graphics device).
        record->renderer = std::make_unique<render::VulkanRasterRenderer>();

        // D6: default 1920x1080 unless overridden.
        record->raster_options.width = (desc->width != 0u) ? desc->width : 1920u;
        record->raster_options.height =
            (desc->height != 0u) ? desc->height : 1080u;
        if (desc->use_camera_override != 0u) {
            record->raster_options.use_camera_override = true;
            record->raster_options.camera_eye = {desc->camera_eye[0],
                                                 desc->camera_eye[1],
                                                 desc->camera_eye[2]};
            record->raster_options.camera_target = {desc->camera_target[0],
                                                    desc->camera_target[1],
                                                    desc->camera_target[2]};
            const bool up_zero = desc->camera_up[0] == 0.0f &&
                                 desc->camera_up[1] == 0.0f &&
                                 desc->camera_up[2] == 0.0f;
            record->raster_options.camera_up =
                up_zero ? nuka::math::Vec3{0.0f, 0.0f, 1.0f}
                        : nuka::math::Vec3{desc->camera_up[0], desc->camera_up[1],
                                           desc->camera_up[2]};
            record->raster_options.camera_fov_degrees =
                (desc->camera_fov_degrees > 0.0f) ? desc->camera_fov_degrees
                                                  : 45.0f;
        }

        record->publisher = std::make_unique<app::HostDownloadPublisher>();
        record->sim = std::make_unique<app::Simulation>(
            *record->world, *record->publisher, std::move(render_world));
        record->sim->SetEnvIndex(desc->env_index);  // D4: env 0 by default.
        record->sim->EnableRendering(record->renderer.get(),
                                     record->raster_options);

        *out = RecorderTable().Insert(std::move(record));
        return (*out != nullptr) ? NUKA_RESULT_OK : NUKA_RESULT_INTERNAL;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
#endif  // NK_BUILD_VULKAN_VALIDATION
}

void nuka_recorder_destroy(nuka_recorder_handle recorder) {
    (void)RecorderTable().Remove(recorder);
}

// ---------------------------------------------------------------------------
// capture: N frames -> PPM P6 sequence
// ---------------------------------------------------------------------------
nuka_result_t nuka_recorder_capture(nuka_recorder_handle recorder,
                                    uint32_t n_frames, const char* out_dir) {
#ifndef NK_BUILD_VULKAN_VALIDATION
    (void)recorder;
    (void)n_frames;
    (void)out_dir;
    return NUKA_RESULT_NOT_SUPPORTED;  // D5.
#else
    RecorderRecord* r = RecorderTable().Get(recorder);
    if (r == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (n_frames == 0u || out_dir == nullptr || out_dir[0] == '\0') {
        return NUKA_RESULT_INVALID_ARG;
    }
    try {
        std::filesystem::create_directories(out_dir);
        for (uint32_t i = 0u; i < n_frames; ++i) {
            if (!r->sim->Frame()) return NUKA_RESULT_INTERNAL;  // unhealthy step
            if (!r->sim->LastFrameRendered()) return NUKA_RESULT_INTERNAL;
            char name[32];
            std::snprintf(name, sizeof(name), "frame_%06u.ppm", r->frame_index);
            const std::string path =
                (std::filesystem::path(out_dir) / name).string();
            if (!WritePpmP6(path, r->sim->LatestReport())) {
                return NUKA_RESULT_INTERNAL;
            }
            ++r->frame_index;
        }
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
#endif  // NK_BUILD_VULKAN_VALIDATION
}

// ---------------------------------------------------------------------------
// to_video: ffmpeg mux (R6: graceful on missing/failing ffmpeg)
// ---------------------------------------------------------------------------
nuka_result_t nuka_recorder_to_video(nuka_recorder_handle recorder,
                                     const char* out_dir, const char* out_mp4,
                                     uint32_t fps) {
#ifndef NK_BUILD_VULKAN_VALIDATION
    (void)recorder;
    (void)out_dir;
    (void)out_mp4;
    (void)fps;
    return NUKA_RESULT_NOT_SUPPORTED;  // D5.
#else
    RecorderRecord* r = RecorderTable().Get(recorder);
    if (r == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (out_dir == nullptr || out_dir[0] == '\0' || out_mp4 == nullptr ||
        out_mp4[0] == '\0') {
        return NUKA_RESULT_INVALID_ARG;
    }
    try {
        // R6: ffmpeg-on-PATH probe. `command -v` exits non-zero when absent; we
        // return a clean NOT_SUPPORTED rather than letting the mux command fail.
        if (std::system("command -v ffmpeg >/dev/null 2>&1") != 0) {
            return NUKA_RESULT_NOT_SUPPORTED;
        }
        const uint32_t use_fps = (fps != 0u) ? fps : 30u;  // default 30.
        const std::string fps_str = std::to_string(use_fps);
        const std::string pattern =
            (std::filesystem::path(out_dir) / "frame_%06d.ppm").string();
        // -y overwrite; -framerate before -i sets the INPUT rate; -r after -i pins
        // the OUTPUT rate to fps so the muxed video plays at exactly fps. libx264 +
        // yuv420p is the default codec/pixel format. Quote paths defensively.
        std::string cmd = "ffmpeg -y -framerate " + fps_str +
                          " -i \"" + pattern + "\" -r " + fps_str +
                          " -pix_fmt yuv420p -c:v libx264 \"" +
                          std::string(out_mp4) + "\" >/dev/null 2>&1";
        const int rc = std::system(cmd.c_str());
        if (rc != 0) return NUKA_RESULT_INTERNAL;  // ffmpeg failed (no crash).
        return NUKA_RESULT_OK;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
#endif  // NK_BUILD_VULKAN_VALIDATION
}

uint32_t nuka_recorder_frame_count(nuka_recorder_handle recorder) {
#ifndef NK_BUILD_VULKAN_VALIDATION
    (void)recorder;
    return 0u;
#else
    RecorderRecord* r = RecorderTable().Get(recorder);
    return (r != nullptr) ? r->frame_index : 0u;
#endif  // NK_BUILD_VULKAN_VALIDATION
}

}  // extern "C"
