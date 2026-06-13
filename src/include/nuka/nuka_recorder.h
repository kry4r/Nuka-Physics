#ifndef NUKA_NUKA_RECORDER_H
#define NUKA_NUKA_RECORDER_H
// ---------------------------------------------------------------------------
// Nuka C ABI -- the offscreen RECORDER surface (M8 T5). A MINIMAL extern "C"
// shim that drives the M8 host frame loop (runtime::app::Simulation over an
// nk::World) through the offscreen Vulkan forward RASTER renderer and captures
// each frame to disk, then muxes the captured frames into an mp4 via ffmpeg.
//
// THE nk-BACKED WORLD (Decision D2). The recorder reads/steps the nk::World, NOT
// the legacy WorldRecord. nuka_recorder_create COOKS the authored .nks scene
// inside libnuka exactly as the M8 frame-loop smoke / parity gate do:
//   scene::nks::Load(scene_path) -> cook::CookToModel(scene, 1) -> nk::World
// and BUILDS the RenderWorld from the SAME cook's Registry + SceneMap
//   render::BuildRenderWorld(scene.Ecs(), cooked.scene_map)
// so the rendered instances are bound to the very rows the World's Data pose
// fields carry. The capture is therefore the live nk physics state -- M9-proof
// (no legacy world is touched, the c_abi->nk cutover does not churn this).
//
// ENV SELECTION (Decision D4). The render env is env 0 by default; the desc
// carries an optional `env_index` that threads Simulation::SetEnvIndex -> the
// HostDownloadPublisher's DownloadField byte offset.
//
// RENDER CONFIG (Decision D6). Defaults are 1920x1080 @ 30 fps, h264/yuv420p,
// each overridable on the desc / the to_video fps arg. The capture writes a
// binary PPM P6 ("P6\n<W> <H>\n255\n" + RGB bytes) per frame into <out_dir> as
// frame_%06d.ppm; to_video shells ffmpeg over that sequence.
//
// VULKAN GATING (Decision D5). The recorder pulls the Vulkan raster renderer
// into libnuka, so its IMPLEMENTATION is compiled behind NK_BUILD_VULKAN_VALIDATION.
// On a Vulkan-less build (the flag OFF) libnuka still exports these SAME symbols
// but every entry point returns NUKA_RESULT_NOT_SUPPORTED -- libnuka stays
// buildable with NO Vulkan dependency and the python binding still imports.
//
// FFMPEG (Risk R6). nuka_recorder_to_video shells `ffmpeg`. If ffmpeg is absent
// or the mux fails it returns a clean nuka_result_t (NUKA_RESULT_NOT_SUPPORTED
// for "no ffmpeg on PATH", NUKA_RESULT_INTERNAL for a non-zero ffmpeg exit) --
// it NEVER crashes.
//
// WHY A C ABI (not a C++ class bind): same reason as nuka.h -- the engine
// is g++-14, the nanobind binding g++-10; only plain C crosses.
//
// ERROR / HANDLE CONVENTIONS mirror nuka.h (nuka_result_t, zeroed out-params,
// NULL handle -> NUKA_RESULT_NULL_HANDLE, bad arg -> INVALID_ARG, missing asset
// -> FILE_NOT_FOUND, no-Vulkan / no-ffmpeg -> NOT_SUPPORTED).
// ---------------------------------------------------------------------------

#include "nuka/nuka.h"  // nuka_result_t, nuka_device_handle, stdint/stddef

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nuka_recorder_t* nuka_recorder_handle;

// Creation descriptor. `scene_path` is REQUIRED (the authored .nks the recorder
// cooks to an nk::World + RenderWorld); NULL/"" -> NUKA_RESULT_INVALID_ARG. The
// camera fields are passed through to the raster RasterOptions; when
// use_camera_override == 0 the renderer uses the scene's camera (or auto-frames
// the AABB) so a frame is always visible. 0-valued render config selects the
// D6 defaults (width 1920, height 1080).
typedef struct nuka_recorder_desc_t {
    const char* scene_path;        /* REQUIRED authored .nks scene path          */
    uint32_t    width;             /* 0 -> 1920                                  */
    uint32_t    height;            /* 0 -> 1080                                  */
    uint32_t    env_index;         /* which env to render (D4; default 0)        */
    float       dt;                /* <= 0 -> 1/240f (the world's fixed dt)      */

    /* Optional explicit camera (eye/target/up + vertical FOV degrees). When
       use_camera_override != 0 these win over the scene camera.                 */
    uint32_t    use_camera_override; /* 0 -> use scene camera / auto-frame       */
    float       camera_eye[3];
    float       camera_target[3];
    float       camera_up[3];        /* 0,0,0 -> default up (0,0,1)              */
    float       camera_fov_degrees;  /* <= 0 -> 45.0f                            */
} nuka_recorder_desc_t;

// Create a recorder over a freshly cooked nk::World for `scene_path`, bound to
// `device`. Returns NUKA_RESULT_NOT_SUPPORTED on a Vulkan-less build (D5),
// FILE_NOT_FOUND if the scene / its assets are missing, INVALID_ARG on a NULL
// scene_path, and (when no Vulkan graphics device is present) NOT_SUPPORTED.
nuka_result_t nuka_recorder_create(nuka_device_handle device,
                                   const nuka_recorder_desc_t* desc,
                                   nuka_recorder_handle* out);

void nuka_recorder_destroy(nuka_recorder_handle recorder);

// Drive `n_frames` Simulation::Frame()s (step + publish + render) and write each
// frame's raw RGBA framebuffer to <out_dir>/frame_%06d.ppm as binary PPM P6.
// The frame index continues across successive capture calls on the same handle
// (so a multi-segment rollout muxes contiguously). `out_dir` is created if
// absent. Returns NOT_SUPPORTED on a Vulkan-less build, NULL_HANDLE on a bad
// handle, INVALID_ARG on n_frames == 0 / NULL out_dir, INTERNAL on a write
// failure or an unhealthy step.
nuka_result_t nuka_recorder_capture(nuka_recorder_handle recorder,
                                    uint32_t n_frames, const char* out_dir);

// Mux <out_dir>/frame_%06d.ppm -> <out_mp4> via ffmpeg at `fps` (<= 0 -> 30).
// Shells: ffmpeg -y -framerate <fps> -i <out_dir>/frame_%06d.ppm
//                -pix_fmt yuv420p -c:v libx264 <out_mp4>
// R6: returns NUKA_RESULT_NOT_SUPPORTED if ffmpeg is not on PATH (no crash) and
// NUKA_RESULT_INTERNAL if ffmpeg exits non-zero. NOT_SUPPORTED on a Vulkan-less
// build, INVALID_ARG on NULL out_dir/out_mp4.
nuka_result_t nuka_recorder_to_video(nuka_recorder_handle recorder,
                                     const char* out_dir, const char* out_mp4,
                                     uint32_t fps);

// The number of frames captured so far on this handle (0 on a bad handle).
uint32_t nuka_recorder_frame_count(nuka_recorder_handle recorder);

#ifdef __cplusplus
}
#endif

#endif /* NUKA_NUKA_RECORDER_H */
