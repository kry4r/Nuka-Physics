# ---------------------------------------------------------------------------
# nuka_core – foundational utilities (version, logging, …)
# ---------------------------------------------------------------------------
add_library(nuka_core STATIC
    core/version.cpp
    core/perf/perf_recorder.cpp
)
set_target_properties(nuka_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(nuka_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

nuka_set_warnings(nuka_core)

# ---------------------------------------------------------------------------
# nuka_math – header-only math library (Vec3, Quat, Transform, Spatial)
# ---------------------------------------------------------------------------
add_library(nuka_math INTERFACE)

target_include_directories(nuka_math
    INTERFACE
        ${CMAKE_SOURCE_DIR}/src
)

# ---------------------------------------------------------------------------
# nuka_cooker – cook-time asset processing (V-HACD convex decomposition)
# ---------------------------------------------------------------------------
# COOK-TIME / HOST-CPU code (master plan §5.6 sanctioned exception). This is a
# LEAF library: DecomposeMesh() takes only raw pointers + its own POD structs,
# so it carries no scene/runtime types and can be linked PUBLIC by nuka_scene
# (whose CookScene invokes it) without creating a dependency cycle. The
# importers (nuka_import, which depends on nuka_scene) only parse the
# nuka:decompose attribute; they do NOT link nuka_cooker directly.
#
# nuka_set_warnings is intentionally NOT applied: the vendored single-header
# V-HACD implementation is compiled into convex_decomposition.cpp and does not
# build clean under -Werror -Wconversion -Wshadow. It is third-party cook-time
# code (not a D1 / hot-path concern); warnings on it are suppressed via the
# SYSTEM include path and by omitting the strict flag set on this TU.
add_library(nuka_cooker STATIC
    import/cooker/convex_decomposition.cpp
    import/cooker/mesh_surface_cooker.cpp
    import/cooker/mesh_simplifier.cpp
    import/cooker/mesh_surface_cache.cpp
    import/cooker/mesh_query_backend.cpp
    import/cooker/convex_cover.cpp
    import/cooker/sparse_sdf_cooker.cpp
    import/cooker/sdf_bake_backend.cpp
)
set_target_properties(nuka_cooker PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CUDA_COMPILER)
    target_sources(nuka_cooker PRIVATE phi/backend_cuda/cooking/mesh_query.cu)
    target_compile_definitions(nuka_cooker PRIVATE NUKA_COOKER_HAS_CUDA=1)
    target_link_libraries(nuka_cooker PRIVATE CUDA::cudart)
    set_target_properties(nuka_cooker PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

target_include_directories(nuka_cooker
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_cooker
    PUBLIC
        nuka_vhacd
        nuka_sha256
)

# ---------------------------------------------------------------------------
# nuka_scene – Scene IR and runtime cooking
# ---------------------------------------------------------------------------
add_library(nuka_scene STATIC
    scene/scene_ir.cpp
    scene/scene_compose.cpp
    scene/scene_fields.cpp
    scene/cooker.cpp
    scene/graph/scene_graph.cpp
    scene/ecs/registry.cpp
    scene/scene_map.cpp
    # M2c: own asset formats. nka (binary container) + asset_cache (content-
    # addressed cook cache) + json (self-written minimal). nks.cpp lives in
    # nuka_import (it calls the importers/Compose) to avoid a dep cycle.
    scene/asset/nka.cpp
    scene/asset/asset_cache.cpp
    scene/format/json.cpp
)
set_target_properties(nuka_scene PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(nuka_scene
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_scene
    PUBLIC
        nuka_math
        nuka_cooker
)

nuka_set_warnings(nuka_scene)

# ---------------------------------------------------------------------------
# nuka_runtime – Runtime containers, world building, rigid body integration
# ---------------------------------------------------------------------------
add_library(nuka_runtime STATIC
    core/diagnostics/invariants.cpp
    core/diagnostics/trace_sink.cpp
    runtime/world_builder.cpp
    runtime/physics_world.cpp
    runtime/rigid/integrator.cpp
    runtime/batch_scheduler.cpp
)
set_target_properties(nuka_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(nuka_runtime
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

set(NUKA_PUBLIC_HEADERS
    ${CMAKE_SOURCE_DIR}/src/include/nuka/nuka.h
    ${CMAKE_SOURCE_DIR}/src/include/nuka/nuka.hpp
    ${CMAKE_SOURCE_DIR}/src/include/nuka/nuka_noise.h
    ${CMAKE_SOURCE_DIR}/src/include/nuka/nuka_state_sensor.h
    ${CMAKE_SOURCE_DIR}/src/include/nuka/nuka_recorder.h
)

find_package(Threads REQUIRED)
add_library(nuka_particle_geometry STATIC
    runtime/fluid/surface_mesher.cpp
    runtime/particle_skin.cpp
)
set_target_properties(nuka_particle_geometry PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nuka_particle_geometry PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(nuka_particle_geometry PRIVATE Threads::Threads)
nuka_set_warnings(nuka_particle_geometry)

add_library(nuka_render STATIC
    render/render_world.cpp
    # Host image decode (stb_image) -> rt::Texture / rt::EnvironmentMap for the
    # offline beauty texture + HDR-environment pipeline. ZERO CUDA.
    render/texture_image.cpp
    # The ONE general particle-set -> deformed render surface mesh (cloth now,
    # tet/fluid later). Pure host, ZERO CUDA: it lives in nuka_render so a demo
    # gets a deforming surface from live particles with no CUDA dependency.
    runtime/soft/particle_surface.cpp
    # Host, CUDA-free adapter mapping RenderWorld -> the
    # self-written path-tracer scene-desc POD (rt::TwoLevelScene). Zero CUDA
    # token (src/render red-line); the CUDA backend is reached only via the
    # render::RtBackendI interface (render/rt_backend.hpp).
    render/rt_adapter.cpp
    # Host, CUDA-free glue mapping the path-tracer's linear-float rt::Framebuffer
    # to the RGBA8 VulkanOffscreenReport (ACES + sRGB, matching mesh_pbr.frag).
    render/rt_framebuffer_to_report.cpp
    # The shared studio render scene (robot links + deforming particle surface +
    # studio floor + cinematic options) the cloth demo AND the live-render bridge
    # draw. Reaches the offline RT tracer only via render::RtBackendI (zero CUDA).
    render/studio_beauty.cpp
    render/scene_asset.cpp
    render/vulkan_renderer.cpp
    render/raster/vulkan_raster_renderer.cpp
    # Exportable pose slots + timeline semaphore the editor's CUDA scatter writes.
    render/raster/interop_transform_ssbo.cpp
)

find_program(GLSLC_EXECUTABLE glslc HINTS "$ENV{VULKAN_SDK}/Bin")
if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found; Vulkan offscreen renderer requires the Vulkan SDK shader compiler")
endif()

set(NUKA_DEBUG_DRAW_COMP "${CMAKE_CURRENT_SOURCE_DIR}/render/shaders/debug_draw.comp")
set(NUKA_DEBUG_DRAW_SPV "${CMAKE_CURRENT_BINARY_DIR}/render/shaders/debug_draw.comp.spv")
add_custom_command(
    OUTPUT "${NUKA_DEBUG_DRAW_SPV}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/render/shaders"
    COMMAND "${GLSLC_EXECUTABLE}" "${NUKA_DEBUG_DRAW_COMP}" -o "${NUKA_DEBUG_DRAW_SPV}"
    DEPENDS "${NUKA_DEBUG_DRAW_COMP}"
    VERBATIM
)

set(NUKA_RASTER_MESH_VERT "${CMAKE_CURRENT_SOURCE_DIR}/render/raster/shaders/mesh.vert")
set(NUKA_RASTER_MESH_VERT_SPV "${CMAKE_CURRENT_BINARY_DIR}/render/raster/shaders/mesh.vert.spv")
add_custom_command(
    OUTPUT "${NUKA_RASTER_MESH_VERT_SPV}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/render/raster/shaders"
    COMMAND "${GLSLC_EXECUTABLE}" "${NUKA_RASTER_MESH_VERT}" -o "${NUKA_RASTER_MESH_VERT_SPV}"
    DEPENDS "${NUKA_RASTER_MESH_VERT}"
    VERBATIM
)

set(NUKA_RASTER_MESH_FRAG "${CMAKE_CURRENT_SOURCE_DIR}/render/raster/shaders/mesh_pbr.frag")
set(NUKA_RASTER_MESH_FRAG_SPV "${CMAKE_CURRENT_BINARY_DIR}/render/raster/shaders/mesh_pbr.frag.spv")
add_custom_command(
    OUTPUT "${NUKA_RASTER_MESH_FRAG_SPV}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/render/raster/shaders"
    COMMAND "${GLSLC_EXECUTABLE}" "${NUKA_RASTER_MESH_FRAG}" -o "${NUKA_RASTER_MESH_FRAG_SPV}"
    DEPENDS "${NUKA_RASTER_MESH_FRAG}"
    VERBATIM
)

# M11 INT-5 (OD-2): the INSTANCED vertex-shader variant (reads the per-instance
# model matrix from the device-local interop SSBO by gl_InstanceIndex). The DEFAULT
# mesh.vert per-draw pipeline (the D1 oracle path) is unchanged; this is the
# opt-in, interop-only 2nd pipeline variant.
set(NUKA_RASTER_MESH_INSTANCED_VERT "${CMAKE_CURRENT_SOURCE_DIR}/render/raster/shaders/mesh_instanced.vert")
set(NUKA_RASTER_MESH_INSTANCED_VERT_SPV "${CMAKE_CURRENT_BINARY_DIR}/render/raster/shaders/mesh_instanced.vert.spv")
add_custom_command(
    OUTPUT "${NUKA_RASTER_MESH_INSTANCED_VERT_SPV}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/render/raster/shaders"
    COMMAND "${GLSLC_EXECUTABLE}" "${NUKA_RASTER_MESH_INSTANCED_VERT}" -o "${NUKA_RASTER_MESH_INSTANCED_VERT_SPV}"
    DEPENDS "${NUKA_RASTER_MESH_INSTANCED_VERT}"
    VERBATIM
)

# Go2-terrain demo: the DIRECTIONAL SHADOW-MAP depth-only vertex shader (the hero
# Isaac-Lab feature). Same glslc custom_command pattern. Depth-only, no fragment.
set(NUKA_RASTER_SHADOW_VERT "${CMAKE_CURRENT_SOURCE_DIR}/render/raster/shaders/shadow_depth.vert")
set(NUKA_RASTER_SHADOW_VERT_SPV "${CMAKE_CURRENT_BINARY_DIR}/render/raster/shaders/shadow_depth.vert.spv")
add_custom_command(
    OUTPUT "${NUKA_RASTER_SHADOW_VERT_SPV}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/render/raster/shaders"
    COMMAND "${GLSLC_EXECUTABLE}" "${NUKA_RASTER_SHADOW_VERT}" -o "${NUKA_RASTER_SHADOW_VERT_SPV}"
    DEPENDS "${NUKA_RASTER_SHADOW_VERT}"
    VERBATIM
)

# Go2-terrain demo: the SKY GRADIENT fullscreen vertex + fragment shaders.
set(NUKA_RASTER_SKY_VERT "${CMAKE_CURRENT_SOURCE_DIR}/render/raster/shaders/sky_gradient.vert")
set(NUKA_RASTER_SKY_VERT_SPV "${CMAKE_CURRENT_BINARY_DIR}/render/raster/shaders/sky_gradient.vert.spv")
add_custom_command(
    OUTPUT "${NUKA_RASTER_SKY_VERT_SPV}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/render/raster/shaders"
    COMMAND "${GLSLC_EXECUTABLE}" "${NUKA_RASTER_SKY_VERT}" -o "${NUKA_RASTER_SKY_VERT_SPV}"
    DEPENDS "${NUKA_RASTER_SKY_VERT}"
    VERBATIM
)
set(NUKA_RASTER_SKY_FRAG "${CMAKE_CURRENT_SOURCE_DIR}/render/raster/shaders/sky_gradient.frag")
set(NUKA_RASTER_SKY_FRAG_SPV "${CMAKE_CURRENT_BINARY_DIR}/render/raster/shaders/sky_gradient.frag.spv")
add_custom_command(
    OUTPUT "${NUKA_RASTER_SKY_FRAG_SPV}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/render/raster/shaders"
    COMMAND "${GLSLC_EXECUTABLE}" "${NUKA_RASTER_SKY_FRAG}" -o "${NUKA_RASTER_SKY_FRAG_SPV}"
    DEPENDS "${NUKA_RASTER_SKY_FRAG}"
    VERBATIM
)

add_custom_target(nuka_render_shaders DEPENDS
    "${NUKA_DEBUG_DRAW_SPV}"
    "${NUKA_RASTER_MESH_VERT_SPV}"
    "${NUKA_RASTER_MESH_FRAG_SPV}"
    "${NUKA_RASTER_MESH_INSTANCED_VERT_SPV}"
    "${NUKA_RASTER_SHADOW_VERT_SPV}"
    "${NUKA_RASTER_SKY_VERT_SPV}"
    "${NUKA_RASTER_SKY_FRAG_SPV}"
)
add_dependencies(nuka_render nuka_render_shaders)
target_compile_definitions(nuka_render
    PRIVATE
        NUKA_VULKAN_DEBUG_DRAW_SPV="${NUKA_DEBUG_DRAW_SPV}"
        NUKA_RASTER_MESH_VERT_SPV="${NUKA_RASTER_MESH_VERT_SPV}"
        NUKA_RASTER_MESH_FRAG_SPV="${NUKA_RASTER_MESH_FRAG_SPV}"
        NUKA_RASTER_MESH_INSTANCED_VERT_SPV="${NUKA_RASTER_MESH_INSTANCED_VERT_SPV}"
        NUKA_RASTER_SHADOW_VERT_SPV="${NUKA_RASTER_SHADOW_VERT_SPV}"
        NUKA_RASTER_SKY_VERT_SPV="${NUKA_RASTER_SKY_VERT_SPV}"
        NUKA_RASTER_SKY_FRAG_SPV="${NUKA_RASTER_SKY_FRAG_SPV}"
)

target_include_directories(nuka_render
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_render
    PUBLIC
        nuka_scene
        nuka_particle_geometry
        Vulkan::Vulkan
    PRIVATE
        nuka_stb
        Threads::Threads
)

nuka_set_warnings(nuka_render)

add_subdirectory(rhi)

if(NK_BUILD_EDITOR OR NK_BUILD_VULKAN_VALIDATION)
    include("${CMAKE_SOURCE_DIR}/cmake/ViewerFonts.cmake")
    nuka_resolve_viewer_fonts()

    add_library(nuka_imgui STATIC
        render/imgui/nuka_imgui.cpp
    )
    set_target_properties(nuka_imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_include_directories(nuka_imgui
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )
    # The vendored ImGui headers come in as SYSTEM includes via the `imgui` target's
    # PUBLIC SYSTEM include dirs -- so nuka_imgui.cpp's #include "imgui.h" resolves
    # without imgui warnings entering nuka_imgui's -Werror.
    target_link_libraries(nuka_imgui
        PUBLIC
            imgui
            Vulkan::Vulkan
    )

    target_compile_definitions(nuka_imgui
        PRIVATE
            NUKA_IMGUI_FONT_BODY="${NUKA_VIEWER_BODY_FONT}"
            NUKA_IMGUI_FONT_STRONG="${NUKA_VIEWER_STRONG_FONT}"
            NUKA_IMGUI_FONT_DISPLAY="${NUKA_VIEWER_DISPLAY_FONT}"
            NUKA_IMGUI_FONT_MONO="${NUKA_VIEWER_MONO_FONT}"
            NUKA_IMGUI_FONT_ICON="${NUKA_VIEWER_ICON_FONT}"
            NUKA_IMGUI_FONT_ICON_FILL="${NUKA_VIEWER_ICON_FILL_FONT}"
            NUKA_IMGUI_FONT_CJK="${NUKA_VIEWER_CJK_FONT_FILE}"
            NUKA_IMGUI_LOGO="${CMAKE_SOURCE_DIR}/docs/media/nuka-logo.png"
    )
    target_link_libraries(nuka_imgui PUBLIC nuka_render)
    set_property(GLOBAL PROPERTY NUKA_VIEWER_FONT_RESOURCES
        "${NUKA_VIEWER_BODY_FONT}" "${NUKA_VIEWER_STRONG_FONT}" "${NUKA_VIEWER_DISPLAY_FONT}" "${NUKA_VIEWER_MONO_FONT}"
        "${NUKA_VIEWER_ICON_FONT}" "${NUKA_VIEWER_ICON_FILL_FONT}")
    set_property(GLOBAL PROPERTY NUKA_VIEWER_CJK_RESOURCE "${NUKA_VIEWER_CJK_FONT_FILE}")
    set_property(GLOBAL PROPERTY NUKA_VIEWER_FONT_DIR "${NUKA_VIEWER_FONT_DIR}")
    set_property(GLOBAL PROPERTY NUKA_VIEWER_ICON_LICENSE "${NUKA_VIEWER_ICON_LICENSE}")
    function(nuka_copy_viewer_resources target)
        get_property(faces GLOBAL PROPERTY NUKA_VIEWER_FONT_RESOURCES)
        get_property(cjk GLOBAL PROPERTY NUKA_VIEWER_CJK_RESOURCE)
        get_property(font_dir GLOBAL PROPERTY NUKA_VIEWER_FONT_DIR)
        get_property(icon_license GLOBAL PROPERTY NUKA_VIEWER_ICON_LICENSE)
        set(fonts_out "$<TARGET_FILE_DIR:${target}>/assets/viewer/fonts")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${fonts_out}"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/assets/viewer/fonts" "${fonts_out}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${faces} "${cjk}" "${fonts_out}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${font_dir}/UbuntuSans/LICENCE.txt" "${fonts_out}/UbuntuSans-LICENCE.txt"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${font_dir}/MartianMono/LICENSE" "${fonts_out}/MartianMono-OFL.txt"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${font_dir}/GeistMono/LICENSE" "${fonts_out}/GeistMono-OFL.txt"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${icon_license}" "${fonts_out}/Phosphor-LICENSE.txt"
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${target}>/docs/media"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/docs/media/nuka-logo.png" "$<TARGET_FILE_DIR:${target}>/docs/media/nuka-logo.png"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/examples/assets/nuka_lab" "$<TARGET_FILE_DIR:${target}>/examples/assets/nuka_lab"
            VERBATIM)
    endfunction()
    nuka_set_warnings(nuka_imgui)
endif()

if(NK_BUILD_EDITOR OR NK_BUILD_VULKAN_VALIDATION)
    add_library(nuka_present STATIC
        render/raster/vulkan_present_renderer.cpp
        render/window/window_surface_sdl3.cpp
        render/window/window_surface_headless.cpp
    )
    add_dependencies(nuka_present nuka_render_shaders)
    set_target_properties(nuka_present PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_include_directories(nuka_present
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )

    target_compile_definitions(nuka_present
        PRIVATE
            NUKA_RASTER_MESH_VERT_SPV="${NUKA_RASTER_MESH_VERT_SPV}"
            NUKA_RASTER_MESH_FRAG_SPV="${NUKA_RASTER_MESH_FRAG_SPV}"
            # M11 INT-F1: the present path's opt-in instanced pipeline reads the
            # per-instance model matrix from the interop SSBO by gl_InstanceIndex.
            NUKA_RASTER_MESH_INSTANCED_VERT_SPV="${NUKA_RASTER_MESH_INSTANCED_VERT_SPV}"
    )

    target_link_libraries(nuka_present
        PUBLIC
            nuka_render
            Vulkan::Vulkan
        PRIVATE
            SDL3::SDL3-static
    )

    nuka_set_warnings(nuka_present)
endif()

# The viewer panels and camera use render snapshots without a physics backend.
if(NK_BUILD_EDITOR OR NK_BUILD_VULKAN_VALIDATION)
    add_library(nuka_viewer STATIC
        runtime/app/viewer/camera_controller.cpp
        runtime/app/viewer/debug_draw.cpp
        runtime/app/viewer/editor_actions.cpp
        runtime/app/viewer/file_dialog.cpp
        runtime/app/viewer/imgui_layer.cpp
        runtime/app/viewer/nui/nui_capi.cpp
        runtime/app/viewer/nui/nui_core.cpp
        runtime/app/viewer/nui/nui_data.cpp
        runtime/app/viewer/nui/nui_widgets.cpp
    )
    set_target_properties(nuka_viewer PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(nuka_viewer PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/include)
    target_compile_definitions(nuka_viewer PUBLIC NK_BUILD_VULKAN_VALIDATION)
    target_link_libraries(nuka_viewer PUBLIC nuka_scene nuka_present nuka_imgui imgui)
    nuka_set_warnings(nuka_viewer)
endif()

# ---------------------------------------------------------------------------
# nuka_import – MJCF and URDF importers
# ---------------------------------------------------------------------------
add_library(nuka_import STATIC
    import/mjcf_importer.cpp
    import/urdf_importer.cpp
    import/usd_stage_adapter.cpp
    import/usd_importer.cpp
    import/usd_crate_reader.cpp
    import/mesh_file_loader.cpp
    # M2c: .nks scene Save/Load. Lives here (not nuka_scene) because Load resolves
    # "imports" via LoadMjcf/LoadUsd/LoadUrdf + Compose; nuka_import already links
    # nuka_scene so it sees the json/nka/asset_ref surface.
    scene/format/nks.cpp
)
set_target_properties(nuka_import PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(nuka_import
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_import
    PUBLIC
        nuka_scene
        tinyxml2::tinyxml2
)

nuka_set_warnings(nuka_import)

# ---------------------------------------------------------------------------
# nuka_collision – broadphase and geometry queries
# ---------------------------------------------------------------------------
add_library(nuka_collision STATIC
    collision/query.cpp
    collision/dynamic_broadphase.cpp
)
set_target_properties(nuka_collision PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(nuka_collision
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_collision
    PUBLIC
        nuka_math
)

nuka_set_warnings(nuka_collision)

# ---------------------------------------------------------------------------
# nuka_constraint – contact manifolds and constraint blocks
# ---------------------------------------------------------------------------
add_library(nuka_constraint STATIC
    constraint/contact_builder.cpp
    constraint/row_buffers.cpp
    constraint/row_builder.cpp
)
set_target_properties(nuka_constraint PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(nuka_constraint
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_constraint
    PUBLIC
        nuka_math
)

nuka_set_warnings(nuka_constraint)

add_library(nuka_solver INTERFACE)

target_include_directories(nuka_solver
    INTERFACE
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_solver
    INTERFACE
        nuka_constraint
        nuka_math
)

# ---------------------------------------------------------------------------
# nuka_articulation – Articulation graphs and joint drives
# ---------------------------------------------------------------------------
add_library(nuka_articulation STATIC
    runtime/articulation/articulation_state.cpp
    runtime/articulation/articulation_cooker.cpp
    runtime/articulation/joint_constraints.cpp
)
set_target_properties(nuka_articulation PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(nuka_articulation
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_articulation
    PUBLIC
        nuka_constraint
        nuka_scene
        # M11 buffer sweep: articulation_state.cpp's UploadArticulationState now
        # allocates phi v2 Buffer* from the device-level DEFAULT type, so it needs
        # phi v2's InitBestDevice / DeviceBufferType / Buffer wrappers (registry +
        # cuda backend). PUBLIC so every consumer (test exes, runtime libs) resolves
        # these transitively without per-target link edits.
        nuka_phi2
)

nuka_set_warnings(nuka_articulation)

target_link_libraries(nuka_runtime
    PUBLIC
        nuka_scene
        nuka_collision
        nuka_constraint
        nuka_articulation
        nuka_solver
)

nuka_set_warnings(nuka_runtime)

# ---------------------------------------------------------------------------
# nuka_sensor – Sensor graph and queries
# ---------------------------------------------------------------------------
add_library(nuka_sensor STATIC
    sensor/state_sensor.cpp
    sensor/ray_sensor.cpp
)

target_include_directories(nuka_sensor
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_sensor
    PUBLIC
        nuka_runtime
        nuka_collision
)

nuka_set_warnings(nuka_sensor)

# The registry and physics core compile independently of device implementations.
add_library(nuka_phi2 STATIC phi/registry.cpp)
add_dependencies(nuka_phi2 nuka_codegen)
set_target_properties(nuka_phi2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nuka_phi2 PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_options(nuka_phi2 PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4 /permissive- /WX /wd4819>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror>
)

add_library(nuka_nk STATIC
    nk/model/model.cpp
    nk/data/arena.cpp
    nk/data/data.cpp
    nk/pipeline/pipeline.cpp
    nk/pipeline/coupling_provider.cpp
    nk/pipeline/world.cpp
    nk/solve/vertex_block_hierarchy.cpp
    nk/solve/vertex_block_schedule.cpp
    sensor/observation.cpp
    sensor/state_bank.cpp
)
set_target_properties(nuka_nk PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_dependencies(nuka_nk nuka_codegen)
target_include_directories(nuka_nk PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(nuka_nk PUBLIC nuka_phi2 nuka_math)
nuka_set_warnings(nuka_nk)

if(CMAKE_CUDA_COMPILER)
    set_source_files_properties(
        ${NUKA_CODEGEN_OUTPUTS}
        PROPERTIES GENERATED TRUE
    )

    add_library(nuka_codegen_generated STATIC
        ${NUKA_CODEGEN_GENERATED_SOURCES}
        ${NUKA_CODEGEN_GENERATED_HEADERS}
    )
    set_target_properties(nuka_codegen_generated PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_dependencies(nuka_codegen_generated nuka_codegen)

    target_include_directories(nuka_codegen_generated
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_codegen_generated
        PUBLIC
            CUDA::cudart
    )

    add_library(nuka_codegen_v3_validation STATIC
        codegen/v3_validation/fd_harness.cu
        codegen/v3_validation/numerical_jacobian.cu
        codegen/v3_validation/box_pgs_fd.cu
    )
    set_target_properties(nuka_codegen_v3_validation PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_dependencies(nuka_codegen_v3_validation nuka_codegen)

    target_include_directories(nuka_codegen_v3_validation
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_codegen_v3_validation
        PUBLIC
            nuka_codegen_generated
            CUDA::cudart
    )

    # BUF-14 (M11): the legacy phi v1 device-context / owned-stream / stream-view
    # abstraction was DELETED -- every consumer now threads a raw cudaStream_t
    # (stream 0) + int device_id. nuka_phi survives carrying the relocated
    # scoped device guard + Device + platform-contract (BUF-13 already moved every
    # buffer consumer to the opaque phi-v2 Buffer* in nuka_phi2).
    add_library(nuka_phi STATIC
        phi/scoped_device_guard.cpp
        phi/platform_contract.cpp
        phi/backend_cuda/device.cu
    )
    set_target_properties(nuka_phi PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_include_directories(nuka_phi
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_phi
        PUBLIC
            CUDA::cudart
    )

    # Use CUDA-compatible warning flags (not the generic nuka_set_warnings)
    target_compile_options(nuka_phi PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4 /permissive- /WX /wd4819>
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror>
    )

    target_link_libraries(nuka_articulation
        PUBLIC
            nuka_phi
            CUDA::cudart
    )

    # Backend objects share the registry archive so static linking retains op registration.
    add_library(nuka_phi2_cuda OBJECT
        phi/backend_cuda/cuda_backend.cu
        phi/backend_cuda/cuda_buffer.cu
        phi/backend_cuda/ops/registry.cu
        phi/backend_cuda/ops/articulation.cu
        phi/backend_cuda/ops/crba.cu
        phi/backend_cuda/ops/contacts_foot.cu
        phi/backend_cuda/ops/sync_body_pose.cu
        # General contact pipeline Phase 2 (H3): the GENERAL heightfield narrowphase
        # — per-cell TRIANGLE_PRISM midphase routed through the EXISTING cvx GJK/EPA
        # (no terrain-type-specific contact code; the prism corner heights come from
        # the cooked `heights` grid, never an in-kernel SampleTerrainHeight).
        # PairDriven-family-gated (inert for UnionCsr).
        phi/backend_cuda/ops/narrowphase_heightfield.cu
        # Body/artic <-> soft/fluid particle narrowphase (particle = sphere on the
        # ONE path; PairDriven-family-gated).
        phi/backend_cuda/ops/narrowphase_body_particle.cu
        # M5 full-GPU collision: broadphase (BuildAabbs/Lbvh*)
        # + generalized pair-driven narrowphase (analytic prim set + sphere x
        # hull) + the SDF main path (SAMP x cooked sparse SDF grid). Ports of
        # collision/broadphase_lbvh / particle_uniform_grid + the shared
        # runtime/sdf sampler; the union slot-template path early-exits.
        phi/backend_cuda/ops/broadphase.cu
        phi/backend_cuda/ops/narrowphase_prims.cu
        phi/backend_cuda/ops/narrowphase_sdf.cu
        phi/backend_cuda/ops/assemble_rows.cu
        phi/backend_cuda/ops/block_descent.cu
        # Particle predict / projection velocity / finalize, aero drag and surface
        # refits: the particle side of the unified row solve.
        phi/backend_cuda/ops/particles.cu
        phi/backend_cuda/ops/vertex_blocks.cu
        phi/backend_cuda/ops/ogc_detect.cu
        phi/backend_cuda/ops/dat_truncate.cu
        # MLS-MPM transfer ops (P2G deterministic gather + G2P + grid clear) — the
        # inert round-trip scaffold; reachable through RegisterNkMpmOps, not yet in
        # Pipeline::Build's op list.
        phi/backend_cuda/ops/mpm.cu
        phi/backend_cuda/ops/readout.cu
        phi/backend_cuda/ops/energy_ledger.cu
        phi/backend_cuda/ops/sensors.cu
    )
    set_target_properties(nuka_phi2_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(nuka_phi2_cuda PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(nuka_phi2_cuda PRIVATE CUDA::cudart)
    target_compile_definitions(nuka_phi2 PRIVATE NUKA_PHI2_WITH_CUDA)
    target_link_libraries(nuka_phi2 PRIVATE nuka_phi2_cuda)

    # -------------------------------------------------------------------------
    # nuka_scene_cook — CookToModel (SceneIR -> {nk::Model, SceneMap}). Cleanest
    # non-cyclic home: depends on nuka_scene (CookScene + SceneIR/SceneMap),
    # nuka_articulation (CookArticulations), and nuka_nk (the Model tables). Kept
    # OUTSIDE nuka_scene so that library stays free of the nk / articulation deps.
    # Pure C++ (src/scene/cook/ is in the nk_engine lint scope); CUDA-gated only
    # because its nuka_nk dependency lives in this CMAKE_CUDA_COMPILER block.
    # -------------------------------------------------------------------------
    add_library(nuka_scene_cook STATIC
        scene/cook/cook_to_model.cpp
        scene/cook/settle.cpp
        scene/cook/placement.cpp
        # The general heightfield: the parametric grid generators + the self-written
        # grayscale image loader, the ONE source the cook + obs/render share.
        scene/terrain/heightfield_loaders.cpp
        scene/terrain/image_grayscale.cpp
        # L1-c: union_cook.cpp (CookSceneToUnionTemplate — the .nks h1_cup_table
        # -> UnionSceneTemplate cook) was DELETED with the UnionCsr path. The one
        # general cook is cook_to_model.cpp (CookSceneToModel -> PairDriven).
    )
    set_target_properties(nuka_scene_cook PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_include_directories(nuka_scene_cook
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_scene_cook
        PUBLIC
            nuka_nk
            nuka_scene
            nuka_articulation
        PRIVATE
            # The media cook reuses the cloth-topology + lattice cookers
            # (BuildClothVertexBlocks / CookParticleBox) compiled into nuka_runtime_gpu.
            nuka_runtime_gpu
    )

    nuka_set_warnings(nuka_scene_cook)

    # -------------------------------------------------------------------------
    # nuka_policy_mlp — host MLP policy runner (no libtorch, zero deps): the flat-
    # binary loader + linear/ELU forward + RunningMeanStd normalize. Used by the
    # inference controller and the C++-vs-Python validation tool.
    # -------------------------------------------------------------------------
    add_library(nuka_policy_mlp STATIC
        runtime/inference/mlp_policy.cpp
    )
    set_target_properties(nuka_policy_mlp PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(nuka_policy_mlp PUBLIC ${CMAKE_SOURCE_DIR}/src)
    nuka_set_warnings(nuka_policy_mlp)

    # -------------------------------------------------------------------------
    # nuka_inference_gpu — the device-resident policy: general batched MLP kernels
    # + the Go2 obs-assembly / drive-write kernels (runtime/inference/gpu_policy.cu).
    # A control step reads the device fields directly and stays on the world backend
    # stream the C ABI hands out (nuka_world_get_stream), so it issues NO host round-trip.
    # -------------------------------------------------------------------------
    add_library(nuka_inference_gpu STATIC
        runtime/inference/gpu_policy.cu
    )
    set_target_properties(nuka_inference_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(nuka_inference_gpu PUBLIC ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(nuka_inference_gpu PUBLIC CUDA::cudart)

    # -------------------------------------------------------------------------
    # nuka_inference — world controllers built on the policy runner. Go2Policy-
    # Controller reads world state through the C ABI fields, builds the height-scan
    # obs, runs the MLP, and writes PD drive targets. Production drives the GPU-
    # resident path (nuka_inference_gpu); the host MLP stays the numeric oracle.
    # -------------------------------------------------------------------------
    add_library(nuka_inference STATIC
        runtime/inference/go2_policy_controller.cpp
    )
    set_target_properties(nuka_inference PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(nuka_inference PUBLIC ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(nuka_inference PUBLIC nuka_policy_mlp nuka_inference_gpu nuka)
    nuka_set_warnings(nuka_inference)

    # -------------------------------------------------------------------------
    # nuka_app — M8 host frame-loop spine (runtime/app/): PosePublisher +
    # systems + Simulation. Steps an nk::World and (when a renderer is attached
    # + enabled) publishes its FK poses into a render::RenderWorld and rasters an
    # offscreen frame. Lives in this CUDA block because it consumes nuka_nk.
    #
    # Pure host C++ (src/runtime/app/** is in the zero-CUDA-token lint scope):
    # the only device touch is nk::World::GetData().DownloadField (host readback)
    # through nuka_nk's phi v2 vtable. The render dependency is gated per
    # Decision D5 — the VulkanRasterRenderer::Render draw call in systems.cpp is
    # compiled only when NK_BUILD_VULKAN_VALIDATION is defined, so a Vulkan-less
    # toolchain still links (RenderSystem::Run is then inert and Simulation never
    # attaches a renderer). nuka_render is linked for the shared RenderWorld /
    # RasterOptions / VulkanOffscreenReport types (Vulkan is a REQUIRED package
    # for this tree, so the link is always satisfiable here).
    # -------------------------------------------------------------------------
    add_library(nuka_app STATIC
        runtime/app/pose_publisher.cpp
        runtime/app/systems.cpp
        runtime/app/simulation.cpp
    )
    set_target_properties(nuka_app PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_include_directories(nuka_app
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_app
        PUBLIC
            nuka_nk
            nuka_render
            nuka_scene
    )

    if(NK_BUILD_VULKAN_VALIDATION)
        target_compile_definitions(nuka_app PUBLIC NK_BUILD_VULKAN_VALIDATION)
    endif()

    nuka_set_warnings(nuka_app)

    add_library(nuka_runtime_gpu STATIC
        core/diagnostics/invariants_gpu.cu
        # SAP broadphase: O(n^2) pair-slot oracle over a device AABB buffer. The
        # only remaining consumer is the LBVH-vs-SAP differential gate (the legacy
        # whole-world AABB-generation path was removed in the M0 cleanup).
        collision/gpu/broadphase.cu
        # Cook-time HOST builders: cloth/rod vertex blocks, tet lattices and skins.
        runtime/soft/cloth_topology.cpp
        runtime/soft/tetmesh_topology.cpp
        # The box/sphere lattice sampler (ParticleVolumeSpec) the MLS-MPM media cook uses.
        import/cooker/particle_volume_cooker.cpp
        # COL-5 (M11): runtime/sdf/sdf_device_world.cu DELETED. Its sole duty (upload
        # a cooked scene::CookedSdfTable to the device) now lives in nk::Model
        # (model.hpp sdf_cell_* + cook_to_model.cpp + narrowphase_sdf.cu LoadSdfGrid);
        # the cooked-table->device round-trip + cell-count coverage folded into the
        # SDF precision oracle (tests/collision/test_gjk_epa_convex.cpp).
    )
    set_target_properties(nuka_runtime_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_include_directories(nuka_runtime_gpu
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_runtime_gpu
        PUBLIC
            nuka_runtime
            nuka_solver
            nuka_phi
            # phi v2: collision/gpu/broadphase.cu (the SAP oracle) allocates via the
            # opaque phi-v2 Buffer* (stream-0). All nuka_runtime_gpu TUs are on phi-v2.
            nuka_phi2
            # nuka_collision_gpu: retained for the SAP broadphase oracle + the
            # candidate-pair / particle-grid primitives surviving consumers link
            # transitively (the deleted PBF/co-step's particle-grid use is gone, but
            # the dependency edge stays for the rest of the GPU collision estate).
            nuka_collision_gpu
    )

    add_library(nuka_collision_gpu STATIC
        collision/broadphase_lbvh.cu
        collision/lbvh_refit.cu
        # The shared BATCHED, single-launch per-env LBVH build/refit primitive
        # (collision/lbvh_batched.cuh). ONE copy of the env-build kernels called
        # by both the pair-driven broadphase op (phi backend) and the render
        # TLAS. Same library + default --fmad as the single-tree build above.
        collision/lbvh_batched.cu
        collision/particle_uniform_grid.cu
        collision/cross_system_query.cu
        collision/sdf_contact.cu
        collision/sdf_contact_adjoint.cpp
        # v0.8 C2a: the unified broadphase OUTPUT type. CandidatePairStream is a
        # D1 device container of type-tagged CandidatePairs sorted by stable_key
        # (thrust::stable_sort_by_key, no append atomics). C2b/C2c populate it
        # from the real broadphase sources via private-slice fill and call the
        # same builder. candidate_pair.hpp includes the GENERATED collidable
        # registry header -> this TU is the FIRST in nuka_collision_gpu to depend
        # on codegen output, so the lib add_dependencies(nuka_codegen) below.
        collision/candidate_pair.cu
        # v0.8 C2b: the RIGID<->RIGID filtered candidate-pair builder. LBVH over
        # per-shape world AABBs -> per-pair bitmask+exclude filter (count->scan->
        # compact, NO append atomics) -> body-pair CandidatePairs -> host
        # explicit-<pair> merge -> BuildCandidatePairStream (C2a). A NEW
        # validated-not-forced path (rigid-only); NOT wired into the stepper.
        collision/rigid_candidate_pairs.cu
        # v0.8 C2c: the particle<->rigid + particle<->particle candidate-pair
        # builders. REUSE the cross_system_query (particle->rigid LBVH) + the
        # particle_uniform_grid (particle->particle) CSR queries, reframe each CSR
        # entry into its offset slot (NO scan, NO append atomic) -> body/particle
        # CandidatePairs -> BuildCandidatePairStream (C2a) -> adjacent dedup. Gated
        # by the C1c SystemPairMatrix. NEW validated path; NOT wired into production
        # (the PBD co-step wiring is C6a). Excludes articulation-link (no per-link
        # AABB yet -- the C5b-prereq long-pole).
        collision/particle_candidate_pairs.cu
    )
    set_target_properties(nuka_collision_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # candidate_pair.cu includes src/codegen/generated/collidable_registry.hpp
    # (via constraint/collidable.hpp); force the codegen pass to run first so a
    # clean/parallel build never compiles the TU before the header exists.
    add_dependencies(nuka_collision_gpu nuka_codegen)

    target_include_directories(nuka_collision_gpu
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_collision_gpu
        PUBLIC
            nuka_collision
            nuka_constraint
            nuka_phi
            # phi v2: broadphase_lbvh / particle_uniform_grid / cross_system_query /
            # candidate_pair / rigid+particle candidate pairs allocate via the opaque
            # phi-v2 Buffer* (stream-0 DeviceBufferType). All TUs are on phi-v2.
            nuka_phi2
            CUDA::cudart
    )

    # --- v0.7 p12: self-written CUDA ray tracer (camera -> LBVH -> closest-hit
    # depth, box leaves). NO OptiX / NO closed SDK. NAMED CONSUMER that wires the
    # retained LBVH (BuildLbvhForQuery) into the runtime renderer. Stackless
    # parent-pointer ray traversal, one thread per pixel, ZERO atomics -> D1
    # byte-exact. --fmad=false: the leaf slab test (phi/backend_cuda/rt/ray_box.cuh)
    # is a shared __host__ __device__ function whose t must match the CPU oracle
    # bit-for-bit; forbidding FMA contraction on the device side guarantees
    # host==device arithmetic (the oracle TU pairs this with -ffp-contract=off).
    # v0.7 p13: the sensor renderer (scene_render.cu) -- real per-primitive leaf
    # intersections (triangle/sphere/sparse-SDF) + Lambert+GGX shade + hard
    # shadows + full AOV framebuffer. Shares nuka_phi2_rt's --fmad=false so the
    # fp64-internal intersections stay bit-exact vs the CPU oracle. Links
    # nuka_cooker (no link dep here; the SDF cell arrays are passed in by the
    # caller via SparseSdfDevice) -- only the runtime sampler header is used.
    # v0.7 G1-A: the self-written TWO-LEVEL (TLAS/BLAS) tracer
    # (two_level_render.cu). Nests the p12 stackless rt::TraverseRay two deep:
    # a per-frame TLAS over instance world-AABBs whose leaf transforms the ray to
    # each instance's local frame and runs an INNER traversal over that instance's
    # once-built BLAS. RIGID two-level core (p16 demo: H1 + cup + table + USD =
    # many rigid mesh instances). Shares --fmad=false so the fp64 leaf math +
    # fp32 QuatInvRotate stay bit-exact vs the host oracle (-ffp-contract=off).
    #
    # M11 RT-CMAKE/RT-3 (§3.10): the 3 device .cu + 6 .cuh are HOMED to
    # src/phi/backend_cuda/rt/ (the CUDA backend layer); the CUDA-free scene-desc
    # POD / interface headers stay in src/rt (interface-types dir). This DEDICATED
    # lib keeps --fmad=false: it MUST NOT be folded into nuka_phi2 (nvcc default
    # --fmad=true would silently diverge every RT golden). Renamed nuka_rt_gpu ->
    # nuka_phi2_rt to mark it as the RT slice of the CUDA backend.
    add_library(nuka_phi2_rt STATIC
        phi/backend_cuda/rt/bvh_ray_traversal.cu
        phi/backend_cuda/rt/scene_render.cu
        phi/backend_cuda/rt/two_level_render.cu
        # Device instance-transform + world-AABB scatter for the batched sensor
        # path; shares this lib's --fmad=false so the fk*cvl compose is byte-exact.
        phi/backend_cuda/rt/sensor_scatter.cu
        # Batched N-env sensor render: ONE batched LBVH build over per-env TLASes +
        # ONE flat trace over [E*H*W] rays (persistent high-quality RGB/AOV path)
        # into a (N,H,W,ch) device tensor. Reuses the shared traversal nest + the
        # batched LBVH build.
        phi/backend_cuda/rt/batched_sensor_render.cu
        # M11 RT-1/RT-3 (§3.10): the CONCRETE CUDA impl of render::RtBackendI.
        # Host C++ (no kernel launch) that WRAPS BuildTwoLevelScene/RenderFrame;
        # homed in the CUDA backend layer. Linking nuka_phi2_rt provides the
        # strong CreateCudaRtBackend definition. Built here so it shares the
        # lib's include dirs; --fmad=false only affects the CUDA .cu above.
        phi/backend_cuda/rt/rt_backend_cuda.cpp
        # The CONCRETE CUDA impl of render::SensorBackendI (sibling of RtBackendI).
        # Host C++ (no kernel launch) that WRAPS BuildBatchedSensorScene/
        # RenderSensorsMounted; linking nuka_phi2_rt provides the strong
        # CreateCudaSensorBackend definition.
        phi/backend_cuda/rt/sensor_backend_cuda.cpp
    )
    set_target_properties(nuka_phi2_rt PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_options(nuka_phi2_rt PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    )
    target_include_directories(nuka_phi2_rt
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )

    # The FP32 BEAUTY translation unit of the two-level tracer is a DEDICATED lib
    # so --fmad=true (FMA on) applies ONLY here, NEVER folded into the byte-exact
    # golden lib above (which keeps --fmad=false). The beauty kernel instantiates
    # the shared nested-traversal fns with Real=float for full FP32+FMA trace rate;
    # the golden kernel + host oracle stay FP64. Linked into nuka_phi2_rt so callers
    # resolve LaunchBeautyKernel transparently.
    add_library(nuka_phi2_rt_beauty STATIC
        phi/backend_cuda/rt/two_level_render_beauty.cu
    )
    set_target_properties(nuka_phi2_rt_beauty PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_options(nuka_phi2_rt_beauty PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>
    )
    target_include_directories(nuka_phi2_rt_beauty
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_phi2_rt_beauty
        PUBLIC
            nuka_collision
            nuka_collision_gpu
            nuka_phi
            nuka_phi2
            CUDA::cudart
    )

    target_link_libraries(nuka_phi2_rt
        PUBLIC
            nuka_particle_geometry
            nuka_collision
            nuka_collision_gpu
            # RT-7 (M11): the rt device buffers are phi-v2 opaque Buffer*
            # (stream-0); nuka_phi2 supplies BufferAlloc/InitBestDevice/cuda_buffer.
            # nuka_phi stays for the scoped device guard (BUF-14).
            nuka_phi
            nuka_phi2
            nuka_phi2_rt_beauty
            CUDA::cudart
    )

    add_library(nuka_contact_wrench STATIC
        # p14a (v0.7): any-link contact-force readout (per-link wrench + per-slot
        # force). Pure readback of the solved lambda + stage-4/5 contact geometry;
        # depends only on nuka_core/nuka_phi + math, so compiling it INTO this lib
        # (the one that owns the batched world that calls it) avoids any new
        # inter-library dependency cycle.
        sensor/contact_wrench.cu
    )
    set_target_properties(nuka_contact_wrench PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_include_directories(nuka_contact_wrench
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_contact_wrench
        PUBLIC
            nuka_articulation
            nuka_core
            nuka_phi
            CUDA::cudart
    )

    target_link_libraries(nuka_runtime
        PUBLIC
            nuka_contact_wrench
    )

    add_library(nuka_sensor_noise STATIC
        sensor/noise/n1_gaussian.cu
        sensor/noise/n1_poisson.cu
        sensor/noise/n2_domain_randomization.cpp
    )
    set_target_properties(nuka_sensor_noise PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_include_directories(nuka_sensor_noise
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_sensor_noise
        PUBLIC
            nuka_phi
            CUDA::cudart
    )

    add_library(nuka SHARED
        c_abi/device.cpp
        c_abi/world.cpp
        c_abi/params.cpp
        c_abi/checkpoint.cpp
        c_abi/cuda_interop.cpp
        c_abi/buffer.cpp
        c_abi/error.cpp
        c_abi/noise.cpp
        c_abi/state_sensor.cpp
        c_abi/recorder.cpp
        c_abi/scene.cpp
        # The BUILT-SCENE authoring surface (program rigid primitives + TAGGED media
        # records, then cook via the SAME CookSceneToModel a file scene uses). Reuses
        # the scene handle (SceneTable) + the shared world-create helpers in world.cpp
        # -> ONE nk::World on the ONE cook path (no second world type). Host-only.
        c_abi/scene_builder.cpp
        # The device-resident batched camera-sensor surface (attach/render/view):
        # cooks the world's retained scene to the per-env visual binding via
        # BuildRenderWorld + RenderWorldToTwoLevelScene (nuka_render) and drives the
        # batched sensor backend (strong CreateCudaSensorBackend from nuka_phi2_rt).
        c_abi/sensor.cpp
        # The offline beauty render of the LIVE world (host RGB image): the shared
        # studio render scene (BuildStudioScene / PublishStudioScene from nuka_render)
        # + the offline RT beauty tracer (strong CreateCudaRtBackend from nuka_phi2_rt).
        c_abi/render_beauty.cpp
        collision/contact_stream_driver.cpp
    )

    target_include_directories(nuka
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka
        PRIVATE
            nuka_core
            nuka_runtime
            nuka_import
            nuka_contact_wrench
            nuka_sensor_noise
            nuka_phi
            nuka_solver
            nuka_constraint
            nuka_collision_gpu
            # M7: the C-ABI union world cooks examples/scenes/h1_cup_table.nks
            # (CookSceneToUnionTemplate) instead of calling the deleted factory.
            # nuka_scene_cook transitively pulls nuka_nk + nuka_scene +
            # nuka_articulation; nks::Load lives in nuka_import (already linked).
            nuka_scene_cook
            # The media cook (nuka_scene_cook) reuses the cloth-topology + fluid-box
            # cookers (BuildClothConstraints / CookParticleBox) which live in
            # nuka_runtime_gpu; other c_abi surfaces (sensor / render) link it too.
            nuka_runtime_gpu
            # The device-resident batched camera-sensor surface (c_abi/sensor.cpp):
            # nuka_render = the cook->visual bridge (BuildRenderWorld /
            # RenderWorldToTwoLevelScene + the SensorBackendI interface);
            # nuka_phi2_rt = the strong CreateCudaSensorBackend + the batched render.
            nuka_render
            nuka_phi2_rt
        PUBLIC
            CUDA::cudart
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(nuka PUBLIC stdc++)
    endif()

    # v0.8 A2: the grasp path compiles batched_unified_world.cpp, which includes the
    # GENERATED collidable registry (candidate_pair.hpp) -- ensure codegen runs first.
    add_dependencies(nuka nuka_codegen)

    if(NK_BUILD_VULKAN_VALIDATION)
        set_source_files_properties(c_abi/recorder.cpp PROPERTIES
            COMPILE_DEFINITIONS NK_BUILD_VULKAN_VALIDATION)
        target_link_libraries(nuka PRIVATE nuka_app nuka_render)
        if(MSVC)
            # The recorder loads vulkan-1.dll on first use, so headless runs need no Vulkan loader.
            target_link_options(nuka PRIVATE /DELAYLOAD:vulkan-1.dll)
            target_link_libraries(nuka PRIVATE delayimp)
        endif()
    endif()

    set_target_properties(nuka PROPERTIES
        OUTPUT_NAME nuka
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        PUBLIC_HEADER "${NUKA_PUBLIC_HEADERS}"
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )

    nuka_set_warnings(nuka)

    # v0.8 A2: the host contact driver keeps the warning posture of its authoring
    # home (the test target, which does NOT apply nuka_set_warnings) rather than
    # getting the library's strict -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    # -Werror imposed by promotion. This byte-exact, gate-validated source was never
    # written under those flags; re-auditing it would risk the byte-identity the
    # gates pin. `-w` is appended AFTER -Werror per source file, so GCC suppresses
    # the warnings before -Werror can elevate them.
    set_source_files_properties(
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
        PROPERTIES COMPILE_OPTIONS "-w"
    )

    install(TARGETS nuka
        EXPORT nukaTargets
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include/nuka
    )

    install(EXPORT nukaTargets
        FILE nukaTargets.cmake
        NAMESPACE nuka::
        DESTINATION lib/cmake/nuka
    )

    if(NK_BUILD_EDITOR)
        # The pose scatter composes like the host publish only without FMA contraction.
        add_library(nuka_editor_scatter STATIC runtime/app/viewer/pose_scatter.cu)
        set_target_properties(nuka_editor_scatter PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_compile_options(nuka_editor_scatter PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
        target_include_directories(nuka_editor_scatter PUBLIC ${CMAKE_SOURCE_DIR}/src)
        target_link_libraries(nuka_editor_scatter PUBLIC nuka nuka_render CUDA::cudart)

        # The editor reaches physics only through the nuka C ABI; nuka_editor_core is its
        # world binding, scene cache, edits, undo, overlay, physics thread and pose scatter.
        add_library(nuka_editor_core STATIC
            runtime/app/viewer/editor_world.cpp
            runtime/app/viewer/editor_scene.cpp
            runtime/app/viewer/editor_edits.cpp
            runtime/app/viewer/editor_document.cpp
            runtime/app/viewer/debug_overlay.cpp
            runtime/app/viewer/physics_worker.cpp
        )
        set_target_properties(nuka_editor_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_include_directories(nuka_editor_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
        target_link_libraries(nuka_editor_core PUBLIC nuka nuka_viewer nuka_render nuka_import nuka_editor_scatter)
        nuka_set_warnings(nuka_editor_core)

        # nuka_editor -- the editor shared library behind nuka/nuka_editor.h. Python's
        # nuka.editor and the nuka-editor launcher load it; it exports only nuka_editor_*.
        add_library(nuka_editor SHARED
            runtime/app/viewer/editor_host.cpp
            runtime/app/viewer/editor_api.cpp
        )
        add_dependencies(nuka_editor nuka_render_shaders)
        set_target_properties(nuka_editor PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON)
        target_compile_definitions(nuka_editor PRIVATE
            NUKA_EDITOR_BUILD
            NUKA_VIEWER_DEFAULT_SCENE="${CMAKE_SOURCE_DIR}/examples/assets/nuka_lab/gripper.nks")
        target_link_libraries(nuka_editor
            PRIVATE
                nuka
                nuka_editor_core
                nuka_inference
                nuka_viewer
                nuka_present
                nuka_imgui
                nuka_render
                nuka_import
                imgui
                Vulkan::Vulkan
        )
        if(NOT WIN32)
            file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/nuka_editor.map" "{ global: nuka_editor_*; local: *; };\n")
            target_link_options(nuka_editor PRIVATE
                "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/nuka_editor.map")
        endif()
        nuka_copy_viewer_resources(nuka_editor)
        nuka_set_warnings(nuka_editor)
        install(TARGETS nuka_editor LIBRARY DESTINATION lib RUNTIME DESTINATION bin)

        # nuka-editor -- the launcher: an embedded interpreter running `python -m nuka.editor`.
        find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Embed)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import sys; sys.stdout.write(sys.prefix)"
            OUTPUT_VARIABLE NUKA_PY_HOME)
        # Forward slashes keep the Windows prefix a valid C string literal.
        file(TO_CMAKE_PATH "${NUKA_PY_HOME}" NUKA_PY_HOME)
        add_executable(nuka_editor_exe runtime/app/viewer/editor_exe_main.cpp)
        # A distinct output name keeps the launcher's PDB apart from nuka_editor.dll's.
        set_target_properties(nuka_editor_exe PROPERTIES OUTPUT_NAME nuka-editor)
        # Python headers as SYSTEM so libpython's macros never trip -Werror.
        target_include_directories(nuka_editor_exe SYSTEM PRIVATE ${Python3_INCLUDE_DIRS})
        target_compile_definitions(nuka_editor_exe PRIVATE NUKA_PY_HOME="${NUKA_PY_HOME}")
        target_link_libraries(nuka_editor_exe PRIVATE Python3::Python)
        if(WIN32)
            # The interpreter DLL sits beside the launcher so no other python3x.dll on PATH wins.
            add_custom_command(TARGET nuka_editor_exe POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_RUNTIME_DLLS:nuka_editor_exe> $<TARGET_FILE_DIR:nuka_editor_exe>
                COMMAND_EXPAND_LISTS)
        endif()
        nuka_set_warnings(nuka_editor_exe)
        install(TARGETS nuka_editor_exe RUNTIME DESTINATION bin)
    endif()

    add_executable(nuka_pipeline_benchmark
        ${CMAKE_SOURCE_DIR}/tools/perf/pipeline_benchmark.cpp)
    target_include_directories(nuka_pipeline_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(nuka_pipeline_benchmark PRIVATE NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(nuka_pipeline_benchmark PRIVATE
        nuka nuka_scene_cook nuka_render nuka_phi2_rt nuka_nk nuka_phi2 nuka_runtime_gpu nuka_runtime
        nuka_import nuka_cooker nuka_contact_wrench nuka_collision_gpu nuka_core nuka_phi CUDA::cudart)
    nuka_set_warnings(nuka_pipeline_benchmark)

    # The water demo uses CUDA physics and ray tracing without a Vulkan dependency.
    add_executable(nuka_mpm_water_drop_demo
        ${CMAKE_SOURCE_DIR}/examples/demo/mpm_water_drop_demo.cpp)
    add_dependencies(nuka_mpm_water_drop_demo nuka_codegen)
    set_target_properties(nuka_mpm_water_drop_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    target_compile_definitions(nuka_mpm_water_drop_demo PRIVATE NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_include_directories(nuka_mpm_water_drop_demo PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
    target_link_libraries(nuka_mpm_water_drop_demo PRIVATE
        nuka_scene_cook nuka_render nuka_phi2_rt nuka_nk nuka_phi2 nuka_runtime_gpu
        nuka_runtime nuka_import nuka_cooker nuka_contact_wrench nuka_collision_gpu
        nuka_core nuka_phi CUDA::cudart)

    # Compose a robot .nks over a procedural field heightfield (visual-mesh
    # surface + N re-rooted robots on the local surface). HOST-ONLY.
    add_executable(nuka_field_scene
        ${CMAKE_SOURCE_DIR}/tools/scene/field_scene.cpp
    )
    target_include_directories(nuka_field_scene
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_field_scene
        PRIVATE
            nuka_import
            nuka_scene
    )
    nuka_set_warnings(nuka_field_scene)

    # Author a Go2-on-composite-rolling-heightfield .nks the editor simulates: a
    # first-class COLLISION TerrainRecord + a visual surface + K floating dogs.
    add_executable(nuka_go2_terrain_field
        ${CMAKE_SOURCE_DIR}/tools/scene/go2_terrain_field.cpp
    )
    target_include_directories(nuka_go2_terrain_field
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_go2_terrain_field
        PRIVATE
            nuka_import
            nuka_scene_cook   # GenerateHeightField (transitively pulls nuka_scene)
    )
    nuka_set_warnings(nuka_go2_terrain_field)

    # C++-vs-Python policy parity gate: load a flat policy binary + the torch golden
    # (export_go2_policy.py) and assert the host MlpPolicy forward matches < 1e-4.
    add_executable(nuka_mlp_policy_validate
        ${CMAKE_SOURCE_DIR}/tools/inference/mlp_policy_validate.cpp
    )
    target_include_directories(nuka_mlp_policy_validate
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_mlp_policy_validate
        PRIVATE
            nuka_policy_mlp
    )
    nuka_set_warnings(nuka_mlp_policy_validate)

    # nuka_gpu_mlp_policy_validate -- the GPU-vs-host (and GPU-vs-torch) inference
    # parity harness: same weights into the host oracle + device GpuMlp, assert
    # max|GPU - host| < 1e-4.
    add_executable(nuka_gpu_mlp_policy_validate
        ${CMAKE_SOURCE_DIR}/tools/inference/gpu_mlp_policy_validate.cpp
    )
    target_include_directories(nuka_gpu_mlp_policy_validate
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_gpu_mlp_policy_validate
        PRIVATE
            nuka_policy_mlp
            nuka_inference_gpu
    )
    nuka_set_warnings(nuka_gpu_mlp_policy_validate)
endif()

# Host-only scene conversion is available without a CUDA compiler.
add_executable(nuka_cook_scene
    ${CMAKE_SOURCE_DIR}/tools/scene/cook_scene.cpp
)
target_include_directories(nuka_cook_scene
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(nuka_cook_scene
    PRIVATE
        nuka_import
        nuka_scene
)
nuka_set_warnings(nuka_cook_scene)
