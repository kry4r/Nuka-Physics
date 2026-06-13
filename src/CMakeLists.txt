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
    # v0.7 p07: narrow-band sparse SDF cooker (HOST CPU; the cooked SDF is a
    # byte-reproducible golden, so it runs on the deterministic host path rather
    # than taking the cook-time GPU exception). The runtime query header
    # (runtime/sdf/sparse_sdf_query.cuh) is CUDA-free and is included here too.
    import/cooker/sparse_sdf_cooker.cpp
    # v0.7 p07 seam: the swappable SDF bake backend (CpuSdfBakeBackend now; the
    # v1.0 GPU backend slots in behind DefaultSdfBakeBackend). Owner 2026-06-02.
    import/cooker/sdf_bake_backend.cpp
)
set_target_properties(nuka_cooker PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
    scene/cooker.cpp
    scene/pose_graph.cpp
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
    ${CMAKE_SOURCE_DIR}/src/include/nuka/nuka_diffsim.h
    ${CMAKE_SOURCE_DIR}/src/include/nuka/nuka_noise.h
    ${CMAKE_SOURCE_DIR}/src/include/nuka/nuka_grasp.h
    ${CMAKE_SOURCE_DIR}/src/include/nuka/nuka_union.h
    # M8 T5: the offscreen recorder C ABI (PPM P6 capture + ffmpeg mp4). Its
    # IMPLEMENTATION (c_abi/recorder.cpp) is gated behind NK_BUILD_VULKAN_VALIDATION
    # (Decision D5) but the header + the exported symbols are ALWAYS present so the
    # python binding imports on a Vulkan-less build (every entry returns
    # NUKA_RESULT_NOT_SUPPORTED there).
    ${CMAKE_SOURCE_DIR}/src/include/nuka/nuka_recorder.h
)

add_library(nuka_render STATIC
    render/render_scene.cpp
    render/render_world.cpp
    render/vulkan_renderer.cpp
    render/raster/vulkan_raster_renderer.cpp
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

# M8 T3a raster forward-pass shaders (mesh.vert / mesh_pbr.frag) -- same glslc
# custom_command pattern as the compute debug_draw above.
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

add_custom_target(nuka_render_shaders DEPENDS
    "${NUKA_DEBUG_DRAW_SPV}"
    "${NUKA_RASTER_MESH_VERT_SPV}"
    "${NUKA_RASTER_MESH_FRAG_SPV}"
)
add_dependencies(nuka_render nuka_render_shaders)
target_compile_definitions(nuka_render
    PRIVATE
        NUKA_VULKAN_DEBUG_DRAW_SPV="${NUKA_DEBUG_DRAW_SPV}"
        NUKA_RASTER_MESH_VERT_SPV="${NUKA_RASTER_MESH_VERT_SPV}"
        NUKA_RASTER_MESH_FRAG_SPV="${NUKA_RASTER_MESH_FRAG_SPV}"
)

target_include_directories(nuka_render
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_render
    PUBLIC
        nuka_scene
        Vulkan::Vulkan
)

nuka_set_warnings(nuka_render)

# ---------------------------------------------------------------------------
# nuka_imgui (M8.5 T1) -- the nuka-side Dear ImGui glue + custom theme.
#
# Mirrors nuka_render: a STATIC lib that links Vulkan::Vulkan and the vendored
# `imgui` lib (external/imgui, declared in the root CMakeLists, SYSTEM includes).
# GATED behind NK_BUILD_VULKAN_VALIDATION (Decision D3): the default build-cuda128
# config never builds it, so libnuka + core are untouched. HOST-ONLY / zero-CUDA;
# the vendored ImGui is a SYSTEM include so it stays out of the src/render lint scope.
#
# The JetBrains Mono font paths are injected as compile-definitions (the same idiom
# nuka_render uses for its compiled-SPV paths) so nuka_imgui.cpp loads the vendored
# TTFs by absolute path at runtime.
# ---------------------------------------------------------------------------
if(NK_BUILD_VULKAN_VALIDATION)
    add_library(nuka_imgui STATIC
        render/imgui/nuka_imgui.cpp
    )

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
            NUKA_IMGUI_FONT_UI_TTF="${CMAKE_SOURCE_DIR}/external/imgui/fonts/JetBrainsMono-Regular.ttf"
            NUKA_IMGUI_FONT_HEADING_TTF="${CMAKE_SOURCE_DIR}/external/imgui/fonts/JetBrainsMono-Bold.ttf"
    )

    nuka_set_warnings(nuka_imgui)
endif()

# ---------------------------------------------------------------------------
# nuka_present (M8.5 T2) -- the realtime SWAPCHAIN present renderer + the thin
# self-written xcb windowing/surface backend (Decision D1).
#
# STRATEGY-SPLIT: this is a SEPARATE lib from nuka_render. It links nuka_render
# (to borrow the present-capable VulkanRasterRenderer for device/queue/handles)
# but owns the swapchain present path so the offscreen G2 determinism oracle in
# nuka_render stays pristine. The realtime present loop is legitimately non-D1
# (vsync / frames-in-flight) and is isolated here.
#
# GATED behind NK_BUILD_VULKAN_VALIDATION (Decision D3): the default build-cuda128
# config never builds it -> libnuka + core untouched. Links libxcb + X11 (the
# Decision-D1 windowing dep; find_package(X11) is wired in the root CMakeLists
# under the same gate). The raster forward-pass SPV paths are re-injected here so
# the present pipeline loads the SAME compiled shaders as the offscreen renderer.
# HOST-ONLY / zero-CUDA-token: pure C++ / Vulkan / xcb.
# ---------------------------------------------------------------------------
if(NK_BUILD_VULKAN_VALIDATION)
    add_library(nuka_present STATIC
        render/raster/vulkan_present_renderer.cpp
        render/window/window_surface.cpp
    )
    add_dependencies(nuka_present nuka_render_shaders)

    target_include_directories(nuka_present
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
            ${X11_INCLUDE_DIR}
    )

    target_compile_definitions(nuka_present
        PRIVATE
            NUKA_RASTER_MESH_VERT_SPV="${NUKA_RASTER_MESH_VERT_SPV}"
            NUKA_RASTER_MESH_FRAG_SPV="${NUKA_RASTER_MESH_FRAG_SPV}"
    )

    # libxcb: the self-written surface backend uses xcb directly. X11_xcb_LIB is set
    # by find_package(X11) when the xcb component is present; fall back to the bare
    # library name otherwise (the dev header + so are verified present at configure).
    if(X11_xcb_LIB)
        set(NUKA_XCB_LIB ${X11_xcb_LIB})
    else()
        set(NUKA_XCB_LIB xcb)
    endif()

    target_link_libraries(nuka_present
        PUBLIC
            nuka_render
            Vulkan::Vulkan
            ${NUKA_XCB_LIB}
            ${CMAKE_DL_LIBS}
    )

    nuka_set_warnings(nuka_present)
endif()

add_library(nuka_scene_pipeline STATIC
    scene/scene_pipeline.cpp
)

target_include_directories(nuka_scene_pipeline
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_scene_pipeline
    PUBLIC
        nuka_scene
        nuka_runtime
        nuka_render
)

nuka_set_warnings(nuka_scene_pipeline)

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

# ---------------------------------------------------------------------------
# nuka_solver – Iterative rigid body constraint solver (PGS)
# ---------------------------------------------------------------------------
add_library(nuka_solver STATIC
    solver/diff_test_bridge.cpp
    # v0.8 C5a: UnifiedSolve standalone entry (validated-not-wired).
    solver/unified_solve.cpp
)
set_target_properties(nuka_solver PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(nuka_solver
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_solver
    PUBLIC
        nuka_constraint
        nuka_math
)

nuka_set_warnings(nuka_solver)

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

# ---------------------------------------------------------------------------
# nuka_phi – Physics Hardware Interface (CUDA backend)
# ---------------------------------------------------------------------------
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

    # --- V3 finite-difference validation harness (v0.5 p01 adjoint slice) ----
    # Validates each generated reverse-mode adjoint against a central-difference
    # numerical Jacobian of the SAME generated primal. Slice: maximal_drive.
    add_library(nuka_codegen_v3_validation STATIC
        codegen/v3_validation/fd_harness.cu
        codegen/v3_validation/numerical_jacobian.cu
        codegen/v3_validation/box_pgs_fd.cu
        # v0.7 p09-A: XPBD distance-row adjoint FD validator (dense_adjoint).
        codegen/v3_validation/xpbd_distance_fd.cu
        # v0.7 p09-B: XPBD bend (id 7) + volume (id 8) adjoint FD validators.
        codegen/v3_validation/xpbd_bend_fd.cu
        codegen/v3_validation/xpbd_volume_fd.cu
        # v0.7 p09-C: XPBD shape-match (id 9) goal-projection adjoint FD validator.
        codegen/v3_validation/xpbd_shape_match_fd.cu
        # v0.7 p11 (K3): particle-particle contact (id 10) adjoint FD validator --
        # the SAME multilinear XPBD multiplier law as the distance row (only the
        # unilateral box / active-contact sampling differs).
        codegen/v3_validation/particle_particle_contact_fd.cu
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

    add_library(nuka_phi STATIC
        phi/device_context.cpp
        phi/platform_contract.cpp
        phi/backend_cuda/device.cu
        phi/backend_cuda/buffer.cu
        phi/backend_cuda/owned_stream.cu
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

    # -----------------------------------------------------------------------
    # nuka_phi2 — PHI v2: ggml-style runtime multi-backend layer (M1).
    #
    # Built ALONGSIDE the legacy nuka_phi (which all production paths still
    # consume until M9). PHI v2 has NO production consumers yet — only the
    # phi2_smoke scenario gate. Same CUDA-gating / PIC / include / warning
    # policy as nuka_phi.
    #
    # NUKA_PHI2_WITH_CUDA: drives the explicit, linker-safe CUDA registry
    # self-registration in registry.cpp -> RegisterCudaBackendEntry() (see the
    # static-lib self-registration de-risk comment there). A non-CUDA build
    # would compile registry.cpp with the macro undefined and register nothing.
    # -----------------------------------------------------------------------
    add_library(nuka_phi2 STATIC
        phi/registry.cpp
        phi/backend_cuda/cuda_backend.cu
        phi/backend_cuda/cuda_buffer.cu
        phi/backend_cuda/ops/registry.cu
        # M3b articulation pipeline op implementations (line-by-line kernel
        # ports; see each file's header). M4 adds the union-family contact
        # pipeline (contacts_union / assemble_rows / solve_rows — the §3.4
        # device-resident unified row solve, replacing the transitional
        # solve_articulated.cu).
        phi/backend_cuda/ops/articulation.cu
        phi/backend_cuda/ops/crba.cu
        phi/backend_cuda/ops/contacts_foot.cu
        phi/backend_cuda/ops/contacts_union.cu
        # M5 full-GPU collision: broadphase (BuildAabbs/Lbvh*/ParticleGridBuild)
        # + generalized pair-driven narrowphase (analytic prim set + sphere x
        # hull) + the SDF main path (SAMP x cooked sparse SDF grid). Ports of
        # collision/broadphase_lbvh / particle_uniform_grid + the shared
        # runtime/sdf sampler; the union slot-template path early-exits.
        phi/backend_cuda/ops/broadphase.cu
        phi/backend_cuda/ops/narrowphase_prims.cu
        phi/backend_cuda/ops/narrowphase_sdf.cu
        phi/backend_cuda/ops/assemble_rows.cu
        phi/backend_cuda/ops/solve_rows.cu
        # M6 multi-physics coupled co-step: XPBD soft + PBF fluid kernel ports
        # (predict / xpbd project / pbf density-lambda / pbf apply / finalize),
        # the particle side of the unified row solve.
        phi/backend_cuda/ops/particles.cu
        phi/backend_cuda/ops/readout.cu
    )
    set_target_properties(nuka_phi2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_compile_definitions(nuka_phi2 PUBLIC NUKA_PHI2_WITH_CUDA)

    target_include_directories(nuka_phi2
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_phi2
        PUBLIC
            CUDA::cudart
    )

    target_compile_options(nuka_phi2 PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4 /permissive- /WX /wd4819>
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror>
    )

    # -----------------------------------------------------------------------
    # nuka_nk — the new unified core (M3): Model / Data arena / generated field
    # views / Pipeline + World skeleton. PURE C++ (zero CUDA tokens; lint scope
    # nk_engine): all device memory + execution through the phi v2 vtable
    # (nuka_phi2). The generated field headers (src/nk/model/generated/) are
    # committed; they are #included as ordinary headers (regen via
    # tools/codegen/regen.py). Depends only on nuka_phi2 + nuka_math.
    # -----------------------------------------------------------------------
    add_library(nuka_nk STATIC
        nk/model/model.cpp
        nk/data/arena.cpp
        nk/data/data.cpp
        nk/pipeline/pipeline.cpp
        nk/pipeline/world.cpp
        nk/solve/schedule.cpp
    )
    set_target_properties(nuka_nk PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_include_directories(nuka_nk
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_nk
        PUBLIC
            nuka_phi2
            nuka_math
    )

    nuka_set_warnings(nuka_nk)

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
        # M7 T2: the reusable cook primitives — the deterministic settle (runs an
        # already-built nk::World, PD-holds named dof patterns, snapshots +
        # downloads the settled IC) and the geometric rest-placement (drop-to-
        # rest seat). Pure host C++ (the World device work goes through phi v2).
        scene/cook/settle.cpp
        scene/cook/placement.cpp
        # M7 T4 (THE CRUX): CookSceneToUnionTemplate — the .nks h1_cup_table ->
        # union BatchedSceneTemplate cook that REPLACES the 1026-line
        # h1_union_scene_factory authoring (reads the baked settled IC + grasp
        # config instead of running FK/placement/settle). Produces the
        # coresident::BatchedSceneTemplate BuildNkUnionModel / BatchedUnifiedWorld
        # consume (those + the struct stay alive to M9 per R1/R2/R3). Pure host
        # C++; the coresident dep is header-only product types (no kernels).
        scene/cook/union_cook.cpp
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
    )

    nuka_set_warnings(nuka_scene_cook)

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

    # -------------------------------------------------------------------------
    # nuka_viewer (M8.5 T3) -- the windowed realtime viewer: imgui_layer (the
    # custom-beautified UI panels) + camera_controller (orbit/pan/zoom) + the
    # `nuka_viewer` exe (viewer_main). Drives Simulation::FramePublish (step +
    # publish) and presents each frame via nuka_present, with the imgui_layer as
    # the present-pass overlay. GATED behind NK_BUILD_VULKAN_VALIDATION (D3): the
    # default build-cuda128 config never sees it (opt-in, like the recorder). It
    # links nuka_present (swapchain + xcb), nuka_imgui (Dear ImGui glue + theme),
    # nuka_app (the frame-loop spine), and the nk physics libs (for the cook +
    # nk::World fixture). HOST-ONLY / zero-CUDA-token.
    # -------------------------------------------------------------------------
    if(NK_BUILD_VULKAN_VALIDATION)
        add_library(nuka_viewer STATIC
            runtime/app/viewer/camera_controller.cpp
            runtime/app/viewer/imgui_layer.cpp
        )
        set_target_properties(nuka_viewer PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_include_directories(nuka_viewer
            PUBLIC
                ${CMAKE_SOURCE_DIR}/src
        )
        target_compile_definitions(nuka_viewer PUBLIC NK_BUILD_VULKAN_VALIDATION)
        # imgui_layer.cpp includes the vendored ImGui (SYSTEM via the `imgui`
        # target); camera_controller is pure host math (nuka_app/render types).
        target_link_libraries(nuka_viewer
            PUBLIC
                nuka_app
                nuka_present
                nuka_imgui
                imgui
        )
        nuka_set_warnings(nuka_viewer)

        add_executable(nuka_viewer_exe
            runtime/app/viewer/viewer_main.cpp
        )
        add_dependencies(nuka_viewer_exe nuka_render_shaders)
        set_target_properties(nuka_viewer_exe PROPERTIES OUTPUT_NAME nuka_viewer)
        target_include_directories(nuka_viewer_exe
            PRIVATE
                ${CMAKE_SOURCE_DIR}/src
        )
        target_compile_definitions(nuka_viewer_exe PRIVATE NK_BUILD_VULKAN_VALIDATION)
        target_link_libraries(nuka_viewer_exe
            PRIVATE
                nuka_viewer
                nuka_present
                nuka_imgui
                nuka_app
                nuka_scene_cook
                nuka_render
                nuka_nk
                nuka_phi2
                nuka_runtime
                nuka_import
                nuka_cooker
                nuka_articulation_gpu
                nuka_collision_gpu
                nuka_core
                nuka_phi
                imgui
                Vulkan::Vulkan
        )
        nuka_set_warnings(nuka_viewer_exe)
    endif()

    add_library(nuka_runtime_gpu STATIC
        core/diagnostics/invariants_gpu.cu
        # SAP broadphase: O(n^2) pair-slot oracle over a device AABB buffer. The
        # only remaining consumer is the LBVH-vs-SAP differential gate (the legacy
        # whole-world AABB-generation path was removed in the M0 cleanup).
        collision/gpu/broadphase.cu
        # v0.7 p09-A: XPBD soft-body distance world (GPU predict/solve/correct).
        # The first soft path; the soft/fluid particle worlds live in this same
        # library so their cross-system co-step overloads can link directly.
        # p09-B adds the bend (id 7) + volume (id 8) GPU solve sweeps in the same TU.
        runtime/soft/xpbd_world.cu
        # v0.7 p09-B: cook-time HOST topology builders (cloth stretch+bend,
        # tet edge+volume) -> XpbdConstraintSet the GPU solve iterates. Cook-time
        # CPU is allowed (same posture as the p06/p07 cookers); the per-step sim
        # stays GPU-only.
        runtime/soft/cloth_topology.cpp
        runtime/soft/tetmesh_topology.cpp
        # v0.7 p09-D: the XPBD soft-body COOKER (programmatic XpbdSoftBodySpec ->
        # XpbdConstraintSet). THIN cook-time CPU orchestration: maps the authored
        # material params (the USD MaterialX soft schema) + dispatches to the
        # topology builders above (and assembles the id9 cluster). Lives in this
        # library (not nuka_cooker) because it calls the builders here; nuka_cooker
        # is deliberately CUDA-free, so routing the other way would pull CUDA in.
        import/cooker/xpbd_cooker.cpp
        # v0.7 p10-B: the PBF fluid COOKER (programmatic FluidBoxSpec ->
        # runtime::fluid::PbfParticleSet). THIN cook-time CPU: samples a
        # cell-centered uniform lattice filling an AABB (optional inscribed-sphere
        # carve) + the uniform mass = rest_density*spacing^3. Lives in this library
        # (not nuka_cooker) because it includes pbf_world.hpp (a CUDA-language
        # header in this lib); nuka_cooker is deliberately CUDA-free.
        import/cooker/fluid_cooker.cpp
        # v0.7 p08-A: GPU upload of cooked sparse SDFs. Uploads scene::CookedSdfTable
        # to phi::Buffers and exposes a device array of SparseSdfDevice views + the
        # per-piece sdf-index map for the p08 Newton-summed-SDF contact kernel.
        runtime/sdf/sdf_device_world.cu
        # v0.7 p10-A: Position-Based Fluids (Macklin & Mueller 2013) -- the CORE
        # forward density-projection fluid. GPU predict/density-iterate/finalize;
        # FORWARD-ONLY + INTERNAL (no row class, no codegen, no adjoint). Reuses
        # the p05 particle uniform grid (nuka_collision_gpu) for the per-step
        # neighbor list; the cross-system particle co-step links against it, so it
        # lives in the same library. p10-B layers XSPH viscosity + surface tension
        # + the fluid cooker on top.
        runtime/fluid/pbf_world.cu
        # v0.7 p11 (K3): the cross-system particle-particle CONTACT co-step -- the
        # v0.7 infrastructure deliverable for cross-system coupling. Concatenates
        # the XPBD soft + PBF fluid particles into a union, builds the p05 cross-
        # system uniform grid (nuka_collision_gpu) over it, and applies the
        # mass-weighted symmetric non-penetration projection as a D1 Jacobi double-
        # buffer (NO float atomicAdd). Links both worlds, so it lives in this lib.
        runtime/coupling/particle_particle_contact.cu
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
            # v0.7 p10-A: the PBF substep reuses the p05 particle uniform grid.
            nuka_collision_gpu
    )

    # --- v0.7 p04: self-written LBVH rigid broadphase (Karras 2012) ----------
    # Selectable backend (SAP in nuka_runtime_gpu stays the default). Produces a
    # compact, deterministically-sorted candidate-pair list whose reduced SET
    # equals the SAP overlap set. First Thrust user in the tree (CUDA 12.8 bundles
    # Thrust; included directly, no extra find_package needed). D1-critical:
    # integer-only atomics, stable_sort_by_key (radix), sorted compact output.
    add_library(nuka_collision_gpu STATIC
        collision/broadphase_lbvh.cu
        collision/lbvh_refit.cu
        # v0.7 p05: particle uniform grid (spatial hash) + cross-system query
        # (particle AABB -> rigid LBVH). D1-deterministic broadphase primitives
        # feeding the soft/fluid/coupling phases. Same lib as the LBVH (which
        # the cross-system query traverses); integer-only atomics, sorted output.
        collision/particle_uniform_grid.cu
        collision/cross_system_query.cu
        # v0.7 p08-B: Newton-on-summed-SDF contact. find_sdf_contact_newton is a
        # header-inline NUKA_SDF_HD function (host tests call it directly); this
        # TU carries the device emission kernel + the host RowBuffers builders.
        collision/sdf_contact.cu
        # v0.7 p08-C (Channel 2/3): envelope-theorem depth+normal contact-geometry
        # adjoint of the Newton-summed-SDF contact. Host fp32; differentiates the
        # p07 trilinear sampler + pose at the FROZEN witness (valley-proof, no
        # step_curvature). The contact POINT is degenerate -> NOT differentiated.
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
        # v0.8 P2.4c: the BATCHED GRASP NARROWPHASE kernel. One thread per (env x
        # fingertip) wraps the HD-clean cvx::SphereHull (sphere=fingertip x cup hull)
        # -- the SAME math the host narrowphase dispatch calls -- moving the
        # P2.4b-dominant grasp narrowphase (90.9% at N=32) host->GPU. The host launcher
        # (LaunchGraspSphereHullNarrowphase) is the BuildContactManifolds drop-in
        # (StampSides + param merge applied post-download). ADDITIVE; no existing TU edited.
        collision/gpu/narrowphase_grasp.cu
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
            CUDA::cudart
    )

    # --- v0.7 p12: self-written CUDA ray tracer (camera -> LBVH -> closest-hit
    # depth, box leaves). NO OptiX / NO closed SDK. NAMED CONSUMER that wires the
    # retained LBVH (BuildLbvhForQuery) into the runtime renderer. Stackless
    # parent-pointer ray traversal, one thread per pixel, ZERO atomics -> D1
    # byte-exact. --fmad=false: the leaf slab test (rt/ray_box.cuh) is a shared
    # __host__ __device__ function whose t must match the CPU oracle bit-for-bit;
    # forbidding FMA contraction on the device side guarantees host==device
    # arithmetic (the oracle TU pairs this with -ffp-contract=off).
    # v0.7 p13: the sensor renderer (scene_render.cu) -- real per-primitive leaf
    # intersections (triangle/sphere/sparse-SDF) + Lambert+GGX shade + hard
    # shadows + full AOV framebuffer. Shares nuka_rt_gpu's --fmad=false so the
    # fp64-internal intersections stay bit-exact vs the CPU oracle. Links
    # nuka_cooker (no link dep here; the SDF cell arrays are passed in by the
    # caller via SparseSdfDevice) -- only the runtime sampler header is used.
    # v0.7 G1-A: the self-written TWO-LEVEL (TLAS/BLAS) tracer
    # (rt/two_level_render.cu). Nests the p12 stackless rt::TraverseRay two deep:
    # a per-frame TLAS over instance world-AABBs whose leaf transforms the ray to
    # each instance's local frame and runs an INNER traversal over that instance's
    # once-built BLAS. RIGID two-level core (p16 demo: H1 + cup + table + USD =
    # many rigid mesh instances). Shares --fmad=false so the fp64 leaf math +
    # fp32 QuatInvRotate stay bit-exact vs the host oracle (-ffp-contract=off).
    add_library(nuka_rt_gpu STATIC
        rt/bvh_ray_traversal.cu
        rt/scene_render.cu
        rt/two_level_render.cu
    )
    set_target_properties(nuka_rt_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_options(nuka_rt_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    )
    target_include_directories(nuka_rt_gpu
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rt_gpu
        PUBLIC
            nuka_collision
            nuka_collision_gpu
            nuka_phi
            CUDA::cudart
    )

    add_library(nuka_row_solver_gpu STATIC
        solver/gpu/row_scheduler.cu
        solver/gpu/row_solver.cu
    )
    set_target_properties(nuka_row_solver_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_include_directories(nuka_row_solver_gpu
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_row_solver_gpu
        PUBLIC
            nuka_constraint
            nuka_phi
    )

    target_compile_definitions(nuka_solver
        PUBLIC
            NUKA_HAS_CUDA_RUNTIME=1
    )

    target_link_libraries(nuka_solver
        PUBLIC
            nuka_phi
            nuka_row_solver_gpu
    )

    add_library(nuka_articulation_gpu STATIC
        runtime/articulation/featherstone_aba.cu
        runtime/articulation/articulation_drives.cu
        runtime/articulation/articulation_jacobian.cu
        runtime/articulation/articulation_contacts.cu
        runtime/gpu/batched_articulated_world.cu
        # p14a (v0.7): any-link contact-force readout (per-link wrench + per-slot
        # force). Pure readback of the solved lambda + stage-4/5 contact geometry;
        # depends only on nuka_core/nuka_phi + math, so compiling it INTO this lib
        # (the one that owns the batched world that calls it) avoids any new
        # inter-library dependency cycle.
        sensor/contact_wrench.cu
    )
    set_target_properties(nuka_articulation_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_include_directories(nuka_articulation_gpu
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_articulation_gpu
        PUBLIC
            nuka_articulation
            nuka_core
            nuka_phi
            CUDA::cudart
    )

    target_link_libraries(nuka_runtime
        PUBLIC
            nuka_articulation_gpu
    )

    # --- v0.5 p02-A: differentiable single-step spine (ABA 3-pass reverse) ----
    # Contact-free reverse-mode adjoint of drive -> ABA -> integrate. Internal /
    # test-facing C++ API (NOT the tape / C-ABI; that is p02-B). One thread per
    # articulation, atomic-free fixed-order tree accumulation (D1).
    # p02-B/C: the multi-step tape / checkpoint / recompute orchestrator /
    # backward runner ride on the p02-A single-step adjoint (step_backward).
    add_library(nuka_diffsim STATIC
        diffsim/step_backward.cu
        diffsim/step_backward_host.cpp
        diffsim/tape.cu
        diffsim/checkpoint.cu
        diffsim/recompute_orchestrator.cu
        diffsim/backward_runner.cu
        # v0.5 p03 R1: self-written deterministic CG sparse solver (IFT backend).
        # Warp-per-articulation batched-dense SPD CG; D1 bit-exact, no atomics.
        diffsim/sparse_solver_cg.cu
        # v0.7 p01: CG convergence diagnostics host summarizer (the opt-in
        # side-channel's CPU aggregation; the device side is a header consumed by
        # sparse_solver_cg.cu). D1: a fixed-order pass, no device work here.
        diffsim/solver/cg_diagnostics.cpp
        # v0.7 p02: self-written MINRES backend (symmetric INDEFINITE Krylov) +
        # ILU(0) preconditioner (incomplete LU, zero fill-in) + sparse triangular
        # solve. Warp-per-block on the SAME batched-dense seam as the CG backend;
        # D1 bit-exact (Lanczos/Givens are fixed-order per-block, no atomics).
        diffsim/solver/minres_backend.cu
        diffsim/solver/ilu0_preconditioner.cu
        diffsim/solver/triangular_solve.cu
        # v0.7 p03-A: self-written GMRES backend (general NON-SYMMETRIC Krylov,
        # restarted GMRES(m): Arnoldi + modified Gram-Schmidt + Givens rotations) +
        # the non-symmetric auto-routing detector. Warp-per-block on the SAME batched-
        # dense seam as CG/MINRES; D1 bit-exact (MGS/Givens are fixed-order per-block,
        # no atomics, the Krylov basis lives in shared mem). Jacobi precond; ILU0/AMG
        # apply hook reserved for p03-B.
        diffsim/solver/gmres_backend.cu
        # v0.5 p03 R2: contact KKT (Delassus) builder -- assembles the batched
        # per-articulation dense SPD A = J M^-1 J^T the CG backend solves.
        diffsim/kkt_builder.cu
        # v0.5 p03 R3a: IFT-at-convergence contact gradient runner -- backprops
        # through the converged contact solve in ONE Delassus solve (z = A^-1 J M^-1 g),
        # distributing grad_qdot_free = g - J^T z and grad_b_c = z. D1, no atomics.
        diffsim/ift_runner.cu
        # v0.7 p08-C (Channel 1, the §4.B acceptance): SDF-contact IFT d/dM, d/dJ.
        # The v0.5 ift_runner DEFERRED these channels; §4.B owns them for the
        # SDF-contact rows. System-level IFT of A lambda = r (A = J M^-1 J^T):
        # dlambda = A^-1 (dr - dA lambda). Host fp64 reference (the GPU runner
        # already owns the A^-1 apply); FD-validated by perturb-M/J-and-re-solve.
        diffsim/sdf_contact_ift.cpp
        # v0.5 p03 R3b: OSC (operational-space control) differentiable adjoint --
        # reverse-mode gain (Kp,Kd) + task-target gradients of the forward OSC drive
        # (tau = J^T A^-1 (Kp*e_x+Kd*edot_x)); reuses the forward's 3x3 LDL^T to
        # apply A^-1. D1, warp-per-articulation, no atomics. (R8/R9.)
        diffsim/osc_adjoint.cu
    )
    set_target_properties(nuka_diffsim PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_include_directories(nuka_diffsim
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_diffsim
        PUBLIC
            nuka_articulation_gpu
            nuka_articulation
            nuka_core
            nuka_phi
            CUDA::cudart
    )

    # --- v0.5 p04 N1: sim-to-real sensor noise (counter-based Philox) --------
    # Self-contained counter-based Philox4x32-10 RNG + in-place Gaussian/Poisson
    # noise kernels. D1 two-run bit-exact (a sample is a pure function of (seed,
    # element_idx, seq); no atomics, no thread-order reductions). Separate static
    # lib mirroring nuka_diffsim; linked into the `nuka` shared lib and the
    # tests/sensor gtests. Header-only philox.cuh lives alongside.
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
        c_abi/buffer.cpp
        c_abi/error.cpp
        c_abi/diffsim.cpp
        c_abi/noise.cpp
        # v0.8 A2: the BATCHED GRASP WORLD C-ABI shim over BatchedUnifiedWorld (the
        # RL-grasp substrate exposed to Python). Compiles the coresident sources +
        # the SHARED grasp scene factory (the SAME validated scene the 21
        # nuka_batched_unified_world_test gates exercise) + the host contact driver
        # the grasp narrowphase needs, directly into libnuka (g++-14). The binding
        # (python/src/nuka_ext.cpp, g++-10) binds the C surface only -- no C++ class
        # ABI crosses the toolchain boundary.
        c_abi/grasp_world.cpp
        # v0.8 G2 / M7: the H1 WHOLE-BODY UNION WORLD C-ABI shim. The union scene
        # is now COOKED from the authored examples/scenes/h1_cup_table.nks via
        # nuka_scene_cook (CookSceneToUnionTemplate) -- the 1026-line
        # h1_union_scene_factory it used to call is DELETED (M7 T6); the cook
        # reads the baked settled IC + grasp config and produces the SAME
        # BatchedSceneTemplate BatchedUnifiedWorld consumes.
        c_abi/union_world.cpp
        # M8 T5: the offscreen RECORDER C ABI (frame loop -> raster -> PPM P6 ->
        # ffmpeg mp4). ALWAYS compiled (so the symbols export for the python
        # binding), but its render-touching body is behind NK_BUILD_VULKAN_VALIDATION
        # (Decision D5): with the flag OFF this TU pulls NO Vulkan/render headers
        # and every entry returns NUKA_RESULT_NOT_SUPPORTED -> libnuka stays
        # buildable with zero Vulkan deps. With the flag ON the nuka_app + nuka_render
        # link (added below) bring the real Simulation + VulkanRasterRenderer in.
        c_abi/recorder.cpp
        # M9 T4: the GENERIC SCENE-AUTHORING C ABI (load/compose/find/set_local/
        # set_physics_material/settle/save over nuka::scene::SceneIR). Host-only
        # (zero CUDA tokens); nuka_scene_settle cooks Scene->CookToModel->nk::World
        # on the device's OWNED phi v2 backend (mirrors recorder.cpp) and runs
        # cook::Settle. NOT a special grasp/union type -- one entry, any format.
        c_abi/scene.cpp
        runtime/coresident/batched_unified_world.cpp
        runtime/coresident/unified_coresident_stepper.cpp
        runtime/coresident/grasp_scene_factory.cpp
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
            nuka_articulation_gpu
            nuka_diffsim
            nuka_sensor_noise
            nuka_phi
            # v0.8 A2: the unified collision/contact spine the BatchedUnifiedWorld
            # grasp path needs (the same delta the nuka_batched_unified_world_test
            # target adds over the previous libnuka link set).
            nuka_solver
            nuka_constraint
            nuka_collision_gpu
            # M7: the C-ABI union world cooks examples/scenes/h1_cup_table.nks
            # (CookSceneToUnionTemplate) instead of calling the deleted factory.
            # nuka_scene_cook transitively pulls nuka_nk + nuka_scene +
            # nuka_articulation; nks::Load lives in nuka_import (already linked).
            nuka_scene_cook
        PUBLIC
            CUDA::cudart
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(nuka PUBLIC stdc++)
    endif()

    # v0.8 A2: the grasp path compiles batched_unified_world.cpp, which includes the
    # GENERATED collidable registry (candidate_pair.hpp) -- ensure codegen runs first.
    add_dependencies(nuka nuka_codegen)

    # M8 T5 (Decision D5): the offscreen recorder. ONLY when NK_BUILD_VULKAN_VALIDATION
    # is ON does c_abi/recorder.cpp compile its real body + link the frame-loop
    # (nuka_app) and the offscreen raster renderer (nuka_render). With the flag OFF
    # the recorder TU is a pure stub (NUKA_RESULT_NOT_SUPPORTED, no Vulkan headers),
    # so libnuka builds with NO Vulkan dependency and the python binding still imports.
    if(NK_BUILD_VULKAN_VALIDATION)
        set_source_files_properties(c_abi/recorder.cpp PROPERTIES
            COMPILE_DEFINITIONS NK_BUILD_VULKAN_VALIDATION)
        target_link_libraries(nuka PRIVATE nuka_app nuka_render)
    endif()

    set_target_properties(nuka PROPERTIES
        OUTPUT_NAME nuka
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        PUBLIC_HEADER "${NUKA_PUBLIC_HEADERS}"
    )

    nuka_set_warnings(nuka)

    # v0.8 A2: the PROMOTED coresident physics sources (+ the host contact driver)
    # keep the warning posture of their authoring home (the test target, which does
    # NOT apply nuka_set_warnings) rather than getting the library's strict
    # -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror imposed by promotion.
    # These 170 KB of byte-exact, gate-validated sources were never written under
    # those flags; re-auditing them would risk the byte-identity the 21 gates pin.
    # `-w` is appended AFTER -Werror per source file, so GCC suppresses the warnings
    # before -Werror can elevate them. The NEW c_abi/grasp_world.cpp shim is
    # deliberately NOT exempt -- it stays strict to catch real bugs.
    set_source_files_properties(
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/batched_unified_world.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/unified_coresident_stepper.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/grasp_scene_factory.cpp
        # M7: h1_union_scene_factory.cpp DELETED (the .nks cook replaces it); the
        # NEW c_abi/union_world.cpp shim stays strict.
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

    add_executable(nuka_aba_cuda_probe
        ${CMAKE_SOURCE_DIR}/tools/reference/aba_cuda_probe.cpp
    )

    target_include_directories(nuka_aba_cuda_probe
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_aba_cuda_probe
        PRIVATE
            nuka_articulation_gpu
            nuka_import
            nuka_runtime
    )

    nuka_set_warnings(nuka_aba_cuda_probe)

    # M8.5 T5: the visual-mesh cook CLI. HOST-ONLY (import -> .nks/.nka Save),
    # routes visual-only geoms to .nka MESH chunks so robots render as real
    # triangle geometry. Used to (re)generate examples/scenes/go2.nks etc.
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
endif()
