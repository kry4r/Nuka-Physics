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
# nuka_scene – Scene IR and runtime cooking
# ---------------------------------------------------------------------------
add_library(nuka_scene STATIC
    scene/scene_ir.cpp
    scene/cooker.cpp
    scene/scene_graph.cpp
)
set_target_properties(nuka_scene PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(nuka_scene
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_scene
    PUBLIC
        nuka_math
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
    runtime/world_stepper.cpp
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
)

add_library(nuka_render STATIC
    render/render_scene.cpp
    render/vulkan_renderer.cpp
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

add_custom_target(nuka_render_shaders DEPENDS "${NUKA_DEBUG_DRAW_SPV}")
add_dependencies(nuka_render nuka_render_shaders)
target_compile_definitions(nuka_render
    PRIVATE
        NUKA_VULKAN_DEBUG_DRAW_SPV="${NUKA_DEBUG_DRAW_SPV}"
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
    solver/rigid_solver.cpp
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
# nuka_debug_draw – Debug draw command list (no GPU dependency)
# ---------------------------------------------------------------------------
add_library(nuka_debug_draw STATIC
    apps/debug_shell/debug_draw.cpp
    apps/debug_shell/debug_visualization.cpp
    apps/debug_shell/headless_debug_renderer.cpp
    apps/debug_shell/scene_demo.cpp
)

target_include_directories(nuka_debug_draw
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_debug_draw
    PUBLIC
        nuka_collision
        nuka_constraint
        nuka_import
        nuka_math
        nuka_render
        nuka_runtime
        nuka_scene
        nuka_scene_pipeline
)

nuka_set_warnings(nuka_debug_draw)

add_executable(nuka_scene_demo
    apps/debug_shell/main.cpp
)

target_include_directories(nuka_scene_demo
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_scene_demo
    PRIVATE
        nuka_debug_draw
)

nuka_set_warnings(nuka_scene_demo)

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

    add_library(nuka_runtime_gpu STATIC
        core/diagnostics/invariants_gpu.cu
        collision/gpu/broadphase.cu
        constraint/gpu/contact_generation.cu
        runtime/gpu/batched_device_world.cu
        runtime/gpu/cuda_particle_world.cu
        runtime/gpu/device_world.cu
        runtime/gpu/cuda_world_stepper.cu
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
    add_library(nuka_diffsim STATIC
        diffsim/step_backward.cu
        diffsim/step_backward_host.cpp
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

    add_library(nuka SHARED
        c_abi/device.cpp
        c_abi/world.cpp
        c_abi/buffer.cpp
        c_abi/error.cpp
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
            nuka_phi
        PUBLIC
            CUDA::cudart
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(nuka PUBLIC stdc++)
    endif()

    set_target_properties(nuka PROPERTIES
        OUTPUT_NAME nuka
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        PUBLIC_HEADER "${NUKA_PUBLIC_HEADERS}"
    )

    nuka_set_warnings(nuka)

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

    add_library(nuka_sensor_gpu STATIC
        sensor/gpu/cuda_sensors.cu
    )

    target_include_directories(nuka_sensor_gpu
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_sensor_gpu
        PUBLIC
            nuka_runtime_gpu
            nuka_sensor
    )

    target_link_libraries(nuka_debug_draw
        PUBLIC
            nuka_sensor_gpu
            nuka_runtime_gpu
    )

    target_compile_definitions(nuka_debug_draw
        PUBLIC
            NUKA_HAS_CUDA_RUNTIME=1
    )

    add_executable(nuka_cuda_particle_demo
        apps/cuda_particle_demo/main.cpp
    )

    target_include_directories(nuka_cuda_particle_demo
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_cuda_particle_demo
        PRIVATE
            nuka_runtime_gpu
    )

    nuka_set_warnings(nuka_cuda_particle_demo)

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

    add_executable(nuka_go2_stand_demo
        apps/nuka_go2_stand_demo.cpp
    )

    target_include_directories(nuka_go2_stand_demo
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/include
            ${CMAKE_SOURCE_DIR}/tests
    )

    target_link_libraries(nuka_go2_stand_demo
        PRIVATE
            nuka
            CUDA::cudart
    )

    nuka_set_warnings(nuka_go2_stand_demo)
endif()
