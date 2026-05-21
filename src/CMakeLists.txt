# ---------------------------------------------------------------------------
# nuka_core – foundational utilities (version, logging, …)
# ---------------------------------------------------------------------------
add_library(nuka_core STATIC
    core/version.cpp
)

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
    runtime/world_builder.cpp
    runtime/physics_world.cpp
    runtime/world_stepper.cpp
    runtime/rigid/integrator.cpp
    runtime/batch_scheduler.cpp
)

target_include_directories(nuka_runtime
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

add_library(nuka_render STATIC
    render/render_scene.cpp
    render/vulkan_renderer.cpp
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
)

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
    solver/rigid_solver.cpp
)

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
    runtime/articulation/joint_constraints.cpp
)

target_include_directories(nuka_articulation
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_articulation
    PUBLIC
        nuka_constraint
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
    add_library(nuka_phi STATIC
        phi/backend_cuda/device.cu
        phi/backend_cuda/buffer.cu
    )

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
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /WX>
    )

    add_library(nuka_runtime_gpu STATIC
        collision/gpu/broadphase.cu
        constraint/gpu/contact_generation.cu
        runtime/gpu/device_world.cu
        runtime/gpu/cuda_world_stepper.cu
        solver/gpu/cuda_constraint_solver.cu
    )

    target_include_directories(nuka_runtime_gpu
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_runtime_gpu
        PUBLIC
            nuka_runtime
            nuka_phi
    )
endif()
