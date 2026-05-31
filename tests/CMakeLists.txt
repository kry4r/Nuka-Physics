# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
include(GoogleTest)

# --- Smoke test -----------------------------------------------------------
add_executable(nuka_smoke_test
    smoke/test_build_smoke.cpp
)

target_link_libraries(nuka_smoke_test
    PRIVATE
        nuka_core
        GTest::gtest_main
)

gtest_discover_tests(nuka_smoke_test)

# --- Math tests ------------------------------------------------------------
add_executable(nuka_math_test
    math/test_spatial_math.cpp
)

target_link_libraries(nuka_math_test
    PRIVATE
        nuka_core
        GTest::gtest_main
)

gtest_discover_tests(nuka_math_test)

# --- v0.1 e2e foundation tests --------------------------------------------
add_executable(nuka_v01_foundation_e2e_test
    e2e/test_v01_foundation_pipeline.cpp
)

target_compile_definitions(nuka_v01_foundation_e2e_test
    PRIVATE
        NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

if(TARGET nuka_aba_cuda_probe)
    add_dependencies(nuka_v01_foundation_e2e_test nuka_aba_cuda_probe)
    target_compile_definitions(nuka_v01_foundation_e2e_test
        PRIVATE
            NUKA_ABA_CUDA_PROBE="$<TARGET_FILE:nuka_aba_cuda_probe>"
            NUKA_ORACLE_PYTHON="${CMAKE_SOURCE_DIR}/.nuka-oracle-venv/bin/python"
    )
endif()

target_link_libraries(nuka_v01_foundation_e2e_test
    PRIVATE
        nuka_runtime
        nuka_import
        $<$<TARGET_EXISTS:nuka_articulation_gpu>:nuka_articulation_gpu>
        $<$<TARGET_EXISTS:nuka_phi>:nuka_phi>
        GTest::gtest_main
)

gtest_discover_tests(nuka_v01_foundation_e2e_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTIES RUN_SERIAL TRUE
)

if(TARGET nuka)
    add_executable(nuka_c_abi_create_step_destroy_test
        c_abi/test_create_step_destroy.c
    )

    target_include_directories(nuka_c_abi_create_step_destroy_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/include
    )

    target_link_libraries(nuka_c_abi_create_step_destroy_test
        PRIVATE
            nuka
    )

    add_test(NAME CAbi.CreateStepDestroy
             COMMAND nuka_c_abi_create_step_destroy_test)
    set_tests_properties(CAbi.CreateStepDestroy
        PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    if(NUKA_HAS_EXPECTED_WRAPPER)
        add_executable(nuka_cpp_wrapper_raii_test
            c_abi/test_cpp_wrapper_raii.cpp
        )

        target_compile_definitions(nuka_cpp_wrapper_raii_test
            PRIVATE
                NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )

        target_include_directories(nuka_cpp_wrapper_raii_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/src/include
        )

        target_link_libraries(nuka_cpp_wrapper_raii_test
            PRIVATE
                nuka
                CUDA::cudart
                GTest::gtest_main
        )

        gtest_discover_tests(nuka_cpp_wrapper_raii_test
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

        # --- Multi-env C ABI surface (p01-F T7) ---------------------------
        add_executable(nuka_multi_env_world_test
            c_abi/test_multi_env_world.cpp
        )

        target_compile_definitions(nuka_multi_env_world_test
            PRIVATE
                NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )

        target_include_directories(nuka_multi_env_world_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/src/include
        )

        target_link_libraries(nuka_multi_env_world_test
            PRIVATE
                nuka
                CUDA::cudart
                GTest::gtest_main
        )

        gtest_discover_tests(nuka_multi_env_world_test
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

        # --- Per-env drive-target write + q/qd/base read C ABI (sim-val #33) --
        add_executable(nuka_drive_target_io_test
            c_abi/test_drive_target_io.cpp
        )

        target_compile_definitions(nuka_drive_target_io_test
            PRIVATE
                NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )

        target_include_directories(nuka_drive_target_io_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/src/include
        )

        target_link_libraries(nuka_drive_target_io_test
            PRIVATE
                nuka
                CUDA::cudart
                GTest::gtest_main
        )

        gtest_discover_tests(nuka_drive_target_io_test
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    else()
        message(STATUS "Skipping nuka.hpp RAII test: nuka::expected wrapper failed to compile")
    endif()
endif()

# --- Scene tests -----------------------------------------------------------
add_executable(nuka_scene_test
    scene/test_scene_ir.cpp
    scene/test_scene_cooker.cpp
    scene/test_scene_pipeline.cpp
)

target_link_libraries(nuka_scene_test
    PRIVATE
        nuka_scene_pipeline
        nuka_import
        GTest::gtest_main
)

gtest_discover_tests(nuka_scene_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# --- Runtime tests ---------------------------------------------------------
add_executable(nuka_runtime_test
    runtime/test_world_build.cpp
    runtime/test_rigid_integration.cpp
    runtime/test_world_stepper.cpp
)

target_link_libraries(nuka_runtime_test
    PRIVATE
        nuka_runtime
        GTest::gtest_main
)

gtest_discover_tests(nuka_runtime_test)

# --- Import tests ----------------------------------------------------------
add_executable(nuka_import_test
    import/test_mjcf_importer.cpp
    import/test_urdf_importer.cpp
    import/test_usd_importer.cpp
)

target_link_libraries(nuka_import_test
    PRIVATE
        nuka_import
        GTest::gtest_main
)

gtest_discover_tests(nuka_import_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# --- Collision tests -------------------------------------------------------
add_executable(nuka_collision_test
    collision/test_raycast.cpp
    collision/test_broadphase.cpp
)

target_link_libraries(nuka_collision_test
    PRIVATE
        nuka_collision
        GTest::gtest_main
)

gtest_discover_tests(nuka_collision_test)

# --- Constraint tests ------------------------------------------------------
add_executable(nuka_constraint_test
    constraint/test_contact_manifold.cpp
    constraint/test_csr_row_layout.cpp
)

target_link_libraries(nuka_constraint_test
    PRIVATE
        nuka_constraint
        GTest::gtest_main
)

gtest_discover_tests(nuka_constraint_test)

# --- Solver tests ----------------------------------------------------------
add_executable(nuka_solver_test
    solver/test_contact_resting.cpp
    solver/test_contact_material.cpp
    solver/test_joint_limit.cpp
    solver/test_joint_projection.cpp
    solver/test_row_scheduler_bridge.cpp
)

target_link_libraries(nuka_solver_test
    PRIVATE
        nuka_solver
        nuka_articulation
        GTest::gtest_main
)

gtest_discover_tests(nuka_solver_test)

# --- Articulation tests ----------------------------------------------------
add_executable(nuka_articulation_test
    runtime/test_revolute_joint.cpp
    runtime/test_joint_drive.cpp
)

target_link_libraries(nuka_articulation_test
    PRIVATE
        nuka_articulation
        GTest::gtest_main
)

gtest_discover_tests(nuka_articulation_test)

# --- Sensor tests ----------------------------------------------------------
add_executable(nuka_sensor_test
    sensor/test_imu_sensor.cpp
    sensor/test_lidar_sensor.cpp
)

target_link_libraries(nuka_sensor_test
    PRIVATE
        nuka_sensor
        GTest::gtest_main
)

gtest_discover_tests(nuka_sensor_test)

# --- Debug draw tests ------------------------------------------------------
add_executable(nuka_debug_draw_test
    apps/test_debug_draw_list.cpp
)

target_link_libraries(nuka_debug_draw_test
    PRIVATE
        nuka_debug_draw
        nuka_scene_pipeline
        nuka_import
        nuka_constraint
        GTest::gtest_main
)

gtest_discover_tests(nuka_debug_draw_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(nuka_scene_demo_test
    apps/test_scene_demo.cpp
)

target_link_libraries(nuka_scene_demo_test
    PRIVATE
        nuka_debug_draw
        nuka_scene_pipeline
        nuka_import
        nuka_constraint
        GTest::gtest_main
)

if(NK_BUILD_VULKAN_VALIDATION)
    target_compile_definitions(nuka_scene_demo_test
        PRIVATE
            NUKA_ENABLE_VULKAN_VALIDATION_TESTS=1
    )
endif()

gtest_discover_tests(nuka_scene_demo_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# --- Batch context tests ---------------------------------------------------
add_executable(nuka_batch_test
    runtime/test_batch_context.cpp
)

target_link_libraries(nuka_batch_test
    PRIVATE
        nuka_runtime
        GTest::gtest_main
)

gtest_discover_tests(nuka_batch_test)

# --- Regression tests ------------------------------------------------------
add_executable(nuka_oracle_test
    oracle/featherstone_oracle_harness.cpp
)

target_compile_definitions(nuka_oracle_test
    PRIVATE
        NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

target_link_libraries(nuka_oracle_test
    PRIVATE
        nuka_runtime
        nuka_import
        $<$<TARGET_EXISTS:nuka_articulation_gpu>:nuka_articulation_gpu>
        $<$<TARGET_EXISTS:nuka_phi>:nuka_phi>
        GTest::gtest_main
)

target_include_directories(nuka_oracle_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/tests
)

gtest_discover_tests(nuka_oracle_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(nuka_regression_test
    regression/test_free_fall.cpp
    regression/test_go2_stand_5s.cpp
    regression/test_rest_stack.cpp
    regression/test_two_link_arm.cpp
)

target_compile_definitions(nuka_regression_test
    PRIVATE
        NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

target_link_libraries(nuka_regression_test
    PRIVATE
        nuka_solver
        nuka_articulation
        nuka_collision
        nuka_runtime
        $<$<TARGET_EXISTS:nuka>:nuka>
        GTest::gtest_main
)

target_include_directories(nuka_regression_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/src/include
)

gtest_discover_tests(nuka_regression_test)

# --- Performance tests -----------------------------------------------------
add_executable(nuka_perf_test
    perf/test_step_timing.cpp
)

target_link_libraries(nuka_perf_test
    PRIVATE
        nuka_runtime
        nuka_debug_draw
        nuka_render
        nuka_scene_pipeline
        nuka_import
        nuka_solver
        GTest::gtest_main
)

gtest_discover_tests(nuka_perf_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

if(NK_BUILD_DEMO_TIMING_VALIDATION)
    add_executable(nuka_demo_timing_test
        perf/test_render_demo_timing.cpp
    )

    target_link_libraries(nuka_demo_timing_test
        PRIVATE
            nuka_runtime
            nuka_debug_draw
            nuka_render
            nuka_scene_pipeline
            nuka_import
            nuka_solver
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_demo_timing_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

if(NK_BUILD_VULKAN_VALIDATION)
    add_executable(nuka_vulkan_perf_test
        perf/test_vulkan_scene_demo_timing.cpp
        perf/test_batched_vulkan_scene_demo_timing.cpp
        perf/test_vulkan_renderscene_timing.cpp
    )

    target_link_libraries(nuka_vulkan_perf_test
        PRIVATE
            nuka_runtime
            nuka_debug_draw
            nuka_render
            nuka_scene_pipeline
            nuka_import
            nuka_solver
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_vulkan_perf_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# --- PHI tests (requires CUDA) --------------------------------------------
if(CMAKE_CUDA_COMPILER)
    add_executable(nuka_codegen_test
        codegen/test_codegen_roundtrip.cpp
    )

    add_dependencies(nuka_codegen_test nuka_codegen)

    target_compile_definitions(nuka_codegen_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            NUKA_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
    )

    target_link_libraries(nuka_codegen_test
        PRIVATE
            nuka_codegen_generated
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_codegen_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    add_executable(nuka_phi_test
        phi/test_capabilities.cpp
        phi/test_device_context_per_handle.cpp
        phi/test_platform_contract.cpp
        phi/test_stream_view_external.cpp
    )

    set_source_files_properties(
        phi/test_stream_view_external.cpp
        PROPERTIES LANGUAGE CUDA
    )

    target_link_libraries(nuka_phi_test
        PRIVATE
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_phi_test)

    add_executable(nuka_cuda_runtime_test
        runtime/test_cuda_batched_world.cpp
        runtime/test_cuda_device_world.cpp
        runtime/test_cuda_particle_world.cpp
        runtime/test_cuda_world_stepper.cpp
        runtime/test_cuda_contacts.cpp
        runtime/test_cuda_solver.cpp
    )

    target_link_libraries(nuka_cuda_runtime_test
        PRIVATE
            nuka_runtime_gpu
            nuka_runtime
            nuka_articulation_gpu
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_cuda_runtime_test)

    # --- Batched N-articulation replication (p01-F T1) ---------------------
    add_executable(nuka_batched_articulation_replication_test
        runtime/test_batched_articulation_replication.cpp
    )

    target_compile_definitions(nuka_batched_articulation_replication_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_batched_articulation_replication_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_batched_articulation_replication_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    add_executable(nuka_contact_chain_jacobian_test
        runtime/test_contact_chain_jacobian.cpp
    )

    target_compile_definitions(nuka_contact_chain_jacobian_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_contact_chain_jacobian_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_contact_chain_jacobian_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    add_executable(nuka_articulation_contacts_test
        runtime/test_articulation_contacts.cpp
    )

    target_compile_definitions(nuka_articulation_contacts_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_articulation_contacts_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_articulation_contacts_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- Joint-space inertia M (CRBA) + contact effective mass (p01-F T4) ---
    add_executable(nuka_articulation_inertia_m_test
        runtime/test_articulation_inertia_m.cpp
    )

    target_compile_definitions(nuka_articulation_inertia_m_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_articulation_inertia_m_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_articulation_inertia_m_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- Articulated contact rows + block-PGS solve (p01-F T5) --------------
    add_executable(nuka_articulation_contact_rows_test
        runtime/test_articulation_contact_rows.cpp
    )

    target_compile_definitions(nuka_articulation_contact_rows_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_articulation_contact_rows_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_articulation_contact_rows_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- Floating-base (free 6-DOF root) ABA (p01-F T8a) --------------------
    add_executable(nuka_floating_base_aba_test
        runtime/test_floating_base_aba.cpp
    )

    target_compile_definitions(nuka_floating_base_aba_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_floating_base_aba_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_floating_base_aba_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- Floating-base contact coupling (p01-F T8b) ------------------------
    add_executable(nuka_floating_base_contact_test
        runtime/test_floating_base_contact.cpp
    )

    target_compile_definitions(nuka_floating_base_contact_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_floating_base_contact_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_floating_base_contact_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- Go2 PD standing/tracking validation (sim-val.PD) ------------------
    add_executable(nuka_go2_pd_standing_test
        runtime/test_go2_pd_standing.cpp
    )

    target_compile_definitions(nuka_go2_pd_standing_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_go2_pd_standing_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_go2_pd_standing_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- p03 per-env RESET primitive (RL autoreset) ------------------------
    add_executable(nuka_batched_reset_test
        runtime/test_batched_reset.cpp
    )

    target_compile_definitions(nuka_batched_reset_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_batched_reset_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_batched_reset_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- p03 episode-boundary fix: authoritative un-lagged base-pose view ---
    add_executable(nuka_base_pose_view_test
        runtime/test_base_pose_view.cpp
    )

    target_compile_definitions(nuka_base_pose_view_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_base_pose_view_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_base_pose_view_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- Batched articulated-with-contacts step path (p01-F T6) -------------
    add_executable(nuka_batched_articulated_world_test
        runtime/test_batched_articulated_world.cpp
    )

    target_compile_definitions(nuka_batched_articulated_world_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_batched_articulated_world_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_batched_articulated_world_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- CUDA-graph step capture/replay bit-identical gate (p01-W3) ---------
    add_executable(nuka_batched_articulated_graph_test
        runtime/test_batched_articulated_graph.cpp
    )

    target_compile_definitions(nuka_batched_articulated_graph_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_batched_articulated_graph_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_batched_articulated_graph_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- Fine-grained (detail) timing breakdown harness (v0.3 p01 Wave-2) ----
    add_executable(nuka_batched_articulated_world_perf_test
        runtime/test_batched_articulated_world_perf_detail.cpp
    )

    target_compile_definitions(nuka_batched_articulated_world_perf_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_batched_articulated_world_perf_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_batched_articulated_world_perf_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- Parameterized perf-harness CLI (v0.3 p01 Wave-2 #4) ----------------
    # NOT a gtest: a standalone CLI that drives the real batched articulated
    # stepper over a chosen env count / step count and prints the engine
    # PerfRecorder JSON (schema_version + per_tag + gpu/n_envs/steps) to stdout.
    # This is the `--harness` binary tools/perf/run_perf_baseline.py invokes.
    add_executable(nuka_perf_harness
        ${CMAKE_SOURCE_DIR}/tools/perf/perf_harness.cpp
    )

    target_link_libraries(nuka_perf_harness
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
    )

    add_executable(nuka_cuda_sensor_test
        sensor/test_cuda_sensors.cpp
    )

    target_link_libraries(nuka_cuda_sensor_test
        PRIVATE
            nuka_sensor_gpu
            nuka_sensor
            nuka_runtime_gpu
            nuka_runtime
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_cuda_sensor_test)

add_executable(nuka_cuda_perf_test
    perf/test_cuda_batch_timing.cpp
    perf/test_cuda_batch_contact_timing.cpp
    perf/test_cuda_batch_joint_drive_timing.cpp
    perf/test_cuda_batch_sensor_timing.cpp
    perf/test_cuda_contact_timing.cpp
    perf/test_cuda_particle_coupling_timing.cpp
    perf/test_cuda_sensor_timing.cpp
    perf/test_cuda_solver_timing.cpp
    perf/test_cuda_step_timing.cpp
)

    target_link_libraries(nuka_cuda_perf_test
        PRIVATE
            nuka_debug_draw
            nuka_sensor_gpu
            nuka_sensor
            nuka_runtime_gpu
            nuka_runtime
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_cuda_perf_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- Perf timing infrastructure contract test (Task 3.1.1) -------------
    add_executable(nuka_perf_timing_test
        perf/test_perf_timing.cu
    )

    target_link_libraries(nuka_perf_timing_test
        PRIVATE
            nuka_core
            CUDA::cudart
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_perf_timing_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    if(NK_BUILD_DEMO_TIMING_VALIDATION)
        add_executable(nuka_cuda_scene_demo_perf_test
            perf/test_cuda_scene_demo_timing.cpp
        )

        target_link_libraries(nuka_cuda_scene_demo_perf_test
            PRIVATE
                nuka_debug_draw
                nuka_sensor_gpu
                nuka_sensor
                nuka_runtime_gpu
                nuka_runtime
                GTest::gtest_main
        )

        gtest_discover_tests(nuka_cuda_scene_demo_perf_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()
endif()

# --- Vulkan renderer tests -------------------------------------------------
if(NK_BUILD_VULKAN_VALIDATION)
    add_executable(nuka_render_test
        render/test_vulkan_backend.cpp
    )

    target_link_libraries(nuka_render_test
        PRIVATE
            nuka_render
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_render_test)
endif()
