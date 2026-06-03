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

    # --- v0.5 p02-B/C: differentiable rollout tape C ABI (runtime coverage) ---
    add_executable(nuka_diffsim_tape_cabi_test
        c_abi/test_diffsim_tape.cpp
    )

    target_compile_definitions(nuka_diffsim_tape_cabi_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_diffsim_tape_cabi_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/include
    )

    target_link_libraries(nuka_diffsim_tape_cabi_test
        PRIVATE
            nuka
            CUDA::cudart
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_diffsim_tape_cabi_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.5 p04 N1: C-ABI sensor-noise surface (NONE no-op + apply + D1) ----
    add_executable(nuka_sensor_noise_cabi_test
        c_abi/test_sensor_noise.cpp
    )

    target_compile_definitions(nuka_sensor_noise_cabi_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_sensor_noise_cabi_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/include
    )

    target_link_libraries(nuka_sensor_noise_cabi_test
        PRIVATE
            nuka
            CUDA::cudart
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_sensor_noise_cabi_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.5 p04 N2: per-episode domain randomization (sampling + apply + D1) -
    # Reads c_abi/internal.hpp (WorldRecord) for mass/gravity readback + calls
    # SampleEpisodeRandomization directly (nuka_sensor_noise), so it needs the
    # src/ include + the runtime libs (same pattern as the mass-setter FD test).
    if(CMAKE_CUDA_COMPILER)
        add_executable(nuka_n2_domain_randomization_test
            sensor/test_n2_domain_randomization.cpp
        )

        target_compile_definitions(nuka_n2_domain_randomization_test
            PRIVATE
                NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )

        target_include_directories(nuka_n2_domain_randomization_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/src/include
                ${CMAKE_SOURCE_DIR}/src
        )

        target_link_libraries(nuka_n2_domain_randomization_test
            PRIVATE
                nuka
                nuka_sensor_noise
                nuka_runtime
                nuka_articulation_gpu
                nuka_core
                nuka_phi
                CUDA::cudart
                GTest::gtest_main
        )

        gtest_discover_tests(nuka_n2_domain_randomization_test
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()

    # --- v0.5 p04-A0: runtime link-mass setter + C-ABI mass-gradient FD proof --
    # Operations under test are pure C-ABI (set_link_mass / step_with_tape /
    # tape_backward); the internal libs are linked only for setup + reading the
    # tape-evolved device state (the same readback pattern every diffsim test
    # uses). Includes src/ for the c_abi/internal.hpp WorldRecord bridge.
    if(CMAKE_CUDA_COMPILER)
        add_executable(nuka_diffsim_mass_setter_fd_test
            diffsim/test_mass_setter_fd.cpp
        )

        target_compile_definitions(nuka_diffsim_mass_setter_fd_test
            PRIVATE
                NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )

        target_include_directories(nuka_diffsim_mass_setter_fd_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/src/include
                ${CMAKE_SOURCE_DIR}/src
        )

        target_link_libraries(nuka_diffsim_mass_setter_fd_test
            PRIVATE
                nuka
                nuka_diffsim
                nuka_runtime
                nuka_articulation_gpu
                nuka_core
                nuka_phi
                CUDA::cudart
                GTest::gtest_main
        )

        gtest_discover_tests(nuka_diffsim_mass_setter_fd_test
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
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
    import/test_decompose_attribute.cpp
)

target_link_libraries(nuka_import_test
    PRIVATE
        nuka_import
        GTest::gtest_main
)

gtest_discover_tests(nuka_import_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# --- V-HACD convex decomposition tests (v0.7 p06) --------------------------
# Cook-time decomposition: DecomposeMesh / cache (nuka_cooker) + the cooker
# integration (CookScene emits ConvexHull pieces, nuka_scene). Includes
# tests/import/vhacd_test_meshes.hpp via the source-root include dir.
add_executable(nuka_vhacd_test
    import/test_vhacd_simple_cube.cpp
    import/test_vhacd_robot_finger.cpp
    import/test_vhacd_determinism.cpp
    import/test_vhacd_cooker_integration.cpp
)

target_include_directories(nuka_vhacd_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}
)

target_link_libraries(nuka_vhacd_test
    PRIVATE
        nuka_cooker
        nuka_scene
        GTest::gtest_main
)

gtest_discover_tests(nuka_vhacd_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# --- Sparse narrow-band SDF cooker tests (v0.7 p07) ------------------------
# Cook-time SDF generation (nuka_cooker) + dedup through CookScene (nuka_scene).
# Pure host/CPU (no GPU needed); samples the cooked SDF on CPU through the same
# runtime/sdf/sparse_sdf_query.cuh sampler p08 ships on the GPU.
add_executable(nuka_sparse_sdf_test
    import/test_sparse_sdf_known_shapes.cpp
    import/test_sparse_sdf_thin_shell.cpp
    import/test_sparse_sdf_memory.cpp
    import/test_sparse_sdf_determinism.cpp
)

target_include_directories(nuka_sparse_sdf_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_sparse_sdf_test
    PRIVATE
        nuka_cooker
        nuka_scene
        GTest::gtest_main
)

gtest_discover_tests(nuka_sparse_sdf_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

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

    # --- V3 FD adjoint validation: all four base row classes (v0.5 p01) ------
    # maximal_drive (dense), maximal_joint (dense), maximal_contact +
    # featherstone_contact (stop_grad_on_event). Each validates the GENERATED
    # primal/adjoint pair against a central-difference numerical Jacobian.
    add_executable(nuka_adjoint_fd_test
        codegen/test_adjoint_fd_maximal_drive.cpp
        codegen/test_adjoint_fd_maximal_joint.cpp
        codegen/test_adjoint_fd_maximal_contact.cpp
        codegen/test_adjoint_fd_featherstone_contact.cpp
        codegen/test_adjoint_fd_xpbd_distance.cpp
        # v0.7 p09-B: XPBD bend (id 7) + volume (id 8) adjoint FD validation.
        codegen/test_adjoint_fd_xpbd_bend.cpp
        codegen/test_adjoint_fd_xpbd_volume.cpp
        # v0.7 p09-C: XPBD shape-match (id 9) goal-projection adjoint FD validation.
        codegen/test_adjoint_fd_xpbd_shape_match.cpp
        # v0.7 p11 (K3): particle-particle contact (id 10) contact-multiplier
        # adjoint FD validation (active penetrating contacts, unilateral box).
        codegen/test_adjoint_fd_particle_particle_contact.cpp
    )

    add_dependencies(nuka_adjoint_fd_test nuka_codegen)

    target_link_libraries(nuka_adjoint_fd_test
        PRIVATE
            nuka_codegen_v3_validation
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_adjoint_fd_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

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
        # v0.7 p09-A: XPBD distance analytic oscillation oracle + D1 two-run.
        runtime/test_xpbd_distance_oracle.cpp
        # v0.7 p09-B: XPBD bend (cloth) + volume (tet) sim invariants + grad-C FD.
        runtime/test_xpbd_bend_cloth.cpp
        runtime/test_xpbd_volume_tet.cpp
        # v0.7 p09-C: XPBD shape-match (cluster) sim + polar dR/dA host-FD.
        runtime/test_xpbd_shape_match.cpp
        # v0.7 p11 (K3): cross-system particle-particle contact co-step oracles
        # (separation, mass-weighted momentum, soft<->fluid, segment-reduction, D1,
        # inert byte-identity).
        runtime/test_particle_particle_contact.cpp
    )

    target_link_libraries(nuka_cuda_runtime_test
        PRIVATE
            nuka_runtime_gpu
            nuka_runtime
            nuka_articulation_gpu
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_cuda_runtime_test)

    # --- v0.7 p09-D: XPBD soft-body COOKER tests (exit-crit 3) -------------
    # PURE host/CPU: cooks programmatic XpbdSoftBodySpecs and asserts the
    # returned XpbdConstraintSet (counts + the two param->engine mappings +
    # two-cook determinism). No device upload / step, so it needs no GPU -- but
    # it links nuka_runtime_gpu because the cooker + topology builders live
    # there (the lib is CUDA-language, hence inside the CMAKE_CUDA_COMPILER
    # guard). nuka:soft:* schema + mapping documented in
    # docs/architecture/xpbd-soft-schema.md.
    add_executable(nuka_xpbd_cooker_test
        import/test_xpbd_cooker.cpp
    )

    target_include_directories(nuka_xpbd_cooker_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_xpbd_cooker_test
        PRIVATE
            nuka_runtime_gpu
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_xpbd_cooker_test)

    # --- v0.7 p10-B: PBF fluid COOKER (programmatic FluidBoxSpec -> particles) --
    # PURE host/CPU: cooks a box volume + asserts the hand-computed lattice count,
    # the uniform mass = rest_density*spacing^3, cell-centered placement, two-cook
    # byte-identical determinism, degenerate-spec empties, and the optional
    # inscribed-sphere carve. Links nuka_runtime_gpu because the fluid cooker +
    # PbfParticleSet live there (the lib is CUDA-language, hence inside the
    # CMAKE_CUDA_COMPILER guard) -- but the test itself runs no device code.
    add_executable(nuka_fluid_cooker_test
        import/test_fluid_cooker.cpp
    )

    target_include_directories(nuka_fluid_cooker_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_fluid_cooker_test
        PRIVATE
            nuka_runtime_gpu
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_fluid_cooker_test)

    # --- v0.7 p10-A: PBF fluid core (density iteration + volume + D1) -------
    # Analytic/physical invariants for the forward density-projection fluid
    # (Macklin & Mueller 2013): rest-density block, over-compressed relaxation,
    # single column -> stable pool, density-drift volume conservation, and D1
    # two-run byte-exact. NOT a Flex golden (Flex oracle deferred to p15). rho0 is
    # calibrated numerically from the engine's own Poly6 kernel. Links
    # nuka_runtime_gpu (PbfWorld lives there; reuses the p05 particle grid).
    # p10-B adds the polish-layer gates: XSPH viscosity (variance reduction +
    # off-gate skip + D1), surface-tension cohesion (blob contraction + off-gate +
    # D1 + stability), and the PBF V2 particle-count invariant (count conserved
    # with viscosity + cohesion ON; the central V2 sampler is rigid-only, so this
    # is the focused fluid count-conservation test -- see the file header).
    add_executable(nuka_pbf_test
        runtime/test_pbf_density_iteration.cpp
        runtime/test_pbf_volume_conservation.cpp
        runtime/test_pbf_viscosity.cpp
        runtime/test_pbf_surface_tension.cpp
        runtime/test_pbf_v2_invariant.cpp
    )

    target_link_libraries(nuka_pbf_test
        PRIVATE
            nuka_runtime_gpu
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_pbf_test)

    # --- v0.7 p08-A: GPU upload of cooked sparse SDFs ----------------------
    # Cooks known meshes' narrow-band SDFs on the host (p07), uploads via
    # SdfDeviceWorld, launches the shipping p07 sampler kernel through the
    # uploaded device SparseSdfDevice views, and asserts GPU == host sampler
    # (near-equal; out-of-band sentinel exact) + D1 two-run byte-identical.
    # The test holds a __global__, so it is a .cu (CUDA language). Needs both
    # the repo root (tests/import/sdf_*.hpp) and src on the include path.
    add_executable(nuka_sdf_device_test
        runtime/test_sdf_device_upload.cu
    )

    target_include_directories(nuka_sdf_device_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_sdf_device_test
        PRIVATE
            nuka_runtime_gpu
            nuka_cooker
            nuka_scene
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_sdf_device_test)

    # --- v0.7 p04: LBVH vs SAP differential gate + D1 byte-exact -----------
    add_executable(nuka_lbvh_vs_sap_test
        collision/test_lbvh_vs_sap_pair_set.cpp
    )

    target_link_libraries(nuka_lbvh_vs_sap_test
        PRIVATE
            nuka_collision_gpu
            nuka_runtime_gpu
            nuka_runtime
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_lbvh_vs_sap_test)

    # --- v0.7 p05: particle uniform grid + cross-system broadphase ---------
    # Correctness vs CPU brute-force + D1 two-run byte-exact for both the
    # particle neighbor grid and the particle->rigid cross-system query (which
    # traverses the LBVH retained by BuildLbvhForQuery).
    add_executable(nuka_particle_grid_test
        collision/test_particle_grid_correctness.cpp
        collision/test_particle_grid_determinism.cpp
        collision/test_cross_system_query.cpp
    )

    target_link_libraries(nuka_particle_grid_test
        PRIVATE
            nuka_collision_gpu
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_particle_grid_test)

    # --- v0.7 p12: self-written CUDA ray tracer (camera -> LBVH -> closest-hit
    # depth, box leaves). Matches a CPU brute-force ray-vs-all-boxes oracle that
    # REUSES the same __host__ __device__ slab test + closest-hit update + camera
    # ray-gen as the GPU kernel; D1 two-run byte-exact (memcmp); degenerate-deep
    # tree (deeper than stack[64]) renders with no overflow. The oracle TU is
    # built with -ffp-contract=off to pair with the kernel's --fmad=false so the
    # shared slab arithmetic is host==device bit-for-bit (depth max-err == 0).
    add_executable(nuka_rt_traversal_test
        rt/test_bvh_ray_traversal.cpp
    )
    set_source_files_properties(rt/test_bvh_ray_traversal.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    target_include_directories(nuka_rt_traversal_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rt_traversal_test
        PRIVATE
            nuka_rt_gpu
            nuka_collision_gpu
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_rt_traversal_test)

    # --- v0.7 p13: the SENSOR renderer (real per-primitive leaf intersections
    # triangle/sphere/sparse-SDF + Lambert+GGX shade + hard shadows + full AOV
    # framebuffer). Matches a CPU brute-force ray-vs-all-prims oracle that REUSES
    # the same __host__ __device__ intersection + shade + closest-hit functions
    # the kernel uses (triangle hit + bary EXACT, sphere depth + normal EXACT,
    # SDF hit within tolerance, Lambert/GGX == analytic, hard-shadow occlusion);
    # NaN-guard for a surface-origin axis-parallel ray; D1 two-run byte-exact
    # across ALL AOVs. Built -ffp-contract=off to pair with the kernel's
    # --fmad=false so the shared fp64-internal intersection math is host==device
    # bit-for-bit. Links nuka_cooker for the in-test cooked sparse SDF (p07).
    add_executable(nuka_rt_scene_render_test
        rt/test_scene_render.cpp
    )
    set_source_files_properties(rt/test_scene_render.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    target_include_directories(nuka_rt_scene_render_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rt_scene_render_test
        PRIVATE
            nuka_rt_gpu
            nuka_collision_gpu
            nuka_cooker
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_rt_scene_render_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.7 p08-B: Newton-on-summed-SDF contact (forward + D1) ------------
    # Cooks (p07, nuka_cooker) a sphere + box SDF; calls the header-inline
    # find_sdf_contact_newton (the SAME function the device kernel wraps) on the
    # host; emits RigidSDFContactRow / FeatherstoneSDFContactRow rows and runs
    # the existing PGS solve (nuka_solver, CUDA) to prove penetrating bodies are
    # pushed apart. nuka_collision_gpu carries the device kernel + row builders.
    add_executable(nuka_sdf_contact_test
        collision/test_sdf_contact.cpp
    )

    target_include_directories(nuka_sdf_contact_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_sdf_contact_test
        PRIVATE
            nuka_collision_gpu
            nuka_cooker
            nuka_solver
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_sdf_contact_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- p08-C Channel 2/3: envelope-theorem contact-geometry adjoint -------
    # Cooks box+sphere SDFs, runs the SHIPPING find_sdf_contact_newton, and
    # FD-validates the envelope depth + normal gradients by RE-RUNNING the descent
    # under perturbation (cell value / pose). Channel 3: documents the contact
    # point's valley-degeneracy (no point gradient produced/validated).
    add_executable(nuka_sdf_contact_adjoint_test
        collision/test_sdf_contact_adjoint_fd.cpp
    )

    target_include_directories(nuka_sdf_contact_adjoint_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_sdf_contact_adjoint_test
        PRIVATE
            nuka_collision_gpu
            nuka_cooker
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_sdf_contact_adjoint_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

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

    # --- v0.5 p02-A: differentiable single-step ABA reverse (FD ladder) -----
    add_executable(nuka_diffsim_aba_reverse_test
        diffsim/test_aba_reverse_fd.cpp
    )

    target_compile_definitions(nuka_diffsim_aba_reverse_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_diffsim_aba_reverse_test
        PRIVATE
            nuka_diffsim
            nuka_runtime
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_diffsim_aba_reverse_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.5 p02-B/C: multi-step diff-sim (tape/checkpoint/recompute/backward) -
    add_executable(nuka_diffsim_multistep_backward_test
        diffsim/test_multistep_backward.cpp
    )

    target_compile_definitions(nuka_diffsim_multistep_backward_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_diffsim_multistep_backward_test
        PRIVATE
            nuka_diffsim
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_diffsim_multistep_backward_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.5 p03 R1: self-written deterministic CG sparse solver (IFT backend) -
    # CG-vs-Eigen-LDLT dense cross-check + D1 bit-exact gate. Eigen is linked here
    # ONLY (nuka_eigen_testonly INTERFACE include dir), never into any production
    # target.
    add_executable(nuka_diffsim_cg_solver_test
        diffsim/test_cg_vs_dense.cpp
    )

    target_compile_definitions(nuka_diffsim_cg_solver_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_diffsim_cg_solver_test
        PRIVATE
            nuka_diffsim
            nuka_core
            nuka_phi
            nuka_eigen_testonly
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_diffsim_cg_solver_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.7 p01: hardened CG general-path solver tests --------------------
    # The v0.5 CG solver hardened + generalized for worse-conditioned systems:
    #   * test_cg_vs_dense          : Eigen-LDLT cross-check on kappa up to ~1e6 +
    #                                 D1 bit-exact (worse-conditioned coverage).
    #   * test_cg_convergence_rate  : CG A-norm error vs the kappa-theory bound in
    #                                 the informative regime + residual-history diag.
    #   * test_block_jacobi_tuning  : Jacobi-vs-BlockJacobi preconditioner quality
    #                                 sweep + the host diagnostics summarizer.
    # Eigen is linked test-only (the approved dense oracle), never a production path.
    foreach(cg_test
            test_cg_vs_dense
            test_cg_convergence_rate
            test_block_jacobi_tuning)
        add_executable(nuka_diffsim_${cg_test}
            diffsim/solver/${cg_test}.cpp)
        target_compile_definitions(nuka_diffsim_${cg_test}
            PRIVATE NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
        target_link_libraries(nuka_diffsim_${cg_test}
            PRIVATE
                nuka_diffsim
                nuka_core
                nuka_phi
                nuka_eigen_testonly
                GTest::gtest_main)
        gtest_discover_tests(nuka_diffsim_${cg_test}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endforeach()

    # --- v0.7 p02: self-written MINRES (symmetric indefinite) + ILU(0) ----------
    #   * test_minres_indefinite          : MINRES solves a known indefinite system
    #                                       (||Ax-b|| small) + D1 bit-exact.
    #   * test_minres_vs_dense_indefinite : MINRES vs Eigen FullPivLU on indefinite
    #                                       blocks (< 1e-5; FullPivLU, NOT LDLT).
    #   * test_ilu0_factorization         : ILU(0) factor matches a hand reference +
    #                                       host==device + D1 bit-exact.
    #   * test_ilu0_vs_jacobi_convergence : ILU(0)-MINRES converges in <= 0.5x iters
    #                                       vs Jacobi-MINRES on an ill-conditioned
    #                                       structurally-sparse DEFINITE block.
    # Eigen is linked test-only (the approved dense oracle), never a production path.
    foreach(minres_test
            test_minres_indefinite
            test_minres_vs_dense_indefinite
            test_ilu0_factorization
            test_ilu0_vs_jacobi_convergence)
        add_executable(nuka_diffsim_${minres_test}
            diffsim/solver/${minres_test}.cpp)
        target_compile_definitions(nuka_diffsim_${minres_test}
            PRIVATE NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
        target_link_libraries(nuka_diffsim_${minres_test}
            PRIVATE
                nuka_diffsim
                nuka_core
                nuka_phi
                nuka_eigen_testonly
                GTest::gtest_main)
        gtest_discover_tests(nuka_diffsim_${minres_test}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endforeach()

    # --- v0.7 p03-A: self-written GMRES (general NON-SYMMETRIC Krylov) -----------
    #   * test_gmres_nonsymmetric : GMRES vs Eigen PartialPivLU/FullPivLU on
    #                               genuinely non-symmetric blocks (< 1e-5); the m<n
    #                               restart-coverage case; the non-symmetric detector
    #                               byte-safety (symmetric -> 0, non-symmetric -> 1);
    #                               and the D1 two-run byte-exact gate.
    # Eigen is linked test-only (the approved dense oracle), never a production path.
    foreach(gmres_test
            test_gmres_nonsymmetric)
        add_executable(nuka_diffsim_${gmres_test}
            diffsim/solver/${gmres_test}.cpp)
        target_compile_definitions(nuka_diffsim_${gmres_test}
            PRIVATE NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
        target_link_libraries(nuka_diffsim_${gmres_test}
            PRIVATE
                nuka_diffsim
                nuka_core
                nuka_phi
                nuka_eigen_testonly
                GTest::gtest_main)
        gtest_discover_tests(nuka_diffsim_${gmres_test}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endforeach()

    # --- v0.5 p03 R2: contact KKT (Delassus) builder ------------------------
    # Builder (A = J M^-1 J^T) vs an INDEPENDENT host Gram recompute (Eigen,
    # < 1e-5 rel), exact bit-symmetry, the diagonal == 1/effective_mass identity,
    # and the D1 two-run byte-exact gate. Eigen is linked test-only.
    add_executable(nuka_diffsim_kkt_build_test
        diffsim/test_kkt_build.cpp
    )

    target_compile_definitions(nuka_diffsim_kkt_build_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_diffsim_kkt_build_test
        PRIVATE
            nuka_diffsim
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            nuka_eigen_testonly
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_diffsim_kkt_build_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.5 p03 R3a: IFT-at-convergence contact gradient ------------------
    # IFT analytic vs finite-difference through the REAL engine contact step
    # (< 1e-3 rel, the PRIMARY oracle -- p02 is contact-free so there is no
    # contact-tape to compare against), vs a dense Eigen::LDLT solve+distribute
    # reference (< 1e-5, second oracle), and the D1 two-run byte-exact gate.
    # Eigen is linked test-only.
    add_executable(nuka_diffsim_ift_backward_test
        diffsim/test_ift_vs_tape_backward.cpp
    )

    target_compile_definitions(nuka_diffsim_ift_backward_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_diffsim_ift_backward_test
        PRIVATE
            nuka_diffsim
            nuka_articulation_gpu
            nuka_articulation
            nuka_core
            nuka_phi
            nuka_eigen_testonly
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_diffsim_ift_backward_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.7 p08-C Channel 1 (§4.B): SDF-contact IFT d/dM, d/dJ ------------
    # The §4.B acceptance the v0.5 ift_runner DEFERRED. The FD PERTURBS actual
    # entries of M^-1 / J and RE-SOLVES the constraint system to convergence with
    # an INDEPENDENT Eigen LDLT oracle (not the module's own assemble/solve, so a
    # transpose/index bug is caught), then compares to the analytical IFT
    # dlambda = A^-1 (dr - dA lambda). Eigen linked test-only.
    add_executable(nuka_sdf_contact_ift_test
        diffsim/test_sdf_contact_ift_fd.cpp
    )

    target_link_libraries(nuka_sdf_contact_ift_test
        PRIVATE
            nuka_diffsim
            nuka_phi
            nuka_eigen_testonly
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_sdf_contact_ift_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.5 p03 R3b: OSC differentiable adjoint (R8 / R9) -----------------
    # Reverse-mode gain (Kp,Kd) + task-target gradients of the REAL forward OSC
    # drive kernel: analytic adjoint vs central finite-difference of the live
    # LaunchApplyOscDriveKernels (< 1e-3 rel), the D1 two-run byte-exact gate, and
    # the R9 clamp stop-grad (saturated component -> zero param grad). Links the
    # engine (scene cook + world + forward kernel) AND nuka_diffsim (the adjoint).
    add_executable(nuka_diffsim_osc_adjoint_test
        diffsim/test_osc_adjoint.cpp
    )

    target_compile_definitions(nuka_diffsim_osc_adjoint_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_diffsim_osc_adjoint_test
        PRIVATE
            nuka_diffsim
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_diffsim_osc_adjoint_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.5 p04 N1: sim-to-real sensor noise (Philox Gaussian + Poisson) ---
    # Launch-level gtests: Philox KAT + D1 two-run memcmp + statistics. Link the
    # self-contained noise lib directly -- no world/scene needed (fast + robust).
    add_executable(nuka_sensor_noise_test
        sensor/test_n1_gaussian.cu
        sensor/test_n1_poisson.cu
    )

    target_link_libraries(nuka_sensor_noise_test
        PRIVATE
            nuka_sensor_noise
            nuka_phi
            CUDA::cudart
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_sensor_noise_test
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

    # --- p14a (v0.7) any-link contact-force readout (LINK_CONTACT_WRENCH +
    # per-slot CONTACT_FORCE). Synthetic kernel unit test + Go2 stance Sigma F_z =
    # Mg integration + D1 two-run byte-exact. Same link set as the Go2 PD test
    # (nuka_articulation_gpu carries the wrench kernel sensor/contact_wrench.cu).
    add_executable(nuka_link_contact_wrench_test
        sensor/test_link_contact_wrench.cpp
    )

    target_compile_definitions(nuka_link_contact_wrench_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_link_contact_wrench_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_link_contact_wrench_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- p14b (v0.7) RGB+depth+normal+uv CAMERA SENSOR over the LIVE batched sim
    # state. Analytic depth/camera-space-normal/palette-RGB/miss; LIVE-tracking
    # (move a body's DevicePoses() -> image changes); batched per-env; D1 two-run
    # byte-exact across all channels. The poses come from a real BatchedDeviceWorld
    # (nuka_runtime_gpu), the SDFs are real p07 cooked SDFs (nuka_cooker); the
    # renderer is nuka_camera_sensor_gpu. -ffp-contract=off pairs with the kernel's
    # --fmad=false so the shared fp64 SDF sphere-trace is host==device bit-for-bit.
    add_executable(nuka_camera_render_test
        sensor/test_camera_render.cpp
    )
    set_source_files_properties(sensor/test_camera_render.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    target_include_directories(nuka_camera_render_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_camera_render_test
        PRIVATE
            nuka_camera_sensor_gpu
            nuka_runtime_gpu
            nuka_runtime
            nuka_cooker
            nuka_collision_gpu
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_camera_render_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

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

    # --- v0.5 C-fwd: Torque & Velocity control modes ------------------------
    add_executable(nuka_control_modes_test
        runtime/test_control_modes.cpp
    )

    target_compile_definitions(nuka_control_modes_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_control_modes_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_control_modes_test
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

    # --- 4096-env Go2 per-env-step perf-GATE regression test (p01 T3.1.6) ----
    # Records the 4096-env per-env-step µs ALWAYS; asserts the §7 gate only on
    # the designated validation GPU (NUKA_PERF_VALIDATION_GPU), else GTEST_SKIP.
    # Records-and-skips (green) on the dev RTX 4000 Ada -- no false CI failure.
    add_executable(nuka_go2_4096env_step_time_test
        perf/test_go2_4096env_step_time.cpp
    )

    target_compile_definitions(nuka_go2_4096env_step_time_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_go2_4096env_step_time_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_go2_4096env_step_time_test
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
    perf/test_lbvh_50k_bodies.cpp
    perf/test_1m_particles_grid_perf.cpp
)

    target_link_libraries(nuka_cuda_perf_test
        PRIVATE
            nuka_debug_draw
            nuka_sensor_gpu
            nuka_sensor
            nuka_runtime_gpu
            nuka_runtime
            nuka_collision_gpu
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
