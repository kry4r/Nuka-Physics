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
    scene/test_scene_compose.cpp
    scene/test_scene_cooker.cpp
    scene/test_scene_pipeline.cpp
    scene/test_contact_metadata_cook.cpp
    scene/test_filter_precedence.cpp
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
    import/test_mjcf_contact_filtering.cpp
    import/test_urdf_importer.cpp
    import/test_usd_importer.cpp
    import/test_usd_mesh_geometry.cpp
    # v0.8 C7a: the self-written USDC (crate) reader on the REAL newton cup.
    # Links nuka_scene/nuka_cooker for the convex-hull cook gate (CookScene).
    import/test_usdc_cup.cpp
    import/test_decompose_attribute.cpp
    import/test_mesh_file_loader.cpp
)

target_link_libraries(nuka_import_test
    PRIVATE
        nuka_import
        nuka_scene
        nuka_cooker
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

# --- v0.8 C4b: compliant-normal + pyramid-friction row emitter -------------
# Validates the NEW, VALIDATED-NOT-WIRED EmitCompliantContactRows (row_builder):
# row counts (4 normal + 16 pyramid for a condim=3 box), per-normal R/aref
# cross-check vs an independent ComputeCompliantRow (C4a), the pyramid spokes
# summing to the isotropic disk, the ContactRowSides side-output, 2-run host
# byte-identity, and INERTNESS of the legacy BuildContactRows/AppendContactGroup
# path. Plain host C++ (no CUDA): nuka_constraint is host-only.
add_executable(nuka_compliant_rows_test
    constraint/test_compliant_rows.cpp
)
target_link_libraries(nuka_compliant_rows_test
    PRIVATE
        nuka_constraint
        GTest::gtest_main
)
gtest_discover_tests(nuka_compliant_rows_test)

# --- Solver tests ----------------------------------------------------------
add_executable(nuka_solver_test
    solver/test_row_scheduler_bridge.cpp
    # Row-coloring O(rows*K) rewrite EQUIVALENCE gate: production
    # BuildRowColorPartitions / ValidateNoSharedBodiesPerColor vs a test-local
    # O(rows^2) reference, byte-identical over synthetic cliques + randomized
    # rows (incl. invalid-body / body_a==body_b traps) + dense/all-invalid edges.
    solver/test_row_scheduler_coloring_equivalence.cpp
    # v0.8 C5a: UnifiedSolve end-to-end (compliant contact + pyramid + D1).
    solver/test_unified_solve.cpp
    # v0.8 C5b-core: ArticulationChainJ reaction arm wired into UnifiedSolve
    # (closed-form chain oracle + same-articulation coloring + link<->rigid two-
    # way + D1). Hand-built ArticulationReactionState data (no device CRBA).
    solver/test_link_rigid_twoway.cpp
    # v0.8 C5b-core: compliant-PGS regularizer R-SWEEP guard (210c11f). Drives a
    # coupled full-rank SPD A = J M^-1 J^T (dof_stride==N) compliant system through
    # UnifiedSolve and matches the converged lambda to a double-precision dense
    # (A+diag(R))lambda=b solve across R/A from ~1e-3 to ~1e+1, + a large-R
    # discriminator vs the pre-fix A*lambda=b fixed point + D1. MJX-FREE.
    solver/test_compliant_pgs_regularizer.cpp
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

    # --- v0.8 C4a: MuJoCo _efc_row solref/solimp compliance port -----------
    # Tests src/constraint/solref_solimp.hpp (header-only HD). Two regimes:
    #   (1) function-vs-hand-evaluated-table (tolerance, independent oracle), and
    #   (2) HOST-vs-DEVICE BYTE-EXACT parity -- the gate that justifies OPEN-F's
    #       integer-power decision. The .cpp holds a __global__ so it compiles as
    #       CUDA. Built --fmad=false (device) + -ffp-contract=off (host) so the
    #       shared HD arithmetic is host==device bit-for-bit (no FMA contraction
    #       on either side, integer-power => no libm powf divergence). Header-only
    #       => no library link beyond cudart + gtest.
    add_executable(nuka_solref_solimp_test
        constraint/test_solref_solimp.cpp
    )
    set_source_files_properties(constraint/test_solref_solimp.cpp
        PROPERTIES LANGUAGE CUDA
                   COMPILE_OPTIONS "--fmad=false;-Xcompiler=-ffp-contract=off")
    target_include_directories(nuka_solref_solimp_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_solref_solimp_test
        PRIVATE
            CUDA::cudart
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_solref_solimp_test)

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

    # --- v0.8 C2a: unified candidate-pair stream (the broadphase OUTPUT type) ---
    # Canonical stable_key (type-in-high-bits, collision-free), thrust::stable_sort
    # builder over a fixed host-provided pair set, two-run byte-exact + N>=32
    # cross-replica D1. Links nuka_collision_gpu (carries candidate_pair.cu) +
    # nuka_phi. Needs nuka_codegen: candidate_pair.hpp includes the GENERATED
    # collidable_registry.hpp (via constraint/collidable.hpp).
    add_executable(nuka_candidate_stream_test
        collision/test_candidate_stream.cpp
    )
    add_dependencies(nuka_candidate_stream_test nuka_codegen)
    target_include_directories(nuka_candidate_stream_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_candidate_stream_test
        PRIVATE
            nuka_collision_gpu
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_candidate_stream_test)

    # --- v0.8 C2b: RIGID<->RIGID filtered candidate-pair stream validation -----
    # BuildRigidCandidatePairs (LBVH over per-shape AABBs -> bitmask + exclude +
    # parent-child auto-exclude filter -> explicit-<pair> force-include -> sorted
    # D1 stream) vs an independent host oracle driven by real CookScene policy.
    # Gates: unfiltered==brute-force overlap, filtered==oracle (per-case), >=32
    # body D1 two-run byte-exact, empty/no-overlap sanity. Links nuka_collision_gpu
    # (rigid_candidate_pairs.cu) + nuka_scene (CookScene) + nuka_phi. Needs
    # nuka_codegen (candidate_pair.hpp -> generated collidable_registry.hpp).
    add_executable(nuka_lbvh_filtered_pairs_test
        collision/test_lbvh_filtered_pairs.cpp
    )
    add_dependencies(nuka_lbvh_filtered_pairs_test nuka_codegen)
    target_include_directories(nuka_lbvh_filtered_pairs_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_lbvh_filtered_pairs_test
        PRIVATE
            nuka_collision_gpu
            nuka_scene
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_lbvh_filtered_pairs_test)

    # --- v0.8 co-residence (commit 2/3): ARTICULATION-LINK <-> RIGID candidate
    # pairs through the generalized tagged broadphase core. Synthetic mixed AABB
    # sets + per-AABB tags (no cooked robot) exercise: cross-type emission (link
    # AABB overlapping a rigid AABB -> one CandidatePair with correct per-side
    # type + handle, canonical stable_key), same-type alongside, cross-type
    # bitmask + exclude drops (non-vacuous), the rigid wrapper byte-identical to
    # the tagged core (regression), two-run D1 byte-exactness, and the handle<2^28
    # guard. Consumes commit 1's LinkShapeAabbs (link_aabb.hpp). VALIDATED-NOT-
    # WIRED. Links nuka_collision_gpu (rigid_candidate_pairs.cu) + nuka_phi. Needs
    # nuka_codegen (candidate_pair.hpp -> generated collidable_registry.hpp); the
    # scene::CookedFilterPolicy struct comes from nuka_scene (header-only use).
    add_executable(nuka_link_rigid_candidate_pairs_test
        collision/test_link_rigid_candidate_pairs.cpp
    )
    add_dependencies(nuka_link_rigid_candidate_pairs_test nuka_codegen)
    target_include_directories(nuka_link_rigid_candidate_pairs_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_link_rigid_candidate_pairs_test
        PRIVATE
            nuka_collision_gpu
            nuka_scene
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_link_rigid_candidate_pairs_test)

    # --- v0.8 C3a: extended ContactManifold + narrowphase dispatch table -------
    # Validates SelectTier routing (primitive->Analytical, mesh/convex->Convex,
    # SDF-equipped->Sdf), the constexpr (typeA,typeB,tier)->handler dispatch table
    # (looked-up fn-ptr == expected tier stub AND call-through stamps the tier
    # marker), and the extended ContactManifold round-trip (AddPoint clamp@4, Clear
    # reset, CollidableRef a/b type+react+handle, trivially-copyable memcpy
    # byte-stability). Header-only (narrowphase_dispatch.hpp + contact_manifold.hpp
    # + collidable.hpp) -- needs nuka_codegen (collidable.hpp -> generated
    # collidable_registry.hpp) + the src include dir; nuka_phi for completeness.
    add_executable(nuka_narrowphase_dispatch_test
        collision/test_narrowphase_dispatch.cpp
    )
    add_dependencies(nuka_narrowphase_dispatch_test nuka_codegen)
    target_include_directories(nuka_narrowphase_dispatch_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_narrowphase_dispatch_test
        PRIVATE
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_narrowphase_dispatch_test)

    # --- v0.8 C3b: analytical primitive narrowphase (multi-point/face-face) -----
    # Validates the REAL Analytical-tier handlers (analytical_manifold.hpp +
    # narrowphase_dispatch.hpp): box-on-plane 4 corners, box-box face-clip 4-pt
    # patch (order-insensitive SET vs hand oracle), sphere x {sphere,box,plane}
    # 1-pt analytic, capsule x plane (2 endpoints) / capsule x sphere (1-pt),
    # both (a,b) orderings (normal sign = separation dir for side A), the
    # determinism-trap gate (symmetric box-box -> stable 4 distinct corners) and
    # the two-run byte-identical (memcmp on zeroed buffers) D1 gate. Header-only
    # (same deps as the C3a dispatch test) -- needs nuka_codegen + the src include.
    add_executable(nuka_analytical_manifold_test
        collision/test_analytical_manifold.cpp
    )
    add_dependencies(nuka_analytical_manifold_test nuka_codegen)
    target_include_directories(nuka_analytical_manifold_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_analytical_manifold_test
        PRIVATE
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_analytical_manifold_test)

    # --- v0.8 C3c: convex narrowphase (GJK/EPA/face-clip) -----------------------
    # Validates the REAL Convex-tier handler (convex_narrowphase.hpp +
    # narrowphase_dispatch.hpp): two penetrating tetrahedra / two boxes-as-hulls vs
    # analytic depth+normal (and a TIGHT depth/normal cross-check vs the C3b box-box
    # SAT), a separated pair -> no contact, convex x box / convex x sphere / convex
    # x plane (both orderings, the plane special path + sign), and the EPA D1 gate:
    # 2-run byte-identity (memset+memcmp) + N>=32 cross-replica on a SYMMETRIC
    # tie-prone box-box config (a heap-based EPA would diverge there). Header-only
    # (same deps as the C3a/C3b dispatch tests) -- needs nuka_codegen + src include.
    add_executable(nuka_gjk_epa_convex_test
        collision/test_gjk_epa_convex.cpp
    )
    add_dependencies(nuka_gjk_epa_convex_test nuka_codegen)
    target_include_directories(nuka_gjk_epa_convex_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    # nuka_import/nuka_scene/nuka_cooker: the C3-hardening regression
    # (SphereHullShallowPenetrationIsMonotone) loads the REAL cup hull -- the only
    # geometry that reproduces the EPA shallow-penetration dead band -- via
    # LoadUsd + CookScene (asset-gated; SKIPs when the fetch-per-env cup is absent).
    target_link_libraries(nuka_gjk_epa_convex_test
        PRIVATE
            nuka_phi
            nuka_import
            nuka_scene
            nuka_cooker
            GTest::gtest_main
    )
    # WORKING_DIRECTORY = repo root so the relative cup asset path resolves.
    gtest_discover_tests(nuka_gjk_epa_convex_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 C3d: SDF high-precision tier WIRED into the dispatch table --------
    # Validates the REAL Sdf-tier handler (NarrowphaseSdf, narrowphase_dispatch.hpp)
    # -- the FIRST stepping callsite of find_sdf_contact_newton (sdf_contact.hpp,
    # previously zero-callsite dead code). Cooks two unit-cube SDFs (CookSparseSdf,
    # nuka_cooker), stacks them in contact (centers symmetric about y=0 so the
    # handler's center-midpoint guess lands in both narrow bands), drives the
    # resolved Sdf handler, and asserts: ResolveNarrowphase(..,has_sdf=true) is the
    # real handler (not the removed stub); the 1-point manifold's normal (+Y = sep
    # dir for side A) + penetration match the analytical box-on-box contact within
    # voxel tol; a separated/null pair -> empty manifold; 2-run byte-identity (D1,
    # memcmp on zeroed buffers). Header-only SDF path (find_sdf_contact_newton is
    # the inline HD function -- NO device kernel launch), so deps are nuka_cooker +
    # nuka_codegen + nuka_phi + the src/root includes (NOT nuka_collision_gpu).
    add_executable(nuka_sdf_tier_wired_test
        collision/test_sdf_tier_wired.cpp
    )
    add_dependencies(nuka_sdf_tier_wired_test nuka_codegen)
    target_include_directories(nuka_sdf_tier_wired_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_sdf_tier_wired_test
        PRIVATE
            nuka_cooker
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_sdf_tier_wired_test)

    # --- v0.8 C5a: contact stream DRIVER (CandidatePair -> ContactManifold) ------
    # Validates the resolver-decoupled host driver (contact_stream_driver.hpp/.cpp,
    # BuildContactManifolds): resolve each side via a caller-supplied ShapeResolver
    # (the rigid-BodyState / articulation-link-pose decoupling seam), derive
    # has_sdf = a.has_sdf && b.has_sdf (both-sides contract), route through the
    # constexpr dispatch table, invoke the handler, merge the two sides' MuJoCo
    # contact params via scene::MergeContactParams, and append non-empty manifolds.
    # Cases: box-on-plane (rigid x static) geometry == direct ResolveNarrowphase
    # call + merged friction/solimp/solref + manifold_key/a/b; foot sphere-on-plane
    # (articulation-link x static, the C5c foot-ground shape); box-box (rigid x
    # rigid) multi-point == direct call; param merge cross-checked vs an
    # INDEPENDENT MergeContactParams call; has_sdf both-sides (one-sided -> real
    # analytical, both-sided + null seam -> Sdf null-guards -> skipped); skip when a
    # side fails to resolve; D1 two-run byte-exact (memcmp). Compiles the driver
    # .cpp directly (host-only -- the dispatch table is host fn-ptrs; the device
    # driver is C5/later). Header deps == the C3a dispatch test (narrowphase_
    # dispatch.hpp + contact_filter.hpp) -- nuka_codegen + the src include + nuka_phi.
    add_executable(nuka_contact_stream_driver_test
        collision/test_contact_stream_driver.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_contact_stream_driver_test nuka_codegen)
    target_include_directories(nuka_contact_stream_driver_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_contact_stream_driver_test
        PRIVATE
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_contact_stream_driver_test)

    # --- v0.8 C2c: particle<->rigid + particle<->particle candidate-pair builders
    # gated by the C1c SystemPairMatrix. Validates the builders' reframed candidate
    # SET against (1) a hand-reframe of the raw cross-system CSR (non-identity
    # shape->body map + 2-shapes->1-body dedup case), (2) a brute-force i<j radius
    # oracle for p<->p, (3) the matrix-toggle gate (non-vacuous, same scene),
    # (4) >=32-particle 2-run byte-exact D1, (5) 0-particle sanity. Links
    # nuka_collision_gpu (particle_candidate_pairs.cu). Needs nuka_codegen
    # (candidate_pair.hpp -> generated collidable_registry.hpp).
    add_executable(nuka_cross_system_matrix_test
        collision/test_cross_system_matrix.cpp
    )
    add_dependencies(nuka_cross_system_matrix_test nuka_codegen)
    target_include_directories(nuka_cross_system_matrix_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_cross_system_matrix_test
        PRIVATE
            nuka_collision_gpu
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_cross_system_matrix_test)

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

    # --- v0.7 G1-A: the self-written TWO-LEVEL (TLAS/BLAS) ray tracer
    # (rt::BuildTwoLevelScene + rt::RenderFrame). Gates:
    #   * Oracle 1 (host two-level brute-force, shares QuatInvRotate/intersect/
    #     PackPrimId): byte-exact host==device across ALL 6 AOVs for ANY transform
    #     incl. rotation (D1 + host/device-parity + traversal-independence).
    #   * Oracle 2 (the trusted INDEPENDENT flat p13 RenderScene with each
    #     instance baked to world verts): memcmp==0 depth/normal/color/albedo at
    #     IDENTITY instances; near-exact (stated tol) for rotated/translated.
    #   * Analytic anchors: p13's single-triangle t through the nest + a NEW
    #     rotated+translated instance with a hand-computed world t AND an exact
    #     prim_id == PackPrimId(known_instance, known_local) (bit-split + instance-
    #     high ordering gate).
    #   * Instance-transform tracking (move one instance, reuse the BLAS).
    #   * D1 two-run byte-exact across ALL 6 AOVs.
    # Built -ffp-contract=off to pair with the kernel's --fmad=false so Oracle 1's
    # shared fp64-internal math + fp32 QuatInvRotate is host==device bit-for-bit.
    add_executable(nuka_rt_two_level_test
        rt/test_two_level_render.cpp
    )
    set_source_files_properties(rt/test_two_level_render.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    target_include_directories(nuka_rt_two_level_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rt_two_level_test
        PRIVATE
            nuka_rt_gpu
            nuka_collision_gpu
            nuka_cooker
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_rt_two_level_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- H1 cup demo: PBR-style hero scene smoke over the two-level RT path ----
    add_executable(nuka_h1_cup_pbr_render_test
        rt/test_h1_cup_pbr_render.cpp
    )
    set_source_files_properties(rt/test_h1_cup_pbr_render.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    target_include_directories(nuka_h1_cup_pbr_render_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_h1_cup_pbr_render_test
        PRIVATE
            nuka_rt_gpu
            nuka_collision_gpu
            nuka_cooker
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_h1_cup_pbr_render_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.7 G1-C: two-level (TLAS/BLAS) ray-tracer BENCHMARK harness -------
    # MEASURES flat p13 RenderScene (rebuild-all every frame over N*~300 baked
    # world triangles) vs the G1-A two-level RenderFrame (per-mesh BLAS built
    # ONCE + cheap per-frame N-instance TLAS rebuild). Sweeps N=1,16,64,256,1024
    # at 512x512, animates instances each frame, times the WHOLE host per-frame
    # call (wall-clock), reports ms/frame + full-per-frame-wall-time MRays/s +
    # speedup, and prints the explicit G1-B go/no-go verdict. Asserts ONLY the
    # relative structural invariant (two-level < flat at N>=64); absolute numbers
    # are hardware-dependent (nvidia-smi NVML-broken) and never asserted.
    add_executable(nuka_rt_two_level_bench
        rt/test_two_level_bench.cpp
    )
    set_source_files_properties(rt/test_two_level_bench.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    target_include_directories(nuka_rt_two_level_bench
        PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(nuka_rt_two_level_bench
        PRIVATE
            nuka_rt_gpu
            nuka_collision_gpu
            nuka_cooker
            nuka_phi
            GTest::gtest_main
    )
    gtest_discover_tests(nuka_rt_two_level_bench WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})


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

    # --- v0.8 SG G0: DOF honesty above 18 (whole-body H1 precondition) -------
    # Synthetic >18-DOF chains: M^-1 factorization covers the FULL dof block,
    # m_eff uses the full chain-J, implicit damping reaches DOF>18, and beyond
    # kMaxArticulationDof the launchers fail LOUDLY (never a silent clamp).
    add_executable(nuka_dof_above18_honesty_test
        runtime/test_dof_above18_honesty.cpp
    )

    target_compile_definitions(nuka_dof_above18_honesty_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_dof_above18_honesty_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_dof_above18_honesty_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 C4c: reaction providers (rigid invM / chain-J / particle invM) -
    # Validates the NEW, VALIDATED-NOT-WIRED src/constraint/reaction_provider.hpp
    # §0.5 interface: a 1-DOF pendulum chain-J vs a hand-computed J M^-1 J^T (with
    # an OFF-origin contact point -- the C4c arbitrary-point generalization),
    # rigid invM vs the analytic diagonal path, particle scalar invM, StaticNull
    # zero/no-op, and 2-run byte-identity (D1). Round-trips the proven device CRBA
    # M^-1 + chain-J builders the articulation provider consumes, so it links the
    # articulation GPU libs + nuka_constraint (the header-only providers).
    add_executable(nuka_reaction_providers_test
        constraint/test_reaction_providers.cpp
    )

    target_compile_definitions(nuka_reaction_providers_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_link_libraries(nuka_reaction_providers_test
        PRIVATE
            nuka_constraint
            nuka_runtime
            nuka_articulation_gpu
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_reaction_providers_test
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

    # --- v0.8 C5c-1: floating-base go2 foot<->ground contact through the UNIFIED
    # collision/contact spine (contact_stream_driver -> sphere x plane narrowphase
    # -> EmitCompliantContactRows -> UnifiedSolve C5b-core ArticulationChainJ arm).
    # PROVES the foot contact force recoils into the floating base via the 18-wide
    # foot chain Jacobian (6 base + 12 leg): a structural base-column J check
    # (roll=+lever_y, pitch=-lever_x, lin_z=1), the base-z recoil through the
    # unified solve, a summed-normal-impulse ~ m_total*g*dt weight balance, and D1
    # two-run byte-identity. Compiles the host contact_stream_driver.cpp directly
    # (validated-not-wired: it is in no production lib); needs nuka_codegen for the
    # generated collidable_registry.hpp.
    add_executable(nuka_foot_ground_subsume_test
        solver/test_foot_ground_subsume.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_foot_ground_subsume_test nuka_codegen)

    target_compile_definitions(nuka_foot_ground_subsume_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_foot_ground_subsume_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_foot_ground_subsume_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_foot_ground_subsume_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 co-residence PEAK (commit 1/3): per-link/per-shape WORLD AABB
    # extraction from articulation FK (src/collision/link_aabb.hpp). The
    # artic-link<->rigid broadphase PREREQUISITE C2b excluded (no per-link AABB).
    # Cooks go2_float, sweeps >=3 distinct FK poses, asserts each foot sphere's
    # world AABB contains its surface samples + is exactly tight, a synthetic
    # rotated-box AABB == an independent 8-corner reference (and grows under
    # rotation), a capsule's caps are contained, D1 two-run byte-identity, AND a
    # __global__ kernel proves the HD core compiles + runs on-device == host
    # (so commit 2 can call it in a .cu). VALIDATED-NOT-WIRED. The .cu drives the
    # device-core proof; needs nuka_codegen for the generated registry CookScene/
    # BuildWorld pull in (same dep set as the foot-ground subsume exe).
    add_executable(nuka_link_aabb_test
        collision/test_link_aabb.cu
    )
    add_dependencies(nuka_link_aabb_test nuka_codegen)

    target_compile_definitions(nuka_link_aabb_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_link_aabb_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_link_aabb_test
        PRIVATE
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_link_aabb_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 C5c-2: MJX-parity ACCEPTANCE for the foot-ground subsume.
    # At a FIXED penetrating contact state (go2 feet pressed delta into a +Z
    # ground plane, qpos fixed, qvel=0) one mjx.forward -> qacc is compared to
    # the qacc Nuka produces through the UNIFIED contact pipeline (reuses the
    # C5c-1 wiring). PRIMARY assertion: the BASE-6 qacc match (the foot contact
    # force recoils into the floating base MJX-exactly). The MJX golden
    # (go2_foot_contact_step.bin, generated via generate_mjx_golden.py --mode
    # floating-contact-step) is a SOURCE TRANSLATION of the SAME go2_float.usda
    # Nuka cooks, so both sides share M by construction. Single-step, NOT a
    # rollout. Same host contact_stream_driver.cpp compile as the subsume test.
    add_executable(nuka_foot_ground_mjx_parity_test
        solver/test_foot_ground_mjx_parity.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_foot_ground_mjx_parity_test nuka_codegen)

    target_compile_definitions(nuka_foot_ground_mjx_parity_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_foot_ground_mjx_parity_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_foot_ground_mjx_parity_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_foot_ground_mjx_parity_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 co-residence commit 3/3: THE GATE (the grasp crux). A go2
    # articulation FOOT and a MOVABLE RIGID BODY box co-resident in ONE unified
    # contact solve, BOTH reaction arms firing (foot=ArticulationChainJ, box=
    # RigidInvMass) and the box GPU BodyState velocity actually changing. De-orphans
    # commit 1 (link_aabb.hpp ExtractLinkShapeAabbs) + commit 2 (rigid_candidate_
    # pairs.cu BuildArticulationRigidCandidatePairs) by driving the FULL stack:
    # per-link AABB -> cross-type broadphase -> sphere x box narrowphase ->
    # compliant rows -> UnifiedSolve. PRIMARY: kernel lambda vs the DOUBLE exact
    # (A+R) solve of the co-resident system (J_foot M^-1 J_foot^T + box inv-mass).
    # STRUCTURAL: box linear velocity changes (sign-checked away from the foot), the
    # articulation qdot recoils (two-way), box angular ~0 (the linear-only-emitter
    # grasp-blocker finding, reported). VALIDATED-NOT-WIRED. Links nuka_collision_gpu
    # (rigid_candidate_pairs.cu) + compiles contact_stream_driver.cpp like the
    # subsume/parity tests; needs nuka_codegen for the generated registry.
    add_executable(nuka_foot_box_coresidence_test
        solver/test_foot_box_coresidence.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_foot_box_coresidence_test nuka_codegen)

    target_compile_definitions(nuka_foot_box_coresidence_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_foot_box_coresidence_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_foot_box_coresidence_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_foot_box_coresidence_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 W1b: the CO-RESIDENT FREE-BOX foot-contact STEP-0 MJX-parity
    # capstone -- the box-side REALITY ANCHOR to MuJoCo for the unified co-resident
    # contact path. A go2 FL foot penetrates a FREE rigid box; one mjx.forward gives
    # the ground-truth qacc for the articulation base 6 + legs 12 AND the box 6 free
    # DOFs. Nuka's unified pipeline (per-link AABB -> cross-type broadphase ->
    # sphere x box narrowphase -> compliant rows -> UnifiedSolve) must reproduce the
    # BOX 6-DOF qacc (PRIMARY) + the base-6 (SECONDARY, the C5c-2 anchor with the box
    # present). The box mass+inertia are pinned to match MJX exactly (GATE 1). New
    # golden go2_foot_box_contact_step.bin (generated, NOT hand-edited).
    # VALIDATED-NOT-WIRED. Same dep set as nuka_foot_box_coresidence_test.
    add_executable(nuka_foot_box_mjx_parity_test
        solver/test_foot_box_mjx_parity.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_foot_box_mjx_parity_test nuka_codegen)

    target_compile_definitions(nuka_foot_box_mjx_parity_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_foot_box_mjx_parity_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_foot_box_mjx_parity_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_foot_box_mjx_parity_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 W1a: the MULTI-STEP unified co-resident contact stepper. A go2 foot
    # + a movable rigid box co-resident, stepped TOGETHER (predict -> detect ->
    # solve -> integrate -> repeat) so the contact HOLDS over many steps. REUSES
    # FeatherstoneAba:: integration verbatim (the production integrator). Compiles
    # the NEW src/runtime/coresident/unified_coresident_stepper.cpp + the host
    # contact_stream_driver.cpp directly (same pattern as the subsume / parity /
    # coresidence tests). 4 gates: (1) integrator-isolation closed-form ballistic
    # spin (the sharp gate, with a test-side bite proof), (2) multi-step stability +
    # two-way reaction, (3) V2 energy non-increasing, (4) D1 byte-exact two-run.
    # VALIDATED-NOT-WIRED. Same dep set as nuka_foot_box_coresidence_test.
    add_executable(nuka_unified_coresident_stepper_test
        coresident/test_unified_coresident_stepper.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/unified_coresident_stepper.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_unified_coresident_stepper_test nuka_codegen)

    target_compile_definitions(nuka_unified_coresident_stepper_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_unified_coresident_stepper_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_unified_coresident_stepper_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_unified_coresident_stepper_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 P2.1/P2.2: BatchedUnifiedWorld -- the GENERAL, scene-driven, BATCHED
    # world. N parallel envs of co-resident rigid bodies on the env-major buffer
    # layout the batched UnifiedSolve consumes. P2.1: free-fall (NO contact). P2.2:
    # the batched box<->static-ground contact phase (per-env narrowphase -> shared
    # compliant rows with per-env body-id offsets -> ONE UnifiedSolve), byte-identical
    # to UnifiedCoResidentStepper's box<->ground branch at N=1. ADDITIVE: compiles the
    # NEW src/runtime/coresident/batched_unified_world.cpp; modifies no production
    # stepper/solver/golden. P2.2 pulls in the collision/constraint/solver spine, the
    # host contact_stream_driver.cpp (BuildContactManifolds), and -- for Gate A2's
    # production anchor -- the unified_coresident_stepper.cpp (read-only co-resident
    # box-mode oracle; the stepper TU is NOT edited). Same dep set as the co-resident
    # stepper test target. Gates: P2.1 free-fall x4 + P2.2 A1/A2/B/C/D.
    add_executable(nuka_batched_unified_world_test
        coresident/test_batched_unified_world.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/batched_unified_world.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/unified_coresident_stepper.cpp
        # A2: the grasp-scene builders were promoted out of this test into the shared
        # library factory; the test DELEGATES to them, so it compiles the same TU.
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/grasp_scene_factory.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_batched_unified_world_test nuka_codegen)

    target_compile_definitions(nuka_batched_unified_world_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_batched_unified_world_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_batched_unified_world_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_batched_unified_world_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 P2.4a: the BatchedUnifiedWorld env-steps/sec PERF GATE. A COMMITTED
    # throughput harness + per-stage wall-clock attribution (a hypothesis test, not a
    # correctness gate). Links the SAME batched_unified_world.cpp the correctness test
    # links (so any accidental physics change there surfaces in the 16 gates), drives the
    # grasp scene, sweeps N in {1,8,32,64,256,1024}, and dumps the N=32 stage breakdown
    # + the ARTIC_SYNC_STORM vs row_solver verdict. SAME dep set / extra sources as
    # nuka_batched_unified_world_test. ADDITIVE: a NEW target, no production edits.
    add_executable(nuka_batched_unified_world_perf_test
        coresident/test_batched_unified_world_perf.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/batched_unified_world.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/unified_coresident_stepper.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_batched_unified_world_perf_test nuka_codegen)

    target_compile_definitions(nuka_batched_unified_world_perf_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_batched_unified_world_perf_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_batched_unified_world_perf_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_cooker  # G1e: the union scene's CookScene/CookArticulations path.
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_batched_unified_world_perf_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 C7b-1: the HOLD-AGAINST-GRAVITY grasp SPIKE. A synthetic fixed-base
    # 2-finger gripper holds the C7a cup (convex hull) against gravity by FINGER
    # FRICTION ALONE -- NO table. Drives the generalized unified_coresident_stepper
    # in grasp mode (N fingertip<->cup contacts, condim=3 friction, constant grip
    # torque through the control->tau->ABA drive path). Gates: STEP-0 drive-path-is-
    # live probe, the discriminating hold gate (Sigma(finger vert impulse) ~= m*g*dt,
    # no static row, bounded drift), V2 energy, the grip=0 BITE proof (cup falls),
    # D1. Reuses the C7a usdc cup (fetch-per-env; SKIP if absent). VALIDATED-NOT-
    # WIRED. Same dep set as the W1a stepper test (+ compiles the stepper + driver).
    add_executable(nuka_grasp_hold_spike_test
        coresident/test_grasp_hold_spike.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/unified_coresident_stepper.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_grasp_hold_spike_test nuka_codegen)

    target_compile_definitions(nuka_grasp_hold_spike_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            NUKA_GRASP_DIAG
    )

    target_include_directories(nuka_grasp_hold_spike_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_grasp_hold_spike_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_grasp_hold_spike_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 C7b-2a: the REAL h1_with_hand grasp SPIKE ---------------------
    # Replaces the synthetic gripper with an imported unitree h1_with_hand hand
    # (MJCF -> SceneIR -> cook -> ArticulationHostState, base forced Fixed). A real
    # revolute thumb + opposing finger pinch the C7a cup; constant grip torque
    # squeezes; the cup is held against gravity AND a disturbance by finger friction
    # ALONE (NO table). Gates: STEP-ZERO posability (opposing fingertip spheres
    # straddle the cup), the hold gate (Sigma(finger vert impulse) ~= m*g*dt, no
    # static row, bounded drift), the grip=0 BITE, the DISTURBANCE cage gate, V2
    # energy, D1. Asset-gated (SKIP if h1/cup absent). VALIDATED-NOT-WIRED. Same dep
    # set as the C7b-1 grasp spike (+ the MJCF importer is already in nuka_import).
    add_executable(nuka_h1_grasp_spike_test
        coresident/test_h1_grasp_spike.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/unified_coresident_stepper.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_h1_grasp_spike_test nuka_codegen)

    target_compile_definitions(nuka_h1_grasp_spike_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_h1_grasp_spike_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_h1_grasp_spike_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_h1_grasp_spike_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 C7b-2a-v2: the H1 POWER/WRAP grasp + LIFT spike ------------------
    # Mirrors the fingertip h1 spike block. STEP ZERO (FK surround) + a table-
    # supported wrap-close -> table-removal LIFT gate (the discriminator) + dynamic
    # lift load + BITE + D1. Drives the SAME unified_coresident_stepper (grasp mode +
    # the additive, inert-by-default cup<->table support). Asset-gated (SKIP if
    # h1/cup absent). VALIDATED-NOT-WIRED. Same dep set as the fingertip spike.
    add_executable(nuka_h1_power_grasp_lift_test
        coresident/test_h1_power_grasp_lift.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/unified_coresident_stepper.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_h1_power_grasp_lift_test nuka_codegen)

    target_compile_definitions(nuka_h1_power_grasp_lift_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_h1_power_grasp_lift_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_h1_power_grasp_lift_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_h1_power_grasp_lift_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 C7b-2b: the SCALED-CUP H1 power/wrap grasp VALIDATION spike ---
    # Tests ONE variable (the OWNER-chosen option-1 fix): scale the C7a cup UP to
    # ~10cm so it fills the H1 hand aperture + a PALM sphere can co-contact the
    # fingers, closing the ~110deg palm-side void the 6cm cup left open. Re-runs the
    # power-grasp machinery (Step Zero void-closure, table-removal LIFT, BITE, dynamic
    # load, D1) over a scale sweep. Additive parallel TU: the 6cm honest-negative
    # gates in test_h1_power_grasp_lift.cpp are LEFT INTACT. Same dep set.
    add_executable(nuka_h1_scaled_cup_grasp_test
        coresident/test_h1_scaled_cup_grasp.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/unified_coresident_stepper.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_h1_scaled_cup_grasp_test nuka_codegen)

    target_compile_definitions(nuka_h1_scaled_cup_grasp_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_h1_scaled_cup_grasp_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_h1_scaled_cup_grasp_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_h1_scaled_cup_grasp_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 C7b-2c: the DENSE-CONTACT H1 power/wrap grasp DISCRIMINATOR ---
    # Isolates CONTACT DENSITY: the four prior grasp spikes placed ONE sphere per
    # phalanx (~11 contacts), undersampling the finger/cylinder patch ~10x, so the only
    # way to hold was a deep (~7.2mm) passive wedge. This TU re-uses the validated ~10cm
    # scaled-cup machinery (Fixed base, real armature/damping, the same PD gains + table-
    # removal LIFT through the unified spine) but with a DENSE sphere-chain (~37 contacts)
    # + a SHALLOW (<=2mm) placement filter, and a UNIQUE broadphase_handle per sphere
    # (the stepper resolver picks the right geometry; the inert-by-default finger_link<-
    # handle decoupling keeps the chain-J on the real link). Additive parallel TU: the
    # prior honest-negative gates in test_h1_scaled_cup_grasp.cpp are LEFT INTACT. Same
    # dep set + the contact_stream_driver.cpp the grasp path needs.
    add_executable(nuka_h1_dense_grasp_test
        coresident/test_h1_dense_grasp.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/unified_coresident_stepper.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_h1_dense_grasp_test nuka_codegen)

    target_compile_definitions(nuka_h1_dense_grasp_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_h1_dense_grasp_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_h1_dense_grasp_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_h1_dense_grasp_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- H1.1: the CHEAP SCRIPTED force-closure FEASIBILITY PROBE (GO/NO-GO) ---
    # Answers ONE question for the RL-grasp track: does an HONEST force-closure grasp of
    # the cup EXIST for the cooked H1 hand + sphere fingertips? Runs the CONJOINT honest
    # gate -- in ONE scripted close, settle ACTIVE grip -> rotational disturbance ->
    # bounded tilt (HOLD-rotation) AND grip->0 -> cup drops (BITE honesty) -- swept over
    # cup{8.5,9.7,10.9}cm x wrap-depth{shallow,medium,deep} (the gap the prior 4 spikes
    # left: they ran the gate fragments separately, at one config). Reuses the validated
    # wrap/palm/curl scaffold (Fixed base, real armature 0.1/damping 1.0, PD close, table-
    # removal LIFT) from the scaled-cup spike; mass faithful at 0.2kg. Additive parallel
    # TU; the prior grasp gates are LEFT INTACT. Same dep set + contact_stream_driver.cpp.
    add_executable(nuka_h1_grasp_feasibility_probe_test
        coresident/test_h1_grasp_feasibility_probe.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/unified_coresident_stepper.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_h1_grasp_feasibility_probe_test nuka_codegen)

    target_compile_definitions(nuka_h1_grasp_feasibility_probe_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_h1_grasp_feasibility_probe_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_h1_grasp_feasibility_probe_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_h1_grasp_feasibility_probe_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 H1.2: the COOKED H1 HAND cup grasp WRAPPED INTO the batched general world.
    # Takes the H1.1-PROVEN ~10.9cm finger-only-wrap grasp (the feasibility probe's scene)
    # and runs it through the BATCHED general world (BatchedUnifiedWorld) at N envs, then
    # proves the batched H1-hand grasp is BYTE-EXACT (1e-5) vs the N=1 co-resident oracle
    # (UnifiedCoResidentStepper) on the IDENTICAL scene -- the bigger H1-hand articulation
    # (~18 links, dof_stride ~12, ~30 fingertips, ~11 contacts) is the surface where the
    # env-major batching machinery (M^-1 tile @ art_index*dof_stride^2, chain-J over GLOBAL
    # contact links) must generalize. Gates: G1 N=1 byte-exact parity (cup pose/vel +
    # gripper qdot), G2 multi-env independence (per-env / mixed / D1), a throughput lower-
    # bound sweep. Compiles the SAME batched_unified_world.cpp + unified_coresident_stepper.cpp
    # + contact_stream_driver.cpp as nuka_batched_unified_world_test (accidental physics
    # changes there surface in the synthetic 17 gates). Asset-gated SKIP. ADDITIVE: a NEW
    # parallel TU; the feasibility probe + synthetic batched gates are LEFT INTACT.
    add_executable(nuka_batched_h1_hand_grasp_test
        coresident/test_batched_h1_hand_grasp.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/batched_unified_world.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/unified_coresident_stepper.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_batched_h1_hand_grasp_test nuka_codegen)

    target_compile_definitions(nuka_batched_h1_hand_grasp_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_batched_h1_hand_grasp_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_batched_h1_hand_grasp_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_batched_h1_hand_grasp_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- SG G1: the BATCHED UNION WORLD parity TU (G1a floating-base, NO contact) ---
    # G1a proves BatchedUnifiedWorld's env-major machinery batches the FLOATING-BASE
    # whole-body H1 (nuka::demo::LoadFloating, 51 DOF = 6 base + 45 revolute) BYTE-
    # EXACT (tol 0) vs the N=1 UnifiedCoResidentStepper oracle on the IDENTICAL proto,
    # with ZERO contact rows (the cup is parked 5 m away and free-falls alongside) --
    # isolating floating-root replication, per-env IntegrateFloatingBase* activation,
    # and the SetActions seam over a dof_stride with 6 (dead) base columns. Gates:
    # (a) N=1 byte-exact over 120 driven free-fall steps, (b) N=8 mixed-action
    # independence, (c) D1 two-run identity, (d) base action columns 0..5 dead.
    # G1b..G1d (feet/finger/table union rows) will EXTEND this TU. Asset-gated SKIP.
    # ADDITIVE parallel TU mirroring the nuka_batched_h1_hand_grasp_test dep set.
    add_executable(nuka_batched_h1_union_world_test
        coresident/test_batched_h1_union_world.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/batched_unified_world.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/unified_coresident_stepper.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_batched_h1_union_world_test nuka_codegen)

    target_compile_definitions(nuka_batched_h1_union_world_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_batched_h1_union_world_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_batched_h1_union_world_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_batched_h1_union_world_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- SG G2: the PRODUCTION union-scene-factory EQUALITY gate -------------------
    # src/runtime/coresident/h1_union_scene_factory.cpp reproduces the test-side
    # G1c/G1d authoring (h1_union_scene_shared.hpp) VERBATIM so the C-ABI union
    # world can construct it from libnuka. This gate pins the promotion: the
    # factory's BatchedSceneTemplate (incl. the independently-run 200-step settle
    # pre-roll) must be BYTE-IDENTICAL to the shared-header authoring. Compiles the
    # factory + the same coresident sources as the union TU. Asset-gated SKIP.
    add_executable(nuka_h1_union_scene_factory_test
        coresident/test_h1_union_scene_factory.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/h1_union_scene_factory.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/batched_unified_world.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/unified_coresident_stepper.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_h1_union_scene_factory_test nuka_codegen)

    target_compile_definitions(nuka_h1_union_scene_factory_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_h1_union_scene_factory_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_h1_union_scene_factory_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_h1_union_scene_factory_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- SG G2: the C++ REFERENCE RECORDER for the python byte-parity gate ---------
    # Drives the union world THROUGH THE C ABI (links libnuka -- the exact code path
    # the python binding calls, so the python replay has no cross-compile seam) with
    # the closed-loop lift choreography and records per-step action bytes + the
    # C-ABI obs export to .g2_logs/g2_pyref_n4.bin. The python gate harness
    # (examples/training/g2_union_world_gates.py) runs this binary then replays.
    if(TARGET nuka)
        add_executable(nuka_h1_union_pyref_test
            coresident/test_h1_union_pyref.cpp
        )

        target_include_directories(nuka_h1_union_pyref_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/src/include
        )

        target_link_libraries(nuka_h1_union_pyref_test
            PRIVATE
                nuka
                GTest::gtest_main
        )

        gtest_discover_tests(nuka_h1_union_pyref_test
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()

    # --- v0.8 C7b-2 SHOWCASE: the H1 cup-sequence DEMO scaffold (R0/R1) ---
    # The conjunction gate: a floating-base H1 STANDS (RL leg policy + hold-PD) WHILE
    # the right hand co-resides a movable cup + static table + N fingertip collidables
    # -- all in ONE step through the NEW StepStandGrasp() path (foot<->ground +
    # finger<->cup + cup<->table). R0 = pure stand through the conjunction path (cup
    # plumbing inert) proves no stand regression; R1 = cup on a table + resting fingers
    # proves co-residence does not break the stand solve. Reuses the proven bridge-spike
    # / dense-grasp helpers via h1_demo_shared.hpp + tiny_mlp.hpp (the coresident/ dir).
    # Additive parallel TU; the bridge spike + dense grasp gates are LEFT INTACT. Same
    # dep set + the contact_stream_driver.cpp the contact path needs.
    add_executable(nuka_h1_cup_sequence_demo_test
        coresident/test_h1_cup_sequence_demo.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/unified_coresident_stepper.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_h1_cup_sequence_demo_test nuka_codegen)

    target_compile_definitions(nuka_h1_cup_sequence_demo_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_h1_cup_sequence_demo_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/coresident
    )

    target_link_libraries(nuka_h1_cup_sequence_demo_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_h1_cup_sequence_demo_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 S3: the TRANSFER-SMOKE GATE (full-DOF floating-base H1 STANDING) ---
    # Cooks the FULL-DOF FLOATING-BASE h1_with_hand (skip mesh decompose, do NOT
    # force-fix the base), seats it at the bent stance on 4 foot spheres, and runs the
    # PORTED h1_balance_gate.py controller (CoP ankle strategy + posture PD on legs +
    # held torso/arms/fingers, clamped to the MJCF limits) through the NEW StepStand()
    # path (N foot spheres <-> static ground, condim=3 friction, art<->static rows). The
    # S3 gate: does the SAME controller that stood the batched reduced-DOF H1 ALSO stand
    # the full-DOF articulated H1 in the compliant deploy world? Additive parallel TU;
    # the grasp gates in test_h1_dense_grasp.cpp are LEFT INTACT. Same dep set + the
    # contact_stream_driver.cpp the contact path needs.
    add_executable(nuka_h1_costand_transfer_test
        coresident/test_h1_costand_transfer.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/unified_coresident_stepper.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_h1_costand_transfer_test nuka_codegen)

    target_compile_definitions(nuka_h1_costand_transfer_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_h1_costand_transfer_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_h1_costand_transfer_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_h1_costand_transfer_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 POLICY->CO-RESIDENT INFERENCE BRIDGE SPIKE -----------------------
    # The GATE before a multi-week PPO standing campaign: prove the deploy SEAM
    # (PPO trains a torch NN in the BATCHED reduced-16-DOF world; the demo runs it
    # in the CO-RESIDENT full-51-DOF C++ world via flat f32 weights + a ~50-line C++
    # forward, NOT python-bind/ONNX/torch-at-deploy). Piece A = structural leg-index
    # correspondence (reduced 1..10 == full by-name); Gate 1 = EXACT obs+forward
    # parity (1e-5) vs the python MLP from bridge_export.py + its anti-green-wash
    # BITE (permutation + transpose); Gate 2 (SOFT) = closed-loop MLP stand in the
    # full world. Mirrors nuka_h1_costand_transfer_test's dep set + compiles the
    # production stepper + the host contact_stream_driver.cpp (EDITS nothing).
    add_executable(nuka_h1_bridge_spike_test
        coresident/test_h1_bridge_spike.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coresident/unified_coresident_stepper.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_h1_bridge_spike_test nuka_codegen)

    target_compile_definitions(nuka_h1_bridge_spike_test
        PRIVATE
            NUKA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    target_include_directories(nuka_h1_bridge_spike_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/coresident
    )

    target_link_libraries(nuka_h1_bridge_spike_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_runtime
            nuka_import
            nuka_cooker
            nuka_articulation_gpu
            nuka_collision_gpu
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_h1_bridge_spike_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 C6b: cross-system coupling-row framework + ParticleInvMass arm ---
    # The K2 proving pair (a particle pushes a free rigid body two-way through one
    # unified compliant solve). CASE 1 is the closed-form Jv discriminator (the
    # first exercise of the ParticleInvMass arm; the broken-Jv variant differs >>
    # tol). CASE 2/3 prove EmitCouplingRows (re-class to id13 + end-to-end two-way).
    # Header-only framework (coupling_row_framework.hpp) + hand-built manifolds, so
    # no broadphase/narrowphase libs -- just the solver + constraint emitter; needs
    # nuka_codegen for the generated row-class registry (kCouplingContactRowId).
    add_executable(nuka_coupling_test
        coupling/test_coupling_framework.cpp
    )
    add_dependencies(nuka_coupling_test nuka_codegen)

    target_include_directories(nuka_coupling_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_coupling_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_coupling_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # --- v0.8 C6a: the PBD<->rigid unified-contact CO-STEP bridge -------------
    # Drives the REAL front-end together (C2c broadphase -> C5 contact driver ->
    # C6b coupling framework -> C6b unified solver) for the Genesis-style "couple"
    # phase: a PBD particle system consumes unified manifolds + reacts TWO-WAY
    # against rigid bodies through ONE compliant solve. CASE A: multi-particle drop
    # onto a heavy free box, TOTAL-energy (KE+gravPE) non-increasing + settle +
    # two-way. CASE B: the mis-order bite-check (double-applied couple injects
    # energy -> the gate must FAIL). CASE C: fluid rest on a static floor. CASE D:
    # D1. Compiles the bridge (unified_costep.cpp) + the contact driver
    # (contact_stream_driver.cpp) directly, like nuka_foot_box_coresidence_test;
    # needs nuka_codegen for the generated row-class registry.
    add_executable(nuka_pbd_costep_test
        coupling/test_pbd_costep_unified.cpp
        ${CMAKE_SOURCE_DIR}/src/runtime/coupling/unified_costep.cpp
        ${CMAKE_SOURCE_DIR}/src/collision/contact_stream_driver.cpp
    )
    add_dependencies(nuka_pbd_costep_test nuka_codegen)

    target_include_directories(nuka_pbd_costep_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(nuka_pbd_costep_test
        PRIVATE
            nuka_solver
            nuka_constraint
            nuka_collision_gpu
            nuka_runtime
            nuka_core
            nuka_phi
            GTest::gtest_main
    )

    gtest_discover_tests(nuka_pbd_costep_test
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

add_executable(nuka_cuda_perf_test
    perf/test_cuda_solver_timing.cpp
    perf/test_lbvh_50k_bodies.cpp
    perf/test_1m_particles_grid_perf.cpp
)

    target_link_libraries(nuka_cuda_perf_test
        PRIVATE
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
