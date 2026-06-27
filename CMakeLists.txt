cmake_minimum_required(VERSION 3.24)
project(nuka-physics
    VERSION 0.1.0
    LANGUAGES C CXX
)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# MSVC's traditional preprocessor is non-conforming; CUDA 13 CCCL headers reject
# it. Use the conforming preprocessor for both the C++ and CUDA host passes.
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()

# Guard against in-source builds
include(cmake/PreventInSourceBuilds.cmake)

# Compiler-warning helper
include(cmake/Warnings.cmake)
include(CMakePackageConfigHelpers)
include(CheckCXXSourceCompiles)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(NK_BUILD_TESTS "Build unit / smoke tests" ON)
option(NK_BUILD_VULKAN_VALIDATION "Build Vulkan-specific validation and timing tests" OFF)
option(NK_BUILD_DEMO_TIMING_VALIDATION "Build scene demo render timing tests" OFF)
option(NK_REQUIRE_CUDA "Require CUDA support for the preferred production physics backend" ON)
set(NK_PHYSICS_BACKEND "CUDA" CACHE STRING "Preferred production physics backend")
set_property(CACHE NK_PHYSICS_BACKEND PROPERTY STRINGS CUDA CPU_REFERENCE)
set(NK_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures for production physics kernels")

# Probe the project's nuka::expected wrapper contract under the configured C++
# standard (C++20). libstdc++ only exposes std::expected at C++23, so nuka.hpp
# ships a C++20 fallback in nuka/expected.hpp; this confirms that wrapper builds
# before enabling the RAII wrapper test.
set(CMAKE_REQUIRED_INCLUDES "${CMAKE_SOURCE_DIR}/src/include")
check_cxx_source_compiles(
    "#include <nuka/expected.hpp>
struct E { int code; };
int main() {
    nuka::expected<int, E> ok = 1;
    if (!ok.has_value() || *ok != 1) return 1;
    nuka::expected<int, E> bad = nuka::unexpected(E{2});
    if (bad.has_value() || bad.error().code != 2) return 1;
    nuka::expected<void, E> done{};
    return done.has_value() ? 0 : 1;
}"
    NUKA_HAS_EXPECTED_WRAPPER
)
unset(CMAKE_REQUIRED_INCLUDES)

# ---------------------------------------------------------------------------
# CUDA-preferred physics backend and Vulkan renderer backend
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NK_CUDA_ARCHITECTURES STREQUAL "native"
       AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.8)
        set(NK_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures for production physics kernels" FORCE)
        message(STATUS "CUDA ${CMAKE_CUDA_COMPILER_VERSION} does not support native Ada targets; using sm_86")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${NK_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures to build" FORCE)
    message(STATUS "CUDA enabled for preferred physics backend: ${CMAKE_CUDA_COMPILER}")
else()
    if(NK_REQUIRE_CUDA OR NK_PHYSICS_BACKEND STREQUAL "CUDA")
        message(FATAL_ERROR "CUDA compiler not found, but CUDA is the required/preferred production physics backend.")
    endif()
    message(STATUS "CUDA compiler not found; building CPU reference backend only")
endif()

find_package(Vulkan REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(NUKA_CODEGEN_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/src/codegen/generated")
set(NUKA_CODEGEN_GENERATED_SOURCES
    "${NUKA_CODEGEN_OUTPUT_DIR}/maximal_contact_forward.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/maximal_contact_adjoint.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/maximal_joint_forward.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/maximal_joint_adjoint.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/maximal_drive_forward.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/maximal_drive_adjoint.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/featherstone_contact_forward.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/featherstone_contact_adjoint.cu"
    # v0.7 p08-B: SDF contact rows (ids 4/5). FORWARD ONLY -- no _adjoint.* until
    # p08-C adds the IFT-at-convergence adjoint_evaluator to their IR.
    "${NUKA_CODEGEN_OUTPUT_DIR}/rigid_sdf_contact_forward.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/featherstone_sdf_contact_forward.cu"
    # v0.7 p09-A: XPBD soft-body distance row (id 6). Ships a GENUINE dispatchable
    # per-row adjoint (dense_adjoint), so BOTH the forward and the adjoint kernel
    # __global__ symbols must be compiled into nuka_codegen_generated (mirror the
    # maximal rows' _forward.cu + _adjoint.cu pairing -- NOT the forward-only SDF
    # rows). Omitting the adjoint .cu would leave RowClassAdjointKernelSymbol(6)
    # advertising an uncompiled symbol (a non-dispatchable orphan).
    "${NUKA_CODEGEN_OUTPUT_DIR}/xpbd_distance_forward.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/xpbd_distance_adjoint.cu"
    # v0.7 p09-B: XPBD soft-body BEND (id 7) + VOLUME (id 8) rows. Like the
    # distance row each ships a GENUINE dispatchable per-row adjoint, so BOTH the
    # forward and adjoint __global__ symbols must be compiled into
    # nuka_codegen_generated (regen alone leaves the adjoint .cu orphaned -- the
    # p09-A lesson). Mirror the _forward.cu + _adjoint.cu pairing.
    "${NUKA_CODEGEN_OUTPUT_DIR}/xpbd_bend_forward.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/xpbd_bend_adjoint.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/xpbd_volume_forward.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/xpbd_volume_adjoint.cu"
    # v0.7 p09-C: XPBD soft-body SHAPE-MATCH (id 9). Ships a GENUINE dispatchable
    # per-row adjoint (the per-particle goal-projection law), so BOTH the forward
    # and adjoint __global__ symbols must compile into nuka_codegen_generated
    # (regen alone leaves the adjoint .cu orphaned -- the p09-A lesson).
    "${NUKA_CODEGEN_OUTPUT_DIR}/xpbd_shape_match_forward.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/xpbd_shape_match_adjoint.cu"
    # v0.7 p11 (K3): cross-system particle-particle CONTACT row (id 10). Ships a
    # GENUINE dispatchable per-row adjoint (the multilinear XPBD multiplier law, as
    # the distance row), so BOTH the forward and adjoint __global__ symbols must
    # compile into nuka_codegen_generated (regen alone leaves the adjoint .cu
    # orphaned -- the p09-A lesson). Mirror the _forward.cu + _adjoint.cu pairing.
    "${NUKA_CODEGEN_OUTPUT_DIR}/particle_particle_contact_forward.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/particle_particle_contact_adjoint.cu"
    # v0.8 C6b: ids 11/12 RESERVED placeholders (v0.9 Cosserat R2a/R2b) + id13 the
    # cross-system coupling-row class. All THREE are FORWARD-ONLY (no adjoint_evaluator
    # in their IR -> regen emits no _adjoint.* for them, like the id4/id5 SDF rows), so
    # only the _forward.cu stubs compile into nuka_codegen_generated. They reserve the
    # registry symbols (kRowClassCount=14); v0.8 never EMITS rows of 11/12.
    "${NUKA_CODEGEN_OUTPUT_DIR}/xpbd_cosserat_stretch_shear_forward.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/xpbd_cosserat_bend_twist_forward.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/coupling_contact_forward.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/row_dispatch.cu"
)
set(NUKA_CODEGEN_GENERATED_HEADERS
    "${NUKA_CODEGEN_OUTPUT_DIR}/row_class_registry.hpp"
    "${NUKA_CODEGEN_OUTPUT_DIR}/maximal_contact_adjoint.cuh"
    "${NUKA_CODEGEN_OUTPUT_DIR}/maximal_joint_adjoint.cuh"
    "${NUKA_CODEGEN_OUTPUT_DIR}/maximal_drive_adjoint.cuh"
    "${NUKA_CODEGEN_OUTPUT_DIR}/featherstone_contact_adjoint.cuh"
    "${NUKA_CODEGEN_OUTPUT_DIR}/xpbd_distance_adjoint.cuh"
    # v0.7 p09-B: XPBD bend (id 7) + volume (id 8) adjoint headers.
    "${NUKA_CODEGEN_OUTPUT_DIR}/xpbd_bend_adjoint.cuh"
    "${NUKA_CODEGEN_OUTPUT_DIR}/xpbd_volume_adjoint.cuh"
    # v0.7 p09-C: XPBD shape-match (id 9) adjoint header.
    "${NUKA_CODEGEN_OUTPUT_DIR}/xpbd_shape_match_adjoint.cuh"
    # v0.7 p11 (K3): particle-particle contact (id 10) adjoint header.
    "${NUKA_CODEGEN_OUTPUT_DIR}/particle_particle_contact_adjoint.cuh"
    # v0.8 C2a: the collidable-type registry header (enum CollidableType + static
    # metadata table). Emitted by regen.py's SECOND, parallel collidable pass.
    "${NUKA_CODEGEN_OUTPUT_DIR}/collidable_registry.hpp"
)
set(NUKA_CODEGEN_OUTPUTS
    ${NUKA_CODEGEN_GENERATED_SOURCES}
    ${NUKA_CODEGEN_GENERATED_HEADERS}
)
file(GLOB NUKA_ROW_IR_FILES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/tools/codegen/classes/*.yaml"
)
# v0.8 C2a: collidable class YAMLs live in a SUBDIRECTORY (the row glob above is
# non-recursive on purpose, so the row loader never sees them). Glob them too so
# the codegen command re-runs when a collidable type is added/edited.
file(GLOB NUKA_COLLIDABLE_IR_FILES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/tools/codegen/classes/collidable/*.yaml"
)
file(GLOB NUKA_ROW_TEMPLATES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/tools/codegen/templates/*.j2"
)
add_custom_command(
    OUTPUT ${NUKA_CODEGEN_OUTPUTS}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${NUKA_CODEGEN_OUTPUT_DIR}"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tools/codegen/regen.py"
    DEPENDS
        ${NUKA_ROW_IR_FILES}
        ${NUKA_COLLIDABLE_IR_FILES}
        ${NUKA_ROW_TEMPLATES}
        "${CMAKE_SOURCE_DIR}/tools/codegen/schema/row_v0_1.yaml"
        "${CMAKE_SOURCE_DIR}/tools/codegen/schema/collidable_v0_1.yaml"
        "${CMAKE_SOURCE_DIR}/tools/codegen/regen.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Regenerating row class CUDA kernels from IR"
    VERBATIM
)
add_custom_target(nuka_codegen ALL
    DEPENDS ${NUKA_CODEGEN_OUTPUTS}
)

add_custom_target(nuka_run_lint
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/lint/physics_smell.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running physics-smell lint"
)

# ---------------------------------------------------------------------------
# Third-party dependencies
# ---------------------------------------------------------------------------
include(FetchContent)

# tinyxml2 (needed by importers)
FetchContent_Declare(
    tinyxml2
    GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
    GIT_TAG        10.0.0
)
set(tinyxml2_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tinyxml2)

# --- V-HACD (v0.7 p06): cook-time convex decomposition -----------------------
# Vendored as the single-header v4.1 release under external/vhacd/ (BSD
# 3-Clause; see external/vhacd/LICENSE). The implementation is compiled into a
# single TU (src/import/cooker/convex_decomposition.cpp, which defines
# ENABLE_VHACD_IMPLEMENTATION). Exposed here only as an INTERFACE include path.
# COOK-TIME / HOST-CPU code (master plan §5.6 sanctioned exception).
add_library(nuka_vhacd INTERFACE)
target_include_directories(nuka_vhacd SYSTEM INTERFACE
    ${CMAKE_SOURCE_DIR}/external/vhacd)

# --- SHA-256 (v0.7 p06): content-hash cache key ------------------------------
# Public-domain single-header implementation under external/sha256/ (see
# external/sha256/LICENSE). Header-only INTERFACE target.
add_library(nuka_sha256 INTERFACE)
target_include_directories(nuka_sha256 SYSTEM INTERFACE
    ${CMAKE_SOURCE_DIR}/external/sha256)

# --- Dear ImGui (M8.5 T1): the realtime viewport UI --------------------------
# Vendored COPIED sources under external/imgui/ (mirror external/vhacd; NOT a
# submodule) at the pinned tag v1.92.8-docking (see external/imgui/NUKA_VENDOR.md
# for source/SHA/font/license). Built into the `imgui` static lib here and linked
# by `nuka_imgui` (src/render/imgui/) -- the nuka-side glue + custom theme.
#
# GATED (Decision D3): compiled ONLY when NK_BUILD_VULKAN_VALIDATION is ON. The
# default build-cuda128 config (flag OFF) never adds this target, never finds X11,
# and builds libnuka + the core completely unchanged. The vendored ImGui is a
# SYSTEM include everywhere so its warnings are suppressed and it stays OUT of the
# zero-CUDA `src/render/**` lint scope. HOST-ONLY C++/Vulkan (no CUDA).
if(NK_BUILD_VULKAN_VALIDATION)
    # xcb/X11 windowing dep for the Linux swapchain present backend. Windows uses a
    # Win32 surface and the offscreen render path needs no xcb, so the find is skipped.
    if(NOT WIN32)
        find_package(X11 REQUIRED)
        if(NOT X11_xcb_FOUND AND NOT EXISTS "/usr/include/xcb/xcb.h")
            message(FATAL_ERROR
                "M8.5 viewport needs libxcb dev headers (/usr/include/xcb/xcb.h); "
                "X11 found but xcb component missing.")
        endif()
        message(STATUS "M8.5: ImGui viewport gated ON (NK_BUILD_VULKAN_VALIDATION); "
                       "X11 at ${X11_INCLUDE_DIR}, xcb header present.")
    endif()

    add_library(imgui STATIC
        ${CMAKE_SOURCE_DIR}/external/imgui/imgui.cpp
        ${CMAKE_SOURCE_DIR}/external/imgui/imgui_draw.cpp
        ${CMAKE_SOURCE_DIR}/external/imgui/imgui_tables.cpp
        ${CMAKE_SOURCE_DIR}/external/imgui/imgui_widgets.cpp
        ${CMAKE_SOURCE_DIR}/external/imgui/imgui_demo.cpp
        ${CMAKE_SOURCE_DIR}/external/imgui/backends/imgui_impl_vulkan.cpp
    )
    # SYSTEM include so ImGui's own warnings never reach -Werror and its sources are
    # not lint-scoped. The backend includes "imgui.h" relative to the imgui root and
    # "backends/imgui_impl_vulkan.h", so both external/imgui and external/imgui itself
    # are on the path.
    target_include_directories(imgui SYSTEM PUBLIC
        ${CMAKE_SOURCE_DIR}/external/imgui
        ${CMAKE_SOURCE_DIR}/external/imgui/backends)
    target_link_libraries(imgui PUBLIC Vulkan::Vulkan)
    set_target_properties(imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(NK_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
    )
    # Prevent GoogleTest from overriding our compiler/linker options on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # -- Eigen (v0.5 p03): TEST-ONLY dense LDLT reference for the self-written
    #    CG sparse solver. Header-only; pinned at 3.4.0. We FETCH but DO NOT
    #    add_subdirectory()/MakeAvailable() Eigen's CMake project: that would
    #    register install rules / an `Eigen3::Eigen` target that could leak into
    #    libnuka or any production target. Instead we populate the sources and
    #    expose ONLY the include dir via a private INTERFACE target consumed by
    #    the single diffsim CG test executable (see tests/CMakeLists.txt). Eigen
    #    never appears in any production link line.
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_GetProperties(eigen)
    if(NOT eigen_POPULATED)
        FetchContent_Populate(eigen)
    endif()
    add_library(nuka_eigen_testonly INTERFACE)
    target_include_directories(nuka_eigen_testonly SYSTEM INTERFACE
        ${eigen_SOURCE_DIR})
endif()

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(src)

configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/nukaConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/nukaConfig.cmake"
    INSTALL_DESTINATION lib/cmake/nuka
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/nukaConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

if(NK_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/nukaConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/nukaConfigVersion.cmake"
    DESTINATION lib/cmake/nuka
)
