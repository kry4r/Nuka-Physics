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
option(NK_BUILD_EDITOR "Build the editor DLL and its launcher" OFF)
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
    # One shared CUDA runtime, so streams and events cross module boundaries.
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
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
    message(STATUS "CUDA backend disabled; building host components and the backend-independent physics core")
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
    # v0.8 C6b: the cross-system coupling-row class (id 6). FORWARD-ONLY (no
    # adjoint_evaluator in its IR, like the id4/id5 SDF rows).
    "${NUKA_CODEGEN_OUTPUT_DIR}/coupling_contact_forward.cu"
    "${NUKA_CODEGEN_OUTPUT_DIR}/row_dispatch.cu"
)
set(NUKA_CODEGEN_GENERATED_HEADERS
    "${NUKA_CODEGEN_OUTPUT_DIR}/row_class_registry.hpp"
    "${NUKA_CODEGEN_OUTPUT_DIR}/maximal_contact_adjoint.cuh"
    "${NUKA_CODEGEN_OUTPUT_DIR}/maximal_joint_adjoint.cuh"
    "${NUKA_CODEGEN_OUTPUT_DIR}/maximal_drive_adjoint.cuh"
    "${NUKA_CODEGEN_OUTPUT_DIR}/featherstone_contact_adjoint.cuh"
    # v0.8 C2a: the collidable-type registry header (enum CollidableType + static
    # metadata table). Emitted by regen.py's SECOND, parallel collidable pass.
    "${NUKA_CODEGEN_OUTPUT_DIR}/collidable_registry.hpp"
    "${CMAKE_SOURCE_DIR}/src/nk/model/generated/field_ids.hpp"
    "${CMAKE_SOURCE_DIR}/src/nk/model/generated/views.hpp"
    "${CMAKE_SOURCE_DIR}/src/nk/model/generated/arena_layout.hpp"
    "${CMAKE_SOURCE_DIR}/src/nk/model/generated/dlpack_table.hpp"
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
        "${CMAKE_SOURCE_DIR}/tools/codegen/fields/gen_fields.py"
        "${CMAKE_SOURCE_DIR}/src/nk/model/fields.yaml"
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

include(cmake/NukaOpenUSD.cmake)

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

# --- stb_image / stb_image_write: image decode for the beauty texture pipeline
# Public-domain single-header decoders under external/stb/ (see the license text
# at the foot of each header). Header-only INTERFACE target; ONE TU
# (render/texture_image.cpp) defines STB_IMAGE_IMPLEMENTATION.
add_library(nuka_stb INTERFACE)
target_include_directories(nuka_stb SYSTEM INTERFACE
    ${CMAKE_SOURCE_DIR}/external)

# --- toml++ 3.4.0 (MIT, external/tomlplusplus/LICENSE): editor extension manifests and UI
# string tables. Header-only INTERFACE target.
add_library(nuka_toml INTERFACE)
target_include_directories(nuka_toml SYSTEM INTERFACE
    ${CMAKE_SOURCE_DIR}/external/tomlplusplus)

# --- cgltf 1.15 (MIT, external/cgltf/LICENSE): glTF 2.0 mesh import. Header-only INTERFACE
# target; ONE TU (import/gltf_mesh_loader.cpp) defines CGLTF_IMPLEMENTATION.
add_library(nuka_cgltf INTERFACE)
target_include_directories(nuka_cgltf SYSTEM INTERFACE
    ${CMAKE_SOURCE_DIR}/external/cgltf)

# --- Dear ImGui + SDL3: vendored ImGui (external/imgui/NUKA_VENDOR.md) and the SDL3
# window layer, built only for the editor or Vulkan validation so the core build never fetches them.
if(NK_BUILD_EDITOR OR NK_BUILD_VULKAN_VALIDATION)
    # SDL3 is the window, input and Vulkan-surface layer on every platform; it also
    # backs the imgui_impl_sdl3 platform backend (IME, gamepads, Wayland and X11).
    FetchContent_Declare(
        SDL3
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG        release-3.4.18
        GIT_SHALLOW    TRUE
    )
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
    set(SDL_STATIC_PIC ON CACHE BOOL "" FORCE)
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
    set(SDL_TESTS OFF CACHE BOOL "" FORCE)
    set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SDL_INSTALL OFF CACHE BOOL "" FORCE)
    set(SDL_DISABLE_INSTALL ON CACHE BOOL "" FORCE)
    set(SDL_RENDER OFF CACHE BOOL "" FORCE)
    set(SDL_GPU OFF CACHE BOOL "" FORCE)
    set(SDL_OPENGL OFF CACHE BOOL "" FORCE)
    set(SDL_OPENGLES OFF CACHE BOOL "" FORCE)
    set(SDL_CAMERA OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(SDL3)

    set(NUKA_IMGUI_SOURCES
        ${CMAKE_SOURCE_DIR}/external/imgui/imgui.cpp
        ${CMAKE_SOURCE_DIR}/external/imgui/imgui_draw.cpp
        ${CMAKE_SOURCE_DIR}/external/imgui/imgui_tables.cpp
        ${CMAKE_SOURCE_DIR}/external/imgui/imgui_widgets.cpp
        ${CMAKE_SOURCE_DIR}/external/imgui/imgui_demo.cpp
        ${CMAKE_SOURCE_DIR}/external/imgui/backends/imgui_impl_vulkan.cpp
        ${CMAKE_SOURCE_DIR}/external/imgui/backends/imgui_impl_sdl3.cpp
        # ImGuizmo (MIT, vendored): the in-viewport transform gizmo. One TU; it
        # links into the same imgui static lib and shares its SYSTEM include scope.
        ${CMAKE_SOURCE_DIR}/external/imguizmo/ImGuizmo.cpp
        # ImPlot (MIT, vendored): live curves in the editor panels.
        ${CMAKE_SOURCE_DIR}/external/implot/implot.cpp
        ${CMAKE_SOURCE_DIR}/external/implot/implot_items.cpp
        # ImGuiColorTextEdit (MIT, vendored): the script editor with Python highlighting.
        ${CMAKE_SOURCE_DIR}/external/imgui_color_text_edit/TextEditor.cpp
        # The item hooks the editor's demo pointer finds widgets by label with.
        ${CMAKE_SOURCE_DIR}/src/render/imgui/imgui_item_hooks.cpp
    )
    add_library(imgui STATIC ${NUKA_IMGUI_SOURCES})
    target_compile_definitions(imgui PUBLIC IMGUI_USE_WCHAR32)
    # ImGui reports each item to the hooks while they record and skips them otherwise.
    target_compile_definitions(imgui PRIVATE IMGUI_ENABLE_TEST_ENGINE)
    # TextEditor.cpp carries UTF-8 comments; MSVC would read them in the system code page.
    if(MSVC)
        target_compile_options(imgui PRIVATE /utf-8)
    endif()
    # SYSTEM include keeps ImGui warnings out of -Werror; backends include "imgui.h"
    # from the root and callers use "backends/...", so both directories are on the path.
    target_include_directories(imgui SYSTEM PUBLIC
        ${CMAKE_SOURCE_DIR}/external/imgui
        ${CMAKE_SOURCE_DIR}/external/imgui/backends
        ${CMAKE_SOURCE_DIR}/external/imguizmo
        ${CMAKE_SOURCE_DIR}/external/implot
        ${CMAKE_SOURCE_DIR}/external/imgui_color_text_edit)
    target_link_libraries(imgui PUBLIC Vulkan::Vulkan SDL3::SDL3-static)
    set_target_properties(imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # The Dear ImGui core alone, for editor plugins that draw into the editor's context.
    add_library(nuka_imgui_plugin STATIC
        ${CMAKE_SOURCE_DIR}/external/imgui/imgui.cpp
        ${CMAKE_SOURCE_DIR}/external/imgui/imgui_draw.cpp
        ${CMAKE_SOURCE_DIR}/external/imgui/imgui_tables.cpp
        ${CMAKE_SOURCE_DIR}/external/imgui/imgui_widgets.cpp)
    target_compile_definitions(nuka_imgui_plugin PUBLIC IMGUI_USE_WCHAR32)
    target_include_directories(nuka_imgui_plugin SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/external/imgui)
    set_target_properties(nuka_imgui_plugin PROPERTIES POSITION_INDEPENDENT_CODE ON)
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

if(NK_BUILD_EDITOR)
    include(cmake/NukaEditorExtension.cmake)
    add_subdirectory(examples/editor_extensions)
endif()

if(WIN32 AND CMAKE_CUDA_COMPILER)
    # Executables load the shared CUDA runtime from their own directory.
    file(GLOB NUKA_CUDART_DLLS "${CUDAToolkit_BIN_DIR}/cudart64_*.dll"
                               "${CUDAToolkit_BIN_DIR}/x64/cudart64_*.dll")
    file(COPY ${NUKA_CUDART_DLLS} DESTINATION "${CMAKE_BINARY_DIR}/src")
    if(NK_BUILD_TESTS)
        file(COPY ${NUKA_CUDART_DLLS} DESTINATION "${CMAKE_BINARY_DIR}/tests")
    endif()
endif()

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
