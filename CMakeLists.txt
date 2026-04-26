cmake_minimum_required(VERSION 3.24)
project(nuka-physics
    VERSION 0.1.0
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Guard against in-source builds
include(cmake/PreventInSourceBuilds.cmake)

# Compiler-warning helper
include(cmake/Warnings.cmake)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(NK_BUILD_TESTS "Build unit / smoke tests" ON)

# ---------------------------------------------------------------------------
# CUDA (optional – some early tasks are CPU-only)
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Auto-detect GPU architecture; fall back to broad set
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    message(STATUS "CUDA enabled – compiler: ${CMAKE_CUDA_COMPILER}")
else()
    message(STATUS "CUDA compiler not found – CUDA targets will be skipped")
endif()

# ---------------------------------------------------------------------------
# Third-party dependencies
# ---------------------------------------------------------------------------
if(NK_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
    )
    # Prevent GoogleTest from overriding our compiler/linker options on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(src)

if(NK_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
