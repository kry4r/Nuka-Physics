cmake_minimum_required(VERSION 3.18...3.30)
project(nuka_ext LANGUAGES CXX)

# --- Engine locations (override with -DNUKA_ROOT / -DNUKA_BUILD_DIR) ----------
# NUKA_ROOT       : repo root that holds src/include/nuka/nuka.h
# NUKA_BUILD_DIR  : the engine build dir that holds src/libnuka.so
# CUDA_LIB_DIR    : the libcudart.so.12 dir (same runtime libnuka.so links)
if(NOT DEFINED NUKA_ROOT)
  set(NUKA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()
if(NOT DEFINED NUKA_BUILD_DIR)
  set(NUKA_BUILD_DIR "${NUKA_ROOT}/build-cuda128")
endif()
if(NOT DEFINED CUDA_LIB_DIR)
  set(CUDA_LIB_DIR "/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64")
endif()

set(NUKA_INCLUDE_DIR "${NUKA_ROOT}/src/include")
set(NUKA_LIB "${NUKA_BUILD_DIR}/src/libnuka.so")
set(CUDART_LIB "${CUDA_LIB_DIR}/libcudart.so")

if(NOT EXISTS "${NUKA_INCLUDE_DIR}/nuka/nuka.h")
  message(FATAL_ERROR "Cannot find nuka.h at ${NUKA_INCLUDE_DIR}/nuka/nuka.h "
                      "(set -DNUKA_ROOT=/path/to/repo)")
endif()
if(NOT EXISTS "${NUKA_LIB}")
  message(FATAL_ERROR "Cannot find libnuka.so at ${NUKA_LIB} "
                      "(build the engine first, or set -DNUKA_BUILD_DIR=...)")
endif()

# --- Toolchain / ABI ----------------------------------------------------------
# CRITICAL: match the engine's ABI exactly. The engine is built g++-10 with the
# default _GLIBCXX_USE_CXX11_ABI=1; libtorch (NOT linked here) is ABI=0. We pin
# ABI=1 explicitly so a stray ABI=0 toolchain default can never sneak in.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
add_compile_options(-fvisibility=hidden -O3)

# --- nanobind -----------------------------------------------------------------
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

# Locate nanobind's CMake config shipped with the installed package.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_nuka_ext STABLE_ABI NB_STATIC src/nuka_ext.cpp)

target_include_directories(_nuka_ext PRIVATE "${NUKA_INCLUDE_DIR}")
target_link_libraries(_nuka_ext PRIVATE "${NUKA_LIB}" "${CUDART_LIB}")

# RPATH so the installed extension finds libnuka.so + libcudart at runtime
# without LD_LIBRARY_PATH. (libnuka.so already carries a RUNPATH to the CUDA
# libs, but we add both for robustness.)
set_target_properties(_nuka_ext PROPERTIES
  BUILD_WITH_INSTALL_RPATH FALSE
  INSTALL_RPATH "${NUKA_BUILD_DIR}/src;${CUDA_LIB_DIR}")

# Install the compiled extension into the `nuka` package directory.
install(TARGETS _nuka_ext LIBRARY DESTINATION nuka)
