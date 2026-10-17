cmake_minimum_required(VERSION 3.18...3.30)
project(nuka_ext LANGUAGES CXX)

# --- Engine locations (override with -DNUKA_ROOT / -DNUKA_BUILD_DIR) ----------
# NUKA_ROOT       : repo root that holds src/include/nuka/nuka.h
# NUKA_BUILD_DIR  : the engine build dir
# CUDA_LIB_DIR    : Linux libcudart.so directory
if(NOT DEFINED NUKA_ROOT)
  set(NUKA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()
if(NOT DEFINED NUKA_BUILD_DIR)
  if(WIN32)
    set(NUKA_BUILD_DIR "${NUKA_ROOT}/build-win-editor")
  else()
    set(NUKA_BUILD_DIR "${NUKA_ROOT}/build-cuda128")
  endif()
endif()
if(NOT WIN32 AND NOT DEFINED CUDA_LIB_DIR)
  set(CUDA_LIB_DIR "/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64")
endif()

set(NUKA_INCLUDE_DIR "${NUKA_ROOT}/src/include")
if(WIN32)
  set(NUKA_LIB "${NUKA_BUILD_DIR}/src/nuka.lib")
  set(NUKA_RUNTIME "${NUKA_BUILD_DIR}/src/nuka.dll")
  set(NUKA_EDITOR_LIB "${NUKA_BUILD_DIR}/src/nuka_editor.lib")
  set(NUKA_EDITOR_RUNTIME "${NUKA_BUILD_DIR}/src/nuka_editor.dll")
else()
  set(NUKA_LIB "${NUKA_BUILD_DIR}/src/libnuka.so")
  set(NUKA_EDITOR_LIB "${NUKA_BUILD_DIR}/src/libnuka_editor.so")
  set(CUDART_LIB "${CUDA_LIB_DIR}/libcudart.so")
endif()

if(NOT EXISTS "${NUKA_INCLUDE_DIR}/nuka/nuka.h")
  message(FATAL_ERROR "Cannot find nuka.h at ${NUKA_INCLUDE_DIR}/nuka/nuka.h "
                      "(set -DNUKA_ROOT=/path/to/repo)")
endif()
if(NOT EXISTS "${NUKA_LIB}")
  message(FATAL_ERROR "Cannot find the Nuka link library at ${NUKA_LIB} "
                      "(build the engine first, or set -DNUKA_BUILD_DIR=...)")
endif()
if(WIN32 AND NOT EXISTS "${NUKA_RUNTIME}")
  message(FATAL_ERROR "Cannot find the Nuka runtime at ${NUKA_RUNTIME}")
endif()

# --- Toolchain / ABI ----------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT MSVC)
  # Match the Linux engine's g++ ABI. Torch remains DLPack-only and is not linked.
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
  add_compile_options(-fvisibility=hidden -O3)
endif()

# --- nanobind -----------------------------------------------------------------
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

# Locate nanobind's CMake config shipped with the installed package.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_nuka_ext STABLE_ABI NB_STATIC src/nuka_ext.cpp)

target_include_directories(_nuka_ext PRIVATE "${NUKA_INCLUDE_DIR}")
if(WIN32)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(_nuka_ext PRIVATE "${NUKA_LIB}" CUDA::cudart)
else()
  target_link_libraries(_nuka_ext PRIVATE "${NUKA_LIB}" "${CUDART_LIB}")

  # Resolve the engine and CUDA runtime without LD_LIBRARY_PATH.
  set_target_properties(_nuka_ext PROPERTIES
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH "${NUKA_BUILD_DIR}/src;${CUDA_LIB_DIR}")
endif()

# Install the extension and, on Windows, its adjacent engine runtime + the CUDA
# runtime DLL so `import nuka` works on machines without the toolkit on PATH.
install(TARGETS _nuka_ext LIBRARY DESTINATION nuka RUNTIME DESTINATION nuka)
if(WIN32)
  install(FILES "${NUKA_RUNTIME}" DESTINATION nuka)
  file(GLOB NUKA_CUDART_DLLS "${CUDAToolkit_BIN_DIR}/cudart64_*.dll"
                             "${CUDAToolkit_BIN_DIR}/x64/cudart64_*.dll")
  install(FILES ${NUKA_CUDART_DLLS} DESTINATION nuka)
endif()
# Typed stubs, for IDEs and the editor's completion, come from the installed modules; they
# import as top-level modules so that a nuka installed elsewhere cannot shadow them.
nanobind_add_stub(_nuka_ext_stub INSTALL_TIME MODULE _nuka_ext OUTPUT nuka/_nuka_ext.pyi PYTHON_PATH nuka)

# The editor extension exists only when the engine was built with NK_BUILD_EDITOR.
if(EXISTS "${NUKA_EDITOR_LIB}")
  nanobind_add_module(_editor_ext STABLE_ABI NB_STATIC src/editor_ext.cpp)
  target_include_directories(_editor_ext PRIVATE "${NUKA_INCLUDE_DIR}")
  target_link_libraries(_editor_ext PRIVATE "${NUKA_EDITOR_LIB}")
  if(NOT WIN32)
    set_target_properties(_editor_ext PROPERTIES
      BUILD_WITH_INSTALL_RPATH FALSE
      INSTALL_RPATH "${NUKA_BUILD_DIR}/src")
  endif()
  install(TARGETS _editor_ext LIBRARY DESTINATION nuka RUNTIME DESTINATION nuka)
  if(WIN32)
    install(FILES "${NUKA_EDITOR_RUNTIME}" DESTINATION nuka)
  endif()
  nanobind_add_stub(_editor_ext_stub INSTALL_TIME MODULE _editor_ext OUTPUT nuka/_editor_ext.pyi PYTHON_PATH nuka)
  # Fonts, the logo and the default scene resolve next to the editor library.
  install(DIRECTORY "${NUKA_BUILD_DIR}/src/assets" "${NUKA_BUILD_DIR}/src/docs"
                    "${NUKA_BUILD_DIR}/src/examples" DESTINATION nuka)
endif()
