# Standalone C++ inference-validation harness for the trained Go2 policy.
#
# IMPORTANT: This is its OWN project(). It is intentionally NOT added to the
# main Nuka CMakeLists.txt and must be configured/built in a SEPARATE build
# directory (e.g. tools/inference/build) so it never touches build-cuda128 or
# the engine build.
#
# It links ONLY libtorch. The pip libtorch is built with the pre-CXX11 ABI
# (-D_GLIBCXX_USE_CXX11_ABI=0); find_package(Torch) puts that flag in
# TORCH_CXX_FLAGS, which we apply to the target. We deliberately do NOT link
# any Nuka engine library (those are CXX11-ABI=1, which would clash).
#
# Configure (point Torch_DIR at the conda torch cmake dir):
#   cmake -S tools/inference -B tools/inference/build \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_CXX_COMPILER=g++-10 \
#     -DTorch_DIR=/root/miniconda3/envs/nuka-v03/lib/python3.10/site-packages/torch/share/cmake/Torch
#   cmake --build tools/inference/build -j

cmake_minimum_required(VERSION 3.18)
project(go2_policy_check LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Locate libtorch. Caller passes -DTorch_DIR=<torch>/share/cmake/Torch.
find_package(Torch REQUIRED)

# Surface the ABI flag so the build log proves what we compiled against.
message(STATUS "TORCH_CXX_FLAGS = ${TORCH_CXX_FLAGS}")
message(STATUS "TORCH_LIBRARIES = ${TORCH_LIBRARIES}")

add_executable(go2_policy_check go2_policy_check.cpp)

# Apply the libtorch ABI / required compile flags (carries
# -D_GLIBCXX_USE_CXX11_ABI=0 for the pre-CXX11 pip build).
target_compile_options(go2_policy_check PRIVATE ${TORCH_CXX_FLAGS})

# Link ONLY libtorch. (No engine libraries -- ABI clash.)
target_link_libraries(go2_policy_check PRIVATE ${TORCH_LIBRARIES})

# Bake the torch lib dir into the rpath so the executable finds libtorch (and
# its CUDA deps, resolved via torch's own rpath) without LD_LIBRARY_PATH.
set_target_properties(go2_policy_check PROPERTIES
  BUILD_RPATH   "${TORCH_INSTALL_PREFIX}/lib"
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")
