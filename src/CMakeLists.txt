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
