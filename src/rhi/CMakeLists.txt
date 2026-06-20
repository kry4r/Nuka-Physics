# ---------------------------------------------------------------------------
# nuka_rhi -- the OFFLINE render layer (host, CUDA-FREE). A client of PHI: it
# expresses the render pipeline as a RenderGraph of passes selected by a
# RenderProfile, driving the self-written ray tracer through render::RtBackendI.
#
# Names ONLY phi opaque handles + render:: / rt:: CUDA-free types (the same
# zero-CUDA red-line src/render holds), so it compiles under g++ and links
# standalone. It links nuka_render for the RenderWorld -> tracer adapter +
# the RtBackendI interface (which carries the weak CreateCudaRtBackend fallback);
# a consumer that wants a live RT backend links nuka_phi2_rt for the strong
# factory, exactly as the existing rt tests + demos do.
#
# The realtime Vulkan rasterizer stays in src/render (unmoved); this is the
# offline half of the PHI != RHI split.
# ---------------------------------------------------------------------------
add_library(nuka_rhi STATIC
    offline/offline_renderer.cpp
    # The batched device-resident sensor render facade (sibling of OfflineRenderer):
    # drives render::SensorBackendI for the in-loop per-env AOV tensor path.
    offline/sensor_renderer.cpp
)
set_target_properties(nuka_rhi PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(nuka_rhi
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nuka_rhi
    PUBLIC
        nuka_render
)

nuka_set_warnings(nuka_rhi)
