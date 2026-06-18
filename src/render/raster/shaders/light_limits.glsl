// Shared light-array bound for the raster forward pass. SINGLE SOURCE for the
// GLSL light cap so mesh.vert / mesh_instanced.vert / mesh_pbr.frag cannot desync.
//
// CANONICAL VALUE = the C++ `kMaxUboLights` in
//   src/render/raster/vulkan_raster_renderer.cpp  and
//   src/render/raster/vulkan_present_renderer.cpp
// which size the SceneUbo.lights[] array uploaded to set 0, binding 0. Keep this
// value equal to that C++ cap; both .cpp sites carry a matching comment. (C++ and
// GLSL cannot share a literal token, so the contract is documented, not asserted.)
#ifndef NUKA_MAX_LIGHTS
#define NUKA_MAX_LIGHTS 8
#endif
