#version 450
// ---------------------------------------------------------------------------
// nuka::render::raster DIRECTIONAL SHADOW-MAP depth-only vertex shader.
//
// Used by the dedicated depth-only shadow pass (no fragment shader): rasterizes
// the scene depth from the SUN's orthographic view into a D32 shadow map that the
// main forward pass (mesh_pbr.frag) samples to cast crisp directional shadows.
// The per-draw light-space MVP (light_view_proj * model) is supplied as the only
// push constant, so this pass needs NO descriptor sets. Position-only input.
//
// G2-SAFE: this pass is recorded ONLY when RasterOptions.shadow_strength > 0; the
// gated determinism smokes never enter it.
// ---------------------------------------------------------------------------

layout(location = 0) in vec3 inPosition;  // object-space position

layout(push_constant) uniform ShadowPush {
    mat4 light_mvp;  // light_view_proj * model (world->light clip, baked per draw)
} pc;

void main() {
    gl_Position = pc.light_mvp * vec4(inPosition, 1.0);
}
