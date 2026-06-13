#version 450
// ---------------------------------------------------------------------------
// nuka::render::raster forward pass fragment shader (M8 T3a -- MINIMAL).
//
// T3a shading = Lambert diffuse from a single hardcoded default directional
// light + a constant ambient term, modulating the per-instance base color. This
// is deliberately minimal: enough to read the 3D shape of geometry and to keep
// distinct base colors distinct.
//
// T3b SEAM (full PBR): this shader is where metallic/roughness/emissive/opacity
// and the 3 RenderMaterial textures (albedo/normal/metallic_roughness) bind, and
// where RenderWorld.lights replaces the hardcoded directional light with a real
// (Cook-Torrance GGX) light loop fed by a per-frame light UBO/SSBO + sampler
// descriptors. The push-constant already carries base_color; T3b extends the
// push block / adds descriptor sets for the remaining material fields and a
// lights buffer. Tonemap (e.g. ACES) + sRGB encode also land in T3b.
// ---------------------------------------------------------------------------

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vWorldNormal;
layout(location = 2) in vec4 vBaseColor;

layout(location = 0) out vec4 oColor;

void main() {
    vec3 n = normalize(vWorldNormal);
    // Default directional light direction (pointing from the scene toward the
    // light) -- a fixed, deterministic key light from the upper-front.
    const vec3 kLightDir = normalize(vec3(0.4, 0.5, 0.75));
    const float kAmbient = 0.25;
    // Two-sided Lambert so back-facing fallback primitives still shade (cull off).
    float ndl = max(abs(dot(n, kLightDir)), 0.0);
    float lit = kAmbient + (1.0 - kAmbient) * ndl;
    oColor = vec4(vBaseColor.rgb * lit, vBaseColor.a);
}
