#version 450
// ---------------------------------------------------------------------------
// nuka::render::raster forward pass vertex shader (M8 T3a).
//
// Transforms a per-instance object-space vertex by a push-constant MVP and
// passes the world-space position + world-space normal + base color to the
// fragment shader. The model matrix (object->world) is supplied separately from
// the view-projection so the fragment shader can light in world space.
//
// T3a is minimal-shading: a single default directional light + ambient is done
// in the fragment shader. The full PBR material set (metallic/roughness/
// emissive/opacity + textures + RenderWorld.lights) is the T3b seam.
// ---------------------------------------------------------------------------

layout(location = 0) in vec3 inPosition;  // object-space position
layout(location = 1) in vec3 inNormal;    // object-space normal (flat-synthesized if absent)

layout(push_constant) uniform PushBlock {
    mat4 mvp;          // view_proj * model (clip-space transform)
    mat4 model;        // object -> world (for world-space lighting)
    vec4 base_color;   // material base color (rgb) + opacity (a)
} pc;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vWorldNormal;
layout(location = 2) out vec4 vBaseColor;

void main() {
    vec4 world = pc.model * vec4(inPosition, 1.0);
    vWorldPos = world.xyz;
    // Normal matrix: for rigid (rotation+translation) model transforms the upper
    // 3x3 is orthonormal, so the inverse-transpose equals the 3x3 itself.
    vWorldNormal = mat3(pc.model) * inNormal;
    vBaseColor = pc.base_color;
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
}
