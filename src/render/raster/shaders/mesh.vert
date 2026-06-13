#version 450
// ---------------------------------------------------------------------------
// nuka::render::raster forward pass vertex shader (M8.5 T4 -- full PBR seam).
//
// Transforms a per-instance object-space vertex by the per-frame view_proj
// (descriptor set 0) composed with the per-draw model matrix (push constant),
// and passes the world-space position + world-space normal + UV + the full
// per-draw material to the fragment shader, which runs Cook-Torrance GGX.
//
// The model matrix (object->world) is supplied per-draw in the push constant;
// the view_proj (and the camera position + the light rig) live in the per-frame
// SceneUbo (set 0, binding 0) so the lighting environment is uploaded once per
// render rather than per draw. Determinism: no time/random inputs; the entire
// transform + lighting state is a pure function of the fixed scene + camera.
//
// T4 NOTE on normals: cooked MESH chunks may carry zero-filled normals (T5
// flagged). The renderer synthesizes per-vertex flat/area-weighted geometric
// normals at intern time when the normal stream length != position stream, so
// inNormal is always usable and deterministic here.
// ---------------------------------------------------------------------------

layout(location = 0) in vec3 inPosition;  // object-space position
layout(location = 1) in vec3 inNormal;    // object-space normal (flat-synthesized if absent)

// Per-frame scene environment (camera + light rig). Bound once per render; see
// the matching SceneUbo in mesh_pbr.frag.
const int kMaxLights = 8;
struct GpuLight {
    vec4 direction;  // xyz = world-space direction TOWARD the light; w = is_directional
    vec4 position;   // xyz = world-space position (point lights);    w = range (unused)
    vec4 color;      // rgb = color * intensity (radiance);           w = unused
};
layout(set = 0, binding = 0) uniform SceneUbo {
    mat4     view_proj;     // clip = view_proj * model * pos
    vec4     camera_pos;    // xyz = world-space eye; w unused
    vec4     ambient;       // rgb = hemispheric/ambient sky color; w = ground mix
    vec4     ambient_ground;// rgb = hemispheric ground color;      w unused
    ivec4    counts;        // x = active light count
    GpuLight lights[kMaxLights];
} scene;

layout(push_constant) uniform PushBlock {
    mat4 model;          // object -> world (for world-space lighting)
    vec4 base_color;     // material albedo (rgb) + opacity (a)
    vec4 mr;             // x = metallic, y = roughness, z = opacity, w = pad
    vec4 emissive;       // rgb = emissive radiance; a = pad
} pc;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vWorldNormal;
layout(location = 2) out vec2 vUv;          // forward-compat (textures deferred to M11)
layout(location = 3) out vec4 vBaseColor;
layout(location = 4) out vec4 vMr;
layout(location = 5) out vec4 vEmissive;

void main() {
    vec4 world = pc.model * vec4(inPosition, 1.0);
    vWorldPos = world.xyz;
    // Normal matrix: for rigid (rotation+translation) model transforms the upper
    // 3x3 is orthonormal, so the inverse-transpose equals the 3x3 itself.
    vWorldNormal = mat3(pc.model) * inNormal;
    vUv = vec2(0.0, 0.0);
    vBaseColor = pc.base_color;
    vMr = pc.mr;
    vEmissive = pc.emissive;
    gl_Position = scene.view_proj * world;
}
