#version 450
#extension GL_GOOGLE_include_directive : require
// ---------------------------------------------------------------------------
// nuka::render::raster INSTANCED forward-pass vertex shader (M11 INT-5,
// OD-2=(A)). The interop-only variant of mesh.vert: it reads the per-instance
// object->world model matrix from a DEVICE-LOCAL storage buffer (the SSBO the
// CUDA scatter writes via cudaImportExternalMemory) indexed by gl_InstanceIndex,
// instead of from the per-draw push constant.
//
// This is the ZERO-COPY render path: the model matrices are produced on the GPU by
// the physics->render scatter kernel and consumed here with NO host round-trip
// (the firm render<->physics shared-memory directive). The DEFAULT mesh.vert
// per-draw push-constant pipeline is the D1 oracle path and is UNTOUCHED -- this
// 2nd pipeline variant is opt-in and only bound when interop is enabled.
//
// MEMORY LAYOUT: the SSBO holds one column-major mat4 (16 floats) per instance,
// row i at floats [i*16, i*16+16). The CUDA scatter writes exactly this layout
// (DevTransformToMatrix, bit-identical to the host TransformToMatrix). std430 lays
// a mat4[] out at 64-byte stride matching the kernel's 16-float rows.
//
// The material (base_color / mr / emissive) still arrives per-draw via the push
// constant (materials are not part of the physics device state) -- only the model
// matrix moves to the SSBO. Determinism is not a concern on this path (it is the
// realtime viewer path, never a D1 gate).
// ---------------------------------------------------------------------------

layout(location = 0) in vec3 inPosition;  // object-space position
layout(location = 1) in vec3 inNormal;    // object-space normal

#include "light_limits.glsl"  // NUKA_MAX_LIGHTS -- single source, == C++ kMaxUboLights
const int kMaxLights = NUKA_MAX_LIGHTS;
struct GpuLight {
    vec4 direction;
    vec4 position;
    vec4 color;
};
layout(set = 0, binding = 0) uniform SceneUbo {
    mat4     view_proj;
    vec4     camera_pos;
    vec4     ambient;
    vec4     ambient_ground;
    ivec4    counts;
    vec4     fog;
    GpuLight lights[kMaxLights];
    mat4     light_view_proj;
    vec4     shadow_params;
} scene;

// set 1, binding 0: the device-local per-instance transform SSBO (interop). std430
// mat4[] == 16 contiguous floats per instance, matching the CUDA scatter layout.
layout(std430, set = 1, binding = 0) readonly buffer InstanceTransforms {
    mat4 model[];
} instances;

layout(push_constant) uniform PushBlock {
    mat4 model;          // UNUSED on this path (kept so the push range matches the
                         // shared pipeline layout); the model comes from the SSBO.
    vec4 base_color;
    vec4 mr;
    vec4 emissive;
} pc;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vWorldNormal;
layout(location = 2) out vec2 vUv;
layout(location = 3) out vec4 vBaseColor;
layout(location = 4) out vec4 vMr;
layout(location = 5) out vec4 vEmissive;

void main() {
    mat4 model = instances.model[gl_InstanceIndex];
    vec4 world = model * vec4(inPosition, 1.0);
    vWorldPos = world.xyz;
    vWorldNormal = mat3(model) * inNormal;
    vUv = vec2(0.0, 0.0);
    vBaseColor = pc.base_color;
    vMr = pc.mr;
    vEmissive = pc.emissive;
    gl_Position = scene.view_proj * world;
}
