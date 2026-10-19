#version 450
#extension GL_GOOGLE_include_directive : require
// The viewport's sun shadow-map depth pass. Built with NUKA_INSTANCED, the model comes from
// the pose slot's transform rows instead of the push constant.

layout(location = 0) in vec3 inPosition;

#include "light_limits.glsl"
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
} scene;

#ifdef NUKA_INSTANCED
layout(std430, set = 1, binding = 0) readonly buffer InstanceTransforms {
    mat4 model[];
} instances;
#else
layout(push_constant) uniform PushBlock {
    mat4 model;
} pc;
#endif

void main() {
#ifdef NUKA_INSTANCED
    const mat4 model = instances.model[gl_InstanceIndex];
#else
    const mat4 model = pc.model;
#endif
    gl_Position = scene.light_view_proj * (model * vec4(inPosition, 1.0));
}
