#version 450
#extension GL_GOOGLE_include_directive : require
// The viewport sky behind the scene: the horizon gradient and the sun's disc.

#include "light_limits.glsl"
const int kMaxLights = NUKA_MAX_LIGHTS;
struct GpuLight {
    vec4 direction;
    vec4 position;
    vec4 color;
};
#include "viewport_scene.glsl"

layout(location = 0) out vec4 oColor;

void main() {
    vec3 origin, dir;
    ViewRay(gl_FragCoord.xy, origin, dir);
    oColor = vec4(Display(SkyRadiance(dir) + SunGlow(dir)), 1.0);
}
