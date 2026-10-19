#version 450
#extension GL_GOOGLE_include_directive : require
// The viewport's ground for scenes without one: an endless matte plane at sky_ground.w with a
// metre grid, lit and shadowed like the scene, fading into the horizon.

#include "light_limits.glsl"
const int kMaxLights = NUKA_MAX_LIGHTS;
struct GpuLight {
    vec4 direction;
    vec4 position;
    vec4 color;
};
#include "viewport_scene.glsl"

layout(location = 0) out vec4 oColor;

const float kPi = 3.14159265358979323846;
const vec3 kFloor = vec3(0.155, 0.157, 0.16);
const vec3 kLine = vec3(0.07, 0.072, 0.075);
const vec3 kAxisX = vec3(0.55, 0.16, 0.14);
const vec3 kAxisY = vec3(0.20, 0.45, 0.16);

// How much of the pixel a grid line of `cell` covers, faded once the cells shrink to a few pixels.
float GridLine(vec2 p, float cell) {
    const vec2 q = p / cell;
    const vec2 w = max(fwidth(q), vec2(1e-6));
    const vec2 g = abs(fract(q - 0.5) - 0.5) / w;
    return (1.0 - min(min(g.x, g.y), 1.0)) * (1.0 - smoothstep(0.12, 0.45, max(w.x, w.y)));
}

// How much of the pixel the world axis line along `coord` = 0 covers.
float AxisLine(float coord) {
    return 1.0 - min(abs(coord) / max(fwidth(coord), 1e-6) * 0.5, 1.0);
}

void main() {
    vec3 origin, dir;
    ViewRay(gl_FragCoord.xy, origin, dir);
    const float height = scene.sky_ground.w;
    if (dir.z > -1e-5) discard;
    const float t = (height - origin.z) / dir.z;
    if (t <= 0.0) discard;
    const vec3 p = origin + dir * t;
    // Depth from a hair behind the plane, so a surface the scene puts at the same height wins.
    const vec4 clip = scene.view_proj * vec4(origin + dir * (t * 1.001), 1.0);
    gl_FragDepth = clip.z / clip.w;

    vec3 albedo = mix(kFloor, kLine, max(GridLine(p.xy, 1.0) * 0.8, GridLine(p.xy, 0.25) * 0.3));
    albedo = mix(albedo, kAxisX, AxisLine(p.y) * 0.8);
    albedo = mix(albedo, kAxisY, AxisLine(p.x) * 0.8);

    vec3 lit = scene.ambient.rgb * albedo;
    for (int i = 0; i < min(scene.counts.x, kMaxLights); ++i) {
        const GpuLight L = scene.lights[i];
        if (L.direction.w > 0.5) {
            const float n_dot_l = max(normalize(L.direction.xyz).z, 0.0);
            lit += albedo / kPi * L.color.rgb * n_dot_l * SunVisibility(p, n_dot_l);
        } else {
            const vec3 to_light = L.position.xyz - p;
            const float dist2 = max(dot(to_light, to_light), 1e-4);
            lit += albedo / kPi * L.color.rgb / dist2 * max(to_light.z * inversesqrt(dist2), 0.0);
        }
    }

    const float dist = length(p - scene.camera_pos.xyz);
    const float horizon = max(80.0, abs(scene.camera_pos.z - height) * 60.0);
    vec3 color = mix(lit, SkyRadiance(dir), smoothstep(horizon * 0.3, horizon, dist));
    if (scene.fog.w > 0.0) color = mix(color, scene.fog.rgb, clamp(1.0 - exp(-scene.fog.w * dist), 0.0, 1.0));
    oColor = vec4(Display(color), 1.0);
}
