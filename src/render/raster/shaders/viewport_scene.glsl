// The viewport's per-view SceneUbo and its sky, sun and shadow helpers (realtime present path
// only). Include after GpuLight and kMaxLights; the block extends mesh_pbr.frag's offscreen one.
layout(set = 0, binding = 0) uniform SceneUbo {
    mat4     view_proj;
    vec4     camera_pos;
    vec4     ambient;          // rgb sky irradiance
    vec4     ambient_ground;   // rgb ground irradiance
    ivec4    counts;           // x active lights
    vec4     fog;              // rgb haze, w density (0 off)
    GpuLight lights[kMaxLights];
    mat4     light_view_proj;  // world -> the sun's shadow-map clip space
    vec4     shadow_params;    // strength (0 off), depth bias, texel size, PCF spacing
    mat4     inv_view_proj;
    vec4     viewport;         // the view's framebuffer x, y, width, height
    vec4     sky_zenith;       // rgb; w 1 when the sky lights the scene
    vec4     sky_horizon;
    vec4     sky_ground;       // rgb below the horizon; w the ground grid's height
    vec4     sun;              // xyz toward the sun; w the disc's cosine radius, 0 without a sun
    vec4     sun_color;        // rgb radiance
} scene;

layout(set = 0, binding = 1) uniform sampler2D uShadowMap;
// Every material image map as one layer, raw texels (albedo still sRGB-encoded).
layout(set = 0, binding = 2) uniform sampler2DArray uMaps;

// The sky's radiance along a world direction (+Z up), the sun left out.
vec3 SkyRadiance(vec3 dir) {
    if (dir.z >= 0.0) return mix(scene.sky_horizon.rgb, scene.sky_zenith.rgb, sqrt(dir.z));
    return mix(scene.sky_horizon.rgb, scene.sky_ground.rgb, smoothstep(0.0, 0.08, -dir.z));
}

// The sun's disc and glow along a world direction.
vec3 SunGlow(vec3 dir) {
    if (scene.sun.w <= 0.0) return vec3(0.0);
    const float c = max(dot(dir, scene.sun.xyz), 0.0);
    const float disc = smoothstep(scene.sun.w, mix(scene.sun.w, 1.0, 0.25), c);
    return scene.sun_color.rgb * (disc * 6.0 + pow(c, 96.0) * 0.10 + pow(c, 8.0) * 0.025);
}

// The world-space ray through a framebuffer pixel of the view.
void ViewRay(vec2 pixel, out vec3 origin, out vec3 dir) {
    const vec2 ndc = (pixel - scene.viewport.xy) / scene.viewport.zw * 2.0 - 1.0;
    const vec4 near = scene.inv_view_proj * vec4(ndc, 0.0, 1.0);
    const vec4 far = scene.inv_view_proj * vec4(ndc, 1.0, 1.0);
    origin = near.xyz / near.w;
    dir = normalize(far.xyz / far.w - origin);
}

// The sun's visibility at a world point, 3x3 PCF; 1 off the map or with shadows off.
float SunVisibility(vec3 world, float n_dot_l) {
    if (scene.shadow_params.x <= 0.0) return 1.0;
    const vec4 clip = scene.light_view_proj * vec4(world, 1.0);
    const vec3 ndc = clip.xyz / clip.w;
    if (any(greaterThan(abs(ndc.xy), vec2(1.0))) || ndc.z < 0.0 || ndc.z > 1.0) return 1.0;
    const vec2 uv = ndc.xy * 0.5 + 0.5;
    const float bias = scene.shadow_params.y * (0.5 + 2.0 * clamp(1.0 - n_dot_l, 0.0, 1.0));
    const float texel = scene.shadow_params.z * max(scene.shadow_params.w, 1.0);
    float lit = 0.0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx)
            lit += ndc.z - bias <= texture(uShadowMap, uv + vec2(dx, dy) * texel).r ? 1.0 : 0.0;
    }
    return mix(1.0, lit / 9.0, scene.shadow_params.x);
}

// The split-sum environment BRDF's scale and bias for F0 (Karis' analytic fit).
vec2 EnvBrdf(float roughness, float n_dot_v) {
    const vec4 r = roughness * vec4(-1.0, -0.0275, -0.572, 0.022) + vec4(1.0, 0.0425, 1.04, -0.04);
    const float a = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    return vec2(-1.04, 1.04) * a + r.zw;
}

// Triplanar blend weights from a unit normal: n^4, normalised, as the beauty tracer blends.
vec3 TriplanarWeights(vec3 n) {
    vec3 w = n * n;
    w *= w;
    return w / max(w.x + w.y + w.z, 1e-12);
}

// A map layer projected on the X (yz), Y (xz) and Z (xy) planes at a point measured in tiles.
vec4 Triplanar(float layer, vec3 p, vec3 w) {
    return texture(uMaps, vec3(p.yz, layer)) * w.x + texture(uMaps, vec3(p.xz, layer)) * w.y +
           texture(uMaps, vec3(p.xy, layer)) * w.z;
}

// A GL-convention normal map layer, whiteout-blended over the three planes about unit normal n.
vec3 TriplanarNormal(float layer, vec3 p, vec3 n, vec3 w) {
    const vec3 tx = texture(uMaps, vec3(p.yz, layer)).xyz * 2.0 - 1.0;
    const vec3 ty = texture(uMaps, vec3(p.xz, layer)).xyz * 2.0 - 1.0;
    const vec3 tz = texture(uMaps, vec3(p.xy, layer)).xyz * 2.0 - 1.0;
    return normalize(vec3(abs(tx.z) * n.x, tx.x + n.y, tx.y + n.z) * w.x +
                     vec3(ty.x + n.x, abs(ty.z) * n.y, ty.y + n.z) * w.y +
                     vec3(tz.x + n.x, tz.y + n.y, abs(tz.z) * n.z) * w.z);
}

vec3 SrgbToLinear(vec3 c) {
    return mix(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92, lessThanEqual(c, vec3(0.04045)));
}

// Linear HDR to the display: the ACES fit then the sRGB curve, as mesh_pbr.frag ends.
vec3 Display(vec3 c) {
    c = clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), 0.0, 1.0);
    return mix(1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, c * 12.92, lessThanEqual(c, vec3(0.0031308)));
}
