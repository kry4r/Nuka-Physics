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
    vec4     occlusion;        // radius in metres, strength
    vec4     environment;      // enabled, yaw, intensity
    vec4     irradiance[9];
} scene;

layout(set = 0, binding = 1) uniform sampler2D uShadowMap;
// Every material image map as one layer, raw texels (albedo still sRGB-encoded).
layout(set = 0, binding = 2) uniform sampler2DArray uMaps;
layout(set = 0, binding = 4) uniform sampler2D uSceneDepth;
layout(set = 0, binding = 5) uniform sampler2D uEnvironment;

vec3 EnvironmentDirection(vec3 direction) {
    const float c = cos(scene.environment.y), s = sin(scene.environment.y);
    return vec3(c * direction.x - s * direction.y, s * direction.x + c * direction.y, direction.z);
}

vec3 EnvironmentDiffuse(vec3 normal) {
    const vec3 n = EnvironmentDirection(normal);
    const float basis[9] = float[9](0.2820948, 0.4886025 * n.y, 0.4886025 * n.z, 0.4886025 * n.x,
        1.0925484 * n.x * n.y, 1.0925484 * n.y * n.z, 0.3153916 * (3.0 * n.z * n.z - 1.0),
        1.0925484 * n.x * n.z, 0.5462742 * (n.x * n.x - n.y * n.y));
    vec3 light = vec3(0.0);
    for (int i = 0; i < 9; ++i) light += scene.irradiance[i].rgb * basis[i];
    return max(light, vec3(0.0)) * scene.environment.z;
}

// Reconstruct neighbors from this view's depth prepass; only indirect light is occluded.
float AmbientVisibility(vec3 position, vec3 normal) {
    const float radius = scene.occlusion.x;
    if (radius <= 0.0) return 1.0;
    const vec4 clip = scene.view_proj * vec4(position, 1.0);
    const float vertical_scale = length(vec3(scene.view_proj[0][1], scene.view_proj[1][1], scene.view_proj[2][1]));
    const float pixels = clamp(radius * vertical_scale * scene.viewport.w /
                               (2.0 * max(clip.w, 0.001)), 1.0, 128.0);
    const vec2 center = gl_FragCoord.xy;
    float blocked = 0.0;
    const float rotation = fract(sin(dot(floor(center), vec2(12.9898, 78.233))) * 43758.5453) * 6.2831853;
    for (int i = 0; i < 16; ++i) {
        const float angle = rotation + float(i) * 2.3999632;
        const vec2 pixel = center + vec2(cos(angle), sin(angle)) * pixels * sqrt((float(i) + 0.5) / 16.0);
        if (any(lessThan(pixel, scene.viewport.xy)) ||
            any(greaterThanEqual(pixel, scene.viewport.xy + scene.viewport.zw))) continue;
        const float depth = texelFetch(uSceneDepth, ivec2(pixel), 0).r;
        if (depth >= 1.0) continue;
        const vec2 ndc = (floor(pixel) + 0.5 - scene.viewport.xy) / scene.viewport.zw * 2.0 - 1.0;
        const vec4 projected = scene.inv_view_proj * vec4(ndc, depth, 1.0);
        const vec3 delta = projected.xyz / projected.w - position;
        const float distance = length(delta);
        if (distance <= 0.001 || distance >= radius) continue;
        blocked += max(dot(normal, delta / distance) - 0.05, 0.0) *
                   (1.0 - smoothstep(0.0, radius, distance));
    }
    return clamp(1.0 - scene.occlusion.y * blocked * (2.0 / 16.0), 0.0, 1.0);
}

// Per material, its map layers in uMaps and how they project; the push block's mr.w picks one.
struct MaterialMaps {
    vec4 layers;   // albedo, roughness, normal, metallic layers (-1 none)
    vec4 extra;    // x emissive layer, y roughness channel, z metallic channel (0 R, 1 G, 2 B)
    vec4 mapping;  // x 1 projects triplanar, y the UV scale or tiles per metre
};
layout(std430, set = 0, binding = 3) readonly buffer MaterialMapsBlock {
    MaterialMaps slots[];
} materials;

// The sky's radiance along a world direction (+Z up), the sun left out.
vec3 SkyRadiance(vec3 dir) {
    if (scene.environment.x > 0.5) {
        const vec3 d = EnvironmentDirection(dir);
        const vec2 uv = vec2(atan(d.y, d.x) / 6.2831853 + 0.5, acos(clamp(d.z, -1.0, 1.0)) / 3.1415927);
        return texture(uEnvironment, uv).rgb * scene.environment.z;
    }
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

// A map layer's texel at texture coordinate st, and one channel of it (0 R, 1 G, 2 B).
vec4 Map(float layer, vec2 st) {
    return texture(uMaps, vec3(st, layer));
}

float MapChannel(vec4 c, float channel) {
    return channel > 1.5 ? c.b : channel > 0.5 ? c.g : c.r;
}

// A tangent frame about unit normal n from the screen derivatives of the position and the v-up UV.
mat3 CotangentFrame(vec3 n, vec3 p, vec2 uv) {
    const vec3 dp1 = dFdx(p), dp2 = dFdy(p);
    const vec2 duv1 = dFdx(uv), duv2 = dFdy(uv);
    const vec3 dp2perp = cross(dp2, n), dp1perp = cross(n, dp1);
    const vec3 t = dp2perp * duv1.x + dp1perp * duv2.x;
    const vec3 b = dp2perp * duv1.y + dp1perp * duv2.y;
    const float scale = inversesqrt(max(max(dot(t, t), dot(b, b)), 1e-20));
    return mat3(t * scale, b * scale, n);
}

// Linear HDR to the display: the ACES fit then the sRGB curve, as mesh_pbr.frag ends.
vec3 Display(vec3 c) {
    c = clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), 0.0, 1.0);
    return mix(1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, c * 12.92, lessThanEqual(c, vec3(0.0031308)));
}
