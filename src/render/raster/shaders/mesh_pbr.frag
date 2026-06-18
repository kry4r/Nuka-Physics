#version 450
#extension GL_GOOGLE_include_directive : require
// ---------------------------------------------------------------------------
// nuka::render::raster forward pass fragment shader (M8.5 T4 -- FULL PBR).
//
// Analytic scalar physically-based shading (Decision D4): a metallic/roughness
// Cook-Torrance microfacet BRDF (Trowbridge-Reitz D_GGX + Smith-Schlick G +
// Schlick Fresnel F) evaluated against the per-frame light rig (set 0 SceneUbo),
// plus a hemispheric ambient term so unlit faces are not pure black, plus a
// material emissive add, then a Narkowicz ACES filmic tonemap and a linear->sRGB
// encode for display. No textures (no TEXB cook / image decoder exists -> M11),
// no shadow maps, no GI -- those are M11. Everything is a pure function of the
// fixed material + light rig + geometry: byte-deterministic for the G2 oracle.
//
// Normals: vWorldNormal is the interpolated world-space geometric normal
// (synthesized flat at mesh-intern time when the cooked stream is zero-filled).
// We renormalize and flip to face the camera (two-sided) since cull is off.
// ---------------------------------------------------------------------------

#include "light_limits.glsl"  // NUKA_MAX_LIGHTS -- single source, == C++ kMaxUboLights
const int kMaxLights = NUKA_MAX_LIGHTS;
struct GpuLight {
    vec4 direction;  // xyz = world-space direction TOWARD the light; w = is_directional (1.0/0.0)
    vec4 position;   // xyz = world-space position (point lights);    w = range (unused)
    vec4 color;      // rgb = color * intensity (radiance);           w = unused
};
layout(set = 0, binding = 0) uniform SceneUbo {
    mat4     view_proj;
    vec4     camera_pos;     // xyz = world-space eye
    vec4     ambient;        // rgb = hemispheric SKY color; w = ground mix weight
    vec4     ambient_ground; // rgb = hemispheric GROUND color
    ivec4    counts;         // x = active light count
    vec4     fog;            // rgb = haze colour; w = density (0 => fog OFF)
    GpuLight lights[kMaxLights];
    mat4     light_view_proj;   // sun's view-proj (world -> light clip) for shadows
    vec4     shadow_params;     // x = strength (0=off), y = bias, z = texel, w pad
} scene;

// Directional shadow map (depth from the sun). ALWAYS bound (a 1x1 dummy when
// shadows are OFF); sampled ONLY when scene.shadow_params.x > 0 so the gated
// determinism path is byte-identical (the branch below is never taken).
layout(set = 0, binding = 1) uniform sampler2D uShadowMap;

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vWorldNormal;
layout(location = 2) in vec2 vUv;
layout(location = 3) in vec4 vBaseColor;
layout(location = 4) in vec4 vMr;        // x metallic, y roughness, z opacity
layout(location = 5) in vec4 vEmissive;  // rgb emissive
layout(location = 6) in vec4 vLightClip; // position in the sun's clip space

// Per-draw push block (must match mesh.vert byte-for-byte). The fragment only
// reads ground_shadow: a soft procedural CONTACT SHADOW the renderer fills for
// the GROUND disc (z = radius = 0 on every other draw, so this is a no-op there
// and their output stays byte-identical to the pre-M10 oracle).
layout(push_constant) uniform PushBlock {
    mat4 model;
    vec4 base_color;
    vec4 mr;
    vec4 emissive;
    vec4 ground_shadow;  // xy = world contact centre, z = radius (0=off), w = strength
} pc;

layout(location = 0) out vec4 oColor;

const float kPi = 3.14159265358979323846;

// Trowbridge-Reitz GGX normal distribution.
float DistributionGGX(float n_dot_h, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = (n_dot_h * n_dot_h) * (a2 - 1.0) + 1.0;
    return a2 / max(kPi * d * d, 1e-7);
}

// Schlick-GGX geometry term (one direction).
float GeometrySchlickGGX(float n_dot_v, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;  // direct-lighting remap
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

// Smith geometry term (view + light occlusion).
float GeometrySmith(float n_dot_v, float n_dot_l, float roughness) {
    return GeometrySchlickGGX(n_dot_v, roughness) *
           GeometrySchlickGGX(n_dot_l, roughness);
}

// Schlick Fresnel.
vec3 FresnelSchlick(float cos_theta, vec3 f0) {
    return f0 + (vec3(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Narkowicz ACES filmic tonemap (approximation).
vec3 AcesFilmic(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 LinearToSrgb(vec3 c) {
    // Standard sRGB OETF.
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(hi, lo, lessThanEqual(c, vec3(0.0031308)));
}

// Directional shadow factor in [0,1] (1 = fully lit, 0 = fully in shadow), from a
// 3x3 PCF of the sun's depth map. Returns 1.0 (no shadow) when shadows are off or
// the fragment is outside the shadow frustum. n_dot_l drives a slope-scaled bias
// so steep treads do not self-shadow (acne) while contact stays tight.
float SunShadow(float n_dot_l) {
    if (scene.shadow_params.x <= 0.0) return 1.0;          // shadows OFF -> fully lit
    // Perspective divide (ortho w==1, but divide for generality) -> NDC.
    vec3 ndc = vLightClip.xyz / max(vLightClip.w, 1e-6);
    // Outside the light frustum in xy -> treat as lit (CLAMP_TO_EDGE + white border).
    if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z > 1.0) {
        return 1.0;
    }
    vec2 uv = ndc.xy * 0.5 + 0.5;          // [-1,1] -> [0,1] texel space
    float frag_depth = ndc.z;              // already in [0,1] (Vulkan ortho)
    float slope = clamp(1.0 - n_dot_l, 0.0, 1.0);
    float bias = scene.shadow_params.y * (0.5 + 2.0 * slope);
    float texel = scene.shadow_params.z;
    float lit = 0.0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            float sd = texture(uShadowMap, uv + vec2(float(dx), float(dy)) * texel).r;
            lit += (frag_depth - bias <= sd) ? 1.0 : 0.0;
        }
    }
    lit /= 9.0;
    // shadow_params.x scales how DARK the shadow gets (1 => fully dark occluder).
    return mix(1.0, lit, scene.shadow_params.x);
}

void main() {
    vec3  albedo    = vBaseColor.rgb;
    float metallic  = clamp(vMr.x, 0.0, 1.0);
    float roughness = clamp(vMr.y, 0.04, 1.0);  // floor to avoid a singular highlight
    float opacity   = clamp(vMr.z, 0.0, 1.0);

    vec3 n = normalize(vWorldNormal);
    vec3 v = normalize(scene.camera_pos.xyz - vWorldPos);
    // Two-sided: faces with cull off can present a back-facing normal; flip toward
    // the viewer so both winding orders shade consistently (deterministic).
    if (dot(n, v) < 0.0) n = -n;
    float n_dot_v = max(dot(n, v), 1e-4);

    // F0: 0.04 dielectric baseline, lerped toward the albedo for metals.
    vec3 f0 = mix(vec3(0.04), albedo, metallic);

    vec3 lo = vec3(0.0);  // accumulated outgoing radiance from analytic lights
    int  count = min(scene.counts.x, kMaxLights);
    for (int i = 0; i < count; ++i) {
        GpuLight L = scene.lights[i];
        vec3  l;
        vec3  radiance = L.color.rgb;
        if (L.direction.w > 0.5) {
            // Directional: constant direction, no attenuation.
            l = normalize(L.direction.xyz);
        } else {
            // Point: direction + inverse-square attenuation.
            vec3 to_light = L.position.xyz - vWorldPos;
            float dist2 = max(dot(to_light, to_light), 1e-4);
            l = to_light * inversesqrt(dist2);
            radiance /= dist2;
        }
        vec3  h       = normalize(v + l);
        float n_dot_l = max(dot(n, l), 0.0);
        if (n_dot_l <= 0.0) continue;
        float n_dot_h = max(dot(n, h), 0.0);
        float v_dot_h = max(dot(v, h), 0.0);

        float dist  = DistributionGGX(n_dot_h, roughness);
        float geom  = GeometrySmith(n_dot_v, n_dot_l, roughness);
        vec3  fres  = FresnelSchlick(v_dot_h, f0);

        vec3  spec  = (dist * geom * fres) / max(4.0 * n_dot_v * n_dot_l, 1e-4);
        vec3  kd    = (vec3(1.0) - fres) * (1.0 - metallic);
        vec3  diff  = kd * albedo / kPi;

        // Directional lights (the sun) are occluded by the shadow map; point
        // lights are not shadowed (the map is the sun's ortho view). When shadows
        // are OFF, SunShadow() returns 1.0 so this is an exact no-op (G2-safe).
        float shadow = (L.direction.w > 0.5) ? SunShadow(n_dot_l) : 1.0;
        lo += (diff + spec) * radiance * n_dot_l * shadow;
    }

    // Hemispheric ambient (cheap image-based-lighting stand-in): blend a sky and a
    // ground color by the normal's up-axis (world +Z is up in this scene).
    float hemi = n.z * 0.5 + 0.5;
    vec3  ambient_color = mix(scene.ambient_ground.rgb, scene.ambient.rgb, hemi);

    // Diffuse ambient is albedo-tinted (a black body absorbs it) -- this term
    // carries the bright/dark material distinction (white shell vs black plastic),
    // so keep it the DOMINANT ambient and do NOT let the recovery terms below
    // flatten it. Two SMALL albedo-INDEPENDENT terms then recover form on a near-
    // black surface so it reads as a shaded DARK-grey shape (not #000) while
    // staying clearly darker than the light parts:
    //  (1) a specular/reflective ambient on the dielectric+metal F0 (the surface
    //      reflects the environment regardless of albedo), Fresnel-boosted at
    //      grazing angles so silhouettes get a soft rim-glow against the dark bg;
    //  (2) a faint constant "form floor" scaled by the GGX gloss so an unlit black
    //      face is separated from the background -- kept low to preserve contrast.
    vec3  diffuse_ambient = ambient_color * albedo * (1.0 - metallic);
    float fresnel_rim     = pow(clamp(1.0 - n_dot_v, 0.0, 1.0), 4.0);
    vec3  spec_ambient    = ambient_color * (f0 + (max(vec3(1.0 - roughness), f0) - f0) * fresnel_rim) * 0.6;
    float gloss           = 1.0 - roughness;
    vec3  form_floor      = ambient_color * (0.035 + 0.05 * gloss * gloss);
    vec3  ambient         = diffuse_ambient + spec_ambient + form_floor;

    vec3 color = lo + ambient + vEmissive.rgb;

    // Procedural CONTACT SHADOW (M10 grounded look). Only the ground disc carries
    // a non-zero radius; for every other draw pc.ground_shadow.z == 0 so this whole
    // block is skipped and the fragment is byte-identical to the pre-M10 oracle.
    // A soft radial occlusion of the floor radiance anchors the subject's feet.
    if (pc.ground_shadow.z > 0.0) {
        float d   = length(vWorldPos.xy - pc.ground_shadow.xy);
        float occ = 1.0 - smoothstep(0.0, pc.ground_shadow.z, d);
        occ = occ * occ;  // tighten the core, soften the skirt
        color *= (1.0 - pc.ground_shadow.w * occ);
    }

    // Atmospheric HAZE (Go2-terrain demo: the Isaac-Lab look). Exponential
    // distance fog fading the fragment toward a bright horizon colour by the
    // distance from the camera eye. When scene.fog.w (density) == 0 the factor is
    // 1 - exp(0) == 0, so color is UNCHANGED -> byte-identical to the pre-fog
    // oracle for every non-fog render (G2-safe). Applied in linear HDR space so it
    // tonemaps together with the lit surface (a natural, photographic fade).
    if (scene.fog.w > 0.0) {
        float view_dist = length(vWorldPos - scene.camera_pos.xyz);
        float fog_amt   = 1.0 - exp(-scene.fog.w * view_dist);
        color = mix(color, scene.fog.rgb, clamp(fog_amt, 0.0, 1.0));
    }

    // Tonemap (HDR -> LDR) then sRGB encode.
    color = AcesFilmic(color);
    color = LinearToSrgb(color);

    oColor = vec4(color, opacity * vBaseColor.a);
}
