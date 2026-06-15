#version 450
// ---------------------------------------------------------------------------
// nuka::render::raster SKY GRADIENT fullscreen fragment shader.
//
// Lerps a vertical gradient between sky_bottom (horizon, vT=0) and sky_top
// (zenith, vT=1), smoothstepped for a soft band, then sRGB-encodes it (the colours
// are authored linear, matching the scene's linear->sRGB output so the sky and the
// fogged terrain meet seamlessly). The two colours come in as push constants.
// G2-SAFE: only used when RasterOptions.sky_gradient is true.
// ---------------------------------------------------------------------------

layout(location = 0) in float vT;   // 0 = horizon, 1 = zenith
layout(location = 0) out vec4 oColor;

layout(push_constant) uniform SkyPush {
    vec4 sky_top;     // rgb zenith (linear)
    vec4 sky_bottom;  // rgb horizon (linear)
} pc;

vec3 LinearToSrgb(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(hi, lo, lessThanEqual(c, vec3(0.0031308)));
}

void main() {
    float t = smoothstep(0.0, 1.0, clamp(vT, 0.0, 1.0));
    vec3 c = mix(pc.sky_bottom.rgb, pc.sky_top.rgb, t);
    oColor = vec4(LinearToSrgb(c), 1.0);
}
