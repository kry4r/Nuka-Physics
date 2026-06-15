#version 450
// ---------------------------------------------------------------------------
// nuka::render::raster SKY GRADIENT fullscreen vertex shader.
//
// Emits a single fullscreen triangle from gl_VertexIndex (no vertex buffer): the
// classic (-1,-1)/(3,-1)/(-1,3) cover triangle. Passes a 0..1 vertical coordinate
// (vT: 0 at the bottom of the frame = horizon, 1 at the top = zenith) to the
// fragment, which lerps the two sky colours. Drawn BEFORE the scene with depth
// test/write OFF so the scene overwrites it. G2-SAFE: only recorded when
// RasterOptions.sky_gradient is true (the gated smokes never enter it).
// ---------------------------------------------------------------------------

layout(location = 0) out float vT;  // 0 = bottom/horizon, 1 = top/zenith

void main() {
    // Fullscreen triangle: positions in clip space.
    vec2 p = vec2((gl_VertexIndex == 1) ? 3.0 : -1.0,
                  (gl_VertexIndex == 2) ? 3.0 : -1.0);
    // Vulkan clip space has y DOWN, so clip-y = -1 is the TOP of the frame. We want
    // vT = 1 at the top (zenith): vT = (1 - clip_y) * 0.5 maps clip_y=-1 -> 1 (top),
    // clip_y=+1 -> 0 (bottom/horizon).
    vT = (1.0 - p.y) * 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
