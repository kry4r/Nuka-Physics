#version 450
// The selection outline: a pixel just outside a selected instance of the id image takes the
// selection colour, one just outside a hovered instance the hover colour.

layout(set = 0, binding = 0) uniform usampler2D uIds;

layout(push_constant) uniform OutlineBlock {
    vec4 selected;  // straight alpha
    vec4 hovered;
    ivec4 rect;     // the view's x, y, width and height in framebuffer pixels
} pc;

layout(location = 0) out vec4 oColor;

const uint kSelected = 0x80000000u;
const uint kHovered = 0x40000000u;
const int kRadius = 2;

void main() {
    const ivec2 p = ivec2(gl_FragCoord.xy);
    const ivec2 lo = pc.rect.xy, hi = pc.rect.xy + pc.rect.zw - 1;
    const uint self = texelFetch(uIds, p, 0).r;
    bool near_selected = false, near_hovered = false;
    for (int dy = -kRadius; dy <= kRadius; ++dy) {
        for (int dx = -kRadius; dx <= kRadius; ++dx) {
            const uint id = texelFetch(uIds, clamp(p + ivec2(dx, dy), lo, hi), 0).r;
            near_selected = near_selected || (id & kSelected) != 0u;
            near_hovered = near_hovered || (id & kHovered) != 0u;
        }
    }
    if (near_selected && (self & kSelected) == 0u) oColor = pc.selected;
    else if (near_hovered && (self & (kSelected | kHovered)) == 0u) oColor = pc.hovered;
    else discard;
}
