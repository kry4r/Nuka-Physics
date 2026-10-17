#version 450
// The viewport's id pass: each draw writes the id its push block carries after the material.

layout(push_constant) uniform IdBlock {
    layout(offset = 112) uint id;
} pc;

layout(location = 0) out uint oId;

void main() {
    oId = pc.id;
}
