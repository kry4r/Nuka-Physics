#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace nuka::nk {

struct ArticulationTree {
    std::vector<uint32_t> depth;
    std::vector<uint32_t> max_depth;
    std::vector<std::array<uint32_t, 2>> child_range;
    std::vector<uint32_t> children;
};

inline ArticulationTree BuildArticulationTree(const std::vector<uint32_t>& parents) {
    const uint32_t count = static_cast<uint32_t>(parents.size());
    ArticulationTree tree;
    tree.depth.resize(count);
    tree.child_range.resize(count);
    tree.children.resize(count);
    std::vector<std::vector<uint32_t>> children(count);
    uint32_t root = 0u;
    for (uint32_t link = 0u; link < count; ++link) {
        if (parents[link] == ~0u) {
            root = link;
            tree.max_depth.push_back(0u);
        } else {
            const uint32_t parent = root + parents[link];
            tree.depth[link] = tree.depth[parent] + 1u;
            tree.max_depth.back() = std::max(tree.max_depth.back(), tree.depth[link]);
            children[parent].push_back(link);
        }
    }
    uint32_t cursor = 0u;
    for (uint32_t link = 0u; link < count; ++link) {
        tree.child_range[link][0] = cursor;
        for (auto child = children[link].rbegin(); child != children[link].rend(); ++child)
            tree.children[cursor++] = *child;
        tree.child_range[link][1] = cursor;
    }
    return tree;
}

} // namespace nuka::nk
