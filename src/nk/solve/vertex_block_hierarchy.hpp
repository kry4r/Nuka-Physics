#pragma once

#include <cstdint>

namespace nuka::nk {

class Model;

// Each vertex interpolates at most this many nodes of a level.
inline constexpr uint32_t kVbdCoarseParents = 3u;
// Coarsening stops at a level of at most this many nodes, which is then solved densely.
inline constexpr uint32_t kVbdCoarseDenseNodes = 64u;

// Coarse spaces of the vertex blocks on the rest metric. Each level samples vertices at a doubling
// geodesic spacing as nodes; every vertex interpolates its level's nodes, reproducing linear fields.
void BuildVertexBlockHierarchy(Model* model);

}  // namespace nuka::nk
