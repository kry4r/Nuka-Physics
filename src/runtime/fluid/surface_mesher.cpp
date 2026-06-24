// ---------------------------------------------------------------------------
// nuka::runtime::fluid -- PBF particle-set -> marching-cubes isosurface (host).
// ---------------------------------------------------------------------------
// HOST cook/render-bridge code (no device): samples the SPH density field on a
// uniform voxel grid (uniform-grid neighbor accel), marches cubes to a watertight
// triangle MeshGeometry, and assigns per-vertex normals from the analytic density
// gradient. Pure host so nuka_render links it without CUDA.
// ---------------------------------------------------------------------------

#include "runtime/fluid/surface_mesher.hpp"

#include "import/cooker/fluid_cooker_types.hpp"  // Poly6FromR2Host / Poly6GradientHost

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace nuka::runtime::fluid {
namespace {

// Standard marching-cubes tables (Lorensen & Cline 1987; the canonical
// Bourke/Geiss 256-case edge mask + per-case triangle edge list). edge_table[c]
// is the 12-bit mask of cube edges the isosurface crosses for corner-sign code c;
// tri_table[c] lists the crossed edges three-per-triangle, terminated by -1.
const int kEdgeTable[256] = {
    0x0,   0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905,
    0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 0x190, 0x99,  0x393, 0x29a,
    0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93,
    0xf99, 0xe90, 0x230, 0x339, 0x33,  0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9,
    0x1a3, 0xaa,  0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6,
    0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569, 0x663, 0x76a, 0x66,  0x16f,
    0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff,  0x3f5, 0x2fc, 0xdfc, 0xcf5,
    0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a,
    0x256, 0x35f, 0x55,  0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53,
    0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9,
    0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc,  0x1c5, 0x2cf, 0x3c6,
    0x4ca, 0x5c3, 0x6c9, 0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f,
    0xf55, 0xe5c, 0x15c, 0x55,  0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5,
    0xff,  0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a,
    0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x66,  0x76a, 0x663,
    0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa,  0x1a3, 0x2a9, 0x3a0, 0xd30, 0xc39,
    0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636,
    0x13a, 0x33,  0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f,
    0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99,  0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605,
    0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0};

const int kTriTable[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
    {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
    {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
    {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
    {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
    {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
    {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
    {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
    {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
    {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
    {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
    {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
    {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
    {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
    {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
    {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
    {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
    {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
    {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
    {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
    {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
    {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
    {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
    {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
    {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
    {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
    {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
    {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
    {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
    {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
    {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
    {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
    {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
    {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
    {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
    {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
    {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
    {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
    {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
    {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
    {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
    {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
    {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
    {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
    {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
    {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
    {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
    {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
    {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
    {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
    {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
    {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
    {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
    {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
    {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
    {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
    {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
    {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
    {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
    {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
    {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
    {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
    {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
    {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
    {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
    {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
    {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
    {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
    {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
    {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
    {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
    {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
    {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
    {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
    {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
    {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
    {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
    {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
    {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};

// The 8 cube corners (unit-cube local offsets) and the 12 edges as corner pairs,
// in the Lorensen-Cline convention the tables above are indexed against.
const int kCorner[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                           {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
const int kEdge[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                          {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

// A uniform grid of particle indices for O(neighbors) density queries. Cells are
// sized to the SPH support radius h so a query gathers from the 3x3x3 block.
struct ParticleGrid {
    math::Vec3 origin{0.0f, 0.0f, 0.0f};
    float cell = 1.0f;
    int nx = 0, ny = 0, nz = 0;
    std::vector<int> cell_start;     // CSR offsets into `order` (nx*ny*nz + 1).
    std::vector<uint32_t> order;     // particle indices grouped by cell.

    int Clamp(int v, int hi) const { return v < 0 ? 0 : (v >= hi ? hi - 1 : v); }
    int CellOf(int ix, int iy, int iz) const { return (iz * ny + iy) * nx + ix; }
};

ParticleGrid BuildParticleGrid(const std::vector<math::Vec3>& pts, float h,
                               const math::Vec3& lo, const math::Vec3& hi) {
    ParticleGrid g;
    g.cell = h;
    g.origin = lo;
    g.nx = std::max(1, static_cast<int>(std::floor((hi.x - lo.x) / h)) + 1);
    g.ny = std::max(1, static_cast<int>(std::floor((hi.y - lo.y) / h)) + 1);
    g.nz = std::max(1, static_cast<int>(std::floor((hi.z - lo.z) / h)) + 1);
    const int ncell = g.nx * g.ny * g.nz;
    std::vector<int> count(static_cast<size_t>(ncell), 0);
    auto cell_index = [&](const math::Vec3& p) {
        const int ix = g.Clamp(static_cast<int>(std::floor((p.x - lo.x) / h)), g.nx);
        const int iy = g.Clamp(static_cast<int>(std::floor((p.y - lo.y) / h)), g.ny);
        const int iz = g.Clamp(static_cast<int>(std::floor((p.z - lo.z) / h)), g.nz);
        return g.CellOf(ix, iy, iz);
    };
    for (const auto& p : pts) {
        ++count[static_cast<size_t>(cell_index(p))];
    }
    g.cell_start.assign(static_cast<size_t>(ncell) + 1u, 0);
    for (int c = 0; c < ncell; ++c) {
        g.cell_start[static_cast<size_t>(c) + 1u] = g.cell_start[static_cast<size_t>(c)] + count[static_cast<size_t>(c)];
    }
    g.order.assign(pts.size(), 0u);
    std::vector<int> cursor(g.cell_start.begin(), g.cell_start.end() - 1);
    // Particles enter their cell bucket in ascending original index, so the
    // neighbor accumulation order is fixed run-to-run (D1).
    for (uint32_t i = 0; i < pts.size(); ++i) {
        const int c = cell_index(pts[i]);
        g.order[static_cast<size_t>(cursor[static_cast<size_t>(c)]++)] = i;
    }
    return g;
}

// Density and (optionally) its gradient at x, from particles within h via the
// grid's 3x3x3 neighborhood. Accumulates in ascending neighbor-index order (D1).
float DensityAt(const math::Vec3& x, const std::vector<math::Vec3>& pts,
                const ParticleGrid& g, const FluidSurfaceParams& p,
                math::Vec3* out_grad) {
    const int cx = g.Clamp(static_cast<int>(std::floor((x.x - g.origin.x) / g.cell)), g.nx);
    const int cy = g.Clamp(static_cast<int>(std::floor((x.y - g.origin.y) / g.cell)), g.ny);
    const int cz = g.Clamp(static_cast<int>(std::floor((x.z - g.origin.z) / g.cell)), g.nz);
    float rho = 0.0f;
    math::Vec3 grad{0.0f, 0.0f, 0.0f};
    for (int dz = -1; dz <= 1; ++dz) {
        const int iz = cz + dz;
        if (iz < 0 || iz >= g.nz) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            const int iy = cy + dy;
            if (iy < 0 || iy >= g.ny) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                const int ix = cx + dx;
                if (ix < 0 || ix >= g.nx) continue;
                const int c = g.CellOf(ix, iy, iz);
                const int b = g.cell_start[static_cast<size_t>(c)];
                const int e = g.cell_start[static_cast<size_t>(c) + 1u];
                for (int k = b; k < e; ++k) {
                    const math::Vec3 d = x - pts[g.order[static_cast<size_t>(k)]];
                    const float r2 = d.Dot(d);
                    rho += p.particle_mass * Poly6FromR2Host(r2, p.h);
                    if (out_grad != nullptr) {
                        // Analytic gradient of the SAME Poly6 field (not the Spiky
                        // pressure gradient); -grad points outward for the normal.
                        grad += p.particle_mass * Poly6GradientHost(d, r2, p.h);
                    }
                }
            }
        }
    }
    if (out_grad != nullptr) *out_grad = grad;
    return rho;
}

// Per-particle anisotropic kernel transform G (Yu & Turk 2013, Eq. 16): a symmetric
// PD 3x3 mapping world offsets into the unit kernel space so a stretched water sheet
// flattens and an isolated splash droplet stays round. `det` caches det(G).
struct AnisoKernel {
    float g[6];  // packed symmetric: xx, yy, zz, xy, xz, yz.
    float det;
    math::Vec3 center;  // smoothed kernel center x-bar (Eq. 6).
};

// Cyclic Jacobi eigensolve of a symmetric 3x3 into descending eigenvalues `w` and
// orthonormal columns `V` (Eq. 12-13). Fixed sweep count keeps it deterministic.
void JacobiEig3(const float a[6], float w[3], float V[9]) {
    float A[3][3] = {{a[0], a[3], a[4]}, {a[3], a[1], a[5]}, {a[4], a[5], a[2]}};
    V[0] = V[4] = V[8] = 1.0f;
    V[1] = V[2] = V[3] = V[5] = V[6] = V[7] = 0.0f;
    for (int sweep = 0; sweep < 12; ++sweep) {
        const int pq[3][2] = {{0, 1}, {0, 2}, {1, 2}};
        for (int s = 0; s < 3; ++s) {
            const int pp = pq[s][0], qq = pq[s][1];
            const float apq = A[pp][qq];
            if (std::fabs(apq) < 1e-20f) continue;
            const float theta = (A[qq][qq] - A[pp][pp]) / (2.0f * apq);
            const float t = (theta >= 0.0f ? 1.0f : -1.0f) /
                            (std::fabs(theta) + std::sqrt(theta * theta + 1.0f));
            const float c = 1.0f / std::sqrt(t * t + 1.0f);
            const float sn = t * c;
            for (int k = 0; k < 3; ++k) {
                const float akp = A[k][pp], akq = A[k][qq];
                A[k][pp] = c * akp - sn * akq;
                A[k][qq] = sn * akp + c * akq;
            }
            for (int k = 0; k < 3; ++k) {
                const float apk = A[pp][k], aqk = A[qq][k];
                A[pp][k] = c * apk - sn * aqk;
                A[qq][k] = sn * apk + c * aqk;
            }
            for (int k = 0; k < 3; ++k) {
                const float vkp = V[k * 3 + pp], vkq = V[k * 3 + qq];
                V[k * 3 + pp] = c * vkp - sn * vkq;
                V[k * 3 + qq] = sn * vkp + c * vkq;
            }
        }
    }
    w[0] = A[0][0]; w[1] = A[1][1]; w[2] = A[2][2];
    // Sort eigenpairs descending so sigma1 >= sigma2 >= sigma3 (Eq. 12).
    for (int i = 0; i < 2; ++i)
        for (int j = i + 1; j < 3; ++j)
            if (w[j] > w[i]) {
                std::swap(w[i], w[j]);
                for (int k = 0; k < 3; ++k) std::swap(V[k * 3 + i], V[k * 3 + j]);
            }
}

// Build the per-particle anisotropic transforms G (Eq. 6-16) over the particle set.
// Neighbors are gathered within 2h (the grid cell == h, so a 5x5x5 block) so the
// covariance/smoothing see the full kernel support. Reconstruction only -- the
// smoothed centers never feed back to the sim. Ascending neighbor order keeps D1.
std::vector<AnisoKernel> BuildAnisoKernels(const std::vector<math::Vec3>& pts,
                                           const ParticleGrid& g,
                                           const FluidSurfaceParams& p) {
    const size_t n = pts.size();
    const float h = p.h;
    const float two_h = 2.0f * h;
    const float two_h2 = two_h * two_h;
    auto weight = [&](float r) {
        if (r >= two_h) return 0.0f;
        const float u = r / two_h;
        return 1.0f - u * u * u;  // Eq. 11.
    };
    auto gather = [&](const math::Vec3& x, auto&& fn) {
        const int cx = g.Clamp(static_cast<int>(std::floor((x.x - g.origin.x) / g.cell)), g.nx);
        const int cy = g.Clamp(static_cast<int>(std::floor((x.y - g.origin.y) / g.cell)), g.ny);
        const int cz = g.Clamp(static_cast<int>(std::floor((x.z - g.origin.z) / g.cell)), g.nz);
        for (int dz = -2; dz <= 2; ++dz) {
            const int iz = cz + dz; if (iz < 0 || iz >= g.nz) continue;
            for (int dy = -2; dy <= 2; ++dy) {
                const int iy = cy + dy; if (iy < 0 || iy >= g.ny) continue;
                for (int dx = -2; dx <= 2; ++dx) {
                    const int ix = cx + dx; if (ix < 0 || ix >= g.nx) continue;
                    const int c = g.CellOf(ix, iy, iz);
                    const int b = g.cell_start[static_cast<size_t>(c)];
                    const int e = g.cell_start[static_cast<size_t>(c) + 1u];
                    for (int k = b; k < e; ++k) fn(g.order[static_cast<size_t>(k)]);
                }
            }
        }
    };

    // Eq. 6: Laplacian position pre-smooth -> kernel centers x-bar.
    std::vector<math::Vec3> centers(n);
    for (size_t i = 0; i < n; ++i) {
        const math::Vec3 xi = pts[i];
        math::Vec3 sum{0, 0, 0}; float wsum = 0.0f;
        gather(xi, [&](uint32_t j) {
            const math::Vec3 d = pts[j] - xi;
            const float w = weight(std::sqrt(d.Dot(d)));
            if (w <= 0.0f) return;
            sum += pts[j] * w; wsum += w;
        });
        const math::Vec3 mean = wsum > 0.0f ? sum * (1.0f / wsum) : xi;
        centers[i] = xi * (1.0f - p.aniso_lambda) + mean * p.aniso_lambda;
    }

    std::vector<AnisoKernel> out(n);
    for (size_t i = 0; i < n; ++i) {
        const math::Vec3 xi = pts[i];
        // Eq. 10: weighted mean position of the neighborhood.
        math::Vec3 mean{0, 0, 0}; float wsum = 0.0f; uint32_t cnt = 0u;
        gather(xi, [&](uint32_t j) {
            const math::Vec3 d = pts[j] - xi;
            if (d.Dot(d) >= two_h2) return;
            const float w = weight(std::sqrt(d.Dot(d)));
            if (w <= 0.0f) return;
            mean += pts[j] * w; wsum += w; ++cnt;
        });
        if (wsum > 0.0f) mean = mean * (1.0f / wsum);
        // Eq. 9: weighted covariance about the weighted mean.
        float cov[6] = {0, 0, 0, 0, 0, 0}; float cwsum = 0.0f;
        gather(xi, [&](uint32_t j) {
            const math::Vec3 d0 = pts[j] - xi;
            if (d0.Dot(d0) >= two_h2) return;
            const float w = weight(std::sqrt(d0.Dot(d0)));
            if (w <= 0.0f) return;
            const math::Vec3 d = pts[j] - mean;
            cov[0] += w * d.x * d.x; cov[1] += w * d.y * d.y; cov[2] += w * d.z * d.z;
            cov[3] += w * d.x * d.y; cov[4] += w * d.x * d.z; cov[5] += w * d.y * d.z;
            cwsum += w;
        });
        AnisoKernel ak;
        ak.center = centers[i];
        if (cwsum <= 0.0f || cnt <= 1u) {
            // No neighborhood: a spherical kernel of radius ~h.
            const float s = p.aniso_kn / h;
            ak.g[0] = ak.g[1] = ak.g[2] = s; ak.g[3] = ak.g[4] = ak.g[5] = 0.0f;
            ak.det = s * s * s;
            out[i] = ak; continue;
        }
        for (float& c : cov) c /= cwsum;
        float w[3], V[9];
        JacobiEig3(cov, w, V);
        // Eq. 12-15: clamp small eigenvalues, scale by k_s, or go isotropic if isolated.
        float sigma[3];
        if (cnt > p.aniso_n_eps) {
            const float s1 = w[0];
            for (int k = 0; k < 3; ++k)
                sigma[k] = p.aniso_ks * std::max(w[k], s1 / p.aniso_kr);
        } else {
            sigma[0] = sigma[1] = sigma[2] = p.aniso_kn;
        }
        // Eq. 16: G = (1/h) R Sigma^{-1} R^T. Sigma diagonal -> inverse is reciprocal.
        const float inv[3] = {1.0f / sigma[0], 1.0f / sigma[1], 1.0f / sigma[2]};
        float G[6];
        G[0] = (V[0] * V[0] * inv[0] + V[1] * V[1] * inv[1] + V[2] * V[2] * inv[2]);
        G[1] = (V[3] * V[3] * inv[0] + V[4] * V[4] * inv[1] + V[5] * V[5] * inv[2]);
        G[2] = (V[6] * V[6] * inv[0] + V[7] * V[7] * inv[1] + V[8] * V[8] * inv[2]);
        G[3] = (V[0] * V[3] * inv[0] + V[1] * V[4] * inv[1] + V[2] * V[5] * inv[2]);
        G[4] = (V[0] * V[6] * inv[0] + V[1] * V[7] * inv[1] + V[2] * V[8] * inv[2]);
        G[5] = (V[3] * V[6] * inv[0] + V[4] * V[7] * inv[1] + V[5] * V[8] * inv[2]);
        const float inv_h = 1.0f / h;
        for (float& gg : G) gg *= inv_h;
        for (int k = 0; k < 6; ++k) ak.g[k] = G[k];
        // det(G) = (1/h)^3 / (sigma1 sigma2 sigma3) (R orthonormal).
        ak.det = (inv_h * inv_h * inv_h) / (sigma[0] * sigma[1] * sigma[2]);
        out[i] = ak;
    }
    return out;
}

// Anisotropic density at x (Eq. 7-8): sum_j det(G_j) * P(||G_j (x - center_j)||)
// over particles whose stretched kernel reaches x (2h grid block). The radial
// profile P is the SAME Poly6 the isotropic field uses, evaluated at the kernel-
// space radius mapped back into [0,h]. Accumulates ascending neighbor order (D1).
float DensityAtAniso(const math::Vec3& x, const std::vector<AnisoKernel>& ker,
                     const ParticleGrid& g, const FluidSurfaceParams& p) {
    const int cx = g.Clamp(static_cast<int>(std::floor((x.x - g.origin.x) / g.cell)), g.nx);
    const int cy = g.Clamp(static_cast<int>(std::floor((x.y - g.origin.y) / g.cell)), g.ny);
    const int cz = g.Clamp(static_cast<int>(std::floor((x.z - g.origin.z) / g.cell)), g.nz);
    const float h = p.h;
    float rho = 0.0f;
    for (int dz = -2; dz <= 2; ++dz) {
        const int iz = cz + dz; if (iz < 0 || iz >= g.nz) continue;
        for (int dy = -2; dy <= 2; ++dy) {
            const int iy = cy + dy; if (iy < 0 || iy >= g.ny) continue;
            for (int dx = -2; dx <= 2; ++dx) {
                const int ix = cx + dx; if (ix < 0 || ix >= g.nx) continue;
                const int c = g.CellOf(ix, iy, iz);
                const int b = g.cell_start[static_cast<size_t>(c)];
                const int e = g.cell_start[static_cast<size_t>(c) + 1u];
                for (int k = b; k < e; ++k) {
                    const uint32_t j = g.order[static_cast<size_t>(k)];
                    const AnisoKernel& a = ker[j];
                    const math::Vec3 d = x - a.center;
                    // r' = G_j d (symmetric G); kernel-space radius in [0, h].
                    const math::Vec3 gr{
                        a.g[0] * d.x + a.g[3] * d.y + a.g[4] * d.z,
                        a.g[3] * d.x + a.g[1] * d.y + a.g[5] * d.z,
                        a.g[4] * d.x + a.g[5] * d.y + a.g[2] * d.z};
                    const float kr2 = gr.Dot(gr);   // (||G d|| in 1/h units).
                    const float r2 = kr2 * h * h;    // map back to [0,h] for Poly6.
                    if (r2 >= h * h) continue;
                    rho += p.particle_mass * a.det * h * h * h * Poly6FromR2Host(r2, h);
                }
            }
        }
    }
    return rho;
}

}  // namespace

render::MeshGeometry MarchFluidSurface(const std::vector<math::Vec3>& particle_positions,
                                       const FluidSurfaceParams& p) {
    render::MeshGeometry out;
    out.uvs.clear();
    const size_t n = particle_positions.size();
    if (n == 0u || p.h <= 0.0f) {
        return out;
    }

    // Particle AABB, padded by h so the sampled field fully encloses the surface
    // (density falls to zero by h beyond the outermost particle).
    math::Vec3 lo = particle_positions[0];
    math::Vec3 hi = particle_positions[0];
    for (const auto& q : particle_positions) {
        lo.x = std::min(lo.x, q.x); lo.y = std::min(lo.y, q.y); lo.z = std::min(lo.z, q.z);
        hi.x = std::max(hi.x, q.x); hi.y = std::max(hi.y, q.y); hi.z = std::max(hi.z, q.z);
    }
    const float h = p.h;
    const ParticleGrid grid = BuildParticleGrid(particle_positions, h, lo, hi);

    const float ch = p.CellSize();
    const math::Vec3 grid_lo{lo.x - h, lo.y - h, lo.z - h};
    const math::Vec3 grid_hi{hi.x + h, hi.y + h, hi.z + h};
    const int gx = std::max(1, static_cast<int>(std::ceil((grid_hi.x - grid_lo.x) / ch)));
    const int gy = std::max(1, static_cast<int>(std::ceil((grid_hi.y - grid_lo.y) / ch)));
    const int gz = std::max(1, static_cast<int>(std::ceil((grid_hi.z - grid_lo.z) / ch)));
    float iso = p.iso_fraction * p.rest_density_rho0;

    // Anisotropic kernels (opt-in): precompute the per-particle ellipsoid transforms.
    // The unified `density`/`gradient` close over the active field so the rest of the
    // marcher is identical for both paths.
    const std::vector<AnisoKernel> ker =
        p.anisotropic ? BuildAnisoKernels(particle_positions, grid, p)
                      : std::vector<AnisoKernel>{};
    auto density = [&](const math::Vec3& x) -> float {
        return p.anisotropic ? DensityAtAniso(x, ker, grid, p)
                             : DensityAt(x, particle_positions, grid, p, nullptr);
    };
    if (p.anisotropic) {
        // Calibrate the iso level to the anisotropic field's own interior density so
        // `iso_fraction` keeps its meaning regardless of k_s units: sample the field
        // at every smoothed kernel center and take a robust upper percentile as rho0.
        std::vector<float> rho_at(n, 0.0f);
        for (size_t i = 0; i < n; ++i) rho_at[i] = density(ker[i].center);
        std::sort(rho_at.begin(), rho_at.end());
        const float rho0 = rho_at[(rho_at.size() * 90u) / 100u];
        iso = p.iso_fraction * rho0;
    }
    auto gradient = [&](const math::Vec3& x) -> math::Vec3 {
        if (!p.anisotropic) {
            math::Vec3 grad{0, 0, 0};
            DensityAt(x, particle_positions, grid, p, &grad);
            return grad;
        }
        // Central finite difference of the anisotropic field (the analytic Poly6
        // gradient no longer matches a stretched kernel).
        const float e = 0.25f * ch;
        return math::Vec3{
            (density(math::Vec3{x.x + e, x.y, x.z}) - density(math::Vec3{x.x - e, x.y, x.z})) / (2.0f * e),
            (density(math::Vec3{x.x, x.y + e, x.z}) - density(math::Vec3{x.x, x.y - e, x.z})) / (2.0f * e),
            (density(math::Vec3{x.x, x.y, x.z + e}) - density(math::Vec3{x.x, x.y, x.z - e})) / (2.0f * e)};
    };

    // Sample the density field at every voxel corner once (cached so each sample
    // costs one neighbor query, not eight). Corner (i,j,k) -> linear (k*(gy+1)+j)*(gx+1)+i.
    const int sx = gx + 1, sy = gy + 1, sz = gz + 1;
    auto sample_pos = [&](int i, int j, int k) {
        return math::Vec3{grid_lo.x + ch * static_cast<float>(i),
                          grid_lo.y + ch * static_cast<float>(j),
                          grid_lo.z + ch * static_cast<float>(k)};
    };
    auto sidx = [&](int i, int j, int k) {
        return (static_cast<size_t>(k) * sy + j) * sx + i;
    };
    std::vector<float> field(static_cast<size_t>(sx) * sy * sz, 0.0f);
    for (int k = 0; k < sz; ++k) {
        for (int j = 0; j < sy; ++j) {
            for (int i = 0; i < sx; ++i) {
                field[sidx(i, j, k)] = density(sample_pos(i, j, k));
            }
        }
    }

    // Edge-keyed vertex dedup. The key is the canonical grid-edge id (its lower
    // corner + axis); an ordered map keeps first-seen indices deterministic.
    std::map<uint64_t, uint32_t> edge_vert;

    const uint64_t kdy = static_cast<uint64_t>(sy) + 1u;
    const uint64_t kdz = static_cast<uint64_t>(sz) + 1u;
    auto edge_key = [&](int ci, int cj, int ck, int e) -> uint64_t {
        // The two corners of edge e on cube (ci,cj,ck), as absolute corner coords.
        const int a = kEdge[e][0], b = kEdge[e][1];
        int ax = ci + kCorner[a][0], ay = cj + kCorner[a][1], az = ck + kCorner[a][2];
        int bx = ci + kCorner[b][0], by = cj + kCorner[b][1], bz = ck + kCorner[b][2];
        // Canonicalize to the lower corner + an axis 0/1/2; pack against the actual
        // grid extent so distinct edges never collide (no magic dimension bound).
        int lx = std::min(ax, bx), ly = std::min(ay, by), lz = std::min(az, bz);
        int axis = (ax != bx) ? 0 : (ay != by) ? 1 : 2;
        uint64_t key = static_cast<uint64_t>(lx);
        key = key * kdy + static_cast<uint64_t>(ly);
        key = key * kdz + static_cast<uint64_t>(lz);
        key = key * 3u + static_cast<uint64_t>(axis);
        return key;
    };

    auto emit_edge_vertex = [&](int ci, int cj, int ck, int e) -> uint32_t {
        const uint64_t key = edge_key(ci, cj, ck, e);
        auto it = edge_vert.find(key);
        if (it != edge_vert.end()) {
            return it->second;
        }
        const int a = kEdge[e][0], b = kEdge[e][1];
        const int aix = ci + kCorner[a][0], aiy = cj + kCorner[a][1], aiz = ck + kCorner[a][2];
        const int bix = ci + kCorner[b][0], biy = cj + kCorner[b][1], biz = ck + kCorner[b][2];
        const float da = field[sidx(aix, aiy, aiz)];
        const float db = field[sidx(bix, biy, biz)];
        const math::Vec3 pa = sample_pos(aix, aiy, aiz);
        const math::Vec3 pb = sample_pos(bix, biy, biz);
        // Linear interpolation of the crossing along the density field.
        float t = 0.5f;
        const float denom = db - da;
        if (denom != 0.0f) t = (iso - da) / denom;
        if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
        const math::Vec3 vp{pa.x + t * (pb.x - pa.x), pa.y + t * (pb.y - pa.y),
                            pa.z + t * (pb.z - pa.z)};
        const uint32_t vid = static_cast<uint32_t>(out.positions.size() / 3u);
        out.positions.push_back(vp.x);
        out.positions.push_back(vp.y);
        out.positions.push_back(vp.z);
        // Outward normal = -grad(density) (toward decreasing density); fall back to
        // +Z if the gradient is degenerate so the shader never normalizes zero.
        const math::Vec3 grad = gradient(vp);
        math::Vec3 nrm{-grad.x, -grad.y, -grad.z};
        const float nl = std::sqrt(nrm.Dot(nrm));
        if (nl > 1e-12f) {
            nrm.x /= nl; nrm.y /= nl; nrm.z /= nl;
        } else {
            nrm = math::Vec3{0.0f, 0.0f, 1.0f};
        }
        out.normals.push_back(nrm.x);
        out.normals.push_back(nrm.y);
        out.normals.push_back(nrm.z);
        edge_vert.emplace(key, vid);
        return vid;
    };

    // March cubes in a fixed z->y->x order (D1). For each cube classify the 8
    // corners against iso, emit triangles per the standard table.
    for (int ck = 0; ck < gz; ++ck) {
        for (int cj = 0; cj < gy; ++cj) {
            for (int ci = 0; ci < gx; ++ci) {
                int code = 0;
                float dval[8];
                for (int c = 0; c < 8; ++c) {
                    const int vi = ci + kCorner[c][0];
                    const int vj = cj + kCorner[c][1];
                    const int vk = ck + kCorner[c][2];
                    dval[c] = field[sidx(vi, vj, vk)];
                    if (dval[c] >= iso) code |= (1 << c);
                }
                const int em = kEdgeTable[code];
                if (em == 0) continue;
                int edge_vid[12];
                for (int e = 0; e < 12; ++e) {
                    edge_vid[e] = (em & (1 << e)) ? static_cast<int>(emit_edge_vertex(ci, cj, ck, e)) : -1;
                }
                for (int t = 0; kTriTable[code][t] != -1; t += 3) {
                    uint32_t a = static_cast<uint32_t>(edge_vid[kTriTable[code][t + 0]]);
                    uint32_t b = static_cast<uint32_t>(edge_vid[kTriTable[code][t + 1]]);
                    uint32_t c = static_cast<uint32_t>(edge_vid[kTriTable[code][t + 2]]);
                    // Orient by the density gradient (outward = -grad) at the centroid
                    // so winding is consistent regardless of the table's convention.
                    const math::Vec3 pa{out.positions[a * 3], out.positions[a * 3 + 1], out.positions[a * 3 + 2]};
                    const math::Vec3 pb{out.positions[b * 3], out.positions[b * 3 + 1], out.positions[b * 3 + 2]};
                    const math::Vec3 pc{out.positions[c * 3], out.positions[c * 3 + 1], out.positions[c * 3 + 2]};
                    const math::Vec3 gn = (pb - pa).Cross(pc - pa);
                    const math::Vec3 ctr{(pa.x + pb.x + pc.x) / 3.0f, (pa.y + pb.y + pc.y) / 3.0f,
                                         (pa.z + pb.z + pc.z) / 3.0f};
                    const math::Vec3 grad = gradient(ctr);
                    if (gn.Dot(math::Vec3{-grad.x, -grad.y, -grad.z}) < 0.0f) {
                        std::swap(b, c);  // flip to outward winding.
                    }
                    out.indices.push_back(a);
                    out.indices.push_back(b);
                    out.indices.push_back(c);
                }
            }
        }
    }
    return out;
}

}  // namespace nuka::runtime::fluid
