// ---------------------------------------------------------------------------
// nuka::collision::gpu -- compiled home for the shared batched LBVH build.
//
// The build/refit launchers + their env-kernels live in lbvh_batched.cuh as
// inline / static __global__ (one source copy, internal-linkage per TU). This
// TU includes the header so the primitive has a compiled home in this library
// and the header is verified to build standalone here. Both the pair-driven
// collision broadphase op and the render TLAS include the same header.
// ---------------------------------------------------------------------------

#include "collision/lbvh_batched.cuh"
