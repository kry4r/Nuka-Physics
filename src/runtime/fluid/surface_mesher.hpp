#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::fluid -- the ONE general PBF particle-set -> render surface mesh.
//
// Turns a fluid particle set into a watertight, smooth-shaded triangle
// MeshGeometry by marching cubes over the SPH density field the fluid itself
// defines: density(x) = sum_p m * Poly6(|x - p|^2, h), iso = iso_fraction * rho0.
// The result flows through the SAME rt_adapter MeshToBlas path the robot/cloth
// meshes use -- the medium is data (an isosurface), never a per-demo render fork.
//
// HOST-ONLY, ZERO CUDA: a pure-host TU so nuka_render links it without pulling
// CUDA. The density Poly6 and the gradient Spiky kernel are the CUDA-free host
// mirrors in import/cooker/fluid_cooker_types.hpp (one shared definition each).
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "render/render_world.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime::fluid {

// Parameters of the marching-cubes isosurface over a PBF particle set. A CUDA-free
// POD with physically grounded defaults: `h` is the PBF support radius, the iso
// level is `iso_fraction * rest_density_rho0`, and `cell_size` is the MC voxel
// edge (a fraction of `h`; <= 0 => derived from `h` at cook time).
struct FluidSurfaceParams {
    float h = 0.1f;                  // SPH support radius (PBF kernel support).
    float rest_density_rho0 = 1.0f;  // rho0; iso = iso_fraction * rho0.
    float iso_fraction = 0.5f;       // surface at this fraction of rest density.
    float particle_mass = 1.0f;      // m (uniform across the fluid).
    float cell_size = 0.0f;          // MC voxel edge; <= 0 => 0.5 * h.

    // Anisotropic surface kernels (Yu & Turk 2013): per-particle covariance shapes
    // each kernel into an ellipsoid so the surface flattens along the fluid and
    // splash particles stay round. Opt-in; default off keeps the isotropic field
    // byte-identical for every existing caller.
    bool anisotropic = false;
    float aniso_lambda = 0.95f;      // position pre-smooth blend (Eq. 6).
    float aniso_kr = 4.0f;           // eigenvalue clamp ratio sigma1/k_r (Eq. 12-14).
    float aniso_kn = 0.5f;           // isolated-particle isotropic scale (Eq. 15).
    float aniso_ks = 1.0f;           // interior anisotropy scale, recalibrated to units.
    uint32_t aniso_n_eps = 25u;      // neighbor count below which a particle is isolated.

    // Resolved MC voxel edge (the explicit cell_size, else a fraction of h).
    float CellSize() const { return cell_size > 0.0f ? cell_size : 0.5f * h; }
};

// March the fluid isosurface and return it as a render::MeshGeometry (positions,
// indices, gradient-based normals, empty uvs) ready for MeshToBlas. Outward
// winding + outward normals (toward decreasing density). Deterministic (D1-safe):
// identical positions + params always yield a byte-identical mesh. Empty input or
// a fully-sub-iso field yields an empty (valid) mesh.
render::MeshGeometry MarchFluidSurface(const std::vector<math::Vec3>& particle_positions,
                                       const FluidSurfaceParams& p);

}  // namespace nuka::runtime::fluid
