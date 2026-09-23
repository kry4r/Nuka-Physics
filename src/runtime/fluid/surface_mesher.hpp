#pragma once
// Shared host reconstruction of particle density isosurfaces for rendering and sensors.

#include "math/vec3.hpp"
#include "render/mesh_geometry.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime::fluid {

// Marching cubes over sum(m * Poly6); the isovalue is iso_fraction * rest_density_rho0.
struct FluidSurfaceParams {
    float h = 0.1f;                  // SPH support radius (PBF kernel support).
    float rest_density_rho0 = 1.0f;  // rho0; iso = iso_fraction * rho0.
    float iso_fraction = 0.5f;       // surface at this fraction of rest density.
    float particle_mass = 1.0f;      // m (uniform across the fluid).
    float cell_size = 0.0f;          // MC voxel edge; <= 0 => 0.5 * h.

    // Yu & Turk anisotropic kernels use neighbor covariance to shape each ellipsoid.
    // Disabling anisotropy retains isotropic Poly6 kernels.
    bool anisotropic = false;
    float aniso_lambda = 0.95f;      // position pre-smooth blend (Eq. 6).
    float aniso_kr = 4.0f;           // eigenvalue clamp ratio sigma1/k_r (Eq. 12-14).
    float aniso_kn = 0.5f;           // isolated-particle isotropic scale (Eq. 15).
    float aniso_ks = 1.0f;           // interior anisotropy scale, recalibrated to units.
    uint32_t aniso_n_eps = 25u;      // neighbor count below which a particle is isolated.

    // Resolved MC voxel edge (the explicit cell_size, else a fraction of h).
    float CellSize() const { return cell_size > 0.0f ? cell_size : 0.5f * h; }
};

// A uniform sampling lattice defines particle volume, kernel support and reconstruction resolution.
FluidSurfaceParams DensitySurfaceParams(float spacing);

// Deterministic outward triangles and gradient normals; a sub-isovalue field yields an empty mesh.
render::MeshGeometry MarchFluidSurface(const std::vector<math::Vec3>& particle_positions,
                                       const FluidSurfaceParams& p);

}  // namespace nuka::runtime::fluid
