# Coupled physics

Rigid bodies, articulations, MLS-MPM and XPBD exchange contact impulses through the same constraint rows. Dynamic endpoints contribute their finite mass and receive the opposite reaction; pinned vertices and static colliders act as supports.

| Pair | Contact geometry and reaction |
| --- | --- |
| Rigid ↔ articulation | Collider manifolds; rigid inertia and articulation Jacobians |
| Rigid ↔ MLS-MPM | Active grid nodes against colliders; grid mass and rigid inertia |
| Articulation ↔ MLS-MPM | Active grid nodes against colliders; grid mass and articulation Jacobians |
| Rigid ↔ XPBD | Particle spheres against colliders; particle mass and rigid inertia |
| Articulation ↔ XPBD | Particle spheres against colliders; particle mass and articulation Jacobians |
| MLS-MPM ↔ XPBD | Active grid nodes against the deformed triangle surface; grid mass and interpolated vertex reactions |

Cloth, tetrahedral soft bodies and cable slabs retain their authored surface topology when cooked. A point cloud or cable centreline alone does not define a collision surface. Multiple XPBD media can share a world with one MLS-MPM medium. An MLS-MPM/PBF mixture is currently rejected.

## Authoring

Add the media to the same `SceneBuilder` or `nuka.author.Scene`. The shared world provides stepping, control, graph execution and selective environment reset. For example:

```python
with nuka.SceneBuilder.create("robot.nks") as scene:
    scene.add_media(
        kind=nuka.MEDIA_CLOTH, method=nuka.MEDIA_METHOD_XPBD,
        cloth_nx=13, cloth_ny=13, cloth_spacing=0.02,
        cloth_origin=[0.0, 0.0, 0.1], xpbd_particle_mass=0.01,
    )
    scene.add_media(
        kind=nuka.MEDIA_SOFT_TET, method=nuka.MEDIA_METHOD_MLSMPM,
        tet_center=[0.0, 0.0, 0.2], tet_radius=0.05,
        tet_cells=10, tet_cell_len=0.012,
        mpm_youngs=10000.0, mpm_poisson=0.3, mpm_density=1000.0,
        mpm_dx=0.02, mpm_substeps=8, mpm_contact_capacity=4096,
    )
    world = scene.build(device, env_count=4, dt=1.0 / 240.0)
```

The contact capacity is an explicit storage budget. Check `ENV_STATUS` and `GRID_CONTACT_OVERFLOW`: a nonzero overflow means some candidate contacts were not retained. `GRID_CONTACT_ATTEMPTED`, `GRID_CONTACT_RETAINED` and `GRID_CONTACT_PEAK` expose the corresponding counts.

## Contact readout

`CONTACT_SIDE_A_KIND` and `CONTACT_SIDE_B_KIND` identify each reaction owner. Their companion `*_INDEX` fields address a world body, link, particle, grid node or interpolated point endpoint, according to the kind. Indices include the environment offset.

`ContactSideKind.POINT_ENDPOINT` identifies an interpolated XPBD surface endpoint. Its index selects `POINT_ENDPOINT_RANGES`, a `uint32` array of `[first, count]` pairs. `first` addresses `POINT_ENDPOINT_TERMS` globally. A zero count denotes an unused range.

`POINT_ENDPOINT_TERMS` exposes the bytes of `nuka_point_endpoint_term_t` from `nuka.h`. Each 44-byte record contains a `uint32 kind`, a `uint32 index` and a column-major 3×3 float velocity map. The map interpolates endpoint velocity; its transpose distributes contact impulse to that term's point mass. Only records referenced by a nonempty range are valid.

Python DLPack views have shapes `[env, endpoint_capacity, 2]` for ranges and `[env, term_capacity, 44]` for bytes. A host download can be decoded with:

```python
term_dtype = np.dtype([
    ("kind", "<u4"), ("index", "<u4"), ("columns", "<f4", (3, 3)),
])
terms = world.download_field(nuka.POINT_ENDPOINT_TERMS).view(term_dtype)
```

Reset clears the selected environments' ranges, terms and contact outputs while preserving the addresses of existing views.

## Resolution limits

The surface and its velocity maps are rebuilt each physical substep and held fixed during the contact solve. Thin layers sharing an MLS-MPM grid cell still share one grid velocity. Surface CCD and multilayer self-contact are not provided by this coupling. Rigid/XPBD collision currently samples particle spheres rather than the full cloth surface.

Large mass ratios and nearly redundant friction contacts can require substantially more solver iterations. Assess contact residuals and timestep convergence for the intended load and geometry. Offline `render_beauty` displays the deforming media; camera sensor rendering does not yet include those deforming surfaces.

APIC transfer dissipates velocity modes that the grid cannot represent. A smaller timestep repeats that transfer more often, so elastic vibration at a fixed grid resolution can change even after the contact solve converges. Elastic surface-loading timestep convergence remains unresolved; this coupling should not yet serve as a converged elastic performance benchmark. Assess grid resolution, particle sampling and timestep together.
