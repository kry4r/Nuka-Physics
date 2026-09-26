# Coupled physics

Rigid bodies, articulations, MLS-MPM and XPBD exchange contact impulses through the same constraint rows. Dynamic endpoints contribute their finite mass and receive the opposite reaction; pinned vertices and static colliders act as supports.

| Pair | Contact geometry and reaction |
| --- | --- |
| Rigid ↔ articulation | Collider manifolds; rigid inertia and articulation Jacobians |
| Rigid ↔ MLS-MPM | Material points against colliders; interpolated grid response and rigid inertia |
| Articulation ↔ MLS-MPM | Material points against colliders; interpolated grid response and articulation Jacobians |
| Rigid ↔ XPBD | Particle spheres against colliders; particle mass and rigid inertia |
| Articulation ↔ XPBD | Particle spheres against colliders; particle mass and articulation Jacobians |
| MLS-MPM ↔ XPBD | Material points against the deformed triangle surface; grid response and interpolated vertex reactions |

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

## Contact exchange frequency

`world.set_coupling_passes(count)` sets the number of contact exchanges in each physical substep. The default, `0`, follows the largest material projection budget. Positive values distribute the same XPBD, PBF, particle-contact and contact-solver iteration budgets across that many exchanges. Material projection order, timestep and substep count remain unchanged. The C API provides `nuka_world_set_coupling_passes`; C++ uses `Pipeline::SolverConfig::coupling_passes` or `World::SetCouplingPasses`.

Changing this setting preserves world state and buffer addresses and invalidates the captured execution graph. Fewer exchanges change the finite-iteration coupling response: compare contact residuals, deformation and timestep convergence before using a lower count. The G1 demo exposes the setting as `--coupling-passes` and records it in `metrics.json`.

MLS-MPM evaluates stress from the accepted particle history during particle-to-grid transfer. Contacts modify the grid velocity; the final grid velocity drives particle transfer and the material history update. Constitutive evaluation reads the accepted history into a separate trial, and a successful trial commits once per physical interval. Repeated trial evaluation leaves the accepted history unchanged.

The material integration remains explicit. Contact exchanges do not iterate the material stress to convergence, so their count does not remove the material stiffness restriction on the timestep.

## Contact readout

`CONTACT_SIDE_A_KIND` and `CONTACT_SIDE_B_KIND` identify each reaction owner. Their companion `*_INDEX` fields address a world body, link, particle, grid node or interpolated point endpoint, according to the kind. Indices include the environment offset.

`ContactSideKind.POINT_ENDPOINT` identifies an interpolated MLS-MPM grid or XPBD surface endpoint. Its index selects `POINT_ENDPOINT_RANGES`, a `uint32` array of `[first, count]` pairs. `first` addresses `POINT_ENDPOINT_TERMS` globally. A zero count denotes an unused range.

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

Box fills use cell-centered particles with mass `density * spacing³`. Counts near an integer account for the floating-point precision of both endpoints and spacing, so adjacent lattice-aligned fills do not lose a complete particle layer. Genuine fractional cells remain unfilled. Adjacent fills with the same spacing and render material share a reconstructed surface.

MLS-MPM grid bounds describe allocated storage, not container walls. Containers need finite collision geometry; the authored floor remains a physical plane. Water can flow over a wall's top. Allocate enough grid headroom for that motion: a transfer stencil leaving any grid face sets `ENV_STATUS` rather than supplying an invisible collision wall.

Liquid render surfaces are reconstructed from current particle positions. The shared mesher reflects the density field at rigid solid boundaries and closes the surface just inside the solid, preventing missing kernel support from rounding the water away from pool walls. This reconstruction changes neither particle positions nor material history. Display rendering and camera sensors use the same reconstruction and live boundary poses.

The surface and its velocity maps are rebuilt each physical substep and held fixed during the contact solve. Thin layers sharing an MLS-MPM grid cell still share one grid velocity. Surface CCD and multilayer self-contact are not provided by this coupling. Rigid/XPBD collision currently samples particle spheres rather than the full cloth surface.

Large mass ratios and nearly redundant friction contacts can require substantially more solver iterations. Assess contact residuals and timestep convergence for the intended load and geometry. Offline `render_beauty` and camera sensors include the registered deforming surfaces.

APIC transfer dissipates velocity modes that the grid cannot represent. A smaller timestep repeats that transfer more often, so elastic vibration at a fixed grid resolution can change even after the contact solve converges. Elastic surface-loading timestep convergence remains unresolved; this coupling should not yet serve as a converged elastic performance benchmark. Assess grid resolution, particle sampling and timestep together.
