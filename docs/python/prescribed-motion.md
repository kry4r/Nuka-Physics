# Prescribed rigid motion

`nuka.motion.JointMotion` computes body poses and world COM linear and angular
velocities from named revolute and prismatic coordinates. Fixed joints are
supported too. Calibration uses metres, radians and quaternions in `wxyz` order:

```python
from nuka.motion import JointMotion, KinematicBoundary

motion = JointMotion(
    [dict(name="hinge", parent=-1, child=0, type="revolute",
          parent_frame=[0, 0, 1, 1, 0, 0, 0],
          child_frame=[0, 0, 0, 1, 0, 0, 0], axis=[0, 0, 1])],
    body_poses=[[0, 0, 1, 1, 0, 0, 0]], body_com=[[0, 0, 0]],
)
boundary = KinematicBoundary(world, motion, body_indices=[0])
boundary.apply({"hinge": 0.2}, {"hinge": 0.5})
world.step()
```

The cooked row must be an independent body with zero inverse mass. Binding
rejects dynamic bodies, articulation rows, compound proxy rows, repeated indices and mismatched COMs.
`world.body_binding(body_index, env_index=0)` exposes the live inverse mass,
articulation link index, compound owner body index (both `UINT32_MAX` for an
independent body), and local rigid-row COM
through the C API as well. Joint frames satisfy
`X_body = X_parent * X_parent_joint * X_motion * inverse(X_child_joint)`.
Coordinate dictionaries must supply every moving joint exactly once. Array
values broadcast over environments; scalar inputs apply the same world motion
to all environments. Independent fixed roots retain their calibrated world poses.

This boundary describes one-way coupling to an external motion driver. Contacts
use the same solver and geometry as other rigid contacts; their reactions do not
accelerate the driver. Keep ordinary dynamic robots articulated and use their
actuator inputs when reaction dynamics are required. Partial prescribed coordinates
inside a dynamic articulation are not represented by this interface.
Applying motion also refreshes compound collider rows and live rendering poses.

Supply the geometry pose and twist at the time used by the external coupling
scheme before every physics substep. Poses remain fixed during that substep;
the twist supplies the moving surface velocity. Publishing an end pose for state
readout is a separate explicit call. Reset restores the cooked boundary pose;
reapply the requested joint coordinates before stepping again. Uploading a pose
does not add a continuous collision sweep for its external motion. Motion must
be resolved by the chosen substeps; independent CCD acceptance remains necessary.

`nuka.geometry.cone_mesh(radius, half_height, max_error=...)` supplies a closed,
inscribed triangle mesh when an interchange format cannot express an analytic
cone. The cone points along +Z, with base at `-half_height` and tip at
`+half_height`. The result exposes vertices, outward oriented triangles and a
Hausdorff/support error bound `radius * (1 - cos(pi / sides))`. Preserve the
original solid's mass, COM and inertia when exporting a dynamic cone. This
geometric bound does not establish equal contact forces between different laws.
