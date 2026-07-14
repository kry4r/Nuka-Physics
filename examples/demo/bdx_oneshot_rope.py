#!/usr/bin/env python3
"""Author the hanging stone-slab assembly as a .usda fragment: a revolute-Y
capsule rope chain, a rigid V-yoke (ONE link, TWO angled capsule geoms forking to
the slab's top corners), and a stone slab fixed under the yoke. Composed into a
scene through the general importer + SceneBuilder.compose.

The assembly is a fixed-base tree: a kinematic anchor (mass 0 => Fixed root),
`n_links` capsule links joined by coaxial Revolute-Y joints (a planar pendulum
swinging in X-Z), the yoke joined revolute-Y to the chain tail, and the slab
fixed to the yoke. Every xformOp:translate is the PARENT-RELATIVE offset down
the chain, so the assembly hangs straight below the anchor and swings freely
under gravity. Visually the yoke's two legs read as a two-rope suspension while
the topology stays a tree (no closed loop).
"""

from __future__ import annotations

import math


def rope_usda(n_links: int = 7, seg: float = 0.043, radius: float = 0.008,
              link_mass: float = 0.02,
              yoke=(0.11, 0.046, 0.03),
              slab=(0.025, 0.11, 0.065, 0.6)) -> str:
    """Return a .usda string: anchor -> n_links capsules -> V-yoke -> slab.
    ``yoke`` = (leg_dy, leg_dz, mass): each leg runs from the yoke apex to a slab
    top corner at (0, +-leg_dy, -leg_dz). ``slab`` = (hx, hy, hz, mass)."""
    cyl_h = max(seg - 2.0 * radius, 0.008)
    P = ['#usda 1.0\n(\n    defaultPrim = "World"\n    metersPerUnit = 1\n'
         '    upAxis = "Z"\n)\n', 'def Xform "World"\n{']
    P.append('''    def Xform "anchor" ( prepend apiSchemas = ["PhysicsRigidBodyAPI","PhysicsMassAPI"] )
    {
        bool physics:rigidBodyEnabled = true
        bool physics:kinematicEnabled = true
        float physics:mass = 0
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }''')
    for i in range(n_links):
        P.append(f'''    def Xform "link_{i}" ( prepend apiSchemas = ["PhysicsRigidBodyAPI","PhysicsMassAPI"] )
    {{
        bool physics:rigidBodyEnabled = true
        float physics:mass = {link_mass}
        float3 physics:diagonalInertia = (8e-6, 8e-6, 1e-6)
        double3 xformOp:translate = (0, 0, {-seg:.5f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Capsule "geom" ( prepend apiSchemas = ["PhysicsCollisionAPI"] )
        {{
            double radius = {radius}
            double height = {cyl_h:.5f}
            uniform token axis = "Z"
        }}
    }}''')

    # The V-yoke: apex at the link frame, two capsule legs forking to the slab's
    # top corners at (0, +-leg_dy, -leg_dz) -- multi-geom on ONE rigid link.
    leg_dy, leg_dz, yoke_mass = yoke
    leg_len = math.hypot(leg_dy, leg_dz)
    leg_cyl = max(leg_len - 2.0 * radius, 0.01)
    # Rotate the capsule's +Z axis onto the leg direction (0, +-leg_dy, -leg_dz):
    # a pure X rotation by -+(90 + atan(leg_dz/leg_dy)) degrees.
    ang = 90.0 + math.degrees(math.atan2(leg_dz, leg_dy))
    legs = []
    for sy, rx in ((1.0, -ang), (-1.0, ang)):
        legs.append(f'''        def Capsule "leg_{'p' if sy > 0 else 'n'}" ( prepend apiSchemas = ["PhysicsCollisionAPI"] )
        {{
            double radius = {radius}
            double height = {leg_cyl:.5f}
            uniform token axis = "Z"
            double3 xformOp:translate = ({0.0}, {sy * 0.5 * leg_dy:.5f}, {-0.5 * leg_dz:.5f})
            float3 xformOp:rotateXYZ = ({rx:.3f}, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
        }}''')
    P.append(f'''    def Xform "yoke" ( prepend apiSchemas = ["PhysicsRigidBodyAPI","PhysicsMassAPI"] )
    {{
        bool physics:rigidBodyEnabled = true
        float physics:mass = {yoke_mass}
        float3 physics:diagonalInertia = (5e-5, 3e-5, 5e-5)
        double3 xformOp:translate = (0, 0, {-seg:.5f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
{chr(10).join(legs)}
    }}''')

    hx, hy, hz, sm = slab
    P.append(f'''    def Xform "slab" ( prepend apiSchemas = ["PhysicsRigidBodyAPI","PhysicsMassAPI"] )
    {{
        bool physics:rigidBodyEnabled = true
        float physics:mass = {sm}
        float3 physics:diagonalInertia = ({sm / 12.0 * (4 * hy * hy + 4 * hz * hz):.6f}, {sm / 12.0 * (4 * hx * hx + 4 * hz * hz):.6f}, {sm / 12.0 * (4 * hx * hx + 4 * hy * hy):.6f})
        double3 xformOp:translate = (0, 0, {-(leg_dz + hz):.5f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "geom" ( prepend apiSchemas = ["PhysicsCollisionAPI"] )
        {{
            double size = 2.0
            float3 xformOp:scale = ({hx}, {hy}, {hz})
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }}
    }}''')

    def rev(name, b0, b1):
        return f'''    def PhysicsRevoluteJoint "{name}" {{
        rel physics:body0 = </World/{b0}>
        rel physics:body1 = </World/{b1}>
        token physics:axis = "Y"
        float physics:lowerLimit = -175
        float physics:upperLimit = 175
    }}'''

    P.append(rev("j_anchor", "anchor", "link_0"))
    for i in range(1, n_links):
        P.append(rev(f"j_{i}", f"link_{i - 1}", f"link_{i}"))
    P.append(rev("j_yoke", f"link_{n_links - 1}", "yoke"))
    P.append(f'''    def PhysicsFixedJoint "j_slab" {{
        rel physics:body0 = </World/yoke>
        rel physics:body1 = </World/slab>
    }}''')
    P.append('}\n')
    return "\n".join(P)


def write_rope_usda(path: str, **kw) -> str:
    with open(path, "w") as f:
        f.write(rope_usda(**kw))
    return path


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "examples/scenes/bdx_oneshot_rope.usda"
    write_rope_usda(out)
    print(f"[rope] wrote {out}")
