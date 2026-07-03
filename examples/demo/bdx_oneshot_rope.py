#!/usr/bin/env python3
"""Author a hanging-rope articulation as a .usda fragment (a serial revolute
capsule chain + a box payload), for composing into a scene through the general
importer + SceneBuilder.compose.

The rope is a fixed-base articulation: a kinematic anchor (mass 0 => Fixed root),
`n_links` capsule links each joined to the one above by a Revolute joint about Y
(a planar pendulum swinging in X-Z), and a box block on the bottom link. Each
link's xformOp:translate is the PARENT-RELATIVE offset (0,0,-seg) down the joint
chain (the articulation FK composes translate down the chain), so the chain hangs
straight below the anchor and swings freely under gravity with no actuator.
Impact momentum reaches a struck body through the general contact path.
"""

from __future__ import annotations


def rope_usda(n_links: int = 9, seg: float = 0.05, radius: float = 0.007,
              link_mass: float = 0.02,
              block=(0.06, 0.04, 0.025, 0.30)) -> str:
    """Return a .usda string for a fixed-base revolute capsule chain anchored at
    the local origin, hanging in -Z. ``block`` = (hx,hy,hz,mass) box payload on the
    last link (None to omit). Compose it at the beam's world pose to place it."""
    hh = seg * 0.5
    cyl_h = max(seg - 2.0 * radius, 0.01)
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
        float3 physics:diagonalInertia = (1.5e-5, 1.5e-5, 2e-6)
        double3 xformOp:translate = (0, 0, {-seg:.5f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Capsule "geom" ( prepend apiSchemas = ["PhysicsCollisionAPI"] )
        {{
            double radius = {radius}
            double height = {cyl_h:.5f}
            uniform token axis = "Z"
        }}
    }}''')
    if block is not None:
        hx, hy, hz, bm = block
        P.append(f'''    def Xform "block" ( prepend apiSchemas = ["PhysicsRigidBodyAPI","PhysicsMassAPI"] )
    {{
        bool physics:rigidBodyEnabled = true
        float physics:mass = {bm}
        float3 physics:diagonalInertia = (3e-4, 3e-4, 3e-4)
        double3 xformOp:translate = (0, 0, {-(hh + hz):.5f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "geom" ( prepend apiSchemas = ["PhysicsCollisionAPI"] )
        {{
            double size = 2.0
            float3 xformOp:scale = ({hx}, {hy}, {hz})
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }}
    }}''')

    def joint(name, b0, b1):
        return f'''    def PhysicsRevoluteJoint "{name}" {{
        rel physics:body0 = </World/{b0}>
        rel physics:body1 = </World/{b1}>
        token physics:axis = "Y"
        float physics:lowerLimit = -175
        float physics:upperLimit = 175
    }}'''

    P.append(joint("j_anchor", "anchor", "link_0"))
    for i in range(1, n_links):
        P.append(joint(f"j_{i}", f"link_{i - 1}", f"link_{i}"))
    if block is not None:
        P.append(joint("j_block", f"link_{n_links - 1}", "block"))
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
