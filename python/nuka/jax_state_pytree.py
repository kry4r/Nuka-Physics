"""nuka.jax_state_pytree -- a JAX pytree node for the simulator state arrays.

This is the v0.5 JAX-frontend deliverable 1 (spec Task 5.4.4). It is PURE
PYTHON over the proven nanobind binding -- no C++/CUDA, no rebuild.

``NukaWorldState`` is a small, registered ``@jax.tree_util.register_pytree_node_class``
container holding the three observation-bearing state arrays of a single-env
contact-free rollout:

  * ``joint_pos``  -- actuated JOINT_POSITION ``(action_dim,)``
  * ``joint_vel``  -- actuated JOINT_VELOCITY ``(action_dim,)``
  * ``root_vel``   -- root-link spatial velocity ``(6,)`` (omega-first
    ``[wx,wy,wz,vx,vy,vz]``)

Registering it as a pytree node means JAX transformations -- ``jax.jit``,
``jax.vmap``, ``jax.grad`` -- traverse it automatically: the three arrays are the
*leaves* (the differentiable / batchable children) and the structure itself is
static metadata (here: empty). This lets you ``jit``/``vmap``/``grad`` a PURE-JAX
function of a ``NukaWorldState`` (e.g. a cost over the state arrays) WITHOUT
touching the (side-effecting, foreign) engine step. The engine adjoint lives in
:mod:`nuka.jax_frontend`'s ``custom_vjp``; this pytree is the complementary
pure-array surface.

jax is imported at this submodule's top level, so ``import nuka`` must NOT import
this eagerly -- it is resolved lazily via ``nuka/__init__.py``'s PEP 562
``__getattr__`` (mirroring ``nuka.autograd``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class NukaWorldState:
    """A JAX pytree of the single-env simulator observation state.

    Leaves (in flatten order): ``(joint_pos, joint_vel, root_vel)``. There is no
    auxiliary/static structure -- the container is a pure 3-leaf node, so
    ``jax.jit`` / ``jax.vmap`` / ``jax.grad`` map straight through onto the three
    arrays.

    The arrays are stored as-is (``jnp.asarray`` is applied so plain numpy /
    python lists are accepted at construction); under ``vmap`` the leaves carry a
    leading batch axis and the same flatten/unflatten round-trips correctly.
    """

    __slots__ = ("joint_pos", "joint_vel", "root_vel")

    def __init__(self, joint_pos, joint_vel, root_vel):
        # ``asarray`` keeps tracers as tracers (vmap/jit) and promotes concrete
        # numpy/lists to jax arrays. It must NOT force-copy a tracer, and asarray
        # does not -- it is a no-op on an existing jax array/tracer.
        self.joint_pos = jnp.asarray(joint_pos)
        self.joint_vel = jnp.asarray(joint_vel)
        self.root_vel = jnp.asarray(root_vel)

    # -- pytree protocol -----------------------------------------------------
    def tree_flatten(self):
        """Return ``(children, aux_data)``: the three arrays are children
        (leaves), and there is no static aux metadata."""
        children = (self.joint_pos, self.joint_vel, self.root_vel)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):  # noqa: ARG003 (aux unused)
        """Rebuild from flattened children. ``__new__`` + direct slot assignment
        bypasses ``__init__``'s ``asarray`` so JAX may place *tracers* (during
        jit/vmap/grad tracing) back into the structure unmodified."""
        obj = object.__new__(cls)
        obj.joint_pos, obj.joint_vel, obj.root_vel = children
        return obj

    def __repr__(self):  # pragma: no cover - convenience only
        return (
            f"NukaWorldState(joint_pos={self.joint_pos!r}, "
            f"joint_vel={self.joint_vel!r}, root_vel={self.root_vel!r})"
        )


__all__ = ["NukaWorldState"]
