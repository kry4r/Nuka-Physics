"""pytest: a COUPLED world (robot + cloth/fluid particles) through the python binding.

Run (single GPU only):
    export CUDA_VISIBLE_DEVICES=0
    python -m pytest python/tests/test_coupled_world.py -v

Proves the reachability FLOOR for a coupled world through the python surface:
  * nuka.World.create_coupled_from_scene cooks a Go2 + cloth + fluid world on the
    ONE general contact pipeline (the SAME scene->cook path create_from_scene uses,
    plus the engine particle cook);
  * the live particle state is readable zero-copy via
    buffer_view(Field.PARTICLE_POSITION / PARTICLE_VELOCITY) -> torch CUDA tensors,
    shaped (env_count * particle_count, 3);
  * stepping advances the world and the foot two-way couples the media: the at-feet
    fluid is held above its floor more than an identical control whose media are sunk
    far out of reach (the body<->particle reaction acts through the python step path).

This is NOT a full RL stack on coupled worlds (obs/action/reward over the particle
state) -- that is a separate RL-track follow-on. This test only proves create + step
+ read reachability.
"""

from __future__ import annotations

import os
import numpy as np

import pytest
import torch

import nuka
from nuka.author.materials import Soft

SCENE = "/root/Nuka-Physics/examples/scenes/go2_stand.usda"

# The Go2 (fixed base z=0.445) settles with its four foot spheres centred at z~0.2545
# (front pair x~0.062, rear pair x~-0.325) -- the media sit just under the feet.
FOOT_Z = 0.2545
N_CLOTH = 13 * 13  # the soft slice [0, N_CLOTH); the fluid is the upper slice.


def _media_kwargs(at_feet: bool) -> dict:
    # The control sinks the SAME geometry below the feet (a small offset so the
    # CookFluidBox floor(L/s) lattice count is invariant -- a large offset would
    # cancel float bits in the box span and recount the lattice). At dz=-0.30 the
    # fluid surface sits at ~ -0.045 m, far below the foot spheres at z~0.237.
    dz = 0.0 if at_feet else -0.30
    return dict(
        cloth_nx=13, cloth_ny=13, cloth_spacing=0.018,
        cloth_origin_x=0.062, cloth_origin_y=0.0,
        cloth_origin_z=(FOOT_Z - 0.012) + dz,
        cloth_particle_mass=0.01, cloth_friction=0.6, cloth_bend_alpha=1.0e-4,
        cloth_iters=24,
        fluid_min_x=-0.39, fluid_min_y=-0.06, fluid_min_z=(FOOT_Z - 0.10) + dz,
        fluid_max_x=-0.26, fluid_max_y=0.09, fluid_max_z=(FOOT_Z + 0.01) + dz,
        fluid_spacing=0.022, fluid_rest_density=1000.0,
        fluid_floor_z=(FOOT_Z - 0.105) + dz, fluid_friction=0.0, fluid_iters=4,
        contact_radius=0.012,
    )


@pytest.fixture(scope="module")
def device():
    if not os.path.exists(SCENE):
        pytest.skip("go2_stand.usda not present")
    dev = nuka.Device.create(0)
    yield dev
    dev.close()


def test_create_step_read_couples_through_python(device):
    """A coupled world is creatable + steppable + its particle state readable, and
    the foot two-way couples the media through the python step path."""

    def run(at_feet: bool):
        with nuka.World.create_coupled_from_scene(
            device, SCENE, env_count=1, **_media_kwargs(at_feet)
        ) as w:
            # The particle fields RESOLVE zero-copy as CUDA float tensors.
            pos = torch.from_dlpack(w.buffer_view(nuka.Field.PARTICLE_POSITION))
            vel = torch.from_dlpack(w.buffer_view(nuka.Field.PARTICLE_VELOCITY))
            assert w.particle_count > 0
            assert pos.is_cuda and vel.is_cuda
            assert pos.numel() == w.particle_count * 3
            assert pos.shape[-1] == 3
            assert w.download_field(nuka.PARTICLE_PLASTIC_DEFORMATION_GRADIENT).size == 0
            for _ in range(200):
                w.step()
            # Copy out the final particle positions (env 0; single-env world).
            return pos.reshape(-1, 3).detach().clone().cpu()

    at_feet = run(at_feet=True)
    control = run(at_feet=False)
    assert at_feet.shape == control.shape, "the two worlds cooked different counts"
    assert torch.isfinite(at_feet).all(), "coupled-world particles went non-finite"
    assert torch.isfinite(control).all(), "control-world particles went non-finite"

    # Floor-relative fluid-slice comparison cancels the control's -0.30 m offset:
    # WITHOUT coupling the two would settle identically (same box, same physics); the
    # residual is the foot displacing the at-feet pool (the two-way reaction through
    # the python step path). Each world's fluid floor: FOOT_Z-0.105 (+dz for control).
    at_floor = FOOT_Z - 0.105
    ctl_floor = at_floor - 0.30
    fluid_at = at_feet[N_CLOTH:, 2] - at_floor
    fluid_ctl = control[N_CLOTH:, 2] - ctl_floor
    l1 = (fluid_at - fluid_ctl).abs().sum().item()
    surf_at = fluid_at.max().item()
    surf_ctl = fluid_ctl.max().item()
    print(
        f"[py-coupled] particles={at_feet.shape[0]} fluid={fluid_at.numel()} "
        f"at_feet_surf(rel)={surf_at:.4f} control_surf(rel)={surf_ctl:.4f} "
        f"fluid_floor_rel_L1={l1:.4f}"
    )
    assert l1 > 0.02, (
        "the at-feet fluid settled identically to the undisturbed control -> no "
        "body<->particle coupling acted through the python step path"
    )


def test_coupled_world_requires_a_medium(device):
    """A coupled-create with NEITHER medium present is rejected (use
    create_from_scene for a particle-free world)."""
    with pytest.raises(RuntimeError):
        nuka.World.create_coupled_from_scene(device, SCENE, env_count=1)


def test_plastic_material_cook_graph_reset_and_readout(device, tmp_path):
    material = Soft.ElastoPlastic(youngs=30000.0, poisson=0.0, yield_stress=200.0,
                                 hardening_modulus=4000.0, dx=0.025, substeps=10)
    fields = (nuka.PARTICLE_DEFORMATION_GRADIENT,
              nuka.PARTICLE_PLASTIC_DEFORMATION_GRADIENT,
              nuka.PARTICLE_EQUIVALENT_PLASTIC_STRAIN)
    saved = str(tmp_path / "plastic.nks")
    states = []
    with nuka.SceneBuilder.create() as builder:
        builder.add_media(kind=nuka.MEDIA_SOFT_TET, method=nuka.MEDIA_METHOD_MLSMPM,
                          tet_center=[0.0, 0.0, 0.15], tet_radius=0.08,
                          tet_cells=12, tet_cell_len=0.016,
                          **material.media_material_kwargs())
        builder.save(saved)
        for source in (builder, nuka.SceneBuilder.create(saved)):
            try:
                with source.build(device, env_count=2, dt=1.0 / 240.0) as world:
                    world.set_execution_mode("graph")
                    initial = [world.download_field(f).copy().reshape(2, -1) for f in fields]
                    for _ in range(120):
                        world.step()
                    assert not np.any(world.download_field(nuka.ENV_STATUS))
                    final = [world.download_field(f).copy().reshape(2, -1) for f in fields]
                    assert final[2].max() > 0.001
                    np.testing.assert_allclose(np.linalg.det(final[1].reshape(-1, 3, 3)),
                                               1.0, rtol=0.0, atol=2.0e-4)
                    for value in final:
                        assert np.isfinite(value).all()
                        np.testing.assert_array_equal(value[0], value[1])
                    states.append(final)
                    world.reset_envs([0])
                    for f, seed, held in zip(fields, initial, final):
                        actual = world.download_field(f).reshape(2, -1)
                        np.testing.assert_array_equal(actual[0], seed[0])
                        np.testing.assert_array_equal(actual[1], held[1])
                    world.step()
                    world.reset()
                    for f, seed in zip(fields, initial):
                        np.testing.assert_array_equal(world.download_field(f).reshape(2, -1), seed)
                    invalid = initial[1].copy()
                    invalid[0, 0] = 0.0
                    world.upload_field(fields[1], invalid.ravel())
                    world.step()
                    status = world.download_field(nuka.ENV_STATUS)
                    assert status[0] & nuka.ENV_STATUS_CONSTITUTIVE_FAILURE
                    assert status[1] == 0
                    np.testing.assert_array_equal(world.download_field(fields[1])[:9], invalid[0, :9])
                    assert world.download_field(fields[2])[0] == 0.0
                    world.reset()
                    assert not np.any(world.download_field(nuka.ENV_STATUS))
            finally:
                if source is not builder:
                    source.destroy()
    for original, reloaded in zip(*states):
        np.testing.assert_array_equal(original, reloaded)
    with pytest.raises(ValueError):
        Soft.ElastoPlastic(yield_stress=200.0, poisson=0.5)
    with nuka.SceneBuilder.create() as builder:
        with pytest.raises(RuntimeError):
            builder.add_media(kind=nuka.MEDIA_SOFT_TET, method=nuka.MEDIA_METHOD_MLSMPM,
                              mpm_model_kind=5.0, mpm_youngs=30000.0, mpm_density=1000.0)
