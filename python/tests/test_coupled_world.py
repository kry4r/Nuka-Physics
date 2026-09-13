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
from dataclasses import replace
from pathlib import Path
import numpy as np

import pytest
import torch

import nuka
from nuka.author.materials import Soft

SCENE = str(Path(__file__).resolve().parents[2] / "examples/scenes/go2_stand.usda")

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


def test_particle_surfaces_camera_lidar_graph_and_reset(device, tmp_path):
    saved = str(tmp_path / "observed_surfaces.nks")
    recorded = []
    with nuka.SceneBuilder.create() as builder:
        builder.add_rigid_primitive(nuka.PRIMITIVE_PLANE)
        cloth_material = builder.add_material("cloth", base_color=[0.1, 0.7, 0.2])
        tet_material = builder.add_material("soft", base_color=[0.8, 0.1, 0.1])
        builder.add_media(kind=nuka.MEDIA_CLOTH, method=nuka.MEDIA_METHOD_XPBD,
                          cloth_nx=7, cloth_ny=7, cloth_spacing=0.08,
                          cloth_origin=[0.0, 0.0, 0.5], cloth_free=True,
                          xpbd_particle_mass=0.02, xpbd_iters=8,
                          skin_normal_offset=0.007, skin_smooth_iters=2,
                          skin_smooth_lambda=0.2, render_material_id=cloth_material)
        builder.add_media(kind=nuka.MEDIA_SOFT_TET, method=nuka.MEDIA_METHOD_XPBD,
                          tet_center=[0.8, 0.0, 0.5], tet_radius=0.12,
                          tet_cells=8, tet_cell_len=0.04,
                          xpbd_particle_mass=0.02, xpbd_iters=8,
                          render_material_id=tet_material)
        builder.save(saved)
        for source in (builder, nuka.SceneBuilder.create(saved)):
            try:
                with source.build(device, env_count=3, dt=0.001, gravity_z=0.0) as world:
                    world.set_gravity_z(0.0)
                    for x in (0.0, 0.8):
                        world.attach_camera_sensor(nuka.SensorMount.WORLD.value, 0,
                                                   (x, 0, 2, 1, 0, 0, 0), 30, 65, 65)
                    q = np.sqrt(0.5)
                    world.attach_lidar_sensor(nuka.SensorMount.WORLD.value, 0, (0, 0, 2, q, 0, q, 0),
                                              1, 1, 0, 0, 0, 0, max_range=10.0)

                    def image(channel):
                        return torch.from_dlpack(world.get_sensor_view(channel)).cpu().numpy().copy()

                    world.render_sensors()
                    depth = image(nuka.SensorChannel.DEPTH).reshape(3, 2, 65, 65)
                    np.testing.assert_allclose(depth[:, 0, 32, 32], 1.493, atol=2.0e-6)
                    assert np.all(depth[:, 1, 32, 32] < 1.45)
                    np.testing.assert_allclose(image(nuka.SensorChannel.RANGE).ravel(),
                                               depth[:, 0, 32, 32], atol=2.0e-6)
                    albedo = image(nuka.SensorChannel.ALBEDO).reshape(3, 2, 65, 65, 3)
                    for env in range(3):
                        np.testing.assert_allclose(albedo[env, :, 32, 32],
                                                   [[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]], atol=2.0e-6)
                    recorded.append(depth.copy())
                    initial = world.download_field(nuka.Field.PARTICLE_POSITION).reshape(3, -1, 3).copy()
                    velocity = np.zeros_like(initial)
                    velocity[:, :, 2] = np.array([0.1, 0.3, -0.1], dtype=np.float32)[:, None]
                    world.upload_field(nuka.Field.PARTICLE_VELOCITY, velocity)
                    world.set_execution_mode("graph")
                    world.step_n(8)
                    before_render = world.download_field(nuka.Field.PARTICLE_POSITION).copy()
                    world.render_sensors()
                    moved = image(nuka.SensorChannel.DEPTH).reshape(3, 2, 65, 65)
                    np.testing.assert_allclose(moved[:, 0, 32, 32],
                        depth[:, 0, 32, 32] - np.array([0.1, 0.3, -0.1]) * 0.008, atol=1.0e-5)
                    np.testing.assert_array_equal(world.download_field(nuka.Field.PARTICLE_POSITION), before_render)
                    tilted = initial.copy()
                    tilted[:, :49, 2] += 0.1 * tilted[:, :49, 0] + 0.2 * tilted[:, :49, 1]
                    world.upload_field(nuka.Field.PARTICLE_POSITION, tilted)
                    world.render_sensors()
                    tilted_depth = image(nuka.SensorChannel.DEPTH).reshape(3, 2, 65, 65)
                    np.testing.assert_allclose(tilted_depth[:, 0, 32, 32],
                                               1.5 - 0.007 * np.sqrt(1.05), atol=3.0e-6)
                    normal = image(nuka.SensorChannel.NORMAL).reshape(3, 2, 65, 65, 3)
                    for env in range(3):
                        np.testing.assert_allclose(normal[env, 0, 32, 32],
                                                   np.array([-0.1, -0.2, 1.0]) / np.sqrt(1.05), atol=3.0e-6)
                    world.render_sensors()
                    np.testing.assert_array_equal(image(nuka.SensorChannel.DEPTH).reshape(3, 2, 65, 65), tilted_depth)
                    world.reset_envs([1])
                    world.render_sensors()
                    reset = image(nuka.SensorChannel.DEPTH).reshape(3, 2, 65, 65)
                    np.testing.assert_array_equal(reset[1], depth[1])
                    np.testing.assert_array_equal(reset[[0, 2]], tilted_depth[[0, 2]])
                    assert not np.any(world.download_field(nuka.ENV_STATUS))
            finally:
                if source is not builder:
                    source.destroy()
    np.testing.assert_array_equal(*recorded)

    with nuka.SceneBuilder.create() as builder:
        builder.add_media(kind=nuka.MEDIA_CLOTH, method=nuka.MEDIA_METHOD_XPBD,
                          cloth_nx=3, cloth_ny=3, cloth_spacing=0.1,
                          cloth_origin=[0.0, 0.0, 0.5], cloth_free=True)
        with builder.build(device, env_count=2) as world:
            world.attach_camera_sensor(nuka.SensorMount.WORLD.value, 0,
                                       (0, 0, 2, 1, 0, 0, 0), 30, 33, 33)
            world.render_sensors()
            depth = torch.from_dlpack(world.get_sensor_view(nuka.SensorChannel.DEPTH))
            np.testing.assert_allclose(depth.cpu().numpy().reshape(2, 33, 33)[:, 16, 16],
                                       1.5, atol=2.0e-6)


def test_mpm_surface_endpoints_roundtrip_graph_and_reset(device, tmp_path):
    saved = str(tmp_path / "mpm_cloth.nks")
    states = []
    term_dtype = np.dtype([("kind", "<u4"), ("index", "<u4"), ("columns", "<f4", (3, 3))])
    assert term_dtype.itemsize == 44
    with nuka.SceneBuilder.create() as builder:
        for x in (-0.04, 0.04):
            builder.add_media(kind=nuka.MEDIA_CLOTH, method=nuka.MEDIA_METHOD_XPBD,
                              cloth_nx=6, cloth_ny=6, cloth_spacing=0.016,
                              cloth_origin=[x, 0.0, 0.06], cloth_free=True,
                              xpbd_particle_mass=0.01, xpbd_iters=8)
        builder.add_media(kind=nuka.MEDIA_SOFT_TET, method=nuka.MEDIA_METHOD_MLSMPM,
                          tet_center=[0.0, 0.0, 0.085], tet_radius=0.04,
                          tet_cells=8, tet_cell_len=0.012,
                          mpm_youngs=10000.0, mpm_poisson=0.3, mpm_density=1000.0,
                          mpm_dx=0.02, mpm_substeps=2, mpm_floor_d=-1.0,
                          mpm_contact_capacity=2048)
        builder.save(saved)
        for source in (builder, nuka.SceneBuilder.create(saved)):
            try:
                with source.build(device, env_count=3, dt=0.0005, gravity_z=0.0,
                                  solver_vel_iters=128) as world:
                    nmpm = world.particle_count - 72
                    assert nmpm > 0
                    velocity = np.zeros((3, world.particle_count, 3), dtype=np.float32)
                    velocity[:, :nmpm, 2] = -0.2
                    world.upload_field(nuka.Field.PARTICLE_VELOCITY, velocity)
                    initial = world.download_field(nuka.Field.PARTICLE_POSITION).reshape(3, -1).copy()
                    ranges_view = torch.from_dlpack(world.buffer_view(nuka.POINT_ENDPOINT_RANGES))
                    terms_view = torch.from_dlpack(world.buffer_view(nuka.POINT_ENDPOINT_TERMS))
                    assert ranges_view.dtype == torch.uint32 and ranges_view.shape[-1] == 2
                    assert terms_view.dtype == torch.uint8 and terms_view.shape[-1] == 44
                    assert ranges_view.shape[0] == terms_view.shape[0] == 3
                    address = terms_view.data_ptr()
                    world.set_execution_mode("graph")
                    world.step_n(8)
                    assert not np.any(world.download_field(nuka.ENV_STATUS))
                    ranges = world.download_field(nuka.POINT_ENDPOINT_RANGES).reshape(3, -1, 2)
                    terms = world.download_field(nuka.POINT_ENDPOINT_TERMS).reshape(3, -1, 44)
                    np.testing.assert_array_equal(ranges_view.cpu().numpy(), ranges)
                    np.testing.assert_array_equal(terms_view.cpu().numpy(), terms)
                    records = terms.reshape(-1, 44).view(term_dtype).reshape(-1)
                    kinds = world.download_field(nuka.CONTACT_SIDE_B_KIND).reshape(3, -1)
                    indices = world.download_field(nuka.CONTACT_SIDE_B_INDEX).reshape(3, -1)
                    forces = world.download_field(nuka.CONTACT_FORCE).reshape(3, -1, 3)
                    for env in range(3):
                        active = kinds[env] == nuka.ContactSideKind.POINT_ENDPOINT.value
                        assert active.any() and np.max(np.linalg.norm(forces[env, active], axis=1)) > 1.0e-6
                        participants = set()
                        for index in indices[env, active]:
                            assert env * ranges.shape[1] <= index < (env + 1) * ranges.shape[1]
                            first, count = ranges[env, index - env * ranges.shape[1]]
                            assert count == 3
                            endpoint = records[first:first + count]
                            assert np.all(endpoint["kind"] == nuka.ContactSideKind.PARTICLE.value)
                            local = endpoint["index"].astype(np.int64) - env * world.particle_count
                            assert np.all((local >= nmpm) & (local < world.particle_count))
                            participants.update(((local - nmpm) // 36).tolist())
                            np.testing.assert_allclose(endpoint["columns"].sum(axis=0), np.eye(3), atol=2.0e-5)
                        assert participants == {0, 1}
                    before = world.download_field(nuka.Field.PARTICLE_POSITION).reshape(3, -1).copy()
                    assert np.isfinite(before).all() and not np.array_equal(before, initial)
                    states.append((before, ranges.copy(), terms.copy(), kinds.copy(), indices.copy()))
                    world.reset_envs([1])
                    after = world.download_field(nuka.Field.PARTICLE_POSITION).reshape(3, -1)
                    np.testing.assert_array_equal(after[[0, 2]], before[[0, 2]])
                    np.testing.assert_array_equal(after[1], initial[1])
                    reset_ranges = world.download_field(nuka.POINT_ENDPOINT_RANGES).reshape(ranges.shape)
                    reset_terms = world.download_field(nuka.POINT_ENDPOINT_TERMS).reshape(terms.shape)
                    assert not np.any(reset_ranges[1])
                    cleared = reset_terms[1].view(term_dtype).reshape(-1)
                    assert not np.any(cleared["columns"])
                    assert np.all(cleared["index"] == np.iinfo(np.uint32).max)
                    np.testing.assert_array_equal(reset_ranges[[0, 2]], ranges[[0, 2]])
                    np.testing.assert_array_equal(reset_terms[[0, 2]], terms[[0, 2]])
                    assert terms_view.data_ptr() == address
                    world.step()
                    assert not np.any(world.download_field(nuka.ENV_STATUS))
            finally:
                if source is not builder:
                    source.destroy()
    for original, reloaded in zip(*states):
        np.testing.assert_array_equal(original, reloaded)


@pytest.mark.parametrize("mode", range(6))
@pytest.mark.parametrize("entry", ["coupled", "builder", "author_coupled", "author_built"])
def test_control_creation_entries(device, mode, entry):
    from nuka.author import Scene, SimOptions, materials, morphs

    options = dict(env_count=2, control_mode=mode, osc_task_link=3)
    if entry == "coupled":
        world = nuka.World.create_coupled_from_scene(device, SCENE, **options, **_media_kwargs(True))
    elif entry == "builder":
        with nuka.SceneBuilder.create(SCENE) as builder:
            world = builder.build(device, **options)
    else:
        authored = Scene(SimOptions(**options))
        authored.add_entity(morphs.NKS(SCENE))
        authored.add_entity(morphs.Grid(5, 5, 0.018, origin=(0.062, 0.0, FOOT_Z)),
                            materials.Cloth.XPBD())
        if entry == "author_built":
            authored.add_entity(morphs.Sphere(0.03, pos=(3.0, 0.0, 0.3)), materials.Rigid())
        world = authored.build(device)
    with world:
        initial = world.download_field(nuka.JOINT_POSITION).copy().reshape(2, -1)
        field = [nuka.DRIVE_TARGET, nuka.TORQUE_INPUT, nuka.VELOCITY_TARGET,
                 nuka.ACCELERATION_TARGET, nuka.TASK_TARGET, nuka.TORQUE_INPUT][mode]
        targets = torch.from_dlpack(world.buffer_view(field))
        if mode == nuka.CONTROL_MODE_OSC:
            assert targets.numel() == 6
            poses = torch.from_dlpack(world.buffer_view(nuka.ARTICULATION_LINK_POSE))
            targets.reshape(2, 3).copy_(poses.reshape(2, -1, 7)[:, 3, :3])
            targets.reshape(2, 3)[:, 0] += 0.002
            assert world.download_field(nuka.TASK_NULLSPACE_STIFFNESS).size == 2
            assert world.download_field(nuka.TASK_NULLSPACE_DAMPING).size == 2
        else:
            targets.add_(0.02)
        persistent = targets.clone()
        nuka.sync()
        world.set_execution_mode("graph")
        world.step_n(4)
        assert not np.any(world.download_field(nuka.ENV_STATUS))
        assert np.isfinite(world.download_field(nuka.JOINT_POSITION)).all()
        assert np.isfinite(world.download_field(nuka.ACTUATOR_EFFORT)).all()
        world.reset_envs([1])
        np.testing.assert_array_equal(world.download_field(nuka.JOINT_POSITION).reshape(2, -1)[1], initial[1])
        assert torch.equal(targets, persistent)
        targets.reshape(2, -1)[0, -1] = float("nan")
        nuka.sync()
        world.step()
        status = world.download_field(nuka.ENV_STATUS).ravel()
        assert status[0] & nuka.ENV_STATUS_CONTROL_FAILURE
        assert status[1] == 0
        assert np.isfinite(world.download_field(nuka.JOINT_POSITION)).all()
        targets.copy_(persistent)
        nuka.sync()
        world.reset_envs([0])
        assert not np.any(world.download_field(nuka.ENV_STATUS))


def test_plastic_material_cook_graph_reset_and_readout(device, tmp_path):
    material = Soft.ElastoPlastic(youngs=30000.0, poisson=0.0, yield_stress=200.0,
                                 hardening_modulus=4000.0, dx=0.025, substeps=10)
    material = replace(material, contact_capacity=4096)
    fields = (nuka.PARTICLE_DEFORMATION_GRADIENT,
              nuka.PARTICLE_PLASTIC_DEFORMATION_GRADIENT,
              nuka.PARTICLE_EQUIVALENT_PLASTIC_STRAIN,
              nuka.GRID_CONTACT_ATTEMPTED, nuka.GRID_CONTACT_RETAINED,
              nuka.GRID_CONTACT_PEAK, nuka.GRID_CONTACT_OVERFLOW,
              nuka.MPM_BOUNDARY_IMPULSE, nuka.MPM_BOUNDARY_ANGULAR_IMPULSE)
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
                    for f in (nuka.GRID_CONTACT_ATTEMPTED, nuka.GRID_CONTACT_PEAK,
                              nuka.GRID_CONTACT_OVERFLOW):
                        host = world.download_field(f)
                        view = torch.from_dlpack(world.buffer_view(f))
                        assert host.dtype == np.uint64 and view.dtype == torch.uint64
                        np.testing.assert_array_equal(view.cpu().numpy().ravel(), host.ravel())
                    assert np.all(final[5] > 0) and not np.any(final[6])
                    assert np.all(final[5] <= material.contact_capacity)
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
                    peak_view = torch.from_dlpack(world.buffer_view(nuka.GRID_CONTACT_PEAK))
                    large_counts = np.array([2**40 + 3, 2**40 + 7], dtype=np.uint64)
                    peak_view.copy_(torch.from_numpy(large_counts).to(peak_view.device).reshape(peak_view.shape))
                    torch.cuda.synchronize()
                    np.testing.assert_array_equal(world.download_field(nuka.GRID_CONTACT_PEAK), large_counts)
                    world.reset_envs([0])
                    np.testing.assert_array_equal(world.download_field(nuka.GRID_CONTACT_PEAK),
                                                  np.array([0, large_counts[1]], dtype=np.uint64))
                    world.reset()
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
