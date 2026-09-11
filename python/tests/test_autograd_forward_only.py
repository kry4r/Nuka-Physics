"""Production forward snapshots and explicit failure for unrecorded gradients."""

from pathlib import Path

import pytest
import torch

import nuka

SCENE = str(Path(__file__).resolve().parents[2] / "examples/scenes/go2_float.usda")


@pytest.fixture(scope="module")
def device():
    with nuka.Device.create(0) as device:
        yield device


def test_action_dim_is_actuated_dof(device):
    with nuka.World.create_from_scene(device, SCENE, 4) as world:
        assert world.base_link_count == 13
        assert world.action_dim == 12
        targets = torch.from_dlpack(world.buffer_view(nuka.DRIVE_TARGET))
        assert tuple(targets[:, 1:].shape) == (world.env_count, world.action_dim)


def test_autograd_forward_only(device):
    with nuka.World.create_from_scene(device, SCENE, 4) as world:
        actions = torch.zeros(world.env_count, world.action_dim, device="cuda", dtype=torch.float32)
        output = nuka.autograd.step(world, actions)
        nuka.sync()
        assert output.is_cuda and output.dtype == torch.float32
        assert tuple(output.shape) == tuple(actions.shape)
        assert torch.isfinite(output).all()
        assert not output.requires_grad
        saved = output.clone()
        nuka.autograd.step(world, torch.full_like(actions, 0.05))
        nuka.sync()
        torch.testing.assert_close(output, saved, rtol=0, atol=0)


def test_autograd_rejects_unrecorded_gradients_before_mutation(device):
    with nuka.World.create_from_scene(device, SCENE, 4) as world:
        actions = torch.full((world.env_count, world.action_dim), 0.05,
                             device="cuda", dtype=torch.float32, requires_grad=True)
        fields = (nuka.DRIVE_TARGET, nuka.JOINT_POSITION, nuka.JOINT_VELOCITY, nuka.BASE_POSE)
        before = [torch.from_dlpack(world.buffer_view(field)).clone() for field in fields]
        with pytest.raises(NotImplementedError, match="does not record an adjoint"):
            nuka.autograd.step(world, actions)
        for field, expected in zip(fields, before):
            actual = torch.from_dlpack(world.buffer_view(field))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actions.grad is None
        with torch.no_grad():
            output = nuka.autograd.step(world, actions)
        nuka.sync()
        assert torch.isfinite(output).all()
        assert not output.requires_grad
