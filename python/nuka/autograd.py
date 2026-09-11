"""PyTorch interfaces for production forward steps and recorded PD rollouts.

Torch is optional for nuka itself and is imported when this module is requested.
"""

from __future__ import annotations

import numpy as np
import torch

from ._nuka_ext import Field


class _NukaPhysicsStep(torch.autograd.Function):
    """Snapshot a production step; an unrecorded step has no backward operation."""

    @staticmethod
    def forward(ctx, world, actions: torch.Tensor) -> torch.Tensor:
        target = torch.from_dlpack(world.buffer_view(Field.DRIVE_TARGET))
        expected = (world.env_count, world.action_dim)
        if tuple(actions.shape) != expected or tuple(target[:, 1:].shape) != expected:
            raise ValueError(f"actions must have shape {expected}")
        if actions.dtype != target.dtype or actions.device != target.device:
            raise ValueError(f"actions must use {target.dtype} on {target.device}")
        target[:, 1:].copy_(actions.detach())
        world.step()
        positions = torch.from_dlpack(world.buffer_view(Field.JOINT_POSITION))
        return positions[:, 1:].clone()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        raise NotImplementedError("step() does not record an adjoint; gradients are unavailable")


def step(world, actions: torch.Tensor) -> torch.Tensor:
    """Advance production physics and return a snapshot of actuated positions.

    Actions have shape (env_count, action_dim) and match the world's device
    and float32 storage. Differentiation is rejected before changing the world;
    use torch.no_grad() for inference with a tensor that requires gradients.
    """
    if torch.is_grad_enabled() and actions.requires_grad:
        raise NotImplementedError("step() does not record an adjoint; gradients are unavailable")
    return _NukaPhysicsStep.apply(world, actions)


_OBS = ("q", "qdot", "v_root")


def _seed_from_grad(obs: str, grad_out: torch.Tensor, n: int) -> np.ndarray:
    """Pack q, qdot, or root velocity into the C ABI's float32[8*n] loss seed."""
    seed = np.zeros(8 * n, dtype=np.float32)
    gradient = grad_out.detach().reshape(-1).to("cpu", torch.float32).contiguous().numpy()
    if obs == "q":
        seed[1:1 + gradient.size] = gradient
    elif obs == "qdot":
        seed[n + 1:n + 1 + gradient.size] = gradient
    elif obs == "v_root":
        seed[2 * n:2 * n + 6] = gradient
    else:
        raise ValueError(f"obs must be one of {_OBS}, got {obs!r}")
    return seed


class _NukaDiffRollout(torch.autograd.Function):
    """Recorded contact-free PD rollout with action and scalar-mass adjoints."""

    @staticmethod
    def forward(ctx, world, tape, actions_seq, params, param_link_indices, obs):
        if obs not in _OBS:
            raise ValueError(f"obs must be one of {_OBS}, got {obs!r}")
        if world.env_count != 1:
            raise ValueError("the recorded PD rollout requires env_count == 1")
        if actions_seq.ndim != 2 or actions_seq.shape[1] != world.base_link_count - 1:
            raise ValueError("actions_seq must have shape (steps, base_link_count - 1)")
        steps, action_dim = actions_seq.shape
        if params.numel() > 0:
            values = params.detach().to("cpu", torch.float32).contiguous().reshape(-1)
            for index, link in enumerate(param_link_indices):
                world.set_link_mass(int(link), float(values[index]))

        targets = torch.from_dlpack(world.buffer_view(Field.DRIVE_TARGET))
        for index in range(steps):
            targets[:, 1:].copy_(actions_seq[index].detach().reshape(1, action_dim))
            tape.step_with_tape()

        if obs == "q":
            out = torch.from_dlpack(tape.state_view(Field.JOINT_POSITION))[:, 1:].clone()
        elif obs == "qdot":
            out = torch.from_dlpack(tape.state_view(Field.JOINT_VELOCITY))[:, 1:].clone()
        else:
            out = torch.from_dlpack(tape.state_view(Field.LINK_VELOCITY))[0, 0, :].clone()

        ctx.save_for_backward(actions_seq, params)
        ctx.tape = tape
        ctx.obs = obs
        ctx.n = tape.link_count
        ctx.action_dim = action_dim
        ctx.param_link_indices = tuple(int(index) for index in param_link_indices)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        actions_seq, params = ctx.saved_tensors
        seed = _seed_from_grad(ctx.obs, grad_out, ctx.n)
        action_gradient, parameter_gradient = ctx.tape.backward(seed)
        grad_actions = (
            torch.from_numpy(np.ascontiguousarray(action_gradient[:, 1:1 + ctx.action_dim]))
            .to(device=actions_seq.device, dtype=actions_seq.dtype)
            .reshape(actions_seq.shape)
        )
        if params.numel() > 0:
            grad_params = (
                torch.from_numpy(np.ascontiguousarray(parameter_gradient[list(ctx.param_link_indices)]))
                .to(params)
                .reshape(params.shape)
            )
        else:
            grad_params = torch.zeros_like(params)
        return None, None, grad_actions, grad_params, None, None


def differentiable_rollout(world, tape, actions_seq, *, params=None,
                           param_link_indices=None, obs="qdot"):
    """Run the recorded contact-free PD map and differentiate its final observation.

    This map uses explicit damping and does not match the production contact
    pipeline's implicit damping. Its gradients do not describe world.step().
    Floating-base pose derivatives are incomplete for multi-step rollouts.

    actions_seq is (steps, action_dim). Observations are actuated q or qdot with
    shape (1, action_dim), or v_root with shape (6,) in angular/linear order.
    Optional masses and param_link_indices have equal length. Each forward
    evaluation requires a fresh single-environment world and tape; the recorded
    rollout changes the world in place.
    """
    if params is None:
        params = torch.empty(0, dtype=torch.float32)
        param_link_indices = ()
    elif param_link_indices is None:
        raise ValueError("param_link_indices is required when params is given")
    return _NukaDiffRollout.apply(
        world, tape, actions_seq, params, tuple(param_link_indices), obs
    )


def differentiable_step(world, tape, actions, *, params=None,
                        param_link_indices=None, obs="qdot"):
    """Run one recorded PD step; see differentiable_rollout for its limitations."""
    if actions.dim() == 1:
        actions_seq = actions.reshape(1, -1)
    else:
        actions_seq = actions.reshape(1, actions.shape[-1])
    return differentiable_rollout(
        world, tape, actions_seq, params=params,
        param_link_indices=param_link_indices, obs=obs,
    )


__all__ = [
    "step", "_NukaPhysicsStep", "differentiable_rollout", "differentiable_step",
    "_NukaDiffRollout", "_seed_from_grad",
]
