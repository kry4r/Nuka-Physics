"""pytest: caller-supplied CUDA stream -- torch stream becomes the engine stream.

Proves the p02 stream exit criterion:
  * nuka.Device.create(ordinal, stream_ptr=<torch stream ptr>) constructs and a
    World on it steps cleanly, producing finite post-step state;
  * under `with torch.cuda.stream(s):`, a torch op reading the post-step
    buffer_view on the SAME stream `s` -- with NO explicit nuka.sync() between the
    physics step and the torch read -- yields the correct post-step value
    (bit-identical to a reference run that DOES sync). This shows shared-stream
    ordering holds: torch and physics serialize on the one stream.

HONEST FRAMING (see report item (e)): this is a POSITIVE correctness check, not
an adversarial "stale data if streams differ" proof. The terminal device->host
transfer of the result scalar imposes ordering regardless, and a negative
control would depend on a nondeterministic race -- which would violate this
project's D1 determinism discipline. We assert the shared-stream path is correct
and sync-free up to the result read; we do NOT claim to have proven the absence
of any implicit sync.

Run (single GPU only):
    export CUDA_VISIBLE_DEVICES=0
    python -m pytest python/tests/test_caller_stream.py -v
"""

from __future__ import annotations

import pytest
import torch

import nuka

SCENE = "/root/Nuka-Physics/examples/scenes/go2_float.usda"
GO2_BLC = 13


def _reference_settled_q(env_count, steps):
    """Engine-owned-default-stream run WITH explicit sync -- the ground truth the
    shared-stream run must reproduce bit-for-bit (D1 determinism)."""
    with nuka.Device.create(0) as dev:
        with nuka.World.create_from_scene(dev, SCENE, env_count) as w:
            tgt = torch.from_dlpack(w.buffer_view(nuka.DRIVE_TARGET))
            tgt[:, 1:GO2_BLC] += 0.05
            w.step_n(steps)
            nuka.sync()
            q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
            return q.detach().cpu().clone()


def test_torch_stream_ptr_helper_nonzero():
    """An EXPLICIT torch stream has a non-zero cuda_stream pointer, so the
    stream-ptr branch in Device::create is actually exercised (torch's *default*
    stream can be 0, which the engine treats as 'own a default stream')."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        ptr = nuka.torch_stream_ptr()
    assert ptr == s.cuda_stream
    assert ptr != 0, "explicit torch stream must have a non-null cuda_stream"


def test_caller_stream_constructs_and_steps():
    """The stream-ptr path constructs a Device + World and steps without error,
    producing finite state (the minimum guaranteed by the task)."""
    env_count = 16
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        dev = nuka.Device.create(0, stream_ptr=s.cuda_stream)
        try:
            w = nuka.World.create_from_scene(dev, SCENE, env_count)
            try:
                w.step_n(5)
                nuka.sync()
                q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
                assert q.is_cuda and tuple(q.shape) == (env_count, GO2_BLC)
                assert torch.isfinite(q).all()
            finally:
                w.destroy()
        finally:
            dev.close()


def test_caller_stream_shared_ordering_no_explicit_sync():
    """Shared-stream ordering: on torch stream `s`, run physics on `s`, then a
    torch reduction on `s` over the post-step buffer view -- with NO nuka.sync()
    in between -- and assert the result reproduces the reference (synced) run
    bit-for-bit. If torch and physics did NOT share a stream, the torch op could
    read pre-step memory; reproducing the reference exactly shows they serialize.
    """
    env_count = 16
    steps = 8
    ref_q = _reference_settled_q(env_count, steps)

    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        dev = nuka.Device.create(0, stream_ptr=s.cuda_stream)
        try:
            w = nuka.World.create_from_scene(dev, SCENE, env_count)
            try:
                tgt = torch.from_dlpack(w.buffer_view(nuka.DRIVE_TARGET))
                tgt[:, 1:GO2_BLC] += 0.05  # same perturbation as the reference
                w.step_n(steps)
                # NO nuka.sync() here: the torch read below runs on the SAME
                # stream `s` the engine ran on, so it is ordered after the step.
                q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
                got = q.detach().cpu().clone()  # the d2h transfer is on `s` too
            finally:
                w.destroy()
        finally:
            dev.close()

    assert torch.isfinite(got).all()
    # Bit-exact match to the reference (synced, engine-default-stream) run:
    # shared-stream ordering held AND determinism (D1) is intact.
    assert torch.equal(got, ref_q), (
        "shared-stream run diverged from reference "
        f"(max|d|={(got - ref_q).abs().max().item():.3e}) -- either ordering or "
        "determinism broke"
    )
