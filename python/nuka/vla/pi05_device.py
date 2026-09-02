"""Strict local LeRobot pi0.5 inference for the Panda checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file


@dataclass(frozen=True)
class Pi05InferenceResult:
    actions: torch.Tensor
    inference_seconds: float
    allocated_gib: float
    reserved_gib: float
    host_image_copies: int


class Pi05DevicePolicy:
    """Load all pi0.5 weights strictly and infer 50-step Panda action chunks."""

    _EMBED_ALIAS = (
        "model.paligemma_with_expert.paligemma.model."
        "language_model.embed_tokens.weight"
    )
    _EMBED_SOURCE = "model.paligemma_with_expert.paligemma.lm_head.weight"

    def __init__(self, checkpoint: str | Path, tokenizer: str | Path):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        import lerobot.policies.pi05.processor_pi05  # noqa: F401

        self.checkpoint = Path(checkpoint)
        self.tokenizer = Path(tokenizer)
        if not (self.checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(self.checkpoint / "model.safetensors")
        if not (self.tokenizer / "tokenizer.json").is_file():
            raise FileNotFoundError(self.tokenizer / "tokenizer.json")

        config = PreTrainedConfig.from_pretrained(self.checkpoint, local_files_only=True)
        if not isinstance(config, PI05Config):
            raise TypeError(f"expected PI05Config, got {type(config).__name__}")
        config.compile_model = False
        config.gradient_checkpointing = False
        if config.device != "cuda" or config.dtype != "bfloat16":
            raise RuntimeError(
                f"unexpected pi0.5 runtime contract: device={config.device} dtype={config.dtype}")
        self.config = config

        self.policy = PI05Policy(config)
        state = load_file(str(self.checkpoint / "model.safetensors"), device="cpu")
        fixed = self.policy._fix_pytorch_state_dict_keys(state, config)
        remapped = {
            key if key.startswith("model.") else f"model.{key}": value
            for key, value in fixed.items()
        }
        if self._EMBED_ALIAS in remapped or self._EMBED_SOURCE not in remapped:
            raise RuntimeError("unexpected PaliGemma tied-embedding checkpoint layout")
        remapped[self._EMBED_ALIAS] = remapped[self._EMBED_SOURCE]
        incompatible = self.policy.load_state_dict(remapped, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"pi0.5 key mismatch: missing={incompatible.missing_keys} "
                f"unexpected={incompatible.unexpected_keys}")
        self.policy.eval()
        del state, fixed, remapped

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={
                "tokenizer_processor": {"tokenizer_name": str(self.tokenizer)}
            },
        )
        self.parameter_count = sum(parameter.numel() for parameter in self.policy.parameters())
        if self.parameter_count != 3_616_757_520:
            raise RuntimeError(f"unexpected pi0.5 parameter count: {self.parameter_count}")

    @property
    def device(self) -> torch.device:
        return next(self.policy.parameters()).device

    def _image_tensor(self, image: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, int]:
        host_copy = 0
        if isinstance(image, np.ndarray):
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError(f"expected uint8 HWC image (480,640,3), got {image.shape} {image.dtype}")
            tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
            host_copy = 1
        else:
            tensor = image
            if tensor.shape == (480, 640, 3):
                tensor = tensor.permute(2, 0, 1)
            if tensor.shape != (3, 480, 640):
                raise ValueError(f"expected CHW image (3,480,640), got {tuple(tensor.shape)}")
        if tensor.dtype == torch.uint8:
            tensor = tensor.to(device=self.device, dtype=torch.float32).div_(255.0)
        else:
            tensor = tensor.to(device=self.device, dtype=torch.float32)
            if float(tensor.detach().amax()) > 1.0:
                tensor = tensor.div(255.0)
        return tensor, host_copy

    def predict_chunk(
        self,
        state: np.ndarray | torch.Tensor,
        image: np.ndarray | torch.Tensor,
        task: str,
        *,
        seed: int | None = None,
    ) -> Pi05InferenceResult:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(8)
        if not torch.isfinite(state_tensor).all():
            raise ValueError("non-finite Panda state")
        image_tensor, host_copy = self._image_tensor(image)
        observation: dict[str, Any] = {
            "observation.state": state_tensor,
            "observation.images.camera_1": image_tensor,
            "task": task,
        }
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        started = time.perf_counter()
        with torch.inference_mode():
            batch = self.preprocessor(observation)
            raw_actions = self.policy.predict_action_chunk(batch)
            actions = self.postprocessor(raw_actions)
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        actions = actions.detach().to(device="cpu", dtype=torch.float32)
        if actions.shape != (1, 50, 8) or not torch.isfinite(actions).all():
            raise RuntimeError(
                f"invalid pi0.5 action chunk: shape={tuple(actions.shape)} "
                f"finite={bool(torch.isfinite(actions).all())}")
        return Pi05InferenceResult(
            actions=actions[0],
            inference_seconds=elapsed,
            allocated_gib=torch.cuda.memory_allocated(self.device) / (2**30),
            reserved_gib=torch.cuda.memory_reserved(self.device) / (2**30),
            host_image_copies=host_copy,
        )


__all__ = ["Pi05DevicePolicy", "Pi05InferenceResult"]
