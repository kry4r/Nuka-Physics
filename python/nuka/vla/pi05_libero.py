"""Strict local pi0.5 inference for the LeRobot LIBERO v0.4.4 checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file


@dataclass(frozen=True)
class LiberoPi05InferenceResult:
    actions: torch.Tensor
    inference_seconds: float
    allocated_gib: float
    reserved_gib: float
    host_image_copies: int
    preprocess_seconds: float = 0.0
    model_seconds: float = 0.0
    postprocess_seconds: float = 0.0
    synchronize_seconds: float = 0.0


class LiberoPi05Policy:
    """Load every pi0.5 weight strictly and infer 50-step LIBERO chunks."""

    _EMBED_ALIAS = (
        "model.paligemma_with_expert.paligemma.model."
        "language_model.embed_tokens.weight"
    )
    _EMBED_SOURCE = "model.paligemma_with_expert.paligemma.lm_head.weight"
    _IMAGE_KEYS = (
        "observation.images.image",
        "observation.images.image2",
    )

    def __init__(self, checkpoint: str | Path, tokenizer: str | Path):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        import lerobot.policies.pi05.processor_pi05  # noqa: F401

        self.checkpoint = Path(checkpoint)
        self.tokenizer = Path(tokenizer)
        adapter_weights_path = self.checkpoint / "adapter_model.safetensors"
        full_weights_path = self.checkpoint / "model.safetensors"
        self.is_peft = adapter_weights_path.is_file()
        if self.is_peft:
            config_path = self.checkpoint / "config.json"
            adapter_config_path = self.checkpoint / "adapter_config.json"
            if not config_path.is_file() or not adapter_config_path.is_file():
                raise FileNotFoundError(
                    "PEFT checkpoint requires config.json and adapter_config.json"
                )
            with adapter_config_path.open(encoding="utf-8") as stream:
                adapter_config = json.load(stream)
            base_name = adapter_config.get("base_model_name_or_path")
            if not base_name:
                raise RuntimeError("PEFT checkpoint has no base_model_name_or_path")
            self.base_checkpoint = Path(base_name)
            if not self.base_checkpoint.is_dir() and base_name == "lerobot/pi05_base":
                self.base_checkpoint = self.checkpoint.parent / "pi05-base"
            config_source = self.checkpoint
        else:
            if not full_weights_path.is_file():
                raise FileNotFoundError(full_weights_path)
            self.base_checkpoint = self.checkpoint
            config_source = self.checkpoint
        if not (self.tokenizer / "tokenizer.json").is_file():
            raise FileNotFoundError(self.tokenizer / "tokenizer.json")

        config = PreTrainedConfig.from_pretrained(config_source, local_files_only=True)
        if not isinstance(config, PI05Config):
            raise TypeError(f"expected PI05Config, got {type(config).__name__}")
        config.compile_model = False
        config.gradient_checkpointing = False
        if config.device != "cuda" or config.dtype != "bfloat16":
            raise RuntimeError(
                "unexpected LIBERO pi0.5 runtime contract: "
                f"device={config.device} dtype={config.dtype}"
            )
        image_keys = tuple(
            key for key in config.image_features if "empty_camera" not in key
        )
        state_shape = tuple(config.input_features["observation.state"].shape)
        action_shape = tuple(config.output_features["action"].shape)
        if (
            image_keys != self._IMAGE_KEYS
            or state_shape != (8,)
            or action_shape != (7,)
            or config.chunk_size != 50
            or config.num_inference_steps != 10
            or config.empty_cameras not in (0, 1)
        ):
            raise RuntimeError(
                "unexpected LIBERO checkpoint features: "
                f"images={image_keys} state={state_shape} action={action_shape} "
                f"chunk={config.chunk_size} flow_steps={config.num_inference_steps} "
                f"empty_cameras={config.empty_cameras}"
            )
        self.config = config

        if self.is_peft:
            from peft import PeftConfig, PeftModel

            if not self.base_checkpoint.is_dir():
                raise FileNotFoundError(
                    "PEFT base checkpoint is not local: "
                    f"{self.base_checkpoint}. Download lerobot/pi05_base first."
                )
            peft_config = PeftConfig.from_pretrained(self.checkpoint)
            base_config = PreTrainedConfig.from_pretrained(
                self.base_checkpoint, local_files_only=True
            )
            base_config.compile_model = False
            base_config.gradient_checkpointing = False
            base_policy = PI05Policy.from_pretrained(
                self.base_checkpoint,
                # Keep the adapter's 8D LIBERO feature contract while loading
                # the base model's architecture and weights.
                config=config,
                local_files_only=True,
            )
            self.base_parameter_count = sum(
                parameter.numel() for parameter in base_policy.parameters()
            )
            if self.base_parameter_count != 3_616_757_520:
                raise RuntimeError(
                    "unexpected pi0.5 base parameter count: "
                    f"{self.base_parameter_count}"
                )
            self.policy = PeftModel.from_pretrained(
                base_policy, self.checkpoint, config=peft_config, is_trainable=False
            )
        else:
            self.policy = PI05Policy(config)
            state = load_file(str(full_weights_path), device="cpu")
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
                    f"unexpected={incompatible.unexpected_keys}"
                )
            del state, fixed, remapped
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={
                "tokenizer_processor": {"tokenizer_name": str(self.tokenizer)},
                "device_processor": {"device": "cuda"},
            },
            postprocessor_overrides={
                "device_processor": {"device": "cuda"},
            },
        )
        self.parameter_count = sum(
            parameter.numel() for parameter in self.policy.parameters()
        )
        self.trainable_parameter_count = sum(
            parameter.numel()
            for parameter in self.policy.parameters()
            if parameter.requires_grad
        )
        if not self.is_peft and self.parameter_count != 3_616_757_520:
            raise RuntimeError(
                f"unexpected pi0.5 parameter count: {self.parameter_count}"
            )
        if self.is_peft and self.base_parameter_count != 3_616_757_520:
            raise RuntimeError(
                f"unexpected pi0.5 base parameter count: {self.base_parameter_count}"
            )

    @property
    def device(self) -> torch.device:
        return next(self.policy.parameters()).device

    def _image_tensor(
        self, image: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        host_copy = int(isinstance(image, np.ndarray))
        tensor = (
            torch.from_numpy(np.ascontiguousarray(image))
            if isinstance(image, np.ndarray)
            else image
        )
        if tensor.ndim != 3:
            raise ValueError(f"expected a 3D LIBERO RGB image, got {tuple(tensor.shape)}")
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        elif tensor.shape[0] != 3:
            raise ValueError(
                f"expected HWC or CHW LIBERO RGB image, got {tuple(tensor.shape)}"
            )
        if min(tensor.shape[1:]) < 64:
            raise ValueError(f"LIBERO image is too small: {tuple(tensor.shape)}")
        if tensor.dtype == torch.uint8:
            tensor = tensor.to(device=self.device, dtype=torch.float32).div_(255.0)
        else:
            tensor = tensor.to(device=self.device, dtype=torch.float32)
            if isinstance(image, np.ndarray) and float(np.nanmax(image)) > 1.0:
                tensor = tensor.div(255.0)
        if not torch.isfinite(tensor).all():
            raise ValueError("non-finite LIBERO image")
        return tensor, host_copy

    def predict_chunk(
        self,
        state: np.ndarray | torch.Tensor,
        agentview: np.ndarray | torch.Tensor,
        eye_in_hand: np.ndarray | torch.Tensor,
        task: str,
        *,
        seed: int | None = None,
        profile: bool = False,
    ) -> LiberoPi05InferenceResult:
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).reshape(8)
        if not torch.isfinite(state_tensor).all():
            raise ValueError("non-finite LIBERO state")
        agent_tensor, agent_copy = self._image_tensor(agentview)
        wrist_tensor, wrist_copy = self._image_tensor(eye_in_hand)
        observation: dict[str, Any] = {
            "observation.state": state_tensor,
            "observation.images.image": agent_tensor,
            "observation.images.image2": wrist_tensor,
            "task": task,
        }
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        started = time.perf_counter()
        with torch.inference_mode():
            if profile:
                torch.cuda.synchronize(self.device)
            stage_started = time.perf_counter()
            batch = self.preprocessor(observation)
            if profile:
                torch.cuda.synchronize(self.device)
            preprocess_seconds = time.perf_counter() - stage_started
            if profile:
                print(
                    f"[libero-infer] preprocess_done seconds={preprocess_seconds:.3f}",
                    flush=True,
                )
            stage_started = time.perf_counter()
            raw_actions = self.policy.predict_action_chunk(batch)
            if profile:
                torch.cuda.synchronize(self.device)
            model_seconds = time.perf_counter() - stage_started
            if profile:
                print(
                    f"[libero-infer] model_done seconds={model_seconds:.3f}",
                    flush=True,
                )
            stage_started = time.perf_counter()
            actions = self.postprocessor(raw_actions)
            if profile:
                torch.cuda.synchronize(self.device)
            postprocess_seconds = time.perf_counter() - stage_started
            if profile:
                print(
                    f"[libero-infer] postprocess_done seconds={postprocess_seconds:.3f}",
                    flush=True,
                )
        sync_started = time.perf_counter()
        torch.cuda.synchronize(self.device)
        synchronize_seconds = time.perf_counter() - sync_started
        elapsed = time.perf_counter() - started
        actions = actions.detach().to(device=self.device, dtype=torch.float32)
        if actions.shape != (1, 50, 7) or not torch.isfinite(actions).all():
            raise RuntimeError(
                f"invalid LIBERO pi0.5 action chunk: shape={tuple(actions.shape)} "
                f"finite={bool(torch.isfinite(actions).all())}"
            )
        return LiberoPi05InferenceResult(
            actions=actions[0],
            inference_seconds=elapsed,
            allocated_gib=torch.cuda.memory_allocated(self.device) / (2**30),
            reserved_gib=torch.cuda.memory_reserved(self.device) / (2**30),
            host_image_copies=agent_copy + wrist_copy,
            preprocess_seconds=preprocess_seconds,
            model_seconds=model_seconds,
            postprocess_seconds=postprocess_seconds,
            synchronize_seconds=synchronize_seconds,
        )


__all__ = ["LiberoPi05InferenceResult", "LiberoPi05Policy"]
