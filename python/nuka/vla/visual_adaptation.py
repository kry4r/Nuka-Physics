"""Visual preprocessing to match LeRobot LIBERO training distribution."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def match_histogram_numpy(
    source: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Match source image histogram to reference (per-channel)."""
    if source.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: source={source.shape} ref={reference.shape}"
        )
    matched = np.zeros_like(source)
    for channel in range(source.shape[2]):
        src_ch = source[:, :, channel].ravel()
        ref_ch = reference[:, :, channel].ravel()
        src_values, src_counts = np.unique(src_ch, return_counts=True)
        ref_values, ref_counts = np.unique(ref_ch, return_counts=True)
        src_quantiles = np.cumsum(src_counts).astype(np.float64)
        src_quantiles /= src_quantiles[-1]
        ref_quantiles = np.cumsum(ref_counts).astype(np.float64)
        ref_quantiles /= ref_quantiles[-1]
        interp = np.interp(src_quantiles, ref_quantiles, ref_values)
        matched[:, :, channel] = interp[
            np.searchsorted(src_values, src_ch)
        ].reshape(source.shape[:2])
    return matched.astype(source.dtype)


class LiberoVisualAdapter:
    """Adapt Nuka renders to match LIBERO MuJoCo visual distribution."""

    def __init__(self, reference_dir: str):
        from pathlib import Path

        ref_path = Path(reference_dir)
        self.ref_agentview = np.array(
            Image.open(ref_path / "libero_agentview_policy_input.png")
        )
        self.ref_eye_in_hand = np.array(
            Image.open(ref_path / "libero_eye_in_hand_policy_input.png")
        )

    def adapt_agentview(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Match agentview histogram to LIBERO reference."""
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
        else:
            image_np = image
        matched = match_histogram_numpy(image_np, self.ref_agentview)
        return torch.from_numpy(matched).to(
            device=image.device if isinstance(image, torch.Tensor) else "cpu"
        )

    def adapt_eye_in_hand(
        self, image: np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """Match eye_in_hand histogram to LIBERO reference."""
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
        else:
            image_np = image
        matched = match_histogram_numpy(image_np, self.ref_eye_in_hand)
        return torch.from_numpy(matched).to(
            device=image.device if isinstance(image, torch.Tensor) else "cpu"
        )


__all__ = ["LiberoVisualAdapter", "match_histogram_numpy"]
