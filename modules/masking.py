from __future__ import annotations

import torch


def apply_block_mask(
    batch: torch.Tensor,
    mask_ratio: float,
    seed: int = 0,
    grid_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask images with a grid of square patches and return masked batch plus binary mask."""
    if not 0 < mask_ratio < 1:
        raise ValueError("mask_ratio must be between 0 and 1")

    generator = torch.Generator(device=batch.device).manual_seed(seed)
    n, _, h, w = batch.shape
    patch = max(2, h // grid_size)
    gh, gw = h // patch, w // patch
    total = gh * gw
    count = max(1, int(total * mask_ratio))
    mask = torch.ones((n, 1, h, w), device=batch.device)

    for i in range(n):
        ids = torch.randperm(total, generator=generator, device=batch.device)[:count]
        for idx in ids:
            y = int(idx // gw) * patch
            x = int(idx % gw) * patch
            mask[i, :, y : y + patch, x : x + patch] = 0.0

    return batch * mask, mask
