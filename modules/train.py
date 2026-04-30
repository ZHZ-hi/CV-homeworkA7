from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from modules.data import (
    SampleImage,
    augmented_batch,
    make_augmented_view,
    samples_to_batch,
    tensor_to_pil,
)
from modules.masking import apply_block_mask
from modules.models import TinyAutoEncoder, TinySimCLR, nt_xent_loss


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _inpaint(masked: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked * mask + prediction * (1.0 - mask)


def _reconstruction_loss(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    missing = 1.0 - mask
    masked_loss = ((output - target).pow(2) * missing).sum() / missing.sum().clamp_min(1.0)
    full_loss = F.mse_loss(output, target)
    return 0.75 * masked_loss + 0.25 * full_loss


def run_mask_reconstruction_demo(
    samples: list[SampleImage],
    mask_ratio: float,
    epochs: int,
    seed: int,
    learning_rate: float = 0.003,
    steps_per_epoch: int = 4,
    mask_grid_size: int = 8,
) -> dict:
    _set_seed(seed)
    batch = samples_to_batch(samples)
    model = TinyAutoEncoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    with torch.no_grad():
        masked, mask = apply_block_mask(batch, mask_ratio, seed, grid_size=mask_grid_size)
        before_raw = model(masked)
        before = _inpaint(masked, before_raw, mask)
        initial_loss = F.mse_loss(before, batch).item()

    history = [{"epoch": 0, "loss": initial_loss}]
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for step in range(steps_per_epoch):
            masked, mask = apply_block_mask(batch, mask_ratio, seed + epoch * 97 + step, grid_size=mask_grid_size)
            output = model(masked)
            loss = _reconstruction_loss(output, batch, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach())

        with torch.no_grad():
            eval_masked, eval_mask = apply_block_mask(batch, mask_ratio, seed + 999, grid_size=mask_grid_size)
            eval_output = _inpaint(eval_masked, model(eval_masked), eval_mask)
            eval_loss = F.mse_loss(eval_output, batch).item()
        history.append({"epoch": epoch, "loss": eval_loss})

    with torch.no_grad():
        masked, mask = apply_block_mask(batch, mask_ratio, seed + 999, grid_size=mask_grid_size)
        after_raw = model(masked)
        after = _inpaint(masked, after_raw, mask)
        final_loss = F.mse_loss(after, batch).item()

    comparison = [
        {"设置": "训练前补全", "MSE Loss": round(initial_loss, 4)},
        {"设置": "训练后补全", "MSE Loss": round(final_loss, 4)},
    ]
    for ratio in (0.25, 0.50):
        eval_masked, eval_mask = apply_block_mask(batch, ratio, seed + int(ratio * 1000), grid_size=mask_grid_size)
        eval_inpainted = _inpaint(eval_masked, model(eval_masked), eval_mask)
        eval_loss = F.mse_loss(eval_inpainted, batch).item()
        comparison.append({"设置": f"{int(ratio * 100)}% 遮挡评估", "MSE Loss": round(eval_loss, 4)})

    preview_index = 0
    return {
        "history": history,
        "comparison": comparison,
        "preview": (
            tensor_to_pil(before[preview_index]),
            tensor_to_pil(masked[preview_index]),
            tensor_to_pil(after[preview_index]),
            tensor_to_pil(batch[preview_index]),
        ),
    }


def run_simclr_demo(
    samples: list[SampleImage],
    augment_mode: str,
    epochs: int,
    seed: int,
    learning_rate: float = 0.003,
    steps_per_epoch: int = 3,
    temperature: float = 0.25,
    top1_margin: float = 0.02,
) -> dict:
    _set_seed(seed)
    model = TinySimCLR()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    view_a, view_b = augmented_batch(samples, augment_mode, seed)
    with torch.no_grad():
        initial_loss, initial_sim, initial_neg, initial_top1 = nt_xent_loss(
            model(view_a), model(view_b), temperature=temperature, top1_margin=top1_margin
        )

    history = [
        {
            "epoch": 0,
            "loss": float(initial_loss),
            "alignment_gap": float(initial_sim - initial_neg),
            "positive_similarity": float(initial_sim),
            "negative_similarity": float(initial_neg),
            "top1": float(initial_top1),
        }
    ]
    for epoch in range(1, epochs + 1):
        for step in range(steps_per_epoch):
            view_a, view_b = augmented_batch(samples, augment_mode, seed + epoch * 37 + step * 1009)
            loss, pos_sim, neg_sim, top1 = nt_xent_loss(
                model(view_a), model(view_b), temperature=temperature, top1_margin=top1_margin
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        view_a, view_b = augmented_batch(samples, augment_mode, seed + epoch * 37)
        with torch.no_grad():
            loss, pos_sim, neg_sim, top1 = nt_xent_loss(
                model(view_a), model(view_b), temperature=temperature, top1_margin=top1_margin
            )
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach()),
                "alignment_gap": float((pos_sim - neg_sim).detach()),
                "positive_similarity": float(pos_sim.detach()),
                "negative_similarity": float(neg_sim.detach()),
                "top1": float(top1.detach()),
            }
        )

    with torch.no_grad():
        view_a, view_b = augmented_batch(samples, augment_mode, seed + 999)
        final_loss, final_sim, final_neg, final_top1 = nt_xent_loss(
            model(view_a), model(view_b), temperature=temperature, top1_margin=top1_margin
        )

    comparison = [
        {
            "设置": "训练前",
            "Loss": round(float(initial_loss), 4),
            "正负间隔": round(float(initial_sim - initial_neg), 4),
            "Top-1": round(float(initial_top1), 4),
        },
        {
            "设置": "训练后",
            "Loss": round(float(final_loss), 4),
            "正负间隔": round(float(final_sim - final_neg), 4),
            "Top-1": round(float(final_top1), 4),
        },
    ]
    for mode in ("crop_color", "flip_gray"):
        eval_a, eval_b = augmented_batch(samples, mode, seed + 555)
        eval_loss, eval_sim, eval_neg, eval_top1 = nt_xent_loss(
            model(eval_a), model(eval_b), temperature=temperature, top1_margin=top1_margin
        )
        label = "随机裁剪+颜色扰动" if mode == "crop_color" else "翻转+灰度扰动"
        comparison.append(
            {
                "设置": label,
                "Loss": round(float(eval_loss.detach()), 4),
                "正负间隔": round(float((eval_sim - eval_neg).detach()), 4),
                "Top-1": round(float(eval_top1.detach()), 4),
            }
        )

    return {
        "history": history,
        "comparison": comparison,
        "views": (
            make_augmented_view(samples[0].image, augment_mode, seed + 1),
            make_augmented_view(samples[0].image, augment_mode, seed + 2),
        ),
    }
