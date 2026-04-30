from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from sklearn.datasets import load_digits
import torch
from torchvision.datasets import CIFAR10


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "samples"
CIFAR_DIR = ROOT / "data" / "cifar10"
CIFAR_SUBSET_DIR = ROOT / "data" / "cifar_subset"
_CIFAR_DOWNLOAD_FAILED = False
CIFAR_CLASSES = [
    "飞机",
    "汽车",
    "鸟",
    "猫",
    "鹿",
    "狗",
    "青蛙",
    "马",
    "船",
    "卡车",
]


@dataclass(frozen=True)
class SampleImage:
    name: str
    image: Image.Image


def ensure_sample_images() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    if list(SAMPLE_DIR.glob("*.png")):
        return

    specs = [
        ("gradient_blocks.png", _gradient_blocks),
        ("rings_and_lines.png", _rings_and_lines),
        ("soft_landscape.png", _soft_landscape),
        ("color_tiles.png", _color_tiles),
    ]
    for filename, builder in specs:
        builder(160).save(SAMPLE_DIR / filename)


def load_sample_images(image_size: int = 64, limit: int = 32) -> list[SampleImage]:
    subset_samples = load_packaged_cifar_subset(image_size=image_size, limit=limit)
    if subset_samples:
        return subset_samples

    try:
        cifar_samples = load_cifar10_images(image_size=image_size, limit=limit)
        if cifar_samples:
            return cifar_samples
    except Exception:
        pass

    digit_samples = load_digit_images(image_size=image_size, limit=min(limit, 10))
    if digit_samples:
        return digit_samples

    ensure_sample_images()
    samples: list[SampleImage] = []
    for path in sorted(SAMPLE_DIR.glob("*.png")):
        img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
        samples.append(SampleImage(path.stem.replace("_", " "), img))
    return samples


def load_packaged_cifar_subset(image_size: int = 64, limit: int = 32) -> list[SampleImage]:
    samples: list[SampleImage] = []
    for path in sorted(CIFAR_SUBSET_DIR.glob("*.png"))[:limit]:
        img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
        label = path.stem.split("_", 1)[-1].replace("_", " ")
        samples.append(SampleImage(f"CIFAR-10 子集：{label}", img))
    return samples


def load_cifar10_images(image_size: int = 64, limit: int = 8) -> list[SampleImage]:
    dataset = _load_cifar10_dataset()
    selected: list[SampleImage] = []
    for index, (image, label) in enumerate(dataset):
        img = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
        selected.append(SampleImage(f"CIFAR-10 #{index + 1}：{CIFAR_CLASSES[label]}", img))
        if len(selected) >= limit:
            break
    return selected


def dataset_source_label() -> str:
    if list(CIFAR_SUBSET_DIR.glob("*.png")):
        return "真实小数据集：CIFAR-10 子集（随仓库部署）"
    if _cifar_exists():
        return "真实小数据集：CIFAR-10"
    return "真实小数据集：scikit-learn digits（CIFAR-10 未缓存时自动使用）"


def load_digit_images(image_size: int = 64, limit: int = 8) -> list[SampleImage]:
    digits = load_digits()
    selected: list[SampleImage] = []
    seen: set[int] = set()
    for image, label in zip(digits.images, digits.target):
        label = int(label)
        if label in seen:
            continue
        seen.add(label)
        normalized = (image / image.max() * 255).astype(np.uint8)
        pil = Image.fromarray(normalized, mode="L").convert("RGB")
        pil = ImageOps.colorize(ImageOps.grayscale(pil), black="#102a43", white="#f7d070")
        pil = pil.resize((image_size, image_size), Image.Resampling.NEAREST)
        selected.append(SampleImage(f"Digits：数字 {label}", pil))
        if len(selected) >= limit:
            break
    return selected


def _load_cifar10_dataset() -> CIFAR10:
    global _CIFAR_DOWNLOAD_FAILED
    if _CIFAR_DOWNLOAD_FAILED and not _cifar_exists():
        raise RuntimeError("CIFAR-10 download failed earlier in this process.")

    download = not _cifar_exists()
    try:
        return CIFAR10(root=str(CIFAR_DIR), train=True, download=download)
    except Exception:
        _CIFAR_DOWNLOAD_FAILED = True
        raise


def _cifar_exists() -> bool:
    return (CIFAR_DIR / "cifar-10-batches-py").exists()


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def samples_to_batch(samples: list[SampleImage]) -> torch.Tensor:
    return torch.stack([pil_to_tensor(item.image) for item in samples])


def make_augmented_view(image: Image.Image, mode: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = image.convert("RGB")

    if mode == "crop_color":
        w, h = img.size
        crop = int(min(w, h) * rng.uniform(0.70, 0.90))
        left = rng.randint(0, w - crop)
        top = rng.randint(0, h - crop)
        img = img.crop((left, top, left + crop, top + crop)).resize((w, h), Image.Resampling.BICUBIC)
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.45, 1.65))
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))
        return img

    if mode == "flip_gray":
        if rng.random() > 0.35:
            img = ImageOps.mirror(img)
        gray = ImageOps.grayscale(img).convert("RGB")
        return Image.blend(img, gray, rng.uniform(0.45, 0.80))

    raise ValueError(f"Unknown augment mode: {mode}")


def augmented_batch(samples: list[SampleImage], mode: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    view_a = [make_augmented_view(item.image, mode, seed + i * 11) for i, item in enumerate(samples)]
    view_b = [make_augmented_view(item.image, mode, seed + i * 11 + 101) for i, item in enumerate(samples)]
    return torch.stack([pil_to_tensor(img) for img in view_a]), torch.stack([pil_to_tensor(img) for img in view_b])


def _gradient_blocks(size: int) -> Image.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    arr[..., 0] = (xx * 255 / size).astype(np.uint8)
    arr[..., 1] = (yy * 255 / size).astype(np.uint8)
    arr[..., 2] = (((xx // 20 + yy // 20) % 2) * 120 + 70).astype(np.uint8)
    return Image.fromarray(arr)


def _rings_and_lines(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), "#18324a")
    draw = ImageDraw.Draw(img)
    for r, color in [(70, "#f5c542"), (50, "#ef476f"), (30, "#06d6a0"), (12, "#f7fff7")]:
        draw.ellipse((size // 2 - r, size // 2 - r, size // 2 + r, size // 2 + r), outline=color, width=6)
    for x in range(0, size, 18):
        draw.line((x, 0, size - x // 2, size), fill="#86bbd8", width=2)
    return img


def _soft_landscape(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), "#8ecae6")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, size * 0.55, size, size), fill="#2a9d8f")
    draw.polygon([(0, 100), (45, 38), (92, 100)], fill="#577590")
    draw.polygon([(55, 105), (118, 24), (160, 105)], fill="#4d908e")
    draw.ellipse((110, 14, 145, 49), fill="#ffdd57")
    draw.rectangle((18, 112, 48, 144), fill="#f4a261")
    draw.polygon([(14, 112), (33, 92), (53, 112)], fill="#e76f51")
    return img


def _color_tiles(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), "#f8f9fa")
    draw = ImageDraw.Draw(img)
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#457b9d"]
    tile = size // 4
    for y in range(4):
        for x in range(4):
            color = colors[(x + y * 2) % len(colors)]
            pad = 5 + ((x + y) % 3) * 3
            draw.rounded_rectangle(
                (x * tile + pad, y * tile + pad, (x + 1) * tile - pad, (y + 1) * tile - pad),
                radius=4,
                fill=color,
            )
    return img
