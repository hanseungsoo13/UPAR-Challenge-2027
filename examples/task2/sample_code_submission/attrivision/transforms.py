"""Image transforms matching CLIP ViT-B/32 input conventions."""
from __future__ import annotations

import io
import random
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
from torchvision.transforms import InterpolationMode

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CameraAugmentation:
    """Camera and lighting shifts bounded to retain clothing and accessory cues."""

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.25))
            gamma = random.uniform(0.8, 1.2)
            lut = [round(255 * (i / 255) ** gamma) for i in range(256)]
            image = image.point(lut * 3)
        if random.random() < 0.5:
            factor = random.uniform(0.6, 0.85)
            reduced = image.resize(
                (max(1, round(image.width * factor)), max(1, round(image.height * factor))),
                Image.Resampling.BICUBIC,
            )
            image = reduced.resize(image.size, Image.Resampling.BICUBIC)
        if random.random() < 0.35:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
        if random.random() < 0.4:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=random.randint(60, 85))
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                image = decoded.convert("RGB")
        return image


class SequentialPhotoAugmentation:
    """Apply several label-preserving illumination transforms in sequence."""

    def __call__(self, image: Image.Image) -> Image.Image:
        operations = ["brightness", "contrast", "gamma"]
        random.shuffle(operations)
        for operation in operations:
            if random.random() < 0.8:
                if operation == "brightness":
                    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
                elif operation == "contrast":
                    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
                else:
                    gamma = random.uniform(0.85, 1.15)
                    lut = [round(255 * (i / 255) ** gamma) for i in range(256)]
                    image = image.point(lut * 3)
        return image


class AugMixPhoto:
    """Mix three short, photometric-only branches into one PIL image."""

    def __init__(self, alpha: float = 1.0, branches: int = 3) -> None:
        if alpha <= 0 or branches < 2:
            raise ValueError("AugMix alpha must be positive and branches must be at least 2")
        self.alpha = float(alpha)
        self.branches = int(branches)

    @staticmethod
    def _apply_random_op(image: Image.Image) -> Image.Image:
        operation = random.choice(("brightness", "contrast", "gamma"))
        if operation == "brightness":
            return ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        if operation == "contrast":
            return ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        gamma = random.uniform(0.85, 1.15)
        lut = [round(255 * (i / 255) ** gamma) for i in range(256)]
        return image.point(lut * 3)

    def __call__(self, image: Image.Image) -> Image.Image:
        chains: list[np.ndarray] = []
        for _ in range(self.branches):
            branch = image.copy()
            for _ in range(random.randint(1, 3)):
                branch = self._apply_random_op(branch)
            chains.append(np.asarray(branch, dtype=np.float32))
        weights = np.random.dirichlet([self.alpha] * self.branches).astype(np.float32)
        mixed = np.zeros_like(chains[0], dtype=np.float32)
        for weight, branch in zip(weights, chains):
            mixed += weight * branch
        original_weight = float(np.random.beta(self.alpha, self.alpha))
        original = np.asarray(image, dtype=np.float32)
        result = original_weight * original + (1.0 - original_weight) * mixed
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")


class PolicyTrainTransform:
    """Apply shared geometry, then one of the controlled photometric policies."""

    def __init__(self, geometry: transforms.Compose, policy: str, image_size: int,
                 alpha: float = 1.0) -> None:
        self.geometry = geometry
        self.policy = policy
        self.augmix = AugMixPhoto(alpha=alpha)
        self.normalize = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
        if image_size <= 0:
            raise ValueError("image_size must be positive")

    def __call__(self, image: Image.Image) -> Any:
        image = self.geometry(image)
        if self.policy == "sequential":
            return self.normalize(SequentialPhotoAugmentation()(image))
        if self.policy == "augmix":
            return self.normalize(self.augmix(image))
        if self.policy == "augmix_js":
            return (
                self.normalize(image),
                self.normalize(self.augmix(image)),
                self.normalize(self.augmix(image)),
            )
        return self.normalize(image)


def build_train_transform(
    image_size: int = 224, rotation: float = 10.0,
    augmentation: str = "current",
    hard_augmentation: bool = False,
    augmentation_policy: str = "baseline",
    augmix_alpha: float = 1.0,
) -> Any:
    policies = {"baseline", "sequential", "augmix", "augmix_js"}
    if augmentation_policy not in policies:
        raise ValueError(f"Unknown augmentation policy: {augmentation_policy}")
    if augmentation == "current":
        spatial = [transforms.RandomResizedCrop(
            image_size, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC,
        )]
    elif augmentation == "rrc_scale_050":
        spatial = [transforms.RandomResizedCrop(
            image_size, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC,
        )]
    elif augmentation == "resize_pad_crop":
        spatial = [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.Pad(10),
            transforms.RandomCrop((image_size, image_size)),
        ]
    elif augmentation == "paper_like":
        spatial = [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(image_size),
        ]
    else:
        raise ValueError(f"Unknown training augmentation: {augmentation}")
    if hard_augmentation and augmentation_policy == "baseline":
        # A bounded affine change models detector-box scale and offset while
        # keeping the existing full-person crop policy for A7.
        geometry = transforms.RandomApply([
            transforms.RandomAffine(
                degrees=0, translate=(0.03, 0.03), scale=(0.94, 1.06),
                interpolation=InterpolationMode.BICUBIC,
            ),
        ], p=0.7)
        if augmentation == "resize_pad_crop":
            spatial.insert(-1, geometry)  # apply inside the padded canvas
        else:
            spatial.append(geometry)
    geometry = transforms.Compose([
        *spatial,
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(rotation, interpolation=InterpolationMode.BICUBIC),
    ])
    if augmentation_policy != "baseline":
        return PolicyTrainTransform(geometry, augmentation_policy, image_size, augmix_alpha)
    return transforms.Compose([
        geometry,
        *([CameraAugmentation()] if hard_augmentation else []),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def build_eval_transform(
    image_size: int = 224, augmentation: str = "current",
) -> transforms.Compose:
    if augmentation == "resize_pad_crop":
        spatial = [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ]
    else:
        spatial = [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(image_size),
        ]
    return transforms.Compose([
        *spatial,
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
