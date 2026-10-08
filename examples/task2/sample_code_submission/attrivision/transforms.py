"""Image transforms matching CLIP ViT-B/32 input conventions."""
from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import InterpolationMode

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_train_transform(
    image_size: int = 224, rotation: float = 10.0,
    augmentation: str = "current",
) -> transforms.Compose:
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
    return transforms.Compose([
        *spatial,
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(rotation, interpolation=InterpolationMode.BICUBIC),
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
