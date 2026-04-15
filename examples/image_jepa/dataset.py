"""
Dataset and augmentation utilities for self-supervised learning.

Supports CIFAR-10 (32×32, 10 classes) and ImageNet1k (224×224, 1000 classes).
"""

from typing import Optional

import torch
import torch.utils.data
import torchvision.transforms as transforms

# (image_size, mean, std, num_classes)
DATASET_INFO = {
    "cifar10": (32, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 10),
    "imagenet1k": (224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 1000),
}


DATASET_ABBREV = {"cifar10": "c10", "imagenet1k": "in1k"}


def get_dataset_info(dataset: str) -> tuple[int, int]:
    """Return (image_size, num_classes) for a dataset.

    Args:
        dataset: Dataset name (e.g. "cifar10", "imagenet1k").

    Returns:
        Tuple of (image_size, num_classes).
    """
    if dataset not in DATASET_INFO:
        raise ValueError(
            f"Unknown dataset: {dataset}. Choose from {list(DATASET_INFO)}"
        )
    info = DATASET_INFO[dataset]
    return info[0], info[3]


class RandomResizedCrop:
    """Random resized crop augmentation."""

    def __init__(self, size, scale=(0.2, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, img):
        return transforms.RandomResizedCrop(self.size, scale=self.scale)(img)


class ColorJitter:
    """Color jitter augmentation."""

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, prob=0.8):
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return self.transform(img)
        return img


class Grayscale:
    """Grayscale augmentation."""

    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return transforms.Grayscale(num_output_channels=3)(img)
        return img


class Solarization:
    """Solarization augmentation."""

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            img = transforms.functional.solarize(img, threshold=128)
        return img


class GaussianBlur:
    """Gaussian blur augmentation."""

    def __init__(self, kernel_size=23, sigma=(0.1, 2.0), prob=0.5):
        self.transform = transforms.GaussianBlur(kernel_size, sigma=sigma)
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return self.transform(img)
        return img


class HorizontalFlip:
    """Horizontal flip augmentation."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return transforms.functional.hflip(img)
        return img


def get_train_transforms(
    dataset: str = "cifar10",
    crop_scale: Optional[tuple[float, float]] = None,
    image_size: Optional[int] = None,
) -> transforms.Compose:
    """Get training transforms for self-supervised learning.

    Args:
        dataset: Dataset name. Determines normalization stats and augmentation
            pipeline. ImageNet-1K uses the VICReg recipe (scale=0.08,
            GaussianBlur); other datasets use a simpler pipeline.
        crop_scale: Override the default RandomResizedCrop scale range.
            If None, uses dataset defaults (0.08 for ImageNet, 0.2 for CIFAR).
        image_size: Override the output crop resolution. If None, uses the
            dataset default (224 for ImageNet, 32 for CIFAR).
    """
    default_size, mean, std, _ = DATASET_INFO[dataset]
    image_size = image_size if image_size is not None else default_size

    if dataset == "imagenet1k":
        scale = crop_scale if crop_scale is not None else (0.08, 1.0)
        aug_ops = [
            RandomResizedCrop(image_size, scale=scale),
            HorizontalFlip(prob=0.5),
            ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, prob=0.8
            ),
            Grayscale(prob=0.2),
            GaussianBlur(kernel_size=23, prob=0.5),
            Solarization(prob=0.1),
        ]
    else:
        scale = crop_scale if crop_scale is not None else (0.2, 1.0)
        aug_ops = [
            RandomResizedCrop(image_size, scale=scale),
            ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, prob=0.8
            ),
            Grayscale(prob=0.2),
            Solarization(prob=0.1),
            HorizontalFlip(prob=0.5),
        ]

    return transforms.Compose(
        aug_ops + [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )


def make_transforms(cfg, dataset: str) -> list[transforms.Compose]:
    """Build a list of per-view transforms from config.

    Args:
        cfg: OmegaConf config with ``data.num_views``, ``data.multi_crop``,
            ``data.n_global_views``, ``data.global_crop_scale``,
            ``data.local_crop_scale``.
        dataset: Dataset name (e.g. "cifar10", "imagenet1k").

    Returns:
        List of transforms, one per view. Length equals ``cfg.data.num_views``.
    """
    num_views = cfg.data.get("num_views", 2)
    multi_crop = cfg.data.get("multi_crop", False)

    if not multi_crop:
        t = get_train_transforms(dataset)
        return [t] * num_views

    n_global = cfg.data.get("n_global_views", 2)
    global_scale = tuple(cfg.data.get("global_crop_scale", [0.3, 1.0]))
    local_scale = tuple(cfg.data.get("local_crop_scale", [0.05, 0.3]))
    local_crop_size = cfg.data.get("local_crop_size", None)

    global_t = get_train_transforms(dataset, crop_scale=global_scale)
    local_t = get_train_transforms(
        dataset, crop_scale=local_scale, image_size=local_crop_size
    )
    return [global_t] * n_global + [local_t] * (num_views - n_global)


def get_val_transforms(dataset: str = "cifar10") -> transforms.Compose:
    """Get validation transforms.

    Args:
        dataset: Dataset name. Determines normalization stats and resize/crop.
    """
    image_size, mean, std, _ = DATASET_INFO[dataset]
    ops = []
    if image_size > 32:
        ops += [transforms.Resize(256), transforms.CenterCrop(image_size)]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


class ImageDataset(torch.utils.data.Dataset):
    """Dataset that applies per-view augmentations to create multiple views.

    Args:
        dataset: Base dataset returning (image, label) tuples.
        transforms: List of transform pipelines, one per view.
    """

    def __init__(self, dataset, transforms: list):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        views = [t(image) for t in self.transforms]
        return views, label
